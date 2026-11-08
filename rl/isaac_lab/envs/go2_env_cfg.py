"""Isaac Lab environment configuration for Unitree Go2.

Migrates the mujoco-warp Go2WarpEnv to Isaac Lab ManagerBasedRLEnvCfg.
Maintains 45-dim observation and 12-dim action spaces for exact parity
with contracts.py.

Supports terrain types: plane, slope, rough, generator (multi-terrain)
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    CommandTermCfg as CmdTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from isaaclab.envs.mdp.events import reset_root_state_uniform, reset_joints_by_scale
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

import isaaclab.sim as sim_utils

from rl.contracts import KP, KV, TORQUE_LIMIT_HIP_THIGH, TORQUE_LIMIT_CALF

from rl.isaac_lab.mdp import (
    track_lin_vel_xy_exp,
    track_ang_vel_z_exp,
    base_height_l2,
    flat_orientation_l2,
    joint_torques_l2,
    joint_acc_l2,
    action_rate_l2,
    joint_deviation_l2,
    feet_air_time,
    forward_progress,
    is_alive,
    undesired_contacts,
    generated_commands,
)
from rl.isaac_lab.events import (
    randomize_joint_friction,
    randomize_rigid_body_mass,
    randomize_rigid_body_com,
    randomize_friction,
    randomize_motor_strength,
    push_robot,
)
from rl.isaac_lab.terminations import base_height as term_base_height, bad_orientation, time_out
from rl.isaac_lab.curriculum.velocity import VelocityCurriculum, velocity_curriculum_term
from rl.isaac_lab.commands import velocity_command

# Import terrain configs from rl/envs/terrain_isaac.py (direct, avoid circular import via rl/envs/__init__)
from rl.envs.terrain_isaac import (
    Go2FlatTerrainCfg,
    Go2SlopeTerrainCfg,
    Go2RoughTerrainCfg,
    flat_terrain_importer_cfg,
    slope_terrain_importer_cfg,
    rough_terrain_importer_cfg,
)


# ---------------------------------------------------------------------------
# Observation Terms (using configclass for proper structure)
# ---------------------------------------------------------------------------

@configclass
class Go2ObservationsCfg:
    """Observations configuration with policy group - exact 45-dim matching contracts.py.

    Order: [ang_vel(3), gravity(3), commands(3), dq(12), qd(12), prev_action(12)]
    """
    @configclass
    class PolicyCfg(ObsGroup):
        # Base angular velocity (3) - indices 0:3
        base_ang_vel = ObsTerm(func=base_ang_vel, params={})
        # Projected gravity (3) - indices 3:6
        projected_gravity = ObsTerm(func=projected_gravity, params={})
        # Velocity command (3) - indices 6:9
        velocity_commands = ObsTerm(func=generated_commands, params={"command_name": "base_velocity"})
        # Joint positions relative to default (12) - indices 9:21
        joint_pos_rel = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Joint velocities (12) - indices 21:33
        joint_vel_rel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Previous action (12) - indices 33:45
        prev_action = ObsTerm(func=last_action, params={})
    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Action Terms (using configclass)
# ---------------------------------------------------------------------------

@configclass
class Go2ActionsCfg:
    """Action configuration - 12-dim joint positions."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale=0.5,
        use_default_offset=True,
        clip={".*": [-1.0, 1.0]},
    )


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Scene with Go2 robot, terrain, and contact sensors."""

    # Robot (using official Unitree Go2 USD from isaaclab_assets)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),  # crouched stance equilibrium hip height (stance sweep, Aristot2026-11-08)
            joint_pos={
                "FL_hip_joint": 0.0, "FL_thigh_joint": 0.7, "FL_calf_joint": -1.6,
                "FR_hip_joint": 0.0, "FR_thigh_joint": 0.7, "FR_calf_joint": -1.6,
                "RL_hip_joint": 0.0, "RL_thigh_joint": 0.7, "RL_calf_joint": -1.6,
                "RR_hip_joint": 0.0, "RR_thigh_joint": 0.7, "RR_calf_joint": -1.6,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                # Per-joint effort limits from contracts.TORQUE_LIMIT_* (hip/thigh 23.7, calf 45.43).
                # A single flat value would silently grant the hips/thighs calf-range torque
                # and let the policy learn what the real Go2 actuators cannot deliver.
                effort_limit={
                    ".*_hip_joint": TORQUE_LIMIT_HIP_THIGH,
                    ".*_thigh_joint": TORQUE_LIMIT_HIP_THIGH,
                    ".*_calf_joint": TORQUE_LIMIT_CALF,
                },
                # The MuJoCo XML has NO joint velocity limit, and the warp PD kernel
                # clamps torque only (wp.clamp(tau, -lim, lim)). Any finite cap here
                # would clip a hard calf swing and diverge from the warp stack, so keep
                # it non-constraining rather than "realistic".
                velocity_limit=1000.0,
                # Stiffness/damping tuned on the Isaac Lab + Newton backend (2026-11-08).
                # The old KP=60/KV=3 (MuJoCo XML values, via contracts.KP/KV) cannot hold
                # the Go2 stance under gravity in Newton: zero-action probes tip over in
                # ~0.4 s and PPO converges to a tipping policy. KP=400/KV=8 with the
                # crouched stance (thigh 0.7, calf -1.6) stands stably and absorbs a
                # 0.3 m/s lateral push with <3 deg tilt — see experiments/_stance_sweep.py.
                stiffness=400.0,
                damping=8.0,
            ),
        },
    )

    # Terrain - default is flat plane (will be overridden by terrain-specific subclasses)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.8, 0.8),
            metallic=0.0,
            roughness=0.8,
        ),
        debug_vis=False,
    )

    # Contact sensors on feet
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=6,
        track_air_time=True,
    )


# ---------------------------------------------------------------------------
# Terrain-Specific Scene Configurations
# ---------------------------------------------------------------------------

@configclass
class Go2FlatSceneCfg(Go2SceneCfg):
    """Scene with flat terrain (plane)."""
    terrain = flat_terrain_importer_cfg()


@configclass
class Go2SlopeSceneCfg(Go2SceneCfg):
    """Scene with slope terrain (constant-grade ramp 10-15 deg)."""
    terrain = slope_terrain_importer_cfg()


@configclass
class Go2RoughSceneCfg(Go2SceneCfg):
    """Scene with rough terrain (smoothed noise amplitude 0.05-0.15m)."""
    terrain = rough_terrain_importer_cfg()


# ---------------------------------------------------------------------------
# Base Environment Configuration (45-dim obs, 12-dim action)
# ---------------------------------------------------------------------------

@configclass
class Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Base flat-terrain Go2 config with exact 45-dim contract observation."""

    # Scene
    scene: Go2FlatSceneCfg = Go2FlatSceneCfg(num_envs=4096, env_spacing=2.5)

    # Simulation (matching contracts.py exactly)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.002,           # SIM_DT
        device="cuda",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=200,
                nconmax=64,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        ),
    )

    # Episode length
    decimation: int = 10    # DECIMATION
    episode_length_s: float = 20.0

    # Observations
    observations: Go2ObservationsCfg = Go2ObservationsCfg()

    # Actions
    actions: Go2ActionsCfg = Go2ActionsCfg()

    # Commands - use a nested configclass for CommandsCfg
    @configclass
    class CommandsCfg:
        """Command specifications for the MDP."""
        base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(20.0, 20.0),  # == episode_length_s: commands stay constant per episode (Rudin et al. protocol)
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
            ),
        )
    commands: CommandsCfg = CommandsCfg()

    # Rewards (stage-specific weights applied in subclasses)
    @configclass
    class RewardsCfg:
        is_alive = RewTerm(func=is_alive, weight=0.5)  # survival floor: makes living non-negative
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        joint_deviation = RewTerm(func=joint_deviation_l2, weight=0.0, params={})
        flat_orientation = RewTerm(func=flat_orientation_l2, weight=0.0, params={})
        joint_acc = RewTerm(func=joint_acc_l2, weight=0.0, params={})
        action_rate = RewTerm(func=action_rate_l2, weight=0.0, params={})
        joint_torques = RewTerm(func=joint_torques_l2, weight=0.0, params={})
        base_height = RewTerm(func=base_height_l2, weight=0.0, params={"target_height": 0.27, "sensor_cfg": SceneEntityCfg("robot")})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.0, params={"command_name": "base_velocity", "threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        forward_progress = RewTerm(func=forward_progress, weight=0.0, params={"command_name": "base_velocity"})
    rewards: RewardsCfg = RewardsCfg()

    # Events (domain randomization)
    @configclass
    class EventsCfg:
        randomize_mass = EventTerm(
            func=randomize_rigid_body_mass,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.9, 1.1)},
        )
        randomize_friction = EventTerm(
            func=randomize_friction,
            mode="reset",
            params={
                # MuJoCo's RandomizationSpec randomises the ground *and* foot geoms, but the
                # Go2 feet carry priority="1" so the feet ARE the realised contact friction.
                # Newton's material term only binds Articulation/RigidObject shapes, so we
                # randomise the foot colliders here and leave the terrain material static.
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (0.6, 1.4),
                "dynamic_friction_range": (0.6, 1.4),
                "restitution_range": (0.0, 0.0),
            },
        )
        randomize_joint_friction = EventTerm(
            func=randomize_joint_friction,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "friction_distribution_params": (0.01, 0.1)},
        )
        randomize_motor_strength = EventTerm(
            func=randomize_motor_strength,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "strength_distribution_params": (0.9, 1.1)},
        )
        push_robot = EventTerm(
            func=push_robot,
            mode="reset",
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)}},
        )
        reset_robot = EventTerm(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        reset_joints = EventTerm(
            func=reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    events: EventsCfg = EventsCfg()

    # Terminations
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=time_out, time_out=True)
        base_height = DoneTerm(func=term_base_height, params={"minimum_height": 0.16})
        bad_orientation = DoneTerm(func=bad_orientation, params={"limit_angle": 1.0})
    terminations: TerminationsCfg = TerminationsCfg()

    # Curriculum
    @configclass
    class CurriculumCfg:
        velocity_curriculum = CurrTerm(
            func=velocity_curriculum_term,
            params={},
        )
    curriculum: CurriculumCfg = CurriculumCfg()


# ---------------------------------------------------------------------------
# Stage-Specific Configurations
# ---------------------------------------------------------------------------

@configclass
class Go2StandEnvCfg(Go2FlatEnvCfg):
    """Stage 0: Stand from rest. No velocity tracking."""

    def __post_init__(self):
        super().__post_init__()
        # Spawn at the crouched-stance equilibrium height (same as base config).
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(pos=(0.0, 0.0, 0.38))
        )

    @configclass
    class RewardsCfg(Go2FlatEnvCfg.RewardsCfg):
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        joint_deviation = RewTerm(func=joint_deviation_l2, weight=-0.5, params={})
        flat_orientation = RewTerm(func=flat_orientation_l2, weight=-1.0, params={})
        joint_acc = RewTerm(func=joint_acc_l2, weight=-1.0e-5, params={})
        action_rate = RewTerm(func=action_rate_l2, weight=-0.01, params={})
        joint_torques = RewTerm(func=joint_torques_l2, weight=-1.0e-4, params={})
        base_height = RewTerm(func=base_height_l2, weight=-0.5, params={"target_height": 0.37, "sensor_cfg": SceneEntityCfg("robot")})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.0, params={"command_name": "base_velocity", "threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        forward_progress = RewTerm(func=forward_progress, weight=0.0, params={"command_name": "base_velocity"})

    rewards: RewardsCfg = RewardsCfg()

    @configclass
    class TerminationsCfg(Go2FlatEnvCfg.TerminationsCfg):
        base_height = DoneTerm(func=term_base_height, params={"minimum_height": 0.16})
    terminations: TerminationsCfg = TerminationsCfg()

    @configclass
    class CurriculumCfg:
        pass
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class Go2BalanceEnvCfg(Go2FlatEnvCfg):
    """Stage 1: Stand + balance under perturbations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(pos=(0.0, 0.0, 0.38))
        )

    @configclass
    class RewardsCfg(Go2FlatEnvCfg.RewardsCfg):
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z_exp, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        joint_deviation = RewTerm(func=joint_deviation_l2, weight=-0.5, params={})
        flat_orientation = RewTerm(func=flat_orientation_l2, weight=-1.0, params={})
        joint_acc = RewTerm(func=joint_acc_l2, weight=-1.0e-5, params={})
        action_rate = RewTerm(func=action_rate_l2, weight=-0.01, params={})
        joint_torques = RewTerm(func=joint_torques_l2, weight=-1.0e-4, params={})
        base_height = RewTerm(func=base_height_l2, weight=-0.5, params={"target_height": 0.37, "sensor_cfg": SceneEntityCfg("robot")})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.0, params={"command_name": "base_velocity", "threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        forward_progress = RewTerm(func=forward_progress, weight=0.0, params={"command_name": "base_velocity"})

    rewards: RewardsCfg = RewardsCfg()

    @configclass
    class EventsCfg(Go2FlatEnvCfg.EventsCfg):
        push_robot = EventTerm(
            func=push_robot,
            mode="reset",
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)}},
        )
    events: EventsCfg = EventsCfg()

    @configclass
    class CurriculumCfg:
        pass
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class Go2TransitionEnvCfg(Go2FlatEnvCfg):
    """Stage 2: Stand -> walk transition with curriculum."""

    @configclass
    class RewardsCfg(Go2FlatEnvCfg.RewardsCfg):
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy_exp, weight=1.0, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z_exp, weight=0.5, params={"std": 0.25, "command_name": "base_velocity"})
        joint_deviation = RewTerm(func=joint_deviation_l2, weight=-0.5, params={})
        flat_orientation = RewTerm(func=flat_orientation_l2, weight=-1.0, params={})
        joint_acc = RewTerm(func=joint_acc_l2, weight=-1.0e-5, params={})
        action_rate = RewTerm(func=action_rate_l2, weight=-0.01, params={})
        joint_torques = RewTerm(func=joint_torques_l2, weight=-1.0e-4, params={})
        base_height = RewTerm(func=base_height_l2, weight=-0.5, params={"target_height": 0.37, "sensor_cfg": SceneEntityCfg("robot")})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.25, params={"command_name": "base_velocity", "threshold": 0.3, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        forward_progress = RewTerm(func=forward_progress, weight=0.25, params={"command_name": "base_velocity"})

    rewards: RewardsCfg = RewardsCfg()

    @configclass
    class EventsCfg(Go2FlatEnvCfg.EventsCfg):
        push_robot = EventTerm(
            func=push_robot,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0)}},
        )
    events: EventsCfg = EventsCfg()

    @configclass
    class CurriculumCfg(Go2FlatEnvCfg.CurriculumCfg):
        pass
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class Go2WalkEnvCfg(Go2FlatEnvCfg):
    """Stage 3: Walk with velocity tracking."""

    @configclass
    class RewardsCfg(Go2FlatEnvCfg.RewardsCfg):
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy_exp, weight=1.5, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z_exp, weight=0.5, params={"std": 0.25, "command_name": "base_velocity"})
        joint_deviation = RewTerm(func=joint_deviation_l2, weight=-0.5, params={})
        flat_orientation = RewTerm(func=flat_orientation_l2, weight=-1.0, params={})
        joint_acc = RewTerm(func=joint_acc_l2, weight=-1.0e-5, params={})
        action_rate = RewTerm(func=action_rate_l2, weight=-0.01, params={})
        joint_torques = RewTerm(func=joint_torques_l2, weight=-1.0e-4, params={})
        base_height = RewTerm(func=base_height_l2, weight=-0.5, params={"target_height": 0.37, "sensor_cfg": SceneEntityCfg("robot")})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.5, params={"command_name": "base_velocity", "threshold": 0.3, "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")})
        forward_progress = RewTerm(func=forward_progress, weight=0.5, params={"command_name": "base_velocity"})

    rewards: RewardsCfg = RewardsCfg()

    @configclass
    class CurriculumCfg(Go2FlatEnvCfg.CurriculumCfg):
        pass
    curriculum: CurriculumCfg = CurriculumCfg()


# ---------------------------------------------------------------------------
# Terrain-Specific Environment Configurations
# ---------------------------------------------------------------------------

@configclass
class Go2WalkSlopeEnvCfg(Go2WalkEnvCfg):
    """Stage 3 Walk on slope terrain (10-15 degree grade)."""
    scene: Go2SlopeSceneCfg = Go2SlopeSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2WalkRoughEnvCfg(Go2WalkEnvCfg):
    """Stage 3 Walk on rough terrain (amplitude 0.05-0.15m)."""
    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2TransitionSlopeEnvCfg(Go2TransitionEnvCfg):
    """Stage 2 Transition on slope terrain."""
    scene: Go2SlopeSceneCfg = Go2SlopeSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2TransitionRoughEnvCfg(Go2TransitionEnvCfg):
    """Stage 2 Transition on rough terrain."""
    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2BalanceSlopeEnvCfg(Go2BalanceEnvCfg):
    """Stage 1 Balance on slope terrain."""
    scene: Go2SlopeSceneCfg = Go2SlopeSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2BalanceRoughEnvCfg(Go2BalanceEnvCfg):
    """Stage 1 Balance on rough terrain."""
    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2StandSlopeEnvCfg(Go2StandEnvCfg):
    """Stage 0 Stand on slope terrain."""
    scene: Go2SlopeSceneCfg = Go2SlopeSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2StandRoughEnvCfg(Go2StandEnvCfg):
    """Stage 0 Stand on rough terrain."""
    scene: Go2RoughSceneCfg = Go2RoughSceneCfg(num_envs=4096, env_spacing=2.5)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "Go2SceneCfg",
    "Go2FlatSceneCfg",
    "Go2SlopeSceneCfg",
    "Go2RoughSceneCfg",
    "Go2ObservationsCfg",
    "Go2ActionsCfg",
    "Go2FlatEnvCfg",
    "Go2StandEnvCfg",
    "Go2BalanceEnvCfg",
    "Go2TransitionEnvCfg",
    "Go2WalkEnvCfg",
    "Go2WalkSlopeEnvCfg",
    "Go2WalkRoughEnvCfg",
    "Go2TransitionSlopeEnvCfg",
    "Go2TransitionRoughEnvCfg",
    "Go2BalanceSlopeEnvCfg",
    "Go2BalanceRoughEnvCfg",
    "Go2StandSlopeEnvCfg",
    "Go2StandRoughEnvCfg",
]