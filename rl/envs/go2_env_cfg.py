"""Isaac Lab ManagerBasedEnvCfg for Go2 locomotion.

Phase 2 of the migration: Go2EnvCfg extends ManagerBasedEnvCfg with:
- scene (ground plane, Go2 robot articulation, lights)
- actions (12 joint positions, scale to torque limits)
- observations (45-dim: base lin/ang vel, projected gravity, joint pos/vel, commands)
- events (domain randomization via EventManager)
- rewards (heuristic terms ported from rl/heuristic.py)
- commands (velocity command generator with curriculum)
- terminations (fall detection, timeout)
"""

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysxAutoCfg, PhysicsCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    EventTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    TerminationTermCfg,
    ActionTermCfg,
)
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg

from rl.envs.constants_isaac import (
    SIM_DT,
    DECIMATION,
    OBS_DIM,
    NUM_ACTIONS,
    JOINT_NAMES,
    BASE_BODY,
    FOOT_BODIES,
    HOME_BASE_HEIGHT,
    KP,
    KV,
    TORQUE_LIMIT_HIP_THIGH,
    TORQUE_LIMIT_CALF,
    GAIT_REGIME_THRESHOLDS_MS,
    DEFAULT_JOINT_POS,
    MAX_EPISODE_STEPS,
    TERMINATE_BASE_HEIGHT,
    TERMINATE_MAX_ROLL_PITCH,
    OBS_IDX_ANG_VEL,
    OBS_IDX_GRAVITY,
    OBS_IDX_COMMANDS,
    OBS_IDX_DQ,
    OBS_IDX_QD,
    OBS_IDX_PREV_ACTION,
)

from rl.envs.scene_isaac import Go2SceneCfg
from rl.envs.terrain_isaac import Go2TerrainCfg, terrain_importer_cfg
from rl.envs.rewards_isaac import REWARD_WEIGHTS
from rl.envs.events_isaac import EVENT_CONFIG
from rl.envs.commands_isaac import Go2VelocityCommandCfg

# Import newton physics configs
import isaaclab_newton.physics as newton_physics


@configclass
class ObservationsCfg:
    """Observation specifications for the Go2 env."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Policy observation group."""

        base_ang_vel = ObservationTermCfg(func=lambda env: env.scene["robot"].data.root_ang_vel_b[:, :])
        projected_gravity = ObservationTermCfg(func=lambda env: env.scene["robot"].data.projected_gravity_b[:, :])
        velocity_commands = ObservationTermCfg(func=lambda env: env.command_manager.get_command("base_velocity"))
        joint_pos = ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_pos[:, :])
        joint_vel = ObservationTermCfg(func=lambda env: env.scene["robot"].data.joint_vel[:, :])
        prev_action = ObservationTermCfg(func=lambda env: env.action_manager.prev_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: ObservationGroupCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the Go2 env."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class EventsCfg:
    """Event configurations for domain randomization."""

    # Startup events
    add_base_mass = EventTermCfg(
        func="rl.envs.events_isaac:randomize_rigid_body_mass",
        mode="startup",
        params={
            "asset_cfg": "robot",
            "mass_distribution_params": (0.9, 1.1),
        },
    )

    # Reset events
    reset_robot_joints = EventTermCfg(
        func="rl.envs.events_isaac:reset_robot_joints",
        mode="reset",
        params={
            "asset_cfg": "robot",
            "position_range": {"noise": 0.1},
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_root_state = EventTermCfg(
        func="rl.envs.events_isaac:reset_root_state",
        mode="reset",
        params={
            "asset_cfg": "robot",
            "pose_range": {"pos_noise": 0.1, "rot_noise": 0.1},
            "velocity_range": {"lin_scale": 0.1, "ang_scale": 0.1},
        },
    )

    # Interval events
    push_robot = EventTermCfg(
        func="rl.envs.events_isaac:push_robot",
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": "robot",
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)},
        },
    )
    randomize_mass = EventTermCfg(
        func="rl.envs.events_isaac:randomize_rigid_body_mass",
        mode="interval",
        interval_range_s=(10.0, 20.0),
        params={
            "asset_cfg": "robot",
            "mass_distribution_params": (0.9, 1.1),
        },
    )
    randomize_com = EventTermCfg(
        func="rl.envs.events_isaac:randomize_rigid_body_com",
        mode="interval",
        interval_range_s=(10.0, 20.0),
        params={
            "asset_cfg": "robot",
            "com_distribution_params": (-0.02, 0.02),
        },
    )
    randomize_gains = EventTermCfg(
        func="rl.envs.events_isaac:randomize_actuator_gains",
        mode="interval",
        interval_range_s=(10.0, 20.0),
        params={
            "asset_cfg": "robot",
            "stiffness_range": (0.8, 1.2),
            "damping_range": (0.8, 1.2),
        },
    )


@configclass
class RewardsCfg:
    """Reward term configurations."""

    # Tracking rewards
    track_lin_vel_xy = RewardTermCfg(
        func="rl.envs.rewards_isaac:track_lin_vel_xy",
        weight=REWARD_WEIGHTS["track_lin_vel_xy"],
        params={"std": 0.5, "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewardTermCfg(
        func="rl.envs.rewards_isaac:track_ang_vel_z",
        weight=REWARD_WEIGHTS["track_ang_vel_z"],
        params={"std": 0.5, "command_name": "base_velocity"},
    )

    # Posture rewards
    base_height = RewardTermCfg(
        func="rl.envs.rewards_isaac:base_height",
        weight=REWARD_WEIGHTS["base_height"],
        params={"target_height": HOME_BASE_HEIGHT, "std": 0.05},
    )
    upright = RewardTermCfg(
        func="rl.envs.rewards_isaac:upright",
        weight=REWARD_WEIGHTS["upright"],
        params={"std": 0.2},
    )
    stand_still_zero_cmd = RewardTermCfg(
        func="rl.envs.rewards_isaac:stand_still_zero_cmd",
        weight=REWARD_WEIGHTS["stand_still_zero_cmd"],
        params={"std": 0.1},
    )

    # Penalty terms
    joint_pos_limits = RewardTermCfg(
        func="rl.envs.rewards_isaac:joint_pos_limits",
        weight=REWARD_WEIGHTS["joint_pos_limits"],
    )
    torques = RewardTermCfg(
        func="rl.envs.rewards_isaac:torques",
        weight=REWARD_WEIGHTS["torques"],
    )
    action_rate = RewardTermCfg(
        func="rl.envs.rewards_isaac:action_rate",
        weight=REWARD_WEIGHTS["action_rate"],
    )
    dof_vel = RewardTermCfg(
        func="rl.envs.rewards_isaac:dof_vel",
        weight=REWARD_WEIGHTS["dof_vel"],
    )
    dof_acc = RewardTermCfg(
        func="rl.envs.rewards_isaac:dof_acc",
        weight=REWARD_WEIGHTS["dof_acc"],
    )
    feet_air_time = RewardTermCfg(
        func="rl.envs.rewards_isaac:feet_air_time",
        weight=REWARD_WEIGHTS["feet_air_time"],
    )
    feet_contact_forces = RewardTermCfg(
        func="rl.envs.rewards_isaac:feet_contact_forces",
        weight=REWARD_WEIGHTS["feet_contact_forces"],
    )
    base_lin_vel_z = RewardTermCfg(
        func="rl.envs.rewards_isaac:base_lin_vel_z",
        weight=REWARD_WEIGHTS["base_lin_vel_z"],
    )
    base_ang_vel_xy = RewardTermCfg(
        func="rl.envs.rewards_isaac:base_ang_vel_xy",
        weight=REWARD_WEIGHTS["base_ang_vel_xy"],
    )
    joint_powers = RewardTermCfg(
        func="rl.envs.rewards_isaac:joint_powers",
        weight=REWARD_WEIGHTS["joint_powers"],
    )


@configclass
class CommandsCfg:
    """Command configurations for velocity commands."""

    base_velocity = Go2VelocityCommandCfg()


@configclass
class TerminationsCfg:
    """Termination conditions."""

    base_height_too_low = TerminationTermCfg(
        func=lambda env: env.scene["robot"].data.root_pos_w[:, 2] < TERMINATE_BASE_HEIGHT,
    )

    base_roll_pitch_too_high = TerminationTermCfg(
        func=lambda env: torch.any(torch.abs(env.scene["robot"].data.projected_gravity_b[:, :2]) > TERMINATE_MAX_ROLL_PITCH, dim=1),
    )

    time_out = TerminationTermCfg(func=lambda env: env.episode_length_buf >= MAX_EPISODE_STEPS)


@configclass
class Go2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 ManagerBasedEnv."""

    # Simulation settings - use Newton physics backend
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        # Use Newton physics engine with Featherstone solver
        physics=NewtonCfg(
            solver_cfg=newton_physics.FeatherstoneSolverCfg(
                class_type=newton_physics.NewtonFeatherstoneManager,
                # TGS-like settings for Newton Featherstone
                friction_smoothing=1.0,
                angular_damping=0.05,
                fuse_cholesky=True,
            ),
        ),
        use_newton_actuators=True,
    )

    # Decimation (physics steps per policy step)
    decimation: int = DECIMATION

    # Decimation
    decimation: int = DECIMATION

    # Episode length
    episode_length_s: float = MAX_EPISODE_STEPS * SIM_DT * DECIMATION

    # Scene configuration
    scene: Go2SceneCfg = Go2SceneCfg()

    # Observation/action specs
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    # Events, rewards, commands, terminations
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Seed
    seed: int = 42


__all__ = ["Go2EnvCfg", "ObservationsCfg", "ActionsCfg", "EventsCfg", "RewardsCfg", "CommandsCfg", "TerminationsCfg"]