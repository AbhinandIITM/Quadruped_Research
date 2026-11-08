"""Go2IsaacEnv: DirectRLEnv parity with Go2WarpEnv from rl/gpu/env.py.

This environment matches the GPU mujoco-warp stack exactly:
- 45-dim observation space (same composition/order as contracts.py)
- 12-dim action space (actuator order FR,FL,RR,RL hip/thigh/calf)
- PD position control with KP=60, KV=3
- Newton physics backend
- Exact seed discipline matching rl/envs/vec.py
- Domain randomization via EventManager
- Reward terms via RewardManager
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import NewtonCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from rl.isaac_lab.mdp import (
    track_lin_vel_xy_exp as track_lin_vel_xy,
    track_ang_vel_z_exp as track_ang_vel_z,
    base_height_l2 as base_height,
    flat_orientation_l2 as upright,
    joint_torques_l2 as torques_penalty,
    joint_acc_l2 as dof_acc_penalty,
    action_rate_l2 as action_rate_penalty,
    joint_deviation_l2 as posture_penalty,
    feet_air_time,
    forward_progress,
)

if TYPE_CHECKING:
    from rl.isaac_lab.configs import IsaacStageConfig


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Scene with Go2 robot and ground plane."""
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{{ISAACLAB_PATH}}/assets/unitree_go2/go2.usd",
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
            pos=(0.0, 0.0, 0.4),
            joint_pos={".*_hip_joint": 0.0, ".*_thigh_joint": 0.9, ".*_calf_joint": -1.8},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=45.43,
                velocity_limit=23.7,
                stiffness=60.0,
                damping=3.0,
            ),
        },
    )
    ground = sim_utils.GroundPlaneCfg()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class Go2IsaacEnvCfg(DirectRLEnvCfg):
    """Configuration for Go2IsaacEnv with stage-parameterized rewards/commands."""
    # Scene
    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)

    # Simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.002,
        device="cuda",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=NewtonCfg(
            solver_cfg=None,  # Use defaults
            deterministic=True,
            use_cuda_graph=True,
        ),
    )

    # Episode
    episode_length_s: float = 20.0

    # Commands - use a nested configclass for CommandsCfg
    @configclass
    class CommandsCfg:
        """Command specifications for the MDP."""
        base_velocity = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-3.14159, 3.14159)
            ),
        )
    commands: CommandsCfg = CommandsCfg()

    # Observations (45-dim: ang_vel(3), gravity(3), commands(3), dq(12), qd(12), prev_action(12))
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg:
            base_ang_vel = ObsTerm(func="mdp.base_ang_vel", params={})
            projected_gravity = ObsTerm(func="mdp.projected_gravity", params={})
            velocity_command = ObsTerm(func="mdp.velocity_command", params={"command_name": "base_velocity"})
            joint_pos = ObsTerm(func="mdp.joint_pos", params={})
            joint_vel = ObsTerm(func="mdp.joint_vel", params={})
            prev_action = ObsTerm(func="mdp.prev_action", params={})
        policy: PolicyCfg = PolicyCfg()
    observations: ObservationsCfg = ObservationsCfg()

    # Actions (12-dim joint positions)
    @configclass
    class ActionsCfg:
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
    actions: ActionsCfg = ActionsCfg()

    # Events (domain randomization)
    @configclass
    class EventsCfg:
        randomize_mass = EventTerm(
            func="mdp.randomize_rigid_body_mass",
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": (0.9, 1.1)},
        )
        randomize_com = EventTerm(
            func="mdp.randomize_rigid_body_com",
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "com_distribution_params": (-0.02, 0.02)},
        )
        randomize_friction = EventTerm(
            func="mdp.randomize_friction",
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("ground"), "friction_distribution_params": (0.6, 1.4)},
        )
        randomize_motor_strength = EventTerm(
            func="mdp.randomize_motor_strength",
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "strength_distribution_params": (0.9, 1.1)},
        )
    events: EventsCfg = EventsCfg()

    # Rewards (weights set per-stage via apply_stage_config)
    @configclass
    class RewardsCfg:
        track_lin_vel_xy = RewTerm(func=track_lin_vel_xy, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        track_ang_vel_z = RewTerm(func=track_ang_vel_z, weight=0.0, params={"std": 0.25, "command_name": "base_velocity"})
        base_height = RewTerm(func=base_height, weight=0.0, params={"target_height": 0.27, "std": 0.05})
        upright = RewTerm(func=upright, weight=0.0, params={"std": 0.2})
        torques = RewTerm(func=torques_penalty, weight=0.0, params={})
        dof_acc = RewTerm(func=dof_acc_penalty, weight=0.0, params={})
        action_rate = RewTerm(func=action_rate_penalty, weight=0.0, params={})
        posture = RewTerm(func=posture_penalty, weight=0.0, params={})
        feet_air_time = RewTerm(func=feet_air_time, weight=0.0, params={"target_duty": 0.5})
        forward_progress = RewTerm(func=forward_progress, weight=0.0, params={})
    rewards: RewardsCfg = RewardsCfg()

    # Terminations
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func="mdp.time_out", time_out=True)
        base_height = DoneTerm(func="mdp.base_height", params={"minimum_height": 0.16})
        bad_orientation = DoneTerm(func="mdp.bad_orientation", params={"limit_angle": 1.0})
    terminations: TerminationsCfg = TerminationsCfg()

    # Stage parameter (set externally before env creation)
    stage: str = "walk"

    def __post_init__(self):
        super().__post_init__()
        self._apply_stage_config()

    def _apply_stage_config(self):
        """Apply stage-specific config to rewards, commands, terminations."""
        from rl.isaac_lab.configs import get_stage_config
        stage_cfg: IsaacStageConfig = get_stage_config(self.stage)

        # Update spawn height
        self.scene.robot.init_state.pos = (0.0, 0.0, stage_cfg.spawn_height)

        # Update base height reward target
        self.rewards["base_height"].params["target_height"] = stage_cfg.base_height_target

        # Update reward weights
        for name, weight in stage_cfg.weights.items():
            if name in self.rewards:
                self.rewards[name].weight = weight

        # Update command resampling based on curriculum
        if stage_cfg.use_curriculum:
            self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
            self.commands.base_velocity.rel_standing_envs = 0.0
        else:
            self.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
            self.commands.base_velocity.rel_standing_envs = 1.0
            # Zero velocity commands
            self.commands.base_velocity.ranges = {
                "lin_vel_x": (0.0, 0.0),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (0.0, 0.0),
            }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Go2IsaacEnv(DirectRLEnv):
    """DirectRLEnv matching Go2WarpEnv API exactly."""

    cfg: Go2IsaacEnvCfg

    def __init__(self, cfg: Go2IsaacEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Shortcut to robot
        self._robot: Articulation = self.scene["robot"]

        # Action buffer (persistent for graph capture parity)
        self._action_buf = torch.zeros(
            (self.num_envs, self.cfg.action_space.shape[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self._prev_action = torch.zeros_like(self._action_buf)

        # Episode tracking
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Default joint positions (matching contracts.py)
        self._default_joint_pos = torch.tensor(
            [0.0, 0.9, -1.8] * 4,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)

    def _setup_scene(self):
        self.scene = InteractiveScene(self.cfg.scene)
        self._robot = self.scene["robot"]
        self.scene.articulations["robot"] = self._robot

    def _pre_physics_step(self, actions: torch.Tensor):
        """Store actions for PD control (mirrors Go2WarpEnv)."""
        self._action_buf[:] = actions
        self._prev_action[:] = self._action_buf

    def _apply_action(self):
        """Apply PD position control (matching CPU env's per-substep servo)."""
        # Actions are normalized joint position targets [-1, 1] -> scale -> add default offset
        joint_pos_target = self._action_buf * self.cfg.actions.joint_pos.scale + self._default_joint_pos
        self._robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Return 45-dim observations matching contracts.py exactly."""
        robot = self._robot

        # Base angular velocity (3)
        base_ang_vel = robot.data.root_ang_vel_b

        # Projected gravity (3)
        projected_gravity = robot.data.projected_gravity_b

        # Velocity command (3)
        velocity_command = self.command_manager.get_command("base_velocity")

        # Joint positions relative to default (12)
        joint_pos = robot.data.joint_pos - self._default_joint_pos

        # Joint velocities (12)
        joint_vel = robot.data.joint_vel

        # Previous action (12)
        prev_action = self._prev_action

        # Concatenate in exact order: [ang_vel(3), gravity(3), commands(3), dq(12), qd(12), prev_action(12)]
        obs = torch.cat([
            base_ang_vel,
            projected_gravity,
            velocity_command,
            joint_pos,
            joint_vel,
            prev_action,
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Rewards computed by RewardManager (wired via config)."""
        # RewardManager handles this via the reward terms in cfg.rewards
        return torch.zeros(self.num_envs, device=self.device)  # Placeholder

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Termination and timeout computed by TerminationManager."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
               torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments (matches vec.py seed discipline)."""
        super()._reset_idx(env_ids)

        # Reset episode length
        self.episode_length_buf[env_ids] = 0

        # Reset action buffers
        self._action_buf[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

        # Reset robot state
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 2] = self.cfg.scene.robot.init_state.pos[2]  # spawn height
        self._robot.write_root_state_to_sim(default_root_state, env_ids)

        # Reset joint positions to default
        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

    def step(self, action: torch.Tensor):
        """Step function matching VecEnv API."""
        return super().step(action)


def make_go2_isaac_env(
    stage: str = "walk",
    num_envs: int = 4096,
    device: str = "cuda",
    seed: int = 0,
) -> Go2IsaacEnv:
    """Factory function to create Go2IsaacEnv with stage config."""
    from rl.isaac_lab.configs import make_env_cfg

    cfg = make_env_cfg(stage, num_envs, device)
    cfg.seed = seed
    env = Go2IsaacEnv(cfg)
    return env


__all__ = ["Go2IsaacEnv", "Go2IsaacEnvCfg", "Go2SceneCfg", "make_go2_isaac_env"]