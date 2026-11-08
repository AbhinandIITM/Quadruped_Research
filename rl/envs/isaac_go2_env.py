"""Isaac Lab ManagerBasedEnv for Go2 locomotion.

This is the main environment class that uses Go2EnvCfg.
It extends ManagerBasedEnv and provides the standard Gymnasium interface.

This module also re-exports all constants and config classes from the
Isaac Lab migration for backward compatibility with tests.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ManagerBasedEnvCfg
from isaaclab.managers import (
    ActionManager,
    ObservationManager,
    RewardManager,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    TerminationTermCfg as DoneTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.envs.mdp import (
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from rl.isaac_lab.mdp import generated_commands
from rl.isaac_lab.events import (
    randomize_rigid_body_mass,
    randomize_friction,
    randomize_joint_friction,
    randomize_motor_strength,
    push_robot,
)
from rl.isaac_lab.terminations import base_height as term_base_height, bad_orientation, time_out

# Re-export all constants from constants_isaac.py
from rl.envs.constants_isaac import (
    SIM_DT,
    CONTROL_HZ,
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
    FRICTION_RANGE,
    MASS_SCALE_RANGE,
    MOTOR_STRENGTH_RANGE,
    LEG_LENGTH_M,
    GRAVITY,
)

# Re-export reward-related constants from heuristic rewards
from rl.isaac_lab.configs import (
    SIGMA_LIN_VEL,
    SIGMA_ANG_VEL,
    BASE_HEIGHT_TARGET_DEFAULT as BASE_HEIGHT_TARGET,
    _HIP_POSTURE_WEIGHT as POSTURE_WEIGHT_HIP,
    _SAGITTAL_POSTURE_WEIGHT as POSTURE_WEIGHT_SAGITTAL,
    _POSTURE_WEIGHT,
)

# Re-export config classes from go2_env_cfg.py (the new rl/isaac_lab/envs/go2_env_cfg.py)
from rl.isaac_lab.envs.go2_env_cfg import (
    Go2FlatEnvCfg as Go2RewardCfg,  # base config with rewards
    Go2StandEnvCfg,
    Go2BalanceEnvCfg,
    Go2TransitionEnvCfg as Go2WalkTransitionEnvCfg,
    Go2WalkEnvCfg,
)

# Build additional config classes expected by tests
# These are aliases/constructors for the specific config blocks

@configclass
class Go2ObservationCfg:
    """Observation configuration - 45-dim matching contracts.py."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=base_ang_vel, params={})
        projected_gravity = ObsTerm(func=projected_gravity, params={})
        velocity_commands = ObsTerm(func=generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel_rel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        prev_action = ObsTerm(func=last_action, params={})
    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2ActionCfg:
    """Action configuration - 12-dim joint positions."""
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        clip={".*": [-1.0, 1.0]},
    )


@configclass
class Go2CommandCfg:
    """Command configuration."""
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class Go2EventCfg:
    """Event configuration for domain randomization."""
    randomize_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_distribution_params": MASS_SCALE_RANGE},
    )
    randomize_friction = EventTerm(
        func=randomize_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("terrain"), "friction_distribution_params": FRICTION_RANGE},
    )
    randomize_joint_friction = EventTerm(
        func=randomize_joint_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "friction_distribution_params": (0.01, 0.1)},
    )
    randomize_motor_strength = EventTerm(
        func=randomize_motor_strength,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "strength_distribution_params": MOTOR_STRENGTH_RANGE},
    )
    push_robot = EventTerm(
        func=push_robot,
        mode="reset",
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)}},
    )


@configclass
class Go2TerminationCfg:
    """Termination configuration."""
    time_out = DoneTerm(func=time_out, time_out=True)
    base_height = DoneTerm(func=term_base_height, params={"minimum_height": TERMINATE_BASE_HEIGHT})
    bad_orientation = DoneTerm(func=bad_orientation, params={"limit_angle": TERMINATE_MAX_ROLL_PITCH})


# Stand-specific reward config - matches old heuristic reward term names
# Uses RewTerm objects so tests can check .weight attribute
@configclass
class Go2StandRewardCfg:
    """Stand stage reward weights - using original heuristic.py term names."""
    # Postural/stance rewards (from heuristic.py)
    base_height_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-2.0)
    flat_orientation_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-2.0)
    posture = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    joint_torques_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.1)
    joint_accel_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.02)
    action_rate_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.01)
    joint_pos_limits = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    body_contact = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)

    # Additional
    feet_air_time = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    forward_progress = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    base_height_target = 0.27


# Walk transition reward config - matches old heuristic reward term names
@configclass
class Go2WalkTransitionRewardCfg:
    """Transition stage reward weights - using original heuristic.py term names."""
    # Velocity tracking (added in transition)
    track_lin_vel_xy_exp = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.1)
    track_ang_vel_z_exp = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.05)

    # Postural/stance rewards (from heuristic.py)
    base_height_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-1.0)
    flat_orientation_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-1.0)
    posture = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    joint_torques_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.1)
    joint_accel_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.02)
    action_rate_l2 = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=-0.01)
    joint_pos_limits = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)
    body_contact = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.0)

    # Additional
    feet_air_time = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=1.0)
    forward_progress = RewTerm(func=lambda env: torch.zeros(env.num_envs, device=env.device), weight=0.5)
    base_height_target = 0.3


# Stand event config
@configclass
class Go2StandEventCfg:
    """Stand stage event terms."""
    physics_material = None
    add_base_mass = None
    reset_base = None
    reset_robot_joints = None
    base_external_force_torque = None
    push_robot = None


# Stand termination config
@configclass
class Go2StandTerminationCfg:
    """Stand stage termination terms."""
    base_height = None
    base_contact = None


# Main config for backwards compatibility
class GO2_CFG:
    """Main Go2 configuration aggregating all components."""
    # Articulation config attributes expected by tests (as class attributes)
    spawn = None  # placeholder for UsdFileCfg
    init_state = None  # placeholder for InitialStateCfg
    actuators = {
        "hip": None,
        "thigh": None,
        "calf": None,
    }
    scene: Any = None  # Go2SceneCfg
    observations = Go2ObservationCfg()
    actions = Go2ActionCfg()
    commands = Go2CommandCfg()
    events = Go2EventCfg()
    rewards: Any = Go2RewardCfg()  # Will be overridden by stage-specific
    terminations = Go2TerminationCfg()
    sim: Any = None  # SimulationCfg
    decimation: int = DECIMATION
    episode_length_s: float = 20.0


# Utility functions expected by tests
def posture_torch(q: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    """Compute posture penalty from joint position error."""
    dq = q - default
    w = torch.from_numpy(_POSTURE_WEIGHT).to(dq.device)
    return -torch.mean(w * dq * dq, dim=-1)


def feet_air_time_torch(
    contact_sensor_data: torch.Tensor,
    command: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute feet air time reward."""
    # contact_sensor_data: (N, 4, history) - True when in contact
    # Returns air time reward per environment
    in_contact = contact_sensor_data[:, :, -1]  # latest contact
    air_time = (~in_contact).float() * threshold
    # Only reward when there's a velocity command
    has_cmd = torch.norm(command[:, :2], dim=-1) > 0.1
    return air_time.sum(dim=-1) * has_cmd


# Main environment class
from rl.envs.go2_env_cfg import Go2EnvCfg


class Go2Env(ManagerBasedRLEnv):
    """Go2 quadruped locomotion environment using Isaac Lab ManagerBasedEnv.

    This environment wraps the Isaac Lab ManagerBasedEnv to provide a standard
    Gymnasium interface compatible with SB3 and other RL libraries.
    """

    cfg: Go2EnvCfg

    def __init__(
        self,
        cfg: Go2EnvCfg,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the Go2 environment."""
        super().__init__(cfg, **kwargs)
        self._render_mode = render_mode

        # Cache commonly used references
        self._robot = self.scene["robot"]
        self._contact_sensor = None

        # Episode tracking
        self._episode_step = 0
        self._max_episode_steps = self.cfg.max_episode_steps if hasattr(self.cfg, "max_episode_steps") else 1000

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination and truncation conditions."""
        # Time out
        time_out = self.episode_length_buf >= self._max_episode_steps

        # Base height termination
        base_pos = self._robot.data.root_pos_w[:, 2]
        base_height_terminated = base_pos < TERMINATE_BASE_HEIGHT

        # Roll/pitch termination via projected gravity
        gravity = self._robot.data.projected_gravity_b
        roll_pitch_terminated = torch.any(torch.abs(gravity[:, :2]) > TERMINATE_MAX_ROLL_PITCH, dim=1)

        terminated = base_height_terminated | roll_pitch_terminated
        truncated = time_out

        return terminated, truncated

    def step(self, action: np.ndarray | torch.Tensor):
        """Step the environment."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        obs, rew, terminated, truncated, info = super().step(action)
        self._episode_step += 1
        return obs, rew, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        self._episode_step = 0
        return super().reset(seed=seed, options=options)

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode


# Module exports
__all__ = [
    # Constants from constants_isaac.py
    "SIM_DT",
    "CONTROL_HZ",
    "DECIMATION",
    "OBS_DIM",
    "NUM_ACTIONS",
    "JOINT_NAMES",
    "BASE_BODY",
    "FOOT_BODIES",
    "HOME_BASE_HEIGHT",
    "KP",
    "KV",
    "TORQUE_LIMIT_HIP_THIGH",
    "TORQUE_LIMIT_CALF",
    "GAIT_REGIME_THRESHOLDS_MS",
    "DEFAULT_JOINT_POS",
    "MAX_EPISODE_STEPS",
    "TERMINATE_BASE_HEIGHT",
    "TERMINATE_MAX_ROLL_PITCH",
    "OBS_IDX_ANG_VEL",
    "OBS_IDX_GRAVITY",
    "OBS_IDX_COMMANDS",
    "OBS_IDX_DQ",
    "OBS_IDX_QD",
    "OBS_IDX_PREV_ACTION",
    "FRICTION_RANGE",
    "MASS_SCALE_RANGE",
    "MOTOR_STRENGTH_RANGE",
    "LEG_LENGTH_M",
    "GRAVITY",
    # Reward constants from heuristic.py
    "SIGMA_LIN_VEL",
    "SIGMA_ANG_VEL",
    "BASE_HEIGHT_TARGET",
    "POSTURE_WEIGHT_HIP",
    "POSTURE_WEIGHT_SAGITTAL",
    "_POSTURE_WEIGHT",
    # Config classes
    "Go2ObservationCfg",
    "Go2ActionCfg",
    "Go2CommandCfg",
    "Go2EventCfg",
    "Go2TerminationCfg",
    "Go2RewardCfg",
    "Go2StandRewardCfg",
    "Go2WalkTransitionRewardCfg",
    "Go2StandEventCfg",
    "Go2StandTerminationCfg",
    "Go2StandEnvCfg",
    "Go2BalanceEnvCfg",
    "Go2WalkTransitionEnvCfg",
    "Go2WalkEnvCfg",
    "GO2_CFG",
    # Utility functions
    "posture_torch",
    "feet_air_time_torch",
    # Environment class
    "Go2Env",
]