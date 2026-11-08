"""DirectRLEnv-compatible reward functions for Go2.

These mirror the mujoco-warp GPU stack reward terms exactly, but are adapted
for DirectRLEnv which manages commands via self._commands buffer instead of
the ManagerBasedRLEnv command_manager.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from rl.contracts import (
    DEFAULT_JOINT_POS,
    ACTUATOR_TO_QPOS,
    ACTUATOR_TO_QVEL,
)


def _get_isaac_indices():
    """Convert contract DOF indices to Isaac Lab joint indices (0-11)."""
    return {
        "qpos": [int(x) - 7 for x in ACTUATOR_TO_QPOS],
        "qvel": [int(x) - 6 for x in ACTUATOR_TO_QVEL],
        "tau_write": [int(x) - 7 for x in ACTUATOR_TO_QPOS],
    }


_ISAAC_INDICES = _get_isaac_indices()
ACTUATOR_TO_QPOS_ISAAC = _ISAAC_INDICES["qpos"]
ACTUATOR_TO_QVEL_ISAAC = _ISAAC_INDICES["qvel"]
TAU_WRITE_ISAAC = _ISAAC_INDICES["tau_write"]


def track_lin_vel(env) -> torch.Tensor:
    """Exponential tracking of linear velocity in x-y plane."""
    asset = env._robot
    lin_vel_error = torch.sum(
        (asset.data.root_lin_vel_b[:, :2] - env._commands[:, :2]) ** 2,
        dim=1,
    )
    return torch.exp(-lin_vel_error / (0.25**2))  # SIGMA_LIN_VEL = 0.25


def track_ang_vel(env) -> torch.Tensor:
    """Exponential tracking of angular velocity around z."""
    asset = env._robot
    ang_vel_error = (asset.data.root_ang_vel_b[:, 2] - env._commands[:, 2]) ** 2
    return torch.exp(-ang_vel_error / (0.25**2))  # SIGMA_ANG_VEL = 0.25


def lin_vel_z_l2(env) -> torch.Tensor:
    """L2 penalty on vertical linear velocity."""
    asset = env._robot
    return asset.data.root_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(env) -> torch.Tensor:
    """L2 penalty on roll/pitch angular velocity."""
    asset = env._robot
    return torch.sum(asset.data.root_ang_vel_b[:, :2] ** 2, dim=1)


def base_height_l2(env, target_height: float = 0.27) -> torch.Tensor:
    """L2 penalty on base height deviation from target."""
    asset = env._robot
    return (asset.data.root_pos_w[:, 2] - target_height) ** 2


def flat_orientation_l2(env) -> torch.Tensor:
    """L2 penalty on orientation deviation from flat (projected gravity xy)."""
    asset = env._robot
    proj_gravity = quat_apply_inverse(
        asset.data.root_quat_w,
        torch.tensor([0.0, 0.0, -1.0], device=asset.device, dtype=torch.float32),
    )
    return proj_gravity[:, 0] ** 2 + proj_gravity[:, 1] ** 2


def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint torques."""
    asset = env._robot
    return torch.sum(asset.data.applied_torque**2, dim=1)


def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint velocity L2 (smoothness in GPU = -(qd*qd).mean())."""
    asset = env._robot
    joint_vel = asset.data.joint_vel[:, ACTUATOR_TO_QVEL_ISAAC]
    return torch.sum(joint_vel ** 2, dim=1) / 12.0


def normalized_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Normalized torque penalty: GPU uses tau = action (actual applied), normalized by per-joint torque limit.
    Provides -((tau/tau_limit)^2).mean().
    """
    asset = env._robot
    # Compute per-env torque limits
    torque_limits = torch.tensor(
        [23.7 if i % 3 != 2 else 45.43 for i in range(12)],
        device=asset.device, dtype=torch.float32,
    )
    tau = asset.data.applied_torque
    return torch.mean((tau / torque_limits) ** 2, dim=1)


def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint accelerations."""
    asset = env._robot
    return torch.sum(asset.data.joint_acc**2, dim=1)


def action_rate_l2(env) -> torch.Tensor:
    """L2 penalty on action rate (difference from prev action)."""
    return torch.sum((env._actions - env._prev_action) ** 2, dim=1)


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint position deviation from default (weighted)."""
    asset = env._robot
    joint_pos = asset.data.joint_pos[:, ACTUATOR_TO_QPOS_ISAAC]
    default_q = torch.as_tensor(DEFAULT_JOINT_POS, device=asset.device, dtype=torch.float32)
    dq = joint_pos - default_q
    # Weighted L2: posture weights [0.5, 1.0, 1.0] * 4
    posture_weights = torch.tensor([0.5, 1.0, 1.0] * 4, device=asset.device, dtype=torch.float32)
    return torch.sum((dq * posture_weights) ** 2, dim=1)


def feet_air_time(env, threshold: float = 0.5, sensor_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for feet air time using L2 kernel.

    Rewards steps longer than threshold. No reward for zero commands.
    """
    try:
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

        # No reward for zero command
        cmd = env._commands[:, :2]
        reward *= torch.linalg.norm(cmd, dim=1) > 0.1
        return reward
    except (AttributeError, KeyError):
        # Contact sensor not available
        return torch.zeros(env._num_envs, device=env._device)


def forward_progress(env) -> torch.Tensor:
    """Reward for forward progress along commanded heading."""
    asset = env._robot
    command = env._commands[:, :2]
    vel = asset.data.root_lin_vel_b[:, :2]
    return torch.sum(command * vel, dim=1)


# Exported for DirectRLEnv._get_rewards
STAGE_WEIGHTS = {
    "stand": {
        "track_lin_vel": 0.0,
        "track_ang_vel": 0.0,
        "posture": 2.0,
        "orientation": 2.0,
        "smoothness": 0.02,
        "action_rate": 0.01,
        "torque_limit": 0.1,
        "base_height": 2.0,
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
    "balance": {
        "track_lin_vel": 0.0,
        "track_ang_vel": 0.0,
        "posture": 2.0,
        "orientation": 2.0,
        "smoothness": 0.02,
        "action_rate": 0.01,
        "torque_limit": 0.1,
        "base_height": 2.0,
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
    "transition": {
        "track_lin_vel": 0.1,
        "track_ang_vel": 0.05,
        "posture": 2.0,
        "orientation": 1.0,
        "smoothness": 0.02,
        "action_rate": 0.01,
        "torque_limit": 0.1,
        "base_height": 1.0,
        "feet_air_time": 1.0,
        "forward_progress": 0.5,
    },
    "walk": {
        "track_lin_vel": 1.0,
        "track_ang_vel": 0.5,
        "posture": 0.2,
        "orientation": 0.3,
        "smoothness": 0.02,
        "action_rate": 0.01,
        "torque_limit": 0.1,
        "base_height": 0.2,
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
}


__all__ = [
    "track_lin_vel",
    "track_ang_vel",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "base_height_l2",
    "flat_orientation_l2",
    "joint_torques_l2",
    "joint_acc_l2",
    "joint_vel_l2",
    "normalized_torques_l2",
    "action_rate_l2",
    "joint_deviation_l2",
    "feet_air_time",
    "forward_progress",
    "STAGE_WEIGHTS",
]