"""Isaac Lab reward term functions for Go2 locomotion.

Ports all heuristic reward terms from the MuJoCo stack (rl/heuristic.py)
to Isaac Lab reward term functions with proper signatures.
Each term: (env: ManagerBasedEnv, ...) -> torch.Tensor
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def track_lin_vel_xy(
    env: ManagerBasedEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward for tracking linear velocity commands in XY plane."""
    command = env.command_manager.get_command(command_name)
    vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error = torch.sum(torch.square(vel - command[:, :2]), dim=1)
    return torch.exp(-error / std)


def track_ang_vel_z(
    env: ManagerBasedEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward for tracking angular velocity command (yaw rate)."""
    command = env.command_manager.get_command(command_name)
    vel = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error = torch.square(vel - command[:, 2])
    return torch.exp(-error / std)


def base_height(
    env: ManagerBasedEnv,
    target_height: float = 0.27,
    std: float = 0.05,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Reward for maintaining target base height."""
    asset = env.scene[asset_cfg]
    height = asset.data.root_pos_w[:, 2]
    error = torch.square(height - target_height)
    return torch.exp(-error / std)


def upright(
    env: ManagerBasedEnv,
    std: float = 0.2,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Reward for keeping base upright (gravity vector aligned)."""
    asset = env.scene[asset_cfg]
    # projected gravity should be [0, 0, -1] when upright
    gravity = asset.data.projected_gravity_b
    # upright when projected gravity = [0, 0, -1], so z component near -1
    error = torch.sum(torch.square(gravity[:, :2]), dim=1)  # x,y components should be 0
    return torch.exp(-error / std)


def joint_pos_limits(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
    limit_factor: float = 0.9,
) -> torch.Tensor:
    """Penalty for exceeding joint position soft limits."""
    asset = env.scene[asset_cfg]
    joint_pos = asset.data.joint_pos
    joint_limits = asset.data.soft_joint_pos_limits
    lower = joint_limits[..., 0] * limit_factor
    upper = joint_limits[..., 1] * limit_factor
    violations = torch.relu(joint_pos - upper) + torch.relu(lower - joint_pos)
    return -torch.sum(violations, dim=1)


def torques(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for high joint torques (effort)."""
    asset = env.scene[asset_cfg]
    torques = asset.data.applied_torque
    return -torch.sum(torch.square(torques), dim=1)


def action_rate(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for large action changes between steps."""
    asset = env.scene[asset_cfg]
    if not hasattr(asset, "_prev_action"):
        asset._prev_action = asset.data.joint_pos.clone()
    action = asset.data.joint_pos - asset._prev_action
    asset._prev_action = asset.data.joint_pos.clone()
    return -torch.sum(torch.square(action), dim=1)


def dof_vel(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for high joint velocities."""
    asset = env.scene[asset_cfg]
    qd = asset.data.joint_vel
    return -torch.sum(torch.square(qd), dim=1)


def dof_acc(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for high joint accelerations."""
    asset = env.scene[asset_cfg]
    qdd = asset.data.joint_acc
    return -torch.sum(torch.square(qdd), dim=1)


def feet_air_time(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
    asset_cfg: str = "robot",
    min_air_time: float = 0.1,
    max_air_time: float = 0.5,
) -> torch.Tensor:
    """Reward for feet air time within target range (encourages gait).
    
    Uses foot body velocity z as proxy for contact detection (no contact sensor needed).
    """
    asset = env.scene[asset_cfg]
    
    # Find foot bodies
    feet_ids = [i for i, name in enumerate(asset.data.body_names) if 'foot' in name.lower()]
    if not feet_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Use foot velocity z as contact proxy: positive z = moving up/airborne
    feet_vel_z = asset.data.body_vel_w[:, feet_ids, 2]  # (N, num_feet)
    
    # Track air time per foot using velocity proxy
    if not hasattr(asset, "_air_time"):
        asset._air_time = torch.zeros(env.num_envs, len(feet_ids), device=env.device, dtype=torch.float32)
        asset._in_air_prev = torch.zeros(env.num_envs, len(feet_ids), device=env.device, dtype=torch.bool)
    
    # Contact proxy: foot is in contact if vel_z is near zero (not moving up)
    # Simple threshold: vel_z < 0.05 means likely in contact
    in_contact = feet_vel_z.abs() < 0.05
    in_air = ~in_contact
    
    # Update air time
    first_contact = asset._in_air_prev & in_contact
    last_contact = in_contact & ~asset._in_air_prev
    
    asset._air_time[first_contact] = 0.0
    asset._air_time[in_air] += env.step_dt * env.cfg.decimation
    
    # Reward air time in target range
    reward = torch.zeros(env.num_envs, device=env.device)
    # last_contact is (N, num_feet), get air_time for each foot that just landed
    landed_feet = last_contact.nonzero(as_tuple=True)
    if len(landed_feet[0]) > 0:
        air_times = asset._air_time[landed_feet]
        # Group by env and take max air time per env
        for env_idx in landed_feet[0].unique():
            env_air = air_times[landed_feet[0] == env_idx]
            max_air = env_air.max()
            reward[env_idx] = torch.clamp(max_air - min_air_time, min=0.0, max=max_air_time - min_air_time)
    
    asset._in_air_prev = in_air.clone()
    return reward


def feet_contact_forces(
    env: ManagerBasedEnv,
    threshold: float = 100.0,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for high contact forces on feet - using joint torques as proxy."""
    asset = env.scene[asset_cfg]
    # Find foot/calf joint indices
    feet_joint_ids = [i for i, name in enumerate(asset.data.joint_names) if 'foot' in name.lower() or 'calf' in name.lower()]
    if not feet_joint_ids:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Use joint torques as proxy for contact forces
    feet_torques = asset.data.applied_torque[:, feet_joint_ids]  # (N, num_feet_joints)
    force_magnitude = feet_torques.abs()  # torque magnitude as force proxy
    violations = torch.relu(force_magnitude - threshold)
    return -torch.sum(violations, dim=1)


def stand_still(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for keeping joints near default position (for standing)."""
    asset = env.scene[asset_cfg]
    from rl.envs.constants_isaac import DEFAULT_JOINT_POS
    default_pos = torch.tensor(DEFAULT_JOINT_POS, device=env.device)
    joint_pos = asset.data.joint_pos
    error = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    return torch.exp(-error / std)


def zero_velocity_command(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Zero reward when velocity command is zero (prevents standing-still hack)."""
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :3], dim=1)
    return (cmd_norm < 1e-3).float() * -10.0  # penalty for zero command


def base_lin_vel_z(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for vertical base velocity (bouncing)."""
    asset = env.scene[asset_cfg]
    vel_z = asset.data.root_lin_vel_w[:, 2]
    return -torch.square(vel_z)


def base_ang_vel_xy(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for base angular velocity in XY (roll/pitch rate)."""
    asset = env.scene[asset_cfg]
    ang_vel_xy = asset.data.root_ang_vel_b[:, :2]
    return -torch.sum(torch.square(ang_vel_xy), dim=1)


def joint_powers(
    env: ManagerBasedEnv,
    asset_cfg: str = "robot",
) -> torch.Tensor:
    """Penalty for joint power (torque * velocity)."""
    asset = env.scene[asset_cfg]
    torques = asset.data.applied_torque
    qd = asset.data.joint_vel
    power = torch.abs(torques * qd)
    return -torch.sum(power, dim=1)


def stand_still_zero_cmd(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
    asset_cfg: str = "robot",
    std: float = 0.1,
) -> torch.Tensor:
    """Stand-still reward ONLY when command is zero."""
    asset = env.scene[asset_cfg]
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :3], dim=1)

    from rl.envs.constants_isaac import DEFAULT_JOINT_POS
    default_pos = torch.tensor(DEFAULT_JOINT_POS, device=env.device)
    joint_pos = asset.data.joint_pos
    error = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    stand_reward = torch.exp(-error / std)

    return stand_reward * (cmd_norm < 1e-3).float()


# Weight dictionary matching the old heuristic.py weights
# These are the DEFAULT weights; can be overridden in env config
REWARD_WEIGHTS = {
    "track_lin_vel_xy": 1.0,
    "track_ang_vel_z": 0.5,
    "base_height": 0.5,
    "upright": 1.0,
    "joint_pos_limits": -0.1,
    "torques": -0.0002,
    "action_rate": -0.01,
    "dof_vel": -0.0001,
    "dof_acc": -0.00001,
    "feet_air_time": 1.0,
    "feet_contact_forces": -0.001,
    "stand_still_zero_cmd": 1.0,
    "base_lin_vel_z": -1.0,
    "base_ang_vel_xy": -0.5,
    "joint_powers": -0.0001,
}


__all__ = [
    "track_lin_vel_xy",
    "track_ang_vel_z",
    "base_height",
    "upright",
    "joint_pos_limits",
    "torques",
    "action_rate",
    "dof_vel",
    "dof_acc",
    "feet_air_time",
    "feet_contact_forces",
    "stand_still",
    "zero_velocity_command",
    "base_lin_vel_z",
    "base_ang_vel_xy",
    "joint_powers",
    "stand_still_zero_cmd",
    "REWARD_WEIGHTS",
]