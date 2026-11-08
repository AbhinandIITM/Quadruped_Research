"""MDP (Markov Decision Process) components for Go2 Isaac Lab environment.

Ports the mujoco-warp reward, observation, termination, and event logic
to Isaac Lab manager-based terms.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp import (
    base_lin_vel as isaac_base_lin_vel,
    JointPositionActionCfg,
    base_ang_vel as isaac_base_ang_vel,
    projected_gravity as isaac_projected_gravity,
    joint_pos_rel as isaac_joint_pos_rel,
    joint_vel_rel as isaac_joint_vel_rel,
    last_action as isaac_last_action,
    is_alive as isaac_is_alive,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Observations (use Isaac Lab built-ins where available)
# -----------------------------------------------------------------------------


def base_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base angular velocity in body frame. Shape: (N, 3)."""
    return isaac_base_ang_vel(env)


def base_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Base linear velocity in body frame. Shape: (N, 3)."""
    return isaac_base_lin_vel(env)


def projected_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Projected gravity vector in body frame. Shape: (N, 3)."""
    return isaac_projected_gravity(env)


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Generated commands from command manager. Shape: (N, 3) for base_velocity."""
    return env.command_manager.get_command(command_name)


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default. Shape: (N, 12)."""
    return isaac_joint_pos_rel(env, asset_cfg)


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities relative to default. Shape: (N, 12)."""
    return isaac_joint_vel_rel(env, asset_cfg)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Previous action. Shape: (N, 12)."""
    return isaac_last_action(env)


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Exponential tracking of linear velocity in x-y plane."""
    asset: Articulation = env.scene["robot"]
    lin_vel_error = torch.sum(
        (asset.data.root_lin_vel_b[:, :2] - env.command_manager.get_command(command_name)[:, :2]) ** 2,
        dim=1,
    )
    return torch.exp(-lin_vel_error / (std**2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Exponential tracking of angular velocity around z."""
    asset: Articulation = env.scene["robot"]
    ang_vel_error = (asset.data.root_ang_vel_b[:, 2] - env.command_manager.get_command(command_name)[:, 2]) ** 2
    return torch.exp(-ang_vel_error / (std**2))


def lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on vertical linear velocity."""
    asset: Articulation = env.scene["robot"]
    return asset.data.root_lin_vel_b[:, 2] ** 2


def ang_vel_xy_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on roll/pitch angular velocity."""
    asset: Articulation = env.scene["robot"]
    return torch.sum(asset.data.root_ang_vel_b[:, :2] ** 2, dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """L2 penalty on base height deviation from target."""
    asset: Articulation = env.scene[sensor_cfg.name]
    return (asset.data.root_pos_w[:, 2] - target_height) ** 2


def flat_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on orientation deviation from flat (projected gravity xy)."""
    asset: Articulation = env.scene["robot"]
    proj_gravity = quat_apply_inverse(asset.data.root_quat_w.torch, torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1))
    return torch.sum(proj_gravity[:, :2] ** 2, dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.applied_torque[:, asset_cfg.body_ids] ** 2, dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint acceleration."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.joint_acc[:, asset_cfg.body_ids] ** 2, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action rate (current - previous)."""
    return torch.sum((env.action_manager.action - env.action_manager.prev_action) ** 2, dim=1)


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """L2 penalty on joint position deviation from default."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum((asset.data.joint_pos[:, asset_cfg.body_ids] - asset.data.default_joint_pos[:, asset_cfg.body_ids]) ** 2, dim=1)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward for feet air time using L2 kernel.

    Rewards steps longer than threshold. No reward for zero commands.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # No reward for zero command
    cmd = env.command_manager.get_command(command_name)[:, :2]
    reward *= torch.linalg.norm(cmd, dim=1) > 0.1
    return reward


def forward_progress(
    env: ManagerBasedRLEnv, command_name: str
) -> torch.Tensor:
    """Reward for forward progress along commanded heading."""
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, :2]
    vel = asset.data.root_lin_vel_b[:, :2]
    return torch.sum(command * vel, dim=1)


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward +1 for each step the episode hasn't terminated (incl. timeouts)."""
    return isaac_is_alive(env)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the number of contact-force violations above threshold (e.g. thigh hits)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    return torch.sum(is_contact, dim=1).float()


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode timeout termination."""
    return env.episode_length_buf >= env.max_episode_length


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when base height falls below minimum."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when orientation exceeds limit (roll/pitch)."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_gravity = quat_apply_inverse(asset.data.root_quat_w.torch, torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1))
    return torch.linalg.norm(proj_gravity[:, :2], dim=1) > limit_angle


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


def push_robot(
    env: ManagerBasedRLEnv, velocity_range: dict[str, tuple[float, float]]
) -> None:
    """Apply random velocity impulse to robot base."""
    asset: Articulation = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    vx_range = velocity_range.get("x", (-0.5, 0.5))
    vy_range = velocity_range.get("y", (-0.5, 0.5))

    velocities = torch.zeros((num_envs, 6), device=device)
    velocities[:, 0] = torch.rand(num_envs, device=device) * (vx_range[1] - vx_range[0]) + vx_range[0]
    velocities[:, 1] = torch.rand(num_envs, device=device) * (vy_range[1] - vy_range[0]) + vy_range[0]

    asset.write_root_velocity_to_sim(asset.data.root_vel_w + velocities)


def randomize_rigid_body_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str,
) -> None:
    """Randomize rigid body mass at startup."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    if operation == "add":
        noise = torch.rand_like(masses[:, asset_cfg.body_ids]) * (mass_distribution_params[1] - mass_distribution_params[0]) + mass_distribution_params[0]
        masses[:, asset_cfg.body_ids] += noise
    elif operation == "scale":
        scale = torch.rand_like(masses[:, asset_cfg.body_ids]) * (mass_distribution_params[1] - mass_distribution_params[0]) + mass_distribution_params[0]
        masses[:, asset_cfg.body_ids] *= (1.0 + scale)
    asset.root_physx_view.set_masses(masses)


def randomize_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float],
    operation: str,
) -> None:
    """Randomize friction on specified bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Friction randomization is done via physics material
    # This is a placeholder - actual implementation depends on Isaac Lab version
    pass


def randomize_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float],
    operation: str,
) -> None:
    """Randomize joint friction."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Joint friction randomization
    pass


def resample_commands(env: ManagerBasedRLEnv, command_name: str) -> None:
    """Resample commands for curriculum."""
    env.command_manager.resample(env.reset_buf.nonzero(as_tuple=False).squeeze(-1), command_name)


def terrain_levels_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> None:
    """Curriculum for terrain levels based on velocity tracking."""
    # Placeholder - implement based on tracking performance
    pass


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------


class JointPositionActionCfg:
    """Joint position action configuration."""

    def __init__(
        self,
        asset_name: str,
        joint_names: list[str],
        scale: float = 1.0,
        use_default_offset: bool = True,
        clip: dict[str, tuple[float, float]] | None = None,
    ):
        self.asset_name = asset_name
        self.joint_names = joint_names
        self.scale = scale
        self.use_default_offset = use_default_offset
        self.clip = clip or {".*": (-1.0, 1.0)}