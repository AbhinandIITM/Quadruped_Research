"""Isaac Lab event terms for domain randomization.

Ports MuJoCoRandomizer from the GPU stack to Isaac Lab EventManager terms.
Each event: (env: ManagerBasedEnv, env_ids: torch.Tensor | None, ...) -> None
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    mass_distribution_params: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
) -> None:
    """Randomize rigid body mass using Isaac Lab's set_masses API."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get current body masses (N, num_bodies)
    masses = asset.data.body_mass[env_ids].clone()

    # Sample random scale factors per env (N,)
    scale = torch.rand(
        (len(env_ids),),
        dtype=masses.dtype,
        device=env.device,
    ) * (mass_distribution_params[1] - mass_distribution_params[0]) + mass_distribution_params[0]

    new_masses = masses * scale.unsqueeze(-1)
    # Use Isaac Lab's set_masses API
    asset.set_masses(new_masses, env_ids=env_ids)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    com_distribution_params: tuple[float, float] = (-0.02, 0.02),
    operation: str = "add",
) -> None:
    """Randomize center of mass position using Isaac Lab's set_coms API."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get current COM for root body (N, 3)
    com = asset.data.body_com[env_ids, 0].clone()  # root body index 0

    # Sample random offsets
    offset = torch.rand(
        (len(env_ids), 3),
        dtype=com.dtype,
        device=env.device,
    ) * (com_distribution_params[1] - com_distribution_params[0]) + com_distribution_params[0]

    new_com = com + offset
    # Use Isaac Lab's set_coms API (expects shape N, 3 for root body)
    asset.set_coms(new_com, env_ids=env_ids)


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    stiffness_range: tuple[float, float] = (0.8, 1.2),
    damping_range: tuple[float, float] = (0.8, 1.2),
    operation: str = "scale",
) -> None:
    """Randomize actuator stiffness and damping."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    actuators = asset.actuators
    for actuator in actuators.values():
        if operation == "scale":
            stiffness_scale = torch.rand(
                (len(env_ids),),
                dtype=torch.float32,
                device=env.device,
            ) * (stiffness_range[1] - stiffness_range[0]) + stiffness_range[0]
            damping_scale = torch.rand(
                (len(env_ids),),
                dtype=torch.float32,
                device=env.device,
            ) * (damping_range[1] - damping_range[0]) + damping_range[0]

            actuator.stiffness[env_ids] *= stiffness_scale.unsqueeze(-1)
            actuator.damping[env_ids] *= damping_scale.unsqueeze(-1)


def randomize_joint_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    friction_range: tuple[float, float] = (0.01, 0.1),
    operation: str = "abs",
) -> None:
    """Randomize joint friction."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Isaac Lab doesn't expose joint friction directly on the asset
    # This would need to be handled via physics material properties
    # or actuator friction if using DCMotor actuator model
    pass


def randomize_joint_armature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    armature_range: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
) -> None:
    """Randomize joint armature (inertia)."""
    # Armature is typically baked into the USD/URDF
    # In Isaac Lab, this would require modifying articulation properties
    pass


def push_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    velocity_range: dict = None,
    push_interval: int = 500,
    operation: str = "add",
) -> None:
    """Apply random external forces to robot base (pushes)."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    if velocity_range is None:
        velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)}

    # Only push at intervals
    if env.common_step_counter % push_interval == 0:
        velocity = torch.zeros(
            (len(env_ids), 6), dtype=torch.float32, device=env.device
        )
        velocity[:, 0] = torch.rand(len(env_ids), device=env.device) * \
            (velocity_range["x"][1] - velocity_range["x"][0]) + velocity_range["x"][0]
        velocity[:, 1] = torch.rand(len(env_ids), device=env.device) * \
            (velocity_range["y"][1] - velocity_range["y"][0]) + velocity_range["y"][0]
        velocity[:, 2] = torch.rand(len(env_ids), device=env.device) * \
            (velocity_range["z"][1] - velocity_range["z"][0]) + velocity_range["z"][0]

        # Apply as root velocity perturbation
        asset.data.root_lin_vel_w[env_ids] += velocity[:, :3]
        asset.data.root_ang_vel_w[env_ids] += velocity[:, 3:]


def reset_robot_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    position_range: dict = None,
    velocity_range: tuple[float, float] = (-0.1, 0.1),
    operation: str = "abs",
) -> None:
    """Reset robot joint positions and velocities."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    from rl.envs.constants_isaac import DEFAULT_JOINT_POS, JOINT_NAMES

    # Sample joint positions near default
    if position_range is None:
        pos_noise = 0.1
    else:
        pos_noise = position_range.get("noise", 0.1)

    joint_pos = torch.tensor(DEFAULT_JOINT_POS, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    joint_pos += torch.randn_like(joint_pos) * pos_noise

    joint_vel = torch.rand(
        (len(env_ids), asset.num_joints),
        dtype=torch.float32,
        device=env.device,
    ) * (velocity_range[1] - velocity_range[0]) + velocity_range[0]

    # Set joint state
    asset.data.joint_pos[env_ids] = joint_pos
    asset.data.joint_vel[env_ids] = joint_vel


def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: str = "robot",
    pose_range: dict = None,
    velocity_range: dict = None,
    operation: str = "abs",
) -> None:
    """Reset robot root state (position, orientation, velocities)."""
    asset = env.scene[asset_cfg]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Reset to default pose with noise
    from rl.envs.constants_isaac import HOME_BASE_HEIGHT

    if pose_range is None:
        pos_noise = 0.1
        rot_noise = 0.1
    else:
        pos_noise = pose_range.get("pos_noise", 0.1)
        rot_noise = pose_range.get("rot_noise", 0.1)

    # Position
    pos = torch.tensor([0.0, 0.0, HOME_BASE_HEIGHT], device=env.device).repeat(len(env_ids), 1)
    pos += torch.randn((len(env_ids), 3), device=env.device) * pos_noise
    pos[:, 2] = torch.clamp(pos[:, 2], HOME_BASE_HEIGHT - 0.05, HOME_BASE_HEIGHT + 0.05)

    # Orientation (quaternion w, x, y, z)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(len(env_ids), 1)
    # Add small random rotation
    rot_noise_tensor = torch.randn((len(env_ids), 3), device=env.device) * rot_noise
    # Convert to quaternion (approximate for small angles)
    quat[:, 1:] += rot_noise_tensor
    quat = torch.nn.functional.normalize(quat, dim=1)

    # Velocities
    lin_vel = torch.zeros((len(env_ids), 3), device=env.device)
    ang_vel = torch.zeros((len(env_ids), 3), device=env.device)
    if velocity_range is not None:
        lin_vel = torch.randn((len(env_ids), 3), device=env.device) * velocity_range.get("lin_scale", 0.1)
        ang_vel = torch.randn((len(env_ids), 3), device=env.device) * velocity_range.get("ang_scale", 0.1)

    asset.data.root_pos_w[env_ids] = pos
    asset.data.root_quat_w[env_ids] = quat
    asset.data.root_lin_vel_w[env_ids] = lin_vel
    asset.data.root_ang_vel_w[env_ids] = ang_vel


# Event configuration dict for EventManager
EVENT_CONFIG = {
    "startup": {
        "physics_material": {
            "func": randomize_rigid_body_mass,  # Placeholder
            "mode": "startup",
            "params": {
                "asset_cfg": "robot",
                "mass_distribution_params": (0.9, 1.1),
            },
        },
    },
    "reset": {
        "reset_robot_joints": {
            "func": reset_robot_joints,
            "mode": "reset",
            "params": {
                "asset_cfg": "robot",
                "position_range": {"noise": 0.1},
                "velocity_range": (-0.1, 0.1),
            },
        },
        "reset_root_state": {
            "func": reset_root_state,
            "mode": "reset",
            "params": {
                "asset_cfg": "robot",
                "pose_range": {"pos_noise": 0.1, "rot_noise": 0.1},
                "velocity_range": {"lin_scale": 0.1, "ang_scale": 0.1},
            },
        },
    },
    "interval": {
        "push_robot": {
            "func": push_robot,
            "mode": "interval",
            "interval_range_s": (5.0, 10.0),
            "params": {
                "asset_cfg": "robot",
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0)},
            },
        },
        "randomize_rigid_body_mass": {
            "func": randomize_rigid_body_mass,
            "mode": "interval",
            "interval_range_s": (10.0, 20.0),
            "params": {
                "asset_cfg": "robot",
                "mass_distribution_params": (0.9, 1.1),
            },
        },
        "randomize_rigid_body_com": {
            "func": randomize_rigid_body_com,
            "mode": "interval",
            "interval_range_s": (10.0, 20.0),
            "params": {
                "asset_cfg": "robot",
                "com_distribution_params": (-0.02, 0.02),
            },
        },
        "randomize_actuator_gains": {
            "func": randomize_actuator_gains,
            "mode": "interval",
            "interval_range_s": (10.0, 20.0),
            "params": {
                "asset_cfg": "robot",
                "stiffness_range": (0.8, 1.2),
                "damping_range": (0.8, 1.2),
            },
        },
    },
}


__all__ = [
    "randomize_rigid_body_mass",
    "randomize_rigid_body_com",
    "randomize_actuator_gains",
    "randomize_joint_friction",
    "randomize_joint_armature",
    "push_robot",
    "reset_robot_joints",
    "reset_root_state",
    "EVENT_CONFIG",
]