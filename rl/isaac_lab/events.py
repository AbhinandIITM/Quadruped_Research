"""Isaac Lab event terms for domain randomization.

Ports MuJoCoRandomizer from the GPU stack to Isaac Lab EventManager terms.
Each event: (env: ManagerBasedEnv, env_ids: torch.Tensor | None, ...) -> None
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_distribution_params: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
) -> None:
    """Randomize rigid body mass using Isaac Lab's set_masses API."""
    asset = env.scene[asset_cfg.name]
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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    com_distribution_params: tuple[float, float] = (-0.02, 0.02),
    operation: str = "add",
) -> None:
    """Randomize center of mass position using Isaac Lab's set_coms API."""
    asset = env.scene[asset_cfg.name]
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
    asset.set_coms(new_com, env_ids=env_ids)


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("terrain"),
    friction_distribution_params: tuple[float, float] | None = None,
    static_friction_range: tuple[float, float] = (0.6, 1.4),
    dynamic_friction_range: tuple[float, float] = (0.6, 1.4),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 256,
    make_consistent: bool = True,
) -> None:
    """Randomize physics material (friction and restitution) on terrain/ground plane using Isaac Lab's
    randomize_rigid_body_material event term.

    This uses the backend's tensor API (PhysX/Newton/OVPhysX) to modify physics materials at runtime.
    The material is randomized per environment by sampling from buckets.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all environments.
        asset_cfg: Scene entity config for the terrain/ground asset.
        friction_distribution_params: Legacy parameter - range for both static and dynamic friction.
            If provided, overrides static_friction_range and dynamic_friction_range.
        static_friction_range: Range for static friction coefficient.
        dynamic_friction_range: Range for dynamic friction coefficient.
        restitution_range: Range for restitution coefficient.
        num_buckets: Number of material buckets to pre-sample (PhysX limit: 64000).
        make_consistent: If True, ensures dynamic_friction <= static_friction.
    """
    from isaaclab.envs.mdp.events import randomize_rigid_body_material
    from isaaclab.managers import EventTermCfg

    # Backward compatibility: if friction_distribution_params is provided, use it for both
    if friction_distribution_params is not None:
        static_friction_range = friction_distribution_params
        dynamic_friction_range = friction_distribution_params

    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Create event term config matching the ManagerTermBase signature
    cfg = EventTermCfg(
        func=randomize_rigid_body_material,
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": static_friction_range,
            "dynamic_friction_range": dynamic_friction_range,
            "restitution_range": restitution_range,
            "num_buckets": num_buckets,
            "make_consistent": make_consistent,
        },
        mode="reset",
    )

    # Instantiate the event term and call it
    term = randomize_rigid_body_material(cfg, env)
    term(
        env,
        env_ids,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
        asset_cfg=asset_cfg,
        make_consistent=make_consistent,
    )


def randomize_joint_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    friction_distribution_params: tuple[float, float] = (0.01, 0.1),
    operation: str = "add",
) -> None:
    """Randomize joint friction using Isaac Lab's write_joint_friction_coefficient API."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get number of joints
    num_joints = asset.num_joints

    # Sample random friction values per env
    friction = torch.rand(
        (len(env_ids), num_joints),
        dtype=torch.float32,
        device=env.device,
    ) * (friction_distribution_params[1] - friction_distribution_params[0]) + friction_distribution_params[0]

    # Write to sim
    asset.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction,
        joint_ids=slice(None),
        env_ids=env_ids,
    )


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    strength_distribution_params: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
) -> None:
    """Randomize motor strength by scaling actuator stiffness and damping."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Get current stiffness/damping
    stiffness = asset.data.joint_stiffness[env_ids].clone()  # (N, num_joints)
    damping = asset.data.joint_damping[env_ids].clone()

    # Sample random scale factors per env
    scale = torch.rand(
        (len(env_ids), 1),
        dtype=stiffness.dtype,
        device=env.device,
    ) * (strength_distribution_params[1] - strength_distribution_params[0]) + strength_distribution_params[0]

    new_stiffness = stiffness * scale
    new_damping = damping * scale

    # Write to sim
    asset.write_joint_stiffness_to_sim_index(
        stiffness=new_stiffness,
        joint_ids=slice(None),
        env_ids=env_ids,
    )
    asset.write_joint_damping_to_sim_index(
        damping=new_damping,
        joint_ids=slice(None),
        env_ids=env_ids,
    )


def push_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_range: dict[str, tuple[float, float]] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.0),
    },
) -> None:
    """Apply random velocity push to robot base at reset."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Sample random push velocity
    num_envs = len(env_ids)
    push_vel = torch.zeros((num_envs, 3), device=env.device)

    for i, axis in enumerate(["x", "y", "z"]):
        lo, hi = velocity_range[axis]
        push_vel[:, i] = torch.rand(num_envs, device=env.device) * (hi - lo) + lo

    # Apply to root body linear velocity
    asset.data.root_lin_vel_b[env_ids] = push_vel


__all__ = [
    "randomize_rigid_body_mass",
    "randomize_rigid_body_com",
    "randomize_friction",
    "randomize_joint_friction",
    "randomize_motor_strength",
    "push_robot",
]