"""Isaac Lab termination terms for Go2.

Ports termination conditions from the GPU stack to Isaac Lab TerminationManager.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Termination due to episode timeout.

    Returns True for envs that have reached max_episode_length.
    This is handled automatically by the environment, but we provide
    it as a termination term for explicit control.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    return env.episode_length_buf[env_ids] >= env.max_episode_length


def base_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimum_height: float = 0.16,
) -> torch.Tensor:
    """Termination when base height falls below minimum.

    Matches contracts.py TERMINATE_BASE_HEIGHT = 0.16
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    height = asset.data.root_pos_w[env_ids, 2]
    return height < minimum_height


def bad_orientation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_angle: float = 1.0,
) -> torch.Tensor:
    """Termination when base orientation exceeds limit.

    Checks projected gravity Z component (should be near -1 when upright).
    When |roll| or |pitch| > limit_angle, gravity vector deviates significantly.
    Matches contracts.py TERMINATE_MAX_ROLL_PITCH = 1.0 rad (~57 degrees)
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Projected gravity in body frame: [0, 0, -1] when upright
    gravity = asset.data.projected_gravity_b[env_ids]

    # Z component should be near -1; if it's > cos(limit_angle), we're tilted too much
    # cos(1.0) ≈ 0.54
    return gravity[:, 2] > -0.54  # > cos(1.0) means tilt > 1 rad


__all__ = ["time_out", "base_height", "bad_orientation"]