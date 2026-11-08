"""Isaac Lab command terms for Go2 velocity commands.

Provides the velocity_command observation term that returns
the current velocity command from the command manager.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def velocity_command(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Get velocity command from command manager.

    Returns 3D tensor: [lin_vel_x, lin_vel_y, ang_vel_z]
    Shape: (N, 3)
    """
    return env.command_manager.get_command(command_name)


__all__ = ["velocity_command"]