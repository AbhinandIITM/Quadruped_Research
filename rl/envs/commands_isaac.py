"""Velocity command and curriculum manager for Go2 Isaac Lab migration.

Ports the VelocityCurriculum from rl/rl/curriculum.py to Isaac Lab's
CommandManager and CurriculumManager.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, MISSING
from typing import Any
import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg

from rl.envs.constants_isaac import (
    CONTROL_HZ,
    NUM_ACTIONS,
)

# Curriculum configuration matching rl/rl/curriculum.py


@configclass
class Go2VelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command configuration for Go2."""
    asset_name: str = "robot"
    heading_command: bool = True
    resampling_time_range: tuple[float, float] = (3.0, 5.0)
    rel_standing_envs: float = 0.05
    rel_heading_envs: float = 1.0

    # Velocity ranges (m/s and rad/s)
    ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
    )


class VelocityCurriculum:
    """Time-accurate command curriculum for Go2 locomotion.

    Port of rl/rl/curriculum.py VelocityCurriculum.
    The command issued at control step s is a pure function of the
    curriculum seed and s. Starts from zero commands, ramps up linearly
    over total_steps control steps.
    """

    def __init__(
        self,
        seed: int = 0,
        total_steps: int = 1_000_000,
        stages: list[dict] = None,
        cycle: bool = True,
        cycle_steps: int = 2_000_000,
    ):
        self._seed = seed
        self._total_steps = total_steps
        self._cycle = cycle
        self._cycle_steps = cycle_steps
        self._counter = 0

        if stages is None:
            from rl.isaac_lab.curriculum.velocity import DEFAULT_CURRICULUM_STAGES
            self._stages = [
                {"steps": s.n_steps, "command_range": {"vx": s.vx_range, "vy": (-s.vy_abs_max, s.vy_abs_max), "yaw": (-s.yaw_rate_abs_max, s.yaw_rate_abs_max)}}
                for s in DEFAULT_CURRICULUM_STAGES
            ]
        else:
            self._stages = stages

    def command_for(self, step: int) -> torch.Tensor:
        """Compute command (vx, vy, yaw_rate) for given global step counter."""
        idx = step % self._cycle_steps if self._cycle else min(step, self._cycle_steps)

        # Build cumulative schedule
        schedule = []
        start = 0
        for stage in self._stages:
            end = start + stage.get('steps', 10000)
            cmd_range = stage.get('command_range', {})
            vx_range = cmd_range.get('vx', (-0.5, 0.5))
            vy_range = cmd_range.get('vy', (-0.2, 0.2))
            yaw_range = cmd_range.get('yaw', (-0.5, 0.5))
            schedule.append((start, end, vx_range, vy_range, yaw_range))
            start = end

        # Find current stage
        current = schedule[0]
        for s, e, vxr, vyr, yawr in schedule:
            if s <= idx < e:
                current = (s, e, vxr, vyr, yawr)
                break
            elif idx >= e:
                current = (s, e, vxr, vyr, yawr)

        start, end, vx_range, vy_range, yaw_range = current
        progress = (idx - start) / max(1, end - start)
        progress = min(1.0, max(0.0, progress))

        # Scale command ranges by progress within stage
        vx_half = (vx_range[1] - vx_range[0]) / 2 * progress
        vy_half = (vy_range[1] - vy_range[0]) / 2 * progress
        yaw_half = (yaw_range[1] - yaw_range[0]) / 2 * progress

        import random
        rng = random.Random(self._seed + step)

        vx = rng.uniform(-vx_half, vx_half)
        vy = rng.uniform(-vy_half, vy_half)
        yaw = rng.uniform(-yaw_half, yaw_half)

        return torch.tensor([vx, vy, yaw])

    def __call__(self, env: ManagerBasedEnv) -> None:
        """Apply curriculum to environment."""
        if not hasattr(self, '_step_counter'):
            self._step_counter = 0

        cmds = self.command_for(self._step_counter)
        if hasattr(env, 'command_manager'):
            env.command_manager._commands = cmds.unsqueeze(0).repeat(env.num_envs, 1)
        self._step_counter += 1


@configclass
class CurriculumTerm:
    """Configuration of a curriculum term."""
    name: str = MISSING
    func: Any = MISSING

    # Curriculum levels
    max_velocity: float = 1.0
    min_velocity: float = 0.3
    ramp_steps: int = 100_000


def curriculum_command_scale(
    env: ManagerBasedEnv,
    term_name: str = "base_velocity",
) -> torch.Tensor:
    """Scale command ranges based on episode success."""
    # This is a placeholder for the CurriculumManager's actual protocol.
    # In real Isaac Lab, this would be called by the CurriculumManager
    # to adjust command ranges based on success metrics.
    return torch.tensor(0.0, device=env.device)


__all__ = [
    "Go2VelocityCommandCfg",
    "VelocityCurriculum",
    "CurriculumTerm",
    "curriculum_command_scale",
]