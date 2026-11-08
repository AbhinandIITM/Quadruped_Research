"""Isaac Lab migration package for Go2 quadruped.

This package provides a complete Isaac Lab + Newton physics implementation
of the Go2 quadruped RL training stack, maintaining exact parity with the
mujoco-warp GPU stack.

Modules
-------
configs       — Stage configurations (stand/balance/transition/walk)
envs          — Go2DirectRLEnv with exact warp parity
rewards       — Reward terms matching gpu/env.py
curriculum    — VelocityCurriculum deterministic schedule
runner        — IsaacLabRunner wrapping RSL-RL OnPolicyRunner
eval_adapter  — load_isaac_policy for checkpoint evaluation
visualize     — Standalone Isaac Lab viewer for trained policies
scripts       — Training CLI and smoke tests
"""

from __future__ import annotations

__all__ = [
    "configs",
    "envs",
    "rewards",
    "curriculum",
    "runner",
    "eval_adapter",
    "visualize",
    "scripts",
]