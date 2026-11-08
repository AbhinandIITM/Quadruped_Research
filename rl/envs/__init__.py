"""Environments for the LLM-vs-RL quadruped reward-source study."""

from __future__ import annotations

# Lazy imports to avoid circular dependency with rl.isaac_lab.envs.go2_env_cfg
# which imports rl.envs.terrain_isaac, which triggers this __init__.py

_go2_env = None


def __getattr__(name: str):
    global _go2_env
    if name == "Go2Env":
        if _go2_env is None:
            from .isaac_go2_env import Go2Env as _Go2Env
            _go2_env = _Go2Env
        return _go2_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Go2Env"]
