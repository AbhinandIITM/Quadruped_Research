"""Isaac Lab checkpoint loader for rsl-rl .pt checkpoints.

Provides ``load_isaac_policy`` — the Isaac Lab counterpart of the GPU
``load_gpu_policy`` adapter. Builds a minimal env + OnPolicyRunner, loads
the checkpoint weights, and returns a callable mapping observations to actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from tensordict import TensorDict

from rl.isaac_lab.envs.go2_env_cfg import Go2StandEnvCfg, Go2BalanceEnvCfg, Go2TransitionEnvCfg, Go2WalkEnvCfg, Go2FlatEnvCfg
from rl.isaac_lab.runner import IsaacLabRunner, _build_runner_cfg, _build_env_cfg
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg
from rl.contracts import OBS_DIM, NUM_ACTIONS
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class _InActorNormalization:
    """Truthy stand-in for ``NormalizedPolicyFn.vec_normalize`` on rsl-rl policies.

    The rsl-rl actor carries observation normalisation *inside* its weights
    (an ``EmpiricalNormalization`` module saved in ``actor_state_dict``) rather
    than as a separate ``VecNormalize`` object.
    """

    def __repr__(self) -> str:
        return "EmpiricalNormalization(in-actor)"


_IN_ACTOR_NORM = _InActorNormalization()


class IsaacPolicyFn:
    """Callable policy loaded from a rsl-rl .pt checkpoint.

    Mimics the SB3 ``NormalizedPolicyFn`` interface exactly:
    ``__call__(obs: np.ndarray, t: int, command: Any) -> np.ndarray``.
    """

    def __init__(
        self,
        runner: IsaacLabRunner,
        device: str,
        *,
        deterministic: bool = True,
    ) -> None:
        self.runner = runner
        self._device = torch.device(device)
        self.deterministic = deterministic
        self.n_calls = 0
        self.vec_normalize = _IN_ACTOR_NORM  # provenance flag for eval scripts

        # Ensure model is in eval mode
        self.runner._runner.alg.actor.eval()
        self.runner._runner.alg.critic.eval()

    def __call__(self, obs: np.ndarray, t: int = 0, command: Any = None) -> np.ndarray:
        """Forward pass through the actor.

        Args:
            obs: Raw observation array (45,).
            t: Timestep (unused, for interface parity).
            command: Command (unused, for interface parity).
        """
        self.n_calls += 1
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)  # (1, 45)

        # rsl-rl actor expects TensorDict with 'policy' key
        obs_td = TensorDict({"policy": obs_t}, batch_size=[1], device=self._device)

        with torch.no_grad():
            action_out = self.runner._runner.alg.actor(obs_td, stochastic_output=not self.deterministic)

        # rsl-rl actor may return a Tensor or a TensorDict
        if isinstance(action_out, TensorDict):
            action = action_out.get("action", None)
            if action is None:
                action = action_out
        else:
            action = action_out
        if isinstance(action, TensorDict):
            action = action.get("action", action)

        action_np = action.cpu().numpy().reshape(-1)
        return np.asarray(action_np, dtype=np.float64)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """SB3-compatible predict shim."""
        old_deterministic = self.deterministic
        self.deterministic = deterministic
        out = self(obs)
        self.deterministic = old_deterministic
        return out, None


def _resolve_device(device: str) -> str:
    """Map 'auto' to 'cuda' if available, else 'cpu'."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _infer_stage(checkpoint: str | Path) -> str:
    """Infer stage from checkpoint path (parent dir name)."""
    p = Path(checkpoint)
    parent = p.parent.name.lower()
    for stage in ("stand", "balance", "transition", "walk"):
        if stage in parent:
            return stage
    # Fallback: check if it's a numbered run dir
    import re
    m = re.search(r"gpu_run_(\d+)", parent)
    if m:
        return "walk"  # default
    return "walk"


def _ensure_app_launched_flag() -> None:
    """Set the ``ISAAC_LAUNCHED_FROM_TERMINAL`` builtins flag.

    Isaac Lab's ``ManagerBasedEnv`` refuses to reuse an existing
    ``SimulationContext`` singleton unless this flag is set (normally set by
    ``isaaclab.app.AppLauncher``). The eval adapter builds its own env inside
    an already-running Isaac Sim process, so it sets the flag before env
    construction instead of launching the full app.
    """
    import builtins

    if not hasattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL"):
        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True


# Cache of (stage, device) -> env for reuse across eval calls
_EVAL_ENV_CACHE: dict[tuple[str, str], ManagerBasedRLEnv] = {}


def _get_or_create_eval_env(stage: str, device: str):
    """Get or create a reusable eval env for the given stage/device.

    This avoids the SimulationContext singleton collision by reusing the same
    env across multiple policy loads for the same stage.
    """
    key = (stage, device)
    if key in _EVAL_ENV_CACHE:
        return _EVAL_ENV_CACHE[key]

    # When switching stages, we must clear the SimulationContext singleton
    # because each stage has a different scene configuration.
    if _EVAL_ENV_CACHE:
        # Clear all existing cached envs and the singleton
        _clear_eval_env_cache()

    # Allow env reuse of the SimulationContext singleton (no AppLauncher here).
    _ensure_app_launched_flag()

    # Build a minimal env + runner for loading (match training env type)
    env_cfg = _build_env_cfg(stage, num_envs=1, device=device)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.device = device

    # Create env (ManagerBasedRLEnv like training)
    env = ManagerBasedRLEnv(env_cfg)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    env.seed(0)

    _EVAL_ENV_CACHE[key] = env
    return env


def _clear_eval_env_cache() -> None:
    """Clear the eval env cache, closing all envs."""
    global _EVAL_ENV_CACHE
    for env in _EVAL_ENV_CACHE.values():
        try:
            env.close()
        except Exception:
            pass
    _EVAL_ENV_CACHE.clear()
    # Clear the SimulationContext singleton
    try:
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
    except Exception:
        pass


def load_isaac_policy(
    checkpoint: str | Path,
    stage: str | None = None,
    device: str = "auto",
    *,
    deterministic: bool = True,
) -> IsaacPolicyFn:
    """Load an Isaac Lab rsl-rl checkpoint and return an ``IsaacPolicyFn``.

    Args:
        checkpoint: Path to ``model_final.pt`` or run directory containing it.
        stage: Stage name (stand/balance/transition/walk). If None, inferred from path.
        device: Torch device ("cuda", "cpu", "auto").
        deterministic: Whether to use deterministic actions.

    Returns:
        ``IsaacPolicyFn`` callable with ``(obs, t, command) -> action`` signature.
    """
    ckpt = Path(checkpoint)
    if ckpt.is_dir():
        # Look for model_final.pt or latest model_*.pt
        final = ckpt / "model_final.pt"
        if final.exists():
            ckpt = final
        else:
            models = list(ckpt.glob("model_*.pt"))
            if not models:
                raise FileNotFoundError(f"No .pt checkpoint in {checkpoint}")
            ckpt = max(models, key=lambda p: int(p.stem.split("_")[-1]))

    if stage is None:
        stage = _infer_stage(ckpt)

    device = _resolve_device(device)

    # Get or create shared eval env for this stage/device
    env = _get_or_create_eval_env(stage, device)

    # Build runner config
    runner_cfg = _build_runner_cfg(stage, num_envs=1, device=device)
    runner_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # Migrate deprecated RSL-RL config (must match training)
    installed_version = check_rsl_rl_version()
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, installed_version)

    # Create runner with checkpoint's log_dir
    from isaaclab_rl.rsl_rl import create_rsl_rl_runner
    log_dir = str(ckpt.parent)
    runner_obj = create_rsl_rl_runner(env, runner_cfg, log_dir=log_dir)

    # Load checkpoint
    runner_obj.load(str(ckpt))

    # Wrap runner_obj in a lightweight holder
    class _Shim:
        pass

    il_runner = _Shim()
    il_runner._runner = runner_obj

    return IsaacPolicyFn(il_runner, device, deterministic=deterministic)


def is_isaac_checkpoint(path: str | Path) -> bool:
    """Check if path looks like an Isaac Lab rsl-rl checkpoint."""
    p = Path(path)
    if p.suffix == ".pt":
        return True
    if p.is_dir():
        return (p / "model_final.pt").exists() or len(list(p.glob("model_*.pt"))) > 0
    return False


__all__ = [
    "IsaacPolicyFn",
    "load_isaac_policy",
    "is_isaac_checkpoint",
    "_InActorNormalization",
    "_IN_ACTOR_NORM",
]