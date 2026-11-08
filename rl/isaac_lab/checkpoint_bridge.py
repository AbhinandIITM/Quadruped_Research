"""Checkpoint format bridge: rsl-rl .pt <-> SB3 .zip compatibility.

Converts between rsl-rl OnPolicyRunner checkpoints and stable-baselines3 PPO
checkpoints, in both directions, and verifies that policy behaviour survives
the round-trip.

What is preserved
-----------------
* Actor MLP weights (``mlp.0/2/4/6.{weight,bias}`` <-> ``mlp_extractor.policy_net.*``
  + ``action_net``). Layer *even* indices ``2*i`` hold the linear layers, so
  rsl-rl ``mlp.{2i}`` maps to SB3 ``mlp_extractor.policy_net.{2i}``.
* Exploration scale: rsl-rl ``distribution.std_param`` (scalar-space std) is
  converted to SB3 ``log_std`` with ``log_std = log(std)`` and back with
  ``std = exp(log_std)``. Deterministic actions never touch this value.
* Observation normalization stats (rsl-rl ``normalizer_state_dict`` /
  in-actor ``EmpiricalNormalization`` <-> SB3 ``VecNormalize.obs_rms``).

What is NOT preserved (documented limitations)
---------------------------------------------
* The **critic** is dropped on ``rsl -> sb3``. ``rsl-rl`` keeps actor and
  critic as separate state dicts inside one ``.pt``; the SB3 export only fills
  the policy head of a freshly-constructed ``ActorCriticPolicy`` from the
  *actor* weights, so the value head keeps SB3's random init (this is why the
  load reports missing ``value_net`` keys - expected, not a bug).
  Converting ``sb3 -> rsl`` therefore writes ``critic_state_dict=None``.
* The **optimizer** state is dropped on ``rsl -> sb3`` and cannot be restored;
  resumed training from a converted SB3 checkpoint restarts the optimizer.
* rsl-rl uses state-independent Gaussian exploration (state-independent
  ``std_param``), which matches SB3 PPO's default - but SB3 checkpoints using
  gSDE (``use_sde=True``) are not supported.
* The SB3 action space written by :func:`convert_rsl_to_sb3` is a wide
  ``Box(-10, 10)`` placeholder (SB3 refuses unbounded action spaces). rsl-rl
  behaviour is unchanged because actions come out of the network *before* any
  Box is consulted, but SB3-side code that clips to the action space will clip
  at +/-10.
* ``VecNormalize`` and ``EmpiricalNormalization`` use different update
  schedules (SB3 applies ``clip_obs=10`` by default). Only the *snapshot* of
  the stats at conversion time is preserved, not the normalizer's runtime
  behaviour.
* Only MLP policies with a single observation group are supported. RNN, CNN
  and multi-group (privileged / asymmetric) observations are not handled.

Usage
-----
>>> verify_conversion("model_0.pt", "model_0.zip")     # round-trip via a zip
True, 0.0
>>> verify_conversion("model_0.pt", "roundtrip.pt")     # direct .pt vs .pt
True, 0.0
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.models.mlp_model import MLPModel

ACTIVATIONS: Dict[str, type] = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
"""Supported activation functions, by lowercase name."""


# ---------------------------------------------------------------------------
# rsl-rl side helpers
# ---------------------------------------------------------------------------


def extract_rsl_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Dict[str, Any]:
    """Extract all components from rsl-rl checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    result = {
        "actor_state_dict": None,
        "critic_state_dict": None,
        "optimizer_state_dict": None,
        "normalizer_state_dict": None,
        "iteration": 0,
        "infos": {},
    }

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        # Full model state (actor + critic)
        model_state = ckpt["model_state_dict"]
        actor_state = {}
        critic_state = {}
        for k, v in model_state.items():
            if k.startswith("actor."):
                actor_state[k[6:]] = v
            elif k.startswith("critic."):
                critic_state[k[7:]] = v
        result["actor_state_dict"] = actor_state
        result["critic_state_dict"] = critic_state
    elif "actor_state_dict" in ckpt:
        result["actor_state_dict"] = ckpt["actor_state_dict"]
        result["critic_state_dict"] = ckpt.get("critic_state_dict")

    result["optimizer_state_dict"] = ckpt.get("optimizer_state_dict")
    result["normalizer_state_dict"] = ckpt.get("normalizer_state_dict")
    result["iteration"] = ckpt.get("iter", ckpt.get("iteration", 0))
    result["infos"] = ckpt.get("infos", {})

    # Also check for obs_norm directly
    if "obs_norm" in ckpt:
        result["obs_norm"] = ckpt["obs_norm"]

    return result


def _action_dim_of(actor_state_dict: Dict[str, torch.Tensor]) -> int:
    """Infer the action dimension from an rsl-rl actor state dict."""
    if "distribution.std_param" in actor_state_dict:
        return int(actor_state_dict["distribution.std_param"].shape[0])
    if "distribution.log_std_param" in actor_state_dict:
        return int(actor_state_dict["distribution.log_std_param"].shape[0])
    # Fall back on the final MLP layer's output features.
    final = max(
        int(k.split(".")[1]) for k in actor_state_dict if k.startswith("mlp.") and k.endswith(".bias")
    )
    return int(actor_state_dict[f"mlp.{final}.bias"].shape[0])


def build_actor_model(
    obs_dim: int = 45,
    action_dim: int = 12,
    hidden_dims: list[int] = (128, 128, 128),
    activation: str = "elu",
    init_std: float = 1.0,
    std_type: str = "scalar",
    device: str = "cpu",
) -> MLPModel:
    """Build the rsl-rl actor (``MLPModel`` + Gaussian exploration distribution).

    The returned model matches the architecture rsl-rl's ``MLPModel`` builds
    from ``actor=RslRlMLPModelCfg(hidden_dims=..., activation="elu",
    distribution_cfg=...)`` - i.e. a plain ``MLP`` plus a state-independent
    Gaussian head - so real checkpoints and freshly-constructed models share
    the very same state-dict layout.
    """
    if activation.lower() not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{activation}'. Expected one of {sorted(ACTIVATIONS)}.")

    # MLPModel computes its obs_dim from the TensorDict, so hand it a zero batch.
    dummy_obs = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    distribution_cfg = {
        "class_name": "rsl_rl.modules.GaussianDistribution",
        "init_std": float(init_std),
        "std_type": std_type,
        "learn_std": True,
    }

    model = MLPModel(
        obs=dummy_obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=list(hidden_dims),
        activation=activation.lower(),
        obs_normalization=False,
        distribution_cfg=distribution_cfg,
    )
    model.eval()
    return model.to(device)


def _extract_obs_norm(extracted: Dict[str, Any], obs_dim: int) -> Dict[str, Any] | None:
    """Pull observation-normalization stats out of an extracted rsl-rl checkpoint."""
    if extracted.get("normalizer_state_dict") is not None:
        norm = EmpiricalNormalization(shape=(obs_dim,), until=1_000_000_000)
        norm.load_state_dict(extracted["normalizer_state_dict"])
        return {
            "mean": norm.running_mean.detach().cpu().numpy(),
            "var": norm.running_var.detach().cpu().numpy(),
            "count": float(norm.count.item()),
        }

    norm = extracted.get("obs_norm")
    if hasattr(norm, "running_mean"):
        return {
            "mean": norm.running_mean.detach().cpu().numpy(),
            "var": norm.running_var.detach().cpu().numpy(),
            "count": float(norm.count.item()),
        }
    return None


# ---------------------------------------------------------------------------
# rsl-rl -> SB3
# ---------------------------------------------------------------------------


def _make_dummy_env(obs_space, act_space):
    """Build a throwaway Gym env satisfying SB3's constructor type checks."""
    import gymnasium as gym

    class _DummyEnv(gym.Env):
        observation_space = obs_space
        action_space = act_space

        def reset(self, *, seed=None, options=None):  # pragma: no cover - never stepped
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):  # pragma: no cover - never stepped
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

    return _DummyEnv()


def convert_rsl_to_sb3(
    rsl_checkpoint: str | Path,
    sb3_output: str | Path,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    hidden_dims: list[int] = (128, 128, 128),
    activation: str = "elu",
    init_std: float = 1.0,
    std_type: str = "scalar",
    device: str = "cpu",
) -> None:
    """Convert an rsl-rl checkpoint to a real SB3 PPO checkpoint (``.zip``).

    The result is written through ``stable_baselines3.BaseAlgorithm.save``, so
    ``PPO.load(path)`` reconstructs it - policy, normalizer and all.

    Args:
        rsl_checkpoint: Path to the rsl-rl ``.pt`` checkpoint.
        sb3_output: Output path for the SB3 ``.zip``.
        obs_dim: Observation dimension. ``None`` infers it from the checkpoint.
        action_dim: Action dimension. ``None`` infers it from the checkpoint.
        hidden_dims: Policy network architecture of the source checkpoint.
        activation: Activation function of the source checkpoint.
        init_std: Fallback exploration std for the SB3 policy.
        std_type: Exploration parameterization of the rsl-rl checkpoint.
        device: Torch device to build the SB3 model on.
    """
    from stable_baselines3 import PPO
    from gymnasium import spaces

    rsl_checkpoint = Path(rsl_checkpoint)
    sb3_output = Path(sb3_output)

    extracted = extract_rsl_checkpoint(rsl_checkpoint, device)
    rsl_actor = extracted["actor_state_dict"]
    if rsl_actor is None:
        raise ValueError(f"No actor state dict found in {rsl_checkpoint}")
    rsl_actor = {k: v.to(device) for k, v in rsl_actor.items()}

    # Infer shapes from the checkpoint so callers need not know the architecture.
    if action_dim is None:
        action_dim = _action_dim_of(rsl_actor)
    if obs_dim is None:
        obs_dim = int(rsl_actor["mlp.0.weight"].shape[1])

    obs_norm_data = _extract_obs_norm(extracted, obs_dim)

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32)

    policy_kwargs = dict(
        net_arch=list(hidden_dims),
        activation_fn=ACTIVATIONS[activation.lower()],
    )
    model = PPO(
        "MlpPolicy",
        _make_dummy_env(obs_space, act_space),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )

    # rsl-rl's linear layers live at even indices 0,2,4,...; SB3's policy_net
    # uses the same convention for its hidden layers, with the action head split
    # out into a standalone ``action_net``.
    with torch.no_grad():
        sb3_state: Dict[str, torch.Tensor] = {}
        for i in range(len(hidden_dims)):
            idx = 2 * i
            if f"mlp.{idx}.weight" in rsl_actor:
                sb3_state[f"mlp_extractor.policy_net.{idx}.weight"] = rsl_actor[f"mlp.{idx}.weight"]
                sb3_state[f"mlp_extractor.policy_net.{idx}.bias"] = rsl_actor[f"mlp.{idx}.bias"]
        final_idx = 2 * len(hidden_dims)
        if f"mlp.{final_idx}.weight" in rsl_actor:
            sb3_state["action_net.weight"] = rsl_actor[f"mlp.{final_idx}.weight"]
            sb3_state["action_net.bias"] = rsl_actor[f"mlp.{final_idx}.bias"]

        # scalar std -> log std (affects exploration only, not deterministic actions)
        if "distribution.std_param" in rsl_actor:
            sb3_state["log_std"] = torch.log(rsl_actor["distribution.std_param"].clamp(min=1e-6))
        elif "distribution.log_std_param" in rsl_actor:
            sb3_state["log_std"] = rsl_actor["distribution.log_std_param"].clone()

        missing, unexpected = model.policy.load_state_dict(sb3_state, strict=False)
        # The value head is intentionally left at SB3's random init (see docstring).
        real_missing = sorted(k for k in missing if "value_net" not in k)
        if real_missing:
            print(f"[Warning] Missing keys when writing SB3 checkpoint: {real_missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys when writing SB3 checkpoint: {sorted(unexpected)}")

    if obs_norm_data is not None:
        _attach_vec_normalize(model, obs_norm_data)

    model.save(sb3_output)
    print(f"Converted {rsl_checkpoint} -> {sb3_output} (SB3-compatible)")


def _attach_vec_normalize(model, obs_norm_data: Dict[str, Any]) -> None:
    """Attach rsl-rl normalizer stats to an SB3 model as a ``VecNormalize``."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    obs_space = model.policy.observation_space
    act_space = model.policy.action_space
    venv = DummyVecEnv([lambda: _make_dummy_env(obs_space, act_space)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean = obs_norm_data["mean"].astype(np.float32)
    venv.obs_rms.var = np.maximum(obs_norm_data["var"].astype(np.float32), 1e-8)
    venv.obs_rms.count = max(float(obs_norm_data["count"]), 1e-4)
    model.vec_normalize = venv


# ---------------------------------------------------------------------------
# SB3 -> rsl-rl
# ---------------------------------------------------------------------------


def convert_sb3_to_rsl(
    sb3_checkpoint: str | Path,
    rsl_output: str | Path,
    device: str = "cpu",
) -> None:
    """Convert an SB3 PPO checkpoint to rsl-rl format (``.pt``).

    The architecture (hidden dims, activation) and - when present - the
    ``VecNormalize`` stats are read back from the SB3 checkpoint, so this
    function takes no explicit architecture arguments.

    Args:
        sb3_checkpoint: Path to a ``PPO.load``-able SB3 ``.zip``.
        rsl_output: Output path for the rsl-rl ``.pt``.
        device: Torch device to build the actor on.
    """
    from stable_baselines3 import PPO

    sb3_checkpoint = Path(sb3_checkpoint)
    rsl_output = Path(rsl_output)

    model = PPO.load(sb3_checkpoint, device=device)
    policy = model.policy

    obs_dim = int(policy.observation_space.shape[0])
    action_dim = int(policy.action_space.shape[0])

    hidden_dims = [
        int(layer.out_features)
        for layer in policy.mlp_extractor.policy_net
        if isinstance(layer, nn.Linear)
    ]
    activation = "elu"
    for layer in policy.mlp_extractor.policy_net:
        if isinstance(layer, nn.ELU):
            activation = "elu"
        elif isinstance(layer, nn.ReLU):
            activation = "relu"
        elif isinstance(layer, nn.Tanh):
            activation = "tanh"

    actor = build_actor_model(obs_dim, action_dim, hidden_dims, activation, device=device)

    sb3_state = {k: v.to(device) for k, v in policy.state_dict().items()}
    rsl_state: Dict[str, torch.Tensor] = {}
    for i in range(len(hidden_dims)):
        idx = 2 * i
        w_key = f"mlp_extractor.policy_net.{idx}.weight"
        b_key = f"mlp_extractor.policy_net.{idx}.bias"
        if w_key in sb3_state:
            rsl_state[f"mlp.{idx}.weight"] = sb3_state[w_key]
            rsl_state[f"mlp.{idx}.bias"] = sb3_state[b_key]

    final_idx = 2 * len(hidden_dims)
    if "action_net.weight" in sb3_state:
        rsl_state[f"mlp.{final_idx}.weight"] = sb3_state["action_net.weight"]
        rsl_state[f"mlp.{final_idx}.bias"] = sb3_state["action_net.bias"]
    if "log_std" in sb3_state:
        rsl_state["distribution.std_param"] = torch.exp(sb3_state["log_std"])

    actor.load_state_dict(rsl_state)
    actor.eval()

    obs_norm_state = None
    vec_normalize = getattr(model, "vec_normalize", None)
    if vec_normalize is not None:
        obs_norm_state = {
            "running_mean": torch.as_tensor(vec_normalize.obs_rms.mean, dtype=torch.float32, device=device),
            "running_var": torch.as_tensor(vec_normalize.obs_rms.var, dtype=torch.float32, device=device),
            "count": torch.tensor(float(vec_normalize.obs_rms.count), device=device),
        }

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": None,  # dropped: see module docstring
            "optimizer_state_dict": None,  # dropped: see module docstring
            "normalizer_state_dict": obs_norm_state,
            "iter": 0,
            "infos": {},
        },
        rsl_output,
    )
    print(f"Converted {sb3_checkpoint} -> {rsl_output}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_conversion(
    original_ckpt: str | Path,
    converted_ckpt: str | Path,
    device: str = "cpu",
    num_tests: int = 100,
    obs_dim: int | None = None,
    action_dim: int | None = None,
    hidden_dims: list[int] = (128, 128, 128),
    activation: str = "elu",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[bool, float]:
    """Verify that a conversion preserves policy behaviour.

    Both directions are supported; the direction is taken from the file
    extensions of the two arguments:

    * ``.pt`` + ``.zip`` -> round-trips the converted zip back to ``.pt``
      (``rsl -> sb3 -> rsl``) and compares against the original ``.pt``.
    * ``.zip`` + ``.pt`` -> round-trips the converted ``.pt`` back to ``.zip``
      (``sb3 -> rsl -> sb3``) and compares against the original ``.zip``.
    * ``.pt`` + ``.pt`` -> compares the two rsl-rl checkpoints directly (use
      this when you have already produced the round-trip file yourself).

    Returns ``(success, max_abs_action_difference)``. Comparison uses the
    *deterministic* action (the distribution mean), which is what deployment
    and evaluation consume; stochastic sampling would differ by construction.
    """
    original_ckpt = Path(original_ckpt)
    converted_ckpt = Path(converted_ckpt)
    orig_is_pt = original_ckpt.suffix == ".pt"
    conv_is_zip = converted_ckpt.suffix == ".zip"
    tolerance = max(rtol, atol)

    if orig_is_pt and conv_is_zip:
        # rsl-rl -> SB3 -> rsl-rl
        with tempfile.TemporaryDirectory() as tmpdir:
            roundtrip = Path(tmpdir) / "roundtrip.pt"
            convert_sb3_to_rsl(converted_ckpt, roundtrip, device=device)
            return _compare_rsl_checkpoints(
                original_ckpt, roundtrip, device, num_tests, hidden_dims, tolerance
            )

    if not orig_is_pt and not conv_is_zip:
        # SB3 -> rsl-rl -> SB3
        with tempfile.TemporaryDirectory() as tmpdir:
            roundtrip = Path(tmpdir) / "roundtrip.zip"
            extracted = extract_rsl_checkpoint(converted_ckpt, device)
            if extracted["actor_state_dict"] is None:
                raise ValueError(f"No actor state dict found in {converted_ckpt}")
            if obs_dim is None:
                obs_dim = int(extracted["actor_state_dict"]["mlp.0.weight"].shape[1])
            if action_dim is None:
                action_dim = _action_dim_of(extracted["actor_state_dict"])
            convert_rsl_to_sb3(
                converted_ckpt, roundtrip, obs_dim, action_dim, hidden_dims, activation, device=device
            )
            return _compare_sb3_checkpoints(
                original_ckpt, roundtrip, device, num_tests, obs_dim, tolerance
            )

    if orig_is_pt and not conv_is_zip:
        return _compare_rsl_checkpoints(
            original_ckpt, converted_ckpt, device, num_tests, hidden_dims, tolerance
        )

    raise ValueError(
        f"Unsupported checkpoint combination: {original_ckpt.suffix} + {converted_ckpt.suffix}. "
        "Expected any pairing of .pt and .zip."
    )


def _load_rsl_actor(path: Path, hidden_dims: list[int], device: str) -> MLPModel:
    """Load a rsl-rl actor from a ``.pt`` checkpoint."""
    extracted = extract_rsl_checkpoint(path, device)
    if extracted["actor_state_dict"] is None:
        raise ValueError(f"No actor state dict found in {path}")
    state = extracted["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    actor = build_actor_model(obs_dim, _action_dim_of(state), hidden_dims, device=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def _compare_rsl_checkpoints(
    ckpt_a: Path,
    ckpt_b: Path,
    device: str,
    num_tests: int,
    hidden_dims: list[int],
    tolerance: float,
) -> Tuple[bool, float]:
    """Run two rsl-rl actors on identical observations and diff the actions."""
    actor_a = _load_rsl_actor(ckpt_a, hidden_dims, device)
    actor_b = _load_rsl_actor(ckpt_b, hidden_dims, device)

    obs_dim = int(actor_a.obs_dim)
    if int(actor_b.obs_dim) != obs_dim:
        raise ValueError(f"Observation dim mismatch: {actor_a.obs_dim} vs {actor_b.obs_dim}")

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(num_tests):
            td = TensorDict(
                {"policy": torch.randn(1, obs_dim, device=device)}, batch_size=[1], device=device
            )
            act_a = actor_a(td, stochastic_output=False)
            act_b = actor_b(td, stochastic_output=False)
            max_diff = max(max_diff, (act_a - act_b).abs().max().item())

    success = max_diff < tolerance
    print(f"Verification: max action difference = {max_diff:.8f} (threshold: {tolerance:g})")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success, max_diff


def _compare_sb3_checkpoints(
    ckpt_a: Path,
    ckpt_b: Path,
    device: str,
    num_tests: int,
    obs_dim: int,
    tolerance: float,
) -> Tuple[bool, float]:
    """Run two SB3 policies on identical observations and diff the actions."""
    from stable_baselines3 import PPO

    model_a = PPO.load(ckpt_a, device=device)
    model_b = PPO.load(ckpt_b, device=device)

    max_diff = 0.0
    for _ in range(num_tests):
        obs = np.random.randn(obs_dim).astype(np.float32)
        act_a, _ = model_a.predict(obs, deterministic=True)
        act_b, _ = model_b.predict(obs, deterministic=True)
        max_diff = max(max_diff, float(np.abs(act_a - act_b).max()))

    success = max_diff < tolerance
    print(f"Verification: max action difference = {max_diff:.8f} (threshold: {tolerance:g})")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    return success, max_diff


__all__ = [
    "convert_rsl_to_sb3",
    "convert_sb3_to_rsl",
    "verify_conversion",
    "extract_rsl_checkpoint",
    "build_actor_model",
]
