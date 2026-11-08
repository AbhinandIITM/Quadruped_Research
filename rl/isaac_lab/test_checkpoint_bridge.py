#!/usr/bin/env python3
"""Tests for ``rl.isaac_lab.checkpoint_bridge``.

Run with the repo root on ``sys.path``::

    PYTHONPATH=. python rl/isaac_lab/test_checkpoint_bridge.py

Tests
-----
1. ``test_direct_model_comparison`` -- sanity check that two models built with
   identical weights produce identical actions (rules out harness bugs).
2. ``test_rsl_to_sb3_loadable`` -- a converted ``.zip`` is a *real* SB3
   checkpoint: ``PPO.load`` reconstructs it and ``predict`` runs.
3. ``test_rsl_to_sb3_to_rsl_roundtrip`` -- the full rsl-rl -> SB3 -> rsl-rl
   round-trip preserves deterministic actions to 1e-5.
4. ``test_sb3_to_rsl_to_sb3_roundtrip`` -- the reverse SB3 -> rsl-rl -> SB3
   round-trip preserves deterministic actions to 1e-5.
5. ``test_verify_conversion_both_directions`` -- ``verify_conversion`` works
   for all three argument pairings (.pt/.zip, .zip/.pt, .pt/.pt).

The tests prefer a pre-trained checkpoint if one exists on disk, otherwise
they synthesise a tiny rsl-rl checkpoint in a temp dir and use that. Nothing
here touches a real environment, so the suite runs in seconds on CPU.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from rl.isaac_lab.checkpoint_bridge import (
    build_actor_model,
    convert_rsl_to_sb3,
    convert_sb3_to_rsl,
    extract_rsl_checkpoint,
    verify_conversion,
)

OBS_DIM = 45
ACTION_DIM = 12
HIDDEN_DIMS = (128, 128, 128)
ACTIVATION = "elu"
TOLERANCE = 1e-5

# Prefer a real trained checkpoint; fall back to one synthesised in a temp dir.
_CANDIDATE_CHECKPOINTS = [
    Path(__file__).resolve().parents[2] / "logs" / "rsl_rl" / "model_0.pt",
    Path(__file__).resolve().parents[2] / "logs" / "rsl_rl" / "model_final.pt",
]


def _find_or_make_checkpoint(tmpdir: Path) -> Path:
    """Return a pre-trained rsl-rl checkpoint, or synthesise one in ``tmpdir``."""
    for candidate in _CANDIDATE_CHECKPOINTS:
        if candidate.exists():
            print(f"  [info] using pre-trained checkpoint: {candidate}")
            return candidate

    print("  [info] no pre-trained checkpoint found; synthesising a tiny one")
    path = tmpdir / "model_0.pt"
    torch.manual_seed(0)
    actor = build_actor_model(OBS_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION)
    with torch.no_grad():
        for param in actor.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": build_actor_model(OBS_DIM, 1, HIDDEN_DIMS, ACTIVATION).state_dict(),
            "optimizer_state_dict": None,
            "iter": 0,
            "infos": {},
        },
        path,
    )
    return path


def _load_rsl_actor(path: Path, hidden_dims=HIDDEN_DIMS, activation=ACTIVATION):
    """Load a rsl-rl actor directly (bypassing the Isaac env machinery)."""
    extracted = extract_rsl_checkpoint(path, "cpu")
    state = extracted["actor_state_dict"]
    actor = build_actor_model(OBS_DIM, ACTION_DIM, list(hidden_dims), activation)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def _action_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_model_comparison() -> bool:
    """Sanity check: identical weights -> identical deterministic actions."""
    print("=" * 70)
    print("Test 1: Direct model comparison (identical weights)")
    print("=" * 70)

    model_a = build_actor_model(OBS_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION)
    model_b = build_actor_model(OBS_DIM, ACTION_DIM, HIDDEN_DIMS, ACTIVATION)
    model_b.load_state_dict(model_a.state_dict())

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(100):
            td = TensorDict({"policy": torch.randn(1, OBS_DIM)}, batch_size=[1])
            diff = _action_diff(
                model_a(td, stochastic_output=False), model_b(td, stochastic_output=False)
            )
            max_diff = max(max_diff, diff)

    print(f"  max action difference: {max_diff:.10f}")
    success = max_diff < 1e-7
    print(f"  Result: {'PASS' if success else 'FAIL'} (threshold 1e-7)")
    return success


def test_rsl_to_sb3_loadable(src_ckpt: Path) -> bool:
    """A converted .zip must be a genuine SB3 checkpoint."""
    print("=" * 70)
    print("Test 2: Converted SB3 .zip is loadable by stable_baselines3")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        sb3_path = Path(tmpdir) / "converted.zip"
        convert_rsl_to_sb3(
            src_ckpt,
            sb3_path,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=list(HIDDEN_DIMS),
            activation=ACTIVATION,
            device="cpu",
        )

        from stable_baselines3 import PPO

        model = PPO.load(sb3_path, device="cpu")
        print(f"  policy type : {type(model.policy).__name__}")
        print(f"  obs space   : {model.policy.observation_space}")
        print(f"  action space: {model.policy.action_space}")

        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        print(f"  sample action: {np.asarray(action)[:4]}")
        success = np.asarray(action).shape == (ACTION_DIM,) and np.all(np.isfinite(action))
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        return success


def test_rsl_to_sb3_to_rsl_roundtrip(src_ckpt: Path) -> bool:
    """rsl-rl .pt -> SB3 .zip -> rsl-rl .pt preserves actions."""
    print("=" * 70)
    print("Test 3: rsl-rl .pt -> SB3 .zip -> rsl-rl .pt round-trip")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        sb3_path = Path(tmpdir) / "converted.zip"
        rt_path = Path(tmpdir) / "roundtrip.pt"

        convert_rsl_to_sb3(
            src_ckpt,
            sb3_path,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=list(HIDDEN_DIMS),
            activation=ACTIVATION,
            device="cpu",
        )
        convert_sb3_to_rsl(sb3_path, rt_path, device="cpu")

        actor_orig = _load_rsl_actor(src_ckpt)
        actor_rt = _load_rsl_actor(rt_path)

        max_diff = 0.0
        with torch.no_grad():
            for _ in range(100):
                td = TensorDict({"policy": torch.randn(1, OBS_DIM)}, batch_size=[1])
                diff = _action_diff(
                    actor_orig(td, stochastic_output=False),
                    actor_rt(td, stochastic_output=False),
                )
                max_diff = max(max_diff, diff)

        print(f"  max action difference: {max_diff:.10f}")
        success = max_diff < TOLERANCE
        print(f"  Result: {'PASS' if success else 'FAIL'} (threshold {TOLERANCE:g})")
        return success


def test_sb3_to_rsl_to_sb3_roundtrip(src_ckpt: Path) -> bool:
    """SB3 .zip -> rsl-rl .pt -> SB3 .zip preserves actions (reverse direction)."""
    print("=" * 70)
    print("Test 4: SB3 .zip -> rsl-rl .pt -> SB3 .zip round-trip (reverse)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        sb3_path = Path(tmpdir) / "converted.zip"
        rsl_path = Path(tmpdir) / "roundtrip.pt"
        sb3_rt_path = Path(tmpdir) / "roundtrip.zip"

        # Baseline SB3 checkpoint produced from the rsl-rl source.
        convert_rsl_to_sb3(
            src_ckpt,
            sb3_path,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=list(HIDDEN_DIMS),
            activation=ACTIVATION,
            device="cpu",
        )
        convert_sb3_to_rsl(sb3_path, rsl_path, device="cpu")
        convert_rsl_to_sb3(
            rsl_path,
            sb3_rt_path,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=list(HIDDEN_DIMS),
            activation=ACTIVATION,
            device="cpu",
        )

        from stable_baselines3 import PPO

        model_orig = PPO.load(sb3_path, device="cpu")
        model_rt = PPO.load(sb3_rt_path, device="cpu")

        max_diff = 0.0
        for _ in range(100):
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            a1, _ = model_orig.predict(obs, deterministic=True)
            a2, _ = model_rt.predict(obs, deterministic=True)
            max_diff = max(max_diff, float(np.abs(a1 - a2).max()))

        print(f"  max action difference: {max_diff:.10f}")
        success = max_diff < TOLERANCE
        print(f"  Result: {'PASS' if success else 'FAIL'} (threshold {TOLERANCE:g})")
        return success


def test_verify_conversion_both_directions(src_ckpt: Path) -> bool:
    """verify_conversion itself works for every supported argument pairing."""
    print("=" * 70)
    print("Test 5: verify_conversion for all argument pairings")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        sb3_path = Path(tmpdir) / "converted.zip"
        rt_path = Path(tmpdir) / "roundtrip.pt"

        convert_rsl_to_sb3(
            src_ckpt,
            sb3_path,
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=list(HIDDEN_DIMS),
            activation=ACTIVATION,
            device="cpu",
        )
        convert_sb3_to_rsl(sb3_path, rt_path, device="cpu")

        ok_rt, diff_rt = verify_conversion(src_ckpt, sb3_path, device="cpu", num_tests=25)
        ok_direct, diff_direct = verify_conversion(src_ckpt, rt_path, device="cpu", num_tests=25)
        ok_rev, diff_rev = verify_conversion(sb3_path, rt_path, device="cpu", num_tests=25)

        print(f"  .pt/.zip (rsl->sb3 round-trip): {ok_rt} (diff {diff_rt:.2e})")
        print(f"  .pt/.pt (direct)              : {ok_direct} (diff {diff_direct:.2e})")
        print(f"  .zip/.pt (sb3->rsl round-trip): {ok_rev} (diff {diff_rev:.2e})")
        success = ok_rt and ok_direct and ok_rev
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        return success


def main() -> int:
    print("\ncheckpoint_bridge test suite\n")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_ckpt = _find_or_make_checkpoint(Path(tmpdir))

        results = [
            ("direct model comparison", test_direct_model_comparison()),
            ("SB3 .zip loadable", test_rsl_to_sb3_loadable(src_ckpt)),
            ("rsl->sb3->rsl round-trip", test_rsl_to_sb3_to_rsl_roundtrip(src_ckpt)),
            ("sb3->rsl->sb3 round-trip", test_sb3_to_rsl_to_sb3_roundtrip(src_ckpt)),
            ("verify_conversion pairings", test_verify_conversion_both_directions(src_ckpt)),
        ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        print(f"  {name:32s} {'PASS' if ok else 'FAIL'}")
        all_pass = all_pass and ok
    print("=" * 70)
    print("All tests PASSED!" if all_pass else "Some tests FAILED!")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

