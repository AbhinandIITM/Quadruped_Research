#!/usr/bin/env python3
"""Smoke test for Isaac Lab Go2 environment across all 4 stages.

Runs a few steps of each stage to verify environment setup, physics,
observations, rewards, and termination without training.

Usage
-----
    python -m rl.isaac_lab.scripts.smoke_test --stages all --steps 100
    python -m rl.isaac_lab.scripts.smoke_test --stage walk --steps 200 --num-envs 4
"""

from __future__ import annotations

import argparse
import time
from typing import List

import torch

from isaaclab.envs import ManagerBasedRLEnv
from rl.isaac_lab.envs.go2_env_cfg import (
    Go2StandEnvCfg,
    Go2BalanceEnvCfg,
    Go2TransitionEnvCfg,
    Go2WalkEnvCfg,
)


STAGE_CONFIGS = {
    "stand": Go2StandEnvCfg,
    "balance": Go2BalanceEnvCfg,
    "transition": Go2TransitionEnvCfg,
    "walk": Go2WalkEnvCfg,
}


def run_stage_smoke(
    stage: str,
    num_envs: int = 4,
    steps: int = 100,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> bool:
    """Run a smoke test for a single stage.

    Returns True if the test passes (no errors, valid obs/rewards).
    """
    cfg_class = STAGE_CONFIGS[stage]
    cfg = cfg_class()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.device = device

    if verbose:
        print(f"\n{'='*60}")
        print(f"SMOKE TEST: {stage.upper()} (num_envs={num_envs}, steps={steps})")
        print(f"{'='*60}")

    try:
        env = ManagerBasedRLEnv(cfg)
    except Exception as e:
        print(f"  [FAIL] Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Reset (Isaac Lab 3.0 returns (obs, info))
        ret = env.reset()
        obs_dict = ret[0] if isinstance(ret, tuple) else ret
        obs = obs_dict["policy"]

        # Check observation shape
        expected_shape = (num_envs, 45)
        if obs.shape != expected_shape:
            print(f"  [FAIL] Observation shape mismatch: got {obs.shape}, expected {expected_shape}")
            return False
        if verbose:
            print(f"  [OK] Observation shape: {obs.shape}")

        # Check observation values are finite
        if not torch.isfinite(obs).all():
            print(f"  [FAIL] Observation contains NaN/Inf")
            return False

        # Run a few steps
        total_reward = 0.0
        episode_steps = 0
        terminated_count = 0

        start_time = time.time()
        for step in range(steps):
            # Random action in [-1, 1]
            action = torch.rand(num_envs, 12, device=device) * 2 - 1

            # Step
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]

            total_reward += reward.mean().item()
            episode_steps += 1

            # Check obs
            if not torch.isfinite(obs).all():
                print(f"  [FAIL] Observation NaN at step {step}")
                return False

            # Check reward
            if not torch.isfinite(reward).all():
                print(f"  [FAIL] Reward NaN at step {step}")
                return False

            # Check termination
            if terminated.any():
                terminated_count += terminated.sum().item()

            # Check done handling
            done = terminated | truncated
            if done.any():
                # Should auto-reset
                pass

            if verbose and (step % 50 == 0):
                print(f"  Step {step}: reward_mean={reward.mean():.3f}, terminated={terminated.sum().item()}")

        elapsed = time.time() - start_time
        fps = (num_envs * steps) / elapsed

        if verbose:
            print(f"  [OK] Completed {steps} steps in {elapsed:.2f}s ({fps:.0f} env-steps/s)")
            print(f"  [OK] Mean reward: {total_reward/episode_steps:.3f}")
            print(f"  [OK] Terminations: {terminated_count}")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] Error during step: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab Go2 Smoke Test")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["stand", "balance", "transition", "walk", "all"],
                        help="Stage to test")
    parser.add_argument("--stages", type=str, nargs="+",
                        default=None,
                        help="Stages to test (overrides --stage)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Steps per stage")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stages:
        stages = args.stages
    elif args.stage == "all":
        stages = ["stand", "balance", "transition", "walk"]
    else:
        stages = [args.stage]

    verbose = not args.quiet

    print(f"Isaac Lab Go2 Smoke Test")
    print(f"Stages: {stages}")
    print(f"Device: {args.device}")
    print(f"Num envs: {args.num_envs}")
    print(f"Steps per stage: {args.steps}")

    results = {}
    for stage in stages:
        results[stage] = run_stage_smoke(
            stage,
            num_envs=args.num_envs,
            steps=args.steps,
            device=args.device,
            seed=args.seed,
            verbose=verbose,
        )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for stage, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {stage:12s}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ All smoke tests PASSED")
        return 0
    else:
        print("\n✗ Some smoke tests FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())