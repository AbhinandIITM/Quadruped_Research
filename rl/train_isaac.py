#!/usr/bin/env python3
"""Unified Isaac Lab training CLI with multi-stage driver.

Usage:
    python -m rl.train_isaac --stage walk --steps 20000000 --seed 0
    python -m rl.train_isaac --stage all --steps 5000000 --num-seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add rl to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.isaac_lab.runner import train_stage, _build_env_cfg
from rl.isaac_lab.scripts.train_isaac_multiseed import train_multistage_multiseed


def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Lab Go2 Training")
    parser.add_argument("--stage", type=str, default="walk",
                        choices=["stand", "balance", "transition", "walk", "all"],
                        help="Training stage")
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Total environment steps")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run directory")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity")
    parser.add_argument("--max-wall-clock-s", type=int, default=86400,
                        help="Max wall clock time (seconds)")
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Number of seeds for multi-seed training")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Specific seeds (overrides --num-seeds)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == "all":
        stages = ["stand", "balance", "transition", "walk"]
    else:
        stages = [args.stage]

    seeds = args.seeds if args.seeds is not None else list(range(args.num_seeds))

    if len(stages) == 1 and len(seeds) == 1:
        # Single stage, single seed
        print(f"Training {stages[0]} for {args.steps} steps, seed={seeds[0]}")
        runner = train_stage(
            stage=stages[0],
            seed=seeds[0],
            total_steps=args.steps,
            run_dir=args.run_dir,
            resume=args.resume,
            device=args.device,
            num_envs=args.num_envs,
            verbose=args.verbose,
        )
        print(f"Run directory: {runner.run_json_path}")
    else:
        # Multi-stage or multi-seed
        print(f"Training stages: {stages}")
        print(f"Seeds: {seeds}")
        print(f"Steps per stage: {args.steps}")
        print(f"Envs: {args.num_envs}")

        results = train_multistage_multiseed(
            stages=stages,
            seeds=seeds,
            steps_per_stage=args.steps,
            num_envs=args.num_envs,
            device=args.device,
            run_dir=args.run_dir,
            verbose=args.verbose,
            max_parallel=1 if args.device == "cuda" else len(seeds),
        )

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 60)
        for stage, agg in results.items():
            print(f"  {stage:12s}: reward={agg['reward_mean']:.3f}±{agg['reward_std']:.3f}, "
                  f"ep_len={agg['ep_len_mean']:.1f}±{agg['ep_len_std']:.1f}")


if __name__ == "__main__":
    main()
