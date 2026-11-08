#!/usr/bin/env python3
"""Multi-seed training for Isaac Lab Go2.

Runs multiple seeds in parallel and aggregates metrics (mean ± std).
Mirrors rl/train_multistage_multiseed.py for Isaac Lab.
"""

from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

# Add rl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl.isaac_lab.runner import train_stage, IsaacLabRunner


@dataclass
class SeedResult:
    """Results from a single seed."""
    seed: int
    stage: str
    final_reward: float
    final_episode_length: float
    total_steps: int
    wall_time: float
    run_dir: str
    metrics_history: list


def train_single_seed(
    stage: str,
    seed: int,
    total_steps: int,
    num_envs: int,
    device: str,
    run_dir: Optional[Path],
    verbose: int,
) -> SeedResult:
    """Train a single seed (worker function for parallel execution)."""
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create unique run directory for this seed
    if run_dir:
        seed_run_dir = run_dir / f"seed{seed}"
    else:
        seed_run_dir = Path(f"logs/isaac_lab/{stage}/seed{seed}_{int(time.time())}")

    seed_run_dir.mkdir(parents=True, exist_ok=True)

    # Train
    runner = train_stage(
        stage=stage,
        seed=seed,
        total_steps=total_steps,
        run_dir=seed_run_dir,
        device=device,
        num_envs=num_envs,
        verbose=verbose,
    )

    # Get final metrics
    metrics = runner._metrics_history[-1] if runner._metrics_history else None

    return SeedResult(
        seed=seed,
        stage=stage,
        final_reward=metrics.ep_rew_mean if metrics else 0.0,
        final_episode_length=metrics.ep_len_mean if metrics else 0.0,
        total_steps=total_steps,
        wall_time=time.time() - runner._start_time if hasattr(runner, "_start_time") else 0.0,
        run_dir=str(seed_run_dir),
        metrics_history=[asdict(m) for m in runner._metrics_history],
    )


def train_multistage_multiseed(
    stages: List[str],
    seeds: List[int],
    steps_per_stage: int,
    num_envs: int,
    device: str,
    run_dir: Optional[Path] = None,
    verbose: int = 1,
    max_parallel: int = 1,  # GPU memory limits parallelism
) -> dict:
    """Train multiple stages with multiple seeds.

    Args:
        stages: List of stages to train in sequence
        seeds: List of random seeds
        steps_per_stage: Steps per stage per seed
        num_envs: Number of environments
        device: Device (cuda/cpu)
        run_dir: Base run directory
        verbose: Verbosity level
        max_parallel: Max parallel seeds (1 for GPU, >1 for CPU)

    Returns:
        Aggregated results with mean ± std across seeds
    """
    if run_dir is None:
        run_dir = Path(f"logs/isaac_lab/multistage_multiseed_{int(time.time())}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.upper()} - {len(seeds)} seeds")
        print(f"{'='*60}")

        stage_results = []

        if max_parallel > 1 and device == "cpu":
            # Parallel execution for CPU
            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                futures = {
                    executor.submit(
                        train_single_seed,
                        stage, seed, steps_per_stage, num_envs, device,
                        run_dir / stage, verbose,
                    ): seed for seed in seeds
                }

                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        stage_results.append(result)
                        print(f"  Seed {seed}: reward={result.final_reward:.3f}, "
                              f"ep_len={result.final_episode_length:.1f}")
                    except Exception as e:
                        print(f"  Seed {seed}: FAILED - {e}")
        else:
            # Sequential execution (required for GPU)
            for seed in seeds:
                print(f"  Training seed {seed}...")
                try:
                    result = train_single_seed(
                        stage, seed, steps_per_stage, num_envs, device,
                        run_dir / stage, verbose,
                    )
                    stage_results.append(result)
                    print(f"    reward={result.final_reward:.3f}, "
                          f"ep_len={result.final_episode_length:.1f}")
                except Exception as e:
                    print(f"    FAILED - {e}")

        # Aggregate results
        if stage_results:
            rewards = [r.final_reward for r in stage_results]
            ep_lens = [r.final_episode_length for r in stage_results]

            agg = {
                "stage": stage,
                "seeds": [r.seed for r in stage_results],
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "ep_len_mean": float(np.mean(ep_lens)),
                "ep_len_std": float(np.std(ep_lens)),
                "individual": [asdict(r) for r in stage_results],
            }

            all_results[stage] = agg
            print(f"  Aggregated: reward={agg['reward_mean']:.3f}±{agg['reward_std']:.3f}, "
                  f"ep_len={agg['ep_len_mean']:.1f}±{agg['ep_len_std']:.1f}")

    # Save combined results
    output_file = run_dir / "multiseed_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab multi-seed training")
    parser.add_argument("--stages", nargs="+", default=["stand", "balance", "transition", "walk"],
                        help="Stages to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds")
    parser.add_argument("--steps", type=int, default=5_000_000,
                        help="Steps per stage per seed")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Number of environments")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Base run directory")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity")
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Max parallel seeds (1 for GPU)")

    args = parser.parse_args()

    train_multistage_multiseed(
        stages=args.stages,
        seeds=args.seeds,
        steps_per_stage=args.steps,
        num_envs=args.num_envs,
        device=args.device,
        run_dir=args.run_dir,
        verbose=args.verbose,
        max_parallel=args.max_parallel,
    )


if __name__ == "__main__":
    main()
