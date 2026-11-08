#!/usr/bin/env python3
"""Multi-seed training script for Isaac Lab Go2 stages.

Runs multiple seeds in parallel or sequentially and aggregates results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


def run_single_seed(
    stage: str,
    seed: int,
    steps: int,
    num_envs: int,
    device: str,
    run_dir: Path,
    config: str,
    verbose: int = 1,
) -> dict:
    """Run a single training seed as a subprocess."""
    cmd = [
        sys.executable,
        "-m", "rl.isaac_lab.scripts.train_isaac",
        "--stage", stage,
        "--seed", str(seed),
        "--steps", str(steps),
        "--num-envs", str(num_envs),
        "--device", device,
        "--run-dir", str(run_dir / f"seed_{seed}"),
    ]
    if verbose > 0:
        cmd.append("--verbose")
        cmd.append(str(verbose))

    if verbose > 0:
        print(f"[seed {seed}] Starting: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=verbose == 0,
            text=True,
            check=True,
        )
        if verbose > 0:
            print(f"[seed {seed}] Completed successfully")
            if result.stdout:
                print(result.stdout[-500:])  # Last 500 chars

        # Try to load final metrics from run.json (IsaacLabRunner's log format)
        metrics_file = run_dir / f"seed_{seed}" / "run.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                run_data = json.load(f)
            history = run_data.get("metrics_history", [])
            metrics = history[-1] if history else {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

        return {"seed": seed, "success": True, "metrics": metrics, "stdout": result.stdout}

    except subprocess.CalledProcessError as e:
        if verbose > 0:
            print(f"[seed {seed}] FAILED: {e}")
            if e.stdout:
                print(e.stdout[-500:])
            if e.stderr:
                print(e.stderr[-500:])
        return {"seed": seed, "success": False, "error": str(e), "stdout": e.stdout, "stderr": e.stderr}


def aggregate_results(results: list[dict], output_dir: Path) -> dict:
    """Aggregate metrics across seeds."""
    successful = [r for r in results if r["success"]]

    if not successful:
        return {"error": "No successful runs"}

    # Collect numeric metrics
    metric_keys = set()
    for r in successful:
        metric_keys.update(r.get("metrics", {}).keys())

    aggregated = {
        "num_seeds": len(results),
        "num_successful": len(successful),
        "seeds": [r["seed"] for r in successful],
        "metrics": {},
    }

    for key in metric_keys:
        values = [r["metrics"].get(key) for r in successful if key in r["metrics"]]
        if values and all(isinstance(v, (int, float)) for v in values):
            import numpy as np
            aggregated["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    # Save aggregated results
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_file = output_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training for Isaac Lab Go2")
    parser.add_argument("--stage", type=str, required=True, choices=["stand", "balance", "transition", "walk"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of seeds to run")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total timesteps per seed")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/isaac_multiseed"), help="Output directory")
    parser.add_argument("--config", type=str, default="default", help="Config name")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel jobs (1=sequential)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    run_dir = args.run_dir / args.stage
    print(f"Multi-seed training: stage={args.stage}, seeds={args.seeds}")
    print(f"Steps per seed: {args.steps}, Num envs: {args.num_envs}")
    print(f"Parallel jobs: {args.parallel}")
    print(f"Run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run seeds
    if args.parallel > 1:
        print(f"Running {len(args.seeds)} seeds in parallel ({args.parallel} at a time)...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    run_single_seed,
                    args.stage, seed, args.steps, args.num_envs, args.device,
                    run_dir, args.config, args.verbose
                ): seed for seed in args.seeds
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())
    else:
        print(f"Running {len(args.seeds)} seeds sequentially...")
        results = []
        for seed in args.seeds:
            results.append(run_single_seed(
                args.stage, seed, args.steps, args.num_envs, args.device,
                run_dir, args.config, args.verbose
            ))

    # Aggregate
    print("\nAggregating results...")
    aggregated = aggregate_results(results, run_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY: {args.stage}")
    print(f"{'='*60}")
    print(f"Total seeds: {aggregated.get('num_seeds', 0)}")
    print(f"Successful: {aggregated.get('num_successful', 0)}")
    print(f"Failed: {aggregated.get('num_seeds', 0) - aggregated.get('num_successful', 0)}")

    if "metrics" in aggregated:
        for key, stats in aggregated["metrics"].items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})")

    # Save full results
    results_file = run_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if aggregated.get("num_successful", 0) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()