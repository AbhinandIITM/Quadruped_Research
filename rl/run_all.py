#!/usr/bin/env python3
"""
Complete training + plotting + evaluation pipeline for the 4-stage Go2 curriculum.

Runs in serial:
  1. Stage 0: Stand (300 iterations ≈ 29.5M steps, 4096 envs)
  2. Stage 1: Balance (300 iterations ≈ 29.5M steps, 4096 envs)  <- warm-started from stand
  3. Stage 2: Transition (400 iterations ≈ 39.3M steps, 4096 envs) <- warm-started from balance
  4. Stage 3: Walk (1500 iterations ≈ 147.5M steps, 4096 envs)      <- warm-started from transition
  5. Training progress plots (per stage)
  6. Evaluation sweep (all stages, all speeds)
  7. Multistage comparison plot

Usage:
    python -m rl.run_all --seed 0 --device cuda
    python -m rl.run_all --seed 0 --device cuda --stages walk  # single stage
    python -m rl.run_all --seed 0 --device cuda --skip-train --only-eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

RL_ROOT = Path(__file__).resolve().parent  # …/rl/ — canonical home for runs/ and results/
PYTHON = sys.executable


STAGE_SPECS = [
    ("stand", 29491200),      # 300 iterations (× 98,304 steps/iter)
    ("balance", 29491200),    # 300 iterations
    ("transition", 39321600), # 400 iterations
    ("walk", 147456000),      # 1500 iterations
]

EVAL_SPEEDS = [0.0, 0.25, 0.5, 0.75, 1.0]
EVAL_EPISODES_PER_SPEED = 10


def run_cmd(cmd: list[str], description: str, env: dict | None = None) -> int:
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    result = subprocess.run(cmd, env=env)
    return result.returncode


def write_comparison_summary(stages: Sequence[str], seed: int) -> Path:
    """Merge per-stage eval JSONs into a single comparison.json.

    Schema matches the previous eval_all_stages output:
    {generated_at, stages: [{stage, metrics: {...}}], comparison: {stage: {...}}}
    """
    import datetime

    out = RL_ROOT / "results" / "eval" / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    comparison = {}
    for stage in stages:
        eval_dir = RL_ROOT / "results" / "eval" / stage
        files = sorted(eval_dir.glob("eval_*.json")) if eval_dir.exists() else []
        if not files:
            continue
        d = json.load(open(files[-1], encoding="utf-8"))
        summary = d.get("summary", {})
        metrics = {
            "velocity_mae_mean": summary.get("velocity_mae_mean", 0.0),
            "survival_rate": summary.get("survival_rate", 0.0),
            "command_success_rate": summary.get("command_success_rate", 0.0),
            "gait_match_rate": summary.get("gait_match_rate", 0.0),
            "cot_mean": summary.get("cot_mean", 0.0),
            "n_episodes": summary.get("n_episodes", 0),
        }
        rows.append({"stage": stage, "metrics": metrics})
        comparison[stage] = metrics

    out.write_text(json.dumps({
        "generated_at": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "stages": rows,
        "comparison": comparison,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return out


def train_stage(
    stage: str,
    seed: int,
    device: str,
    num_envs: int,
    steps: int,
    override_steps: int | None = None,
    resume_from: Path | None = None,
) -> int:
    """Train a single stage using the unified train_isaac.py CLI."""
    run_dir = RL_ROOT / "runs" / stage / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    actual_steps = override_steps if override_steps is not None else steps

    cmd = [
        PYTHON, "-m", "rl.isaac_lab.scripts.train_isaac",
        "--stage", stage,
        "--seed", str(seed),
        "--device", device,
        "--num-envs", str(num_envs),
        "--steps", str(actual_steps),
        "--run-dir", str(run_dir),
    ]
    if resume_from is not None and resume_from.exists():
        cmd += ["--resume", str(resume_from)]
    tag = f" (warm-start {resume_from.name})" if resume_from is not None else ""
    return run_cmd(cmd, f"Training {stage} (seed={seed}, {actual_steps/num_envs/24:.0f} iters){tag}")


def plot_training(stage: str, seed: int) -> int:
    """Generate training progress plot for a stage."""
    run_dir = RL_ROOT / "runs" / stage / f"seed_{seed}"
    cmd = [PYTHON, "-m", f"rl.plot_stage_{stage}", "--run-dir", str(run_dir)]
    return run_cmd(cmd, f"Plotting {stage} training progress")


def evaluate_stage(stage: str, seed: int, device: str) -> int:
    """Evaluate a single stage checkpoint."""
    run_dir = RL_ROOT / "runs" / stage / f"seed_{seed}"
    out_dir = RL_ROOT / "results" / "eval" / stage
    speeds = ",".join(str(s) for s in EVAL_SPEEDS)

    cmd = [
        PYTHON, "-m", "rl.runs.eval_stage",
        "--run-dir", str(run_dir),
        "--stage", stage,
        "--num-episodes", str(EVAL_EPISODES_PER_SPEED),
        "--device", device,
        "--speeds", speeds,
        "--out-dir", str(out_dir),
    ]
    return run_cmd(cmd, f"Evaluating {stage} (seed={seed})")


def plot_multistage() -> int:
    """Generate the cross-stage comparison plot."""
    cmd = [PYTHON, "-m", "rl.plot_multistage", "--metrics-dir", str(RL_ROOT / "results")]
    return run_cmd(cmd, "Generating multistage comparison plot")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Full Go2 curriculum pipeline")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu / auto")
    p.add_argument("--num-envs", type=int, default=4096, help="Parallel envs")
    p.add_argument("--steps", type=int, default=None,
                   help="Override steps for all stages (default: per-stage spec)")
    p.add_argument("--stages", nargs="*", default=None,
                   choices=["stand", "balance", "transition", "walk"],
                   help="Stages to run (default: all 4)")
    p.add_argument("--skip-train", action="store_true", help="Skip training, only plot/eval")
    p.add_argument("--skip-plot", action="store_true", help="Skip training plots")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--only-eval", action="store_true", help="Only run evaluation + multistage plot")
    args = p.parse_args(argv)

    stages = args.stages or [s for s, _ in STAGE_SPECS]

    # Resolve device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'#'*60}")
    print(f"  GO2 CURRICULUM PIPELINE")
    print(f"  Stages: {stages}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"  Num envs: {args.num_envs}")
    print(f"{'#'*60}\n")

    start_time = time.time()

    # 1. Training (chain: each stage warm-starts from the previous stage's final checkpoint)
    if not (args.skip_train or args.only_eval):
        prev_final = None
        for stage, steps in STAGE_SPECS:
            if stage not in stages:
                continue
            rc = train_stage(stage, args.seed, device, args.num_envs, steps, args.steps, prev_final)
            if rc != 0:
                print(f"FAILED: {stage} training exited with {rc}")
                return rc
            prev_final = RL_ROOT / "runs" / stage / f"seed_{args.seed}" / "model_final.pt"
    else:
        print("Skipping training (--skip-train or --only-eval)")

    # 2. Training progress plots
    if not (args.skip_plot or args.only_eval):
        for stage in stages:
            rc = plot_training(stage, args.seed)
            if rc != 0:
                print(f"FAILED: {stage} plot exited with {rc}")
                return rc
    else:
        print("Skipping training plots (--skip-plot or --only-eval)")

    # 3. Evaluation
    if not args.skip_eval:
        for stage in stages:
            rc = evaluate_stage(stage, args.seed, device)
            if rc != 0:
                print(f"FAILED: {stage} evaluation exited with {rc}")
                return rc

        # 4. Multistage comparison plot
        rc = plot_multistage()
        if rc != 0:
            print(f"FAILED: multistage plot exited with {rc}")
            return rc

        # 5. Aggregated comparison.json summary
        write_comparison_summary(stages, args.seed)
    else:
        print("Skipping evaluation (--skip-eval)")

    elapsed = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE in {elapsed/60:.1f} min")
    print(f"  Runs:    rl/runs/{{stage}}/seed_{args.seed}/")
    print(f"  Plots:   rl/runs/{{stage}}/plots/")
    print(f"  Eval:    rl/results/eval/{{stage}}/")
    print(f"  Compare: rl/results/plots/multistage_comparison.png")
    print(f"{'#'*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())