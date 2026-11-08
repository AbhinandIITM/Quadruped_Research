#!/usr/bin/env python3
"""
Sequential master driver: Train 1 seed × 1 stage at a time, across all 4 stages,
then plot each finished stage from its training log CSV, then move to the next seed.

Pipeline per seed (seed0 → seed1 → seed2 → seed3 → seed4):
  stand (5M) → balance (5M) → transition (5M) → walk (5M)
      each stage: train → write run.json → plot from CSV → checkpoint handoff to next stage

Usage
-----
    python scripts/run_all_training.py --config walk --steps 5000000 --seeds 0 1 2 3 4 --slots 1 --device cuda

This runs 1 process at a time (16 envs, in-process SubprocVecEnv), so memory stays
flat at ~1 GPU + 1 policy regardless of seed count. After every stage finishes it:
  1. prints progress,
  2. drops run.json + the plot PDF/PNG next to the checkpoint,
  3. hands the latest_model.zip into the next stage,
  4. only then starts the next stage/seed.

Logs
----
    scripts/run_all_YYYYMMDD_HHMMSS.log   (master driver stdout/stderr)
    scripts/logs/seed_{seed}/{stage}/worker.log   (per-stage fd-level log)

Run from the repo root:
    cd /run/media/abhi/Abhi/Projects/Quadruped
    nohup python scripts/run_all_training.py --config walk --steps 5000000 --seeds 0 1 2 3 4 > scripts/run_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "scripts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Training-run output moved into the rl package (the repo-root runs/ is gone); the
# stage trainers anchor their defaults here too, so driver and workers agree.
RL_RUNS = REPO / "rl" / "runs"

# (stage_name, training_module)
STAGES = [
    ("stand",  "rl.train_stage_stand"),
    ("balance","rl.train_stage_balance"),
    ("transition","rl.train_stage_transition"),
    ("walk",   "rl.train_stage_walk"),
]


def master_log(seed: int, stage: str, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] seed={seed} stage={stage}: {msg}"
    print(line, flush=True)


def stage_run_dir(stage: str, run_root: Path) -> Path:
    """runs/<stage>/<run_N>/<stage>/"""
    run_dirs = sorted(run_root.glob("run_*"))
    if not run_dirs:
        return run_root / "run_1"
    return run_dirs[-1]


def save_run_json(stage: str, run_root: Path, seed: int, total_steps: int, run_dir: Path) -> None:
    """Persist run.json inside the specific run_N/<stage>/ so plots/egmal find it."""
    latest = run_dir / "run.json"
    if not latest.exists():
        return
    data = json.loads(latest.read_text())
    data["seed"] = seed
    data["total_steps"] = total_steps
    latest.write_text(json.dumps(data, indent=2))


def run_stage(config: str, seed: int, stage: str, module: str, steps: int,
              device: str, local: bool, resume: Path | None, allow_over_budget: bool) -> tuple[int, Path | None]:
    """Launch one training stage in-process; return its return code and run dir."""
    master_log(seed, stage, "starting")
    cmd = [
        sys.executable, "-m", module,
        "--config", config,
        "--steps", str(steps),
        "--seed", str(seed),
        "--device", device,
        "--verbose", "0",
    ]
    if local:
        cmd.append("--local")
    if resume is not None:
        cmd.extend(["--resume", str(resume)])
    if allow_over_budget:
        cmd.append("--allow-over-budget")

    log_path = LOG_DIR / f"seed_{seed}" / stage / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    master_log(seed, stage, f"log -> {log_path}")
    with open(log_path, "a") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf, cwd=str(REPO))
    master_log(seed, stage, f"exit={proc.returncode}")

    actual_run_dir = None
    try:
        for line in log_path.read_text().splitlines():
            if "output dir:" in line:
                actual_run_dir = Path(line.split("output dir:")[-1].strip())
                break
    except Exception:
        pass
    return proc.returncode, actual_run_dir


def plot_stage(stage: str, run_dir: Path) -> int:
    """Plot a finished stage from its training log CSV."""
    module = f"rl.plot_stage_{stage}"
    log_csv = run_dir / "training_log.csv"
    if not log_csv.exists():
        # fallback: run.json based plotter if available
        run_json = run_dir / "run.json"
        if not run_json.exists():
            master_log(-1, stage, f"no training_log.csv or run.json in {run_dir}; skipping plot")
            return 0
    master_log(-1, stage, "plotting")
    cmd = [sys.executable, "-m", module, "--run-dir", str(run_dir)]
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def latest_checkpoint(run_root: Path, stage: str) -> Path | None:
    target = stage_run_dir(stage, run_root) / stage
    cand = target / "latest_model.zip"
    return cand if cand.exists() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential train→plot per stage, one seed at a time")
    p.add_argument("--config", default="walk", help="Config name from rl/configs/")
    p.add_argument("--steps", type=int, default=5_000_000, help="Steps per stage per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds to run")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--local", action="store_true", help="in-process envs (no SubprocVecEnv)")
    p.add_argument("--allow-over-budget", action="store_true", help="Allow training to exceed wall-clock budget")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    master_log(-1, "driver", f"seeds={args.seeds} stages={[s[0] for s in STAGES]} steps={args.steps:,}")

    for seed in args.seeds:
        prev_ckpt: Path | None = None
        cumulative_steps = 0
        for i, (stage, module) in enumerate(STAGES):
            run_root = RL_RUNS / stage
            cumulative_steps += args.steps
            rc, actual_dir = run_stage(args.config, seed, stage, module, cumulative_steps,
                                       args.device, args.local, prev_ckpt, args.allow_over_budget)
            if rc != 0:
                master_log(seed, stage, f"FAILED rc={rc}; aborting seed {seed}")
                return 1

            target_dir = (actual_dir / stage) if (actual_dir and (actual_dir / stage).exists()) else (stage_run_dir(stage, run_root) / stage)
            save_run_json(stage, run_root, seed, cumulative_steps, target_dir)
            plot_stage(stage, target_dir)

            prev_ckpt = target_dir / "latest_model.zip"
            master_log(seed, stage, f"done; checkpoint {prev_ckpt} (cumulative steps={cumulative_steps:,})")

        master_log(seed, "all", f"seed {seed} complete across all 4 stages")

    master_log(-1, "driver", "ALL SEEDS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
