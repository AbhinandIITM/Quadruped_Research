#!/usr/bin/env python3
"""Unified training CLI for all Isaac Lab stages.

Usage:
    python -m rl.isaac_lab.scripts.train_isaac --stage walk --seed 0 --steps 5000000
    python -m rl.isaac_lab.scripts.train_isaac --stage stand --steps 2000 --num-envs 64  # smoke check
"""

from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

# Add the rl directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl.isaac_lab.runner import train_stage


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Isaac Lab training – unified CLI")
    p.add_argument("--stage", type=str, required=True,
                   choices=["stand", "balance", "transition", "walk"],
                   help="Training stage")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--config", type=str, default=None, help="Config name (defaults to stage)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Determine config name
    config_name = args.config or args.stage

    runner = train_stage(
        stage=args.stage,
        seed=args.seed,
        total_steps=args.steps,
        run_dir=args.run_dir,
        resume=args.resume,
        device=args.device,
        num_envs=args.num_envs,
        verbose=args.verbose,
        config_name=config_name,
    )

    print(f"Training complete. run.json: {runner.run_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())