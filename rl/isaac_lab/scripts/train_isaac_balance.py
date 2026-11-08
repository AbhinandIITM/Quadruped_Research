#!/usr/bin/env python3
"""Train Balance stage on Isaac Lab (Newton physics + RSL-RL PPO)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl.isaac_lab.runner import train_stage


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Isaac Lab training – Balance stage")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--config", type=str, default="balance")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_stage(
        "balance",
        seed=args.seed,
        total_steps=args.steps,
        run_dir=args.run_dir,
        resume=args.resume,
        device=args.device,
        num_envs=args.num_envs,
        verbose=args.verbose,
        config_name=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())