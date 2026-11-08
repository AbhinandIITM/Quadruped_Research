#!/usr/bin/env python3
"""Standalone Isaac Lab viewer for trained Go2 policies.

Loads an rsl-rl .pt checkpoint, builds the Go2 DirectRLEnv with rendering
enabled, and steps through the policy in real-time using the Isaac Lab
viewport.

Usage
-----
    python -m rl.isaac_lab.visualize --checkpoint runs/walk/gpu_run_0/model_final.pt --stage walk
    python -m rl.isaac_lab.visualize --checkpoint runs/walk/gpu_run_0 --episodes 3 --speed 1.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.isaac_lab.eval_adapter import load_isaac_policy, is_isaac_checkpoint
from rl.isaac_lab.envs.direct_env import Go2WalkEnvCfg
from isaaclab.app import AppLauncher


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_visualization(
    checkpoint: str | Path,
    *,
    stage: str = "walk",
    device: str = "auto",
    episodes: int = 5,
    episode_length: int = 1000,
    speed: float = 1.0,
    deterministic: bool = True,
    headless: bool = False,
):
    """Run visualization of a trained policy.

    Args:
        checkpoint: Path to .pt file or run directory.
        stage: Stage name (stand, balance, transition, walk).
        device: Torch device.
        episodes: Number of episodes to run.
        episode_length: Max steps per episode (default 1000 = 20s at 50Hz).
        speed: Playback speed multiplier.
        deterministic: Use deterministic actions.
        headless: Run without GUI (for CI/record).
    """
    # Load policy
    device = _resolve_device(device)
    policy = load_isaac_policy(checkpoint, stage=stage, device=device, deterministic=deterministic)

    # Configure app
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=False,
    )
    app = app_launcher.app

    # Build env with rendering
    from rl.isaac_lab.envs.direct_env import Go2DirectRLEnv, Go2DirectRLEnvCfg

    # Map stage to env cfg class
    stage_to_cfg = {
        "stand": "stand",
        "balance": "balance",
        "transition": "transition",
        "walk": "walk",
    }

    env_cfg = Go2DirectRLEnvCfg(
        stage=stage,
        num_envs=1,
        seed=0,
        device=device,
        decimation=10,
        episode_length_s=20.0,
        obs_noise=0.0,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.device = device

    env = Go2DirectRLEnv(env_cfg)

    print(f"[visualize] Loaded policy from {checkpoint}")
    print(f"[visualize] Stage: {stage}, Device: {device}")
    print(f"[visualize] Running {episodes} episodes...")

    try:
        for ep in range(episodes):
            obs_dict = env.reset()
            obs = obs_dict["policy"]  # (1, 45)

            total_reward = 0.0
            for step in range(episode_length):
                # Policy forward
                action_np = policy(obs.squeeze(0).cpu().numpy(), step)
                action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

                # Env step
                obs_dict, reward, terminated, truncated, info = env.step(action)
                obs = obs_dict["policy"]

                total_reward += reward.item()

                if terminated or truncated:
                    break

                # Frame pacing for real-time visualization
                target_dt = (1.0 / 50.0) / speed
                time.sleep(max(0, target_dt - 0.001))

            print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={step + 1}")

    finally:
        env.close()
        app_launcher.app.close()

    print("[visualize] Done.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="visualize", description="Watch a trained Go2 policy in Isaac Lab viewer."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file or run directory")
    parser.add_argument("--stage", default="walk", choices=["stand", "balance", "transition", "walk"])
    parser.add_argument("--device", default="auto", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not is_isaac_checkpoint(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found or invalid: {args.checkpoint}")

    run_visualization(
        args.checkpoint,
        stage=args.stage,
        device=args.device,
        episodes=args.episodes,
        episode_length=args.episode_length,
        speed=args.speed,
        deterministic=not args.stochastic,
        headless=args.headless,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())