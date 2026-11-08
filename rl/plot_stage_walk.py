#!/usr/bin/env python3
"""
Plot training progress for Stage 3 (walk) training runs.

Reads from run_{x}/ folders (run.json files with metrics_history) and produces
a single figure with subplots for:
- Episode reward (mean ± std across seeds)
- Episode length
- Velocity tracking error (linear x-y and angular z)
- Command success rate (fraction of steps within tracking tolerance)
- Value loss / policy loss
- Mean episode reward over wall clock time

Saves as PNG to run_{x}/plots/ folder.

Usage:
    python -m rl.plot_stage_walk --run-dir rl/runs/walk/run_1
    python -m rl.plot_stage_walk --run-dir rl/runs/walk --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np


# Reference palette from dataviz skill (validated for light/dark, CVD-safe)
SERIES_COLORS = {
    "light": {
        "series_1": "#2a78d6",   # blue - primary series
        "series_2": "#eb6834",   # orange - secondary series
        "series_3": "#1baf7a",   # aqua - tertiary
        "series_4": "#eda100",   # yellow
        "fill": "#2a78d633",     # blue with 20% alpha for fill_between
        "grid": "#52514e4d",     # muted grid
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "surface": "#fcfcfb",
    },
    "dark": {
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "series_3": "#199e70",
        "series_4": "#c98500",
        "fill": "#3987e533",
        "grid": "#c3c2b74d",
        "text_primary": "#f0efec",
        "text_secondary": "#c3c2b7",
        "surface": "#1a1a19",
    },
}


def load_run_json(run_path: Path) -> dict[str, Any]:
    """Load a run.json file."""
    json_path = run_path / "run.json"
    if not json_path.exists():
        return {}
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def discover_runs(run_dir: Path, seeds: list[int] | None = None) -> dict[int, dict[str, Any]]:
    """Discover run.json files for each seed under run_dir."""
    runs = {}
    
    # Check for stage-specific subdirectories (e.g., walk/run.json)
    for stage_name in ["walk", "stand", "balance", "transition"]:
        stage_dir = run_dir / stage_name
        if stage_dir.exists() and (stage_dir / "run.json").exists():
            data = load_run_json(stage_dir)
            if data:
                seed = data.get("seed", 0)
                if seeds is None or seed in seeds:
                    runs[seed] = data
            return runs
    
    # Check for run.json directly in run_dir
    if (run_dir / "run.json").exists():
        data = load_run_json(run_dir)
        if data:
            seed = data.get("seed", 0)
            if seeds is None or seed in seeds:
                runs[seed] = data
        return runs
    
    # Parent directory with run_N subdirectories (run_N or gpu_run_N)
    for subdir in sorted(run_dir.iterdir()):
        if not (subdir.is_dir() and (subdir.name.startswith("run_") or subdir.name.startswith("gpu_run_"))):
            continue
        # Check stage subdirectories
        for stage_name in ["walk", "stand", "balance", "transition"]:
            stage_dir = subdir / stage_name
            if stage_dir.exists() and (stage_dir / "run.json").exists():
                data = load_run_json(stage_dir)
                if data:
                    seed = data.get("seed", 0)
                    if seeds is None or seed in seeds:
                        runs[seed] = data
                break
        else:
            # Check directly in subdir
            if (subdir / "run.json").exists():
                data = load_run_json(subdir)
                if data:
                    seed = data.get("seed", 0)
                    if seeds is None or seed in seeds:
                        runs[seed] = data
    return runs


def interpolate_metrics(
    runs: dict[int, dict[str, Any]],
    metric: str,
    x_key: str = "steps",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a metric across seeds to common x grid."""
    all_x = []
    all_y = []
    for seed, data in runs.items():
        history = data.get("metrics_history", [])
        if not history:
            continue
        x = [h.get(x_key, i * 8192) for i, h in enumerate(history)]
        y = []
        for h in history:
            v = h.get(metric)
            if v is None:
                y.append(np.nan)
            else:
                try:
                    y.append(float(v))
                except (ValueError, TypeError):
                    y.append(np.nan)
        y = np.array(y, dtype=float)
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            continue
        all_x.append(np.array(x)[valid])
        all_y.append(np.array(y)[valid])

    if not all_x:
        return np.array([]), np.array([]), np.array([])

    # Common x grid
    x_min = max(x[0] for x in all_x)
    x_max = min(x[-1] for x in all_x)
    if x_min >= x_max:
        return np.array([]), np.array([]), np.array([])

    x_common = np.linspace(x_min, x_max, 100)
    y_interp = []
    for x, y in zip(all_x, all_y):
        y_interp.append(np.interp(x_common, x, y))

    y_arr = np.array(y_interp)
    return x_common, np.mean(y_arr, axis=0), np.std(y_arr, axis=0)


def plot_training_curves(
    runs: dict[int, dict[str, Any]],
    output_path: Path,
    theme: str = "light",
) -> None:
    """Create the training progress figure."""
    colors = SERIES_COLORS[theme]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor(colors["surface"])
    ax_list = axes.flatten()

    # Metrics to plot
    metrics_config = [
        ("ep_rew_mean", "Episode Reward", ax_list[0]),
        ("ep_len_mean", "Episode Length", ax_list[1]),
        ("value_loss", "Value Loss", ax_list[2]),
        ("policy_gradient_loss", "Policy Gradient Loss", ax_list[3]),
        ("approx_kl", "Approx KL", ax_list[4]),
        ("entropy_loss", "Entropy Loss", ax_list[5]),
    ]

    for metric_key, title, ax in metrics_config:
        x, mean, std = interpolate_metrics(runs, metric_key)
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, color=colors["text_secondary"])
            ax.set_title(title, fontsize=10, color=colors["text_primary"])
            continue
        ax.plot(x, mean, color=colors["series_1"], linewidth=1.6, label=title)
        ax.fill_between(x, mean - std, mean + std,
                        color=colors["fill"], alpha=0.5, label="±1 std")
        ax.set_title(title, fontsize=10, color=colors["text_primary"])
        ax.set_xlabel("Steps", fontsize=9, color=colors["text_primary"])
        ax.set_ylabel(title, fontsize=9, color=colors["text_primary"])
        ax.set_facecolor(colors["surface"])
        for spine in ax.spines.values():
            spine.set_color(colors["text_secondary"])
        ax.tick_params(colors=colors["text_primary"], labelsize=7)
        ax.grid(True, color=colors["grid"], linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=7, frameon=False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage 3 (walk) training progress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory (specific run_N or parent containing run_N subdirs)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to include (default: all)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file (default: run_dir/plots/training_progress.png)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = discover_runs(args.run_dir, args.seeds)

    if not runs:
        print(f"No run.json files found in {args.run_dir}")
        return 1

    print(f"Found {len(runs)} run(s):")
    for seed in sorted(runs):
        print(f"  seed {seed}")

    if args.output:
        output_path = args.output
    else:
        output_path = args.run_dir / "plots" / "training_progress.png"

    plot_training_curves(runs, output_path, theme=args.theme)
    print(f"[ok]   {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
