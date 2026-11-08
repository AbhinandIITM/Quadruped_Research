#!/usr/bin/env python3
"""
Plot training progress for Stage 1 (balance) training runs.

Reads from run_{x}/ folders (run.json files with metrics_history) and produces
a single figure with subplots for:
- Episode reward (mean ± std across seeds)
- Episode length
- Value loss
- Policy loss (policy gradient loss)
- Mean episode reward over time (wall clock)
- Perturbation command statistics

Saves as PNG to run_{x}/plots/ folder.

Usage:
    python -m rl.plot_stage_balance --run-dir rl/runs/balance/run_1
    python -m rl.plot_stage_balance --run-dir rl/runs/balance --seeds 0 1 2
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
# Only the roles this chart uses
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
    if (run_dir / "run.json").exists():
        # Single run directory (handles both run_N and gpu_run_N naming)
        data = load_run_json(run_dir)
        if data:
            seed = data.get("seed", 0)
            runs[seed] = data
    else:
        # Parent directory: scan any subdir that contains run.json
        for subdir in sorted(run_dir.iterdir()):
            if not subdir.is_dir():
                continue
            data = load_run_json(subdir)
            if data:
                seed = data.get("seed", 0)
                if seeds is None or seed in seeds:
                    runs[seed] = data
    return runs


def interpolate_metrics(
    runs: dict[int, dict[str, Any]],
    metric: str,
    x_key: str = "global_step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a metric across seeds to common x grid."""
    all_x = []
    all_y = []
    for seed, data in runs.items():
        history = data.get("metrics_history", [])
        if not history:
            continue
        x = [h.get(x_key, i * 8192) for i, h in enumerate(history)]
        y = [h.get(metric, np.nan) for h in history]
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
        ("episode_reward_mean", "Episode Reward", ax_list[0]),
        ("episode_length_mean", "Episode Length", ax_list[1]),
        ("value_loss", "Value Loss", ax_list[2]),
        ("policy_loss", "Policy Gradient Loss", ax_list[3]),
        ("rollout/ep_rew_mean", "Mean Episode Reward (SB3)", ax_list[4]),
        ("perturbation_magnitude", "Perturbation Command Magnitude", ax_list[5]),
    ]

    for metric_key, title, ax in metrics_config:
        x, mean, std = interpolate_metrics(runs, metric_key)
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, color=colors["text_secondary"])
            ax.set_title(title, color=colors["text_primary"])
            continue

        ax.plot(x, mean, color=colors["series_1"], linewidth=2, label="Mean")
        ax.fill_between(x, mean - std, mean + std, color=colors["fill"], alpha=0.5, label="±1 std")
        ax.set_title(title, color=colors["text_primary"], fontsize=11)
        ax.set_xlabel("Global Steps", color=colors["text_secondary"], fontsize=9)
        ax.tick_params(colors=colors["text_secondary"], labelsize=8)
        ax.grid(True, color=colors["grid"], linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])

        # Format x-axis in scientific notation
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    # Wall clock subplot (separate because x is time, not steps)
    ax = ax_list[5]
    all_times = []
    all_rews = []
    for seed, data in runs.items():
        history = data.get("metrics_history", [])
        if not history:
            continue
        times = [h.get("wall_time_s", 0) for h in history]
        rews = [h.get("episode_reward_mean", np.nan) for h in history]
        valid = ~np.isnan(rews)
        if np.sum(valid) < 2:
            continue
        all_times.append(np.array(times)[valid])
        all_rews.append(np.array(rews)[valid])

    if all_times:
        t_min = max(t[0] for t in all_times)
        t_max = min(t[-1] for t in all_times)
        if t_min < t_max:
            t_common = np.linspace(t_min, t_max, 100)
            rew_interp = []
            for t, r in zip(all_times, all_rews):
                rew_interp.append(np.interp(t_common, t, r))
            r_arr = np.array(rew_interp)
            ax.plot(t_common, np.mean(r_arr, axis=0), color=colors["series_1"], linewidth=2)
            ax.fill_between(t_common, np.mean(r_arr, axis=0) - np.std(r_arr, axis=0),
                           np.mean(r_arr, axis=0) + np.std(r_arr, axis=0),
                           color=colors["fill"], alpha=0.5)

    ax.set_title("Episode Reward vs Wall Clock", color=colors["text_primary"], fontsize=11)
    ax.set_xlabel("Wall Clock (s)", color=colors["text_secondary"], fontsize=9)
    ax.tick_params(colors=colors["text_secondary"], labelsize=8)
    ax.grid(True, color=colors["grid"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(colors["grid"])

    # Add seed info
    seed_list = sorted(runs.keys())
    fig.suptitle(f"Stage 1 (Balance) Training Progress — Seeds: {seed_list}",
                 color=colors["text_primary"], fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, facecolor=colors["surface"],
                bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot Stage 1 (Balance) training progress")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory (specific run_N or parent containing run_N subdirs)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Specific seed numbers to plot (default: all discovered)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                        help="Color theme for plots")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: run_dir/plots)")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        return 1

    runs = discover_runs(run_dir, args.seeds)
    if not runs:
        print(f"No run.json files found in {run_dir}")
        return 1

    if args.output_dir:
        output_dir = args.output_dir
    elif run_dir.name.startswith("run_"):
        output_dir = run_dir / "plots"
    else:
        output_dir = run_dir / "plots"

    output_path = output_dir / f"stage_balance_training_progress"
    plot_training_curves(runs, output_path, args.theme)
    print(f"Saved plots to {output_path}.pdf and {output_path}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())