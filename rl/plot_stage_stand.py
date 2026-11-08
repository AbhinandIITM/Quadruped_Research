#!/usr/bin/env python3
"""
Plot training progress for Stage 0 (stand) training runs.

Reads from run_{x}/ folders (run.json files with metrics_history) and produces
a single figure with subplots for:
- Episode reward (mean ± std across seeds)
- Episode length
- Value loss
- Policy loss (policy gradient loss)
- Mean episode reward over time (wall clock)

Saves as PNG to run_{x}/plots/ folder.

Usage:
    python -m rl.plot_stage_stand --run-dir rl/runs/stand_local/run_1
    python -m rl.plot_stage_stand --run-dir rl/runs/stand_local --seeds 0 1 2
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


def extract_metrics(run_data: dict[str, Any]) -> dict[str, list]:
    """Extract time series from run.json metrics_history."""
    history = run_data.get("metrics_history", [])
    if not history:
        return {}

    metrics = {
        "steps": [],
        "value_loss": [],
        "policy_gradient_loss": [],
        "ep_rew_mean": [],
        "ep_len_mean": [],
        "wall_clock_s": [],
        "approx_kl": [],
        "entropy_loss": [],
        "explained_variance": [],
        "clip_fraction": [],
    }

    for entry in history:
        metrics["steps"].append(entry.get("steps"))
        metrics["value_loss"].append(entry.get("value_loss"))
        metrics["policy_gradient_loss"].append(entry.get("policy_gradient_loss"))
        metrics["ep_rew_mean"].append(entry.get("ep_rew_mean"))
        metrics["ep_len_mean"].append(entry.get("ep_len_mean"))
        metrics["wall_clock_s"].append(entry.get("wall_clock_s"))
        metrics["approx_kl"].append(entry.get("approx_kl"))
        metrics["entropy_loss"].append(entry.get("entropy_loss"))
        metrics["explained_variance"].append(entry.get("explained_variance"))
        metrics["clip_fraction"].append(entry.get("clip_fraction"))

    # Filter out None values for each metric
    for k in metrics:
        metrics[k] = [v if v is not None else float("nan") for v in metrics[k]]

    return metrics


def discover_runs(run_dir: Path, seeds: list[int] | None = None) -> list[Path]:
    """Discover run directories under run_dir."""
    runs = []

    # If run_dir looks like a specific run (contains run.json)
    if (run_dir / "run.json").exists():
        runs.append(run_dir)
        return runs

    # Otherwise scan for run_{x} subdirectories (both CPU run_* and GPU gpu_run_*)
    for subdir in sorted(run_dir.iterdir()):
        if subdir.is_dir() and (subdir.name.startswith("run_") or subdir.name.startswith("gpu_run_")):
            # Check for seed subdirectories
            if seeds:
                for seed in seeds:
                    seed_dir = subdir / f"seed_{seed}"
                    if (seed_dir / "run.json").exists():
                        runs.append(seed_dir)
            else:
                # Look for any seed_* subdirs with run.json
                for seed_dir in sorted(subdir.glob("seed_*")):
                    if (seed_dir / "run.json").exists():
                        runs.append(seed_dir)
                # Also check the run_X dir itself
                if (subdir / "run.json").exists():
                    runs.append(subdir)

    return runs


def aggregate_metrics(runs: list[Path]) -> dict[str, np.ndarray]:
    """Aggregate metrics across multiple runs (seeds) by aligning on steps."""
    all_metrics = []

    for run_path in runs:
        run_data = load_run_json(run_path)
        metrics = extract_metrics(run_data)
        if metrics and metrics["steps"]:
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Find common step range (use the longest run as reference)
    max_steps = max(len(m["steps"]) for m in all_metrics)
    ref_steps = all_metrics[np.argmax([len(m["steps"]) for m in all_metrics])]["steps"]

    # Interpolate all metrics onto reference steps
    agg = {"steps": np.array(ref_steps)}
    metric_keys = ["value_loss", "policy_gradient_loss", "ep_rew_mean", "ep_len_mean",
                   "wall_clock_s", "approx_kl", "entropy_loss", "explained_variance", "clip_fraction"]

    for key in metric_keys:
        values_list = []
        for m in all_metrics:
            if len(m["steps"]) > 1 and len(m[key]) > 1:
                # Interpolate
                interp = np.interp(ref_steps, m["steps"], m[key])
                values_list.append(interp)
            elif len(m[key]) == 1:
                values_list.append(np.full_like(ref_steps, m[key][0]))

        if values_list:
            arr = np.array(values_list)
            agg[f"{key}_mean"] = np.nanmean(arr, axis=0)
            agg[f"{key}_std"] = np.nanstd(arr, axis=0)
            agg[f"{key}_all"] = arr

    return agg


def apply_style(ax: plt.Axes, theme: str = "light") -> None:
    """Apply clean matplotlib style matching project conventions."""
    colors = SERIES_COLORS[theme]
    ax.set_facecolor(colors["surface"])
    ax.grid(True, alpha=0.3, linewidth=0.5, color=colors["grid"])
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=colors["text_secondary"])
    for spine in ax.spines.values():
        spine.set_color(colors["text_secondary"])
        spine.set_linewidth(0.5)
    ax.title.set_color(colors["text_primary"])
    ax.xaxis.label.set_color(colors["text_primary"])
    ax.yaxis.label.set_color(colors["text_primary"])


def create_figure(agg: dict[str, np.ndarray], run_dirs: list[Path], theme: str = "light") -> plt.Figure:
    """Create the multi-subplot figure."""
    colors = SERIES_COLORS[theme]

    # 2x3 grid for 5 plots + 1 empty or combined
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=140, constrained_layout=True)
    axes = axes.flatten()

    steps = agg.get("steps", np.array([]))

    # 1. Value Loss
    ax = axes[0]
    apply_style(ax, theme)
    if "value_loss_mean" in agg:
        mean = agg["value_loss_mean"]
        std = agg["value_loss_std"]
        ax.plot(steps, mean, color=colors["series_1"], linewidth=1.6, label="Value Loss")
        ax.fill_between(steps, mean - std, mean + std, color=colors["fill"], alpha=0.5)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Value Loss", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)

    # 2. Policy Gradient Loss
    ax = axes[1]
    apply_style(ax, theme)
    if "policy_gradient_loss_mean" in agg:
        mean = agg["policy_gradient_loss_mean"]
        std = agg["policy_gradient_loss_std"]
        ax.plot(steps, mean, color=colors["series_2"], linewidth=1.6, label="Policy Grad Loss")
        ax.fill_between(steps, mean - std, mean + std, color=colors["series_2"] + "33", alpha=0.5)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("Policy Gradient Loss", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # 3. Episode Reward (Mean)
    ax = axes[2]
    apply_style(ax, theme)
    if "ep_rew_mean_mean" in agg:
        mean = agg["ep_rew_mean_mean"]
        std = agg["ep_rew_mean_std"]
        ax.plot(steps, mean, color=colors["series_3"], linewidth=1.6, label="Mean Episode Reward")
        ax.fill_between(steps, mean - std, mean + std, color=colors["series_3"] + "33", alpha=0.5)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.set_title("Episode Reward", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # 4. Episode Length
    ax = axes[3]
    apply_style(ax, theme)
    if "ep_len_mean_mean" in agg:
        mean = agg["ep_len_mean_mean"]
        std = agg["ep_len_mean_std"]
        ax.plot(steps, mean, color=colors["series_4"], linewidth=1.6, label="Mean Episode Length")
        ax.fill_between(steps, mean - std, mean + std, color=colors["series_4"] + "33", alpha=0.5)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Length (steps)", fontsize=9)
    ax.set_title("Episode Length", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # 5. Mean Episode Reward over Wall Clock Time
    ax = axes[4]
    apply_style(ax, theme)
    if "ep_rew_mean_mean" in agg and "wall_clock_s_mean" in agg:
        mean_rew = agg["ep_rew_mean_mean"]
        mean_wall = agg["wall_clock_s_mean"]
        # Cumulative wall clock
        cum_wall = np.cumsum(mean_wall)
        ax.plot(cum_wall, mean_rew, color=colors["series_1"], linewidth=1.6, label="Reward vs Time")
    ax.set_xlabel("Wall Clock Time (s)", fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=9)
    ax.set_title("Reward vs Wall Clock", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # 6. Approx KL / Entropy / Explained Variance (auxiliary)
    ax = axes[5]
    apply_style(ax, theme)
    has_aux = False
    if "approx_kl_mean" in agg:
        ax.plot(steps, agg["approx_kl_mean"], color=colors["series_1"], linewidth=1.2, label="Approx KL")
        has_aux = True
    if "entropy_loss_mean" in agg:
        ax.plot(steps, -np.array(agg["entropy_loss_mean"]), color=colors["series_2"], linewidth=1.2, label="-Entropy")
        has_aux = True
    if "explained_variance_mean" in agg:
        ax.plot(steps, agg["explained_variance_mean"], color=colors["series_3"], linewidth=1.2, label="Explained Var")
        has_aux = True
    if has_aux:
        ax.set_xlabel("Steps", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_title("Auxiliary Metrics", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.set_visible(False)

    # Overall title
    run_names = [d.name for d in run_dirs]
    fig.suptitle(f"Stage 0 (Stand) Training Progress\n{', '.join(run_names[:3])}{'...' if len(run_names) > 3 else ''}",
                 fontsize=11, color=colors["text_primary"])

    return fig


def save_figure(fig: plt.Figure, out_dir: Path, theme: str = "light") -> Path:
    """Save figure as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "training_progress.png"

    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    print(f"Saved: {png_path}")
    return png_path
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot Stage 0 (Stand) training progress from run.json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory (specific run_N or parent containing run_N subdirs)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Specific seed numbers to plot (default: all discovered)",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="light",
        help="Color theme for plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: run_dir/plots)",
    )

    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1

    # Discover runs
    run_dirs = discover_runs(run_dir, args.seeds)
    if not run_dirs:
        print(f"No run.json files found under {run_dir}")
        return 1

    print(f"Found {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d}")

    # Aggregate metrics
    agg = aggregate_metrics(run_dirs)
    if not agg:
        print("No metrics data found in run files")
        return 1

    # Determine output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        # Use parent of first run's parent (e.g., run_1/plots or runs/stand_local/plots)
        out_dir = run_dirs[0].parent / "plots"

    # Create figure
    fig = create_figure(agg, run_dirs, args.theme)

    # Save
    save_figure(fig, out_dir, args.theme)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())