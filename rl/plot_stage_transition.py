#!/usr/bin/env python3
"""
Plot training progress for Stage 2 (Stand → Walk Transition) training runs.

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
    python -m rl.plot_stage_transition --run-dir rl/runs/transition/run_1
    python -m rl.plot_stage_transition --run-dir rl/runs/transition --seeds 0 1 2
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


def extract_metrics(run_data: dict[str, Any]) -> dict[str, list[Any]]:
    """Extract metric series from a run.json document."""
    history = run_data.get("metrics_history", []) if run_data else []
    metrics: dict[str, list[Any]] = {
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
        "track_lin_vel_xy": [],
        "track_ang_vel_z": [],
        "command_success_rate": [],
        "joint_action_rate": [],
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
        metrics["track_lin_vel_xy"].append(entry.get("track_lin_vel_xy"))
        metrics["track_ang_vel_z"].append(entry.get("track_ang_vel_z"))
        metrics["command_success_rate"].append(entry.get("command_success_rate"))
        metrics["joint_action_rate"].append(entry.get("joint_action_rate"))

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

    # Otherwise scan for run dirs (run_{x} or gpu_run_{x}) with run.json
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
                # Also check the run_X / gpu_run_X dir itself (GPU runs have run.json flat)
                if (subdir / "run.json").exists():
                    runs.append(subdir)

    return runs


def aggregate_across_seeds(runs: list[Path]) -> dict[str, Any]:
    """Aggregate metric series across multiple seed runs.

    Returns a dict mapping each metric to mean/std lists aligned on the union of
    all step coordinates. Runs must share step coordinates to aggregate meaningfully;
    otherwise each run is kept as its own series under ``raw``.
    """
    all_metrics = [extract_metrics(load_run_json(r)) for r in runs]
    all_metrics = [m for m in all_metrics if m["steps"]]

    if not all_metrics:
        return {}

    # Check whether all runs share identical step coordinates
    reference_steps = all_metrics[0]["steps"]
    aligned = all(m["steps"] == reference_steps for m in all_metrics)

    agg: dict[str, Any] = {"steps": list(reference_steps) if aligned else None}
    if aligned:
        n = len(all_metrics)
        for key in all_metrics[0]:
            if key == "steps":
                continue
            series = np.array([m[key] for m in all_metrics], dtype=np.float64)
            agg[f"{key}_mean"] = series.mean(axis=0).tolist()
            agg[f"{key}_std"] = series.std(axis=0).tolist()
    else:
        agg["raw"] = all_metrics

    return agg


def apply_style(ax: plt.Axes, theme: str = "light") -> None:
    """Apply the shared theme style to an axes."""
    colors = SERIES_COLORS[theme]
    ax.set_facecolor(colors["surface"])
    for spine in ax.spines.values():
        spine.set_color(colors["text_secondary"])
    ax.tick_params(colors=colors["text_primary"], labelsize=7)
    ax.xaxis.label.set_color(colors["text_primary"])
    ax.yaxis.label.set_color(colors["text_primary"])
    ax.title.set_color(colors["text_primary"])
    ax.grid(True, color=colors["grid"], linewidth=0.6, alpha=0.7)


def build_figure(run_dirs: list[Path], theme: str = "light") -> plt.Figure:
    """Build the Stage 2 training-progress figure."""
    colors = SERIES_COLORS[theme]
    agg = aggregate_across_seeds(run_dirs)

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 7.2))
    axes = axes.flatten()

    steps = agg.get("steps") if agg else None

    # 1. Episode reward (mean ± std across seeds)
    ax = axes[0]
    apply_style(ax, theme)
    if agg and "ep_rew_mean_mean" in agg:
        mean = np.array(agg["ep_rew_mean_mean"])
        std = np.array(agg["ep_rew_mean_std"])
        ax.plot(steps, mean, color=colors["series_1"], linewidth=1.6, label="Episode Reward")
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors["series_1"], alpha=0.25, label="±1 std across seeds")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=9)
    ax.set_title("Episode Reward (across seeds)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 2. Episode length
    ax = axes[1]
    apply_style(ax, theme)
    if agg and "ep_len_mean_mean" in agg:
        mean = np.array(agg["ep_len_mean_mean"])
        std = np.array(agg["ep_len_mean_std"])
        ax.plot(steps, mean, color=colors["series_2"], linewidth=1.6, label="Episode Length")
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors["series_2"], alpha=0.25, label="±1 std across seeds")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Mean Steps per Episode", fontsize=9)
    ax.set_title("Episode Length", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 3. Velocity tracking error (linear x-y)
    ax = axes[2]
    apply_style(ax, theme)
    has_lin = agg and "track_lin_vel_xy_mean" in agg
    if has_lin:
        mean = np.array(agg["track_lin_vel_xy_mean"])
        std = np.array(agg["track_lin_vel_xy_std"])
        ax.plot(steps, mean, color=colors["series_1"], linewidth=1.6, label="|err| lin vel xy")
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors["series_1"], alpha=0.25, label="±1 std across seeds")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Tracking Error (m/s)", fontsize=9)
    ax.set_title("Linear Velocity Tracking Error", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 4. Velocity tracking error (angular z)
    ax = axes[3]
    apply_style(ax, theme)
    has_ang = agg and "track_ang_vel_z_mean" in agg
    if has_ang:
        mean = np.array(agg["track_ang_vel_z_mean"])
        std = np.array(agg["track_ang_vel_z_std"])
        ax.plot(steps, mean, color=colors["series_2"], linewidth=1.6, label="|err| ang vel z")
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors["series_2"], alpha=0.25, label="±1 std across seeds")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Tracking Error (rad/s)", fontsize=9)
    ax.set_title("Angular Velocity Tracking Error", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 5. Command success rate
    ax = axes[4]
    apply_style(ax, theme)
    has_succ = agg and "command_success_rate_mean" in agg
    if has_succ:
        mean = np.array(agg["command_success_rate_mean"])
        std = np.array(agg["command_success_rate_std"])
        ax.plot(steps, mean, color=colors["series_3"], linewidth=1.6, label="Command Success Rate")
        ax.fill_between(steps, mean - std, mean + std,
                        color=colors["series_3"], alpha=0.25, label="±1 std across seeds")
        ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Fraction of Steps in Tolerance", fontsize=9)
    ax.set_title("Command Success Rate", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 6. Value loss / policy loss
    ax = axes[5]
    apply_style(ax, theme)
    has_loss = agg and ("value_loss_mean" in agg or "policy_gradient_loss_mean" in agg)
    if has_loss:
        if "value_loss_mean" in agg:
            mean = np.array(agg["value_loss_mean"])
            ax.plot(steps, mean, color=colors["series_1"], linewidth=1.2, label="Value Loss")
        if "policy_gradient_loss_mean" in agg:
            mean = np.array(agg["policy_gradient_loss_mean"])
            ax.plot(steps, mean, color=colors["series_2"], linewidth=1.2,
                    label="Policy Gradient Loss")
    ax.set_xlabel("Steps", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("PPO Losses", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # Overall title
    run_names = [d.name for d in run_dirs]
    fig.suptitle(f"Stage 2 (Stand → Walk Transition) Training Progress\n"
                 f"{', '.join(run_names[:3])}{'...' if len(run_names) > 3 else ''}",
                 fontsize=11, color=colors["text_primary"])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def save_figure(fig: plt.Figure, out_dir: Path, theme: str = "light") -> Path:
    """Save figure as PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "training_progress.png"

    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[ok]   {png_path}")
    return png_path
    return pdf_path, png_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage 2 (Stand → Walk Transition) training progress"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "transition",
        help="Directory containing run_{x}/ folders (default: rl/runs/transition)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Specific seeds to include (default: all seed_* dirs)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir: Path = args.run_dir

    if not run_dir.exists():
        print(f"[skip] {run_dir} does not exist")
        return

    run_dirs = discover_runs(run_dir, args.seeds)
    if not run_dirs:
        print(f"[skip] no run.json found under {run_dir}")
        return

    print(f"[plot_stage_transition] found {len(run_dirs)} run(s) under {run_dir}")
    fig = build_figure(run_dirs, theme="light")
    save_figure(fig, run_dir / "plots")


if __name__ == "__main__":
    main()
