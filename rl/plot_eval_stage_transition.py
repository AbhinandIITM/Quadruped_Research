#!/usr/bin/env python3
"""
Plot evaluation results for Stage 2 (Stand → Walk Transition) policies.

Reads the evaluation manifest JSON (metrics.json produced by eval_stage_transition.py)
and produces a figure with subplots for:
- Velocity tracking MAE (linear + angular) vs commanded speed
- Survival rate vs commanded speed
- Command success rate vs commanded speed
- Gait match rate vs commanded speed
- Cost of transport vs commanded speed (if available)

Saves as PNG.

Usage:
    python -m rl.plot_eval_stage_transition --metrics rl/results/eval/transition_eval_YYYYMMDD_HHMMSS/metrics.json
    python -m rl.plot_eval_stage_transition --metrics rl/results/eval/transition_eval_*/metrics.json --compare
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Reference palette from dataviz skill (validated for light/dark, CVD-safe)
SERIES_COLORS = {
    "light": {
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
        "series_3": "#1baf7a",
        "series_4": "#eda100",
        "fill": "#2a78d633",
        "grid": "#52514e4d",
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


def load_metrics(json_path: Path) -> dict[str, Any]:
    """Load evaluation metrics from JSON."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def extract_speed_summary(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the speed sweep summary from the evaluation document."""
    return doc.get("transition_eval", {}).get("speed_summary", [])


def plot_evaluation(metrics_paths: list[Path], out_dir: Path, dark: bool = False) -> Path:
    """Create the evaluation figure."""
    palette = SERIES_COLORS["dark" if dark else "light"]

    n_runs = len(metrics_paths)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for idx, metrics_path in enumerate(metrics_paths):
        doc = load_metrics(metrics_path)
        summary = extract_speed_summary(doc)

        if not summary:
            print(f"[plot_eval_stage_transition] WARNING: no speed_summary in {metrics_path}")
            continue

        speeds = np.array([row["command_speed_ms"] for row in summary])
        survival = np.array([row["survival_rate"] for row in summary])
        success = np.array([row["command_success_rate"] for row in summary])
        vel_mae = np.array([row["vel_mae_mean"] for row in summary])
        gait_match = np.array([row["gait_match_rate"] for row in summary])

        label = doc.get("policy_name", f"run_{idx}")
        color = list(palette.values())[idx % 4] if "series" in list(palette.keys())[0] else palette["series_1"]

        # Survival rate
        axes[0].plot(speeds, survival, "o-", label=label, color=color, linewidth=2, markersize=6)
        axes[0].set_ylabel("Survival Rate", fontsize=12)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, color=palette["grid"], linewidth=0.8)
        axes[0].set_title("Survival Rate vs Commanded Speed", fontsize=13)

        # Command success rate
        axes[1].plot(speeds, success, "s-", label=label, color=color, linewidth=2, markersize=6)
        axes[1].set_ylabel("Command Success Rate", fontsize=12)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, color=palette["grid"], linewidth=0.8)
        axes[1].set_title("Velocity Tracking Success vs Speed", fontsize=13)

        # Velocity MAE
        axes[2].plot(speeds, vel_mae, "^-", label=label, color=color, linewidth=2, markersize=6)
        axes[2].set_ylabel("Velocity MAE (m/s)", fontsize=12)
        axes[2].set_xlabel("Commanded |v| (m/s)", fontsize=12)
        axes[2].grid(True, color=palette["grid"], linewidth=0.8)
        axes[2].set_title("Velocity Tracking Error vs Speed", fontsize=13)

        # Gait match rate
        axes[3].plot(speeds, gait_match, "d-", label=label, color=color, linewidth=2, markersize=6)
        axes[3].set_ylabel("Gait Match Rate", fontsize=12)
        axes[3].set_xlabel("Commanded |v| (m/s)", fontsize=12)
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].grid(True, color=palette["grid"], linewidth=0.8)
        axes[3].set_title("Gait Pattern Match vs Speed", fontsize=13)

    # Legend outside plots if multiple runs
    if n_runs > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
                   ncol=min(n_runs, 4), fontsize=10, frameon=True)

    fig.suptitle("Stage 2 (Transition) Evaluation: Velocity Tracking from Stand",
                 fontsize=14, fontweight="bold", y=0.995 if n_runs <= 1 else 0.95)
    fig.tight_layout()
    if n_runs > 1:
        fig.subplots_adjust(top=0.88)

    # Save (PNG only)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"eval_transition_{'multi' if n_runs > 1 else metrics_paths[0].parent.name}"
    png_path = out_dir / f"{base}.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=palette["surface"])
    plt.close(fig)

    print(f"[plot_eval_stage_transition] saved: {png_path}")
    return png_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Stage 2 (transition) evaluation results."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to metrics.json from eval_stage_transition.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: alongside first metrics.json)",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Use dark theme",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay multiple runs for comparison (implied by multiple --metrics)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    metrics_paths = [Path(p) for p in args.metrics]
    for p in metrics_paths:
        if not p.exists():
            raise FileNotFoundError(f"metrics file not found: {p}")

    out_dir = Path(args.out_dir) if args.out_dir else metrics_paths[0].parent / "plots"
    plot_evaluation(metrics_paths, out_dir, dark=args.dark)


if __name__ == "__main__":
    main()