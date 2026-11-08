#!/usr/bin/env python3
"""
Multistage comparison plot: Stand → Balance → Transition → Walk.

Reads evaluation metrics from all four stages and produces a comparison figure:
- Velocity tracking MAE vs speed (all stages)
- Survival rate vs speed
- Command success rate vs speed
- Gait match rate vs speed
- Cost of transport vs speed
- Stage comparison bar chart (final metrics at 0.5 m/s)

Saves as PNG.

Usage:
    python -m rl.plot_multistage --metrics-dir rl/results/
    python -m rl.plot_multistage --metrics-dir rl/results/ --stages stand balance transition walk
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


# Reference palette (validated for light/dark, CVD-safe)
SERIES_COLORS = {
    "light": {
        "stand": "#2a78d6",      # blue
        "balance": "#eb6834",    # orange
        "transition": "#1baf7a", # aqua
        "walk": "#eda100",       # yellow
        "fill": "#2a78d633",
        "grid": "#52514e4d",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "surface": "#fcfcfb",
    },
    "dark": {
        "stand": "#3987e5",
        "balance": "#d95926",
        "transition": "#199e70",
        "walk": "#c98500",
        "fill": "#3987e533",
        "grid": "#c3c2b74d",
        "text_primary": "#f0efec",
        "text_secondary": "#c3c2b7",
        "surface": "#1a1a19",
    },
}

STAGE_ORDER = ["stand", "balance", "transition", "walk"]
STAGE_LABELS = {
    "stand": "Stand (0)",
    "balance": "Balance (1)",
    "transition": "Transition (2)",
    "walk": "Walk (3)",
}


def load_metrics(json_path: Path) -> dict[str, Any]:
    """Load evaluation metrics from JSON."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def discover_stage_metrics(metrics_dir: Path, stages: list[str] | None = None) -> dict[str, dict]:
    """Discover the most recent eval_*.json for each stage.

    Accepts either:
      * ``metrics_dir`` = project ``rl/results/`` (eval files at ``rl/results/eval/{stage}/eval_*.json``)
      * ``metrics_dir`` = ``rl/results/eval/`` (eval files at ``{metrics_dir}/{stage}/eval_*.json``)
    """
    stages = stages or STAGE_ORDER
    results = {}
    for stage in stages:
        candidates: list[Path] = []
        for base in (metrics_dir, metrics_dir / "eval", metrics_dir / "eval" / stage):
            if base.exists():
                candidates += list(base.rglob(f"eval_*.json")) if base.is_dir() else []
        # filter to those whose content indicates this stage, or path contains stage
        stage_cands = []
        for c in candidates:
            if stage in str(c).lower():
                stage_cands.append(c)
        # fallback: any eval json
        if not stage_cands:
            stage_cands = candidates
        if not stage_cands:
            continue
        stage_cands = sorted(set(stage_cands), key=lambda p: (p.stat().st_mtime, p.name))
        results[stage] = load_metrics(stage_cands[-1])
    return results


def extract_speed_curves(metrics: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract per-speed metrics from evaluation results."""
    # Expected structure: metrics["results"] is list of per-episode results
    # Each has: command, survival, vel_mae, cot, gait_label, gait_match
    speeds = []
    vel_mae = []
    survival = []
    cmd_success = []
    gait_match = []
    cot = []

    if "results" not in metrics:
        return {}

    for r in metrics["results"]:
        cmd = r.get("command", {})
        speed = np.hypot(cmd.get("vx", 0), cmd.get("vy", 0))
        speeds.append(speed)
        vel_mae.append(r.get("vel_mae", np.nan))
        survival.append(1.0 if r.get("survival", False) else 0.0)
        cmd_success.append(1.0 if r.get("command_success", False) else 0.0)
        gait_match.append(1.0 if r.get("gait_match", False) else 0.0)
        cot.append(r.get("cot", np.nan))

    return {
        "speeds": np.array(speeds),
        "vel_mae": np.array(vel_mae),
        "survival": np.array(survival),
        "cmd_success": np.array(cmd_success),
        "gait_match": np.array(gait_match),
        "cot": np.array(cot),
    }


def plot_comparison(
    stage_metrics: dict[str, dict],
    output_path: Path,
    theme: str = "light",
) -> None:
    """Create the multistage comparison figure."""
    colors = SERIES_COLORS[theme]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.patch.set_facecolor(colors["surface"])
    ax_list = axes.flatten()

    # Speed bins for x-axis
    speed_bins = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.1])

    # Subplot 1: Velocity MAE vs Speed
    ax = ax_list[0]
    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        if len(curves.get("speeds", [])) == 0:
            continue
        # Bin by speed
        binned_mae = []
        binned_speeds = []
        for i in range(len(speed_bins) - 1):
            mask = (curves["speeds"] >= speed_bins[i]) & (curves["speeds"] < speed_bins[i+1])
            if np.any(mask):
                binned_speeds.append((speed_bins[i] + speed_bins[i+1]) / 2)
                binned_mae.append(np.nanmean(curves["vel_mae"][mask]))
        if binned_mae:
            ax.plot(binned_speeds, binned_mae, "o-", label=STAGE_LABELS[stage],
                   color=colors[stage], linewidth=2, markersize=6)

    ax.set_xlabel("Commanded Speed (m/s)", color=colors["text_secondary"])
    ax.set_ylabel("Velocity MAE (m/s)", color=colors["text_secondary"])
    ax.set_title("Velocity Tracking Error", color=colors["text_primary"])
    ax.legend(fontsize=9)
    ax.grid(True, color=colors["grid"], linewidth=0.5)

    # Subplot 2: Survival Rate vs Speed
    ax = ax_list[1]
    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        if len(curves.get("speeds", [])) == 0:
            continue
        binned_surv = []
        binned_speeds = []
        for i in range(len(speed_bins) - 1):
            mask = (curves["speeds"] >= speed_bins[i]) & (curves["speeds"] < speed_bins[i+1])
            if np.any(mask):
                binned_speeds.append((speed_bins[i] + speed_bins[i+1]) / 2)
                binned_surv.append(np.mean(curves["survival"][mask]) * 100)
        if binned_surv:
            ax.plot(binned_speeds, binned_surv, "o-", label=STAGE_LABELS[stage],
                   color=colors[stage], linewidth=2, markersize=6)

    ax.set_xlabel("Commanded Speed (m/s)", color=colors["text_secondary"])
    ax.set_ylabel("Survival Rate (%)", color=colors["text_secondary"])
    ax.set_title("Survival Rate", color=colors["text_primary"])
    ax.set_ylim(0, 105)
    ax.grid(True, color=colors["grid"], linewidth=0.5)

    # Subplot 3: Command Success Rate vs Speed
    ax = ax_list[2]
    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        if len(curves.get("speeds", [])) == 0:
            continue
        binned_succ = []
        binned_speeds = []
        for i in range(len(speed_bins) - 1):
            mask = (curves["speeds"] >= speed_bins[i]) & (curves["speeds"] < speed_bins[i+1])
            if np.any(mask):
                binned_speeds.append((speed_bins[i] + speed_bins[i+1]) / 2)
                binned_succ.append(np.mean(curves["cmd_success"][mask]) * 100)
        if binned_succ:
            ax.plot(binned_speeds, binned_succ, "o-", label=STAGE_LABELS[stage],
                   color=colors[stage], linewidth=2, markersize=6)

    ax.set_xlabel("Commanded Speed (m/s)", color=colors["text_secondary"])
    ax.set_ylabel("Command Success (%)", color=colors["text_secondary"])
    ax.set_title("Command Tracking Success", color=colors["text_primary"])
    ax.set_ylim(0, 105)
    ax.grid(True, color=colors["grid"], linewidth=0.5)

    # Subplot 4: Gait Match vs Speed
    ax = ax_list[3]
    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        if len(curves.get("speeds", [])) == 0:
            continue
        binned_gait = []
        binned_speeds = []
        for i in range(len(speed_bins) - 1):
            mask = (curves["speeds"] >= speed_bins[i]) & (curves["speeds"] < speed_bins[i+1])
            if np.any(mask):
                binned_speeds.append((speed_bins[i] + speed_bins[i+1]) / 2)
                binned_gait.append(np.mean(curves["gait_match"][mask]) * 100)
        if binned_gait:
            ax.plot(binned_speeds, binned_gait, "o-", label=STAGE_LABELS[stage],
                   color=colors[stage], linewidth=2, markersize=6)

    ax.set_xlabel("Commanded Speed (m/s)", color=colors["text_secondary"])
    ax.set_ylabel("Gait Match (%)", color=colors["text_secondary"])
    ax.set_title("Gait Pattern Match", color=colors["text_primary"])
    ax.set_ylim(0, 105)
    ax.grid(True, color=colors["grid"], linewidth=0.5)

    # Subplot 5: Cost of Transport vs Speed
    ax = ax_list[4]
    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        if len(curves.get("speeds", [])) == 0:
            continue
        binned_cot = []
        binned_speeds = []
        for i in range(len(speed_bins) - 1):
            mask = (curves["speeds"] >= speed_bins[i]) & (curves["speeds"] < speed_bins[i+1])
            if np.any(mask):
                binned_speeds.append((speed_bins[i] + speed_bins[i+1]) / 2)
                binned_cot.append(np.nanmean(curves["cot"][mask]))
        if binned_cot:
            ax.plot(binned_speeds, binned_cot, "o-", label=STAGE_LABELS[stage],
                   color=colors[stage], linewidth=2, markersize=6)

    ax.set_xlabel("Commanded Speed (m/s)", color=colors["text_secondary"])
    ax.set_ylabel("Cost of Transport (J/m)", color=colors["text_secondary"])
    ax.set_title("Energy Efficiency (CoT)", color=colors["text_primary"])
    ax.grid(True, color=colors["grid"], linewidth=0.5)

    # Subplot 6: Bar chart at 0.5 m/s (summary)
    ax = ax_list[5]
    target_speed = 0.5
    x_pos = np.arange(len(STAGE_ORDER))
    bar_width = 0.2
    metrics_at_target = {
        "vel_mae": [],
        "survival": [],
        "cmd_success": [],
    }

    for stage in STAGE_ORDER:
        if stage not in stage_metrics:
            metrics_at_target["vel_mae"].append(np.nan)
            metrics_at_target["survival"].append(np.nan)
            metrics_at_target["cmd_success"].append(np.nan)
            continue
        curves = extract_speed_curves(stage_metrics[stage])
        mask = np.abs(curves["speeds"] - target_speed) < 0.15
        if np.any(mask):
            metrics_at_target["vel_mae"].append(np.nanmean(curves["vel_mae"][mask]))
            metrics_at_target["survival"].append(np.mean(curves["survival"][mask]) * 100)
            metrics_at_target["cmd_success"].append(np.mean(curves["cmd_success"][mask]) * 100)
        else:
            metrics_at_target["vel_mae"].append(np.nan)
            metrics_at_target["survival"].append(np.nan)
            metrics_at_target["cmd_success"].append(np.nan)

    # Normalize MAE for plotting (invert so higher is better)
    mae_vals = np.array(metrics_at_target["vel_mae"])
    surv_vals = np.array(metrics_at_target["survival"])
    succ_vals = np.array(metrics_at_target["cmd_success"])

    # Plot normalized bars (0-1 scale)
    mae_norm = 1.0 - (mae_vals / np.nanmax(mae_vals) if np.nanmax(mae_vals) > 0 else 1)
    surv_norm = surv_vals / 100.0
    succ_norm = succ_vals / 100.0

    ax.bar(x_pos - bar_width, mae_norm, bar_width, label="Tracking (1-MAE)", color=colors["stand"])
    ax.bar(x_pos, surv_norm, bar_width, label="Survival", color=colors["balance"])
    ax.bar(x_pos + bar_width, succ_norm, bar_width, label="Cmd Success", color=colors["transition"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Normalized Score (0-1)", color=colors["text_secondary"])
    ax.set_title(f"Summary at {target_speed} m/s", color=colors["text_primary"])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(True, color=colors["grid"], linewidth=0.5, axis="y")

    fig.suptitle("Multistage Curriculum Comparison: Stand → Balance → Transition → Walk",
                 color=colors["text_primary"], fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, facecolor=colors["surface"],
                bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multistage curriculum comparison plot")
    parser.add_argument("--metrics-dir", type=Path, required=True,
                        help="Directory containing stage evaluation results")
    parser.add_argument("--stages", nargs="*", default=None,
                        help="Stages to include (default: all four)")
    parser.add_argument("--theme", choices=["light", "dark"], default="light",
                        help="Color theme")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: metrics_dir/plots)")
    args = parser.parse_args(argv)

    stage_metrics = discover_stage_metrics(args.metrics_dir, args.stages)
    if not stage_metrics:
        print(f"No evaluation metrics found in {args.metrics_dir}")
        return 1

    output_dir = args.output_dir or (args.metrics_dir / "plots")
    output_path = output_dir / "multistage_comparison"
    plot_comparison(stage_metrics, output_path, args.theme)
    print(f"Saved comparison to {output_path}.pdf and {output_path}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())