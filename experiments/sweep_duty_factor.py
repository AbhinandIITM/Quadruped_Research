"""
sweep_duty_factor.py
====================
Duty-factor sweep for the CoT investigation.

Motivation
----------
The benchmark harness reports two cost-of-transport conventions (added
2026-10-17): a *gross* CoT from ``sum |tau*qdot|`` and a *net* CoT from
``sum tau*qdot``. On ``waypoint_square`` the gross figure was 1.65 while the
net figure was 0.92, i.e. 44% of the mechanical work was recovered in place.
That means most of the gap to the 1.25-1.45 target band is **measurement
convention**, not mechanics.

This sweep answers the residual question: after correcting the convention,
does the *duty factor* still leave anything on the table? Duty factor is
``stance / (stance + swing)``; raising it toward high-duty trot is the
classical CoT reducer because it cuts vertical oscillation and impact loss.

Each trial overrides ``GaitConfig`` inside the harness and runs the ``s_curve``
scenario headlessly, reporting both CoT conventions plus peak attitude, so a
duty factor that improves CoT at the cost of stability margin is visible.

Usage:
    python experiments/sweep_duty_factor.py
    python experiments/sweep_duty_factor.py --trials 0.50,0.58,0.65,0.70
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bootstrap() -> str:
    """Resolve the repo root and put import roots on sys.path."""
    root = REPO_ROOT if os.path.exists(os.path.join(REPO_ROOT, "unitree_go2")) else os.getcwd()
    for p in (str(root), os.path.join(str(root), "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


_ROOT = _bootstrap()

import evaluate_champ_trajectories as harness  # noqa: E402
from go2_champ_control import GaitConfig  # noqa: E402

#: Duty factors swept by default. 0.50 is the shipped value.
DEFAULT_TRIALS = (0.50, 0.58, 0.65, 0.70)

#: Fixed stride period. Only the stance/swing split changes across trials, so
#: results are comparable rather than confounded by a speed change.
STRIDE_PERIOD_S = 0.44


def _scenario_path(root: str) -> str:
    """Locate the s_curve scenario JSON in either repo layout."""
    for candidate in (os.path.join(root, "experiments", "scenarios", "s_curve.json"),
                      os.path.join(root, "scenarios", "s_curve.json")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("s_curve.json not found")


def _patch_gait(duty: float):
    """Monkeypatch the harness controller to use a specific duty factor."""
    stance = STRIDE_PERIOD_S * duty
    swing = STRIDE_PERIOD_S - stance
    cfg = GaitConfig(stance_duration=stance, swing_duration=swing)
    original = harness.Go2ChampController

    class _Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["gait_config"] = cfg
            super().__init__(*args, **kwargs)

    harness.Go2ChampController = _Patched
    return original, cfg


def run_trial(duty: float, scenario_cfg: Dict) -> Dict:
    """Run one trial at the given duty factor and summarise the metrics."""
    original, cfg = _patch_gait(duty)
    try:
        runner = harness.ScenarioRunner()
        result = runner.run_scenario(
            scenario_cfg,
            output_dir=None,
            save_plots=False,
            gui=False,
            vision=False,
        )
    finally:
        harness.Go2ChampController = original

    return {
        "duty": duty,
        "stance_s": cfg.stance_duration,
        "swing_s": cfg.swing_duration,
        "cot_abs": float(result.get("estimated_cost_of_transport", math.nan)),
        "cot_signed": float(result.get("estimated_cost_of_transport_signed", math.nan)),
        "recovery": float(result.get("recovery_fraction", math.nan)),
        "path_len_m": float(result.get("total_path_length_m", math.nan)),
        "roll_peak_deg": math.degrees(float(result.get("base_roll_peak_abs_rad", 0.0))),
        "pitch_peak_deg": math.degrees(float(result.get("base_pitch_peak_abs_rad", 0.0))),
        "passed": bool(result.get("passed")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep gait duty factor and report CoT.")
    parser.add_argument("--trials", type=str, default=",".join(str(t) for t in DEFAULT_TRIALS))
    args = parser.parse_args()

    duties = [float(t) for t in args.trials.split(",") if t.strip()]

    scenario_cfg = harness.load_scenario(_scenario_path(_ROOT))

    print("=" * 94)
    print(f"DUTY FACTOR SWEEP  (stride period fixed at {STRIDE_PERIOD_S} s, "
          f"{float(scenario_cfg.get('duration_s', 0)):.0f} s scenario)")
    print("=" * 94)
    print(
        f"{'duty':>6} {'stance':>7} {'swing':>7} {'CoT gross':>10} {'CoT net':>8} "
        f"{'recovery':>9} {'path m':>7} {'roll deg':>9} {'pitch deg':>10} {'pass':>6}"
    )
    print("-" * 94)

    rows: List[Dict] = []
    for duty in duties:
        row = run_trial(duty, scenario_cfg)
        rows.append(row)
        print(
            f"{row['duty']:>6.2f} {row['stance_s']:>7.3f} {row['swing_s']:>7.3f} "
            f"{row['cot_abs']:>10.2f} {row['cot_signed']:>8.2f} {row['recovery'] * 100:>8.0f}% "
            f"{row['path_len_m']:>7.2f} {row['roll_peak_deg']:>9.1f} {row['pitch_peak_deg']:>10.1f} "
            f"{'YES' if row['passed'] else 'NO':>6}"
        )

    print("-" * 94)

    ok = [r for r in rows if r["passed"]]
    if not ok:
        print("\nNo trial passed the gate -- duty-factor change is not viable as shipped.")
        return 1

    best_net = min(ok, key=lambda r: r["cot_signed"])
    best_gross = min(ok, key=lambda r: r["cot_abs"])
    shipped = next((r for r in rows if abs(r["duty"] - 0.50) < 1e-6), None)

    print(f"\nBest NET CoT   : duty {best_net['duty']:.2f} -> {best_net['cot_signed']:.3f}")
    print(f"Best GROSS CoT : duty {best_gross['duty']:.2f} -> {best_gross['cot_abs']:.3f}")
    if shipped is not None and shipped["cot_signed"] > 0 and shipped["cot_abs"] > 0:
        d_net = best_net["cot_signed"] - shipped["cot_signed"]
        d_gross = best_gross["cot_abs"] - shipped["cot_abs"]
        print(f"vs shipped (0.50): net {d_net:+.3f} ({d_net / shipped['cot_signed'] * 100:+.1f}%), "
              f"gross {d_gross:+.3f} ({d_gross / shipped['cot_abs'] * 100:+.1f}%)")

    worst_roll = max(r["roll_peak_deg"] for r in ok)
    worst_pitch = max(r["pitch_peak_deg"] for r in ok)
    print(f"\nStability envelope across passing trials: roll peak {worst_roll:.1f} deg, "
          f"pitch peak {worst_pitch:.1f} deg")
    if worst_roll > 12.0 or worst_pitch > 14.0:
        print("  ^ WARNING: attitude peaks high enough that CoT gains should NOT be trusted "
              "without checking contact duty factors and fall margin.")

    print("\nReminder: CoT gross credits no regeneration; CoT net is the comparable "
          "convention for MIT Cheetah / RL baselines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
