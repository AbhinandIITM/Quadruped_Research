"""
language_ablation.py
====================
Ablation of the natural-language abstraction layer.

Question
--------
The paper claims that putting a natural-language interface in front of the
quadruped controller has a *measurable cost*, and that the paper will report
that cost instead of only the successes of the controller underneath.  This
script measures that cost.

Design
------
Three conditions are run over the same benchmark scenarios, through the *same*
harness (`src/evaluate_champ_trajectories.py`), so the comparison is
apples-to-apples:

  D (direct)     The scenario's continuous velocity commands, exactly as the
                  unmodified harness runs them today.  This is the
                  "no language interface" reference.

  Q (quantised)  The same continuous velocity commands, but pushed through a
                  quantiser that emulates the language vocabulary: each
                  commanded (vx, vy, yaw_rate) triple is snapped to the nearest
                  point of a discrete vocabulary and held for a fixed dwell
                  time (1.0 s, the contract's typical command duration).

  S (quantised + trailing stand)
                  As Q, but every quantised command block is terminated by the
                  contract's 1.0 s stationary stand -- the same invariant the
                  refinement chain enforces in
                  `src/go2_llm_commands.py::parse_and_validate_response`.

WHERE the quantiser is applied
------------------------------
The quantiser is applied to the **command**, inside the controller, before the
controller sees it:

    harness -> MotionCommand -> [quantiser] -> MotionCommand -> controller

The recorded telemetry is never touched.  Every degradation reported below is
produced by the physics and the controller reacting to a coarser command
stream, never by post-processing the measurements.  A reviewer can verify this
by reading `CommandQuantiser.apply` below and by checking that the patched
controller in `_make_patched_controller` substitutes the command *before*
calling `super().step(...)`.

The monkeypatch follows the established pattern in
`experiments/sweep_duty_factor.py`: the harness is imported as a module and
`harness.Go2ChampController` is replaced by a subclass for the duration of one
run.

Sensitivity variants
--------------------
The headline numbers use one explicitly documented configuration.  Three
further variants are run so that the reader can see which modelling choices
the conclusion depends on:

  primary              V1 vocabulary, ties broken away from zero, trailing
                       stand after every command block.
  tie_toward_zero      same vocabulary, ties broken toward zero (the
                       opposite convention for the handful of scenario speeds
                       that fall exactly between two vocabulary levels).
  brief_vocabulary     the vocabulary proposed in the experiment brief, whose
                       +/-0.90 rad/s yaw levels lie outside the shipped
                       SafetyBounds and are therefore trimmed by the
                       controller's own clamp.
  single_trailing_stand
                       trailing stand applied ONCE, at the end of the whole
                       run, instead of after every command block.  This is the
                       conservative reading of the invariant and bounds how
                       much of the condition-S effect is due to the
                       recurrence of the standstill.

Usage
-----
    python experiments/language_ablation.py

Headless, no GUI, no network, no model server, no API keys.  Exit code is
non-zero only on an internal error; a scenario failing its declared gate is a
RESULT, not an error.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bootstrap() -> str:
    """Resolve the repo root and put import roots on sys.path.

    Same bootstrap as `experiments/sweep_duty_factor.py`: the repo root (for
    `unitree_go2/`, the model XML) and `<root>/src` (for the harness and the
    controller modules).
    """
    root = REPO_ROOT if os.path.exists(os.path.join(REPO_ROOT, "unitree_go2")) else os.getcwd()
    for p in (str(root), os.path.join(str(root), "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


_ROOT = _bootstrap()

import evaluate_champ_trajectories as harness  # noqa: E402
from go2_champ_control import (  # noqa: E402
    MotionCommand,
    MotionMode,
    SafetyBounds,
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

#: Command-driven scenarios of the benchmark suite.  `waypoint_square` is
#: included because the language interface also has to express waypoint goals;
#: note that the closed-loop navigator derives its own velocity, so the
#: velocity vocabulary does not bind on that path (see the writeup).
DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "forward_speed_step",
    "s_curve",
    "stop_and_hold",
    "turn_in_place",
    "waypoint_square",
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vocabulary:
    """A discrete command vocabulary: the set of magnitudes a language model
    can actually express for each channel."""

    name: str
    vx: Tuple[float, ...]
    vy: Tuple[float, ...]
    yaw: Tuple[float, ...]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vx_mps": list(self.vx),
            "vy_mps": list(self.vy),
            "yaw_rate_radps": list(self.yaw),
            "rationale": self.rationale,
        }


#: Primary vocabulary.  Every level is inside the *shipped* SafetyBounds
#: (src/go2_champ_control.py:219-231: vx in [-0.30, 0.60], vy in [-0.35, 0.35],
#: yaw_rate in [-0.80, 0.80]), and the grid is evenly spaced, so the
#: quantisation error is a pure statement about the grid, not about the safety
#: clamp.  This is the vocabulary used for the headline numbers.
PRIMARY_VOCABULARY = Vocabulary(
    name="V1_in_bounds_uniform",
    vx=(0.0, 0.15, 0.30, 0.45, 0.60),
    vy=(0.0, -0.20, 0.20),
    yaw=(0.0, -0.30, -0.60, 0.30, 0.60),
    rationale=(
        "Uniform 0.15 m/s grid over the shipped vx range [0, 0.60] with 0 as the "
        "canonical 'stop' token; uniform 0.20 m/s grid over the shipped vy range "
        "[-0.35, 0.35]; uniform 0.30 rad/s grid over the shipped yaw_rate range "
        "[-0.80, 0.80].  Every level is admissible under the shipped "
        "SafetyBounds, so no level is silently trimmed by the safety clamp."
    ),
)

#: Sensitivity vocabulary: the literal grid proposed in the experiment brief
#: (yaw_rate in {0, +/-0.30, +/-0.90}).  The +/-0.90 levels lie *outside* the
#: shipped SafetyBounds (max_yaw_rate = 0.80), so a real pipeline would clamp
#: them.  Running it shows that the headline conclusion is a property of the
#: abstraction, not of one particular grid.
BRIEF_VOCABULARY = Vocabulary(
    name="V2_brief_literal",
    vx=(0.0, 0.15, 0.30, 0.45, 0.60),
    vy=(0.0, -0.20, 0.20),
    yaw=(0.0, -0.30, -0.90, 0.30, 0.90),
    rationale=(
        "The vocabulary proposed in the experiment brief.  Its +/-0.90 rad/s "
        "yaw levels exceed the shipped SafetyBounds max_yaw_rate = 0.80, so the "
        "controller's own clamp trims them to +/-0.80.  Included as a "
        "sensitivity check on the primary grid."
    ),
)

#: Tie-breaking conventions for q(v) = argmin_{c in V} |v - c|.
TIE_BREAKS = ("away_from_zero", "toward_zero")

#: Distances closer than this are treated as a tie.  Without it, a value that
#: is exactly between two levels is not a tie in binary floating point
#: (|0.45-0.30| = 0.15000000000000002 vs |0.45-0.60| = 0.14999999999999997)
#: and the rule would silently pick whichever side the rounding noise favours.
TIE_EPS = 1e-9


def quantise(value: float, grid: Sequence[float], tie_break: str = "away_from_zero") -> float:
    """Snap `value` to the nearest point of `grid`.

    The operator is

        q(v) = argmin_{c in V} |v - c|

    A value exactly between two levels is a tie; `tie_break` decides it:

    * ``"away_from_zero"``  -> round half away from zero (the primary rule);
    * ``"toward_zero"``     -> round half toward zero (the conservative rule).

    Both rules are deterministic and symmetric about the origin.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break '{tie_break}'")
    best = grid[0]
    best_key: Optional[Tuple[float, float, float]] = None
    for c in grid:
        d = round(abs(value - c), 9)  # collapse float noise so real ties are ties
        if tie_break == "away_from_zero":
            key = (d, -abs(c), -c)   # nearest, then larger |c|
        else:
            key = (d, abs(c), -c)     # nearest, then smaller |c|
        if best_key is None or key < best_key:
            best_key = key
            best = c
    return float(best)


# ---------------------------------------------------------------------------
# Conditions and variants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionSpec:
    """One experimental condition."""

    label: str                 # "direct" | "quantised" | "quantised+stand" | ...
    identity: bool             # pass the command through untouched
    trailing_stand_mode: str   # "none" | "per_block" | "once_at_end"
    description: str


CONDITION_DIRECT = ConditionSpec(
    label="direct",
    identity=True,
    trailing_stand_mode="none",
    description=(
        "D -- the scenario's continuous velocity commands, exactly as the "
        "unmodified harness runs them today. The 'no language interface' "
        "reference."
    ),
)
CONDITION_QUANTISED = ConditionSpec(
    label="quantised",
    identity=False,
    trailing_stand_mode="none",
    description=(
        "Q -- each commanded (vx, vy, yaw_rate) is snapped to the nearest "
        "vocabulary point and held for one dwell period (1.0 s). The model can "
        "only say a small number of things, and only every second or so."
    ),
)
CONDITION_QUANTISED_STAND = ConditionSpec(
    label="quantised+stand",
    identity=False,
    trailing_stand_mode="per_block",
    description=(
        "S -- as Q, but every command block is terminated by the contract's "
        "1.0 s stationary stand, so the interface periodically commands a "
        "standstill that a continuous velocity reference would not have."
    ),
)
CONDITION_QUANTISED_STAND_ONCE = ConditionSpec(
    label="quantised+stand-once",
    identity=False,
    trailing_stand_mode="once_at_end",
    description=(
        "S-once -- as Q, with a single 1.0 s stationary stand appended at the "
        "very end of the run (the conservative reading of the trailing-stand "
        "invariant, applied once instead of after every command block)."
    ),
)


@dataclass(frozen=True)
class Variant:
    """A full experimental configuration."""

    name: str
    vocabulary: Vocabulary
    tie_break: str
    conditions: Tuple[ConditionSpec, ...]
    description: str


PRIMARY_VARIANT = Variant(
    name="primary",
    vocabulary=PRIMARY_VOCABULARY,
    tie_break="away_from_zero",
    conditions=(CONDITION_DIRECT, CONDITION_QUANTISED, CONDITION_QUANTISED_STAND),
    description=(
        "Headline configuration: the in-bounds uniform vocabulary, ties broken "
        "away from zero, trailing stand after every command block."
    ),
)

SENSITIVITY_VARIANTS: Tuple[Variant, ...] = (
    Variant(
        name="tie_toward_zero",
        vocabulary=PRIMARY_VOCABULARY,
        tie_break="toward_zero",
        conditions=(CONDITION_DIRECT, CONDITION_QUANTISED, CONDITION_QUANTISED_STAND),
        description=(
            "Same vocabulary, but a value exactly between two levels is rounded "
            "toward zero instead of away from zero.  This flips the sign of the "
            "quantisation error for the scenario speeds that fall on a tie."
        ),
    ),
    Variant(
        name="brief_vocabulary",
        vocabulary=BRIEF_VOCABULARY,
        tie_break="away_from_zero",
        conditions=(CONDITION_DIRECT, CONDITION_QUANTISED, CONDITION_QUANTISED_STAND),
        description=(
            "The vocabulary proposed in the experiment brief, whose +/-0.90 rad/s "
            "yaw levels are outside the shipped SafetyBounds and are trimmed by "
            "the controller's own clamp."
        ),
    ),
    Variant(
        name="single_trailing_stand",
        vocabulary=PRIMARY_VOCABULARY,
        tie_break="away_from_zero",
        conditions=(CONDITION_DIRECT, CONDITION_QUANTISED, CONDITION_QUANTISED_STAND_ONCE),
        description=(
            "Trailing stand applied ONCE at the end of the run instead of after "
            "every command block: bounds how much of the condition-S effect is "
            "due to the standstill recurring rather than to its existence."
        ),
    ),
)


# ---------------------------------------------------------------------------
# The quantiser
# ---------------------------------------------------------------------------

def _mode_str(cmd: MotionCommand) -> str:
    mode = cmd.mode
    return mode.value if isinstance(mode, MotionMode) else str(mode)


class CommandQuantiser:
    """Rewrites the *command* the controller receives.

    This object never touches telemetry.  It is called once per control step
    with the raw command the harness built, and returns the command the
    controller is actually allowed to see.

    Behaviours, selected by the constructor:

    * ``identity=True``  -- the command is passed through untouched (condition
      D).  Also used for the sanity check that the monkeypatch does not
      perturb the baseline.

    * ``trailing_stand_mode="none"`` -- the command is sampled at the start of
      every dwell block, snapped to the vocabulary, and held for the whole
      block (condition Q).

    * ``trailing_stand_mode="per_block"`` -- as above, but the dwell grid
      alternates "move block, stand block": each quantised command block is
      terminated by the contract's 1.0 s stationary stand (condition S).  The
      stand block emits `MotionCommand(mode=STAND, vx=vy=yaw_rate=0)`, exactly
      the command `parse_and_validate_response` appends.

    * ``trailing_stand_mode="once_at_end"`` -- as "none", except that the last
      `dwell_s` seconds of the run are a stand block.

    The dwell grid is anchored at t = 0 and has period `dwell_s`, so block k
    covers [k*dwell, (k+1)*dwell).  The block index is a pure function of the
    step time, so the schedule can be reconstructed exactly from the telemetry
    timestamps.
    """

    def __init__(
        self,
        vocabulary: Optional[Vocabulary] = None,
        dwell_s: float = 1.0,
        trailing_stand_mode: str = "none",
        tie_break: str = "away_from_zero",
        identity: bool = False,
        total_duration_s: Optional[float] = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.dwell_s = float(dwell_s)
        self.trailing_stand_mode = trailing_stand_mode
        self.tie_break = tie_break
        self.identity = bool(identity)
        self.total_duration_s = total_duration_s
        #: step_time -> (vx, vy, yaw_rate, mode) actually delivered to the controller
        self.delivered: Dict[float, Tuple[float, float, float, str]] = {}
        self._active_key: Optional[int] = None
        self._active_cmd: Optional[MotionCommand] = None

    # -- block geometry ---------------------------------------------------

    def block_kind(self, t: float) -> Tuple[str, int]:
        """Return (kind, key) of the block covering time `t`.

        kind is "move" or "stand".  The key is unique across both kinds.
        """
        if self.trailing_stand_mode == "once_at_end":
            end = self.total_duration_s if self.total_duration_s is not None else math.inf
            if t >= end - self.dwell_s:
                return "stand", -1
            return "move", int(math.floor(t / self.dwell_s))
        if self.trailing_stand_mode == "per_block":
            cycle = 2.0 * self.dwell_s
            c = int(math.floor(t / cycle))
            phase = t - c * cycle
            if phase < self.dwell_s:
                return "move", 2 * c
            return "stand", 2 * c + 1
        return "move", int(math.floor(t / self.dwell_s))

    # -- the command rewrite ----------------------------------------------

    def apply(self, command: MotionCommand, step_time: float) -> MotionCommand:
        """Return the command the controller should actually receive."""
        if self.identity:
            self.delivered[step_time] = (
                float(command.vx),
                float(command.vy),
                float(command.yaw_rate),
                _mode_str(command),
            )
            return command

        kind, key = self.block_kind(step_time)

        if kind == "stand":
            # The contract's trailing stand: a stationary hold for one
            # command duration, with the velocities zeroed.
            out = MotionCommand(
                vx=0.0,
                vy=0.0,
                yaw_rate=0.0,
                duration=self.dwell_s,
                mode=MotionMode.STAND,
                gait=command.gait,
            )
            self._active_key = key
            self._active_cmd = out
            self.delivered[step_time] = (0.0, 0.0, 0.0, MotionMode.STAND.value)
            return out

        if key != self._active_key or self._active_cmd is None:
            self._active_key = key
            self._active_cmd = self._quantise_command(command)

        out = self._active_cmd
        self.delivered[step_time] = (
            float(out.vx),
            float(out.vy),
            float(out.yaw_rate),
            _mode_str(out),
        )
        return out

    def _quantise_command(self, command: MotionCommand) -> MotionCommand:
        """Build the quantised version of one command."""
        if command.mode == MotionMode.WAYPOINT:
            # The controller's WAYPOINT branch derives its velocity from the
            # closed-loop navigator and ignores the typed velocity triple
            # (src/go2_champ_control.py:1278-1298).  The waypoint *target* is
            # passed through live on purpose: re-issuing a stale target would
            # reset the navigator (the controller compares the commanded
            # waypoint against its own current one), which would be an artefact
            # of this patch rather than a property of the language interface.
            return MotionCommand(
                vx=0.0,
                vy=0.0,
                yaw_rate=0.0,
                duration=command.duration,
                mode=command.mode,
                waypoint=command.waypoint,
                waypoints=command.waypoints,
                gait=command.gait,
            )

        assert self.vocabulary is not None
        vx = quantise(float(command.vx), self.vocabulary.vx, self.tie_break)
        vy = quantise(float(command.vy), self.vocabulary.vy, self.tie_break)
        yaw = quantise(float(command.yaw_rate), self.vocabulary.yaw, self.tie_break)
        return MotionCommand(
            vx=vx,
            vy=vy,
            yaw_rate=yaw,
            duration=command.duration,
            mode=command.mode,
            heading_target=command.heading_target,
            stop_after=command.stop_after,
            gait=command.gait,
        )

    # -- reporting ---------------------------------------------------------

    def segments(self) -> List[Dict[str, Any]]:
        """Collapse the per-step delivery record into contiguous segments.

        This is the audit trail of what the controller was actually asked to
        do: it makes the quantisation visible without any telemetry
        manipulation.
        """
        if not self.delivered:
            return []
        out: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None
        for t in sorted(self.delivered):
            vx, vy, yaw, mode = self.delivered[t]
            if cur is None or (cur["vx"], cur["vy"], cur["yaw_rate"], cur["mode"]) != (vx, vy, yaw, mode):
                cur = {
                    "t_start_s": t,
                    "t_end_s": t,
                    "vx": vx,
                    "vy": vy,
                    "yaw_rate": yaw,
                    "mode": mode,
                }
                out.append(cur)
            else:
                cur["t_end_s"] = t
        return out


# ---------------------------------------------------------------------------
# Monkeypatching the harness
# ---------------------------------------------------------------------------

def _make_patched_controller(quantiser: CommandQuantiser):
    """Return a Go2ChampController subclass that quantises the command.

    Same pattern as `experiments/sweep_duty_factor.py`: the harness is
    imported as a module and the class it instantiates is replaced for the
    duration of one run.
    """
    original = harness.Go2ChampController

    class _Patched(original):  # type: ignore[misc, valid-type]
        """Controller that receives the quantised command, not the raw one."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._quantiser = quantiser

        def step(  # type: ignore[override]
            self,
            command,
            step_time,
            dt,
            base_pos,
            base_euler,
            base_linvel_body,
            base_angvel_body,
            joint_positions,
            sector_clearance=None,
        ):
            delivered = self._quantiser.apply(command, step_time)
            return super().step(
                delivered,
                step_time,
                dt,
                base_pos,
                base_euler,
                base_linvel_body,
                base_angvel_body,
                joint_positions,
                sector_clearance,
            )

    return _Patched


class _CapturingRunner(harness.ScenarioRunner):
    """ScenarioRunner that keeps the telemetry list.

    `_compute_metrics` is a pure pass-through to the harness implementation;
    only the history is stashed so that the commanded-vs-actual error can be
    computed against the command that was actually delivered.
    """

    def _compute_metrics(  # type: ignore[override]
        self,
        scenario_config,
        history,
        requested_duration,
        trajectory_start_time: float = 0.0,
    ):
        self.last_history = history
        return super()._compute_metrics(
            scenario_config, history, requested_duration, trajectory_start_time
        )


# ---------------------------------------------------------------------------
# Running one (scenario, condition) pair
# ---------------------------------------------------------------------------

#: Metrics that get a delta row in the report.
DELTA_METRICS: Tuple[str, ...] = (
    "forward_displacement_m",
    "lateral_displacement_m",
    "total_path_length_m",
    "velocity_tracking_rmse_mps",
    "yaw_rate_tracking_rmse_radps",
    "joint_tracking_rmse_rad",
    "base_height_mean_m",
    "base_roll_peak_abs_rad",
    "base_pitch_peak_abs_rad",
    "estimated_cost_of_transport",
    "estimated_cost_of_transport_signed",
    "estimated_energy_j",
    "estimated_energy_signed_j",
    "recovery_fraction",
    "contact_duty_factor_mean",
    "interface_cmd_vx_rmse_mps",
    "interface_cmd_vy_rmse_mps",
    "interface_cmd_planar_rmse_mps",
    "interface_cmd_yaw_rmse_radps",
)

#: The metric keys the harness itself computes (`_compute_metrics`).  The
#: sanity check compares exactly these between the unmodified harness and the
#: patched harness with an identity quantiser; bookkeeping fields (wall clock,
#: sample counts) are excluded because they are not measurements.
HARNESS_METRIC_KEYS: Tuple[str, ...] = (
    "run_duration_s",
    "requested_duration_s",
    "forward_displacement_m",
    "lateral_displacement_m",
    "total_path_length_m",
    "max_tracking_error_m",
    "rms_tracking_error_m",
    "final_heading_rad",
    "velocity_tracking_rmse_mps",
    "yaw_rate_tracking_rmse_radps",
    "joint_tracking_rmse_rad",
    "base_height_mean_m",
    "base_height_std_m",
    "base_height_min_m",
    "base_roll_std_rad",
    "base_pitch_std_rad",
    "base_roll_peak_abs_rad",
    "base_pitch_peak_abs_rad",
    "estimated_energy_j",
    "estimated_average_power_w",
    "estimated_cost_of_transport",
    "estimated_cost_of_transport_signed",
    "estimated_energy_signed_j",
    "estimated_average_power_signed_w",
    "estimated_energy_recovered_j",
    "recovery_fraction",
)


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so the JSON stays strict (no NaN).

    `json.dump(..., allow_nan=False)` refuses NaN/Infinity, and a NaN in the
    payload would silently poison a downstream table cell.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _interface_tracking(history: List[Dict[str, Any]], quantiser: CommandQuantiser) -> Dict[str, Any]:
    """Commanded-vs-actual error against the command the interface delivered.

    The harness's own `velocity_tracking_rmse_mps` compares the measured base
    velocity against the *post-slew-limiter* command that the controller
    actually applied (`info["cmd_vx"]`).  This adds the complementary number:
    the error against the command the *interface* asked for, i.e. the
    quantised (held) triple, before the slew limiter.
    """
    delivered = quantiser.delivered
    if not delivered:
        return {
            "interface_cmd_vx_rmse_mps": None,
            "interface_cmd_vy_rmse_mps": None,
            "interface_cmd_planar_rmse_mps": None,
            "interface_cmd_yaw_rmse_radps": None,
            "interface_samples": 0,
            "interface_missing_samples": 0,
        }

    dvx: List[float] = []
    dvy: List[float] = []
    dyaw: List[float] = []
    missing = 0
    for rec in history:
        t = float(rec["time_s"])
        entry = delivered.get(t)
        if entry is None:
            missing += 1
            continue
        vx_req, vy_req, yaw_req, _mode = entry
        dvx.append(float(rec["base_vx_body_mps"]) - vx_req)
        dvy.append(float(rec["base_vy_body_mps"]) - vy_req)
        dyaw.append(float(rec["base_wz_radps"]) - yaw_req)

    if not dvx:
        return {
            "interface_cmd_vx_rmse_mps": None,
            "interface_cmd_vy_rmse_mps": None,
            "interface_cmd_planar_rmse_mps": None,
            "interface_cmd_yaw_rmse_radps": None,
            "interface_samples": 0,
            "interface_missing_samples": missing,
        }

    rmse_x = math.sqrt(sum(v * v for v in dvx) / len(dvx))
    rmse_y = math.sqrt(sum(v * v for v in dvy) / len(dvy))
    rmse_planar = math.sqrt(sum(a * a + b * b for a, b in zip(dvx, dvy)) / len(dvx))
    rmse_yaw = math.sqrt(sum(v * v for v in dyaw) / len(dyaw))
    return {
        "interface_cmd_vx_rmse_mps": rmse_x,
        "interface_cmd_vy_rmse_mps": rmse_y,
        "interface_cmd_planar_rmse_mps": rmse_planar,
        "interface_cmd_yaw_rmse_radps": rmse_yaw,
        "interface_samples": len(dvx),
        "interface_missing_samples": missing,
    }


def run_one(
    scenario_cfg: Dict[str, Any],
    condition: ConditionSpec,
    variant: Variant,
    dwell_s: float,
    patched: bool,
) -> Dict[str, Any]:
    """Run one (scenario, condition) pair and return the summary dict.

    `patched=False` runs the completely unmodified harness (used once per
    scenario for the sanity check).  `patched=True` installs the quantising
    controller; condition "direct" uses an identity quantiser, which is the
    apples-to-apples baseline for Q and S.
    """
    quantiser = CommandQuantiser(
        vocabulary=variant.vocabulary,
        dwell_s=dwell_s,
        trailing_stand_mode=condition.trailing_stand_mode,
        tie_break=variant.tie_break,
        identity=condition.identity,
        total_duration_s=float(scenario_cfg.get("duration_s", 5.0)),
    )

    original_cls = harness.Go2ChampController
    if patched:
        harness.Go2ChampController = _make_patched_controller(quantiser)

    t0 = time.perf_counter()
    try:
        runner = _CapturingRunner()
        result = runner.run_scenario(
            scenario_cfg,
            output_dir=None,
            save_plots=False,
            gui=False,
            vision=False,
        )
    finally:
        harness.Go2ChampController = original_cls
    wall_s = time.perf_counter() - t0

    history = getattr(runner, "last_history", []) or []
    summary: Dict[str, Any] = {
        "scenario": scenario_cfg.get("name"),
        "condition": condition.label,
        "condition_id": {"direct": "D", "quantised": "Q",
                         "quantised+stand": "S",
                         "quantised+stand-once": "S-once"}.get(condition.label, condition.label),
        "variant": variant.name,
        "patched": patched,
        "wall_clock_s": wall_s,
    }
    # Flat copy of the harness metrics: the LaTeX table generator reads the
    # metric keys straight off the run dict.
    summary.update(result)
    summary["wall_clock_s"] = wall_s
    summary["patched"] = patched

    duty = result.get("contact_duty_factor", {}) or {}
    summary["contact_duty_factor_mean"] = (
        float(sum(duty.values()) / len(duty)) if duty else None
    )
    summary.update(_interface_tracking(history, quantiser))

    # Audit trail of the command stream the controller actually received.
    segs = quantiser.segments()
    summary["command_segments"] = segs
    summary["command_segment_count"] = len(segs)
    summary["delivered_command_modes"] = sorted({s["mode"] for s in segs})
    # Seconds during which the controller was handed a stationary stand
    # command.  In condition D this is the scenario's own stand time; in Q it
    # is the scenario's stand time plus the effect of holding a stand command
    # across a dwell block; in S it additionally contains the contract's
    # trailing-stand blocks.
    summary["stand_seconds_delivered"] = float(
        sum(
            (s["t_end_s"] - s["t_start_s"])
            for s in segs
            if s["mode"] == MotionMode.STAND.value
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def scenario_path(name: str) -> str:
    for candidate in (
        os.path.join(str(harness.SCENARIOS_DIR), f"{name}.json"),
        os.path.join(_ROOT, "experiments", "scenarios", f"{name}.json"),
        os.path.join(_ROOT, "scenarios", f"{name}.json"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"scenario '{name}' not found under {harness.SCENARIOS_DIR}")


def vocabulary_error_table(
    scenario_cfg: Dict[str, Any], vocabulary: Vocabulary, tie_break: str
) -> List[Dict[str, Any]]:
    """Quantisation error for every distinct command value in the scenario.

    `tie` marks the values that fall exactly between two vocabulary levels,
    i.e. the ones whose quantised value depends on the tie-breaking rule.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()
    for entry in scenario_cfg.get("commands", []) or []:
        key = (
            round(float(entry.get("vx", 0.0)), 6),
            round(float(entry.get("vy", 0.0)), 6),
            round(float(entry.get("yaw_rate", 0.0)), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        vx, vy, yaw = key
        qvx = quantise(vx, vocabulary.vx, tie_break)
        qvy = quantise(vy, vocabulary.vy, tie_break)
        qyaw = quantise(yaw, vocabulary.yaw, tie_break)

        def is_tie(value: float, grid: Sequence[float]) -> bool:
            ds = sorted(round(abs(value - c), 9) for c in grid)
            return len(ds) > 1 and abs(ds[0] - ds[1]) < TIE_EPS

        rows.append(
            {
                "raw": {"vx": vx, "vy": vy, "yaw_rate": yaw},
                "quantised": {"vx": qvx, "vy": qvy, "yaw_rate": qyaw},
                "abs_error": {
                    "vx": abs(qvx - vx),
                    "vy": abs(qvy - vy),
                    "yaw_rate": abs(qyaw - yaw),
                },
                "rel_error": {
                    "vx": (abs(qvx - vx) / abs(vx)) if vx else 0.0,
                    "vy": (abs(qvy - vy) / abs(vy)) if vy else 0.0,
                    "yaw_rate": (abs(qyaw - yaw) / abs(yaw)) if yaw else 0.0,
                },
                "tie": {
                    "vx": is_tie(vx, vocabulary.vx),
                    "vy": is_tie(vy, vocabulary.vy),
                    "yaw_rate": is_tie(yaw, vocabulary.yaw),
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------------

def _safe_rel(delta: float, base: float) -> Optional[float]:
    if base is None or base == 0.0 or not math.isfinite(base) or not math.isfinite(delta):
        return None
    return delta / base


def compute_deltas(runs: Dict[str, Dict[str, Any]], conditions: Sequence[str]) -> Dict[str, Any]:
    """Delta of every quantised condition relative to condition D."""
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, run in runs.items():
        scen = run.get("scenario", key)
        by_scenario.setdefault(scen, {})[run.get("condition", "")] = run

    deltas: Dict[str, Any] = {}
    for scen, conds in by_scenario.items():
        base = conds.get("direct")
        if base is None:
            continue
        entry: Dict[str, Any] = {}
        for cond in conditions:
            if cond == "direct":
                continue
            cur = conds.get(cond)
            if cur is None:
                continue
            metric_deltas: Dict[str, Any] = {}
            for metric in DELTA_METRICS:
                b = base.get(metric)
                c = cur.get(metric)
                if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
                    continue
                b = float(b)
                c = float(c)
                if not (math.isfinite(b) and math.isfinite(c)):
                    continue
                d = c - b
                rel = _safe_rel(d, b)
                metric_deltas[metric] = {
                    "direct": b,
                    "condition": c,
                    "delta": d,
                    "rel": rel,
                    "rel_pct": (None if rel is None else 100.0 * rel),
                }
            gate = {
                "passed_direct": bool(base.get("passed")),
                "passed_condition": bool(cur.get("passed")),
                "failure_reasons_direct": list(base.get("failure_reasons", []) or []),
                "failure_reasons_condition": list(cur.get("failure_reasons", []) or []),
                "gate_broken": bool(base.get("passed")) and not bool(cur.get("passed")),
            }
            entry[cond] = {"metrics": metric_deltas, "gate": gate}
        deltas[scen] = entry
    return deltas


def compute_marginal_effects(runs: Dict[str, Dict[str, Any]], conditions: Sequence[str]) -> Dict[str, Any]:
    """Marginal effect of the trailing stand: condition S minus condition Q.

    S is Q plus the trailing stand, so S - Q isolates the cost of the
    standstill from the cost of the quantised command stream itself.
    """
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, run in runs.items():
        scen = run.get("scenario", key)
        by_scenario.setdefault(scen, {})[run.get("condition", "")] = run

    out: Dict[str, Any] = {}
    for scen, conds in by_scenario.items():
        q = conds.get("quantised")
        if q is None:
            continue
        entry: Dict[str, Any] = {}
        for cond in conditions:
            if cond in ("direct", "quantised"):
                continue
            s = conds.get(cond)
            if s is None:
                continue
            metric_deltas: Dict[str, Any] = {}
            for metric in DELTA_METRICS:
                b = q.get(metric)
                c = s.get(metric)
                if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
                    continue
                b = float(b)
                c = float(c)
                if not (math.isfinite(b) and math.isfinite(c)):
                    continue
                d = c - b
                rel = _safe_rel(d, b)
                metric_deltas[metric] = {
                    "quantised": b,
                    "condition": c,
                    "delta": d,
                    "rel": rel,
                    "rel_pct": (None if rel is None else 100.0 * rel),
                }
            entry[cond] = {
                "baseline": "quantised",
                "metrics": metric_deltas,
                "gate": {
                    "passed_quantised": bool(q.get("passed")),
                    "passed_condition": bool(s.get("passed")),
                    "gate_broken": bool(q.get("passed")) and not bool(s.get("passed")),
                },
            }
        if entry:
            out[scen] = entry
    return out


def find_gate_failures(runs: Dict[str, Dict[str, Any]], conditions: Sequence[str]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, run in runs.items():
        scen = run.get("scenario", key)
        by_scenario.setdefault(scen, {})[run.get("condition", "")] = run

    broken: List[Dict[str, Any]] = []
    for scen, conds in by_scenario.items():
        base = conds.get("direct")
        if base is None or not base.get("passed"):
            continue
        for cond in conditions:
            if cond == "direct":
                continue
            cur = conds.get(cond)
            if cur is None or cur.get("passed"):
                continue
            broken.append(
                {
                    "scenario": scen,
                    "condition": cond,
                    "passed_direct": True,
                    "passed_condition": False,
                    "failure_reasons": list(cur.get("failure_reasons", []) or []),
                    "failure_reasons_direct": list(base.get("failure_reasons", []) or []),
                }
            )
    return broken


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _f(value: Any, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "--"
    return fmt.format(f)


def print_condition_table(
    runs: Dict[str, Dict[str, Any]], scenarios: Sequence[str], conditions: Sequence[str]
) -> None:
    header = (
        f"{'scenario':<20} {'cond':<22} {'pass':<5} {'fwd_m':>7} {'path_m':>8} "
        f"{'vX_rmse':>8} {'wZ_rmse':>8} {'jnt_rmse':>8} {'h_m':>6} "
        f"{'roll_d':>7} {'pit_d':>7} {'CoT_g':>7} {'CoT_n':>7} {'rec':>6} {'duty':>5} {'stand_s':>7}"
    )
    print(header)
    print("-" * len(header))
    for scen in scenarios:
        for cond in conditions:
            run = runs.get(f"{scen}::{cond}")
            if run is None:
                continue
            print(
                f"{scen:<20} {cond:<22} {'PASS' if run.get('passed') else 'FAIL':<5} "
                f"{_f(run.get('forward_displacement_m'), '{:.2f}'):>7} "
                f"{_f(run.get('total_path_length_m'), '{:.2f}'):>8} "
                f"{_f(run.get('velocity_tracking_rmse_mps')):>8} "
                f"{_f(run.get('yaw_rate_tracking_rmse_radps')):>8} "
                f"{_f(run.get('joint_tracking_rmse_rad')):>8} "
                f"{_f(run.get('base_height_mean_m')):>6} "
                f"{_f(math.degrees(float(run.get('base_roll_peak_abs_rad', 0.0))), '{:.1f}'):>7} "
                f"{_f(math.degrees(float(run.get('base_pitch_peak_abs_rad', 0.0))), '{:.1f}'):>7} "
                f"{_f(run.get('estimated_cost_of_transport'), '{:.2f}'):>7} "
                f"{_f(run.get('estimated_cost_of_transport_signed'), '{:.2f}'):>7} "
                f"{_f(100.0 * float(run.get('recovery_fraction') or 0.0), '{:.0f}'):>6} "
                f"{_f(run.get('contact_duty_factor_mean'), '{:.2f}'):>5} "
                f"{_f(run.get('stand_seconds_delivered'), '{:.1f}'):>7}"
            )
    print("-" * len(header))


def print_delta_table(deltas: Dict[str, Any], scenarios: Sequence[str], conditions: Sequence[str]) -> None:
    header = (
        f"{'scenario':<20} {'cond':<22} {'d_path_m':>9} {'d_path_%':>9} "
        f"{'d_vX_rmse':>9} {'d_vX_%':>9} {'d_iface_vX':>11} {'d_iface_%':>11} "
        f"{'d_CoTnet':>9} {'d_CoTnet_%':>11} {'d_yaw_rmse':>10}"
    )
    print(header)
    print("-" * len(header))
    for scen in scenarios:
        entry = deltas.get(scen, {})
        for cond in conditions:
            if cond == "direct":
                continue
            block = entry.get(cond)
            if not block:
                continue
            m = block["metrics"]

            def d(key: str) -> str:
                row = m.get(key)
                if row is None:
                    return "--"
                return _f(row.get("delta"))

            def p(key: str) -> str:
                row = m.get(key)
                if row is None or row.get("rel_pct") is None:
                    return "--"
                return _f(row["rel_pct"], "{:+.1f}")

            print(
                f"{scen:<20} {cond:<22} {d('total_path_length_m'):>9} "
                f"{p('total_path_length_m'):>9} "
                f"{d('velocity_tracking_rmse_mps'):>9} {p('velocity_tracking_rmse_mps'):>9} "
                f"{d('interface_cmd_vx_rmse_mps'):>11} {p('interface_cmd_vx_rmse_mps'):>11} "
                f"{d('estimated_cost_of_transport_signed'):>9} "
                f"{p('estimated_cost_of_transport_signed'):>11} "
                f"{d('yaw_rate_tracking_rmse_radps'):>10}"
            )
    print("-" * len(header))


def build_headline(
    runs: Dict[str, Dict[str, Any]], scenarios: Sequence[str], conditions: Sequence[str]
) -> Dict[str, Any]:
    """Aggregate headline numbers: mean relative degradation across scenarios."""
    out: Dict[str, Any] = {}
    for cond in conditions:
        if cond == "direct":
            continue
        rel_path: List[float] = []
        rel_vx: List[float] = []
        rel_planar: List[float] = []
        d_cot: List[float] = []
        d_cot_rel: List[float] = []
        n = 0
        for scen in scenarios:
            base = runs.get(f"{scen}::direct")
            cur = runs.get(f"{scen}::{cond}")
            if base is None or cur is None:
                continue
            n += 1
            b_path = float(base.get("total_path_length_m") or 0.0)
            c_path = float(cur.get("total_path_length_m") or 0.0)
            if b_path > 0:
                rel_path.append(100.0 * (c_path - b_path) / b_path)
            b_vx = float(base.get("velocity_tracking_rmse_mps") or 0.0)
            c_vx = float(cur.get("velocity_tracking_rmse_mps") or 0.0)
            if b_vx > 0:
                rel_vx.append(100.0 * (c_vx - b_vx) / b_vx)
            b_pl = float(base.get("interface_cmd_planar_rmse_mps") or 0.0)
            c_pl = float(cur.get("interface_cmd_planar_rmse_mps") or 0.0)
            if b_pl > 0:
                rel_planar.append(100.0 * (c_pl - b_pl) / b_pl)
            b_cot = float(base.get("estimated_cost_of_transport_signed") or 0.0)
            c_cot = float(cur.get("estimated_cost_of_transport_signed") or 0.0)
            d_cot.append(c_cot - b_cot)
            if b_cot != 0.0:
                d_cot_rel.append(100.0 * (c_cot - b_cot) / b_cot)
        out[cond] = {
            "n_scenarios": n,
            "mean_path_length_rel_pct": (sum(rel_path) / len(rel_path)) if rel_path else None,
            "median_path_length_rel_pct": (sorted(rel_path)[len(rel_path) // 2]) if rel_path else None,
            "mean_velocity_rmse_rel_pct": (sum(rel_vx) / len(rel_vx)) if rel_vx else None,
            "median_velocity_rmse_rel_pct": (sorted(rel_vx)[len(rel_vx) // 2]) if rel_vx else None,
            "mean_interface_planar_rmse_rel_pct": (
                (sum(rel_planar) / len(rel_planar)) if rel_planar else None
            ),
            "mean_delta_cot_net": (sum(d_cot) / len(d_cot)) if d_cot else None,
            "mean_delta_cot_net_pct": (sum(d_cot_rel) / len(d_cot_rel)) if d_cot_rel else None,
            "per_scenario_delta_cot_net": d_cot,
        }
    return out


def crosscheck_published_benchmark(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare condition D against the independently published benchmark.

    `report/figures/benchmark_results.json` holds the metrics of the fresh
    full-suite benchmark run (report/notes/07_benchmark_results.md).  If
    condition D reproduces it exactly, then D is demonstrably the unmodified
    baseline and the deltas are attributable to the language interface alone.
    The check is skipped (not failed) when the published file is absent.
    """
    pub_path = os.path.join(_ROOT, "report", "figures", "benchmark_results.json")
    out: Dict[str, Any] = {
        "source": str(pub_path),
        "description": (
            "Condition D (identity quantiser through the patched harness) versus "
            "the metrics of the independent full-suite benchmark run published in "
            "report/figures/benchmark_results.json."
        ),
        "available": False,
    }
    if not os.path.exists(pub_path):
        out["note"] = "published benchmark file not present; cross-check skipped"
        return out
    try:
        pub = json.loads(open(pub_path, encoding="utf-8").read())
    except Exception as exc:  # a broken reference file is not a result
        out["note"] = f"could not parse published benchmark: {exc}"
        return out

    per_scenario: Dict[str, Any] = {}
    worst = 0.0
    worst_key = ""
    for key, run in runs.items():
        if not key.endswith("::direct"):
            continue
        scen = run.get("scenario")
        p = pub.get(scen) if isinstance(pub, dict) else None
        if not isinstance(p, dict):
            continue
        diffs: Dict[str, float] = {}
        for metric in HARNESS_METRIC_KEYS:
            pv = p.get(metric)
            dv = run.get(metric)
            if not isinstance(pv, (int, float)) or not isinstance(dv, (int, float)):
                continue
            dd = abs(float(pv) - float(dv))
            diffs[metric] = dd
            if dd > worst:
                worst = dd
                worst_key = f"{scen}.{metric}"
        per_scenario[scen] = {
            "max_abs_difference": max(diffs.values()) if diffs else 0.0,
            "per_metric_abs_difference": diffs,
            "bitwise_identical": all(v == 0.0 for v in diffs.values()),
        }
    out.update(
        {
            "available": True,
            "per_scenario": per_scenario,
            "max_abs_difference": worst,
            "worst_metric": worst_key,
            "verdict": (
                "condition D reproduces the published benchmark exactly"
                if worst == 0.0
                else "condition D differs from the published benchmark"
            ),
        }
    )
    return out


# ---------------------------------------------------------------------------
# One variant
# ---------------------------------------------------------------------------

def run_variant(
    variant: Variant,
    loaded: Dict[str, Dict[str, Any]],
    dwell_s: float,
) -> Dict[str, Any]:
    """Run every condition of one variant over every scenario."""
    condition_labels = [c.label for c in variant.conditions]
    runs: Dict[str, Dict[str, Any]] = {}
    unmodified: Dict[str, Dict[str, Any]] = {}

    for name, cfg in loaded.items():
        # 1. the completely unmodified harness, once, as the reference
        unmodified[name] = run_one(cfg, CONDITION_DIRECT, variant, dwell_s, patched=False)
        # 2. every condition through the patched path (identity quantiser for D)
        for cond in variant.conditions:
            run = run_one(cfg, cond, variant, dwell_s, patched=True)
            runs[f"{name}::{cond.label}"] = run

    deltas = compute_deltas(runs, condition_labels)
    marginal = compute_marginal_effects(runs, condition_labels)
    gate_failures = find_gate_failures(runs, condition_labels)

    # Inject the deltas into the run dicts as well, so a consumer can read
    # `run["delta_total_path_length_m"]` straight off the run without walking
    # the `deltas` tree.
    for scen, entry in deltas.items():
        for cond, block in entry.items():
            run = runs.get(f"{scen}::{cond}")
            if run is None:
                continue
            run["delta_vs_direct"] = {
                metric: {
                    "abs": v["delta"],
                    "rel": v["rel"],
                    "rel_pct": v["rel_pct"],
                    "direct": v["direct"],
                    "condition": v["condition"],
                }
                for metric, v in block["metrics"].items()
            }
            for metric, v in block["metrics"].items():
                run[f"delta_{metric}"] = v["delta"]
                run[f"delta_{metric}_pct"] = v["rel_pct"]
            run["gate_broken_vs_direct"] = block["gate"]["gate_broken"]

    # Sanity check: patched identity quantiser vs the unmodified harness.
    sanity: Dict[str, Any] = {}
    max_abs_diff = 0.0
    worst_key = ""
    all_equal = True
    for name in loaded:
        base = unmodified[name]
        patched = runs[f"{name}::direct"]
        diffs: Dict[str, Any] = {}
        for key in HARNESS_METRIC_KEYS:
            value = base.get(key)
            pv = patched.get(key)
            if value is None or pv is None:
                if value != pv:
                    all_equal = False
                    diffs[key] = "mismatch"
                continue
            if not isinstance(value, (int, float)) or not isinstance(pv, (int, float)):
                if value != pv:
                    all_equal = False
                    diffs[key] = "mismatch"
                continue
            d = abs(float(pv) - float(value))
            diffs[key] = d
            if d > max_abs_diff:
                max_abs_diff = d
                worst_key = f"{name}.{key}"
            if d != 0.0:
                all_equal = False
        duty_diffs = {}
        duty_a = base.get("contact_duty_factor", {}) or {}
        duty_b = patched.get("contact_duty_factor", {}) or {}
        for leg in sorted(set(duty_a) | set(duty_b)):
            va = float(duty_a.get(leg, 0.0))
            vb = float(duty_b.get(leg, 0.0))
            duty_diffs[leg] = abs(vb - va)
            if duty_diffs[leg] > max_abs_diff:
                max_abs_diff = duty_diffs[leg]
                worst_key = f"{name}.contact_duty_factor.{leg}"
            if duty_diffs[leg] != 0.0:
                all_equal = False
        sanity[name] = {
            "max_abs_metric_difference": max(
                [v for v in diffs.values() if isinstance(v, (int, float))] or [0.0]
            ),
            "per_metric_abs_difference": diffs,
            "contact_duty_factor_abs_difference": duty_diffs,
            "bitwise_identical": all(
                (v == 0.0 if isinstance(v, (int, float)) else False) for v in diffs.values()
            )
            and all(v == 0.0 for v in duty_diffs.values()),
        }
    sanity["_summary"] = {
        "description": (
            "Condition D is produced by the patched harness with an identity "
            "quantiser (the command is passed through untouched). It is compared "
            "against one run of the completely unmodified harness. Any non-zero "
            "difference would mean the monkeypatch itself perturbs the baseline."
        ),
        "max_abs_difference_over_all_scenarios_and_metrics": max_abs_diff,
        "worst_metric": worst_key,
        "all_differences_exactly_zero": all_equal,
        "verdict": (
            "PASS -- the monkeypatch is numerically neutral; condition D is the "
            "unmodified harness."
            if all_equal
            else "FAIL -- the monkeypatch perturbs the baseline."
        ),
    }

    return {
        "variant": variant.name,
        "description": variant.description,
        "vocabulary": variant.vocabulary.to_dict(),
        "tie_break": variant.tie_break,
        "conditions": {c.label: c.description for c in variant.conditions},
        "runs": runs,
        "deltas": deltas,
        "marginal_effects": marginal,
        "gate_failures": gate_failures,
        "headline": build_headline(runs, list(loaded.keys()), condition_labels),
        "sanity_check": sanity,
        "unmodified_harness_runs": {f"{n}::direct": r for n, r in unmodified.items()},
        "vocabulary_error_per_scenario": {
            n: vocabulary_error_table(cfg, variant.vocabulary, variant.tie_break)
            for n, cfg in loaded.items()
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablate the natural-language abstraction over the benchmark scenarios."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="comma-separated scenario names",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=1.0,
        help="command dwell time in seconds (default: 1.0, the contract's typical duration)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output JSON path (default: report/figures/language_ablation.json)",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="skip the sensitivity variants (primary configuration only)",
    )
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out_path = args.out or os.path.join(_ROOT, "report", "figures", "language_ablation.json")
    dwell_s = float(args.dwell)

    bounds = SafetyBounds()

    variants: List[Variant] = [PRIMARY_VARIANT]
    if not args.no_sensitivity:
        variants.extend(SENSITIVITY_VARIANTS)

    print("=" * 104)
    print("LANGUAGE ABLATION  --  what does the natural-language abstraction cost?")
    print("=" * 104)
    print(
        f"conditions      : D = direct (continuous reference) | Q = quantised (held {dwell_s:.1f} s) | "
        f"S = quantised + trailing stand"
    )
    print(
        f"shipped bounds  : vx [{bounds.min_vx:+.2f}, {bounds.max_vx:+.2f}] m/s | "
        f"vy [{-bounds.max_vy:+.2f}, {bounds.max_vy:+.2f}] m/s | "
        f"yaw_rate [{-bounds.max_yaw_rate:+.2f}, {bounds.max_yaw_rate:+.2f}] rad/s"
    )
    for variant in variants:
        print(
            f"variant         : {variant.name:<22} vocab={variant.vocabulary.name} "
            f"ties={variant.tie_break} conds={[c.label for c in variant.conditions]}"
        )
    print(f"scenarios       : {', '.join(scenarios)}")
    print("-" * 104)

    loaded: Dict[str, Dict[str, Any]] = {}
    for name in scenarios:
        try:
            cfg = harness.load_scenario(scenario_path(name))
        except Exception as exc:  # a missing scenario is a RESULT, not a crash
            print(f"[skip] scenario '{name}' could not be loaded: {exc}")
            continue
        loaded[name] = cfg

    if not loaded:
        print("[error] no scenario could be loaded")
        return 2

    variant_results: Dict[str, Any] = {}
    for variant in variants:
        block = run_variant(variant, loaded, dwell_s)
        variant_results[variant.name] = block

        condition_labels = [c.label for c in variant.conditions]
        print()
        print(f"### variant: {variant.name}  ({variant.description})")
        print_condition_table(block["runs"], list(loaded.keys()), condition_labels)
        print()
        print_delta_table(block["deltas"], list(loaded.keys()), condition_labels)
        print()
        sanity_summary = block["sanity_check"]["_summary"]
        print(
            f"sanity check (identity quantiser vs unmodified harness): "
            f"max |diff| = {sanity_summary['max_abs_difference_over_all_scenarios_and_metrics']:.3e}  -> "
            f"{'IDENTICAL' if sanity_summary['all_differences_exactly_zero'] else 'DIFFERENT (investigate!)'}"
        )
        if block["gate_failures"]:
            print("!! GATE FAILURE introduced by the language interface:")
            for gf in block["gate_failures"]:
                print(f"   {gf['scenario']} under '{gf['condition']}': {gf['failure_reasons']}")
        else:
            print("no declared gate that passes in D is broken by the quantised conditions")
        print("-" * 104)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    primary = variant_results[PRIMARY_VARIANT.name]
    primary_conditions = [c.label for c in PRIMARY_VARIANT.conditions]

    payload: Dict[str, Any] = {
        "meta": {
            "experiment": "language_ablation",
            "research_question": (
                "What does the natural-language abstraction cost the quadruped "
                "controller, in tracking, in distance covered, and in energy?"
            ),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(_ROOT),
            "command": "python experiments/language_ablation.py",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "harness": "src/evaluate_champ_trajectories.py (ScenarioRunner.run_scenario)",
            "controller": "src/go2_champ_control.py (Go2ChampController.step)",
            "llm_contract": "src/go2_llm_commands.py (parse_and_validate_response)",
            "monkeypatch": (
                "harness.Go2ChampController is replaced by a subclass whose step() "
                "quantises the MotionCommand BEFORE calling super().step(); the "
                "telemetry is never modified."
            ),
            "quantiser_applied_to": "command (MotionCommand), not telemetry",
            "quantiser_operator": "q(v) = argmin_{c in V} |v - c|",
            "dwell_s": dwell_s,
            "conditions": primary["conditions"],
            "safety_bounds_shipped": {
                "min_vx": bounds.min_vx,
                "max_vx": bounds.max_vx,
                "max_vy": bounds.max_vy,
                "max_yaw_rate": bounds.max_yaw_rate,
                "max_linear_accel": bounds.max_linear_accel,
                "max_yaw_accel": bounds.max_yaw_accel,
                "source": "src/go2_champ_control.py SafetyBounds",
            },
            "scenarios": list(loaded.keys()),
            "metric_units": {
                "forward_displacement_m": "m",
                "lateral_displacement_m": "m",
                "total_path_length_m": "m",
                "velocity_tracking_rmse_mps": "m/s",
                "yaw_rate_tracking_rmse_radps": "rad/s",
                "joint_tracking_rmse_rad": "rad",
                "base_height_mean_m": "m",
                "base_roll_peak_abs_rad": "rad",
                "base_pitch_peak_abs_rad": "rad",
                "estimated_cost_of_transport": "dimensionless (gross)",
                "estimated_cost_of_transport_signed": "dimensionless (net)",
                "estimated_energy_j": "J",
                "estimated_energy_signed_j": "J",
                "recovery_fraction": "dimensionless fraction",
                "contact_duty_factor_mean": "fraction of the run each foot is in contact",
                "stand_seconds_delivered": "seconds the controller was handed a stand command",
                "interface_cmd_vx_rmse_mps": "m/s (measured vs the command the interface delivered)",
                "interface_cmd_vy_rmse_mps": "m/s",
                "interface_cmd_planar_rmse_mps": "m/s",
                "interface_cmd_yaw_rmse_radps": "rad/s",
            },
            "notes": [
                "velocity_tracking_rmse_mps (from the harness) compares the measured "
                "base velocity against the POST-slew-limiter command that the "
                "controller applied; interface_cmd_*_rmse compare against the "
                "command the interface actually delivered (pre-slew).",
                "The slew limiter is identical in every condition and is therefore "
                "NOT a differentiator.",
                "For waypoint scenarios the controller derives its velocity from the "
                "closed-loop navigator and ignores the typed velocity triple, so "
                "condition Q coincides with D there; only the stand blocks of "
                "condition S bind.",
                "The trailing-stand invariant lives in "
                "src/go2_llm_commands.py::parse_and_validate_response (the final "
                "'safety convention' block appending MotionCommand(mode=STAND, "
                "duration=1.0)).",
                "runs / deltas / gate_failures / headline / sanity_check at the top "
                "level belong to the 'primary' variant; 'variants' holds every "
                "variant including the primary.",
            ],
        },
        # Primary variant, flattened onto the top level so that
        # report/make_tables.py::tab_language_ablation can consume it directly.
        "variant": PRIMARY_VARIANT.name,
        "vocabulary": primary["vocabulary"],
        "tie_break": primary["tie_break"],
        "runs": primary["runs"],
        "deltas": primary["deltas"],
        "marginal_effects": primary["marginal_effects"],
        "gate_failures": primary["gate_failures"],
        "headline": primary["headline"],
        "sanity_check": primary["sanity_check"],
        "vocabulary_error_per_scenario": primary["vocabulary_error_per_scenario"],
        "unmodified_harness_runs": primary["unmodified_harness_runs"],
        "published_benchmark_crosscheck": crosscheck_published_benchmark(primary["runs"]),
        "variants": {
            name: {
                "variant": block["variant"],
                "description": block["description"],
                "vocabulary": block["vocabulary"],
                "tie_break": block["tie_break"],
                "conditions": block["conditions"],
                "runs": block["runs"],
                "deltas": block["deltas"],
                "marginal_effects": block["marginal_effects"],
                "gate_failures": block["gate_failures"],
                "headline": block["headline"],
                "sanity_check": block["sanity_check"],
                "vocabulary_error_per_scenario": block["vocabulary_error_per_scenario"],
                "unmodified_harness_runs": block["unmodified_harness_runs"],
            }
            for name, block in variant_results.items()
        },
    }
    del primary_conditions

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False, default=str)

    print()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # internal error -> non-zero exit
        import traceback

        traceback.print_exc()
        print(f"[internal error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
