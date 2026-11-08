"""Headless evaluation harness and benchmark runner for Go2 CHAMP locomotion scenarios.

Executes deterministic locomotion scenarios (stand hold, speed step, stop & hold,
turn in place, S-curve, waypoint navigation), logs full telemetry, computes
rigorous biomechanical/tracking metrics, generates diagnostic figures, and
produces benchmark acceptance reports.

Usage:
    # Run all benchmark scenarios headlessly
    python evaluate_champ_trajectories.py --scenario all

    # Run a specific scenario
    python evaluate_champ_trajectories.py --scenario stand_hold
    python evaluate_champ_trajectories.py --scenario turn_in_place
    python evaluate_champ_trajectories.py --scenario s_curve
    python evaluate_champ_trajectories.py --scenario waypoint_square
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from go2_scene_obstacles import (
    ObstacleSpec,
    ObstacleSpecError,
    build_model_with_obstacles,
    line_lateral_deviation,
    obstacle_contact,
    obstacle_geom_distance,
    parse_obstacles,
    robot_collidable_geom_ids,
)
from go2_champ_control import (
    DEFAULT_NEUTRAL_Q,
    GRAVITY,
    HIP_OFFSET,
    HIP_POSITIONS,
    IS_LEFT,
    JOINT_NAMES,
    LEG_NAMES,
    NOMINAL_BODY_HEIGHT,
    ChampTrajectoryPlanner,
    CommandSlewLimiter,
    FootholdPlanner,
    GaitConfig,
    Go2ChampController,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    TRAJECTORY_PATH_TYPES,
    TrajectorySpec,
    Waypoint,
    WaypointNavigator,
    configure_model,
    initialize_simulation_standing,
    neutral_joint_targets,
    quaternion_to_euler_xyz,
    solve_ik,
)
from go2_llm_planner import (  # noqa: E402
    MAX_SKILLS_PER_PLAN,
    SKILL_REGISTRY,
    expand_skill,
    skill_budget,
)


_src_dir = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = (
    os.path.dirname(_src_dir)
    if os.path.exists(os.path.join(os.path.dirname(_src_dir), "unitree_go2"))
    else _src_dir
)
ROOT_DIR = _REPO_ROOT
MODEL_PATH = os.path.join(ROOT_DIR, "unitree_go2", "go2_description.xml")
SCENARIOS_DIR = (
    os.path.join(ROOT_DIR, "experiments", "scenarios")
    if os.path.exists(os.path.join(ROOT_DIR, "experiments", "scenarios"))
    else os.path.join(ROOT_DIR, "scenarios")
)
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

# Scenario schema -------------------------------------------------------------------
#: Path types accepted by the parametric trajectory generator. Imported from the
#: controller so the validator can never accept a path the tracker would reject at
#: runtime.
# (TRAJECTORY_PATH_TYPES comes from go2_champ_control.)
#: Per-command modes accepted by the runner's command schedule ("trot" and "velocity"
#: both mean a velocity command; "stand" is a static hold).
COMMAND_MODES = ("stand", "trot", "velocity")
#: Modes a *scenario* may select. ``MotionMode.EMERGENCY_STAND`` is deliberately
#: absent: it is the controller's own safety state (entered on a roll/pitch fault
#: and exited only by recovery), not a motion request. See the ``mode`` check in
#: validate_scenario for why admitting it would be a silent mis-run.
SCENARIO_MODES = (MotionMode.STAND.value, MotionMode.VELOCITY.value, MotionMode.WAYPOINT.value, MotionMode.TRAJECTORY.value, "plan")
#: Metrics the pass/fail gate actually evaluates. An unrecognised key in
#: `expected_metrics` silently disables a check, so the validator rejects it.
KNOWN_EXPECTED_METRICS = (
    "min_base_height_m",
    "max_roll_rad",
    "max_pitch_rad",
    "min_forward_displacement_m",
    "max_forward_displacement_m",
    "min_path_length_m",
    "max_tracking_error_m",
    "min_obstacle_clearance_m",
    "max_obstacle_clearance_m",
    "min_obstacle_engagement_m",
    "min_lateral_deviation_m",
    "min_obstacle_detections",
    "min_plan_skills_completed",
    "min_plan_waypoints_reached",
    "max_final_goal_distance_m",
)
#: Every top-level key the runner understands. The runner reads each field with
#: ``.get()`` and a default, so an unrecognised top-level key is silently ignored --
#: a config carrying ``"trajetcory"`` (typo) validates and then runs as an ordinary
#: command-mode scenario, with the declared trajectory simply never executing. This
#: mirrors KNOWN_EXPECTED_METRICS, which already closes the same hole one level down.
#: ``description`` and ``notes`` are documentation-only: nothing in the harness reads
#: either one, but every shipped scenario carries ``description`` and four carry
#: ``notes``, so they are accepted rather than rejected.
KNOWN_TOP_LEVEL_KEYS = (
    "name",
    "duration_s",
    "mode",
    "commands",
    "trajectory",
    "waypoints",
    "plan",
    "obstacles",
    "vision",
    "expected_metrics",
    "description",
    "notes",
)
#: Workspace bound for waypoint coordinates, matching the LLM command schema.
MAX_WAYPOINT_COORD_M = 10.0
#: Tolerance for schedule arithmetic. Command windows are compared across the
#: float round-trip through JSON, so exact equality is not a safe identity test.
_SCHEDULE_TOL_S = 1e-9

#: How many ``logs/benchmark_suite_*`` runs to keep on disk. Each suite writes one
#: directory per run (plots, CSV, JSON, markdown), so an unbounded history grows
#: without limit. Older suites are pruned by ``prune_old_run_dirs`` at the start of
#: a run; nothing outside the passed directory is ever touched.
KEEP_RECENT_RUNS = 10


class ScenarioValidationError(ValueError):
    """Raised when a scenario file is malformed or fails schema/range validation."""


def _find_non_finite(values: Dict[str, Any], prefix: str = "") -> List[str]:
    """Collect the names of every non-finite (NaN/Inf) numeric value in a mapping.

    The pass/fail gate is built from ordered comparisons, and both ``nan < x`` and
    ``nan > x`` are ``False``, so a NaN-poisoned run would satisfy every check, record
    no failure reason and be reported as a PASS. Walking the recorded telemetry
    channels and the computed metrics lets the gate fail such a run explicitly, naming
    every offending quantity, instead of bolting a guard onto a single comparison.
    """
    offending: List[str] = []
    for key, value in values.items():
        label = f"{prefix}{key}"
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, (int, float)):
            try:
                checked = float(value)
            except (OverflowError, ValueError):
                # An out-of-range magnitude is worse than NaN, not benign:
                # record it so the gate fails rather than silently passing.
                offending.append(label)
                continue
            if not math.isfinite(checked):
                offending.append(label)
        elif isinstance(value, np.ndarray):
            if value.size and not np.all(np.isfinite(value)):
                offending.append(label)
        elif isinstance(value, dict):
            offending.extend(_find_non_finite(value, prefix=f"{label}."))
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                offending.extend(_find_non_finite({f"{label}[{idx}]": item}))
    return offending


def validate_scenario(cfg: Any, source: Any = "scenario") -> Dict[str, Any]:
    """Validate a parsed scenario configuration, returning it unchanged on success.

    Raises:
        ScenarioValidationError: naming `source` and the offending field, so a bad
            scenario fails with an actionable message instead of an uncaught
            traceback (or, worse, an out-of-range value flowing into the runner).
    """
    if not isinstance(cfg, dict):
        raise ScenarioValidationError(
            f"{source}: top level must be a JSON object, got {type(cfg).__name__}"
        )

    bounds = SafetyBounds()

    def fail(field: str, message: str):
        raise ScenarioValidationError(f"{source}: field '{field}' {message}")

    def get_number(
        container: Dict[str, Any],
        field: str,
        label: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        exclusive: bool = False,
    ) -> float:
        if field not in container:
            fail(label, "is required but missing")
        value = container[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(label, f"must be a number, got {type(value).__name__}")
        try:
            value = float(value)
        except (OverflowError, ValueError):
            # JSON parses a 400-digit integer literal as a Python int whose
            # magnitude exceeds float range; float() raises OverflowError
            # before the finite check below, so catch it here and report it
            # as a validation failure rather than an uncaught crash.
            fail(label, "is outside the representable numeric range")
        if not math.isfinite(value):
            fail(label, f"must be finite, got {value!r}")
        if minimum is not None and (value <= minimum if exclusive else value < minimum):
            fail(label, f"must be {'>' if exclusive else '>='} {minimum}, got {value}")
        if maximum is not None and value > maximum:
            fail(label, f"must be <= {maximum}, got {value}")
        return value

    def get_str(container: Dict[str, Any], field: str, label: str) -> str:
        if field not in container:
            fail(label, "is required but missing")
        value = container[field]
        if not isinstance(value, str) or not value.strip():
            fail(label, f"must be a non-empty string, got {value!r}")
        return value

    def validate_commands(entries: Any, label: str = "commands", total_duration: Optional[float] = None):
        if not isinstance(entries, list) or not entries:
            fail(label, f"must be a non-empty list, got {type(entries).__name__}")
        spans: List[Tuple[float, float]] = []
        for idx, entry in enumerate(entries):
            base = f"{label}[{idx}]"
            if not isinstance(entry, dict):
                fail(base, f"must be an object, got {type(entry).__name__}")
            for field, kwargs in (
                ("start_time_s", {"minimum": 0.0, "maximum": bounds.max_duration}),
                ("duration_s", {"minimum": 0.0, "maximum": bounds.max_duration, "exclusive": True}),
                ("vx", {"minimum": bounds.min_vx, "maximum": bounds.max_vx}),
                ("vy", {"minimum": -bounds.max_vy, "maximum": bounds.max_vy}),
                ("yaw_rate", {"minimum": -bounds.max_yaw_rate, "maximum": bounds.max_yaw_rate}),
            ):
                if field in entry:
                    get_number(entry, field, f"{base}.{field}", **kwargs)
            if "mode" in entry:
                cmd_mode = entry["mode"]
                if not isinstance(cmd_mode, str) or cmd_mode.lower() not in COMMAND_MODES:
                    fail(f"{base}.mode", f"must be one of {list(COMMAND_MODES)}, got {cmd_mode!r}")
            # Schedule spans, for the coverage/overlap check below. Fields the entry
            # omits take the runner's own defaults (see run_scenario): a command
            # with no start begins at t=0 and with no explicit duration runs to
            # the end of the scenario.  When the caller supplied no total
            # duration there is no end-of-scenario to default to, so a missing
            # duration is a validation failure rather than a float(None) crash.
            start = float(entry.get("start_time_s", 0.0))
            raw_duration = entry.get("duration_s", total_duration)
            if raw_duration is None:
                fail(
                    f"{base}.duration_s",
                    "is required when the scenario duration is not available to "
                    "default it, and must be a number",
                )
            duration = float(raw_duration)
            spans.append((start, start + duration))

        if total_duration is None:
            return

        # Overlap: two windows covering the same instant makes the runner's
        # first-match rule (list order) decide the command, so the effective
        # schedule is not the one the file spells out.
        ordered = sorted(range(len(spans)), key=lambda i: (spans[i][0], i))
        prev_end: Optional[float] = None
        for i in ordered:
            start, end = spans[i]
            if prev_end is not None and start < prev_end - _SCHEDULE_TOL_S:
                fail(
                    f"{label}[{i}]",
                    f"overlaps an earlier command: starts at {start}s but the preceding "
                    f"command in time already runs to {prev_end}s; the runner applies the "
                    f"first match in list order, so the overlapping entry may never take effect",
                )
            prev_end = end if prev_end is None else max(prev_end, end)

        # Tail coverage: with nothing matching, run_scenario falls back to the
        # *last* command in the list, so any gap silently replays that command
        # for the rest of the run instead of standing still.  The end of the
        # schedule is the maximum end over all commands, not the end of the
        # last entry in list order: the list need not be sorted in time.
        latest_end = max(end for _, end in spans)
        if latest_end < total_duration - _SCHEDULE_TOL_S:
            fail(
                label,
                f"do not cover the full scenario: the command schedule ends at {latest_end}s "
                f"but 'duration_s' is {total_duration}s. The runner falls back to "
                f"{label}[-1] whenever no window matches, so the uncovered tail "
                f"silently repeats that command rather than holding pose",
            )

    def validate_trajectory(spec: Any, label: str = "trajectory", total_duration: Optional[float] = None):
        if not isinstance(spec, dict):
            fail(label, f"must be an object, got {type(spec).__name__}")
        path_type = get_str(spec, "path_type", f"{label}.path_type")
        if path_type not in TRAJECTORY_PATH_TYPES:
            fail(
                f"{label}.path_type",
                f"must be one of {list(TRAJECTORY_PATH_TYPES)}, got {path_type!r}",
            )
        trajectory_duration = get_number(
            spec, "duration", f"{label}.duration",
            minimum=0.0, maximum=bounds.max_duration, exclusive=True,
        )
        # The reference clock reads `trajectory.duration` while the loop runs to
        # `duration_s`, and compute_tracking_error clamps its sample times to the
        # trajectory duration. A shorter trajectory therefore measures the tail of
        # the run against a frozen endpoint the base is not asked to reach, which
        # can pass a tracking gate spuriously.
        if total_duration is not None and trajectory_duration > total_duration + _SCHEDULE_TOL_S:
            fail(
                f"{label}.duration",
                f"({trajectory_duration}s) must not exceed 'duration_s' ({total_duration}s): the "
                f"reference path is clamped at trajectory.duration, so a run past that "
                f"point is scored against a frozen endpoint and can pass the tracking "
                f"gate without ever following the path",
            )
        params = spec.get("params", {})
        if not isinstance(params, dict):
            fail(f"{label}.params", f"must be an object, got {type(params).__name__}")
        for key, value in params.items():
            plabel = f"{label}.params.{key}"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                fail(plabel, f"must be a number, got {type(value).__name__}")
            try:
                checked = float(value)
            except (OverflowError, ValueError):
                # see get_number: huge int literals overflow float() before
                # the finite check can reject them.
                fail(plabel, "is outside the representable numeric range")
            if not math.isfinite(checked):
                fail(plabel, f"must be finite, got {value!r}")
        if "speed" in params:
            try:
                speed = float(params["speed"])
            except (OverflowError, ValueError):
                speed = float("nan")
            if not math.isfinite(speed):
                fail(f"{label}.params.speed", "is outside the representable numeric range")
            if not 0.0 <= speed <= bounds.max_vx:
                fail(f"{label}.params.speed", f"must be within [0.0, {bounds.max_vx}] m/s, got {speed}")
        for key in ("radius", "scale", "scale_x", "scale_y", "omega"):
            if key in params:
                try:
                    par_val = float(params[key])
                except (OverflowError, ValueError):
                    par_val = float("nan")
                if not math.isfinite(par_val):
                    fail(f"{label}.params.{key}", "is outside the representable numeric range")
                if par_val <= 0.0:
                    fail(f"{label}.params.{key}", f"must be > 0, got {params[key]}")

    def validate_waypoints(wps: Any, label: str = "waypoints"):
        if not isinstance(wps, list) or not wps:
            fail(label, f"must be a non-empty list, got {type(wps).__name__}")
        for idx, wp in enumerate(wps):
            base = f"{label}[{idx}]"
            if not isinstance(wp, dict):
                fail(base, f"must be an object, got {type(wp).__name__}")
            get_number(wp, "x", f"{base}.x", minimum=-MAX_WAYPOINT_COORD_M, maximum=MAX_WAYPOINT_COORD_M)
            get_number(wp, "y", f"{base}.y", minimum=-MAX_WAYPOINT_COORD_M, maximum=MAX_WAYPOINT_COORD_M)
            if wp.get("heading") is not None:
                get_number(wp, "heading", f"{base}.heading", minimum=-math.pi, maximum=math.pi)
            if "tolerance" in wp:
                get_number(wp, "tolerance", f"{base}.tolerance", minimum=0.0, exclusive=True)
            if "speed" in wp:
                get_number(wp, "speed", f"{base}.speed", minimum=0.0, maximum=bounds.max_vx)

    def validate_plan(entries: Any, label: str = "plan"):
        """Validate a *plan* block: a list of skills to run in order.

        A plan scenario is the benchmark-side counterpart of the LLM planner
        in ``src/go2_llm_planner.py``: the same closed skill vocabulary, the
        same parameter rules, but with the plan written out in the scenario
        file instead of produced by a model.  The skill names and parameter
        sets are taken from the planner's own registry, so the two cannot
        drift apart -- and the planner's validator is what actually runs, so
        a scenario that validates here is a plan the planner would also
        accept.

        The harness validates the plan *structurally* (closed vocabulary, no
        unknown parameters, finite numbers) and then hands the plan to the
        planner's own validator, which owns the range clamping.  That keeps
        one authority on the ranges.
        """
        if not isinstance(entries, list) or not entries:
            fail(label, f"must be a non-empty list, got {type(entries).__name__}")
        if len(entries) > MAX_SKILLS_PER_PLAN:
            fail(
                label,
                f"has {len(entries)} skills, exceeding the maximum of {MAX_SKILLS_PER_PLAN}",
            )
        for idx, entry in enumerate(entries):
            base = f"{label}[{idx}]"
            if not isinstance(entry, dict):
                fail(base, f"must be an object, got {type(entry).__name__}")
            if "skill" not in entry:
                fail(f"{base}.skill", "is required but missing")
            skill_name = entry["skill"]
            if not isinstance(skill_name, str) or skill_name.strip().lower() not in SKILL_REGISTRY:
                fail(
                    f"{base}.skill",
                    f"must be one of {sorted(SKILL_REGISTRY.keys())}, got {skill_name!r}",
                )
            args = entry.get("args", {})
            if not isinstance(args, dict):
                fail(f"{base}.args", f"must be an object, got {type(args).__name__}")
            spec = SKILL_REGISTRY[skill_name.strip().lower()]
            unknown = [k for k in args if k not in spec.params]
            if unknown:
                fail(
                    f"{base}.args",
                    f"has unknown parameter(s) {sorted(unknown)}; legal parameters are "
                    f"{list(spec.params)}",
                )
            for key in spec.required:
                if key not in args:
                    fail(f"{base}.args.{key}", "is required but missing")
            for key, value in args.items():
                if key in ("path_type", "name"):
                    if not isinstance(value, str):
                        fail(f"{base}.args.{key}", f"must be a string, got {type(value).__name__}")
                    continue
                get_number(args, key, f"{base}.args.{key}")

    # --- required top-level keys ----------------------------------------------------
    for key in cfg:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            fail(key, f"is not a recognised top-level key (known: {list(KNOWN_TOP_LEVEL_KEYS)})")
    name = get_str(cfg, "name", "name")
    if ".." in name or "/" in name or "\\" in name:
        fail("name", f"must not contain path separators or '..', got {name!r}")
    total_duration = get_number(
        cfg, "duration_s", "duration_s",
        minimum=0.0, maximum=bounds.max_duration, exclusive=True,
    )
    for optional_str in ("description", "notes"):
        if optional_str in cfg and not isinstance(cfg[optional_str], str):
            fail(optional_str, f"must be a string, got {type(cfg[optional_str]).__name__}")

    # --- mode ----------------------------------------------------------------------
    mode = cfg.get("mode", MotionMode.VELOCITY.value)
    if not isinstance(mode, str) or mode not in SCENARIO_MODES:
        fail("mode", f"must be one of {list(SCENARIO_MODES)}, got {mode!r}")

    # --- execution block -----------------------------------------------------------
    has_commands = "commands" in cfg
    has_trajectory = "trajectory" in cfg
    has_waypoints = "waypoints" in cfg
    has_plan = "plan" in cfg
    if not (has_commands or has_trajectory or has_waypoints or has_plan):
        fail(
            "commands",
            "scenario must define an execution block: 'commands', 'trajectory', "
            "'waypoints' or 'plan'",
        )
    if has_commands:
        # total_duration is threaded through deliberately.  The schedule
        # overlap and tail-coverage checks below are gated on it, and passing
        # None here left them unreachable: overlapping command windows and a
        # command schedule that did not cover the whole scenario were both
        # accepted.  It also supplies the default duration for a command that
        # omits one, which previously reached float(None).
        validate_commands(cfg["commands"], total_duration=total_duration)
    if has_trajectory:
        # Likewise: the check that the reference path does not outlast the run
        # is gated on total_duration and was unreachable without it.
        validate_trajectory(cfg["trajectory"], total_duration=total_duration)
    if has_waypoints:
        validate_waypoints(cfg["waypoints"])
    if has_plan:
        validate_plan(cfg["plan"])
    if mode == MotionMode.WAYPOINT.value and not has_waypoints:
        fail("waypoints", f"is required for mode '{mode}'")
    if mode == MotionMode.TRAJECTORY.value and not has_trajectory:
        fail("trajectory", f"is required for mode '{mode}'")
    if mode == "plan" and not has_plan:
        fail("plan", "is required for mode 'plan'")
    if mode in (MotionMode.VELOCITY.value, MotionMode.STAND.value, MotionMode.EMERGENCY_STAND.value) and not has_commands:
        fail("commands", f"is required for mode '{mode}'")

    # --- vision flag ---------------------------------------------------------------
    # A scenario may request the camera pipeline explicitly.  The runner's
    # global --vision switch stays the master override: it turns the camera on
    # for every scenario, while a per-scenario "vision": true turns it on for
    # that scenario alone.  Absent the flag, behaviour is unchanged.
    if "vision" in cfg:
        if not isinstance(cfg["vision"], bool):
            fail("vision", f"must be a boolean, got {type(cfg['vision']).__name__}")

    # --- obstacles -----------------------------------------------------------------
    # Optional, off by default.  A scenario that declares obstacles gets them
    # injected into the MuJoCo world at run time (the vendor MJCF is never
    # edited), and the acceptance gate can then measure the robot's clearance
    # from them.
    if "obstacles" in cfg:
        try:
            obstacle_specs = parse_obstacles(cfg["obstacles"])
        except ObstacleSpecError as exc:
            fail("obstacles", str(exc))
        # A scenario that declares obstacles but no vision cannot react to them;
        # the gate would then be measuring a blind robot against a gate written
        # for a seeing one.  Rejecting it here keeps that mistake loud.
        if obstacle_specs and not cfg.get("vision", False):
            fail(
                "obstacles",
                "require 'vision': true -- a scenario that places obstacles must "
                "also enable the camera, otherwise the acceptance gate measures a "
                "blind robot against a threshold written for a seeing one",
            )

    # --- acceptance thresholds -----------------------------------------------------
    expected = cfg.get("expected_metrics", {})
    if not isinstance(expected, dict):
        fail("expected_metrics", f"must be an object, got {type(expected).__name__}")
    thresholds = {
        "min_base_height_m": (0.0, bounds.max_height_m),
        "max_roll_rad": (0.0, bounds.max_roll_rad),
        "max_pitch_rad": (0.0, bounds.max_pitch_rad),
        "min_forward_displacement_m": (0.0, None),
        "max_forward_displacement_m": (0.0, None),
        "min_path_length_m": (0.0, None),
        "max_tracking_error_m": (0.0, None),
        "min_obstacle_clearance_m": (0.0, None),
        "max_obstacle_clearance_m": (0.0, None),
        "min_obstacle_engagement_m": (0.0, None),
        "min_lateral_deviation_m": (0.0, None),
        "min_obstacle_detections": (0, None),
        "min_plan_skills_completed": (0, None),
        "min_plan_waypoints_reached": (0, None),
        "max_final_goal_distance_m": (0.0, None),
    }
    for key in expected:
        label = f"expected_metrics.{key}"
        if key not in KNOWN_EXPECTED_METRICS:
            fail(label, f"is not a gated metric (known: {list(KNOWN_EXPECTED_METRICS)})")
        minimum, maximum = thresholds[key]
        get_number(expected, key, label, minimum=minimum, maximum=maximum)

    return cfg


def load_scenario(path) -> Dict[str, Any]:
    """Load and validate one scenario JSON file.

    Raises:
        ScenarioValidationError: for a missing file, malformed JSON, or any schema or
            range violation, with a message naming the file and the offending field.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise ScenarioValidationError(f"scenario file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ScenarioValidationError(
            f"{path}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    return validate_scenario(cfg, source=path)


def prune_old_run_dirs(logs_dir, pattern: str = "benchmark_suite_*", keep: int = KEEP_RECENT_RUNS):
    """Delete all but the `keep` most recent run directories under `logs_dir`.

    Suites are named ``benchmark_suite_<YYYYmmdd_HHMMSS>``, so the name sorts
    chronologically and no stat call is needed. Returns the removed paths (sorted
    oldest first). Non-matching files are never touched, and nothing is removed when
    `keep` is not a positive integer, so this cannot wipe a directory by accident.
    """
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        return []
    if not os.path.isdir(logs_dir):
        return []

    run_dirs = sorted(p for p in glob.glob(os.path.join(logs_dir, pattern), recursive=True) if os.path.isdir(p))
    stale = run_dirs[: max(0, len(run_dirs) - keep)]
    removed = []
    for path in stale:
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError as exc:
            print(f"[!] Could not prune old run directory {path}: {exc}")
    return removed


def reference_position_xy(path_type: str, params: Dict[str, Any], t_elapsed: float) -> Tuple[float, float]:
    """Analytic horizontal reference position for a parametric path.

    Derived closed-form from the scenario's own geometry, deliberately *not* by
    calling ``TrajectoryTracker``: if the evaluator asked the controller for the
    reference, a broken tracker would agree with itself and the tracking-error metric
    would be blind to exactly the defect it exists to catch.

    Supported (see ``TRAJECTORY_PATH_TYPES``):
        circle    -- ``x = cx + r*cos(w*t)``, ``y = cy + r*sin(w*t)``, ``w = speed/r``
        figure_8  -- Gerono lemniscate ``x = cx + a*sin(t)``, ``y = cy + b*sin(t)*cos(t)``
    """
    if path_type == "circle":
        cx = float(params.get("cx", 0.0))
        cy = float(params.get("cy", 0.0))
        radius = float(params.get("radius", 1.0))
        speed = float(params.get("speed", 0.30))
        omega = speed / max(radius, 0.05)
        theta = omega * t_elapsed
        return cx + radius * math.cos(theta), cy + radius * math.sin(theta)

    if path_type == "figure_8":
        cx = float(params.get("cx", 0.0))
        cy = float(params.get("cy", 0.0))
        scale_x = float(params.get("scale_x", 1.5))
        scale_y = float(params.get("scale_y", 0.75))
        omega = float(params.get("omega", 1.0))
        theta = omega * t_elapsed
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        return cx + scale_x * sin_t, cy + scale_y * sin_t * cos_t

    raise ValueError(
        f"Unsupported trajectory path_type {path_type!r}; supported: {list(TRAJECTORY_PATH_TYPES)}"
    )


def compute_tracking_error(
    scenario_config: Dict[str, Any],
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    trajectory_start_time: float = 0.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Per-sample distance from the base to its reference path, or None if there is none.

    Returns ``None`` for command- or waypoint-driven scenarios (no ``trajectory``
    block), which is the clean skip the gate reports rather than a silent zero. The
    reference is sampled at the same timestamps as the telemetry and is clamped to
    the scenario's trajectory duration, matching the tracker's own time base of
    ``elapsed = t - start_time``.
    """
    spec = scenario_config.get("trajectory")
    if not isinstance(spec, dict):
        return None

    path_type = spec.get("path_type")
    params = spec.get("params", {})
    duration = float(spec.get("duration", scenario_config.get("duration_s", 0.0)))

    errors = np.empty(len(times), dtype=float)
    for idx, t in enumerate(np.asarray(times, dtype=float)):
        t_ref = min(max(0.0, t - trajectory_start_time), duration)
        ref_x, ref_y = reference_position_xy(path_type, params, t_ref)
        errors[idx] = math.hypot(float(x[idx]) - ref_x, float(y[idx]) - ref_y)
    return {"errors": errors}


@dataclass
class _PlanCursor:
    """Which skill of a plan is currently executing, and when it ends.

    A plan is a *sequence of subgoals*, so a skill ends either when its
    declared duration elapses or -- for a ``goto_waypoint`` skill -- earlier,
    when the navigator reports that the goal has been reached.  The
    benchmark runs the plan without a model, so there is no replan: a skill
    that reaches its goal moves on to the next one, and a skill that does
    not simply runs out its window.
    """

    index: int = 0
    skill_end: Optional[float] = None
    #: The waypoint skills that were completed, for the plan metrics.
    reached_waypoints: List[Tuple[float, float]] = field(default_factory=list)
    #: The distance to each reached waypoint, measured at the moment of
    #: arrival.
    arrival_distances: List[float] = field(default_factory=list)
    #: Per-skill records: (skill name, start time, end time).
    timeline: List[Tuple[str, float, float]] = field(default_factory=list)
    skill_start: float = 0.0


class ScenarioRunner:
    """Runs a single scenario in MuJoCo, records telemetry, and generates metrics."""
    def __init__(self, model_path=MODEL_PATH, obstacles: Optional[List[ObstacleSpec]] = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model_path = model_path
        self.obstacle_specs: List[ObstacleSpec] = list(obstacles or [])
        self.obstacle_geom_ids: List[int] = []
        if self.obstacle_specs:
            # The vendor MJCF is never edited: the obstacles are appended to a
            # compiled MjSpec copy, which is byte-for-byte equivalent to the
            # original apart from the extra world geoms.
            self.model, self.obstacle_geom_ids = build_model_with_obstacles(
                self.model_path, self.obstacle_specs
            )
        else:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Configure actuator gains for stable tracking (Kp=60, Kd=5)
        configure_model(self.model)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
        self.foot_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.qpos_addrs = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addrs = self.model.jnt_dofadr[self.joint_ids]

        # Total robot mass
        self.total_mass = float(np.sum(self.model.body_mass))

        # The safety bounds the action contract enforces.  The runner passes
        # them to the planner's plan validator, so a `plan_*` scenario is
        # clamped by exactly the same envelope a model-authored plan is.
        self.bounds = SafetyBounds()

        # Robot's own collidable geoms, for the signed clearance measurement.
        self.robot_geom_ids = robot_collidable_geom_ids(self.model)

    def reset_to_standing(self):
        """Deterministically reset the simulation to the neutral standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        initialize_simulation_standing(self.model, self.data, body_height=NOMINAL_BODY_HEIGHT, settle_time=0.20)

    def foot_contacts(self) -> np.ndarray:
        """Return 4-element binary array indicating contact for each foot."""
        contacts = np.zeros(4, dtype=int)
        foot_lookup = {body_id: idx for idx, body_id in enumerate(self.foot_body_ids)}
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            for b in (b1, b2):
                leg_idx = foot_lookup.get(b)
                if leg_idx is not None:
                    contacts[leg_idx] = 1
        return contacts

    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        output_dir=None,
        save_plots: bool = True,
        gui: bool = False,
        vision: bool = False,
    ) -> Dict[str, Any]:
        """Execute a scenario schedule and record full performance artifacts."""
        name = scenario_config.get("name", "unnamed_scenario")
        total_duration = float(scenario_config.get("duration_s", 5.0))
        commands_list = scenario_config.get("commands", [])
        scenario_mode = scenario_config.get("mode", "velocity")
        waypoints_data = scenario_config.get("waypoints", [])
        trajectory_data = scenario_config.get("trajectory")

        # Reset simulation state
        self.reset_to_standing()

        controller = Go2ChampController()
        wps: List[Waypoint] = []
        if scenario_mode == "waypoint" and waypoints_data:
            wps = [
                Waypoint(
                    x=float(w["x"]),
                    y=float(w["y"]),
                    heading=w.get("heading"),
                    tolerance=float(w.get("tolerance", 0.15)),
                    speed=float(w.get("speed", 0.30)),
                )
                for w in waypoints_data
            ]
            controller.waypoint_navigator.reset(wps)

        # Parametric trajectory spec. Built once and reused every tick: the controller
        # adopts a fresh trajectory start time only when it sees a spec it has not been
        # tracking, and it compares by identity *and* value, so a stable object keeps
        # one time base for the whole run.
        trajectory_spec: Optional[TrajectorySpec] = None
        trajectory_start_time: Optional[float] = None
        if scenario_mode == MotionMode.TRAJECTORY.value and isinstance(trajectory_data, dict):
            path_type = str(trajectory_data["path_type"])
            if path_type not in TRAJECTORY_PATH_TYPES:
                # evaluate_path raises on an unsupported path, which would otherwise
                # surface as a bare traceback from inside the stepping loop.
                raise ScenarioValidationError(
                    f"scenario '{name}': unsupported trajectory path_type {path_type!r}; "
                    f"supported: {list(TRAJECTORY_PATH_TYPES)}"
                )
            trajectory_spec = TrajectorySpec(
                path_type=path_type,
                params=dict(trajectory_data.get("params", {})),
                duration=float(trajectory_data["duration"]),
            )

        # Plan block.  The plan is validated by the *planner's own* validator,
        # so the closed skill vocabulary and the parameter ranges have exactly
        # one authority.  A plan that does not validate is a scenario
        # authoring error, and the harness raises rather than silently
        # running a different schedule.
        plan_skills: List[Any] = []
        plan_skill_count = 0
        plan_data = scenario_config.get("plan")
        if plan_data is not None:
            from go2_llm_planner import parse_plan
            from go2_llm_planner import Skill as _PlannerSkill

            plan_validation = parse_plan({"plan": plan_data}, self.bounds)
            if not plan_validation.ok:
                raise ScenarioValidationError(
                    f"scenario '{name}': invalid plan: {plan_validation.error}"
                )
            plan_skills = list(plan_validation.skills)
            # The contract's own invariant: a moving sequence ends with a
            # stationary hold.  The planner's expansion adds the same hold, so
            # a plan scenario and a planner run execute the same commands.
            # A plan that already ends in a stand is left alone, so the
            # trailing hold is never counted twice.
            if not plan_skills or plan_skills[-1].name != "stand":
                plan_skills.append(
                    _PlannerSkill(name="stand", args={"duration": 1.0})
                )
            # The count is the length of the plan that actually runs, so the
            # gate can be checked against it without the reader having to
            # know whether the trailing hold was appended.
            plan_skill_count = len(plan_skills)

        # Cursor over the plan: which skill is active, and when it ends.
        plan_cursor = _PlanCursor()

        # Vision perception pipeline if enabled
        vision_pipeline = None
        last_clearance = None
        vision_was_available = False
        if vision:
            try:
                from go2_vision_perception import DepthPerceptionPipeline
                vision_pipeline = DepthPerceptionPipeline(self.model)
                vision_was_available = bool(vision_pipeline.is_available)
            except Exception as e:
                print(f"[Warning] Failed to initialize vision pipeline: {e}")
                vision_pipeline = None

        # Camera-observability counters.  These answer "did the camera actually
        # see anything", independently of whether the robot then acted on it:
        # a run in which the camera rendered a featureless scene must not be
        # reported as a successful avoidance.
        vision_frames = 0                # frames the camera was actually rendered
        vision_detections = 0            # frames whose clearance flagged an obstacle
        vision_critical_frames = 0        # frames whose clearance was in the danger band
        avoidance_deflections = 0        # controller steps where the command was deflected
        min_center_clearance = float("inf")
        min_sector_clearance = float("inf")
        sector_clearance_samples: List[Dict[str, float]] = []

        # Obstacle telemetry.  The clearance is a *geometric* signed distance
        # between the robot's collidable geoms and the obstacle geoms, so it
        # answers "did the robot hit it" independently of what the robot
        # believed.
        min_obstacle_clearance = float("inf")
        max_obstacle_clearance = float("-inf")
        max_obstacle_engagement = 0.0
        initial_obstacle_clearance: Optional[float] = None
        obstacle_contact_frames = 0
        max_lateral_deviation = 0.0
        obstacle_start_xy: Optional[Tuple[float, float]] = None
        obstacle_target_xy: Optional[Tuple[float, float]] = None

        # Reference geometry for the lateral-deviation measure.  The reference
        # line runs from the robot's start position to the nearest obstacle's
        # centre, so "deviation" measures how far the base strayed from the
        # straight path to that obstacle.  A robot that drives straight at the
        # obstacle has a deviation of about zero the whole way, and so does a
        # robot that stops in front of it -- stopping is not avoiding.
        if self.obstacle_specs:
            start_x = float(self.data.qpos[0])
            start_y = float(self.data.qpos[1])
            nearest = min(
                self.obstacle_specs,
                key=lambda o: (o.pos[0] - start_x) ** 2 + (o.pos[1] - start_y) ** 2,
            )
            obstacle_start_xy = (start_x, start_y)
            obstacle_target_xy = (float(nearest.pos[0]), float(nearest.pos[1]))
            # The clearance at the start pose, used as the baseline for the
            # engagement measure below.
            initial_obstacle_clearance = obstacle_geom_distance(
                self.model, self.data, self.obstacle_geom_ids, self.robot_geom_ids
            )

        dt = float(self.model.opt.timestep)
        telemetry_rate_hz = 250
        decimation = max(1, int(round(1.0 / (telemetry_rate_hz * dt))))

        history: List[Dict[str, Any]] = []
        step_idx = 0
        current_time = 0.0

        viewer_ctx = None
        if gui:
            try:
                viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
                print(f"[GUI] Passive viewer opened for scenario '{name}' ({total_duration:.1f}s)")
            except Exception as e:
                print(f"[Warning] Failed to launch GUI viewer: {e}. Falling back to headless.")
                viewer_ctx = None

        wall_clock_ref = time.perf_counter()

        try:
            # Run main simulation loop
            while current_time < total_duration:
                # 1. Determine active command
                active_cmd: MotionCommand
                if plan_skills:
                    # Plan mode: the skills run in order, each for its own
                    # duration.  A `goto_waypoint` skill ends *early* when the
                    # navigator reports arrival, which is what makes a
                    # multi-skill plan a sequence of subgoals rather than a
                    # sequence of fixed-length windows.
                    if plan_cursor.index >= len(plan_skills):
                        active_cmd = MotionCommand(mode=MotionMode.STAND)
                    else:
                        current_skill = plan_skills[plan_cursor.index]
                        if plan_cursor.skill_end is None:
                            plan_cursor.skill_end = current_time + skill_budget(current_skill)
                        plan_cmds = expand_skill(current_skill, self.bounds)
                        active_cmd = plan_cmds[0]
                        if active_cmd.mode == MotionMode.WAYPOINT and active_cmd.waypoint:
                            # Use the list form of the waypoint command: the
                            # controller then compares waypoint *lists*, so it
                            # does not re-arm the navigator on every tick once
                            # the goal has been reached.
                            active_cmd = MotionCommand(
                                mode=MotionMode.WAYPOINT,
                                waypoints=[active_cmd.waypoint],
                                waypoint=active_cmd.waypoint,
                            )
                elif scenario_mode == "waypoint":
                    active_cmd = MotionCommand(
                        mode=MotionMode.WAYPOINT,
                        waypoints=wps if wps else None,
                        waypoint=controller.waypoint_navigator.current_waypoint(),
                    )
                elif trajectory_spec is not None:
                    # Trajectory mode: hand the tracker its reference path. The activation
                    # time is latched by the harness here, not read back from the
                    # controller, so the evaluator's reference clock stays independent.
                    if trajectory_start_time is None:
                        trajectory_start_time = current_time
                    active_cmd = MotionCommand(mode=MotionMode.TRAJECTORY, trajectory=trajectory_spec)
                else:
                    # Find matching command from schedule
                    matched = None
                    for cmd_entry in commands_list:
                        t_start = float(cmd_entry.get("start_time_s", 0.0))
                        t_dur = float(cmd_entry.get("duration_s", total_duration))
                        if t_start <= current_time < t_start + t_dur:
                            matched = cmd_entry
                            break
                    if matched is None and len(commands_list) > 0:
                        matched = commands_list[-1]

                    if matched is not None:
                        mode_str = matched.get("mode", "trot").lower()
                        m_mode = MotionMode.STAND if mode_str == "stand" else MotionMode.VELOCITY
                        active_cmd = MotionCommand(
                            vx=float(matched.get("vx", 0.0)),
                            vy=float(matched.get("vy", 0.0)),
                            yaw_rate=float(matched.get("yaw_rate", 0.0)),
                            mode=m_mode,
                        )
                    else:
                        active_cmd = MotionCommand(mode=MotionMode.STAND)

                # 2. Extract current state
                base_pos = self.data.qpos[:3].copy()
                base_quat = self.data.qpos[3:7].copy()
                base_euler = quaternion_to_euler_xyz(base_quat)
                base_rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3).copy()

                # World linear velocity -> Body linear velocity
                linvel_world = self.data.qvel[:3].copy()
                linvel_body = base_rot_mat.T @ linvel_world

                # Angular velocity in body frame (qvel[3:6] in MuJoCo freejoint is body-frame)
                angvel_body = self.data.qvel[3:6].copy()

                joint_positions = np.array([self.data.qpos[addr] for addr in self.qpos_addrs])
                joint_velocities = np.array([self.data.qvel[addr] for addr in self.dof_addrs])

                # 3. Vision perception clearance update (20 Hz)
                sector_clearance = None
                if vision_pipeline is not None and vision_pipeline.is_available:
                    if step_idx % 25 == 0:
                        try:
                            depth_map = vision_pipeline.capture_depth(self.data)
                            last_clearance = vision_pipeline.analyze_clearance(depth_map)
                            if last_clearance is not None:
                                vision_frames += 1
                                min_sector_clearance = min(
                                    min_sector_clearance, float(last_clearance.min_dist)
                                )
                                min_center_clearance = min(
                                    min_center_clearance, float(last_clearance.center_dist)
                                )
                                if last_clearance.obstacle_detected:
                                    vision_detections += 1
                                if last_clearance.is_critical:
                                    vision_critical_frames += 1
                                sector_clearance_samples.append({
                                    "time_s": float(current_time),
                                    "left_m": float(last_clearance.left_dist),
                                    "center_m": float(last_clearance.center_dist),
                                    "right_m": float(last_clearance.right_dist),
                                    "min_m": float(last_clearance.min_dist),
                                    "closest_sector": str(last_clearance.closest_sector),
                                    "obstacle_detected": bool(last_clearance.obstacle_detected),
                                    "is_critical": bool(last_clearance.is_critical),
                                })
                        except Exception:
                            last_clearance = None
                    sector_clearance = last_clearance

                # 4. Controller step
                target_q, desired_feet, st_phases, sw_phases, info = controller.step(
                    command=active_cmd,
                    step_time=current_time,
                    dt=dt,
                    base_pos=base_pos,
                    base_euler=base_euler,
                    base_linvel_body=linvel_body,
                    base_angvel_body=angvel_body,
                    joint_positions=joint_positions,
                    sector_clearance=sector_clearance,
                )

                # 4. Apply actuator control targets
                self.data.ctrl[self.actuator_ids] = target_q

                # 4a. Plan advance.  A skill ends when its window closes, or
                # earlier when a `goto_waypoint` skill's goal has been
                # reached (the controller reports STAND once the navigator is
                # done with the waypoint).  This is what turns a plan into a
                # sequence of subgoals rather than a sequence of fixed-length
                # windows.
                if plan_skills and plan_cursor.index < len(plan_skills):
                    active_skill = plan_skills[plan_cursor.index]
                    skill_done = current_time >= (plan_cursor.skill_end or 0.0)
                    if active_skill.name == "goto_waypoint" and str(info.get("mode")) == "stand":
                        # The navigator has completed the waypoint.  The
                        # standoff distance is measured *here*, at the moment
                        # of arrival: the trailing hold then lets the robot
                        # settle, and measuring at the end of the run would
                        # fold that settling drift into the standoff.
                        skill_done = True
                        goal = active_skill.args
                        gx, gy = float(goal["x"]), float(goal["y"])
                        plan_cursor.reached_waypoints.append((gx, gy))
                        plan_cursor.arrival_distances.append(
                            math.hypot(float(base_pos[0]) - gx, float(base_pos[1]) - gy)
                        )
                    if skill_done:
                        plan_cursor.timeline.append(
                            (active_skill.name, plan_cursor.skill_start, current_time)
                        )
                        plan_cursor.index += 1
                        plan_cursor.skill_end = None
                        plan_cursor.skill_start = current_time
                        # The next skill starts on the next tick, so the
                        # controller is not stepped twice in one tick.
                        if plan_cursor.index < len(plan_skills):
                            next_skill = plan_skills[plan_cursor.index]
                            plan_cursor.skill_end = current_time + skill_budget(next_skill)

                # 4b. Did the camera change the command?  The controller reports
                # the obstacle status it applied; anything other than "none" or
                # "clear" means the reactive layer actually deflected the
                # command this step.
                obstacle_status = str(info.get("obstacle_status", "none"))
                if obstacle_status not in ("none", "clear"):
                    avoidance_deflections += 1

                # 4c. Obstacle clearance, measured geometrically from the
                # simulator's own collision geometry.  This is deliberately
                # independent of what the camera saw: it is the ground truth
                # the acceptance gate is written against.
                if self.obstacle_geom_ids:
                    clearance = obstacle_geom_distance(
                        self.model, self.data, self.obstacle_geom_ids, self.robot_geom_ids
                    )
                    if clearance < min_obstacle_clearance:
                        min_obstacle_clearance = clearance
                    if clearance > max_obstacle_clearance:
                        max_obstacle_clearance = clearance
                    if obstacle_contact(self.model, self.data, self.obstacle_geom_ids):
                        obstacle_contact_frames += 1
                    # "Engagement": how much closer to the obstacle the robot
                    # got than it started.  The raw maximum clearance is
                    # dominated by the starting distance and so cannot tell a
                    # robot that engaged the obstacle from one that stood
                    # still; the *reduction* in clearance can.
                    if initial_obstacle_clearance is not None and math.isfinite(initial_obstacle_clearance):
                        engagement = initial_obstacle_clearance - clearance
                        if engagement > max_obstacle_engagement:
                            max_obstacle_engagement = engagement
                    if obstacle_start_xy is not None and obstacle_target_xy is not None:
                        deviation = line_lateral_deviation(
                            (float(base_pos[0]), float(base_pos[1])),
                            obstacle_start_xy,
                            obstacle_target_xy,
                        )
                        if deviation > max_lateral_deviation:
                            max_lateral_deviation = deviation

                # 5. Advance physics
                mujoco.mj_step(self.model, self.data)

                # 6. Synchronize passive GUI viewer if active
                if viewer_ctx is not None:
                    if not viewer_ctx.is_running():
                        print("\n[Viewer closed by user - finalizing scenario run early]")
                        break
                    viewer_ctx.sync()
                    next_step_wall_time = wall_clock_ref + (step_idx + 1) * dt
                    sleep_dur = next_step_wall_time - time.perf_counter()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)

                # 7. Record telemetry periodically
                if step_idx % decimation == 0:
                    contacts = self.foot_contacts()
                    joint_torques = self.data.actuator_force[self.actuator_ids].copy()
                    # Two cost conventions are recorded so CoT can be reported
                    # both ways (see the CoT block in _finalize_metrics):
                    #   power_abs_w    - sum of |tau * qdot| per joint. This is the
                    #                    "no regeneration" convention: every watt a
                    #                    joint absorbs is charged as cost.
                    #   power_signed_w - sum of tau * qdot. Negative phases (stance
                    #                    braking, swing deceleration, yaw reversal)
                    #                    are credited as energy returned, which is the
                    #                    convention MIT Cheetah and most RL CoT
                    #                    figures use.
                    # The gap between the two is pure measurement artefact on a
                    # quadruped, not mechanical difference, so it must be visible
                    # rather than silently folded into one number.
                    joint_mech_power = joint_torques * joint_velocities
                    power = float(np.sum(np.abs(joint_mech_power)))
                    power_signed = float(np.sum(joint_mech_power))

                    record = {
                        "time_s": current_time,
                        "cmd_vx": info.get("cmd_vx", 0.0),
                        "cmd_vy": info.get("cmd_vy", 0.0),
                        "cmd_yaw": info.get("cmd_yaw", 0.0),
                        "mode": info.get("mode", "stand"),
                        "base_x_m": float(base_pos[0]),
                        "base_y_m": float(base_pos[1]),
                        "base_z_m": float(base_pos[2]),
                        "base_vx_body_mps": float(linvel_body[0]),
                        "base_vy_body_mps": float(linvel_body[1]),
                        "base_vz_body_mps": float(linvel_body[2]),
                        "base_roll_rad": float(base_euler[0]),
                        "base_pitch_rad": float(base_euler[1]),
                        "base_yaw_rad": float(base_euler[2]),
                        "base_wx_radps": float(angvel_body[0]),
                        "base_wy_radps": float(angvel_body[1]),
                        "base_wz_radps": float(angvel_body[2]),
                        "power_w": power,
                        "power_signed_w": power_signed,
                        "fault": info.get("fault"),
                    }
                    for leg_idx, leg_name in enumerate(LEG_NAMES):
                        record[f"{leg_name}_contact"] = int(contacts[leg_idx])
                        record[f"{leg_name}_st_phase"] = float(st_phases[leg_idx])
                        record[f"{leg_name}_sw_phase"] = float(sw_phases[leg_idx])
                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        record[f"{j_name}_target_rad"] = float(target_q[j_idx])
                        record[f"{j_name}_actual_rad"] = float(joint_positions[j_idx])
                        record[f"{j_name}_torque_nm"] = float(joint_torques[j_idx])

                    history.append(record)

                step_idx += 1
                current_time += dt
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()
            if vision_pipeline is not None:
                # Release the renderer's OpenGL context deterministically.  If
                # this were left to the garbage collector, a later renderer
                # could be created first and then have its depth buffer
                # silently degraded by the outgoing one's __del__ (the scene
                # stops being rasterised and every pixel reads the far plane,
                # so obstacles vanish from the depth frame with no error
                # raised).  Closing here, before the next pipeline is built,
                # keeps the ordering safe.
                vision_pipeline.close()

        # Compute summary metrics and evaluate pass criteria
        perception_summary: Dict[str, Any] = {
            # Latched before the pipeline was closed above: `is_available` is
            # False once the renderer has been released, so asking afterwards
            # would report a camera that plainly worked as never having run.
            "vision_enabled": bool(vision_was_available),
            "vision_frames": vision_frames,
            "obstacle_detection_frames": vision_detections,
            "critical_clearance_frames": vision_critical_frames,
            "avoidance_deflections": avoidance_deflections,
            "obstacle_count": len(self.obstacle_specs),
            "obstacle_contact_frames": obstacle_contact_frames,
            "max_lateral_deviation_m": max_lateral_deviation if self.obstacle_specs else None,
            "min_sector_clearance_m": min_sector_clearance if vision_frames else None,
            "min_center_clearance_m": min_center_clearance if vision_frames else None,
            "min_obstacle_clearance_m": (
                min_obstacle_clearance if self.obstacle_specs and math.isfinite(min_obstacle_clearance) else None
            ),
            "max_obstacle_clearance_m": (
                max_obstacle_clearance if self.obstacle_specs and math.isfinite(max_obstacle_clearance) else None
            ),
            # How much closer the robot got to the obstacle than it started.
            # This is the "did the robot actually engage the obstacle" measure:
            # the raw maximum clearance is dominated by the start distance and
            # cannot tell a robot that approached the obstacle from one that
            # stood still.
            "max_obstacle_engagement_m": (
                max_obstacle_engagement if self.obstacle_specs else None
            ),
            "initial_obstacle_clearance_m": (
                initial_obstacle_clearance
                if initial_obstacle_clearance is not None and math.isfinite(initial_obstacle_clearance)
                else None
            ),
        }
        if self.obstacle_specs:
            perception_summary["obstacles"] = [o.to_json() for o in self.obstacle_specs]

        # Plan metrics: which subgoals were reached, and where the robot
        # ended up relative to the last one.  These are the plan-family
        # equivalent of the waypoint metrics, and they are what a `plan_*`
        # scenario's `expected_metrics` gate is checked against.  They are
        # computed *before* `_compute_metrics` so the gate can read them.
        plan_metrics: Optional[Dict[str, Any]] = None
        if plan_skill_count:
            final_x = float(history[-1]["base_x_m"]) if history else 0.0
            final_y = float(history[-1]["base_y_m"]) if history else 0.0
            plan_metrics = {
                "plan_skill_count": plan_skill_count,
                "plan_skills_completed": plan_cursor.index,
                "plan_waypoints_reached": len(plan_cursor.reached_waypoints),
                "plan_timeline": [
                    {"skill": name, "start_s": start, "end_s": end}
                    for (name, start, end) in plan_cursor.timeline
                ],
            }
            # The docking standoff: the distance to the *last* waypoint the
            # plan asked for, measured at the moment the robot arrived there.
            # (The distance from the *final* position is also reported, but
            # the gate uses the arrival distance, because the trailing hold
            # lets the robot settle and drift by a few centimetres.)
            if plan_cursor.arrival_distances:
                plan_metrics["final_goal_distance_m"] = float(
                    plan_cursor.arrival_distances[-1]
                )
            if plan_cursor.reached_waypoints:
                last_gx, last_gy = plan_cursor.reached_waypoints[-1]
                plan_metrics["final_position_distance_m"] = math.hypot(
                    final_x - last_gx, final_y - last_gy
                )

        metrics = self._compute_metrics(
            scenario_config,
            history,
            total_duration,
            trajectory_start_time=float(trajectory_start_time or 0.0),
            perception=perception_summary,
            plan=plan_metrics,
        )
        if plan_metrics is not None:
            metrics["plan"] = plan_metrics

        # Output logging directory
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self._save_csv(history, os.path.join(output_dir, "data.csv"))
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            if sector_clearance_samples:
                # The per-frame camera record, so a reader can check the
                # detection claim frame by frame rather than trusting a
                # single aggregate counter.
                with open(os.path.join(output_dir, "sector_clearance.json"), "w") as f:
                    json.dump(sector_clearance_samples, f, indent=2)
            if save_plots:
                self._generate_plots(history, metrics, scenario_config, os.path.join(output_dir, "plots.png"))

        return metrics

    def _compute_metrics(
        self,
        scenario_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        requested_duration: float,
        trajectory_start_time: float = 0.0,
        perception: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate biomechanical, tracking, and energy metrics."""
        if not history:
            return {"error": "No telemetry recorded"}

        times = np.array([r["time_s"] for r in history])
        run_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        # Kinematic states
        x = np.array([r["base_x_m"] for r in history])
        y = np.array([r["base_y_m"] for r in history])
        z = np.array([r["base_z_m"] for r in history])
        vx_meas = np.array([r["base_vx_body_mps"] for r in history])
        vx_cmd = np.array([r["cmd_vx"] for r in history])
        wz_meas = np.array([r["base_wz_radps"] for r in history])
        wz_cmd = np.array([r["cmd_yaw"] for r in history])
        roll = np.array([r["base_roll_rad"] for r in history])
        pitch = np.array([r["base_pitch_rad"] for r in history])
        yaw = np.array([r["base_yaw_rad"] for r in history])

        # Tracking performance
        vx_rmse = float(np.sqrt(np.mean((vx_meas - vx_cmd) ** 2)))
        wz_rmse = float(np.sqrt(np.mean((wz_meas - wz_cmd) ** 2)))

        # Joint tracking RMSE
        joint_errors = []
        joint_samples: Dict[str, np.ndarray] = {}
        for j_name in JOINT_NAMES:
            actual = np.array([r[f"{j_name}_actual_rad"] for r in history])
            target = np.array([r[f"{j_name}_target_rad"] for r in history])
            joint_samples[f"{j_name}_actual_rad"] = actual
            joint_samples[f"{j_name}_target_rad"] = target
            joint_errors.append(actual - target)
        joint_errors_mat = np.array(joint_errors)
        joint_tracking_rmse = float(np.sqrt(np.mean(joint_errors_mat ** 2)))

        # Attitude and height stability
        base_height_mean = float(np.mean(z))
        base_height_std = float(np.std(z))
        base_height_min = float(np.min(z))
        base_roll_std = float(np.std(roll))
        base_pitch_std = float(np.std(pitch))
        base_roll_peak = float(np.max(np.abs(roll)))
        base_pitch_peak = float(np.max(np.abs(pitch)))

        # Displacement
        delta_x = float(x[-1] - x[0])
        delta_y = float(y[-1] - y[0])
        total_dist = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

        # Reference-path tracking error, for scenarios that declare a `trajectory` block.
        # None (not zero) when there is no reference path, so the gate can say it skipped.
        tracking = compute_tracking_error(
            scenario_config, times, x, y, trajectory_start_time=trajectory_start_time
        )
        tracking_errors = None if tracking is None else tracking["errors"]
        if tracking_errors is not None:
            max_tracking_error = float(np.max(tracking_errors))
            rms_tracking_error = float(np.sqrt(np.mean(tracking_errors ** 2)))
        else:
            max_tracking_error = None
            rms_tracking_error = None

        # Energy & Power
        # Two conventions, reported side by side:
        #   cot_abs    - from sum|tau*qdot|. No regeneration credited; this is the
        #                pessimistic "gross mechanical work" convention.
        #   cot_signed - from sum(tau*qdot). Regenerative phases are credited, which
        #                is what MIT Cheetah / Walk-These-Ways style CoT figures use.
        # cot_net is the difference expressed in energy terms and is the portion of
        # cot_abs that is measurement artefact rather than mechanical cost.
        powers = np.array([r["power_w"] for r in history])
        powers_signed = np.array([r.get("power_signed_w", r["power_w"]) for r in history])
        dt_sample = float(np.mean(np.diff(times))) if len(times) > 1 else 0.004
        energy_j = float(np.sum(powers) * dt_sample)
        energy_signed_j = float(np.sum(powers_signed) * dt_sample)
        avg_power = float(np.mean(powers))
        avg_power_signed = float(np.mean(powers_signed))
        cot = float(energy_j / (self.total_mass * GRAVITY * max(0.01, total_dist)))
        cot_signed = float(energy_signed_j / (self.total_mass * GRAVITY * max(0.01, total_dist)))
        # Net dissative fraction: how much of gross work was recovered in place.
        energy_recovered_j = float(energy_j - energy_signed_j)
        recovery_fraction = float(energy_recovered_j / energy_j) if energy_j > 0.0 else 0.0

        # Contact duty factors
        duty_factors = {}
        for leg_name in LEG_NAMES:
            c = np.array([r[f"{leg_name}_contact"] for r in history])
            duty_factors[leg_name] = float(np.mean(c))

        # Check faults
        faults_recorded = [r["fault"] for r in history if r["fault"] is not None]

        # Non-finite guard: backstops the whole gate, not one comparison. Every check
        # below is an ordered comparison, and `nan < x` / `nan > x` are both False, so
        # a NaN-poisoned run would otherwise record no failure and print PASS.
        telemetry_channels: Dict[str, Any] = {
            "base_x_m": x,
            "base_y_m": y,
            "base_z_m": z,
            "base_vx_body_mps": vx_meas,
            "base_wz_radps": wz_meas,
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "base_yaw_rad": yaw,
            "cmd_vx": vx_cmd,
            "cmd_yaw": wz_cmd,
            "power_w": powers,
            "power_signed_w": powers_signed,
        }
        telemetry_channels.update(joint_samples)
        if tracking_errors is not None:
            telemetry_channels["tracking_error_m"] = tracking_errors

        # Evaluate against scenario expected_metrics
        expected = scenario_config.get("expected_metrics", {})
        passed = True
        failure_reasons: List[str] = []

        non_finite = _find_non_finite(telemetry_channels)
        if non_finite:
            passed = False
            failure_reasons.append(
                "Non-finite telemetry (NaN/Inf) in: " + ", ".join(non_finite)
            )

        if len(faults_recorded) > 0:
            passed = False
            failure_reasons.append(f"Controller faults triggered: {set(faults_recorded)}")

        if "min_base_height_m" in expected and base_height_min < float(expected["min_base_height_m"]):
            passed = False
            failure_reasons.append(f"Min base height {base_height_min:.3f}m < expected {expected['min_base_height_m']}m")

        if "max_roll_rad" in expected and base_roll_peak > float(expected["max_roll_rad"]):
            passed = False
            failure_reasons.append(f"Peak roll {base_roll_peak:.3f}rad > expected {expected['max_roll_rad']}rad")

        if "max_pitch_rad" in expected and base_pitch_peak > float(expected["max_pitch_rad"]):
            passed = False
            failure_reasons.append(f"Peak pitch {base_pitch_peak:.3f}rad > expected {expected['max_pitch_rad']}rad")

        if "min_forward_displacement_m" in expected and delta_x < float(expected["min_forward_displacement_m"]):
            passed = False
            failure_reasons.append(f"Forward displacement {delta_x:.3f}m < expected {expected['min_forward_displacement_m']}m")

        if "max_forward_displacement_m" in expected and delta_x > float(expected["max_forward_displacement_m"]):
            passed = False
            failure_reasons.append(f"Forward displacement {delta_x:.3f}m > expected {expected['max_forward_displacement_m']}m")

        if "min_path_length_m" in expected and total_dist < float(expected["min_path_length_m"]):
            passed = False
            failure_reasons.append(
                f"Path length travelled {total_dist:.3f}m < expected {expected['min_path_length_m']}m "
                "(cumulative distance along the actual base path; a stationary base scores ~0m however "
                "it was commanded, and unlike net displacement this is non-zero for a closed loop)"
            )

        if "max_tracking_error_m" in expected:
            threshold = float(expected["max_tracking_error_m"])
            if max_tracking_error is None:
                failure_reasons.append(
                    f"Skipped max_tracking_error_m > {threshold}m: scenario declares no reference path "
                    "(no 'trajectory' block)"
                )
            elif not math.isfinite(max_tracking_error):
                # Explicit rather than relying on `error > threshold`, which is always
                # False for NaN and would silently pass a NaN-poisoned run.
                passed = False
                failure_reasons.append(
                    f"Tracking error is non-finite ({max_tracking_error!r}); cannot compare against "
                    f"expected {threshold}m"
                )
            elif max_tracking_error > threshold:
                passed = False
                failure_reasons.append(
                    f"Max reference-path tracking error {max_tracking_error:.3f}m > expected {threshold}m "
                    f"(rms {rms_tracking_error:.3f}m); the base never converged onto the commanded path"
                )

        # --- obstacle-avoidance gate -------------------------------------------------
        # These four checks are the ones that make an avoidance scenario mean
        # something.  All of them are *geometric*: they are computed from the
        # simulator's own collision geometry, never from what the robot
        # believed about the world, so a stationary or blind robot cannot pass
        # them.  They are also deliberately complementary:
        #
        #   min_obstacle_clearance_m  the robot never got closer than X to the
        #                              obstacle -- rules out ploughing into it.
        #   min_obstacle_engagement_m the robot *did* close on the obstacle --
        #                              rules out a robot that avoided the
        #                              obstacle by refusing to move at all.
        #   min_lateral_deviation_m   the robot left the straight line to the
        #                              obstacle -- rules out a robot that
        #                              stopped short without swerving.
        #   min_obstacle_detections   the camera actually flagged the obstacle
        #                              -- rules out a run whose gate was met by
        #                              luck while the camera saw nothing.
        #
        # A robot passes only if it both kept its distance AND closed on the
        # obstacle AND left the straight line AND the camera saw the
        # obstacle.  A blind robot (no vision) sees nothing, so it fails
        # min_obstacle_detections and, having no reason to swerve, also fails
        # min_lateral_deviation_m.  A robot that never moves fails
        # min_obstacle_engagement_m and min_lateral_deviation_m.
        perception = perception or {}
        min_obstacle_clearance = perception.get("min_obstacle_clearance_m")
        max_obstacle_engagement = perception.get("max_obstacle_engagement_m")
        max_lateral_deviation = perception.get("max_lateral_deviation_m")
        obstacle_detections = int(perception.get("obstacle_detection_frames", 0) or 0)

        if "min_obstacle_clearance_m" in expected:
            threshold = float(expected["min_obstacle_clearance_m"])
            if min_obstacle_clearance is None or not math.isfinite(float(min_obstacle_clearance)):
                passed = False
                failure_reasons.append(
                    f"Skipped min_obstacle_clearance_m >= {threshold}m: the scenario declares no "
                    "obstacles, so there is no geometry to measure clearance against"
                )
            elif float(min_obstacle_clearance) < threshold:
                passed = False
                failure_reasons.append(
                    f"Minimum obstacle clearance {float(min_obstacle_clearance):.3f}m < expected "
                    f"{threshold}m; the robot's collidable geometry came closer to the obstacle "
                    "than the gate allows (a negative value means the robot interpenetrated it)"
                )

        if "min_obstacle_engagement_m" in expected:
            threshold = float(expected["min_obstacle_engagement_m"])
            if max_obstacle_engagement is None or not math.isfinite(float(max_obstacle_engagement)):
                passed = False
                failure_reasons.append(
                    f"Skipped min_obstacle_engagement_m >= {threshold}m: the scenario declares no "
                    "obstacles, so there is no geometry to measure engagement against"
                )
            elif float(max_obstacle_engagement) < threshold:
                passed = False
                failure_reasons.append(
                    f"Maximum obstacle engagement {float(max_obstacle_engagement):.3f}m < expected "
                    f"{threshold}m; the robot never closed on the obstacle, so it avoided nothing "
                    "-- it simply never went near it"
                )

        if "min_lateral_deviation_m" in expected:
            threshold = float(expected["min_lateral_deviation_m"])
            if max_lateral_deviation is None:
                passed = False
                failure_reasons.append(
                    f"Skipped min_lateral_deviation_m >= {threshold}m: the scenario declares no "
                    "obstacles, so there is no reference line to deviate from"
                )
            elif float(max_lateral_deviation) < threshold:
                passed = False
                failure_reasons.append(
                    f"Maximum lateral deviation {float(max_lateral_deviation):.3f}m < expected "
                    f"{threshold}m; the robot drove straight at the obstacle without steering "
                    "around it, so it stopped short of a collision rather than avoiding"
                )

        if "min_obstacle_detections" in expected:
            threshold = int(expected["min_obstacle_detections"])
            if obstacle_detections < threshold:
                passed = False
                failure_reasons.append(
                    f"Camera obstacle detections {obstacle_detections} < expected {threshold}; "
                    "the depth camera did not flag the obstacle, so the run cannot be counted "
                    "as a vision-driven avoidance"
                )

        # --- plan-family gate ------------------------------------------------
        # A plan scenario's success is a *compositional* property: the run
        # only passes if the skills completed in order AND the robot ended
        # where the last subgoal said it should.  These three checks are what
        # a two-skill plan needs to be measurable, and they are the metrics a
        # `plan_*` scenario declares in `expected_metrics`.
        plan_metrics = plan
        if "min_plan_skills_completed" in expected:
            threshold = int(expected["min_plan_skills_completed"])
            completed = int(plan_metrics.get("plan_skills_completed", 0)) if plan_metrics else 0
            if completed < threshold:
                passed = False
                failure_reasons.append(
                    f"Plan skills completed {completed} < expected {threshold}; the plan did "
                    "not run to its declared subgoal sequence"
                )
        if "min_plan_waypoints_reached" in expected:
            threshold = int(expected["min_plan_waypoints_reached"])
            reached = int(plan_metrics.get("plan_waypoints_reached", 0)) if plan_metrics else 0
            if reached < threshold:
                passed = False
                failure_reasons.append(
                    f"Plan waypoints reached {reached} < expected {threshold}; the robot did "
                    "not arrive at the subgoals the plan asked for"
                )
        if "max_final_goal_distance_m" in expected:
            threshold = float(expected["max_final_goal_distance_m"])
            if not plan_metrics or plan_metrics.get("final_goal_distance_m") is None:
                passed = False
                failure_reasons.append(
                    f"Skipped max_final_goal_distance_m <= {threshold}m: the plan reached no "
                    "waypoint subgoal, so there is no final goal to measure a standoff from"
                )
            elif float(plan_metrics["final_goal_distance_m"]) > threshold:
                passed = False
                failure_reasons.append(
                    f"Final distance to the last subgoal {float(plan_metrics['final_goal_distance_m']):.3f}m "
                    f"> expected {threshold}m; the robot did not end at the standoff the plan asked for"
                )

        metrics = {
            "scenario": scenario_config.get("name"),
            "passed": passed,
            "failure_reasons": failure_reasons,
            "run_duration_s": run_duration,
            "requested_duration_s": requested_duration,
            "forward_displacement_m": delta_x,
            "lateral_displacement_m": delta_y,
            "total_path_length_m": total_dist,
            "max_tracking_error_m": max_tracking_error,
            "rms_tracking_error_m": rms_tracking_error,
            "final_position": [float(x[-1]), float(y[-1]), float(z[-1])],
            "final_heading_rad": float(yaw[-1]),
            "velocity_tracking_rmse_mps": vx_rmse,
            "yaw_rate_tracking_rmse_radps": wz_rmse,
            "joint_tracking_rmse_rad": joint_tracking_rmse,
            "base_height_mean_m": base_height_mean,
            "base_height_std_m": base_height_std,
            "base_height_min_m": base_height_min,
            "base_roll_std_rad": base_roll_std,
            "base_pitch_std_rad": base_pitch_std,
            "base_roll_peak_abs_rad": base_roll_peak,
            "base_pitch_peak_abs_rad": base_pitch_peak,
            "estimated_energy_j": energy_j,
            "estimated_average_power_w": avg_power,
            "estimated_cost_of_transport": cot,
            "estimated_cost_of_transport_signed": cot_signed,
            "estimated_energy_signed_j": energy_signed_j,
            "estimated_average_power_signed_w": avg_power_signed,
            "estimated_energy_recovered_j": energy_recovered_j,
            "recovery_fraction": recovery_fraction,
            "contact_duty_factor": duty_factors,
        }

        # Camera-observability and obstacle-geometry metrics.  These are the
        # machine-readable record that the vision path was actually exercised:
        # how many frames the camera rendered, how many of those flagged an
        # obstacle, how far away the nearest thing in each sector ever was, how
        # many times the reactive layer changed the commanded velocity, and the
        # geometric clearance the robot actually kept from the obstacle.
        #
        # `min_obstacle_clearance_m` is None (not 0.0 and not inf) when the
        # scenario declares no obstacles, so "no obstacle" is never confused
        # with "the robot was touching the obstacle".
        metrics["vision_enabled"] = bool(perception.get("vision_enabled", False))
        metrics["vision_frames"] = int(perception.get("vision_frames", 0) or 0)
        metrics["obstacle_detection_frames"] = int(
            perception.get("obstacle_detection_frames", 0) or 0
        )
        metrics["critical_clearance_frames"] = int(
            perception.get("critical_clearance_frames", 0) or 0
        )
        metrics["avoidance_deflections"] = int(
            perception.get("avoidance_deflections", 0) or 0
        )
        metrics["min_sector_clearance_m"] = perception.get("min_sector_clearance_m")
        metrics["min_center_clearance_m"] = perception.get("min_center_clearance_m")
        metrics["min_obstacle_clearance_m"] = min_obstacle_clearance
        metrics["max_obstacle_clearance_m"] = perception.get("max_obstacle_clearance_m")
        metrics["max_obstacle_engagement_m"] = perception.get("max_obstacle_engagement_m")
        metrics["initial_obstacle_clearance_m"] = perception.get("initial_obstacle_clearance_m")
        metrics["max_lateral_deviation_m"] = max_lateral_deviation
        metrics["obstacle_contact_frames"] = int(
            perception.get("obstacle_contact_frames", 0) or 0
        )
        metrics["obstacle_count"] = int(perception.get("obstacle_count", 0) or 0)
        # The obstacle declarations themselves, so a reader of metrics.json can
        # see what the run was measured against.
        if perception.get("obstacles"):
            metrics["obstacles"] = perception["obstacles"]

        # Backstop: no computed metric may be non-finite either, whatever produced it.
        non_finite_metrics = _find_non_finite(metrics)
        if non_finite_metrics:
            metrics["passed"] = False
            metrics["failure_reasons"].append(
                "Non-finite computed metric(s): " + ", ".join(non_finite_metrics)
            )

        return metrics

    def _save_csv(self, history: List[Dict[str, Any]], filepath):
        """Save telemetry records to CSV file."""
        if not history:
            return
        fieldnames = list(history[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def _generate_plots(
        self,
        history: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        scenario_config: Dict[str, Any],
        filepath,
    ):
        """Generate high-density diagnostic multi-panel figure."""
        times = [r["time_s"] for r in history]
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
        fig.suptitle(
            f"Go2 CHAMP Benchmark: {scenario_config.get('name')} (Status: {'PASS' if metrics.get('passed') else 'FAIL'})",
            fontsize=15, fontweight="bold",
        )

        # 1. Forward and Lateral Velocities
        ax = axes[0, 0]
        ax.plot(times, [r["cmd_vx"] for r in history], "k--", label="Cmd vx (mps)", alpha=0.8)
        ax.plot(times, [r["base_vx_body_mps"] for r in history], "b-", label="Meas vx (mps)", alpha=0.9)
        ax.plot(times, [r["base_vy_body_mps"] for r in history], "g-", label="Meas vy (mps)", alpha=0.7)
        ax.set_title("Body Linear Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 2. Heading / Yaw Rate
        ax = axes[0, 1]
        ax.plot(times, [r["cmd_yaw"] for r in history], "k--", label="Cmd Yaw Rate (rad/s)", alpha=0.8)
        ax.plot(times, [r["base_wz_radps"] for r in history], "m-", label="Meas Yaw Rate (rad/s)", alpha=0.9)
        ax.plot(times, [r["base_yaw_rad"] for r in history], "c-.", label="Heading Yaw (rad)", alpha=0.7)
        ax.set_title("Heading and Rotational Tracking")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rate / Angle (rad/s, rad)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 3. Base Height and Attitude
        ax = axes[1, 0]
        ax.plot(times, [r["base_z_m"] for r in history], "r-", label="Base Height (m)")
        ax.plot(times, [r["base_roll_rad"] for r in history], "b:", label="Roll (rad)")
        ax.plot(times, [r["base_pitch_rad"] for r in history], "g:", label="Pitch (rad)")
        ax.axhline(NOMINAL_BODY_HEIGHT, color="gray", linestyle="--", alpha=0.5, label="Nominal Height")
        ax.set_title("Base Height and Attitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (m) / Angle (rad)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 4. 2D Horizontal Trajectory
        ax = axes[1, 1]
        x = [r["base_x_m"] for r in history]
        y = [r["base_y_m"] for r in history]
        ax.plot(x, y, "b-", label="Robot Path", linewidth=2.0)
        ax.plot(x[0], y[0], "go", markersize=8, label="Start")
        ax.plot(x[-1], y[-1], "rs", markersize=8, label="End")

        # Plot scenario waypoints if available
        waypoints = scenario_config.get("waypoints", [])
        if waypoints:
            wpx = [w["x"] for w in waypoints]
            wpy = [w["y"] for w in waypoints]
            ax.plot(wpx, wpy, "k--o", label="Target Waypoints", alpha=0.7)
            for idx, w in enumerate(waypoints):
                ax.annotate(f"WP{idx+1}", (w["x"], w["y"]), textcoords="offset points", xytext=(5,5))

        ax.set_title("2D Ground Trajectory (X-Y)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        # 5. Foot Contact States
        ax = axes[2, 0]
        colors = ["blue", "orange", "green", "red"]
        for idx, leg_name in enumerate(LEG_NAMES):
            contacts = [r[f"{leg_name}_contact"] * 0.8 + idx for r in history]
            ax.plot(times, contacts, label=f"{leg_name} (Duty: {metrics['contact_duty_factor'][leg_name]:.2f})", color=colors[idx])
        ax.set_yticks(range(4))
        ax.set_yticklabels(LEG_NAMES)
        ax.set_title("Foot Contact Duty Cycle")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Leg")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

        # 6. Electrical / Mechanical Power
        ax = axes[2, 1]
        powers = [r["power_w"] for r in history]
        powers_signed = [r.get("power_signed_w", r["power_w"]) for r in history]
        ax.plot(times, powers, "purple", alpha=0.7, label=f"Gross |tau*qdot| (Avg: {metrics['estimated_average_power_w']:.1f} W)")
        ax.plot(times, powers_signed, "teal", alpha=0.85, label=f"Signed (Avg: {metrics['estimated_average_power_signed_w']:.1f} W)")
        ax.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_title(
            f"Actuator Power (CoT gross {metrics['estimated_cost_of_transport']:.2f} / "
            f"net {metrics['estimated_cost_of_transport_signed']:.2f})"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        plt.savefig(filepath, dpi=150)
        plt.close(fig)


def run_benchmark_suite(
    scenario_names: Optional[List[str]] = None,
    scenarios_dir=SCENARIOS_DIR,
    logs_dir=LOGS_DIR,
    gui: bool = False,
    vision: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the full suite of scenarios and print comparison summary table."""
    if not scenario_names or "all" in scenario_names:
        json_files = sorted(glob.glob(os.path.join(scenarios_dir, "*.json")))
    else:
        resolved_files = []
        for name in scenario_names:
            p = name
            if os.path.exists(p) and os.path.isfile(p):
                resolved_files.append(p)
            elif os.path.exists(os.path.join(scenarios_dir, os.path.basename(p))):
                resolved_files.append(os.path.join(scenarios_dir, os.path.basename(p)))
            else:
                stem = os.path.splitext(os.path.basename(p))[0]
                if os.path.exists(os.path.join(scenarios_dir, f"{stem}.json")):
                    resolved_files.append(os.path.join(scenarios_dir, f"{stem}.json"))
                else:
                    resolved_files.append(os.path.join(scenarios_dir, f"{name}.json" if not name.endswith(".json") else name))
        json_files = resolved_files

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(logs_dir, f"benchmark_suite_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)

    # Retention: each suite leaves a directory of plots/CSV/JSON behind, so keep only
    # the most recent KEEP_RECENT_RUNS. Pruning is name-based and scoped to this logs
    # directory; the current run is created first and is the newest, so it survives.
    pruned = prune_old_run_dirs(logs_dir)
    if pruned:
        print(f"[*] Pruned {len(pruned)} old run dir(s), keeping the {KEEP_RECENT_RUNS} most recent")
        for path in pruned:
            print(f"    removed {os.path.basename(path)}")

    # A runner is built per distinct obstacle set.  Scenarios that declare no
    # obstacles share one plain runner; a scenario that declares obstacles needs
    # its own model, because the obstacles are compiled into the MuJoCo world.
    runner: Optional[ScenarioRunner] = None
    runner_key: Optional[Tuple[Any, ...]] = None
    results = []

    print("\n" + "=" * 80)
    print(f"RUNNING GO2 CHAMP BENCHMARK SUITE ({len(json_files)} SCENARIOS)")
    print("=" * 80)

    for sf in json_files:
        if not os.path.exists(sf):
            print(f"[!] Warning: Scenario file {sf} not found, skipping.")
            continue

        try:
            cfg = load_scenario(sf)
        except ScenarioValidationError as exc:
            print(f"[!] Invalid scenario skipped: {exc}")
            continue
        s_name = cfg.get("name", os.path.splitext(os.path.basename(sf))[0])
        scenario_out = os.path.join(suite_dir, f"scenario_{s_name}")
        print(f"\n--> Running scenario: '{s_name}' ({cfg.get('duration_s')}s)...")

        t0 = time.time()
        try:
            obstacle_specs = parse_obstacles(cfg.get("obstacles"))
            runner_key_now = tuple(
                f"{o.name}:{o.kind}:{o.pos}:{o.size}:{o.yaw}:{o.rgba}" for o in obstacle_specs
            )
            if runner is None or runner_key_now != runner_key:
                runner = ScenarioRunner(obstacles=obstacle_specs)
                runner_key = runner_key_now

            # The global --vision switch is the master override; a scenario may
            # also ask for the camera on its own.  Either turns it on, so a
            # scenario file can request perception without the operator
            # having to remember the flag -- while a scenario that does not ask
            # for it still runs blind under a blind suite.
            scenario_vision = bool(vision or cfg.get("vision", False))

            metrics = runner.run_scenario(
                cfg, output_dir=scenario_out, save_plots=True, gui=gui, vision=scenario_vision
            )
        except ScenarioValidationError as exc:
            # Clear scenario-level error (e.g. a path the tracker cannot execute)
            # instead of a bare traceback from inside the stepping loop.
            print(f"    [!] Scenario could not be executed: {exc}")
            continue
        elapsed = time.time() - t0

        status_str = "[PASS]" if metrics.get("passed") else "[FAIL]"
        print(f"    Result: {status_str} in {elapsed:.2f}s wall-clock")
        print(f"    Displacement: fwd={metrics.get('forward_displacement_m', 0.0):.3f}m, lat={metrics.get('lateral_displacement_m', 0.0):.3f}m")
        print(f"    Path length travelled: {metrics.get('total_path_length_m', 0.0):.3f}m")
        if metrics.get("max_tracking_error_m") is not None:
            print(f"    Reference tracking error: max={metrics['max_tracking_error_m']:.3f}m, rms={metrics['rms_tracking_error_m']:.3f}m")
        print(f"    Tracking: vx RMSE={metrics.get('velocity_tracking_rmse_mps', 0.0):.3f} m/s, yaw RMSE={metrics.get('yaw_rate_tracking_rmse_radps', 0.0):.3f} rad/s")
        print(f"    Attitude: roll peak={math.degrees(metrics.get('base_roll_peak_abs_rad', 0.0)):.1f} deg, pitch peak={math.degrees(metrics.get('base_pitch_peak_abs_rad', 0.0)):.1f} deg")
        if metrics.get("obstacle_count"):
            print(
                f"    Obstacles: {metrics['obstacle_count']} | min clearance="
                f"{metrics.get('min_obstacle_clearance_m')} m | max lateral deviation="
                f"{metrics.get('max_lateral_deviation_m')} m"
            )
        if metrics.get("vision_enabled"):
            print(
                f"    Vision: {metrics.get('vision_frames')} frames, "
                f"{metrics.get('obstacle_detection_frames')} detections, "
                f"{metrics.get('avoidance_deflections')} deflections, "
                f"min sector clearance={metrics.get('min_sector_clearance_m')} m"
            )
        if not metrics.get("passed"):
            print(f"    Failures: {metrics.get('failure_reasons')}")

        results.append(metrics)

    # Save summary report
    summary_path = os.path.join(suite_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown table
    md_lines = [
        f"# Go2 CHAMP Trajectory Benchmark Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Scenario | Status | Fwd Dist (m) | Path Len (m) | Track Err Max (m) | Track Err RMS (m) | Vx RMSE (m/s) | Yaw RMSE (rad/s) | Height (m) | Roll Peak (deg) | Pitch Peak (deg) | CoT **net** | CoT gross | Recovery |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        "",
        "**CoT convention:** *net* = $\\sum(\\tau \\cdot \\dot{q})$, regenerative work credited — this is the MIT Cheetah / RL-comparable figure. "
        "*gross* = $\\sum |\\tau \\cdot \\dot{q}|$, no regeneration credited. *Recovery* is the fraction of gross work returned in place. "
        "Compare CoT on the **net** column; the gross column overstates cost by ~40–44% on these scenarios.",
    ]
    for m in results:
        status_badge = "✅ PASS" if m.get("passed") else "❌ FAIL"
        roll_deg = math.degrees(m.get("base_roll_peak_abs_rad", 0.0))
        pitch_deg = math.degrees(m.get("base_pitch_peak_abs_rad", 0.0))
        track_max = m.get("max_tracking_error_m")
        track_rms = m.get("rms_tracking_error_m")
        track_max_str = "n/a" if track_max is None else f"{track_max:.2f}"
        track_rms_str = "n/a" if track_rms is None else f"{track_rms:.2f}"
        md_lines.append(
            f"| `{m.get('scenario')}` | {status_badge} | {m.get('forward_displacement_m', 0.0):.2f} | "
            f"{m.get('total_path_length_m', 0.0):.2f} | {track_max_str} | {track_rms_str} | "
            f"{m.get('velocity_tracking_rmse_mps', 0.0):.3f} | {m.get('yaw_rate_tracking_rmse_radps', 0.0):.3f} | "
            f"{m.get('base_height_mean_m', 0.0):.3f} | {roll_deg:.1f}° | {pitch_deg:.1f}° | "
            f"**{m.get('estimated_cost_of_transport_signed', 0.0):.2f}** | {m.get('estimated_cost_of_transport', 0.0):.2f} | "
            f"{m.get('recovery_fraction', 0.0) * 100:.0f}% |"
        )
    md_lines.append("")
    report_md = "\n".join(md_lines)
    with open(os.path.join(suite_dir, "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(report_md)
    print(f"\nArtifacts saved in: {suite_dir}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CHAMP trajectory tracking on Unitree Go2.")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        help="Scenario name(s) to run (e.g. stand_hold, turn_in_place, s_curve) or 'all'",
    )
    parser.add_argument("--gui", dest="gui", action="store_true", default=True, help="Launch MuJoCo passive GUI viewer during scenario execution (default: True)")
    parser.add_argument("--no-gui", "--headless", dest="gui", action="store_false", help="Run headlessly without GUI viewer (for automated tests)")
    parser.add_argument("--vision", action="store_true", default=False, help="Enable closed-loop depth vision perception and reactive obstacle avoidance")
    parser.add_argument("--scenarios-dir", type=pathlib.Path, default=SCENARIOS_DIR, help="Directory containing scenario JSON files")
    parser.add_argument("--logs-dir", type=pathlib.Path, default=LOGS_DIR, help="Directory to save run logs")
    args = parser.parse_args()

    run_benchmark_suite(
        scenario_names=args.scenario,
        scenarios_dir=args.scenarios_dir,
        logs_dir=args.logs_dir,
        gui=args.gui,
        vision=args.vision,
    )


if __name__ == "__main__":
    main()
