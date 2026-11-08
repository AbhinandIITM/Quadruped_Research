"""
test_scenario_validation.py
===========================
Unit tests for the benchmark harness's scenario loader/validator and for the
pass/fail gate's non-finite (NaN/Inf) guard and path-length criterion.

These cover two defects the harness previously could not see:

* every gate check was an ordered comparison, and ``nan < x`` / ``nan > x`` are
  both ``False``, so a NaN-poisoned run recorded no failure and printed PASS;
* scenario JSON was loaded with a bare ``json.load`` and read with ``.get()`` +
  defaults, so malformed files raised an uncaught ``JSONDecodeError`` and absurd
  values flowed straight into the runner.
"""

import glob
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import evaluate_champ_trajectories as harness
from evaluate_champ_trajectories import (
    KEEP_RECENT_RUNS,
    ScenarioRunner,
    ScenarioValidationError,
    _find_non_finite,
    load_scenario,
    prune_old_run_dirs,
    reference_position_xy,
    validate_scenario,
)
from go2_champ_control import JOINT_NAMES, LEG_NAMES, NOMINAL_BODY_HEIGHT, TRAJECTORY_PATH_TYPES


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENARIOS_EXPERIMENTS = os.path.join(_ROOT, "experiments", "scenarios")
SCENARIOS_DIR = (
    _SCENARIOS_EXPERIMENTS
    if os.path.exists(_SCENARIOS_EXPERIMENTS)
    else os.path.join(_ROOT, "scenarios")
)


def make_history(n=50, dt=0.01, x_step=0.0, **overrides):
    """Build a synthetic telemetry history carrying every key _compute_metrics reads."""
    history = []
    for i in range(n):
        record = {
            "time_s": i * dt,
            "cmd_vx": 0.30,
            "cmd_yaw": 0.0,
            "base_x_m": i * x_step,
            "base_y_m": 0.0,
            "base_z_m": NOMINAL_BODY_HEIGHT,
            "base_vx_body_mps": x_step / dt,
            "base_wz_radps": 0.0,
            "base_roll_rad": 0.0,
            "base_pitch_rad": 0.0,
            "base_yaw_rad": 0.0,
            "power_w": 30.0,
            "fault": None,
        }
        for leg_name in LEG_NAMES:
            record[f"{leg_name}_contact"] = 1
        for j_name in JOINT_NAMES:
            record[f"{j_name}_actual_rad"] = 0.0
            record[f"{j_name}_target_rad"] = 0.0
        record.update(overrides)
        history.append(record)
    return history


def compute_metrics(history, expected_metrics, duration_s=1.0, trajectory=None, trajectory_start_time=0.0, plan=None):
    """Run _compute_metrics without paying for a MuJoCo model in __init__.

    numpy floating-point warnings are muted: deliberately non-finite telemetry is
    the whole point of several tests below.
    """
    runner = ScenarioRunner.__new__(ScenarioRunner)
    runner.total_mass = 15.0
    cfg = {"name": "synthetic", "duration_s": duration_s, "expected_metrics": expected_metrics}
    if trajectory is not None:
        cfg["trajectory"] = trajectory
    with np.errstate(all="ignore"):
        return runner._compute_metrics(
            cfg, history, duration_s, trajectory_start_time=trajectory_start_time, plan=plan
        )


def valid_config(**overrides):
    """A minimal scenario that passes validate_scenario, for mutation in tests."""
    cfg = {
        "name": "unit_test",
        "duration_s": 2.0,
        "commands": [
            {
                "start_time_s": 0.0,
                "duration_s": 2.0,
                "mode": "trot",
                "vx": 0.2,
                "vy": 0.0,
                "yaw_rate": 0.0,
            }
        ],
        "expected_metrics": {"min_base_height_m": 0.24},
    }
    cfg.update(overrides)
    return cfg


class TestNonFiniteGate(unittest.TestCase):
    """Any non-finite telemetry value must force passed=False with a named reason."""

    def test_finite_run_passes(self):
        metrics = compute_metrics(
            make_history(x_step=0.01),
            {"min_base_height_m": 0.24, "max_roll_rad": 0.2, "max_pitch_rad": 0.2},
        )
        self.assertTrue(metrics["passed"], metrics["failure_reasons"])
        self.assertEqual(metrics["failure_reasons"], [])

    def test_nan_base_height_fails_and_is_named(self):
        metrics = compute_metrics(make_history(base_z_m=float("nan")), {"min_base_height_m": 0.24})
        self.assertFalse(metrics["passed"])
        self.assertTrue(any("base_z_m" in r for r in metrics["failure_reasons"]), metrics["failure_reasons"])

    def test_nan_roll_fails_and_is_named(self):
        metrics = compute_metrics(make_history(base_roll_rad=float("nan")), {"max_roll_rad": 0.2})
        self.assertFalse(metrics["passed"])
        self.assertTrue(any("base_roll_rad" in r for r in metrics["failure_reasons"]), metrics["failure_reasons"])

    def test_inf_pitch_fails(self):
        metrics = compute_metrics(make_history(base_pitch_rad=float("inf")), {"max_pitch_rad": 0.2})
        self.assertFalse(metrics["passed"])
        self.assertTrue(any("base_pitch_rad" in r for r in metrics["failure_reasons"]), metrics["failure_reasons"])

    def test_nan_velocity_fails(self):
        """The reachable path: a NaN command from the LLM backend poisons the state."""
        metrics = compute_metrics(make_history(base_vx_body_mps=float("nan")), {})
        self.assertFalse(metrics["passed"])
        self.assertTrue(any("base_vx_body_mps" in r for r in metrics["failure_reasons"]), metrics["failure_reasons"])

    def test_nan_command_channel_fails(self):
        metrics = compute_metrics(make_history(cmd_vx=float("nan")), {})
        self.assertFalse(metrics["passed"])
        self.assertTrue(any("cmd_vx" in r for r in metrics["failure_reasons"]), metrics["failure_reasons"])

    def test_nan_joint_telemetry_fails(self):
        metrics = compute_metrics(make_history(**{f"{JOINT_NAMES[0]}_actual_rad": float("nan")}), {})
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any(f"{JOINT_NAMES[0]}_actual_rad" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_graceful_without_expected_metrics(self):
        """A bare config with no thresholds must still fail on non-finite telemetry."""
        metrics = compute_metrics(make_history(base_x_m=float("nan")), {})
        self.assertFalse(metrics["passed"])

    def test_non_finite_computed_metric_is_backstopped(self):
        """Even if a metric goes non-finite on its own, the gate must fail."""
        metrics = compute_metrics(make_history(power_w=1e308), {})
        self.assertFalse(np.isfinite(metrics["estimated_energy_j"]))
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any("Non-finite computed metric" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_find_non_finite_walks_nested_containers(self):
        offending = _find_non_finite(
            {
                "ok": 1.0,
                "flag": True,
                "text": "nan",
                "series": np.array([0.0, float("nan")]),
                "nested": {"deep": float("inf")},
                "listing": [0.0, float("nan")],
            }
        )
        self.assertEqual(sorted(offending), ["listing[1]", "nested.deep", "series"])


class TestPathLengthGate(unittest.TestCase):
    """min_path_length_m must separate a stationary base from a moving one."""

    def test_stationary_base_fails_path_length_gate(self):
        metrics = compute_metrics(make_history(x_step=0.0), {"min_path_length_m": 2.6})
        self.assertFalse(metrics["passed"])
        self.assertLess(metrics["total_path_length_m"], 0.01)
        self.assertTrue(
            any("Path length" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_moving_base_passes_path_length_gate(self):
        # 50 samples, 0.1 m per 0.01 s sample -> ~4.9 m of cumulative travel.
        metrics = compute_metrics(make_history(x_step=0.1), {"min_path_length_m": 2.6})
        self.assertAlmostEqual(metrics["total_path_length_m"], 4.9, places=6)
        self.assertTrue(metrics["passed"], metrics["failure_reasons"])

    def test_closed_loop_has_zero_net_displacement_but_nonzero_path_length(self):
        """The reason net displacement cannot gate a circle: it returns to its start."""
        history = make_history(n=40, x_step=0.0)
        for i, record in enumerate(history):
            angle = 2.0 * np.pi * i / (len(history) - 1)  # last sample closes the loop
            record["base_x_m"] = float(np.cos(angle)) * 1.5
            record["base_y_m"] = float(np.sin(angle)) * 1.5
        metrics = compute_metrics(history, {"min_path_length_m": 2.6})
        self.assertLess(abs(metrics["forward_displacement_m"]), 1e-9)
        self.assertAlmostEqual(metrics["total_path_length_m"], 2.0 * np.pi * 1.5, delta=0.1)
        self.assertTrue(metrics["passed"], metrics["failure_reasons"])


class TestScheduleSemanticsAreEnforced(unittest.TestCase):
    """The schedule overlap, tail-coverage, and trajectory-duration checks.

    These three rules were written into the validator and documented in its
    error messages, but they were *unreachable*: each was gated on a
    ``total_duration`` argument that the only two call sites never passed, so
    the validator returned early before reaching them.  A scenario with two
    overlapping command windows, a scenario whose commands stopped before the
    end of the run, and a scenario whose reference path outlasted the run were
    all accepted.

    The runtime behaviours the rules describe are real -- the runner's
    first-match-in-list-order rule really does decide overlapping windows, and
    the runner really does fall back to the last command when nothing matches
    -- so an accepted-but-ambiguous scenario is a silent mis-run, not a
    harmless redundancy.  These tests pin the checks open.
    """

    def test_overlapping_command_windows_are_rejected(self):
        cfg = valid_config(
            duration_s=10.0,
            commands=[
                {"start_time_s": 0.0, "duration_s": 6.0, "mode": "trot", "vx": 0.2},
                {"start_time_s": 3.0, "duration_s": 4.0, "mode": "trot", "vx": 0.2},
            ],
        )
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="scenarios/unit_test.json")
        self.assertIn("overlaps an earlier command", str(ctx.exception))

    def test_touching_windows_are_not_an_overlap(self):
        """A window that starts exactly when the previous one ends is legal."""
        cfg = valid_config(
            duration_s=10.0,
            commands=[
                {"start_time_s": 0.0, "duration_s": 5.0, "mode": "trot", "vx": 0.2},
                {"start_time_s": 5.0, "duration_s": 5.0, "mode": "trot", "vx": 0.3},
            ],
        )
        validate_scenario(cfg, source="scenarios/unit_test.json")

    def test_uncovered_tail_is_rejected(self):
        cfg = valid_config(
            duration_s=10.0,
            commands=[{"start_time_s": 0.0, "duration_s": 4.0, "mode": "trot", "vx": 0.2}],
        )
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="scenarios/unit_test.json")
        self.assertIn("do not cover the full scenario", str(ctx.exception))

    def test_tail_coverage_uses_the_latest_end_not_the_list_order(self):
        """The schedule need not be listed in time order; the latest end counts."""
        cfg = valid_config(
            duration_s=10.0,
            commands=[
                {"start_time_s": 6.0, "duration_s": 4.0, "mode": "trot", "vx": 0.2},
                {"start_time_s": 0.0, "duration_s": 5.0, "mode": "trot", "vx": 0.1},
            ],
        )
        validate_scenario(cfg, source="scenarios/unit_test.json")

    def test_trajectory_longer_than_the_run_is_rejected(self):
        cfg = valid_config(
            mode="trajectory",
            duration_s=5.0,
            trajectory={
                "path_type": "circle",
                "duration": 9.0,
                "params": {"radius": 1.0, "speed": 0.3},
            },
        )
        del cfg["commands"]
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="scenarios/unit_test.json")
        self.assertIn("must not exceed", str(ctx.exception))

    def test_trajectory_matching_the_run_is_accepted(self):
        cfg = valid_config(
            mode="trajectory",
            duration_s=5.0,
            trajectory={
                "path_type": "circle",
                "duration": 5.0,
                "params": {"radius": 1.0, "speed": 0.3},
            },
        )
        del cfg["commands"]
        validate_scenario(cfg, source="scenarios/unit_test.json")

    def test_a_command_may_omit_its_duration_and_run_to_the_end(self):
        """A missing duration defaults to the end of the scenario, not a crash."""
        cfg = valid_config(
            duration_s=10.0,
            commands=[{"start_time_s": 0.0, "mode": "trot", "vx": 0.2}],
        )
        validate_scenario(cfg, source="scenarios/unit_test.json")


class TestScenarioValidation(unittest.TestCase):
    """The loader must reject bad scenarios with a message naming file and field."""

    def assert_rejects(self, cfg, *expected_fragments):
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="scenarios/unit_test.json")
        message = str(ctx.exception)
        self.assertIn("scenarios/unit_test.json", message)
        for fragment in expected_fragments:
            self.assertIn(fragment, message)
        return message

    def test_valid_config_accepted(self):
        cfg = valid_config()
        self.assertIs(validate_scenario(cfg, source="scenarios/unit_test.json"), cfg)

    def test_missing_name(self):
        cfg = valid_config()
        del cfg["name"]
        self.assert_rejects(cfg, "name", "required")

    def test_missing_duration(self):
        cfg = valid_config()
        del cfg["duration_s"]
        self.assert_rejects(cfg, "duration_s", "required")

    def test_wrong_type_for_duration(self):
        self.assert_rejects(valid_config(duration_s="five"), "duration_s", "must be a number")

    def test_non_finite_duration(self):
        self.assert_rejects(valid_config(duration_s=float("nan")), "duration_s", "must be finite")

    def test_duration_beyond_safety_bounds(self):
        self.assert_rejects(valid_config(duration_s=600.0), "duration_s", "must be <=")

    def test_bad_mode_value(self):
        self.assert_rejects(valid_config(mode="teleport"), "mode", "must be one of")

    def test_missing_execution_block(self):
        cfg = valid_config()
        del cfg["commands"]
        self.assert_rejects(cfg, "commands", "execution block")

    def test_absurd_velocity_rejected(self):
        cfg = valid_config(
            commands=[{"start_time_s": 0.0, "duration_s": 1.0, "mode": "trot", "vx": 100.0}]
        )
        self.assert_rejects(cfg, "commands[0].vx", "must be <= 0.6")

    def test_absurd_yaw_rate_rejected(self):
        cfg = valid_config(
            commands=[{"start_time_s": 0.0, "duration_s": 1.0, "mode": "trot", "yaw_rate": 45.0}]
        )
        self.assert_rejects(cfg, "commands[0].yaw_rate", "must be <=")

    def test_unknown_command_mode_rejected(self):
        cfg = valid_config(
            commands=[{"start_time_s": 0.0, "duration_s": 1.0, "mode": "sprint", "vx": 0.2}]
        )
        self.assert_rejects(cfg, "commands[0].mode", "must be one of")

    def test_command_entry_wrong_type(self):
        self.assert_rejects(valid_config(commands=["trot"]), "commands[0]", "must be an object")

    def test_empty_commands_rejected(self):
        self.assert_rejects(valid_config(commands=[]), "commands", "non-empty list")

    def test_trajectory_requires_path_type(self):
        cfg = valid_config(
            mode="trajectory",
            trajectory={"params": {"speed": 0.3}, "duration": 10.0},
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "trajectory.path_type", "required")

    def test_trajectory_rejects_unknown_path_type(self):
        cfg = valid_config(
            mode="trajectory",
            trajectory={"path_type": "helix", "params": {"speed": 0.3}, "duration": 10.0},
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "trajectory.path_type", "must be one of")

    def test_unsupported_trajectory_path_type_rejected(self):
        """The validator must certify only what the tracker can actually execute."""
        self.assertNotIn("spline", TRAJECTORY_PATH_TYPES)
        cfg = valid_config(
            mode="trajectory",
            trajectory={"path_type": "spline", "params": {}, "duration": 10.0},
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "trajectory.path_type", "must be one of")

    def test_accepted_path_types_match_the_tracker(self):
        for path_type in TRAJECTORY_PATH_TYPES:
            with self.subTest(path_type=path_type):
                params = {"radius": 1.5, "speed": 0.3} if path_type == "circle" else {
                    "scale_x": 1.2, "scale_y": 0.6, "omega": 0.33
                }
                cfg = valid_config(
                    mode="trajectory",
                    duration_s=12.0,
                    trajectory={"path_type": path_type, "params": params, "duration": 10.0},
                )
                del cfg["commands"]
                validate_scenario(cfg, source="scenarios/unit_test.json")

    def test_trajectory_rejects_offending_param(self):
        cfg = valid_config(
            mode="trajectory",
            duration_s=12.0,
            trajectory={"path_type": "circle", "params": {"radius": -1.5, "speed": 0.3}, "duration": 10.0},
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "trajectory.params.radius", "must be > 0")

    def test_trajectory_rejects_out_of_range_speed(self):
        cfg = valid_config(
            mode="trajectory",
            duration_s=12.0,
            trajectory={"path_type": "circle", "params": {"radius": 1.5, "speed": 9.9}, "duration": 10.0},
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "trajectory.params.speed", "must be within")

    def test_trajectory_mode_without_block(self):
        """Declaring trajectory mode with only a commands block must be rejected."""
        self.assert_rejects(valid_config(mode="trajectory"), "trajectory", "required for mode")

    def test_waypoint_mode_without_waypoints(self):
        """Declaring waypoint mode with only a commands block must be rejected."""
        self.assert_rejects(valid_config(mode="waypoint"), "waypoints", "required for mode")

    def test_waypoint_out_of_workspace(self):
        cfg = valid_config(
            mode="waypoint",
            waypoints=[{"x": 99.0, "y": 0.0, "tolerance": 0.15, "speed": 0.3}],
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "waypoints[0].x", "must be <=")

    def test_waypoint_non_positive_tolerance(self):
        cfg = valid_config(
            mode="waypoint",
            waypoints=[{"x": 1.0, "y": 0.0, "tolerance": 0.0, "speed": 0.3}],
        )
        del cfg["commands"]
        self.assert_rejects(cfg, "waypoints[0].tolerance", "must be > 0")

    def test_unknown_expected_metric_key_rejected(self):
        """A typo'd threshold silently disables a check, so it must be an error."""
        self.assert_rejects(
            valid_config(expected_metrics={"min_base_heigt_m": 0.24}),
            "expected_metrics.min_base_heigt_m",
            "not a gated metric",
        )

    def test_non_finite_expected_metric_rejected(self):
        self.assert_rejects(
            valid_config(expected_metrics={"min_path_length_m": float("nan")}),
            "expected_metrics.min_path_length_m",
            "must be finite",
        )

    def test_out_of_range_expected_metric_rejected(self):
        self.assert_rejects(
            valid_config(expected_metrics={"max_roll_rad": 3.0}),
            "expected_metrics.max_roll_rad",
            "must be <=",
        )

    def test_name_with_path_separator_rejected(self):
        self.assert_rejects(valid_config(name="../escape"), "name", "path separators")

    def test_top_level_must_be_object(self):
        self.assert_rejects([1, 2, 3], "top level")


class TestScenarioFileLoading(unittest.TestCase):
    """load_scenario must fail cleanly (no traceback) on disk-level problems."""

    def test_all_shipped_scenarios_load_and_validate(self):
        """Every shipped scenario must load, and the set must not shrink.

        The count is a guard against an *accidentally* deleted scenario, so
        it tracks the shipped set by name rather than by number: a new
        scenario family (the `plan_*` family, added with the LLM planner)
        must be added here deliberately, and a deletion shows up as a named
        difference rather than a bare number.
        """
        files = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "*.json")))
        expected = {
            "circle_orbit.json",
            "figure_eight.json",
            "forward_speed_step.json",
            "obstacle_avoidance.json",
            "plan_goto_then_dock.json",
            "plan_three_waypoints.json",
            "s_curve.json",
            "stand_hold.json",
            "stop_and_hold.json",
            "turn_in_place.json",
            "waypoint_square.json",
            "plan_waypoint_square.json",
        }
        self.assertEqual({os.path.basename(f) for f in files}, expected, sorted(os.path.basename(f) for f in files))
        for path in files:
            with self.subTest(scenario=os.path.basename(path)):
                cfg = load_scenario(path)
                self.assertIsInstance(cfg["name"], str)

    def test_malformed_json_reports_file_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "broken.json")
            open(path, "w", encoding="utf-8").write('{"name": "broken", "duration_s": }')
            with self.assertRaises(ScenarioValidationError) as ctx:
                load_scenario(path)
            message = str(ctx.exception)
            self.assertIn("broken.json", message)
            self.assertIn("invalid JSON", message)
            self.assertIn("line", message)
            self.assertNotIsInstance(ctx.exception, json.JSONDecodeError)

    def test_missing_file_reports_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "absent.json")
            with self.assertRaises(ScenarioValidationError) as ctx:
                load_scenario(path)
            self.assertIn("absent.json", str(ctx.exception))

    def test_bad_scenario_on_disk_names_offending_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "speedy.json")
            open(path, "w", encoding="utf-8").write(json.dumps(valid_config(name="speedy", duration_s=-3.0)))
            with self.assertRaises(ScenarioValidationError) as ctx:
                load_scenario(path)
            self.assertIn("speedy.json", str(ctx.exception))
            self.assertIn("duration_s", str(ctx.exception))


class TestReferencePath(unittest.TestCase):
    """The reference must come from the scenario's geometry, not from the tracker."""

    def test_circle_closed_form(self):
        params = {"cx": -1.5, "cy": 0.0, "radius": 1.5, "speed": 0.35}
        omega = 0.35 / 1.5
        # t = 0 is the path's start, which the scenario anchors on the robot's pose.
        self.assertEqual(reference_position_xy("circle", params, 0.0), (0.0, 0.0))
        x, y = reference_position_xy("circle", params, (np.pi / 2) / omega)
        self.assertAlmostEqual(x, -1.5, places=9)
        self.assertAlmostEqual(y, 1.5, places=9)

    def test_figure_eight_closed_form(self):
        params = {"cx": 0.0, "cy": 0.0, "scale_x": 1.2, "scale_y": 0.6, "omega": 0.3304}
        self.assertEqual(reference_position_xy("figure_8", params, 0.0), (0.0, 0.0))
        x, y = reference_position_xy("figure_8", params, (np.pi / 2) / 0.3304)
        self.assertAlmostEqual(x, 1.2, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_reference_raises_on_unsupported_path(self):
        with self.assertRaises(ValueError):
            reference_position_xy("spline", {}, 1.0)

    def test_shipped_scenarios_declare_only_honoured_param_keys(self):
        """The tracker ignores unknown params, so a stale name would silently vanish."""
        honoured = {
            "circle": {"cx", "cy", "radius", "speed"},
            "figure_8": {"cx", "cy", "scale_x", "scale_y", "omega"},
        }
        for name in ("circle_orbit", "figure_eight"):
            with self.subTest(scenario=name):
                spec = load_scenario(os.path.join(SCENARIOS_DIR, f"{name}.json"))["trajectory"]
                path_type = spec["path_type"]
                self.assertIn(path_type, TRAJECTORY_PATH_TYPES)
                self.assertLessEqual(set(spec["params"]), honoured[path_type])


class TestTrackingErrorGate(unittest.TestCase):
    """max_tracking_error_m must separate following the path from not following it."""

    def _scenario(self, name):
        return load_scenario(os.path.join(SCENARIOS_DIR, f"{name}.json"))

    def test_stationary_base_fails_both_trajectory_scenarios(self):
        """A robot standing still cannot satisfy the shipped thresholds."""
        # 2001 samples at 0.01 s spans the longer of the two scenario durations.
        history = make_history(n=2001, x_step=0.0)
        for name, geometric_floor in (("circle_orbit", 2.9), ("figure_eight", 1.15)):
            cfg = self._scenario(name)
            with self.subTest(scenario=name):
                metrics = compute_metrics(
                    history,
                    cfg["expected_metrics"],
                    duration_s=cfg["duration_s"],
                    trajectory=cfg["trajectory"],
                )
                self.assertFalse(metrics["passed"])
                self.assertGreater(metrics["max_tracking_error_m"], geometric_floor)
                self.assertGreater(
                    metrics["max_tracking_error_m"],
                    cfg["expected_metrics"]["max_tracking_error_m"],
                )
                self.assertTrue(
                    any("tracking error" in r for r in metrics["failure_reasons"]),
                    metrics["failure_reasons"],
                )

    def test_straight_line_dash_fails_tracking_but_passes_path_length(self):
        """The blind spot path length alone cannot cover: driving off is not standing still."""
        cfg = self._scenario("circle_orbit")
        # 0.35 m/s forward for 15 s = 5.25 m of genuine travel, well over the 2.6 m bar.
        history = make_history(n=1501, x_step=0.0035)
        metrics = compute_metrics(
            history,
            cfg["expected_metrics"],
            duration_s=cfg["duration_s"],
            trajectory=cfg["trajectory"],
        )
        self.assertGreaterEqual(metrics["total_path_length_m"], 5.2)
        self.assertGreaterEqual(metrics["total_path_length_m"], cfg["expected_metrics"]["min_path_length_m"])
        self.assertGreater(metrics["max_tracking_error_m"], 5.0)
        self.assertFalse(metrics["passed"])
        self.assertFalse(any("Path length" in r for r in metrics["failure_reasons"]))

    def test_reference_following_base_passes(self):
        """A base that traces the circle exactly must pass every shipped threshold."""
        cfg = self._scenario("circle_orbit")
        spec = cfg["trajectory"]
        params = spec["params"]
        n, dt = 1501, 0.01
        history = make_history(n=n, x_step=0.0)
        for i, record in enumerate(history):
            t_ref = min(i * dt, spec["duration"])
            ref_x, ref_y = reference_position_xy(spec["path_type"], params, t_ref)
            record["base_x_m"] = ref_x
            record["base_y_m"] = ref_y
        metrics = compute_metrics(
            history, cfg["expected_metrics"], duration_s=cfg["duration_s"], trajectory=spec
        )
        self.assertLess(metrics["max_tracking_error_m"], 1e-6)
        self.assertTrue(metrics["passed"], metrics["failure_reasons"])

    def test_non_trajectory_scenario_skips_the_gate_cleanly(self):
        metrics = compute_metrics(make_history(x_step=0.01), {"max_tracking_error_m": 0.4})
        self.assertIsNone(metrics["max_tracking_error_m"])
        self.assertIsNone(metrics["rms_tracking_error_m"])
        self.assertTrue(metrics["passed"], metrics["failure_reasons"])
        self.assertTrue(
            any("Skipped max_tracking_error_m" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_nan_tracking_error_trips_the_non_finite_gate(self):
        cfg = self._scenario("circle_orbit")
        metrics = compute_metrics(
            make_history(n=101, base_x_m=float("nan")),
            cfg["expected_metrics"],
            duration_s=cfg["duration_s"],
            trajectory=cfg["trajectory"],
        )
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any("tracking_error_m" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_nan_tracking_error_is_not_silently_passed_by_comparison(self):
        """`nan > threshold` is False, so the gate must test finiteness explicitly."""
        cfg = self._scenario("circle_orbit")
        finite_history = make_history(n=101, x_step=0.0035)
        nan_errors = np.full(len(finite_history), np.nan)
        with mock.patch.object(
            harness, "compute_tracking_error", return_value={"errors": nan_errors}
        ):
            metrics = compute_metrics(
                finite_history,
                cfg["expected_metrics"],
                duration_s=cfg["duration_s"],
                trajectory=cfg["trajectory"],
            )
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any("Tracking error is non-finite" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )


class TestTrajectoryModeWiring(unittest.TestCase):
    """The harness must hand the tracker a path it can execute, or fail cleanly."""

    def test_unsupported_path_surfaces_as_scenario_error(self):
        runner = ScenarioRunner()
        cfg = {
            "name": "bad_path",
            "duration_s": 1.0,
            "mode": "trajectory",
            "trajectory": {"path_type": "spline", "params": {}, "duration": 1.0},
        }
        with self.assertRaises(ScenarioValidationError) as ctx:
            runner.run_scenario(cfg)
        self.assertIn("bad_path", str(ctx.exception))
        self.assertIn("spline", str(ctx.exception))


class TestLogRetention(unittest.TestCase):
    """Old benchmark_suite_* runs must be pruned; unrelated files must survive."""

    def _make_runs(self, root, names):
        for name in names:
            os.makedirs(os.path.join(root, name), exist_ok=True)
            open(os.path.join(root, name, "metrics.json"), "w", encoding="utf-8").write("{}")

    def test_keeps_only_the_most_recent_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = tmp
            names = [f"benchmark_suite_2026010{i}_120000" for i in range(1, 6)]
            self._make_runs(root, names)
            removed = prune_old_run_dirs(root, keep=2)
            self.assertEqual([os.path.basename(p) for p in removed], names[:3])
            self.assertEqual(sorted(os.path.basename(p) for p in os.listdir(root)), names[3:])

    def test_unrelated_entries_are_never_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = tmp
            self._make_runs(root, [f"benchmark_suite_2026010{i}_120000" for i in range(1, 5)])
            keepers = [
                os.path.join(root, "llm_audit.jsonl"),
                os.path.join(root, "champ_trot_20261005_233345"),
                os.path.join(root, "go2_nonimu_mujoco_metrics_20260910_205012.json"),
            ]
            for path in keepers:
                os.mkdir(path)
                open(os.path.join(path, "data.json"), "w", encoding="utf-8").write("{}")
            prune_old_run_dirs(root, keep=1)
            for path in keepers:
                self.assertTrue(os.path.exists(path), os.path.basename(path))
                self.assertTrue(os.path.exists(os.path.join(path, "data.json")))

    def test_no_prune_when_under_the_limit_or_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = tmp
            names = [f"benchmark_suite_2026010{i}_120000" for i in range(1, 4)]
            self._make_runs(root, names)
            self.assertEqual(prune_old_run_dirs(root, keep=10), [])
            self.assertEqual(prune_old_run_dirs(root, keep=0), [])
            self.assertEqual(prune_old_run_dirs(root, keep=-1), [])
            self.assertEqual(len(os.listdir(root)), 3)

    def test_missing_directory_is_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(prune_old_run_dirs(os.path.join(tmp, "nope")), [])

    def test_default_retention_is_a_small_positive_constant(self):
        self.assertIsInstance(KEEP_RECENT_RUNS, int)
        self.assertGreater(KEEP_RECENT_RUNS, 0)


if __name__ == "__main__":
    unittest.main()


class TestOversizedNumericLiterals(unittest.TestCase):
    """Huge integer literals must surface as ScenarioValidationError.

    json.load gives a 400-digit integer a Python int that overflows
    float(), and the overflow happens *before* the finiteness check that
    would otherwise reject the value. load_scenario only catches
    FileNotFoundError and JSONDecodeError, so an unguarded conversion
    escapes as a bare OverflowError traceback instead of the documented
    error type -- a crash, not a validation report.
    """

    BIG = int("1" * 400)

    def _write(self, config):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "scenario.json")
        open(path, "w").write(json.dumps(config))
        return path

    def _assert_rejected(self, config, label):
        with self.assertRaises(ScenarioValidationError, msg=label):
            load_scenario(self._write(config))

    def test_oversized_duration_is_rejected_not_raised(self):
        self._assert_rejected({"name": "big", "duration_s": self.BIG}, "duration_s")

    def test_oversized_command_time_is_rejected(self):
        self._assert_rejected(
            {"name": "big", "duration_s": 5.0,
             "commands": [{"start_time_s": self.BIG, "duration_s": 1.0, "vx": 0.1}]},
            "start_time_s")

    def test_oversized_trajectory_param_is_rejected(self):
        expected = {"max_tracking_error_m": 0.2}
        for key in ("radius", "omega", "scale_x"):
            with self.subTest(key=key):
                self._assert_rejected(
                    {"name": "big", "duration_s": 5.0,
                     "trajectory": {"path_type": "circle", "duration": 5.0,
                                    "params": {key: self.BIG}},
                     "expected_metrics": expected},
                    key)

    def test_non_finite_scan_treats_oversize_as_offending(self):
        offending = _find_non_finite({"a": self.BIG, "b": 1.0})
        self.assertIn("a", offending)
        self.assertNotIn("b", offending)


class TestPlanScenarioFamily(unittest.TestCase):
    """The `plan_*` scenario family: a plan is a sequence of subgoals.

    A plan scenario is the benchmark-side counterpart of the LLM planner in
    `src/go2_llm_planner.py`: the same closed skill vocabulary, but the plan is
    written in the scenario file rather than produced by a model.  The gate
    machinery has to be able to tell a plan that *ran* from one that did not,
    and the validator has to reject a plan the planner would also reject.
    """

    def _plan_scenarios(self):
        return [
            p for p in sorted(glob.glob(os.path.join(SCENARIOS_DIR, "plan_*.json")))
        ]

    def test_at_least_one_plan_scenario_ships(self):
        self.assertGreaterEqual(len(self._plan_scenarios()), 1)

    def test_plan_scenarios_load_and_validate(self):
        for path in self._plan_scenarios():
            with self.subTest(scenario=os.path.basename(path)):
                cfg = load_scenario(path)
                self.assertEqual(cfg["mode"], "plan")
                self.assertIn("plan", cfg)
                self.assertTrue(cfg["plan"])

    def test_plan_scenarios_declare_a_real_gate(self):
        """A plan scenario with no gate would pass by construction."""
        for path in self._plan_scenarios():
            with self.subTest(scenario=os.path.basename(path)):
                cfg = load_scenario(path)
                expected = cfg["expected_metrics"]
                self.assertTrue(expected, f"{os.path.basename(path)} declares no expected_metrics")
                # At least one plan-specific gate, so the run is measured as
                # a plan and not merely as locomotion.
                plan_gates = {
                    "min_plan_skills_completed",
                    "min_plan_waypoints_reached",
                    "max_final_goal_distance_m",
                }
                self.assertTrue(
                    plan_gates & set(expected),
                    f"{os.path.basename(path)} declares no plan-specific gate",
                )

    def test_plan_scenarios_declare_at_least_two_skills(self):
        """A one-skill plan is not a plan; the family is about decomposition."""
        for path in self._plan_scenarios():
            with self.subTest(scenario=os.path.basename(path)):
                cfg = load_scenario(path)
                self.assertGreaterEqual(len(cfg["plan"]), 2)

    def test_unknown_skill_in_a_plan_is_rejected(self):
        cfg = {
            "name": "bad_plan",
            "duration_s": 5.0,
            "mode": "plan",
            "plan": [{"skill": "teleport", "args": {"x": 1.0}}],
        }
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="unit_test")
        self.assertIn("plan[0].skill", str(ctx.exception))

    def test_unknown_parameter_in_a_plan_is_rejected(self):
        cfg = {
            "name": "bad_plan",
            "duration_s": 5.0,
            "mode": "plan",
            "plan": [
                {"skill": "move_velocity",
                 "args": {"vx": 0.3, "duration": 1.0, "thrust": 5.0}}
            ],
        }
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="unit_test")
        self.assertIn("unknown parameter", str(ctx.exception))

    def test_non_finite_parameter_in_a_plan_is_rejected(self):
        cfg = {
            "name": "bad_plan",
            "duration_s": 5.0,
            "mode": "plan",
            "plan": [{"skill": "move_velocity", "args": {"vx": float("nan"), "duration": 1.0}}],
        }
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="unit_test")
        self.assertIn("must be finite", str(ctx.exception))

    def test_plan_mode_requires_a_plan_block(self):
        cfg = {
            "name": "bad_plan",
            "duration_s": 5.0,
            "mode": "plan",
            "commands": [{"start_time_s": 0.0, "duration_s": 5.0, "vx": 0.2}],
        }
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario(cfg, source="unit_test")
        self.assertIn("plan", str(ctx.exception))

    def test_plan_gate_rejects_a_run_that_reached_no_subgoal(self):
        """The plan gate must fail a base that never left the origin."""
        cfg = load_scenario(os.path.join(SCENARIOS_DIR, "plan_goto_then_dock.json"))
        metrics = compute_metrics(
            make_history(n=200, x_step=0.0),
            cfg["expected_metrics"],
            duration_s=cfg["duration_s"],
            plan={
                "plan_skill_count": 3,
                "plan_skills_completed": 0,
                "plan_waypoints_reached": 0,
            },
        )
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any("Plan skills completed" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )
        self.assertTrue(
            any("Plan waypoints reached" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_plan_gate_rejects_a_standoff_that_is_too_far(self):
        """Reaching the goals is not enough: the standoff must hold too."""
        cfg = load_scenario(os.path.join(SCENARIOS_DIR, "plan_goto_then_dock.json"))
        metrics = compute_metrics(
            make_history(n=200, x_step=0.0),
            cfg["expected_metrics"],
            duration_s=cfg["duration_s"],
            plan={
                "plan_skill_count": 3,
                "plan_skills_completed": 3,
                "plan_waypoints_reached": 2,
                "final_goal_distance_m": 0.9,
            },
        )
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            any("Final distance to the last subgoal" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_plan_gate_passes_a_complete_run(self):
        """A run that completed every subgoal and held the standoff passes."""
        cfg = load_scenario(os.path.join(SCENARIOS_DIR, "plan_goto_then_dock.json"))
        metrics = compute_metrics(
            make_history(n=200, x_step=0.0),
            cfg["expected_metrics"],
            duration_s=cfg["duration_s"],
            plan={
                "plan_skill_count": 3,
                "plan_skills_completed": 3,
                "plan_waypoints_reached": 2,
                "final_goal_distance_m": 0.09,
            },
        )
        # The path-length gate is not satisfied by a synthetic stationary
        # history, so only the plan-specific gates are asserted here.
        self.assertFalse(
            any("Plan" in r for r in metrics["failure_reasons"]),
            metrics["failure_reasons"],
        )

    def test_plan_arm_and_direct_arm_visit_the_same_markers(self):
        """The cost-of-planning pair must be the *same* route on both arms.

        `plan_waypoint_square` decomposes the four-corner route into four
        plan skills; `waypoint_square` issues the same four markers as one
        continuous waypoint list.  The difference in cost of transport
        between them is the cost of the decomposition, so the two scenarios
        are only comparable if they ask for the same markers, at the same
        tolerance and speed, and are gated on the same path length.  A
        future edit that changes one arm and not the other would silently
        turn the comparison into a comparison of two different routes.
        """
        plan_cfg = load_scenario(os.path.join(SCENARIOS_DIR, "plan_waypoint_square.json"))
        direct_cfg = load_scenario(os.path.join(SCENARIOS_DIR, "waypoint_square.json"))

        plan_markers = [
            (float(s["args"]["x"]), float(s["args"]["y"]))
            for s in plan_cfg["plan"]
            if s["skill"] == "goto_waypoint"
        ]
        direct_markers = [
            (float(w["x"]), float(w["y"])) for w in direct_cfg["waypoints"]
        ]
        self.assertEqual(plan_markers, direct_markers)

        plan_tols = sorted(
            float(s["args"].get("tolerance", 0.15))
            for s in plan_cfg["plan"]
            if s["skill"] == "goto_waypoint"
        )
        direct_tols = sorted(float(w["tolerance"]) for w in direct_cfg["waypoints"])
        self.assertEqual(plan_tols, direct_tols)

        plan_speeds = sorted(
            float(s["args"].get("speed", 0.30))
            for s in plan_cfg["plan"]
            if s["skill"] == "goto_waypoint"
        )
        direct_speeds = sorted(float(w["speed"]) for w in direct_cfg["waypoints"])
        self.assertEqual(plan_speeds, direct_speeds)

        # The same gate, so neither arm is advantaged by a laxer threshold.
        self.assertEqual(
            plan_cfg["expected_metrics"]["min_path_length_m"],
            direct_cfg["expected_metrics"]["min_path_length_m"],
        )
        # The same duration, so neither arm is given more wall-clock.
        self.assertEqual(plan_cfg["duration_s"], direct_cfg["duration_s"])
