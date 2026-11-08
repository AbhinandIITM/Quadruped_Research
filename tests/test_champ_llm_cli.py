"""Unit tests for the real-time hygiene helpers of go2_champ_llm_cli.

The CLI owns the MuJoCo stepping loop of the project, where every control tick
must fit a 2 ms budget (dt = 0.002 s, 500 Hz). These tests cover the pieces that
keep the loop inside that budget without launching a simulation:

- step/period decimation arithmetic (`steps_for_period`)
- absolute-deadline pacing (`pacing_sleep_seconds`)
- off-thread terminal telemetry (`BackgroundPrinter`)
- the measured vision time budget and its failure accounting (`VisionScheduler`)
- incremental live-plot buffers and the on-exit safe hold
"""

import contextlib
import io
import os
import queue
import subprocess
import sys
import types
import unittest
from unittest import mock

import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg", force=True)

import go2_champ_llm_cli as cli  # noqa: E402  (import after matplotlib backend selection)
from go2_champ_llm_cli import (  # noqa: E402
    LIVE_PLOT_PERIOD_S,
    TELEMETRY_PRINT_PERIOD_S,
    VIEWER_SYNC_PERIOD_S,
    VISION_PERIOD_S,
    BackgroundPrinter,
    Go2ChampSimulationRunner,
    VisionScheduler,
    pacing_sleep_seconds,
    steps_for_period,
)

DT = 0.002  # model.opt.timestep, the per-tick budget denominator


class TestModelServerToken(unittest.TestCase):
    """`--token` reaches the client; omitting it keeps the env-var fallback."""

    def _select(self, **kwargs):
        with mock.patch.object(cli, "LiveModelServiceClient") as client_cls:
            client_cls.return_value.check_availability.return_value = (True, "ok")
            with contextlib.redirect_stdout(io.StringIO()):
                cli.select_live_llm_client(backend="server", **kwargs)
        return client_cls

    def test_explicit_token_is_passed_through(self):
        client_cls = self._select(server_url="http://127.0.0.1:5005", auth_token="shhh")
        client_cls.assert_called_once_with(
            server_url="http://127.0.0.1:5005", auth_token="shhh"
        )

    def test_omitted_token_defers_to_the_client_env_fallback(self):
        # None (not "") must be forwarded so LiveModelServiceClient keeps applying
        # its MODEL_SERVER_TOKEN fallback and the tokenless local flow is untouched.
        client_cls = self._select(server_url="http://127.0.0.1:5005")
        self.assertIsNone(client_cls.call_args.kwargs["auth_token"])

    def test_token_flag_is_declared_and_help_exits_zero(self):
        # --help short-circuits in argparse before any backend is contacted.
        argv_backup = sys.argv
        sys.argv = ["go2_champ_llm_cli.py", "--help"]
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit) as ctx:
                cli.main()
        finally:
            sys.argv = argv_backup
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--token", buffer.getvalue())


class TestPathConstants(unittest.TestCase):
    """Model and log locations are named constants with env-var overrides."""

    def test_defaults_are_unchanged(self):
        self.assertEqual(
            cli.DEFAULT_MODEL_PATH, os.path.join(ROOT_DIR, "unitree_go2", "go2_description.xml")
        )
        self.assertEqual(cli.DEFAULT_LOG_DIR, os.path.join(ROOT_DIR, "logs", "llm_runs"))
        self.assertEqual(cli.MODEL_PATH, cli.DEFAULT_MODEL_PATH)
        self.assertEqual(cli.LOG_DIR, cli.DEFAULT_LOG_DIR)

    def test_constants_are_paths(self):
        self.assertIsInstance(cli.MODEL_PATH, str)
        self.assertIsInstance(cli.LOG_DIR, str)

    def test_environment_overrides_redirect_the_paths(self):
        # Module-level env reads can only be re-evaluated in a fresh interpreter.
        env = dict(os.environ, GO2_MODEL_PATH="override_model.xml", GO2_LOG_DIR="override_logs")
        src_path = os.path.join(ROOT_DIR, "src")
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pp}" if existing_pp else src_path
        code = (
            "import matplotlib; matplotlib.use('Agg');"
            "import go2_champ_llm_cli as c;"
            "print(c.MODEL_PATH); print(c.LOG_DIR)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        model_path, log_dir = result.stdout.split()
        self.assertEqual(model_path, "override_model.xml")
        self.assertEqual(log_dir, "override_logs")


class TestStepDecimation(unittest.TestCase):
    """Periodic work must be decimated in whole control steps, never per tick."""

    def test_model_timestep_periods(self):
        self.assertEqual(steps_for_period(TELEMETRY_PRINT_PERIOD_S, DT), 250)
        self.assertEqual(steps_for_period(VISION_PERIOD_S, DT), 25)
        self.assertEqual(steps_for_period(LIVE_PLOT_PERIOD_S, DT), 100)
        self.assertEqual(steps_for_period(VIEWER_SYNC_PERIOD_S, DT), 10)

    def test_plot_is_decimated_far_below_tick_rate(self):
        # Live plotting must run at a fixed low rate, not once per 500 Hz tick.
        self.assertGreater(steps_for_period(LIVE_PLOT_PERIOD_S, DT), 1)
        self.assertLess(1.0 / (steps_for_period(LIVE_PLOT_PERIOD_S, DT) * DT), 500.0)

    def test_period_never_collapses_to_zero_steps(self):
        self.assertEqual(steps_for_period(1e-9, DT), 1)
        self.assertEqual(steps_for_period(0.0, DT), 1)

    def test_degenerate_timestep_yields_single_step(self):
        self.assertEqual(steps_for_period(0.5, 0.0), 1)
        self.assertEqual(steps_for_period(0.5, -0.001), 1)


class TestPacing(unittest.TestCase):
    """GUI pacing is driven by an absolute deadline, outside the control work."""

    def test_sleeps_until_remaining_slack(self):
        self.assertAlmostEqual(pacing_sleep_seconds(10.0, 9.75), 0.25, places=6)

    def test_never_sleeps_when_already_late(self):
        self.assertEqual(pacing_sleep_seconds(10.0, 10.0), 0.0)
        self.assertEqual(pacing_sleep_seconds(10.0, 12.5), 0.0)

    def test_slow_tick_does_not_compound_into_drift(self):
        # A tick that overran its deadline must consume no sleep at all, so the
        # next deadline is computed from the absolute reference and the sim clock
        # cannot drift further away from wall time.
        reference = 100.0
        late_now = reference + 5 * DT + 0.050  # 50 ms of overrun on tick 5
        self.assertEqual(pacing_sleep_seconds(reference + 5 * DT, late_now), 0.0)

    def test_on_schedule_tick_sleeps_the_full_step(self):
        reference = 100.0
        deadline = reference + 7 * DT
        self.assertAlmostEqual(
            pacing_sleep_seconds(deadline, reference + 6 * DT), DT, places=9
        )


class TestBackgroundPrinter(unittest.TestCase):
    """Terminal telemetry must not perform blocking console I/O on a tick."""

    def test_lines_are_written_in_order(self):
        printer = BackgroundPrinter()
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                for idx in range(5):
                    printer.submit(f"line {idx}")
                printer.close()
        finally:
            printer.close()
        self.assertEqual(
            buffer.getvalue().splitlines(), [f"line {idx}" for idx in range(5)]
        )

    def test_close_without_submissions_is_safe(self):
        printer = BackgroundPrinter()
        printer.close()
        printer.close()

    def test_submit_never_blocks_when_the_writer_stalls(self):
        # Simulate a writer thread that never drains: a blocking submit would hang
        # this test forever, and an unbounded queue would grow without limit.
        printer = BackgroundPrinter.__new__(BackgroundPrinter)
        printer._queue = queue.Queue(maxsize=1)
        printer._thread = None

        for idx in range(100):
            printer.submit(f"line {idx}")

        self.assertEqual(printer._queue.qsize(), 1)
        self.assertEqual(printer._queue.get_nowait(), "line 99")


class TestVisionScheduler(unittest.TestCase):
    """Vision runs on a measured budget, not on a raw modulo."""

    def test_runs_at_the_nominal_period(self):
        scheduler = VisionScheduler(dt=DT)
        self.assertEqual(scheduler.period_steps, 25)

        self.assertTrue(scheduler.due(0))
        for step in range(1, 25):
            self.assertFalse(scheduler.due(step), f"step {step} should not be due")
        self.assertTrue(scheduler.due(25))

    def test_default_budget_is_a_fraction_of_a_tick(self):
        scheduler = VisionScheduler(dt=DT)
        self.assertLess(scheduler.budget_s, DT * 5)
        self.assertGreater(scheduler.budget_s, 0.0)

    def test_vision_is_skipped_when_the_tick_has_no_slack(self):
        scheduler = VisionScheduler(dt=DT)

        # Deadline already blown for this tick: skip, and record that we skipped.
        self.assertFalse(scheduler.due(0, deadline=1.0, now=1.5))
        self.assertEqual(scheduler.skips, 1)

        # The skipped attempt must not consume the period: the very next tick with
        # slack can still run, so a busy loop degrades vision rather than the tick.
        self.assertTrue(scheduler.due(1, deadline=2.0, now=1.9))
        self.assertEqual(scheduler.skips, 1)

    def test_over_budget_run_backs_the_period_off(self):
        scheduler = VisionScheduler(dt=DT)
        scheduler.due(0)
        scheduler.record(DT * 100)  # far slower than the tick budget

        self.assertEqual(scheduler.overruns, 1)
        self.assertGreater(scheduler.period_steps, 25)
        self.assertGreater(scheduler.last_duration_s, scheduler.budget_s)

    def test_period_recovers_towards_the_nominal_rate(self):
        scheduler = VisionScheduler(dt=DT)
        scheduler.due(0)
        scheduler.record(10.0)  # back off hard, repeatedly
        for _ in range(10):
            scheduler.record(10.0)
        self.assertEqual(scheduler.period_steps, scheduler.max_period_steps)

        for _ in range(10):
            scheduler.record(1e-6)  # fast runs pull the period back down
        self.assertEqual(scheduler.period_steps, scheduler.min_period_steps)

    def test_backoff_is_capped(self):
        scheduler = VisionScheduler(dt=DT, max_period_s=0.2)
        for _ in range(50):
            scheduler.record(1.0)
        self.assertEqual(scheduler.period_steps, 100)

    def test_failures_are_counted(self):
        scheduler = VisionScheduler(dt=DT)
        self.assertEqual(scheduler.failures, 0)
        self.assertEqual(scheduler.record_failure(), 1)
        self.assertEqual(scheduler.record_failure(), 2)
        self.assertEqual(scheduler.failures, 2)


class TestLivePlotBuffers(unittest.TestCase):
    """The plot is fed incrementally instead of rebuilding the whole history."""

    @staticmethod
    def _record(time_s: float) -> dict:
        return {
            "time_s": time_s,
            "base_x_m": time_s * 1.0,
            "base_y_m": time_s * 2.0,
            "cmd_vx": 0.3,
            "base_vx_body_mps": 0.29,
            "cmd_vy": 0.0,
            "base_vy_body_mps": 0.01,
            "cmd_yaw": 0.1,
            "base_wz_radps": 0.09,
        }

    def test_only_new_samples_are_appended(self):
        buffers = {
            key: []
            for key in ("t", "x", "y", "cmd_vx", "obs_vx", "cmd_vy", "obs_vy", "cmd_wz", "obs_wz")
        }
        history = [self._record(i * 0.008) for i in range(5)]

        cursor = Go2ChampSimulationRunner._extend_plot_buffers(buffers, history, 0)
        self.assertEqual(cursor, 5)
        self.assertEqual(buffers["t"], [r["time_s"] for r in history])
        self.assertEqual(buffers["obs_vx"], [0.29] * 5)

        # A redraw with no new samples must neither duplicate nor re-scan history.
        cursor = Go2ChampSimulationRunner._extend_plot_buffers(buffers, history, cursor)
        self.assertEqual(len(buffers["t"]), 5)

        history.append(self._record(0.040))
        Go2ChampSimulationRunner._extend_plot_buffers(buffers, history, cursor)
        self.assertEqual(len(buffers["t"]), 6)
        self.assertEqual(buffers["t"][-1], 0.040)


class TestExitSafeHold(unittest.TestCase):
    """Every exit path must leave the actuators in a defined state."""

    def test_hold_commands_the_measured_joint_positions(self):
        runner = Go2ChampSimulationRunner.__new__(Go2ChampSimulationRunner)
        runner.qpos_addrs = [0, 1, 2]
        runner.actuator_ids = [0, 1, 2]
        runner.model = object()
        runner.data = types.SimpleNamespace(
            qpos=np.array([0.1, 0.2, 0.3]), ctrl=np.array([9.9, 9.9, 9.9])
        )

        with mock.patch.object(cli.mujoco, "mj_step") as step:
            runner.hold_current_joints()

        np.testing.assert_allclose(runner.data.ctrl, [0.1, 0.2, 0.3])
        self.assertTrue(step.called)

    def test_hold_failure_never_propagates(self):
        runner = Go2ChampSimulationRunner.__new__(Go2ChampSimulationRunner)
        runner.qpos_addrs = []
        runner.actuator_ids = []
        runner.model = object()
        runner.data = types.SimpleNamespace(qpos=np.zeros(0), ctrl=np.zeros(0))

        with mock.patch.object(cli.mujoco, "mj_step", side_effect=RuntimeError("boom")):
            runner.hold_current_joints()  # must not raise


if __name__ == "__main__":
    unittest.main()
