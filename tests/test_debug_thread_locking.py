"""
test_debug_thread_locking.py
============================
Regression tests for the debug-plot / simulation hand-off in
``go2_mujoco_trot.py`` and ``go2_mujoco_trot_traj.py``.

Both scripts run a matplotlib daemon thread next to the simulation loop. The
loop mutates the shared ``mujoco.MjData`` in place (``mj_step``, ``data.ctrl``
writes) while that thread reads shared state, so every access on both sides has
to go through the single ``debug_lock``; ``MjData`` is a mutable C struct and
unsynchronised concurrent access is undefined behaviour, not just a style
issue. The lock used to exist but was not held by the sim side at all, so the
reader could observe a partially integrated state.

The scripts are top-level programs - importing one would open the viewer and
step forever - so each is loaded here with the driver stubbed out
(``mujoco.viewer.launch_passive`` returns an already-stopped viewer and
``threading.Thread.start`` does nothing). That leaves the module body, its
helpers and its shutdown path importable and exercisable with no display and
without running a single simulation loop iteration.
"""

import importlib.util
import os
import sys
import threading
import time
import unittest
from unittest import mock

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg", force=True)

SCRIPTS = ("go2_mujoco_trot.py", "go2_mujoco_trot_traj.py")

# Keys publish_debug_sample() always writes; the reader must never see a
# snapshot with a subset of them (i.e. a half-written sample).
SAMPLE_KEYS = {"t", "vx_cmd", "vy_cmd", "w_cmd", "vx_est", "vy_est", "w_est", "fr_cmd", "fr_est"}


class _StoppedViewer:
    """Stand-in for ``mujoco.viewer.launch_passive()``: never running, never draws."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def is_running(self):
        return False

    def sync(self):
        pass


def _load_script(filename):
    """Import one of the trot scripts without starting the simulation or a GUI."""
    path = os.path.join(ROOT_DIR, "experiments", filename)
    if not os.path.exists(path):
        path = os.path.join(ROOT_DIR, filename)
    stem = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(stem + "_under_test", path)
    module = importlib.util.module_from_spec(spec)

    cwd = os.getcwd()
    os.chdir(str(ROOT_DIR))  # MODEL_PATH is relative to the repo root
    try:
        with mock.patch("mujoco.viewer.launch_passive", return_value=_StoppedViewer()), \
                mock.patch.object(threading.Thread, "start", lambda self: None):
            spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


class _LockProbe:
    """Stands in for ``debug_lock`` and records every ``with debug_lock:`` entered.

    ``held`` is sampled *inside* the block, so a non-True entry means the code
    only pretended to take the lock.
    """

    def __init__(self, real_lock):
        self._real = real_lock
        self.entries = 0
        self.held = []

    def __enter__(self):
        value = self._real.__enter__()
        self.entries += 1
        self.held.append(self._real.locked())
        return value

    def __exit__(self, *exc_info):
        return self._real.__exit__(*exc_info)


class _ScriptContract:
    """Locking contract exercised identically against both scripts."""

    SCRIPT = None
    STEP_FUNC = None

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script(cls.SCRIPT)

    def setUp(self):
        self.mod.latest_debug_sample = None
        self.mod.debug_stop_event.clear()

    def _publish(self, t=1.0):
        self.mod.publish_debug_sample(t, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0)

    # ---- the sim side --------------------------------------------------

    def test_step_runs_inside_debug_lock(self):
        """mj_step mutates MjData in place, so it must run under debug_lock."""
        step = getattr(self.mod, self.STEP_FUNC)
        observed = []
        real_mj_step = self.mod.mujoco.mj_step

        def spy(model, data):
            observed.append(self.mod.debug_lock.locked())
            return real_mj_step(model, data)

        with mock.patch.object(self.mod.mujoco, "mj_step", spy):
            step(None)

        self.assertTrue(observed, "mj_step was never reached")
        self.assertTrue(all(observed), "mj_step ran without debug_lock held")
        self.assertFalse(self.mod.debug_lock.locked(), "debug_lock was not released after the step")

    def test_control_writes_run_inside_debug_lock(self):
        probe = _LockProbe(self.mod.debug_lock)
        with mock.patch.object(self.mod, "debug_lock", probe):
            self.mod.command_joint_positions(list(self.mod.standing_q))
        self.assertEqual(probe.entries, 1)
        self.assertTrue(all(probe.held))

    def test_state_reads_and_publishes_hold_debug_lock(self):
        probe = _LockProbe(self.mod.debug_lock)
        with mock.patch.object(self.mod, "debug_lock", probe):
            self.mod.get_estimated_body_velocities()
            self.assertEqual(probe.entries, 1)
            self._publish()
            self.assertEqual(probe.entries, 2)
            self.mod.take_debug_sample()
            self.assertEqual(probe.entries, 3)
        self.assertTrue(all(probe.held))

    # ---- the reader side -----------------------------------------------

    def test_snapshot_is_an_independent_copy(self):
        self.assertIsNone(self.mod.take_debug_sample())
        self._publish(t=1.0)

        sample = self.mod.take_debug_sample()
        self.assertEqual(set(sample), SAMPLE_KEYS)

        sample["t"] = 999.0  # "drawing" mutates the copy, never the shared slot
        self.assertEqual(self.mod.take_debug_sample()["t"], 1.0)

    def test_reader_never_observes_a_partially_written_sample(self):
        stop = threading.Event()
        problems = []
        seen = []

        def writer():
            i = 0
            while not stop.is_set():
                i += 1
                self._publish(t=float(i))

        def reader():
            count = 0
            while not stop.is_set():
                sample = self.mod.take_debug_sample()
                if sample is None:
                    continue
                count += 1
                if set(sample) != SAMPLE_KEYS:
                    problems.append(sorted(sample))
                if not all(isinstance(v, float) for v in sample.values()):
                    problems.append(sorted(sample))
                time.sleep(0.0002)
            seen.append(count)

        threads = [threading.Thread(target=writer, daemon=True),
                   threading.Thread(target=reader, daemon=True)]
        for thread in threads:
            thread.start()
        time.sleep(0.2)
        stop.set()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(problems, [], "reader saw a partial or non-plain-data sample")
        self.assertGreater(seen[0], 0, "reader never got a sample")
        self.assertFalse(any(thread.is_alive() for thread in threads))

    def test_plot_worker_snapshots_under_lock_and_stops_on_the_event(self):
        """The real plot loop: locks only to copy, and stops when signalled."""
        mod = self.mod
        probe = _LockProbe(mod.debug_lock)
        worker_errors = []
        figures_before = len(mod.plt.get_fignums())

        def run_worker():
            try:
                mod.debug_plot_worker()
            except BaseException as exc:  # surfaced below instead of dying silently
                worker_errors.append(exc)

        with mock.patch.object(mod, "debug_lock", probe):
            worker = threading.Thread(target=run_worker, daemon=True)
            worker.start()
            try:
                deadline = time.time() + 15.0
                while probe.entries == 0 and time.time() < deadline:
                    time.sleep(0.01)
                self._publish(t=1.0)
                time.sleep(0.2)
            finally:
                mod.debug_stop_event.set()
                worker.join(timeout=5.0)

        self.assertEqual(worker_errors, [])
        self.assertGreater(probe.entries, 0, "plot thread never took a snapshot through debug_lock")
        self.assertTrue(all(probe.held))
        self.assertFalse(worker.is_alive(), "plot thread ignored debug_stop_event")
        self.assertEqual(len(mod.plt.get_fignums()), figures_before,
                         "plot thread did not close its figure")


class TestGo2MujocoTrotLocking(_ScriptContract, unittest.TestCase):
    SCRIPT = "go2_mujoco_trot.py"
    STEP_FUNC = "step_with_viewer"


class TestGo2MujocoTrotTrajLocking(_ScriptContract, unittest.TestCase):
    SCRIPT = "go2_mujoco_trot_traj.py"
    STEP_FUNC = "step_sim"


if __name__ == "__main__":
    unittest.main()
