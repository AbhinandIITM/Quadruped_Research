"""
test_multi_gait_transitions.py
==============================
Regression tests for the multi-gait scheduler's in-flight transition engine,
in particular that an *interrupted* transition continues from the current
blended state instead of restarting from the last completed gait.
"""

import unittest
import numpy as np

from go2_multi_gait import (
    GaitType,
    MultiGaitScheduler,
    GAIT_PRESETS,
)


# Largest difference between any two presets, used to bound the per-step
# change that a continuous blend may produce.
_MAX_STANCE_SPAN = max(p.stance_duration for p in GAIT_PRESETS.values()) - \
    min(p.stance_duration for p in GAIT_PRESETS.values())
_MAX_SWING_SPAN = max(p.swing_duration for p in GAIT_PRESETS.values()) - \
    min(p.swing_duration for p in GAIT_PRESETS.values())


def _run(scheduler, duration, dt=0.002, is_moving=True):
    """Steps the scheduler for `duration` seconds and returns the phase samples."""
    samples = []
    for _ in range(int(round(duration / dt))):
        stance, swing = scheduler.step(dt, is_moving=is_moving)
        samples.append((stance, swing))
    return samples


class TestInterruptedTransitionContinuity(unittest.TestCase):

    def setUp(self):
        self.scheduler = MultiGaitScheduler(initial_gait=GaitType.TROT)

    def test_interruption_does_not_jump_the_blended_durations(self):
        """A new set_gait mid-transition must not snap back to the last completed gait."""
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)

        sched.set_gait(GaitType.PRONK, transition_duration=1.0)
        _run(sched, 0.5, dt)                      # half-way through the transition
        self.assertGreater(sched.transition_progress, 0.0)
        self.assertLess(sched.transition_progress, 1.0)

        blended_stance = sched.stance_duration
        blended_swing = sched.swing_duration
        # The blended state really is part-way between the two presets.
        self.assertNotAlmostEqual(blended_stance, GAIT_PRESETS[GaitType.TROT].stance_duration, places=3)

        # Interrupt with a different target.
        self.assertTrue(sched.set_gait(GaitType.WALK, transition_duration=1.0))
        # set_gait itself must not perturb the state.
        self.assertAlmostEqual(sched.stance_duration, blended_stance, places=12)
        self.assertAlmostEqual(sched.swing_duration, blended_swing, places=12)

        # The first steps after the interruption must continue from that blended
        # state: the per-step change can be at most rate * dt * (target - source),
        # whereas restarting from TROT would jump by ~0.02 s in a single step.
        prev_stance = blended_stance
        prev_swing = blended_swing
        max_step_stance = 0.0
        max_step_swing = 0.0
        for _ in range(50):
            sched.step(dt)
            max_step_stance = max(max_step_stance, abs(sched.stance_duration - prev_stance))
            max_step_swing = max(max_step_swing, abs(sched.swing_duration - prev_swing))
            prev_stance = sched.stance_duration
            prev_swing = sched.swing_duration

        rate = 1.0 / 1.0   # transition_duration = 1.0 s
        self.assertLessEqual(max_step_stance, 2.0 * rate * dt * _MAX_STANCE_SPAN)
        self.assertLessEqual(max_step_swing, 2.0 * rate * dt * _MAX_SWING_SPAN)

    def test_interruption_does_not_jump_the_blended_offsets(self):
        """Phase offsets are likewise blended from the interrupted state."""
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)

        sched.set_gait(GaitType.BOUND, transition_duration=1.0)
        _run(sched, 0.5, dt)
        blended_offsets = sched.current_offsets.copy()

        sched.set_gait(GaitType.WALK, transition_duration=1.0)
        np.testing.assert_allclose(sched.current_offsets, blended_offsets, atol=1e-12)

        prev = blended_offsets
        max_step = 0.0
        for _ in range(50):
            sched.step(dt)
            # Shortest-path modular distance between consecutive offset vectors.
            delta = np.abs(sched.current_offsets - prev)
            max_step = max(max_step, float(np.minimum(delta, 1.0 - delta).max()))
            prev = sched.current_offsets.copy()

        # 0.5 is the largest possible modular distance, 1.0 is the transition rate.
        self.assertLessEqual(max_step, 2.0 * dt * 0.5)

    def test_interrupted_transition_still_reaches_the_new_target(self):
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)
        sched.set_gait(GaitType.PRONK, transition_duration=0.4)
        _run(sched, 0.2, dt)
        sched.set_gait(GaitType.WALK, transition_duration=0.4)

        # Run long enough for the interrupted transition to complete.
        for _ in range(int(0.5 / dt) + 5):
            stance, swing = sched.step(dt)
            self.assertTrue(all(np.isfinite(stance)) and all(np.isfinite(swing)))

        self.assertEqual(sched.current_gait, GaitType.WALK)
        self.assertEqual(sched.target_gait, GaitType.WALK)
        self.assertGreaterEqual(sched.transition_progress, 1.0)
        self.assertAlmostEqual(sched.stance_duration,
                               GAIT_PRESETS[GaitType.WALK].stance_duration, places=9)
        self.assertAlmostEqual(sched.swing_duration,
                               GAIT_PRESETS[GaitType.WALK].swing_duration, places=9)
        np.testing.assert_allclose(sched.current_offsets,
                                   np.array(GAIT_PRESETS[GaitType.WALK].phase_offsets), atol=1e-12)

    def test_uninterrupted_transition_is_continuous(self):
        """Continuity in the plain (non-interrupted) case must be preserved."""
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)

        sched.set_gait(GaitType.WALK, transition_duration=0.5)
        prev_stance = sched.stance_duration
        prev_swing = sched.swing_duration
        max_step_stance = 0.0
        max_step_swing = 0.0
        for _ in range(250):
            sched.step(dt)
            max_step_stance = max(max_step_stance, abs(sched.stance_duration - prev_stance))
            max_step_swing = max(max_step_swing, abs(sched.swing_duration - prev_swing))
            prev_stance = sched.stance_duration
            prev_swing = sched.swing_duration

        rate = 1.0 / 0.5
        self.assertLessEqual(max_step_stance, 2.0 * rate * dt * _MAX_STANCE_SPAN)
        self.assertLessEqual(max_step_swing, 2.0 * rate * dt * _MAX_SWING_SPAN)
        self.assertEqual(sched.current_gait, GaitType.WALK)

    def test_repeated_set_gait_is_idempotent(self):
        """A repeated request for the in-flight target must not restart the blend."""
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)

        sched.set_gait(GaitType.WALK, transition_duration=1.0)
        _run(sched, 0.3, dt)
        progress_before = sched.transition_progress
        stance_before = sched.stance_duration

        self.assertTrue(sched.set_gait(GaitType.WALK, transition_duration=1.0))
        self.assertAlmostEqual(sched.transition_progress, progress_before, places=12)
        self.assertAlmostEqual(sched.stance_duration, stance_before, places=12)

        sched.step(dt)
        self.assertGreater(sched.transition_progress, progress_before)

    def test_blending_does_not_compound_across_a_repeat_of_the_same_request(self):
        """Holding set_gait on the same target must not drift the durations."""
        sched = self.scheduler
        dt = 0.002
        _run(sched, 0.25, dt)
        sched.set_gait(GaitType.WALK, transition_duration=1.0)
        _run(sched, 0.25, dt)

        samples = []
        for _ in range(100):
            sched.set_gait(GaitType.WALK, transition_duration=1.0)
            sched.step(dt)
            samples.append(sched.stance_duration)

        # Monotone ramp toward the target, bounded by the blend rate.
        for before, after in zip(samples, samples[1:]):
            self.assertLessEqual(after - before, 2.0 * dt * _MAX_STANCE_SPAN)
            self.assertGreaterEqual(after, before - 1e-12)


class TestPhaseSweepHasNoDeadTick(unittest.TestCase):
    """The phase normalisation must never yield a tick where both phases are 0.

    A dead tick makes ChampTrajectoryPlanner.generate fall through to its
    degenerate (0, 0) branch, snapping a swing foot from its stance-end position
    back to the nominal centre for one tick.
    """

    SAMPLES = 2000
    STEP_LENGTH = 0.16

    @staticmethod
    def _planner():
        # Imported lazily: go2_champ_control is owned by another agent and only
        # the continuity assertions below need it.
        from go2_champ_control import ChampTrajectoryPlanner
        return ChampTrajectoryPlanner(swing_height=0.06, stance_depth=0.0)

    def _sweep(self, gait):
        """Samples exact stride times over one full stride, leg by leg."""
        scheduler = MultiGaitScheduler(initial_gait=gait)
        stride = scheduler.stride_period
        out = []
        for k in range(self.SAMPLES):
            t = k * stride / self.SAMPLES
            scheduler.phase_time = t     # sample the same step() interface at exact t
            out.append((t, scheduler.step(0.0)))
        return out

    def test_no_dead_ticks_across_a_full_stride(self):
        for gait in GaitType:
            max_stance = 0.0
            max_swing = 0.0
            for t, (stance, swing) in self._sweep(gait):
                for leg in range(4):
                    self.assertFalse(
                        stance[leg] == 0.0 and swing[leg] == 0.0,
                        f"both phases zero for {gait.value} at t={t:.6f}, leg={leg}",
                    )
                    self.assertTrue(0.0 <= stance[leg] <= 1.0)
                    self.assertTrue(0.0 <= swing[leg] <= 1.0)
                    # Exactly one phase is active on every tick.
                    if stance[leg] > 0.0:
                        self.assertEqual(swing[leg], 0.0)
                    else:
                        self.assertEqual(stance[leg], 0.0)
                    max_stance = max(max_stance, stance[leg])
                    max_swing = max(max_swing, swing[leg])
            # Both phases still span (almost) the full [0, 1] range over a stride.
            self.assertGreater(max_stance, 0.99, msg=gait.value)
            self.assertGreater(max_swing, 0.99, msg=gait.value)

    def test_accumulated_stepping_has_no_dead_tick(self):
        """The live accumulation path (phase_time += dt) must be clean too."""
        dt = 0.002
        for gait in GaitType:
            scheduler = MultiGaitScheduler(initial_gait=gait)
            ticks = int(3 * scheduler.stride_period / dt)
            for _ in range(ticks):
                stance, swing = scheduler.step(dt)
                for leg in range(4):
                    self.assertFalse(
                        stance[leg] == 0.0 and swing[leg] == 0.0,
                        f"dead tick for {gait.value}, leg={leg}",
                    )

    def test_no_dead_tick_while_a_transition_is_in_flight(self):
        """Blended durations must satisfy the same invariant."""
        scheduler = MultiGaitScheduler(initial_gait=GaitType.TROT)
        scheduler.set_gait(GaitType.WALK, transition_duration=0.4)
        for _ in range(int(0.6 / 0.002)):
            stance, swing = scheduler.step(0.002)
            for leg in range(4):
                self.assertFalse(
                    stance[leg] == 0.0 and swing[leg] == 0.0,
                    f"dead tick during transition, leg={leg}",
                )

    def test_wrap_point_maps_to_end_of_stance(self):
        """leg_clock == -swing is mapped to end of stance, not to swing_phase 0.0.

        Mapping it into swing would give stance_phase == swing_phase == 0.0, the
        planner's degenerate branch. End of stance renders the same rear extreme
        (x = -L/2) as the first swing tick, so the trajectory stays continuous.
        """
        scheduler = MultiGaitScheduler(initial_gait=GaitType.TROT)
        # Leg 0 (FR) has offset 0.0, so its wrap point is at stride_time == stance.
        scheduler.phase_time = scheduler.stance_duration
        stance, swing = scheduler.step(0.0)

        self.assertAlmostEqual(stance[0], 1.0, places=9)
        self.assertEqual(swing[0], 0.0)

        planner = self._planner()
        L = self.STEP_LENGTH
        x_stance_end, _ = planner.generate(L, swing[0], stance[0])
        x_first_swing, _ = planner.generate(L, 1e-9, 0.0)

        self.assertAlmostEqual(x_stance_end, -L / 2.0, places=6)
        self.assertAlmostEqual(x_stance_end, x_first_swing, places=3)
        # ... and it is nowhere near the degenerate nominal-centre snap.
        self.assertGreater(abs(x_stance_end - 0.0), 0.05)

    def test_planner_output_is_continuous_across_the_sweep(self):
        """No tick may jump by anything like the dead-tick snap (~0.16 m)."""
        planner = self._planner()
        for gait in GaitType:
            prev_x = None
            max_jump = 0.0
            for _t, (stance, swing) in self._sweep(gait):
                x, _z = planner.generate(self.STEP_LENGTH, swing[0], stance[0])
                if prev_x is not None:
                    max_jump = max(max_jump, abs(x - prev_x))
                prev_x = x
            # Observed ~4e-4 m for a continuous sweep; a dead tick snaps ~0.08 m.
            self.assertLess(max_jump, 0.01, msg=f"discontinuity in {gait.value}: {max_jump:.6f} m")


class TestSchedulerStatePreservation(unittest.TestCase):

    def test_reset_restores_the_initial_gait(self):
        sched = MultiGaitScheduler(initial_gait=GaitType.PACE)
        for _ in range(50):
            sched.step(0.002)
        sched.set_gait(GaitType.WALK, transition_duration=0.4)
        for _ in range(25):
            sched.step(0.002)

        sched.reset()

        pace = GAIT_PRESETS[GaitType.PACE]
        self.assertEqual(sched.current_gait, GaitType.PACE)
        self.assertEqual(sched.profile, pace)
        self.assertEqual(sched.phase_time, 0.0)
        self.assertAlmostEqual(sched.transition_progress, 1.0)
        self.assertAlmostEqual(sched.stance_duration, pace.stance_duration, places=12)
        self.assertAlmostEqual(sched.swing_duration, pace.swing_duration, places=12)
        np.testing.assert_allclose(sched.current_offsets, np.array(pace.phase_offsets), atol=1e-12)

        # Still usable and continuous after a reset.
        for _ in range(25):
            stance, swing = sched.step(0.002)
            self.assertTrue(all(np.isfinite(stance)) and all(np.isfinite(swing)))

    def test_set_gait_to_current_gait_is_a_noop(self):
        sched = MultiGaitScheduler(initial_gait=GaitType.TROT)
        for _ in range(25):
            sched.step(0.002)

        self.assertTrue(sched.set_gait(GaitType.TROT))
        self.assertEqual(sched.target_gait, GaitType.TROT)
        self.assertAlmostEqual(sched.transition_progress, 1.0)
        self.assertAlmostEqual(sched.stance_duration,
                               GAIT_PRESETS[GaitType.TROT].stance_duration, places=12)


if __name__ == "__main__":
    unittest.main()
