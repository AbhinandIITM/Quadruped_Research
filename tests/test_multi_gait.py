"""
test_multi_gait.py
==================
Unit tests for the multi-gait phase scheduler and in-flight transitions.
"""

import unittest
import numpy as np

from go2_multi_gait import (
    GaitType,
    MultiGaitScheduler,
    GAIT_PRESETS,
)


class TestMultiGaitScheduler(unittest.TestCase):

    def setUp(self):
        self.scheduler = MultiGaitScheduler(initial_gait=GaitType.TROT)

    def test_preset_definitions(self):
        """Verifies that all 5 canonical gaits are defined with valid physical properties."""
        for g_type in [GaitType.TROT, GaitType.PACE, GaitType.BOUND, GaitType.PRONK, GaitType.WALK]:
            profile = GAIT_PRESETS[g_type]
            self.assertEqual(len(profile.phase_offsets), 4)
            self.assertGreater(profile.stance_duration, 0.0)
            self.assertGreater(profile.swing_duration, 0.0)
            self.assertGreater(profile.stride_period, 0.0)
            self.assertGreater(profile.duty_factor, 0.0)
            self.assertLess(profile.duty_factor, 1.0)

    def test_trot_diagonal_pairing(self):
        """In Trot, FR and RL share offset 0.0, FL and RR share offset 0.5."""
        trot = GAIT_PRESETS[GaitType.TROT]
        self.assertEqual(trot.phase_offsets[0], trot.phase_offsets[3])  # FR == RL
        self.assertEqual(trot.phase_offsets[1], trot.phase_offsets[2])  # FL == RR

    def test_pace_lateral_pairing(self):
        """In Pace, FR and RR share offset 0.0, FL and RL share offset 0.5."""
        pace = GAIT_PRESETS[GaitType.PACE]
        self.assertEqual(pace.phase_offsets[0], pace.phase_offsets[2])  # FR == RR
        self.assertEqual(pace.phase_offsets[1], pace.phase_offsets[3])  # FL == RL

    def test_bound_sagittal_pairing(self):
        """In Bound, FR and FL share offset 0.0, RR and RL share offset 0.5."""
        bound = GAIT_PRESETS[GaitType.BOUND]
        self.assertEqual(bound.phase_offsets[0], bound.phase_offsets[1])  # FR == FL
        self.assertEqual(bound.phase_offsets[2], bound.phase_offsets[3])  # RR == RL

    def test_walk_crawl_duty_factor(self):
        """Walk (crawl) has duty factor >= 0.70 ensuring >= 3 feet on the ground."""
        walk = GAIT_PRESETS[GaitType.WALK]
        self.assertGreaterEqual(walk.duty_factor, 0.70)

    def test_pronk_synchronized_phase(self):
        """In Pronk, all 4 legs have identical zero phase offset."""
        pronk = GAIT_PRESETS[GaitType.PRONK]
        self.assertEqual(pronk.phase_offsets, (0.0, 0.0, 0.0, 0.0))

    def test_stationary_produces_zero_phases(self):
        st, sw = self.scheduler.step(0.01, is_moving=False)
        self.assertEqual(st, [0.0] * 4)
        self.assertEqual(sw, [0.0] * 4)

    def test_smooth_transition_continuity(self):
        """Switching from TROT to PACE must smoothly progress without NaNs or exceptions."""
        dt = 0.002
        # Run in trot for 0.5s
        for _ in range(250):
            st, sw = self.scheduler.step(dt, is_moving=True)
            self.assertTrue(all(np.isfinite(st)) and all(np.isfinite(sw)))

        # Request transition to PACE over 0.4s
        success = self.scheduler.set_gait(GaitType.PACE, transition_duration=0.4)
        self.assertTrue(success)
        self.assertEqual(self.scheduler.target_gait, GaitType.PACE)

        # Step through the transition
        for _ in range(200):
            st, sw = self.scheduler.step(dt, is_moving=True)
            self.assertTrue(all(np.isfinite(st)) and all(np.isfinite(sw)))
            # Phases must strictly remain in [0, 1]
            for p in st + sw:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

        # Transition should complete
        self.assertEqual(self.scheduler.current_gait, GaitType.PACE)
        self.assertGreaterEqual(self.scheduler.transition_progress, 1.0)

    def test_string_gait_selection(self):
        self.assertTrue(self.scheduler.set_gait("walk"))
        self.assertEqual(self.scheduler.target_gait, GaitType.WALK)

        self.assertTrue(self.scheduler.set_gait("bound"))
        self.assertEqual(self.scheduler.target_gait, GaitType.BOUND)

        # Invalid gait returns False
        self.assertFalse(self.scheduler.set_gait("hyper_jump"))


if __name__ == "__main__":
    unittest.main()
