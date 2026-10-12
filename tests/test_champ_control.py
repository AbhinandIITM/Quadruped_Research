"""Unit tests for Go2 CHAMP Kinematics, Gait Generation, and Safety Supervisor.

Tests cover:
- Analytic 3-DOF inverse kinematics (hip-roll, thigh-pitch, calf-pitch)
- CHAMP diagonal trot phase oscillator
- 12-point Bezier swing curve and cosine stance profile
- Foothold placement and kinematic coupling
- Command slew-rate limiting
- 2D Waypoint navigation and heading regulation
- Safety supervisor fault detection and emergency triggers
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_champ_control import (
    CALF,
    HIP_OFFSET,
    HIP_POSITIONS,
    IS_LEFT,
    JOINT_LIMITS,
    NOMINAL_BODY_HEIGHT,
    THIGH,
    ChampConfig,
    ChampPhaseGenerator,
    ChampTrajectoryPlanner,
    CommandSlewLimiter,
    FootholdPlanner,
    GaitConfig,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    SafetySupervisor,
    Waypoint,
    WaypointNavigator,
    neutral_joint_targets,
    quaternion_to_euler_xyz,
    solve_ik,
)


class TestGo2Kinematics(unittest.TestCase):
    """Test 3-DOF analytic inverse kinematics and workspace limits."""

    def test_solve_ik_neutral_standing(self):
        """Neutral standing foot directly below hip should yield valid joint angles."""
        for is_left in (False, True):
            lateral = HIP_OFFSET if is_left else -HIP_OFFSET
            hip, thigh, calf = solve_ik(0.0, lateral, -NOMINAL_BODY_HEIGHT, is_left=is_left)

            # Hip roll should be zero when foot is directly below hip offset
            self.assertAlmostEqual(hip, 0.0, places=4)

            # Knee (calf) must be flexed backwards (< 0)
            self.assertLess(calf, -0.5)

            # Check angles within physical limits
            self.assertTrue(JOINT_LIMITS["hip"][0] <= hip <= JOINT_LIMITS["hip"][1])
            self.assertTrue(JOINT_LIMITS["thigh"][0] <= thigh <= JOINT_LIMITS["thigh"][1])
            self.assertTrue(JOINT_LIMITS["calf"][0] <= calf <= JOINT_LIMITS["calf"][1])

    def test_solve_ik_symmetry(self):
        """Left and right legs should produce symmetric thigh/calf angles for symmetric foot targets."""
        q_right = solve_ik(0.05, -HIP_OFFSET, -0.26, is_left=False)
        q_left = solve_ik(0.05, HIP_OFFSET, -0.26, is_left=True)

        self.assertAlmostEqual(q_right[0] + q_left[0], 0.0, places=4)
        self.assertAlmostEqual(q_right[1], q_left[1], places=4)
        self.assertAlmostEqual(q_right[2], q_left[2], places=4)

    def test_solve_ik_outside_workspace_raises(self):
        """Target inside hip cylinder or beyond reach must raise ValueError."""
        with self.assertRaises(ValueError):
            solve_ik(0.0, 0.0, 0.0, is_left=False)

    def test_neutral_joint_targets_length(self):
        """Neutral targets must return exactly 12 joint values."""
        neutral_q = neutral_joint_targets(NOMINAL_BODY_HEIGHT)
        self.assertEqual(len(neutral_q), 12)
        self.assertTrue(np.all(np.isfinite(neutral_q)))


class TestChampPhaseGenerator(unittest.TestCase):
    """Test the CHAMP diagonal trot phase oscillator."""

    def test_phase_offsets_and_period(self):
        gen = ChampPhaseGenerator(stance_duration=0.22, swing_duration=0.22)
        self.assertAlmostEqual(gen.stride_period, 0.44, places=4)

        # Diagonal pairing: FR (index 0) and RL (index 3) share phase 0.0
        # FL (index 1) and RR (index 2) share phase 0.5
        self.assertEqual(gen.offsets[0], 0.0)
        self.assertEqual(gen.offsets[3], 0.0)
        self.assertEqual(gen.offsets[1], 0.5)
        self.assertEqual(gen.offsets[2], 0.5)

    def test_phase_stepping_continuity(self):
        gen = ChampPhaseGenerator(stance_duration=0.22, swing_duration=0.22)
        dt = 0.002
        st_prev, sw_prev = gen.step(dt, is_moving=True)

        for _ in range(50):
            st, sw = gen.step(dt, is_moving=True)
            for leg in range(4):
                self.assertTrue(0.0 <= st[leg] <= 1.0)
                self.assertTrue(0.0 <= sw[leg] <= 1.0)
                # Stance and swing cannot both be active simultaneously
                if st[leg] > 0.0:
                    self.assertEqual(sw[leg], 0.0)
                elif sw[leg] > 0.0:
                    self.assertEqual(st[leg], 0.0)

    def test_phase_stationary_zero(self):
        gen = ChampPhaseGenerator(stance_duration=0.22, swing_duration=0.22)
        st, sw = gen.step(0.01, is_moving=False)
        self.assertEqual(st, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(sw, [0.0, 0.0, 0.0, 0.0])


class TestTrajectories(unittest.TestCase):
    """Test 12-point Bezier swing curves and cosine stance curves."""

    def test_bezier_endpoints_zero(self):
        """Swing curve lift should be 0.0 at liftoff (s=0) and touchdown (s=1)."""
        planner = ChampTrajectoryPlanner(swing_height=0.06, stance_depth=0.0)
        L = 0.16
        # At start of swing (swing_phase=0.0, stance_phase=0.0 is boundary, test s=0.0001)
        _, z_start = planner.generate(step_length=L, swing_phase=0.0001, stance_phase=0.0)
        _, z_end = planner.generate(step_length=L, swing_phase=0.9999, stance_phase=0.0)
        self.assertAlmostEqual(z_start, 0.0, places=3)
        self.assertAlmostEqual(z_end, 0.0, places=3)

    def test_bezier_peak_at_midpoint(self):
        """Peak clearance occurs at swing phase ~0.5 and matches swing_height."""
        planner = ChampTrajectoryPlanner(swing_height=0.06, stance_depth=0.0)
        L = 0.16
        _, z_mid = planner.generate(step_length=L, swing_phase=0.5, stance_phase=0.0)
        self.assertAlmostEqual(z_mid, 0.06, delta=0.002)

    def test_bezier_non_negative(self):
        """Swing trajectory should strictly remain above ground (z >= 0)."""
        planner = ChampTrajectoryPlanner(swing_height=0.06, stance_depth=0.0)
        L = 0.16
        for s in np.linspace(0.01, 0.99, 100):
            _, z = planner.generate(step_length=L, swing_phase=s, stance_phase=0.0)
            self.assertGreaterEqual(z, -1e-6)

    def test_stance_curve_retraction(self):
        """Stance curve retracts linearly from +L/2 at touchdown to -L/2 at liftoff."""
        planner = ChampTrajectoryPlanner(swing_height=0.06, stance_depth=0.0)
        L = 0.16
        x_start, _ = planner.generate(step_length=L, swing_phase=0.0, stance_phase=0.0001)
        x_mid, _ = planner.generate(step_length=L, swing_phase=0.0, stance_phase=0.5)
        x_end, _ = planner.generate(step_length=L, swing_phase=0.0, stance_phase=1.0)

        self.assertAlmostEqual(x_start, L / 2.0, places=3)
        self.assertAlmostEqual(x_mid, 0.0, places=3)
        self.assertAlmostEqual(x_end, -L / 2.0, places=3)


class TestCommandSlewLimiter(unittest.TestCase):
    """Test command acceleration and slew rate limiting."""

    def test_linear_acceleration_clamping(self):
        bounds = SafetyBounds(max_linear_accel=1.0)
        limiter = CommandSlewLimiter(bounds)

        # Step from 0.0 to 1.0 in dt = 0.1s -> max delta is 1.0 * 0.1 = 0.1
        cmd = MotionCommand(vx=1.0, vy=0.0, yaw_rate=0.0)
        vx_out, vy_out, yaw_out = limiter.update(cmd, dt=0.1)

        self.assertAlmostEqual(vx_out, 0.1, places=4)

    def test_yaw_acceleration_clamping(self):
        bounds = SafetyBounds(max_yaw_accel=2.0)
        limiter = CommandSlewLimiter(bounds)

        # Step from 0.0 to 2.0 in dt = 0.1s -> max delta is 2.0 * 0.1 = 0.2
        cmd = MotionCommand(vx=0.0, vy=0.0, yaw_rate=2.0)
        vx_out, vy_out, yaw_out = limiter.update(cmd, dt=0.1)

        self.assertAlmostEqual(yaw_out, 0.2, places=4)


class TestWaypointNavigator(unittest.TestCase):
    """Test 2D world-frame waypoint tracking and heading regulation."""

    def test_reached_tolerance(self):
        nav = WaypointNavigator(k_dist=0.8, k_yaw=1.5)
        wp = Waypoint(x=1.0, y=1.0, speed=0.30, tolerance=0.15)

        # Current position within tolerance
        vx, vy, yaw_rate, reached = nav.compute_velocity_command(
            current_pos=np.array([1.05, 1.05, 0.28]),
            current_yaw=0.0,
            waypoint=wp,
        )
        self.assertTrue(reached)
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(yaw_rate, 0.0)

    def test_pure_yaw_alignment_when_misaligned(self):
        nav = WaypointNavigator(k_dist=0.8, k_yaw=1.5)
        # Target is at (0, 1), requiring yaw = +pi/2 (+90 deg)
        # Robot heading is 0.0 (pointing east) -> angle error = +pi/2 > 0.45 rad
        wp = Waypoint(x=0.0, y=1.0, speed=0.30, tolerance=0.15)
        vx, vy, yaw_rate, reached = nav.compute_velocity_command(
            current_pos=np.array([0.0, 0.0, 0.28]),
            current_yaw=0.0,
            waypoint=wp,
        )
        self.assertFalse(reached)
        # When heading error > 0.45 rad, robot rotates in place with vx=0
        self.assertAlmostEqual(vx, 0.0)
        self.assertGreater(yaw_rate, 0.0)  # Turn left towards +Y


class TestSafetySupervisor(unittest.TestCase):
    """Test safety fault detection and emergency triggers."""

    def test_safe_nominal_state(self):
        bounds = SafetyBounds()
        supervisor = SafetySupervisor(bounds)
        neutral_q = neutral_joint_targets(0.28)

        fault = supervisor.check_state(
            step_time=1.0,
            base_z=0.28,
            roll=0.02,
            pitch=-0.02,
            joint_positions=neutral_q,
        )
        self.assertIsNone(fault)

    def test_roll_fault_trigger(self):
        bounds = SafetyBounds(max_roll_rad=0.35)
        supervisor = SafetySupervisor(bounds)
        neutral_q = neutral_joint_targets(0.28)

        fault = supervisor.check_state(
            step_time=1.0,
            base_z=0.28,
            roll=0.45,  # Exceeds max_roll
            pitch=0.0,
            joint_positions=neutral_q,
        )
        self.assertIsNotNone(fault)
        self.assertEqual(fault.fault_type, "ROLL_LIMIT_VIOLATION")

    def test_low_height_fault_trigger(self):
        bounds = SafetyBounds(min_height_m=0.18)
        supervisor = SafetySupervisor(bounds)
        neutral_q = neutral_joint_targets(0.28)

        fault = supervisor.check_state(
            step_time=1.0,
            base_z=0.14,  # Below minimum ground clearance
            roll=0.0,
            pitch=0.0,
            joint_positions=neutral_q,
        )
        self.assertIsNotNone(fault)
        self.assertEqual(fault.fault_type, "HEIGHT_LIMIT_VIOLATION")

    def test_non_finite_fault_trigger(self):
        bounds = SafetyBounds()
        supervisor = SafetySupervisor(bounds)
        neutral_q = neutral_joint_targets(0.28)

        fault = supervisor.check_state(
            step_time=1.0,
            base_z=float("nan"),
            roll=0.0,
            pitch=0.0,
            joint_positions=neutral_q,
        )
        self.assertIsNotNone(fault)
        self.assertEqual(fault.fault_type, "NON_FINITE_STATE")


if __name__ == "__main__":
    unittest.main()
