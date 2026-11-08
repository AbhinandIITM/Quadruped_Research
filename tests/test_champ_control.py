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
import os
import sys
import unittest
from unittest.mock import Mock

import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_champ_control import (
    CALF,
    HIP_OFFSET,
    HIP_POSITIONS,
    IS_LEFT,
    JOINT_LIMITS,
    JOINT_NAMES,
    LEG_NAMES,
    NOMINAL_BODY_HEIGHT,
    THIGH,
    ChampConfig,
    ChampPhaseGenerator,
    ChampTrajectoryPlanner,
    CommandSlewLimiter,
    FootholdPlanner,
    GaitConfig,
    Go2ChampController,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    SafetySupervisor,
    TrajectorySpec,
    TrajectoryTracker,
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

    def test_phase_sweep_has_no_dead_tick(self):
        """A full stride must never yield a tick where both phases are zero.

        A dead tick would make ChampTrajectoryPlanner.generate take its degenerate
        (0, 0) branch, snapping a swing foot from its stance-end position back to
        the nominal centre for one tick.
        """
        gen = ChampPhaseGenerator(stance_duration=0.22, swing_duration=0.22)
        samples = 400
        max_stance = 0.0
        max_swing = 0.0
        for k in range(samples):
            t = k * gen.stride_period / samples
            st, sw = gen.run(t, is_moving=True)
            for leg in range(4):
                self.assertFalse(
                    st[leg] == 0.0 and sw[leg] == 0.0,
                    f"both phases zero at t={t:.4f}, leg={leg}",
                )
                self.assertTrue(0.0 <= st[leg] <= 1.0)
                self.assertTrue(0.0 <= sw[leg] <= 1.0)
                # Exactly one phase is active at any tick
                if st[leg] > 0.0:
                    self.assertEqual(sw[leg], 0.0)
                else:
                    self.assertEqual(st[leg], 0.0)
                max_stance = max(max_stance, st[leg])
                max_swing = max(max_swing, sw[leg])
        # Both phases still span close to the full [0, 1] range across a stride
        self.assertGreater(max_stance, 0.99)
        self.assertGreater(max_swing, 0.99)


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

    def test_multi_waypoint_sequence(self):
        nav = WaypointNavigator(k_dist=0.8, k_yaw=1.5)
        wp1 = Waypoint(x=1.0, y=0.0, speed=0.30, tolerance=0.15)
        wp2 = Waypoint(x=1.0, y=1.0, speed=0.30, tolerance=0.15)
        nav.reset([wp1, wp2])

        self.assertEqual(nav.current_waypoint(), wp1)
        # Arrive at wp1
        vx, vy, yaw_rate, reached = nav.compute_command(0.95, 0.02, 0.0)
        self.assertFalse(reached)  # Advanced to wp2, not fully completed yet
        self.assertEqual(nav.current_waypoint(), wp2)

        # Arrive at wp2
        vx, vy, yaw_rate, reached = nav.compute_command(1.02, 0.98, 1.57)
        self.assertTrue(reached)
        self.assertIsNone(nav.current_waypoint())
        self.assertTrue(nav.is_completed)

    def test_motion_command_waypoints_handling(self):
        wp1 = Waypoint(x=1.0, y=0.0)
        wp2 = Waypoint(x=2.0, y=1.0)
        cmd = MotionCommand(mode=MotionMode.WAYPOINT, waypoints=[wp1, wp2])
        clamped = cmd.clamped(SafetyBounds())
        self.assertEqual(clamped.waypoints, [wp1, wp2])
        d = cmd.to_dict()
        self.assertIn("waypoints", d)
        self.assertEqual(len(d["waypoints"]), 2)
        self.assertEqual(d["waypoints"][0]["x"], 1.0)


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


class TestTrajectoryTracking(unittest.TestCase):
    """Test parametric trajectory tracking, start-time adoption, and completion."""

    # Circle of radius 1 m at 0.30 m/s; at theta = 0 the tangent heads along +Y
    CIRCLE_PARAMS = {"cx": 0.0, "cy": 0.0, "radius": 1.0, "speed": 0.30}

    def _make_controller_and_command(self, duration=10.0):
        controller = Go2ChampController()
        spec = TrajectorySpec(path_type="circle", params=dict(self.CIRCLE_PARAMS), duration=duration)
        command = MotionCommand(mode=MotionMode.TRAJECTORY, trajectory=spec)
        return controller, command, spec

    def _step(self, controller, command, step_time):
        """Step the controller with the robot parked on the circle at theta = 0."""
        base_pos = np.array([1.0, 0.0, NOMINAL_BODY_HEIGHT])
        base_euler = np.array([0.0, 0.0, math.pi / 2.0])
        return controller.step(
            command=command,
            step_time=step_time,
            dt=0.01,
            base_pos=base_pos,
            base_euler=base_euler,
            base_linvel_body=np.zeros(3),
            base_angvel_body=np.zeros(3),
            joint_positions=neutral_joint_targets(NOMINAL_BODY_HEIGHT),
        )

    def test_tracker_defaults(self):
        """A tracker with no spec must report completion without a start offset."""
        tracker = TrajectoryTracker()
        self.assertAlmostEqual(tracker.start_time, 0.0)
        self.assertEqual(tracker.compute_command(0.0, 0.0, 0.0, 5.0), (0.0, 0.0, 0.0, True))

    def test_reference_advances_across_successive_steps(self):
        """Elapsed trajectory time must advance while the same trajectory stays active."""
        controller, command, _ = self._make_controller_and_command()
        tracker = controller.trajectory_tracker

        refs = []
        for step_time in (0.0, 0.25, 0.50):
            _, _, _, _, info = self._step(controller, command, step_time)
            self.assertEqual(info["mode"], "trajectory")
            # Start time adopted on activation and never rewritten
            self.assertAlmostEqual(tracker.start_time, 0.0, places=6)
            # Reference the controller evaluated during this step
            refs.append(tracker.evaluate_path(step_time - tracker.start_time))

        # Reference must sweep around the circle, not stay pinned at theta = 0
        self.assertNotAlmostEqual(refs[-1].x, refs[0].x, places=4)
        self.assertLess(refs[-1].x, refs[0].x)
        self.assertGreater(refs[-1].y, refs[0].y)

    def test_step_does_not_mutate_caller_spec(self):
        """The scenario's own TrajectorySpec must be untouched after step()."""
        controller, command, spec = self._make_controller_and_command()
        self.assertEqual(spec.start_time, 0.0)

        for step_time in (1.0, 1.5, 2.0):
            self._step(controller, command, step_time)

        self.assertEqual(spec.start_time, 0.0)
        self.assertIs(command.trajectory, spec)
        self.assertEqual(spec.params, dict(self.CIRCLE_PARAMS))
        self.assertEqual(spec.duration, 10.0)

    def test_completion_uses_elapsed_trajectory_time(self):
        """Completion fires on elapsed trajectory time, not on the absolute sim clock."""
        controller, command, _ = self._make_controller_and_command(duration=1.0)

        # Absolute sim clock is far past the duration, but elapsed time is not
        _, _, _, _, info = self._step(controller, command, step_time=50.0)
        self.assertEqual(info["mode"], "trajectory")

        # 0.5 s elapsed: still tracking
        _, _, _, _, info = self._step(controller, command, step_time=50.5)
        self.assertEqual(info["mode"], "trajectory")

        # 1.5 s elapsed: duration reached, fall back to stand
        _, _, _, _, info = self._step(controller, command, step_time=51.5)
        self.assertEqual(info["mode"], "stand")

    def test_new_trajectory_re_adopts_start_time(self):
        """A rebuilt equal spec keeps the clock; a different path restarts it."""
        controller, command, _ = self._make_controller_and_command()
        tracker = controller.trajectory_tracker

        self._step(controller, command, step_time=5.0)
        self.assertAlmostEqual(tracker.start_time, 5.0, places=6)

        # Value-equal spec rebuilt by the caller must not restart the clock
        rebuilt = MotionCommand(
            mode=MotionMode.TRAJECTORY,
            trajectory=TrajectorySpec(path_type="circle", params=dict(self.CIRCLE_PARAMS), duration=10.0),
        )
        self._step(controller, rebuilt, step_time=5.5)
        self.assertAlmostEqual(tracker.start_time, 5.0, places=6)

        # A genuinely different trajectory does restart the clock
        other = MotionCommand(
            mode=MotionMode.TRAJECTORY,
            trajectory=TrajectorySpec(path_type="figure_8", params={"omega": 0.5}, duration=10.0),
        )
        self._step(controller, other, step_time=6.0)
        self.assertAlmostEqual(tracker.start_time, 6.0, places=6)

    def test_default_path_type_is_supported(self):
        """A bare TrajectorySpec must not be a footgun; its default must evaluate."""
        tracker = TrajectoryTracker(TrajectorySpec())
        ref = tracker.evaluate_path(0.5)
        self.assertTrue(math.isfinite(ref.x) and math.isfinite(ref.y))

    def test_unsupported_path_type_raises(self):
        """An unknown path_type must fail loudly rather than return dead motion."""
        tracker = TrajectoryTracker(TrajectorySpec(path_type="spline"))
        with self.assertRaises(ValueError) as ctx:
            tracker.evaluate_path(0.5)
        self.assertIn("spline", str(ctx.exception))
        self.assertIn("circle", str(ctx.exception))

    def test_completed_trajectory_does_not_restart(self):
        """A finished path must not silently re-run while the caller re-sends it."""
        controller, command, _ = self._make_controller_and_command(duration=1.0)
        tracker = controller.trajectory_tracker

        self._step(controller, command, step_time=0.0)
        _, _, _, _, info = self._step(controller, command, step_time=1.5)
        self.assertEqual(info["mode"], "stand")
        self.assertIs(controller._completed_trajectory, command.trajectory)

        # Caller keeps re-sending the identical command every tick
        for step_time in (1.6, 1.7, 1.8, 2.0):
            _, _, _, _, info = self._step(controller, command, step_time)
            self.assertEqual(info["mode"], "stand")
        self.assertAlmostEqual(tracker.start_time, 0.0, places=6)

        # An intervening non-trajectory command re-arms the path
        self._step(controller, MotionCommand(mode=MotionMode.STAND), step_time=2.1)
        self.assertIsNone(controller._completed_trajectory)
        _, _, _, _, info = self._step(controller, command, step_time=2.2)
        self.assertEqual(info["mode"], "trajectory")
        self.assertAlmostEqual(tracker.start_time, 2.2, places=6)

    def test_new_spec_still_runs_after_completion(self):
        """A genuinely different path must still run after one has completed."""
        controller, command, _ = self._make_controller_and_command(duration=1.0)
        tracker = controller.trajectory_tracker

        self._step(controller, command, step_time=0.0)
        _, _, _, _, info = self._step(controller, command, step_time=1.5)
        self.assertEqual(info["mode"], "stand")

        other = MotionCommand(
            mode=MotionMode.TRAJECTORY,
            trajectory=TrajectorySpec(path_type="figure_8", params={"omega": 0.5}, duration=10.0),
        )
        _, _, _, _, info = self._step(controller, other, step_time=1.6)
        self.assertEqual(info["mode"], "trajectory")
        self.assertAlmostEqual(tracker.start_time, 1.6, places=6)


class TestMotionCommandClamping(unittest.TestCase):
    """Test the safety backstop in MotionCommand.clamped()."""

    def test_non_finite_values_are_clamped(self):
        """NaN / inf must not survive clamped(), which np.clip alone would pass through."""
        bounds = SafetyBounds()
        nan = float("nan")
        inf = float("inf")

        cmd = MotionCommand(vx=nan, vy=inf, yaw_rate=-inf, duration=nan).clamped(bounds)
        self.assertTrue(math.isfinite(cmd.vx))
        self.assertTrue(math.isfinite(cmd.vy))
        self.assertTrue(math.isfinite(cmd.yaw_rate))
        self.assertTrue(math.isfinite(cmd.duration))
        # Non-finite velocities collapse to zero; the duration falls back to the floor
        self.assertEqual((cmd.vx, cmd.vy, cmd.yaw_rate), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(cmd.duration, 0.05)

    def test_finite_values_are_unchanged(self):
        """The guard must not alter the existing finite-value behaviour."""
        bounds = SafetyBounds()
        cmd = MotionCommand(vx=5.0, vy=-5.0, yaw_rate=5.0, duration=0.5).clamped(bounds)
        self.assertAlmostEqual(cmd.vx, bounds.max_vx)
        self.assertAlmostEqual(cmd.vy, -bounds.max_vy)
        self.assertAlmostEqual(cmd.yaw_rate, bounds.max_yaw_rate)
        self.assertAlmostEqual(cmd.duration, 0.5)


class TestControllerFailSafe(unittest.TestCase):
    """Test the controller's fail-safe behaviour: emergency hold and fault auditing."""

    def _step(self, controller, command, step_time, joint_positions=None, base_z=NOMINAL_BODY_HEIGHT):
        return controller.step(
            command=command,
            step_time=step_time,
            dt=0.01,
            base_pos=np.array([0.0, 0.0, base_z]),
            base_euler=np.zeros(3),
            base_linvel_body=np.zeros(3),
            base_angvel_body=np.zeros(3),
            joint_positions=(
                neutral_joint_targets(NOMINAL_BODY_HEIGHT) if joint_positions is None else joint_positions
            ),
        )

    def test_internal_ik_failure_holds_pose_and_records_fault(self):
        """An IK singularity inside step() must not escape; it must hold the pose."""
        controller = Go2ChampController()
        joint_positions = neutral_joint_targets(NOMINAL_BODY_HEIGHT)

        controller.foothold_planner.plan_leg_footholds = Mock(
            side_effect=ValueError("Foot target inside hip roll singularity")
        )
        target_q, _, stance, swing, info = self._step(
            controller, MotionCommand(vx=0.4, mode=MotionMode.VELOCITY), 0.0, joint_positions
        )

        # Fail-safe response, not an escaping exception
        np.testing.assert_allclose(target_q, joint_positions)
        self.assertEqual(stance, [0.0] * 4)
        self.assertEqual(swing, [0.0] * 4)

        # The fault is recorded and visible to the caller
        self.assertEqual(info["fault"], "CONTROL_EXCEPTION")
        self.assertEqual(info["mode"], "emergency_stand")
        self.assertIsNotNone(controller.emergency_fault)
        self.assertEqual(controller.emergency_fault.fault_type, "CONTROL_EXCEPTION")
        self.assertEqual(controller.safety_supervisor.faults[-1].fault_type, "CONTROL_EXCEPTION")

    def test_emergency_stand_commands_measured_positions(self):
        """EMERGENCY_STAND must hold the measured joints, not drive toward neutral."""
        controller = Go2ChampController()
        measured = neutral_joint_targets(NOMINAL_BODY_HEIGHT).copy()
        measured[0] += 0.15
        measured[3] += 0.12

        # Trip a real safety fault with a base height below the limit
        self._step(controller, MotionCommand(mode=MotionMode.STAND), 0.0, measured, base_z=0.05)
        self.assertEqual(controller.mode, MotionMode.EMERGENCY_STAND)

        target_q, _, _, _, info = self._step(controller, MotionCommand(mode=MotionMode.STAND), 0.1, measured)
        self.assertEqual(info["mode"], "emergency_stand")
        np.testing.assert_allclose(target_q, measured)
        self.assertFalse(np.allclose(target_q, controller.neutral_targets))

    def test_faults_are_auditable_after_reset(self):
        """reset() re-enables motion but must preserve the faults that justified it."""
        controller = Go2ChampController()
        self._step(controller, MotionCommand(mode=MotionMode.STAND), 0.0, base_z=0.05)
        self.assertTrue(controller.safety_supervisor.faults)

        controller.reset()

        # Motion re-enabled, live fault list cleared...
        self.assertEqual(controller.mode, MotionMode.STAND)
        self.assertIsNone(controller.emergency_fault)
        self.assertEqual(controller.safety_supervisor.faults, [])
        # ...but the record is retained and exposed to the caller
        self.assertTrue(controller.faults_at_last_reset)
        self.assertEqual(controller.faults_at_last_reset[0].fault_type, "HEIGHT_LIMIT_VIOLATION")


class TestStateFeedWatchdog(unittest.TestCase):
    """Test the state-feed watchdog: a stalled or backwards sim clock must fault."""

    def _step(self, controller, command, step_time):
        return controller.step(
            command=command,
            step_time=step_time,
            dt=0.01,
            base_pos=np.array([0.0, 0.0, NOMINAL_BODY_HEIGHT]),
            base_euler=np.zeros(3),
            base_linvel_body=np.zeros(3),
            base_angvel_body=np.zeros(3),
            joint_positions=neutral_joint_targets(NOMINAL_BODY_HEIGHT),
        )

    def test_advancing_step_time_never_faults(self):
        """A steadily advancing clock (the normal contract) must stay clean."""
        controller = Go2ChampController()
        command = MotionCommand(vx=0.3, mode=MotionMode.VELOCITY)
        for k in range(200):
            _, _, _, _, info = self._step(controller, command, k * 0.01)
            self.assertIsNone(info["fault"])
        self.assertEqual(controller.safety_supervisor.faults, [])
        self.assertEqual(controller.mode, MotionMode.VELOCITY)

    def test_frozen_step_time_faults_after_threshold(self):
        """A stalled feed faults once the consecutive-repeat limit is crossed."""
        controller = Go2ChampController()
        command = MotionCommand(mode=MotionMode.STAND)
        limit = controller.safety_bounds.max_stale_steps
        self.assertGreaterEqual(limit, 2)
        self.assertLessEqual(limit, 10)  # "a handful of ticks", not a loose gate

        # First call establishes the clock
        _, _, _, _, info = self._step(controller, command, 1.0)
        self.assertIsNone(info["fault"])

        # limit - 1 further frozen calls are tolerated (transient repeats)
        for _ in range(limit - 1):
            _, _, _, _, info = self._step(controller, command, 1.0)
            self.assertIsNone(info["fault"])

        # The next frozen call crosses the threshold and faults
        _, _, _, _, info = self._step(controller, command, 1.0)
        self.assertEqual(info["fault"], "STALE_STATE_FEED")
        self.assertEqual(info["mode"], "emergency_stand")
        self.assertEqual(controller.mode, MotionMode.EMERGENCY_STAND)
        self.assertIsNotNone(controller.emergency_fault)
        self.assertEqual(controller.safety_supervisor.faults[-1].fault_type, "STALE_STATE_FEED")

        # A permanently frozen feed must not flood the audit list
        recorded = len(controller.safety_supervisor.faults)
        for _ in range(5):
            _, _, _, _, info = self._step(controller, command, 1.0)
            self.assertEqual(info["fault"], "STALE_STATE_FEED")
        self.assertEqual(len(controller.safety_supervisor.faults), recorded)

    def test_backwards_step_time_faults_immediately(self):
        """A strictly earlier step_time is a hard invariant violation."""
        controller = Go2ChampController()
        command = MotionCommand(mode=MotionMode.STAND)

        _, _, _, _, info = self._step(controller, command, 2.0)
        self.assertIsNone(info["fault"])

        _, _, _, _, info = self._step(controller, command, 1.0)
        self.assertEqual(info["fault"], "NON_MONOTONIC_STEP_TIME")
        self.assertEqual(info["mode"], "emergency_stand")
        self.assertEqual(controller.mode, MotionMode.EMERGENCY_STAND)
        self.assertEqual(controller.safety_supervisor.faults[-1].fault_type, "NON_MONOTONIC_STEP_TIME")

    def test_reset_rearms_watchdog(self):
        """reset() must clear the watchdog so a fresh clock in the past is accepted."""
        controller = Go2ChampController()
        command = MotionCommand(mode=MotionMode.STAND)
        limit = controller.safety_bounds.max_stale_steps

        self._step(controller, command, 1.0)
        for _ in range(limit):
            self._step(controller, command, 1.0)
        self.assertTrue(controller.safety_supervisor.faults)

        controller.reset()
        self.assertEqual(controller.safety_supervisor.faults, [])

        # Without re-arming, a clock at 0.0 would read as NON_MONOTONIC after 1.0
        _, _, _, _, info = self._step(controller, command, 0.0)
        self.assertIsNone(info["fault"])
        _, _, _, _, info = self._step(controller, command, 0.0)
        self.assertIsNone(info["fault"])


class TestGaitTimeBase(unittest.TestCase):
    """Test that the foot trajectory follows the live gait period, not the static config."""

    def _step(self, controller, command, step_time):
        return controller.step(
            command=command,
            step_time=step_time,
            dt=0.01,
            base_pos=np.array([0.0, 0.0, NOMINAL_BODY_HEIGHT]),
            base_euler=np.zeros(3),
            base_linvel_body=np.zeros(3),
            base_angvel_body=np.zeros(3),
            joint_positions=neutral_joint_targets(NOMINAL_BODY_HEIGHT),
        )

    def test_foothold_step_length_tracks_requested_stride_period(self):
        """Step length must scale with the period it is given, not the static one."""
        config = GaitConfig()
        planner = FootholdPlanner(config)
        stance, swing = [0.5] * 4, [0.0] * 4
        expected_speed = 0.1 * (1.0 + config.raibert_kv)  # cmd vx + Raibert correction

        _, _, step_static, _ = planner.plan_leg_footholds(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, stance, swing
        )
        _, _, step_walk, _ = planner.plan_leg_footholds(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, stance, swing, stride_period=0.80
        )

        self.assertAlmostEqual(step_static[0], expected_speed * config.stride_period, places=6)
        self.assertAlmostEqual(step_walk[0], expected_speed * 0.80, places=6)
        # Both stay inside the clip limit, so the comparison is not masked by clamping
        self.assertLess(step_walk[0], config.max_step_length)

    def test_default_trot_live_period_equals_static_config(self):
        """TROT must stay bit-identical: blended == config, so the planner input is unchanged."""
        controller = Go2ChampController()
        self.assertEqual(controller._live_stride_period(), controller.gait_config.stride_period)

        # The planner must therefore produce bit-identical output whether the live
        # period is passed explicitly (new path) or omitted (old path).
        planner = controller.foothold_planner
        stance, swing = [0.5] * 4, [0.0] * 4
        q_omitted = planner.plan_leg_footholds(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, stance, swing)
        q_explicit = planner.plan_leg_footholds(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, stance, swing,
            stride_period=controller._live_stride_period(),
        )
        np.testing.assert_array_equal(q_omitted[0], q_explicit[0])
        self.assertEqual(q_omitted[2], q_explicit[2])
        self.assertEqual(q_omitted[3], q_explicit[3])

    def test_controller_feeds_live_gait_period_to_planner(self):
        """Under WALK the planner must receive the live 0.80 s period, not 0.44 s."""
        controller = Go2ChampController()
        # The controller applies command.gait every tick, so WALK must be requested
        # through the command (set_gait is idempotent, so this does not restart the
        # in-flight transition each tick).
        command = MotionCommand(vx=0.3, mode=MotionMode.VELOCITY, gait="walk")

        # Drive through the controller; the transition completes in 0.5 s
        for k in range(200):
            _, _, _, _, info = self._step(controller, command, k * 0.01)
            self.assertIsNone(info["fault"])

        live = controller._live_stride_period()
        self.assertAlmostEqual(live, 0.80, places=3)
        self.assertNotAlmostEqual(live, controller.gait_config.stride_period, places=3)

        spy = Mock(wraps=controller.foothold_planner.plan_leg_footholds)
        controller.foothold_planner.plan_leg_footholds = spy
        self._step(controller, command, 2.0)

        self.assertTrue(spy.called)
        self.assertAlmostEqual(spy.call_args.kwargs["stride_period"], live, places=12)

    def test_gait_config_duty_split_reaches_phase_generator(self):
        """A tuned GaitConfig duty split must reach the scheduler, not be dropped.

        Regression guard: the controller used to construct MultiGaitScheduler() with
        no arguments, so GAIT_PRESETS[TROT] won and GaitConfig.stance/swing_duration
        were silently dead. A duty-factor sweep therefore produced identical results
        for every duty factor it asked for.
        """
        cfg = GaitConfig(stance_duration=0.286, swing_duration=0.154)  # duty 0.65
        controller = Go2ChampController(gait_config=cfg)

        self.assertAlmostEqual(controller.phase_generator.stance_duration, 0.286, places=12)
        self.assertAlmostEqual(controller.phase_generator.swing_duration, 0.154, places=12)
        self.assertAlmostEqual(controller._live_stride_period(), 0.44, places=12)

        step_cmd = MotionCommand(vx=0.0, mode=MotionMode.STAND)
        self._step(controller, step_cmd, 0.01)
        self.assertAlmostEqual(controller.phase_generator.stance_duration, 0.286, places=12)

    def test_default_gait_config_is_bit_identical_to_trot_preset(self):
        """The shipped config must not perturb the preset gait time base."""
        controller = Go2ChampController()
        self.assertAlmostEqual(controller.phase_generator.stance_duration, 0.22, places=12)
        self.assertAlmostEqual(controller.phase_generator.swing_duration, 0.22, places=12)
        self.assertAlmostEqual(controller._live_stride_period(), 0.44, places=12)

    def test_duty_override_does_not_leak_across_gaits(self):
        """Tuning trot's duty factor must not rewrite WALK's own gait profile."""
        cfg = GaitConfig(stance_duration=0.286, swing_duration=0.154)
        controller = Go2ChampController(gait_config=cfg)

        walk_cmd = MotionCommand(vx=0.3, mode=MotionMode.VELOCITY, gait="walk")
        for k in range(200):  # transition completes in 0.5 s
            self._step(controller, walk_cmd, k * 0.01)

        self.assertAlmostEqual(controller.phase_generator.stance_duration, 0.60, places=12)
        self.assertAlmostEqual(controller.phase_generator.swing_duration, 0.20, places=12)
        self.assertAlmostEqual(controller._live_stride_period(), 0.80, places=3)


class TestJointOrderingContract(unittest.TestCase):
    """Pin the positional joint-array layout the safety and actuation paths assume."""

    def test_joint_array_layout_is_pinned(self):
        self.assertEqual(LEG_NAMES, ("FR", "FL", "RR", "RL"))
        self.assertEqual(len(JOINT_NAMES), 12)
        self.assertEqual(
            JOINT_NAMES,
            tuple(
                f"{leg}_{joint}_joint"
                for leg in ("FR", "FL", "RR", "RL")
                for joint in ("hip", "thigh", "calf")
            ),
        )

    def test_supervisor_reports_the_correct_leg_for_a_violation(self):
        """A violated joint must be attributed to its real name, not a reordered one."""
        bounds = SafetyBounds()
        supervisor = SafetySupervisor(bounds)
        joints = neutral_joint_targets(NOMINAL_BODY_HEIGHT).copy()
        joints[7] = JOINT_LIMITS["thigh"][1] + 1.0  # RR_thigh_joint (index 7)

        fault = supervisor.check_state(
            step_time=0.0, base_z=NOMINAL_BODY_HEIGHT, roll=0.0, pitch=0.0, joint_positions=joints
        )
        self.assertEqual(fault.fault_type, "JOINT_LIMIT_VIOLATION")
        self.assertIn("RR_thigh_joint", fault.message)

    def test_no_contact_torques_are_refused(self):
        """A no-contact tick must fail loudly, not return a plausible all-zero vector."""
        controller = Go2ChampController()
        feet = [
            (0.1934, -0.14, -0.28),
            (0.1934, 0.14, -0.28),
            (-0.1934, -0.14, -0.28),
            (-0.1934, 0.14, -0.28),
        ]
        with self.assertRaises(ValueError):
            controller.compute_stance_torques(
                neutral_joint_targets(NOMINAL_BODY_HEIGHT), feet, [False, False, False, False]
            )


if __name__ == "__main__":
    unittest.main()
