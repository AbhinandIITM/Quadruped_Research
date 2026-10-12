"""
test_balance_qp.py
==================
Unit tests for the Single-Rigid-Body Dynamics (SRBD) QP force solver
and analytic leg Jacobian calculations.
"""

import time
import unittest
import numpy as np

from go2_balance_qp import (
    QpBalanceConfig,
    SingleRigidBodyQpSolver,
    compute_leg_jacobian,
    skew_symmetric,
    ROBOT_MASS,
    GRAVITY,
)
from go2_champ_control import solve_ik, HIP_OFFSET


class TestBalanceQpAndJacobian(unittest.TestCase):

    def setUp(self):
        self.config = QpBalanceConfig(
            mu=0.6,
            f_min=5.0,
            f_max=180.0,
            alpha=1e-4,
        )
        self.solver = SingleRigidBodyQpSolver(self.config)

    def test_skew_symmetric(self):
        r = np.array([1.0, 2.0, 3.0])
        v = np.array([4.0, 5.0, 6.0])
        cross_expected = np.cross(r, v)
        cross_skew = skew_symmetric(r) @ v
        np.testing.assert_allclose(cross_skew, cross_expected, atol=1e-10)

    def test_leg_jacobian_finite_difference(self):
        """Verifies analytic Jacobian against numerical finite differences."""
        # Test configurations for both left and right legs
        test_cases = [
            (0.0, 0.6, -1.2, True),
            (0.1, 0.4, -1.0, False),
            (-0.15, 0.8, -1.5, True),
        ]

        def fk(q, is_left):
            th1, th2, th3 = q
            y_off = HIP_OFFSET if is_left else -HIP_OFFSET
            l1, l2 = 0.213, 0.213
            s1, c1 = np.sin(th1), np.cos(th1)
            s2, c2 = np.sin(th2), np.cos(th2)
            s23, c23 = np.sin(th2 + th3), np.cos(th2 + th3)
            x = -l1 * s2 - l2 * s23
            z_leg = -l1 * c2 - l2 * c23
            y = y_off * c1 - z_leg * s1
            z = y_off * s1 + z_leg * c1
            return np.array([x, y, z])

        eps = 1e-6
        for th1, th2, th3, is_left in test_cases:
            q0 = np.array([th1, th2, th3])
            J_analytic = compute_leg_jacobian(th1, th2, th3, is_left)

            # Numerical Jacobian
            J_numerical = np.zeros((3, 3))
            for i in range(3):
                q_plus = q0.copy()
                q_plus[i] += eps
                q_minus = q0.copy()
                q_minus[i] -= eps
                J_numerical[:, i] = (fk(q_plus, is_left) - fk(q_minus, is_left)) / (2 * eps)

            np.testing.assert_allclose(J_analytic, J_numerical, atol=1e-5)

    def test_four_leg_standing_vertical_force(self):
        """In static 4-leg standing, sum of f_z must equal total robot weight."""
        feet = [
            np.array([ 0.1934, -0.14, -0.28]),  # FR
            np.array([ 0.1934,  0.14, -0.28]),  # FL
            np.array([-0.1934, -0.14, -0.28]),  # RR
            np.array([-0.1934,  0.14, -0.28]),  # RL
        ]
        contacts = [True, True, True, True]
        accel_lin = np.array([0.0, 0.0, 0.0])
        accel_ang = np.array([0.0, 0.0, 0.0])

        forces = self.solver.solve(feet, contacts, accel_lin, accel_ang)
        total_fz = np.sum(forces[:, 2])
        expected_weight = ROBOT_MASS * GRAVITY

        # Total vertical force must equal weight within 1%
        self.assertAlmostEqual(total_fz, expected_weight, delta=1.5)

        # Force distribution should be symmetric
        for i in range(4):
            self.assertAlmostEqual(forces[i, 2], expected_weight / 4.0, delta=1.5)
            self.assertAlmostEqual(forces[i, 0], 0.0, delta=0.5)
            self.assertAlmostEqual(forces[i, 1], 0.0, delta=0.5)

    def test_diagonal_trot_two_leg_stance(self):
        """In diagonal trot (e.g. FR and RL in stance), forces split between the 2 active legs."""
        feet = [
            np.array([ 0.1934, -0.14, -0.28]),  # FR
            np.array([ 0.1934,  0.14, -0.28]),  # FL
            np.array([-0.1934, -0.14, -0.28]),  # RR
            np.array([-0.1934,  0.14, -0.28]),  # RL
        ]
        # Diagonal pair: FR and RL in stance, FL and RR in swing
        contacts = [True, False, False, True]
        accel_lin = np.array([0.0, 0.0, 0.0])
        accel_ang = np.array([0.0, 0.0, 0.0])

        forces = self.solver.solve(feet, contacts, accel_lin, accel_ang)

        # Swing legs receive zero force
        np.testing.assert_allclose(forces[1], [0.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(forces[2], [0.0, 0.0, 0.0], atol=1e-8)

        # Stance legs split vertical load
        expected_half_weight = (ROBOT_MASS * GRAVITY) / 2.0
        self.assertAlmostEqual(forces[0, 2], expected_half_weight, delta=2.5)
        self.assertAlmostEqual(forces[3, 2], expected_half_weight, delta=2.5)

    def test_friction_pyramid_constraints_satisfied(self):
        """Tangential forces must strictly obey |fx|, |fy| <= mu * fz."""
        feet = [
            np.array([ 0.1934, -0.14, -0.28]),
            np.array([ 0.1934,  0.14, -0.28]),
            np.array([-0.1934, -0.14, -0.28]),
            np.array([-0.1934,  0.14, -0.28]),
        ]
        contacts = [True, True, True, True]
        # Request aggressive acceleration to challenge friction limits
        accel_lin = np.array([5.0, 3.0, 0.0])
        accel_ang = np.array([2.0, -1.5, 4.0])

        forces = self.solver.solve(feet, contacts, accel_lin, accel_ang)

        for i in range(4):
            fx, fy, fz = forces[i]
            max_tang = self.config.mu * fz
            self.assertLessEqual(abs(fx), max_tang + 1e-6)
            self.assertLessEqual(abs(fy), max_tang + 1e-6)
            self.assertGreaterEqual(fz, self.config.f_min)

    def test_solver_latency_under_one_millisecond(self):
        """Solver must execute well within the 2ms (500 Hz) simulation budget."""
        feet = [
            np.array([ 0.1934, -0.14, -0.28]),
            np.array([ 0.1934,  0.14, -0.28]),
            np.array([-0.1934, -0.14, -0.28]),
            np.array([-0.1934,  0.14, -0.28]),
        ]
        contacts = [True, False, False, True]
        accel_lin = np.array([0.5, 0.0, 0.0])
        accel_ang = np.array([0.0, 0.0, 0.5])

        # Warmup
        for _ in range(10):
            self.solver.solve(feet, contacts, accel_lin, accel_ang)

        t0 = time.perf_counter()
        iters = 200
        for _ in range(iters):
            self.solver.solve(feet, contacts, accel_lin, accel_ang)
        elapsed_per_solve_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        # Solver should take < 0.5 ms per iteration
        self.assertLess(elapsed_per_solve_ms, 0.5)

    def test_controller_compute_stance_torques(self):
        """Verifies controller integration of Jacobian transpose stance torque generation."""
        from go2_champ_control import Go2ChampController, neutral_joint_targets
        controller = Go2ChampController()
        q_standing = neutral_joint_targets(0.28)
        feet = [
            ( 0.1934, -0.14, -0.28),
            ( 0.1934,  0.14, -0.28),
            (-0.1934, -0.14, -0.28),
            (-0.1934,  0.14, -0.28),
        ]
        # Diagonal trot contacts: FR and RL in stance
        contacts = [True, False, False, True]
        torques = controller.compute_stance_torques(q_standing, feet, contacts)

        self.assertEqual(len(torques), 12)
        # Swing legs FL (indices 3..5) and RR (indices 6..8) should produce 0 torque
        np.testing.assert_allclose(torques[3:6], 0.0, atol=1e-8)
        np.testing.assert_allclose(torques[6:9], 0.0, atol=1e-8)

        # Stance legs FR (indices 0..2) and RL (indices 9..11) should produce non-zero support torques
        self.assertGreater(abs(torques[1]) + abs(torques[2]), 0.5)  # Thigh and calf pitch support
        self.assertGreater(abs(torques[10]) + abs(torques[11]), 0.5)


if __name__ == "__main__":
    unittest.main()
