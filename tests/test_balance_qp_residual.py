"""
test_balance_qp_residual.py
===========================
Regression tests for the SRBD contact force solver covering:
  - the finiteness guard (no NaN/Inf may escape into tau = J^T f),
  - the reported wrench residual (A f - b) when bounds are / are not active,
  - the bounded fixed-point refinement that improves on the old
    "solve unconstrained then clip" behaviour,
  - the regularisation that is scaled relative to ||A||_F^2.
"""

import unittest
import numpy as np

from go2_balance_qp import (
    QpBalanceConfig,
    SingleRigidBodyQpSolver,
    INERTIA_BODY,
    ROBOT_MASS,
    GRAVITY,
    skew_symmetric,
)

# Nominal standing foot positions [FR, FL, RR, RL] relative to the CoM.
FEET = [
    np.array([ 0.1934, -0.14, -0.28]),
    np.array([ 0.1934,  0.14, -0.28]),
    np.array([-0.1934, -0.14, -0.28]),
    np.array([-0.1934,  0.14, -0.28]),
]
ZERO3 = np.zeros(3, dtype=float)


def _wrench_system(feet, contacts, accel_lin, accel_ang, cfg):
    """Rebuilds (A, b, active_indices, W) exactly as the solver does."""
    active = [i for i, c in enumerate(contacts) if c]
    n_act = len(active)
    A = np.zeros((6, 3 * n_act), dtype=float)
    for col_idx, leg_idx in enumerate(active):
        A[0:3, col_idx * 3: col_idx * 3 + 3] = np.eye(3)
        A[3:6, col_idx * 3: col_idx * 3 + 3] = skew_symmetric(feet[leg_idx])
    b = np.concatenate([
        ROBOT_MASS * (np.asarray(accel_lin, dtype=float) + np.array([0.0, 0.0, GRAVITY])),
        INERTIA_BODY @ np.asarray(accel_ang, dtype=float),
    ])
    W = np.diag(np.array([cfg.w_lin] * 3 + [cfg.w_ang] * 3, dtype=float))
    return A, b, active, W


def _unconstrained_solution(feet, contacts, accel_lin, accel_ang, cfg, scale_alpha=True):
    """Reference unconstrained ridge solve; scale_alpha=False = the old fixed-alpha rule."""
    A, b, active, W = _wrench_system(feet, contacts, accel_lin, accel_ang, cfg)
    alpha_eff = cfg.alpha * (float(np.linalg.norm(A, ord="fro")) ** 2 if scale_alpha else 1.0)
    H = A.T @ W @ A + alpha_eff * np.eye(A.shape[1])
    f = np.linalg.solve(H, A.T @ W @ b).reshape(len(active), 3)
    out = np.zeros((4, 3), dtype=float)
    for row, leg_idx in enumerate(active):
        out[leg_idx] = f[row]
    return out


def _naive_projection(feet, contacts, accel_lin, accel_ang, cfg):
    """The pre-fix behaviour: unconstrained solve followed by post-hoc clipping."""
    raw = _unconstrained_solution(feet, contacts, accel_lin, accel_ang, cfg)
    out = np.zeros((4, 3), dtype=float)
    for i, c in enumerate(contacts):
        if not c:
            continue
        fx, fy, fz = raw[i]
        fz = float(np.clip(fz, cfg.f_min, cfg.f_max))
        f_tang_max = cfg.mu * fz
        out[i] = [float(np.clip(fx, -f_tang_max, f_tang_max)),
                  float(np.clip(fy, -f_tang_max, f_tang_max)),
                  fz]
    return out


class TestSolverFinitenessGuard(unittest.TestCase):
    """A non-finite input must never escape into the returned forces."""

    def setUp(self):
        self.solver = SingleRigidBodyQpSolver()

    def test_nan_foot_position_returns_finite_forces(self):
        feet = [np.array([np.nan, 0.14, -0.28])] + FEET[1:]
        forces = self.solver.solve(feet, [True] * 4, ZERO3, ZERO3)

        self.assertTrue(np.all(np.isfinite(forces)))
        self.assertTrue(self.solver.last_inputs_sanitized)
        self.assertTrue(np.all(np.isfinite(self.solver.last_residual)))

    def test_nan_in_every_input_returns_finite_forces(self):
        feet = [np.array([np.nan, np.inf, -np.inf])] * 4
        forces = self.solver.solve(feet, [True] * 4,
                                   np.array([np.nan, np.nan, np.nan]),
                                   np.array([np.nan, np.nan, np.nan]))

        self.assertTrue(np.all(np.isfinite(forces)))
        self.assertTrue(self.solver.last_inputs_sanitized)
        self.assertFalse(self.solver.last_bounds_active)

    def test_non_finite_acceleration_falls_back_to_gravity_only(self):
        """Sanitised accelerations are treated as zero: gravity support only."""
        reference = SingleRigidBodyQpSolver().solve(FEET, [True] * 4, ZERO3, ZERO3)
        nan_forces = self.solver.solve(FEET, [True] * 4,
                                       np.array([np.inf, -np.inf, np.nan]),
                                       np.array([np.nan, np.inf, np.nan]))

        self.assertTrue(self.solver.last_inputs_sanitized)
        np.testing.assert_allclose(nan_forces, reference, atol=1e-9)

    def test_none_acceleration_is_accepted(self):
        forces = self.solver.solve(FEET, [True] * 4, None, None)
        self.assertTrue(np.all(np.isfinite(forces)))
        self.assertAlmostEqual(np.sum(forces[:, 2]), ROBOT_MASS * GRAVITY, delta=1.5)

    def test_degenerate_foot_shapes_return_finite_forces(self):
        for bad_feet in ([np.zeros(2)] * 4, [np.zeros((2, 2))] * 4, []):
            forces = self.solver.solve(bad_feet, [True] * 4, ZERO3, ZERO3)
            self.assertTrue(np.all(np.isfinite(forces)), msg=str(bad_feet))
            self.assertTrue(self.solver.last_inputs_sanitized)

    def test_degenerate_foot_position_is_zero_moment_arm(self):
        """A non-finite foot vector is replaced by a well-defined zero moment arm."""
        sanitized = self.solver.solve([np.array([np.nan, np.nan, np.nan])] + FEET[1:],
                                      [True] * 4, ZERO3, ZERO3)
        zero_arm = SingleRigidBodyQpSolver().solve([np.zeros(3)] + FEET[1:],
                                                   [True] * 4, ZERO3, ZERO3)
        np.testing.assert_allclose(sanitized, zero_arm, atol=1e-9)

    def test_no_contact_returns_zeros_and_reports_full_residual(self):
        forces = self.solver.solve(FEET, [False] * 4, ZERO3, ZERO3)

        np.testing.assert_array_equal(forces, np.zeros((4, 3)))
        self.assertFalse(self.solver.last_bounds_active)
        self.assertEqual(self.solver.last_iterations, 0)
        # Nothing supports the robot, so the entire requested wrench is residual.
        self.assertAlmostEqual(self.solver.last_residual_norm, ROBOT_MASS * GRAVITY, places=6)


class TestWrenchResidualReporting(unittest.TestCase):
    """The achieved wrench error must be observable to the caller."""

    def test_diagnostics_exist_before_first_solve(self):
        solver = SingleRigidBodyQpSolver()
        self.assertEqual(solver.last_residual.shape, (6,))
        self.assertAlmostEqual(solver.last_residual_norm, 0.0)
        self.assertFalse(solver.last_bounds_active)
        self.assertEqual(solver.last_iterations, 0)
        self.assertFalse(solver.last_inputs_sanitized)

    def test_residual_is_negligible_when_no_bound_is_active(self):
        """With a tiny ridge the unconstrained fit delivers the wrench almost exactly."""
        cfg = QpBalanceConfig(alpha=1e-9)
        solver = SingleRigidBodyQpSolver(cfg)
        forces = solver.solve(FEET, [True] * 4, ZERO3, ZERO3)

        self.assertFalse(solver.last_bounds_active)
        self.assertEqual(solver.last_iterations, 1)  # refinement exits immediately
        self.assertLess(solver.last_residual_norm, 1e-4)

        # ... and the solve reduces to the unconstrained ridge least-squares solution.
        expected = _unconstrained_solution(FEET, [True] * 4, ZERO3, ZERO3, cfg)
        np.testing.assert_allclose(forces, expected, atol=1e-4)

    def test_residual_with_default_config_is_bounded_by_the_ridge_term(self):
        """The default ridge biases the fit slightly, so "zero" means "negligible"."""
        solver = SingleRigidBodyQpSolver()
        solver.solve(FEET, [True] * 4, ZERO3, ZERO3)

        self.assertFalse(solver.last_bounds_active)
        self.assertLess(solver.last_residual_norm, 1e-2 * ROBOT_MASS * GRAVITY)
        self.assertAlmostEqual(np.sum(np.abs(solver.last_residual)), solver.last_residual_norm, places=9)

    def test_residual_matches_the_achieved_wrench_when_a_bound_is_active(self):
        cfg = QpBalanceConfig()
        solver = SingleRigidBodyQpSolver(cfg)
        accel_lin = np.array([5.0, 3.0, 0.0])
        accel_ang = np.array([2.0, -1.5, 4.0])

        forces = solver.solve(FEET, [True] * 4, accel_lin, accel_ang)

        # The reported residual is exactly the wrench the returned forces fail to deliver.
        A, b, active, _ = _wrench_system(FEET, [True] * 4, accel_lin, accel_ang, cfg)
        f_final = np.concatenate([forces[i] for i in active])
        expected_residual = A @ f_final - b

        np.testing.assert_allclose(solver.last_residual, expected_residual, atol=1e-9)
        self.assertAlmostEqual(solver.last_residual_norm,
                               float(np.linalg.norm(expected_residual)), places=9)

        # A bound really was active, and the shortfall is visible rather than silent.
        self.assertTrue(solver.last_bounds_active)
        self.assertGreater(solver.last_iterations, 1)
        self.assertGreater(solver.last_residual_norm, 1e-3)

    def test_forces_stay_feasible_when_bounds_are_active(self):
        cfg = QpBalanceConfig()
        solver = SingleRigidBodyQpSolver(cfg)
        forces = solver.solve(FEET, [True] * 4, np.array([20.0, 0.0, 0.0]), ZERO3)

        for i in range(4):
            fx, fy, fz = forces[i]
            self.assertGreaterEqual(fz, cfg.f_min - 1e-9)
            self.assertLessEqual(fz, cfg.f_max + 1e-9)
            self.assertLessEqual(abs(fx), cfg.mu * fz + 1e-6)
            self.assertLessEqual(abs(fy), cfg.mu * fz + 1e-6)

    def test_refinement_improves_tracking_over_naive_projection(self):
        """Pinning the bound-hit components and re-solving must beat plain clipping."""
        cfg = QpBalanceConfig()
        solver = SingleRigidBodyQpSolver(cfg)
        accel_lin = np.array([5.0, 3.0, 0.0])
        accel_ang = np.array([2.0, -1.5, 4.0])

        forces = solver.solve(FEET, [True] * 4, accel_lin, accel_ang)

        A, b, active, _ = _wrench_system(FEET, [True] * 4, accel_lin, accel_ang, cfg)
        naive = _naive_projection(FEET, [True] * 4, accel_lin, accel_ang, cfg)
        naive_residual = float(np.linalg.norm(A @ np.concatenate([naive[i] for i in active]) - b))

        self.assertTrue(solver.last_bounds_active)
        self.assertLess(solver.last_residual_norm, 0.5 * naive_residual)
        # The refinement was bounded and terminated well inside the budget.
        self.assertLessEqual(solver.last_iterations, 8)

    def test_swing_legs_stay_exactly_zero(self):
        solver = SingleRigidBodyQpSolver()
        forces = solver.solve(FEET, [True, False, False, True],
                              np.array([4.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(forces[1], np.zeros(3))
        np.testing.assert_array_equal(forces[2], np.zeros(3))


class TestRegularisationScaling(unittest.TestCase):
    """The Tikhonov term is relative to ||A||_F^2, not a fixed constant."""

    def test_unconstrained_solution_matches_scaled_regularisation(self):
        cfg = QpBalanceConfig()
        accel_lin = np.array([0.5, 0.2, 0.0])
        accel_ang = np.array([0.3, -0.2, 0.4])

        for scale in (1.0, 8.0, 50.0):
            feet = [f * scale for f in FEET]
            solver = SingleRigidBodyQpSolver(cfg)
            forces = solver.solve(feet, [True] * 4, accel_lin, accel_ang)
            self.assertFalse(solver.last_bounds_active)

            scaled = _unconstrained_solution(feet, [True] * 4, accel_lin, accel_ang, cfg, True)
            fixed = _unconstrained_solution(feet, [True] * 4, accel_lin, accel_ang, cfg, False)

            np.testing.assert_allclose(forces, scaled, atol=1e-9)
            # A fixed alpha would no longer reproduce the solve at larger scales.
            self.assertGreater(np.max(np.abs(forces - fixed)), 1e-4)


if __name__ == "__main__":
    unittest.main()
