"""
test_friction_consistency.py
============================
Guards the invariant that the simulated contact friction and the contact QP's
friction coefficient are the same number.

Why this matters: the stack's differentiating claim is that the QP "strictly
enforces friction cone constraints" (0.0% violations). That claim is only
falsifiable if the simulator is willing to let friction fail. The floor
previously declared tangential friction 2.5 while the QP constrained against
mu = 0.6, so the QP was enforcing a 31 deg cone in a 68 deg world: it refused
~76% of the friction the floor actually provides, and could never be observed
in violation. Both the floor XML and the procedural terrain generator are
pinned here to QpBalanceConfig().mu.
"""

import os
import unittest

import numpy as np
import pytest

from go2_balance_qp import (
    QpBalanceConfig,
    ROBOT_MASS,
    GRAVITY,
    SingleRigidBodyQpSolver,
)


def _go2_xml_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(root, "unitree_go2", "go2_description.xml"),
        os.path.join("unitree_go2", "go2_description.xml"),
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("go2_description.xml not found")


class TestFloorFrictionMatchesQp(unittest.TestCase):

    def test_floor_tangential_friction_equals_qp_mu(self):
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path(str(_go2_xml_path()))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.assertGreaterEqual(floor_id, 0, "no floor geom in go2_description.xml")
        floor_mu = float(model.geom_friction[floor_id][0])
        self.assertAlmostEqual(
            floor_mu, QpBalanceConfig().mu, places=6,
            msg="Floor friction and QP mu have diverged. A floor that is more "
                "slip-resistant than the QP assumes makes friction-cone "
                "compliance unfalsifiable.",
        )


class TestTerrainFrictionMatchesQp(unittest.TestCase):

    def test_generated_terrain_uses_qp_mu(self):
        tg = pytest.importorskip("go2_terrain_generator")
        mu = QpBalanceConfig().mu
        self.assertAlmostEqual(tg.TERRAIN_MU, mu, places=6)
        self.assertEqual(tg.TERRAIN_FRICTION.split()[0], str(mu))

        for xml_fragment in (tg.generate_stairs_xml(), tg.generate_slope_xml()):
            for attr in xml_fragment.split('friction="')[1:]:
                declared = attr.split('"')[0].split()[0]
                self.assertAlmostEqual(
                    float(declared), mu, places=6,
                    msg=f"Generated terrain friction {declared} != QP mu {mu}",
                )


if __name__ == "__main__":
    unittest.main()


class TestFrictionConeUtilisation(unittest.TestCase):
    """Friction-cone slack must be measured BEFORE the feasibility clip.

    The projection is what makes the returned forces feasible: it clips
    |f_x|, |f_y| to mu * f_z.  Once that has happened an over-saturated
    iterate and a naturally-saturated one are identical, and the
    "were we about to violate the cone?" question has no answer.  The
    solver therefore records utilisation on the pre-projection iterate,
    which is the only place the slack information still exists.
    """

    def setUp(self):
        self.feet = [
            np.array([0.1934, -0.0955, -0.28]),
            np.array([0.1934, 0.0955, -0.28]),
            np.array([-0.1934, -0.0955, -0.28]),
            np.array([-0.1934, 0.0955, -0.28]),
        ]

    def _solve(self, a_lin, a_ang=(0.0, 0.0, 0.0)):
        solver = SingleRigidBodyQpSolver(QpBalanceConfig(mass=ROBOT_MASS))
        return solver, solver.solve(
            self.feet, [True] * 4, list(a_lin), list(a_ang)
        )

    def test_gravity_only_stance_does_not_saturate_the_cone(self):
        """Standing still needs no tangential force at all."""
        solver, _ = self._solve((0.0, 0.0, 0.0))
        self.assertEqual(len(solver.last_friction_utilisation), 4)
        for util in solver.last_friction_utilisation:
            self.assertLess(util, 1.0)
        self.assertFalse(any(solver.last_friction_violated))

    def test_diagnostics_are_cleared_between_solves(self):
        """A clean solve must not report the previous solve's violations."""
        solver, _ = self._solve((9.0, 0.0, 0.0))
        self.assertTrue(
            any(solver.last_friction_violated),
            msg="precondition: the aggressive solve must actually saturate",
        )

        solver.solve(self.feet, [True] * 4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertFalse(any(solver.last_friction_violated))

    def test_over_mu_request_is_flagged_on_pre_projection_iterate(self):
        """An unreachable tangential demand must be reported, not clipped away.

        The clip makes the RETURNED forces feasible either way, so this is
        the only signal distinguishing "we had margin" from "we needed the
        clip and still came up short".
        """
        solver, forces = self._solve((9.0, 0.0, 0.0))
        utils = solver.last_friction_utilisation
        self.assertTrue(
            any(v > 1.0 + 1e-9 for v in utils),
            msg=f"expected a saturated leg, got utilisation {utils}",
        )
        self.assertTrue(any(solver.last_friction_violated))

        # The forces we hand back are still feasible -- that is the contract.
        mu = QpBalanceConfig().mu
        for leg in forces:
            self.assertLessEqual(abs(leg[0]), mu * leg[2] + 1e-9)
            self.assertLessEqual(abs(leg[1]), mu * leg[2] + 1e-9)

    def test_no_stance_reports_empty_utilisation(self):
        """Nothing in contact means nothing to measure."""
        solver = SingleRigidBodyQpSolver(QpBalanceConfig(mass=ROBOT_MASS))
        solver.solve(self.feet, [False] * 4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(solver.last_friction_utilisation, [])
        self.assertEqual(solver.last_friction_violated, [])

    def test_wide_cone_never_saturates(self):
        """With a near-frictionless-limit cone, modest demand is nowhere near 1.0."""
        solver = SingleRigidBodyQpSolver(
            QpBalanceConfig(mass=ROBOT_MASS, mu=1e9)
        )
        solver.solve(self.feet, [True] * 4, [3.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for util in solver.last_friction_utilisation:
            self.assertGreaterEqual(util, 0.0)
            self.assertLess(util, 1.0)
        self.assertFalse(any(solver.last_friction_violated))

    def test_lateral_demand_scales_with_mu_inverse(self):
        """The same lateral ask saturates sooner as mu shrinks.

        This is the property the diagnostic exists to expose: the flagged
        violation set is a function of the cone the solver was configured
        with, so a mu change is visible in the numbers rather than hidden
        by the clip.
        """
        utils_by_mu = {}
        for mu in (0.6, 0.3, 0.15):
            solver = SingleRigidBodyQpSolver(
                QpBalanceConfig(mass=ROBOT_MASS, mu=mu)
            )
            solver.solve(self.feet, [True] * 4, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            utils_by_mu[mu] = max(solver.last_friction_utilisation)

        # Utilisation must be monotone decreasing in mu: tighter cone, more used.
        self.assertGreater(utils_by_mu[0.15], utils_by_mu[0.3])
        self.assertGreater(utils_by_mu[0.3], utils_by_mu[0.6])


class TestGraspMatrixRank(unittest.TestCase):
    """The contact wrench is only fully reachable with 3+ stance feet.

    The grasp matrix A maps foot forces to the 6-DoF base wrench. With fewer
    than 3 stance feet, A is rank-deficient and part of the requested wrench is
    structurally unreachable, no matter what the solver does -- the residual
    reported by last_residual is then infeasibility, not solver error. This is
    why the QP residual diagnostic matters: without it, an unsupportable
    wrench request is indistinguishable from a good solve.

    Rank of A for N point feet at a flat z == min(6, 3*N).
    """

    def test_grasp_rank_by_contact_count(self):
        from go2_balance_qp import skew_symmetric

        z = -0.28
        flat = [
            (0.1934, -0.0955, z),
            (0.1934, 0.0955, z),
            (-0.1934, -0.0955, z),
            (-0.1934, 0.0955, z),
        ]

        def rank_of(contacts):
            idx = [i for i, c in enumerate(contacts) if c]
            A = np.zeros((6, 3 * len(idx)))
            for k, i in enumerate(idx):
                A[0:3, k * 3:k * 3 + 3] = np.eye(3)
                A[3:6, k * 3:k * 3 + 3] = skew_symmetric(np.array(flat[i]))
            return np.linalg.matrix_rank(A)

        self.assertEqual(rank_of([True, False, False, False]), 3)
        self.assertEqual(rank_of([True, False, True, False]), 5)   # diagonal
        self.assertEqual(rank_of([True, True, False, False]), 5)   # front pair
        self.assertEqual(rank_of([True, True, True, False]), 6)    # 3 feet: full wrench
        self.assertEqual(rank_of([True] * 4), 6)

    def test_solver_reports_residual_when_wrench_unreachable(self):
        """A diagonal stance must report a nonzero residual, not a silent fit."""
        solver = SingleRigidBodyQpSolver(QpBalanceConfig(mass=ROBOT_MASS))
        feet = [
            np.array([0.1934, -0.0955, -0.28]),
            np.array([0.1934, 0.0955, -0.28]),
            np.array([-0.1934, -0.0955, -0.28]),
            np.array([-0.1934, 0.0955, -0.28]),
        ]
        solver.solve(feet, [True, False, True, False], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertGreater(solver.last_residual_norm, 1.0)

    def test_solver_converges_with_full_support(self):
        """A 4-foot flat stance must reach the wrench it was asked for."""
        solver = SingleRigidBodyQpSolver(QpBalanceConfig(mass=ROBOT_MASS))
        feet = [np.array([0.1934, -0.0955, -0.28]), np.array([0.1934, 0.0955, -0.28]),
                np.array([-0.1934, -0.0955, -0.28]), np.array([-0.1934, 0.0955, -0.28])]
        solver.solve(feet, [True] * 4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertLess(solver.last_residual_norm, 1.0)
        self.assertAlmostEqual(float(np.sum(solver.solve(
            feet, [True] * 4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])[:, 2])),
            ROBOT_MASS * GRAVITY, delta=2.0)
