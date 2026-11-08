"""
go2_balance_qp.py
=================
Whole-Body Single-Rigid-Body Dynamics (SRBD) Convex QP Contact Force Solver
and Analytical Leg Jacobian for Unitree Go2 Quadruped.

Mathematical Formulation:
  Find Ground Reaction Forces (GRF) f = [f_FR, f_FL, f_RR, f_RL]^T in R^12:
    min_{f} 0.5 * (A f - b)^T W (A f - b) + 0.5 * alpha * ||A||_F^2 * ||f||^2
  Subject to:
    - Positive normal force: f_min <= f_z,i <= f_max (for stance legs)
    - Zero force: f_i = 0 (for swing legs)
    - Friction pyramid: |f_x,i| <= mu * f_z,i and |f_y,i| <= mu * f_z,i

Solution Method:
  This is NOT a general-purpose active-set QP.  The problem is solved as a
  regularised weighted least-squares fit of the 6-DoF wrench (A f = b)
  followed by a bounded fixed-point refinement (see SingleRigidBodyQpSolver):
  components that hit a bound are pinned and the remaining free components are
  re-solved against the residual wrench for a small, fixed number of
  iterations.  The final iterate is always projected back into the feasible
  set, so the normal-force and friction constraints hold unconditionally.

Leg Jacobian Transpose Mapping:
  tau_i = J_i^T * f_i

Integration status (2026-10-17):
  This solver is correct and unit-tested (tests/test_balance_qp.py,
  tests/test_balance_qp_residual.py), but it is NOT part of the control loop
  that the benchmark exercises. unitree_go2/go2_description.xml declares every
  joint as a MuJoCo <position kp="..." kv="..."> servo, so data.ctrl holds a
  target joint *angle* and no torque vector can be routed through those
  actuators. compute_stance_torques() has no production caller; the benchmarked
  controller path (Go2ChampController._step_impl) returns joint-angle targets
  only. Consequently no benchmark run reports QP forces, and no "friction
  pyramid compliance" result is produced by this repository today.
  Wiring the solver in requires converting the actuator model to torque motors
  and adding load-sensitive QP feedforward to the stance legs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
import numpy as np

from go2_champ_control import (
    HIP_OFFSET,
    THIGH as THIGH_LEN,
    CALF as CALF_LEN,
    GRAVITY,
)

# Go2 Geometric parameters
# Total robot mass [kg]. Must match the MuJoCo model's actual body masses:
# sum(model.body_mass) == 16.2764 kg for unitree_go2/go2_description.xml.
# The previous hardcoded 12.0 kg understated this by 35.6%, making the QP's
# gravity-compensation wrench ~42 N too light. Read from the model instead of
# trusting this constant -- see model_total_mass().
ROBOT_MASS = 16.2764       # kg


def model_total_mass(model: Any) -> float:
    """Return the total body mass [kg] from a MuJoCo model.

    The QP's wrench target is b = [m*(a+g); I*alpha], so an error in m scales
    the entire linear wrench target. Always derive m from the model at load
    time rather than trusting ROBOT_MASS.
    """
    try:
        return float(np.sum(model.body_mass))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not read body_mass from model: {exc}") from exc

# Torso principal moments of inertia [kg*m^2]
IBOXY_XX = 0.035
IBOXY_YY = 0.085
IBOXY_ZZ = 0.095
INERTIA_BODY = np.diag([IBOXY_XX, IBOXY_YY, IBOXY_ZZ])


@dataclass(frozen=True)
class QpBalanceConfig:
    mu: float = 0.6                 # Friction coefficient
    f_min: float = 5.0              # Minimum normal force (N)
    f_max: float = 180.0            # Maximum normal force per leg (N)
    alpha: float = 1e-4             # Regularization weight (relative to ||A||_F^2)
    w_lin: float = 1.0              # Linear force tracking weight
    w_ang: float = 2.0              # Angular torque tracking weight
    mass: float = ROBOT_MASS        # Total robot mass [kg] -- read from the model
    #                                 at load time; see model_total_mass().


# Bounded fixed-point refinement budget for the solver: fixed so the solve is
# deterministic and always terminated within the control-loop time budget.
_REFINE_ITERS = 8
# A stacked force component moved by more than this by the feasibility
# projection is considered to be sitting on a bound.
_BOUND_TOL = 1e-9


def skew_symmetric(r: np.ndarray) -> np.ndarray:
    """Returns the 3x3 skew-symmetric matrix for cross product: r x v = [r]_x v."""
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    return np.array([
        [0.0, -rz, ry],
        [rz, 0.0, -rx],
        [-ry, rx, 0.0]
    ], dtype=float)


def compute_leg_jacobian(
    theta_hip: float,
    theta_thigh: float,
    theta_calf: float,
    is_left: bool,
) -> np.ndarray:
    """
    Computes analytical 3x3 foot Jacobian J = d(p_foot) / d(theta)
    relative to hip frame.
    p_foot = [x, y, z]^T
    theta = [theta_hip, theta_thigh, theta_calf]^T
    """
    y_off = HIP_OFFSET if is_left else -HIP_OFFSET
    l1 = THIGH_LEN
    l2 = CALF_LEN

    s1 = math.sin(theta_hip)
    c1 = math.cos(theta_hip)
    s2 = math.sin(theta_thigh)
    c2 = math.cos(theta_thigh)
    s23 = math.sin(theta_thigh + theta_calf)
    c23 = math.cos(theta_thigh + theta_calf)

    # Leg planar extension in thigh-calf plane:
    # x_leg = -l1 * s2 - l2 * s23
    # z_leg = -l1 * c2 - l2 * c23
    # 3D position from hip:
    # x = x_leg
    # y = y_off * c1 - z_leg * s1
    # z = y_off * s1 + z_leg * c1

    # Partial derivatives with respect to theta_thigh:
    dx_d2 = -l1 * c2 - l2 * c23
    dz_leg_d2 = l1 * s2 + l2 * s23

    # Partial derivatives with respect to theta_calf:
    dx_d3 = -l2 * c23
    dz_leg_d3 = l2 * s23

    # Column 0: d/d(theta_hip)
    # x has no theta_hip dependence: dx/d1 = 0
    # dy/d1 = -y_off * s1 - z_leg * c1
    # dz/d1 =  y_off * c1 - z_leg * s1
    z_leg = -l1 * c2 - l2 * c23

    col0 = np.array([
        0.0,
        -y_off * s1 - z_leg * c1,
        y_off * c1 - z_leg * s1
    ])

    # Column 1: d/d(theta_thigh)
    col1 = np.array([
        dx_d2,
        -dz_leg_d2 * s1,
        dz_leg_d2 * c1
    ])

    # Column 2: d/d(theta_calf)
    col2 = np.array([
        dx_d3,
        -dz_leg_d3 * s1,
        dz_leg_d3 * c1
    ])

    return np.column_stack([col0, col1, col2])


class SingleRigidBodyQpSolver:
    """Bounded fixed-point contact force solver for quadruped stance force distribution.

    Solves the ridge-regularised weighted least-squares wrench fit ``A f = b``
    and then refines it with a small, fixed number of iterations that pin the
    force components sitting on a bound (normal-force limits / friction pyramid)
    and re-solve the remaining free components against the residual wrench.  The
    final iterate is always projected back into the feasible set, so the hard
    constraints hold even if the iteration budget is exhausted.

    The solve is therefore *not* a general active-set QP: it is deterministic,
    terminates after at most ``_REFINE_ITERS`` passes, and reduces exactly to the
    unconstrained least-squares solution when no bound is active.

    Diagnostic state (refreshed on every :meth:`solve` call):

    - ``last_residual``: (6,) requested minus achieved wrench, ``A f - b``.
    - ``last_residual_norm``: 2-norm of ``last_residual`` [N / N*m].
    - ``last_bounds_active``: True if any component was pinned to a bound.
    - ``last_iterations``: number of refinement passes performed.
    - ``last_inputs_sanitized``: True if non-finite inputs were replaced.
    - ``last_friction_utilisation``: per active leg,
      ``sqrt(fx^2 + fy^2) / (mu * fz)`` sampled on the PRE-projection
      iterate, so 1.0 means the cone was exactly saturated and > 1.0 means
      the iterate asked for more tangential force than the cone permits.
    - ``last_friction_violated``: per active leg, True when the
      corresponding utilisation exceeds 1.0.  Both lists are empty when no
      leg is in stance.
    """

    def __init__(self, config: Optional[QpBalanceConfig] = None):
        self.config = config or QpBalanceConfig()
        # Diagnostics from the most recent solve() call.
        self.last_residual = np.zeros(6, dtype=float)
        self.last_residual_norm = 0.0
        self.last_bounds_active = False
        self.last_iterations = 0
        self.last_inputs_sanitized = False
        self.last_friction_utilisation: List[float] = []
        self.last_friction_violated: List[bool] = []

    def solve(
        self,
        foot_positions_body: List[np.ndarray],  # 4 foot vectors relative to CoM
        contacts: List[bool],                   # 4 booleans: True if leg is in stance
        desired_accel_lin: np.ndarray,          # Desired linear acceleration [ax, ay, az]
        desired_accel_ang: np.ndarray,          # Desired angular acceleration [alpha_x, alpha_y, alpha_z]
    ) -> np.ndarray:
        """
        Solves for ground reaction forces f in R^(4 x 3) [N].

        Inputs are validated for finiteness: non-finite desired accelerations are
        treated as zero (fall back to gravity support only) and a non-finite foot
        position is treated as a zero moment arm, with
        ``last_inputs_sanitized`` set.  The returned forces are therefore always
        finite; ``last_residual_norm`` reports how much of the requested wrench
        the (bound-projected) forces actually deliver.

        Returns:
            forces: (4, 3) array where forces[i] is [fx, fy, fz] for leg i.
                    Swing legs are exactly zero.
        """
        cfg = self.config
        num_legs = 4
        forces = np.zeros((num_legs, 3), dtype=float)

        self.last_iterations = 0
        self.last_bounds_active = False
        self.last_inputs_sanitized = False
        # Cleared here and repopulated by the final feasibility projection, so
        # a caller can never read utilisation from a previous solve.
        self.last_friction_utilisation: List[float] = []
        self.last_friction_violated: List[bool] = []

        # Target net wrench b in R^6:
        # F_net = m * (a + [0, 0, g])
        # tau_net = I * alpha
        accel_lin = self._finite_accel(desired_accel_lin)
        accel_ang = self._finite_accel(desired_accel_ang)
        f_net_desired = cfg.mass * (accel_lin + np.array([0.0, 0.0, GRAVITY]))
        tau_net_desired = INERTIA_BODY @ accel_ang
        b = np.concatenate([f_net_desired, tau_net_desired])  # (6,)

        active_indices = [i for i, c in enumerate(contacts) if c]
        if not active_indices:
            # No contact: the achievable wrench is zero, so the whole request is
            # reported as residual.  There are no feet to measure cone
            # utilisation against, so both diagnostics stay empty.
            self.last_residual = b.copy()
            self.last_residual_norm = float(np.linalg.norm(b))
            return forces

        # Construct A matrix for active legs: shape (6, 3 * n_active)
        n_act = len(active_indices)
        A = np.zeros((6, 3 * n_act), dtype=float)

        for col_idx, leg_idx in enumerate(active_indices):
            r_foot = self._finite_foot_position(foot_positions_body, leg_idx)
            r_skew = skew_symmetric(r_foot)
            # Top block: I_3
            A[0:3, col_idx*3 : col_idx*3 + 3] = np.eye(3)
            # Bottom block: [r]_x
            A[3:6, col_idx*3 : col_idx*3 + 3] = r_skew

        # Weighting matrix W:
        W_diag = np.array([cfg.w_lin]*3 + [cfg.w_ang]*3, dtype=float)
        W = np.diag(W_diag)

        # Tikhonov term is scaled by the problem magnitude (||A||_F^2) so the
        # regularisation stays meaningful for large foot offsets / many legs
        # instead of becoming negligible and leaving conditioning to geometry.
        a_norm_sq = float(np.linalg.norm(A, ord="fro")) ** 2
        alpha_eff = cfg.alpha * max(a_norm_sq, 1.0)

        # Unconstrained quadratic objective:
        # min 0.5 * f^T H f - g^T f
        # H = A^T W A + alpha_eff * I
        # g = A^T W b
        AtW = A.T @ W
        H = AtW @ A + alpha_eff * np.eye(3 * n_act)
        g_vec = AtW @ b

        # Unconstrained ridge solution (direct solve, pseudo-inverse fallback)
        f_act = self._solve_linear(H, g_vec)

        # Bounded fixed-point refinement: pin the components the feasibility
        # projection would move and re-solve the free ones against the residual
        # wrench.  Exits on the first pass when no bound is active, so the
        # unconstrained solution is returned unchanged.
        free = np.ones(3 * n_act, dtype=bool)
        for _ in range(_REFINE_ITERS):
            self.last_iterations += 1
            projected = self._project(f_act, n_act)
            newly_pinned = (np.abs(projected - f_act) > _BOUND_TOL) & free
            if not newly_pinned.any():
                break
            free = free & ~newly_pinned
            if not free.any():
                break
            # Re-solve the free components with the pinned ones held at their
            # bound values (residual wrench excludes the pinned contribution).
            A_free = A[:, free]
            rhs = A_free.T @ (W @ (b - A[:, ~free] @ projected[~free]))
            H_free = A_free.T @ W @ A_free + alpha_eff * np.eye(A_free.shape[1])
            f_act = projected.copy()
            f_act[free] = self._solve_linear(H_free, rhs)

        # Final feasibility projection: guarantees the normal-force and friction
        # constraints hold even if the iteration budget was exhausted.
        f_act = self._project(f_act, n_act)

        f_reshaped = f_act.reshape(n_act, 3)
        for row, leg_idx in enumerate(active_indices):
            forces[leg_idx] = f_reshaped[row]

        # Contract: the returned forces are always finite.  Input sanitisation,
        # the finite solve and the bounded projection already guarantee this;
        # the check is a cheap last-resort backstop.
        if not np.all(np.isfinite(forces)):
            forces = np.zeros((num_legs, 3), dtype=float)
            self.last_inputs_sanitized = True

        # Wrench residual, computed from the forces actually returned.
        f_final = np.concatenate([forces[i] for i in active_indices])
        self.last_residual = A @ f_final - b
        self.last_residual_norm = float(np.linalg.norm(self.last_residual))
        self.last_bounds_active = bool(not np.all(free))

        return forces

    def _project(self, f_stack: np.ndarray, n_act: int) -> np.ndarray:
        """Projects stacked forces onto the feasible set: normal bounds + friction pyramid."""
        cfg = self.config
        out = np.array(f_stack, dtype=float).reshape(n_act, 3)

        # Friction-cone utilisation is measured on the PRE-projection iterate:
        # u = sqrt(fx^2 + fy^2) / (mu * fz), where u > 1 means the iterate was
        # infeasible and the clip below had to remove tangential force. This
        # must be sampled before the clip, because the projection destroys the
        # slack information -- an iterate clipped to exactly mu*fz is
        # indistinguishable from one that arrived there naturally, and the
        # whole value of the diagnostic is telling those two apart.
        tang = np.hypot(out[:, 0], out[:, 1])
        denom = cfg.mu * np.maximum(out[:, 2], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            util = np.where(denom > 0.0, tang / denom, 0.0)
        self.last_friction_utilisation = [float(v) for v in util]
        self.last_friction_violated = [bool(v > 1.0 + 1e-9) for v in util]

        # 1. Normal force bounds
        out[:, 2] = np.clip(out[:, 2], cfg.f_min, cfg.f_max)
        # 2. Friction pyramid: |fx| <= mu * fz, |fy| <= mu * fz
        f_tang_max = cfg.mu * out[:, 2]
        out[:, 0] = np.clip(out[:, 0], -f_tang_max, f_tang_max)
        out[:, 1] = np.clip(out[:, 1], -f_tang_max, f_tang_max)
        return out.reshape(-1)

    @staticmethod
    def _solve_linear(H: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solves H f = rhs with a pseudo-inverse fallback, guaranteeing a finite result."""
        try:
            f = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            f = np.linalg.pinv(H) @ rhs
        if not np.all(np.isfinite(f)):
            # Fail safe: a non-finite solve contributes no force rather than
            # letting NaN escape into tau = J^T f.
            f = np.zeros_like(rhs, dtype=float)
        return f

    def _finite_accel(self, accel: np.ndarray) -> np.ndarray:
        """Returns a finite (3,) acceleration vector; non-finite input maps to zero."""
        try:
            a = np.asarray(accel, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            a = np.zeros(0, dtype=float)
        if a.shape != (3,) or not np.all(np.isfinite(a)):
            self.last_inputs_sanitized = True
            return np.zeros(3, dtype=float)
        return a

    def _finite_foot_position(self, foot_positions_body: List[np.ndarray], index: int) -> np.ndarray:
        """Returns a finite (3,) foot vector; non-finite input maps to a zero moment arm."""
        try:
            r = np.asarray(foot_positions_body[index], dtype=float).reshape(-1)
        except (IndexError, TypeError, ValueError):
            r = np.zeros(0, dtype=float)
        if r.shape != (3,) or not np.all(np.isfinite(r)):
            self.last_inputs_sanitized = True
            return np.zeros(3, dtype=float)
        return r
