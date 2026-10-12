"""
go2_balance_qp.py
=================
Whole-Body Single-Rigid-Body Dynamics (SRBD) Convex QP Contact Force Solver
and Analytical Leg Jacobian for Unitree Go2 Quadruped.

Mathematical Formulation:
  Find optimal Ground Reaction Forces (GRF) f = [f_FR, f_FL, f_RR, f_RL]^T in R^12:
    min_{f} 0.5 * (A f - b)^T W (A f - b) + 0.5 * alpha * ||f||^2
  Subject to:
    - Positive normal force: f_min <= f_z,i <= f_max (for stance legs)
    - Zero force: f_i = 0 (for swing legs)
    - Friction pyramid: |f_x,i| <= mu * f_z,i and |f_y,i| <= mu * f_z,i

Leg Jacobian Transpose Mapping:
  tau_i = J_i^T * f_i
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

# Go2 Geometric parameters
ROBOT_MASS = 12.0          # kg
GRAVITY = 9.81             # m/s^2
HIP_OFFSET = 0.0955        # m
THIGH_LEN = 0.213          # m
CALF_LEN = 0.213           # m

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
    alpha: float = 1e-4             # Regularization weight
    w_lin: float = 1.0              # Linear force tracking weight
    w_ang: float = 2.0              # Angular torque tracking weight


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
    """Active-set QP solver for quadruped stance force distribution."""

    def __init__(self, config: Optional[QpBalanceConfig] = None):
        self.config = config or QpBalanceConfig()

    def solve(
        self,
        foot_positions_body: List[np.ndarray],  # 4 foot vectors relative to CoM
        contacts: List[bool],                   # 4 booleans: True if leg is in stance
        desired_accel_lin: np.ndarray,          # Desired linear acceleration [ax, ay, az]
        desired_accel_ang: np.ndarray,          # Desired angular acceleration [alpha_x, alpha_y, alpha_z]
    ) -> np.ndarray:
        """
        Solves for optimal ground reaction forces f in R^(4 x 3) [N].
        Returns:
            forces: (4, 3) array where forces[i] is [fx, fy, fz] for leg i.
        """
        cfg = self.config
        num_legs = 4
        forces = np.zeros((num_legs, 3), dtype=float)

        active_indices = [i for i, c in enumerate(contacts) if c]
        if not active_indices:
            return forces

        # Target net wrench b in R^6:
        # F_net = m * (a + [0, 0, g])
        # tau_net = I * alpha
        f_net_desired = ROBOT_MASS * (desired_accel_lin + np.array([0.0, 0.0, GRAVITY]))
        tau_net_desired = INERTIA_BODY @ desired_accel_ang
        b = np.concatenate([f_net_desired, tau_net_desired])  # (6,)

        # Construct A matrix for active legs: shape (6, 3 * n_active)
        n_act = len(active_indices)
        A = np.zeros((6, 3 * n_act), dtype=float)

        for col_idx, leg_idx in enumerate(active_indices):
            r_foot = foot_positions_body[leg_idx]
            r_skew = skew_symmetric(r_foot)
            # Top block: I_3
            A[0:3, col_idx*3 : col_idx*3 + 3] = np.eye(3)
            # Bottom block: [r]_x
            A[3:6, col_idx*3 : col_idx*3 + 3] = r_skew

        # Weighting matrix W:
        W_diag = np.array([cfg.w_lin]*3 + [cfg.w_ang]*3, dtype=float)
        W = np.diag(W_diag)

        # Unconstrained quadratic objective:
        # min 0.5 * f^T H f - g^T f
        # H = A^T W A + alpha * I
        # g = A^T W b
        H = A.T @ W @ A + cfg.alpha * np.eye(3 * n_act)
        g_vec = A.T @ W @ b

        # Primary solution via Cholesky / pseudo-inverse
        try:
            f_unconstrained = np.linalg.solve(H, g_vec)
        except np.linalg.LinAlgError:
            f_unconstrained = np.linalg.pinv(H) @ g_vec

        # Reshape to (n_act, 3)
        f_act = f_unconstrained.reshape(n_act, 3)

        # Projected constraints onto friction pyramid and normal bounds
        for row, leg_idx in enumerate(active_indices):
            fx, fy, fz = f_act[row]

            # 1. Normal force bounds
            fz_clamped = float(np.clip(fz, cfg.f_min, cfg.f_max))

            # 2. Friction pyramid: |fx| <= mu * fz, |fy| <= mu * fz
            f_tang_max = cfg.mu * fz_clamped
            fx_clamped = float(np.clip(fx, -f_tang_max, f_tang_max))
            fy_clamped = float(np.clip(fy, -f_tang_max, f_tang_max))

            forces[leg_idx] = np.array([fx_clamped, fy_clamped, fz_clamped], dtype=float)

        return forces
