"""Go2 CHAMP Quadruped Control Architecture.

Provides:
- Data contracts: MotionCommand, Waypoint, GaitConfig, SafetyBounds, SafetyFault
- Command slew-rate limiter
- CHAMP 3-DOF analytic Inverse and Forward Kinematics
- CHAMP 12-point Bézier swing and cosine stance trajectory planners
- Omnidirectional foothold planner with hip-yaw coupling and Raibert feedback
- Closed-loop waypoint and heading navigation
- Safety supervisor with emergency stand fail-safe
- Deterministic standing initialization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Go2 Physical & Model Constants
# -----------------------------------------------------------------------------

HIP_OFFSET = 0.0955       # Lateral distance from hip roll joint to thigh pitch joint [m]
THIGH = 0.213            # Thigh link length [m]
CALF = 0.213             # Calf link length [m]
NOMINAL_BODY_HEIGHT = 0.28  # Nominal standing base height [m]
GRAVITY = 9.81           # Acceleration due to gravity [m/s^2]
POSITION_KP = 100.0      # Actuator proportional gain [N*m/rad]
POSITION_KD = 4.0        # Actuator derivative damping gain [N*m*s/rad]

# Motion gates: body velocity / yaw rate above which the robot is treated as moving
MOTION_EPS_LINEAR = 0.01  # [m/s] linear velocity floor
MOTION_EPS_YAW = 0.02     # [rad/s] yaw rate floor

LEG_NAMES = ("FR", "FL", "RR", "RL")
IS_LEFT = (False, True, False, True)

# Hip roll joint positions relative to body center [m]
#
# The lateral coordinate is the position of the hip ROLL axis, which the MJCF
# places at y = +/-0.0465 (FL_hip pos="0.1934 0.0465 0").  It is not the
# roll-to-pitch distance: HIP_OFFSET below holds that, and the thigh pitch
# axis therefore sits at y = +/-0.142, which is where the feet are.
#
# This constant was previously y = +/-0.0955, i.e. the roll-to-pitch distance
# used in the position of the roll axis.  Two consequences followed: the
# reported foot positions were laterally wrong by about 48 mm, and the
# Raibert hip-tangential-velocity coupling, whose lever arm is this
# coordinate, over-weighted the yaw term by a factor of 0.0955/0.0465 = 2.05.
HIP_POSITIONS = np.array(
    [
        [0.1934, -0.0465, 0.0],   # FR
        [0.1934,  0.0465, 0.0],   # FL
        [-0.1934, -0.0465, 0.0],  # RR
        [-0.1934,  0.0465, 0.0],  # RL
    ],
    dtype=float,
)

JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)

FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")

# The 12-element joint array is consumed positionally by the safety path
# (SafetySupervisor.check_state uses joint_types[i % 3]; compute_stance_torques
# uses joint_positions[i * 3 + k]). That silently encodes the
# FR/FL/RR/RL x (hip, thigh, calf) layout, so pin it here: a reorder of
# LEG_NAMES or JOINT_NAMES must fail loudly rather than misattribute a joint
# limit violation (and its fault message) to the wrong leg.
assert LEG_NAMES == ("FR", "FL", "RR", "RL"), f"LEG_NAMES order changed: {LEG_NAMES}"
assert JOINT_NAMES == tuple(
    f"{leg}_{joint}_joint" for leg in LEG_NAMES for joint in ("hip", "thigh", "calf")
), f"JOINT_NAMES must be LEG_NAMES x (hip, thigh, calf); got {JOINT_NAMES}"

# Physical joint angle limits from go2_description.xml [rad]
JOINT_LIMITS = {
    "hip": (-1.0472, 1.0472),
    "thigh": (-1.5708, 3.4907),
    "calf": (-2.7227, -0.83776),
}


# -----------------------------------------------------------------------------
# Data Contracts
# -----------------------------------------------------------------------------

class MotionMode(str, Enum):
    STAND = "stand"
    VELOCITY = "velocity"
    WAYPOINT = "waypoint"
    TRAJECTORY = "trajectory"
    EMERGENCY_STAND = "emergency_stand"


@dataclass
class TrajectoryPoint:
    time_s: float
    x: float
    y: float
    z: float = NOMINAL_BODY_HEIGHT
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    yaw: float = 0.0               # Path tangent heading [rad]

# Parametric path types implemented by TrajectoryTracker.evaluate_path
TRAJECTORY_PATH_TYPES = ("circle", "figure_8")

@dataclass
class TrajectorySpec:
    path_type: str = "circle"  # circle | figure_8
    params: Dict[str, Any] = field(default_factory=dict)
    duration: float = 10.0
    start_time: float = 0.0    # Reference time for relative trajectory evaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_type": self.path_type,
            "params": self.params,
            "duration": self.duration,
            "start_time": self.start_time,
        }

@dataclass
class Waypoint:
    x: float
    y: float
    heading: Optional[float] = None
    tolerance: float = 0.10
    speed: float = 0.30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "tolerance": self.tolerance,
            "speed": self.speed,
        }


@dataclass
class MotionCommand:
    vx: float = 0.0             # Body forward velocity [m/s]
    vy: float = 0.0             # Body lateral velocity [m/s]
    yaw_rate: float = 0.0       # Body yaw rate [rad/s]
    duration: float = 1.0       # Duration for this command [s]
    mode: MotionMode = MotionMode.VELOCITY
    heading_target: Optional[float] = None  # Desired world heading [rad]
    waypoint: Optional[Waypoint] = None     # Target world waypoint
    waypoints: Optional[List[Waypoint]] = None # Target world waypoints list
    trajectory: Optional[TrajectorySpec] = None # Parametric trajectory path
    stop_after: bool = False
    gait: str = "trot"                      # Gait selection: trot | pace | bound | pronk | walk

    def clamped(self, bounds: "SafetyBounds") -> "MotionCommand":
        """Return a new MotionCommand clamped to configured safety bounds."""
        # np.clip propagates NaN, so a non-finite request would otherwise survive
        # this backstop. Collapse non-finite values to zero (a zero velocity, or
        # the minimum duration once clipped) so nothing non-finite reaches the
        # actuators.
        def finite(value: float) -> float:
            return value if math.isfinite(value) else 0.0

        vx_clamped = float(np.clip(finite(self.vx), bounds.min_vx, bounds.max_vx))
        vy_clamped = float(np.clip(finite(self.vy), -bounds.max_vy, bounds.max_vy))
        yaw_clamped = float(np.clip(finite(self.yaw_rate), -bounds.max_yaw_rate, bounds.max_yaw_rate))
        dur_clamped = float(np.clip(finite(self.duration), 0.05, bounds.max_duration))
        return MotionCommand(
            vx=vx_clamped,
            vy=vy_clamped,
            yaw_rate=yaw_clamped,
            duration=dur_clamped,
            mode=self.mode,
            heading_target=self.heading_target,
            waypoint=self.waypoint,
            waypoints=self.waypoints,
            trajectory=self.trajectory,
            stop_after=self.stop_after,
            gait=self.gait,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vx": self.vx,
            "vy": self.vy,
            "yaw_rate": self.yaw_rate,
            "duration": self.duration,
            "mode": self.mode.value if isinstance(self.mode, MotionMode) else str(self.mode),
            "heading_target": self.heading_target,
            "waypoint": self.waypoint.to_dict() if self.waypoint else None,
            "waypoints": [wp.to_dict() for wp in self.waypoints] if self.waypoints else None,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "stop_after": self.stop_after,
            "gait": self.gait,
        }


@dataclass
class GaitConfig:
    stance_duration: float = 0.22      # Stance duration [s]
    swing_duration: float = 0.22       # Swing duration [s]
    swing_height: float = 0.06         # Peak foot lift in swing [m]
    stance_depth: float = 0.0          # Stance curve depth [m]
    max_step_length: float = 0.20      # Maximum allowed step length [m]
    body_height: float = NOMINAL_BODY_HEIGHT
    raibert_kv: float = 0.35           # Raibert velocity feedback gain (dimensionless)

    @property
    def stride_period(self) -> float:
        return self.stance_duration + self.swing_duration


ChampConfig = GaitConfig


@dataclass
class SafetyBounds:
    max_vx: float = 0.60               # Maximum forward velocity [m/s]
    min_vx: float = -0.30              # Maximum backward velocity [m/s]
    max_vy: float = 0.35               # Maximum lateral velocity [m/s]
    max_yaw_rate: float = 0.80         # Maximum yaw rate [rad/s]
    max_linear_accel: float = 1.20     # Maximum linear acceleration [m/s^2]
    max_yaw_accel: float = 2.50        # Maximum yaw acceleration [rad/s^2]
    max_roll_rad: float = 0.40         # ~23 deg max roll before emergency stand
    max_pitch_rad: float = 0.40        # ~23 deg max pitch before emergency stand
    min_height_m: float = 0.16         # Minimum base height [m]
    max_height_m: float = 0.38         # Maximum base height [m]
    max_duration: float = 60.0         # Maximum duration per command [s]
    max_stale_steps: int = 5           # Consecutive frozen step_time values tolerated before STALE_STATE_FEED


@dataclass
class SafetyFault:
    fault_type: str
    message: str
    step_time: float
    values: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Kinematics: 3-DOF Analytic IK and FK
# -----------------------------------------------------------------------------

def solve_ik(x: float, y: float, z: float, is_left: bool = False) -> Tuple[float, float, float]:
    """Solve Go2 hip-roll, thigh-pitch, and calf-pitch for a foot position in hip frame.

    Args:
        x: Foot X in hip frame (+X forward)
        y: Foot Y in hip frame (+Y left)
        z: Foot Z in hip frame (-Z down)
        is_left: True for FL/RL, False for FR/RR

    Returns:
        (theta_hip, theta_thigh, theta_calf) in radians
    """
    hip_offset = HIP_OFFSET if is_left else -HIP_OFFSET
    yz_squared = y * y + z * z
    diff = yz_squared - hip_offset * hip_offset
    if diff <= 0.0:
        raise ValueError(
            f"Foot target ({x:.3f}, {y:.3f}, {z:.3f}) inside hip roll singularity: diff={diff:.6f}"
        )

    theta_hip = math.atan2(y, -z) - math.atan2(hip_offset, math.sqrt(diff))
    z_planar = -math.sqrt(diff)

    distance_squared = x * x + z_planar * z_planar
    cosine_knee = (distance_squared - THIGH * THIGH - CALF * CALF) / (2.0 * THIGH * CALF)
    cosine_knee = float(np.clip(cosine_knee, -1.0, 1.0))
    theta_calf = -math.acos(cosine_knee)

    alpha = math.atan2(-x, -z_planar)
    beta = math.atan2(CALF * math.sin(theta_calf), THIGH + CALF * math.cos(theta_calf))
    theta_thigh = alpha - beta

    return theta_hip, theta_thigh, theta_calf


def forward_kinematics(theta_hip: float, theta_thigh: float, theta_calf: float, is_left: bool = False) -> Tuple[float, float, float]:
    """Calculate foot position (x, y, z) in hip frame from joint angles."""
    hip_offset = HIP_OFFSET if is_left else -HIP_OFFSET
    z_planar = -THIGH * math.cos(theta_thigh) - CALF * math.cos(theta_thigh + theta_calf)
    x = -THIGH * math.sin(theta_thigh) - CALF * math.sin(theta_thigh + theta_calf)
    y = hip_offset * math.cos(theta_hip) + (-z_planar) * math.sin(theta_hip)
    z = -(-hip_offset * math.sin(theta_hip) + (-z_planar) * math.cos(theta_hip))
    return x, y, z


# -----------------------------------------------------------------------------
# CHAMP 12-point Bézier Swing and Cosine Stance Planner
# -----------------------------------------------------------------------------

class ChampTrajectoryPlanner:
    """CHAMP's 12-control-point Bézier swing and cosine stance trajectory planner."""

    FACTORIAL = (
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
        362880.0, 3628800.0, 39916800.0, 479001600.0,
    )
    REF_CP_X = (
        -0.15, -0.2805, -0.3, -0.3, -0.3, 0.0,
        0.0, 0.0, 0.3032, 0.3032, 0.2826, 0.15,
    )
    REF_CP_Y = (
        -0.5, -0.5, -0.3611, -0.3611, -0.3611, -0.3611,
        -0.3611, -0.3214, -0.3214, -0.3214, -0.5, -0.5,
    )

    def __init__(self, swing_height: float = 0.06, stance_depth: float = 0.0):
        self.swing_height = swing_height
        self.stance_depth = stance_depth
        self.total_control_points = 12
        self.polynomial_degree = 11

        height_ratio = swing_height / 0.15
        self.control_points_y = [
            -((cp * height_ratio) + (0.5 * height_ratio))
            for cp in self.REF_CP_Y
        ]

    def generate(self, step_length: float, swing_phase: float, stance_phase: float) -> Tuple[float, float]:
        """Generate forward (x) and vertical (z) trajectory offset relative to nominal.

        Args:
            step_length: Full step displacement along the trajectory direction [m]
            swing_phase: Normalized swing phase in [0, 1]
            stance_phase: Normalized stance phase in [0, 1]

        Returns:
            (x_along_traj, z_vertical) offset in meters
        """
        if step_length <= 1e-6:
            return 0.0, 0.0

        if stance_phase > swing_phase:
            # Stance: move linearly backward from +step_length/2 to -step_length/2
            x = (step_length / 2.0) * (1.0 - 2.0 * stance_phase)
            z = -self.stance_depth * math.cos(math.pi * x / (step_length + 1e-9))
            return x, z

        if swing_phase > stance_phase:
            # Swing: 12-point Bézier curve
            length_ratio = step_length / 0.4
            cpx = [cp * length_ratio for cp in self.REF_CP_X]
            cpx[0] = -step_length / 2.0
            cpx[-1] = step_length / 2.0

            x = 0.0
            z = 0.0
            n = self.polynomial_degree
            for i in range(self.total_control_points):
                coeff = self.FACTORIAL[n] / (self.FACTORIAL[i] * self.FACTORIAL[n - i])
                bernstein = coeff * (swing_phase ** i) * ((1.0 - swing_phase) ** (n - i))
                x += bernstein * cpx[i]
                z -= bernstein * self.control_points_y[i]
            return x, z

        return 0.0, 0.0


# -----------------------------------------------------------------------------
# CHAMP Phase Generator
# -----------------------------------------------------------------------------

class ChampPhaseGenerator:
    """Trot oscillator using CHAMP's diagonal phase offsets: FR/RL together, FL/RR together."""

    def __init__(self, stance_duration: float = 0.22, swing_duration: float = 0.22):
        self.stance_duration = stance_duration
        self.swing_duration = swing_duration
        self.stride_period = stance_duration + swing_duration
        # Leg offsets: FR=0.0, FL=0.5, RR=0.5, RL=0.0
        self.offsets = (0.0, 0.5, 0.5, 0.0)
        self.phase_time = 0.0

    def reset(self):
        self.phase_time = 0.0

    def step(self, dt: float, is_moving: bool = True) -> Tuple[List[float], List[float]]:
        """Advance phase oscillator by dt and return stance/swing phases."""
        if not is_moving:
            return [0.0] * 4, [0.0] * 4
        self.phase_time += dt
        return self._compute_phases(self.phase_time)

    def run(self, elapsed_time: float, is_moving: bool = True) -> Tuple[List[float], List[float]]:
        """Return stance and swing phase values [0, 1] for all 4 legs given elapsed time."""
        if not is_moving:
            return [0.0] * 4, [0.0] * 4
        return self._compute_phases(elapsed_time)

    def _compute_phases(self, t: float) -> Tuple[List[float], List[float]]:
        stance_phases = [0.0] * 4
        swing_phases = [0.0] * 4
        stride_time = t % self.stride_period

        for index, offset in enumerate(self.offsets):
            # Wrap the leg clock into (-swing_duration, stance_duration] so every
            # sample falls in exactly one branch: swing covers (-swing, 0] and
            # stance covers (0, stance]. The wrap point leg_clock == -swing (the
            # start of swing) is mapped onto the end of stance, whose planner
            # output (rear extreme) matches the first swing tick — mapping it to
            # swing instead would yield swing_phase == 0.0, which the planner
            # renders as its degenerate (0, 0) nominal-centre snap.
            leg_clock = (stride_time - offset * self.stride_period) % self.stride_period
            if leg_clock > self.stance_duration:
                leg_clock -= self.stride_period

            if leg_clock > 0.0:
                stance_phases[index] = leg_clock / self.stance_duration
            else:
                swing_phases[index] = (leg_clock + self.swing_duration) / self.swing_duration

        return stance_phases, swing_phases


# -----------------------------------------------------------------------------
# Slew-Rate Limiter
# -----------------------------------------------------------------------------

class CommandSlewLimiter:
    """Smoothly ramps body velocities and yaw rate to avoid destabilizing step commands."""

    def __init__(self, bounds: SafetyBounds):
        self.bounds = bounds
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_yaw_rate = 0.0

    def reset(self, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0):
        self.current_vx = vx
        self.current_vy = vy
        self.current_yaw_rate = yaw_rate

    def step(self, target_vx: float, target_vy: float, target_yaw_rate: float, dt: float) -> Tuple[float, float, float]:
        max_lin_delta = self.bounds.max_linear_accel * dt
        max_yaw_delta = self.bounds.max_yaw_accel * dt

        delta_vx = float(np.clip(target_vx - self.current_vx, -max_lin_delta, max_lin_delta))
        delta_vy = float(np.clip(target_vy - self.current_vy, -max_lin_delta, max_lin_delta))
        delta_yaw = float(np.clip(target_yaw_rate - self.current_yaw_rate, -max_yaw_delta, max_yaw_delta))

        self.current_vx += delta_vx
        self.current_vy += delta_vy
        self.current_yaw_rate += delta_yaw

        # Clamp to absolute velocity limits
        self.current_vx = float(np.clip(self.current_vx, self.bounds.min_vx, self.bounds.max_vx))
        self.current_vy = float(np.clip(self.current_vy, -self.bounds.max_vy, self.bounds.max_vy))
        self.current_yaw_rate = float(np.clip(self.current_yaw_rate, -self.bounds.max_yaw_rate, self.bounds.max_yaw_rate))

        return self.current_vx, self.current_vy, self.current_yaw_rate

    def update(self, cmd: MotionCommand, dt: float) -> Tuple[float, float, float]:
        """Update slew limiter using a MotionCommand object."""
        return self.step(cmd.vx, cmd.vy, cmd.yaw_rate, dt)


# -----------------------------------------------------------------------------
# Omnidirectional Foothold Planner with Hip Yaw Coupling & Raibert Feedback
# -----------------------------------------------------------------------------

class FootholdPlanner:
    """Calculates 3D foot target coordinates in hip frames for omnidirectional locomotion."""

    def __init__(self, gait_config: GaitConfig):
        self.config = gait_config
        self.trajectory_planner = ChampTrajectoryPlanner(
            swing_height=gait_config.swing_height,
            stance_depth=gait_config.stance_depth,
        )

    def plan_leg_footholds(
        self,
        cmd_vx: float,
        cmd_vy: float,
        cmd_yaw_rate: float,
        meas_vx: float,
        meas_vy: float,
        meas_yaw_rate: float,
        stance_phases: List[float],
        swing_phases: List[float],
        stride_period: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[float], List[float]]:
        """Calculate joint angles and foot positions for all 4 legs.

        Args:
            stride_period: Stride period of the phase oscillator actually running.
                MultiGaitScheduler blends stance/swing durations during in-flight
                transitions, and WALK's period (0.80 s) differs from the static
                GaitConfig (0.44 s), so the caller passes the live period; the
                static config is only a fallback, which keeps the default TROT
                gait bit-identical.

        Returns:
            target_q: (12,) joint angle targets [rad]
            desired_feet_body: list of 4 foot positions in body frame [m]
            step_lengths: list of 4 step lengths [m]
            step_rotations: list of 4 trajectory rotations [rad]
        """
        stride_period = stride_period if stride_period is not None else self.config.stride_period
        target_q = []
        desired_feet_body = []
        step_lengths = []
        step_rotations = []

        for leg_index, is_left in enumerate(IS_LEFT):
            hip_pos = HIP_POSITIONS[leg_index]
            hip_x, hip_y = hip_pos[0], hip_pos[1]

            # Hip tangential velocity from body yaw rate
            v_hip_cmd_x = cmd_vx - cmd_yaw_rate * hip_y
            v_hip_cmd_y = cmd_vy + cmd_yaw_rate * hip_x

            v_hip_meas_x = meas_vx - meas_yaw_rate * hip_y
            v_hip_meas_y = meas_vy + meas_yaw_rate * hip_x

            # Raibert velocity feedback: augment commanded hip velocity with velocity error
            # to correct for slip and stance-phase shortening that cause measured speed to lag.
            v_error_x = v_hip_cmd_x - v_hip_meas_x
            v_error_y = v_hip_cmd_y - v_hip_meas_y
            v_eff_x = v_hip_cmd_x + self.config.raibert_kv * v_error_x
            v_eff_y = v_hip_cmd_y + self.config.raibert_kv * v_error_y

            v_hip_norm = math.sqrt(v_eff_x * v_eff_x + v_eff_y * v_eff_y)
            step_len = v_hip_norm * stride_period
            step_len = float(np.clip(step_len, 0.0, self.config.max_step_length))
            rotation = math.atan2(v_eff_y, v_eff_x) if v_hip_norm > 1e-5 else 0.0

            step_lengths.append(step_len)
            step_rotations.append(rotation)

            # Trajectory generator evaluates displacement along trajectory direction
            traj_x, traj_z = self.trajectory_planner.generate(
                step_len, swing_phases[leg_index], stance_phases[leg_index]
            )

            # Rotate trajectory displacement in horizontal plane
            foot_x_hip = traj_x * math.cos(rotation)
            foot_y_hip = traj_x * math.sin(rotation)
            foot_z_hip = -self.config.body_height + traj_z

            # Add lateral offset in hip frame
            nominal_y_hip = HIP_OFFSET if is_left else -HIP_OFFSET
            y_target_hip = nominal_y_hip + foot_y_hip

            # Solve IK for this leg
            theta_hip, theta_thigh, theta_calf = solve_ik(
                foot_x_hip, y_target_hip, foot_z_hip, is_left=is_left
            )
            target_q.extend([theta_hip, theta_thigh, theta_calf])

            # Foot position in body frame.  The lateral coordinate is the hip
            # roll axis plus the IK target's lateral component, which already
            # carries the nominal offset of the foot from the thigh pitch axis.
            foot_pos_body = (
                hip_pos[0] + foot_x_hip,
                hip_pos[1] + y_target_hip,
                foot_z_hip,
            )
            desired_feet_body.append(foot_pos_body)

        return np.asarray(target_q, dtype=float), desired_feet_body, step_lengths, step_rotations


# -----------------------------------------------------------------------------
# Closed-Loop Waypoint and Heading Navigation
# -----------------------------------------------------------------------------

def wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WaypointNavigator:
    """Computes body velocity and yaw rate commands to navigate through world waypoints."""

    def __init__(
        self,
        waypoints: Optional[List[Waypoint]] = None,
        k_heading: float = 1.5,
        k_dist: float = 0.8,
        max_yaw_rate: float = 0.6,
        max_speed: float = 0.35,
        k_yaw: Optional[float] = None,
    ):
        self.waypoints = list(waypoints) if waypoints else []
        self.current_index = 0
        self.k_heading = k_yaw if k_yaw is not None else k_heading
        self.k_dist = k_dist
        self.max_yaw_rate = max_yaw_rate
        self.max_speed = max_speed
        self.is_completed = False

    def reset(self, waypoints: Optional[List[Waypoint]] = None):
        if waypoints is not None:
            self.waypoints = list(waypoints)
        self.current_index = 0
        self.is_completed = len(self.waypoints) == 0

    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    def compute_velocity_command(
        self,
        current_pos: np.ndarray,
        current_yaw: float,
        waypoint: Optional[Waypoint] = None,
    ) -> Tuple[float, float, float, bool]:
        """Compute command from 3D numpy position vector and optional target waypoint."""
        if waypoint is not None:
            self.waypoints = [waypoint]
            self.current_index = 0
            self.is_completed = False

        vx, vy, yaw_rate, arrived = self.compute_command(
            float(current_pos[0]), float(current_pos[1]), current_yaw
        )
        # When arrived, ensure velocities are 0.0
        if arrived:
            return 0.0, 0.0, 0.0, True
        return vx, vy, yaw_rate, arrived

    def compute_command(
        self, current_x: float, current_y: float, current_yaw: float
    ) -> Tuple[float, float, float, bool]:
        """Compute (vx, vy, yaw_rate, arrived) for the current robot state.

        Returns:
            (vx, vy, yaw_rate, is_completed)
        """
        wp = self.current_waypoint()
        if wp is None:
            self.is_completed = True
            return 0.0, 0.0, 0.0, True

        dx = wp.x - current_x
        dy = wp.y - current_y
        dist = math.sqrt(dx * dx + dy * dy)

        # Check if current waypoint reached
        if dist <= wp.tolerance:
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.is_completed = True
                return 0.0, 0.0, 0.0, True
            wp = self.waypoints[self.current_index]
            dx = wp.x - current_x
            dy = wp.y - current_y
            dist = math.sqrt(dx * dx + dy * dy)

        # Desired heading towards waypoint
        target_heading = math.atan2(dy, dx)
        heading_err = wrap_to_pi(target_heading - current_yaw)

        # Yaw rate command proportional to heading error
        yaw_rate = float(np.clip(self.k_heading * heading_err, -self.max_yaw_rate, self.max_yaw_rate))

        # Speed control: slow down during sharp turns, speed up when aligned
        speed_limit = min(wp.speed, self.max_speed)
        alignment_factor = max(0.0, math.cos(heading_err))

        if abs(heading_err) > 0.45:
            # Significant turn needed: rotate in place to align with target
            vx = 0.0
        else:
            # Well aligned: drive towards target with distance deceleration
            vx = float(np.clip(self.k_dist * dist * alignment_factor, 0.10, speed_limit))

        vy = 0.0
        return vx, vy, yaw_rate, False


# -----------------------------------------------------------------------------
# Parametric Trajectory Tracker
# -----------------------------------------------------------------------------

class TrajectoryTracker:
    """Computes body velocity and yaw rate commands to follow parametric trajectories."""

    def __init__(self, spec: Optional[TrajectorySpec] = None, k_pos: float = 1.2, max_speed: float = 0.35, max_yaw_rate: float = 0.6):
        self.spec = spec
        self.start_time = 0.0           # Sim time at which this trajectory was activated [s]
        self.k_pos = k_pos
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate

    def reset(self, spec: Optional[TrajectorySpec] = None, start_time: float = 0.0):
        if spec is not None:
            self.spec = spec
        self.start_time = start_time

    def evaluate_path(self, t: float) -> TrajectoryPoint:
        """Evaluate desired position, velocity, and heading at time t."""
        if not self.spec:
            return TrajectoryPoint(time_s=t, x=0.0, y=0.0)

        path_type = self.spec.path_type.lower()
        p = self.spec.params
        duration = self.spec.duration

        # Normalized progress [0, 1] or clamped time
        t_clamped = max(0.0, min(t, duration))

        if path_type == "circle":
            cx = p.get("cx", 0.0)
            cy = p.get("cy", 0.0)
            r = p.get("radius", 1.0)
            speed = p.get("speed", 0.30)
            # Angular frequency omega = speed / radius
            omega = speed / max(r, 0.05)
            theta = omega * t_clamped

            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            vx = -r * omega * math.sin(theta)
            vy = r * omega * math.cos(theta)
            yaw = theta + math.pi / 2.0
            yaw_rate = omega
            return TrajectoryPoint(time_s=t, x=x, y=y, vx=vx, vy=vy, yaw_rate=yaw_rate, yaw=yaw)

        elif path_type == "figure_8":
            cx = p.get("cx", 0.0)
            cy = p.get("cy", 0.0)
            a = p.get("scale_x", 1.5)
            b = p.get("scale_y", 0.75)
            omega = p.get("omega", 1.0)
            theta = omega * t_clamped

            # Lemniscate of Gerono / parametric figure 8
            # x = cx + a * sin(theta)
            # y = cy + b * sin(theta) * cos(theta)
            s_th = math.sin(theta)
            c_th = math.cos(theta)

            x = cx + a * s_th
            y = cy + b * s_th * c_th

            vx = a * omega * c_th
            vy = b * omega * (c_th * c_th - s_th * s_th)

            # Heading aligned with velocity vector
            yaw = math.atan2(vy, vx) if abs(vx) > 1e-4 or abs(vy) > 1e-4 else 0.0
            yaw_rate = 0.0
            return TrajectoryPoint(time_s=t, x=x, y=y, vx=vx, vy=vy, yaw_rate=yaw_rate, yaw=yaw)

        else:
            raise ValueError(
                f"Unsupported trajectory path_type {self.spec.path_type!r}; "
                f"supported types: {', '.join(TRAJECTORY_PATH_TYPES)}"
            )

    def compute_command(
        self, current_x: float, current_y: float, current_yaw: float, current_time: float
    ) -> Tuple[float, float, float, bool]:
        """Compute body-frame velocities (vx, vy, yaw_rate) to track the trajectory."""
        if not self.spec:
            return 0.0, 0.0, 0.0, True

        # Elapsed trajectory time (t - t0); completion is measured against this
        # rather than the absolute simulation clock.
        elapsed = current_time - self.start_time
        if elapsed > self.spec.duration:
            return 0.0, 0.0, 0.0, True

        t_rel = max(0.0, min(elapsed, self.spec.duration))
        ref = self.evaluate_path(t_rel)

        # Position error in world frame
        ex = ref.x - current_x
        ey = ref.y - current_y

        # Transform feedforward + feedback velocity into body frame
        # v_world = [ref.vx + k_pos * ex, ref.vy + k_pos * ey]
        vw_x = ref.vx + self.k_pos * ex
        vw_y = ref.vy + self.k_pos * ey

        c_yaw = math.cos(current_yaw)
        s_yaw = math.sin(current_yaw)

        # Rotation matrix world -> body
        vx = c_yaw * vw_x + s_yaw * vw_y
        vy = -s_yaw * vw_x + c_yaw * vw_y

        # Clip linear velocities
        vx = float(np.clip(vx, -self.max_speed, self.max_speed))
        vy = float(np.clip(vy, -self.max_speed, self.max_speed))

        # Yaw rate tracking - use path tangent yaw (ref.yaw), not the erroneous yaw_rate reference
        heading_err = wrap_to_pi(ref.yaw - current_yaw)
        yaw_rate = float(np.clip(ref.yaw_rate + 1.5 * heading_err, -self.max_yaw_rate, self.max_yaw_rate))

        is_completed = elapsed >= self.spec.duration
        return vx, vy, yaw_rate, is_completed


# -----------------------------------------------------------------------------
# Safety Supervisor
# -----------------------------------------------------------------------------

class SafetySupervisor:
    """Monitors robot attitude, height, joint limits, numerical state, and state-feed health.

    The state-feed watchdog treats ``step_time`` as a contract: it must strictly
    advance. A frozen feed means the controller would otherwise keep commanding
    the last pose forever with no audit entry, so a non-advancing feed is raised
    as ``STALE_STATE_FEED`` and a backwards feed as ``NON_MONOTONIC_STEP_TIME``.

    No wall-clock deadline check is performed. Two of the three hosts run
    deliberately unpaced (``evaluate_champ_trajectories.py`` runs as fast as it
    can, and ``go2_champ_llm_cli.py`` is only paced in GUI mode), so a real-time
    budget would be exceeded by every healthy headless run and would fault
    constantly. The step_time checks below are pure functions of the state
    contract and are correct regardless of pacing.
    """

    def __init__(self, bounds: SafetyBounds):
        self.bounds = bounds
        self.faults: List[SafetyFault] = []
        self._last_step_time: Optional[float] = None
        self._stale_steps = 0

    def reset(self):
        """Clear recorded faults and re-arm the state-feed watchdog."""
        self.faults.clear()
        self._last_step_time = None
        self._stale_steps = 0

    def check_state(
        self,
        step_time: float,
        base_z: float,
        roll: float,
        pitch: float,
        joint_positions: np.ndarray,
    ) -> Optional[SafetyFault]:
        # State-feed watchdog. Checked first so the clock bookkeeping always runs
        # before any other check returns.
        if self._last_step_time is not None:
            if step_time < self._last_step_time:
                fault = SafetyFault(
                    fault_type="NON_MONOTONIC_STEP_TIME",
                    message=(
                        f"Sim clock went backwards: step_time {step_time:.6f}s < "
                        f"previous {self._last_step_time:.6f}s"
                    ),
                    step_time=step_time,
                    values={"step_time": step_time, "previous_step_time": self._last_step_time},
                )
                self.faults.append(fault)
                self._last_step_time = step_time
                self._stale_steps = 0
                return fault

            if step_time == self._last_step_time:
                self._stale_steps += 1
                # Fault once per episode (on the crossing), not on every tick, so a
                # permanently frozen feed cannot flood the audit list.
                if self._stale_steps == self.bounds.max_stale_steps:
                    fault = SafetyFault(
                        fault_type="STALE_STATE_FEED",
                        message=(
                            f"step_time frozen at {step_time:.6f}s for "
                            f"{self._stale_steps} consecutive calls "
                            f"(limit {self.bounds.max_stale_steps}): state feed appears stalled"
                        ),
                        step_time=step_time,
                        values={"step_time": step_time, "stale_steps": float(self._stale_steps)},
                    )
                    self.faults.append(fault)
                    return fault
            else:
                self._stale_steps = 0

        self._last_step_time = step_time

        # Check finite values
        if not (math.isfinite(base_z) and math.isfinite(roll) and math.isfinite(pitch)):
            fault = SafetyFault(
                fault_type="NON_FINITE_STATE",
                message=f"Non-finite state: z={base_z}, roll={roll}, pitch={pitch}",
                step_time=step_time,
            )
            self.faults.append(fault)
            return fault

        # Check base height
        if not (self.bounds.min_height_m <= base_z <= self.bounds.max_height_m):
            fault = SafetyFault(
                fault_type="HEIGHT_LIMIT_VIOLATION",
                message=f"Base height {base_z:.3f}m out of bounds [{self.bounds.min_height_m}, {self.bounds.max_height_m}]",
                step_time=step_time,
                values={"base_z": base_z},
            )
            self.faults.append(fault)
            return fault

        # Check roll / pitch tilt
        if abs(roll) > self.bounds.max_roll_rad:
            fault = SafetyFault(
                fault_type="ROLL_LIMIT_VIOLATION",
                message=f"Base roll {math.degrees(roll):.1f} deg exceeded max {math.degrees(self.bounds.max_roll_rad):.1f} deg",
                step_time=step_time,
                values={"roll": roll},
            )
            self.faults.append(fault)
            return fault

        if abs(pitch) > self.bounds.max_pitch_rad:
            fault = SafetyFault(
                fault_type="PITCH_LIMIT_VIOLATION",
                message=f"Base pitch {math.degrees(pitch):.1f} deg exceeded max {math.degrees(self.bounds.max_pitch_rad):.1f} deg",
                step_time=step_time,
                values={"pitch": pitch},
            )
            self.faults.append(fault)
            return fault

        # Check joint limits
        joint_types = ("hip", "thigh", "calf")
        for i, q in enumerate(joint_positions):
            jtype = joint_types[i % 3]
            jmin, jmax = JOINT_LIMITS[jtype]
            margin = 0.05  # 0.05 rad tolerance
            if not (jmin - margin <= q <= jmax + margin):
                fault = SafetyFault(
                    fault_type="JOINT_LIMIT_VIOLATION",
                    message=f"Joint {JOINT_NAMES[i]} position {q:.3f} outside limit [{jmin}, {jmax}]",
                    step_time=step_time,
                    values={"joint_index": i, "position": q},
                )
                self.faults.append(fault)
                return fault

        return None


# -----------------------------------------------------------------------------
# Go2 Neutral Standing & Kinematics Helpers
# -----------------------------------------------------------------------------

def neutral_joint_targets(body_height: float = NOMINAL_BODY_HEIGHT) -> np.ndarray:
    """Return the 12 neutral standing joint angles."""
    targets = []
    for is_left in IS_LEFT:
        lateral = HIP_OFFSET if is_left else -HIP_OFFSET
        targets.extend(solve_ik(0.0, lateral, -body_height, is_left))
    return np.asarray(targets, dtype=float)


DEFAULT_NEUTRAL_Q = neutral_joint_targets(NOMINAL_BODY_HEIGHT)


def quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert a MuJoCo wxyz unit quaternion to roll, pitch, yaw [rad]."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array((roll, pitch, yaw), dtype=float)


def configure_model(model: Any, kp: float = POSITION_KP, kd: float = POSITION_KD, gravity: float = GRAVITY):
    """Configure model gravity and actuator PD gains (Kp=POSITION_KP, Kd=POSITION_KD by default)."""
    model.opt.gravity[:] = (0.0, 0.0, -gravity)
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd
    # Publish the model's true total mass for the contact QP's wrench target, so
    # the solver reads the XML rather than a constant that can drift from it.
    # Imported lazily: go2_balance_qp imports from this module at top level, so
    # a top-level reverse import would be circular.
    from go2_balance_qp import model_total_mass
    return model_total_mass(model)


def controller_for_model(model: Any, **kwargs) -> "Go2ChampController":
    """Build a Go2ChampController whose QP uses this model's true total mass.

    The contact QP wrench target is b = [m*(a + g); I*alpha], so an error in the
    mass scales the whole linear wrench. Reading it from the model keeps the QP
    honest without a per-step cost.
    """
    from go2_balance_qp import model_total_mass
    return Go2ChampController(robot_mass=model_total_mass(model), **kwargs)


def initialize_simulation_standing(model: Any, data: Any, body_height: float = NOMINAL_BODY_HEIGHT, settle_time: float = 0.20):
    """Deterministically place the Go2 in standing pose and settle contacts smoothly.

    Prevents free-fall impact bounces from the default floating XML pose.
    """
    import mujoco

    neutral_q = neutral_joint_targets(body_height)

    # Base position and upright orientation
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = body_height
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Assign joint positions
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
    actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
    qpos_addrs = model.jnt_qposadr[joint_ids]

    for addr, q in zip(qpos_addrs, neutral_q):
        data.qpos[addr] = q

    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = neutral_q

    # Update forward kinematics and dynamics
    mujoco.mj_forward(model, data)

    # Micro-settle for contact forces to establish
    settle_steps = max(1, int(round(settle_time / model.opt.timestep)))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)


# -----------------------------------------------------------------------------
# Go2 CHAMP Locomotion Controller
# -----------------------------------------------------------------------------

class Go2ChampController:
    """High-level Go2 CHAMP controller integrating phase, trajectory, slew limiting, and safety."""

    def __init__(
        self,
        gait_config: Optional[GaitConfig] = None,
        safety_bounds: Optional[SafetyBounds] = None,
        config: Optional[GaitConfig] = None,
        bounds: Optional[SafetyBounds] = None,
        robot_mass: Optional[float] = None,
    ):
        self.gait_config = gait_config or config or GaitConfig()
        self.safety_bounds = safety_bounds or bounds or SafetyBounds()
        # Total robot mass [kg] used by the contact QP wrench target. Filled from
        # the loaded MuJoCo model by configure_model(); None means "use the
        # module constant". Keeping it an attribute rather than re-reading the
        # model each solve avoids a per-step cost in the 500 Hz loop.
        self.robot_mass = robot_mass

        try:
            from go2_multi_gait import MultiGaitScheduler

            # Forward the configured duty split only when it actually deviates from
            # the TROT preset. Unconditionally forwarding it would override every
            # other gait's own timing (e.g. WALK's 0.60/0.20 -> 0.80 s period),
            # breaking per-gait profiles and the live-period plumbing. Default
            # GaitConfig values are bit-identical to the TROT preset, so this keeps
            # the shipped behaviour unchanged while making a tuned duty factor live.
            from go2_multi_gait import GAIT_PRESETS, GaitType as _GaitType

            _trot = GAIT_PRESETS[_GaitType.TROT]
            _cfg_dev_from_trot = (
                abs(self.gait_config.stance_duration - _trot.stance_duration) > 1e-12
                or abs(self.gait_config.swing_duration - _trot.swing_duration) > 1e-12
            )

            self.phase_generator = MultiGaitScheduler(
                stance_duration=self.gait_config.stance_duration,
                swing_duration=self.gait_config.swing_duration,
            ) if _cfg_dev_from_trot else MultiGaitScheduler()
        except ImportError:
            self.phase_generator = ChampPhaseGenerator(
                stance_duration=self.gait_config.stance_duration,
                swing_duration=self.gait_config.swing_duration,
            )
        self.foothold_planner = FootholdPlanner(self.gait_config)
        self.slew_limiter = CommandSlewLimiter(self.safety_bounds)
        self.waypoint_navigator = WaypointNavigator()
        self.safety_supervisor = SafetySupervisor(self.safety_bounds)
        self._trajectory_tracker = TrajectoryTracker()
        self._qp_solver = None  # Lazy-initialized in compute_stance_torques

        # Resolved once here rather than lazily on the control tick, so the first
        # obstacle-bearing step pays no import/construction cost. Falls back to
        # None when the optional vision stack (or its mujoco dependency) is absent.
        self._obstacle_avoidance = self._make_obstacle_avoidance()

        self.neutral_targets = neutral_joint_targets(self.gait_config.body_height)
        self.mode = MotionMode.STAND
        self.emergency_fault: Optional[SafetyFault] = None
        self._active_waypoint: Optional[Waypoint] = None
        self._active_trajectory: Optional[TrajectorySpec] = None
        self._completed_trajectory: Optional[TrajectorySpec] = None
        self.faults_at_last_reset: List[SafetyFault] = []

    @staticmethod
    def _make_obstacle_avoidance():
        try:
            from go2_vision_perception import ReactiveObstacleAvoidance
        except ImportError:
            return None
        return ReactiveObstacleAvoidance()

    @property
    def obstacle_avoidance(self):
        if self._obstacle_avoidance is None:
            self._obstacle_avoidance = self._make_obstacle_avoidance()
        return self._obstacle_avoidance

    @property
    def trajectory_tracker(self):
        return self._trajectory_tracker

    def _live_stride_period(self) -> float:
        """Stride period of the phase oscillator actually executing.

        MultiGaitScheduler keeps its own blended stance/swing durations, so the
        value differs from the static GaitConfig during an in-flight transition
        (and permanently for WALK: 0.80 s vs 0.44 s). The foot trajectory must use
        the live period or the feet land short / over-reach relative to the gait
        being executed. Falls back to the static config when the oscillator exposes
        no live period (e.g. the ChampPhaseGenerator fallback), which keeps default
        TROT bit-identical since there blended == config.
        """
        stance = getattr(self.phase_generator, "current_stance_dur", None)
        swing = getattr(self.phase_generator, "current_swing_dur", None)
        if stance is None or swing is None:
            return self.gait_config.stride_period
        return float(stance) + float(swing)

    def reset(self):
        # Preserve the faults that justified this reset: re-enabling motion must
        # not silently discard the evidence for why it was stopped.
        self.faults_at_last_reset = list(self.safety_supervisor.faults)
        if self.faults_at_last_reset:
            logger.warning(
                "Controller reset clearing %d outstanding fault(s): %s",
                len(self.faults_at_last_reset),
                [f.fault_type for f in self.faults_at_last_reset],
            )

        self.slew_limiter.reset()
        self.mode = MotionMode.STAND
        self.emergency_fault = None
        self.safety_supervisor.reset()
        self.phase_generator.reset()
        self._active_waypoint = None
        self._active_trajectory = None
        self._completed_trajectory = None
        if self._trajectory_tracker.spec is not None:
            self._trajectory_tracker.reset(self._trajectory_tracker.spec)

    def step(
        self,
        command: MotionCommand,
        step_time: float,
        dt: float,
        base_pos: np.ndarray,
        base_euler: np.ndarray,
        base_linvel_body: np.ndarray,
        base_angvel_body: np.ndarray,
        joint_positions: np.ndarray,
        sector_clearance: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[float], List[float], Dict[str, Any]]:
        """Fail-safe wrapper around :meth:`_step_impl`.

        Any unexpected internal failure (e.g. an IK singularity raising from the
        foothold planner) is converted into a recorded emergency fault and a
        hold-current-pose response, so the exception never escapes with a stale
        actuator command still standing. The fault is logged and surfaced in the
        returned info["fault"] as well as ``self.emergency_fault``.
        """
        try:
            return self._step_impl(
                command, step_time, dt, base_pos, base_euler,
                base_linvel_body, base_angvel_body, joint_positions, sector_clearance,
            )
        except Exception as exc:
            fault = SafetyFault(
                fault_type="CONTROL_EXCEPTION",
                message=f"{type(exc).__name__} during step(): {exc}",
                step_time=step_time,
            )
            self.safety_supervisor.faults.append(fault)
            self.emergency_fault = fault
            self.mode = MotionMode.EMERGENCY_STAND
            logger.error("Controller step failed (t=%.3fs): %s", step_time, fault.message, exc_info=True)
            return self._emergency_hold(dt, joint_positions, fault)

    def _emergency_hold(
        self,
        dt: float,
        joint_positions: np.ndarray,
        fault: Optional[SafetyFault],
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[float], List[float], Dict[str, Any]]:
        """Freeze at the current measured pose with zero commanded velocity.

        This is the conservative safe state: the controller holds no model handle
        here, so it cannot switch the actuators to a pure damping / kp=0 mode.
        Commanding the neutral pose instead would drive a robot at nonzero roll,
        or with a leg near a joint limit, toward a pose we cannot know is safe.
        """
        cmd_vx, cmd_vy, cmd_yaw = self.slew_limiter.step(0.0, 0.0, 0.0, dt)
        target_q = np.asarray(joint_positions, dtype=float).copy()
        desired_feet = []
        for i in range(4):
            foot_x, foot_y, foot_z = forward_kinematics(
                float(target_q[3 * i]), float(target_q[3 * i + 1]), float(target_q[3 * i + 2]),
                is_left=IS_LEFT[i],
            )
            desired_feet.append(
                (HIP_POSITIONS[i][0] + foot_x, HIP_POSITIONS[i][1] + foot_y, foot_z)
            )
        info = {
            "cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "cmd_yaw": cmd_yaw,
            "mode": self.mode.value,
            "fault": fault.fault_type if fault else None,
            "obstacle_status": "none",
        }
        return target_q, desired_feet, [0.0] * 4, [0.0] * 4, info

    def _step_impl(
        self,
        command: MotionCommand,
        step_time: float,
        dt: float,
        base_pos: np.ndarray,
        base_euler: np.ndarray,
        base_linvel_body: np.ndarray,
        base_angvel_body: np.ndarray,
        joint_positions: np.ndarray,
        sector_clearance: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[float], List[float], Dict[str, Any]]:
        """Compute the next joint angle control targets and telemetry info.

        Args:
            command: Requested MotionCommand
            step_time: Current simulation elapsed time [s]
            dt: Simulation timestep [s]
            base_pos: Base position in world frame (x, y, z) [m]
            base_euler: Base orientation (roll, pitch, yaw) [rad]
            base_linvel_body: Measured linear velocity in body frame [m/s]
            base_angvel_body: Measured angular velocity in body frame [rad/s]
            joint_positions: Measured 12 joint angles [rad]
            sector_clearance: Optional SectorClearance from vision perception pipeline

        Returns:
            target_q: (12,) array of commanded joint angles [rad]
            desired_feet: list of 4 foot positions in body frame [m]
            stance_phases: list of 4 stance phases [0, 1]
            swing_phases: list of 4 swing phases [0, 1]
            info: metadata dictionary with clamped velocities, mode, faults
        """
        # Safety check first
        fault = self.safety_supervisor.check_state(
            step_time=step_time,
            base_z=float(base_pos[2]),
            roll=float(base_euler[0]),
            pitch=float(base_euler[1]),
            joint_positions=joint_positions,
        )
        if fault is not None:
            self.mode = MotionMode.EMERGENCY_STAND
            self.emergency_fault = fault

        # Emergency stand: freeze at the measured pose rather than driving to neutral
        if self.mode == MotionMode.EMERGENCY_STAND:
            return self._emergency_hold(dt, joint_positions, self.emergency_fault)

        # Clamp user command against safety limits
        clamped_cmd = command.clamped(self.safety_bounds)

        # A non-trajectory command clears the completed-trajectory latch, so the
        # caller can explicitly re-run a path after leaving trajectory mode.
        if not (command.mode == MotionMode.TRAJECTORY and clamped_cmd.trajectory is not None):
            self._completed_trajectory = None

        # Determine target velocities based on mode
        if command.mode == MotionMode.WAYPOINT:
            if clamped_cmd.waypoints is not None:
                if self.waypoint_navigator.waypoints != clamped_cmd.waypoints:
                    self.waypoint_navigator.reset(clamped_cmd.waypoints)
                    self._active_waypoint = self.waypoint_navigator.current_waypoint()
            elif clamped_cmd.waypoint is not None:
                if self.waypoint_navigator.current_waypoint() != clamped_cmd.waypoint:
                    self._active_waypoint = clamped_cmd.waypoint
                    self.waypoint_navigator.reset([clamped_cmd.waypoint])

            if self.waypoint_navigator.current_waypoint() is not None:
                wp_vx, wp_vy, wp_yaw, is_arrived = self.waypoint_navigator.compute_command(
                    current_x=float(base_pos[0]),
                    current_y=float(base_pos[1]),
                    current_yaw=float(base_euler[2]),
                )
                req_vx, req_vy, req_yaw = wp_vx, wp_vy, wp_yaw
                self.mode = MotionMode.STAND if is_arrived else MotionMode.WAYPOINT
            else:
                req_vx, req_vy, req_yaw = 0.0, 0.0, 0.0
                self.mode = MotionMode.STAND
        elif command.mode == MotionMode.STAND:
            req_vx, req_vy, req_yaw = 0.0, 0.0, 0.0
            self.mode = MotionMode.STAND
        elif command.mode == MotionMode.TRAJECTORY and clamped_cmd.trajectory is not None:
            spec = clamped_cmd.trajectory
            if self._completed_trajectory is not None and (
                spec is self._completed_trajectory or spec == self._completed_trajectory
            ):
                # A path that already ran to completion must not silently re-run;
                # hold stand until the caller supplies a new path or another mode.
                req_vx, req_vy, req_yaw = 0.0, 0.0, 0.0
                self.mode = MotionMode.STAND
            else:
                # Adopt a fresh start time only when the trajectory is newly
                # activated, so elapsed trajectory time keeps advancing while the
                # same path stays active. Callers may rebuild an equal spec every
                # tick, hence the identity-and-value comparison.
                is_new = self.mode != MotionMode.TRAJECTORY
                if not is_new and spec is not self._active_trajectory and spec != self._active_trajectory:
                    is_new = True
                if is_new:
                    # Never mutate the caller's spec; the tracker owns the start time.
                    self._active_trajectory = spec
                    self._trajectory_tracker.reset(spec, start_time=step_time)
                self.mode = MotionMode.TRAJECTORY
                # Compute trajectory-based velocities
                req_vx, req_vy, req_yaw, is_completed = self._trajectory_tracker.compute_command(
                    current_x=float(base_pos[0]),
                    current_y=float(base_pos[1]),
                    current_yaw=float(base_euler[2]),
                    current_time=step_time
                )
                if is_completed:
                    self._completed_trajectory = spec
                    self.mode = MotionMode.STAND
        else:
            req_vx = clamped_cmd.vx
            req_vy = clamped_cmd.vy
            req_yaw = clamped_cmd.yaw_rate
            self.mode = MotionMode.VELOCITY

        # Heading target closed-loop regulation if specified
        if clamped_cmd.heading_target is not None and command.mode != MotionMode.WAYPOINT:
            heading_err = wrap_to_pi(clamped_cmd.heading_target - float(base_euler[2]))
            req_yaw = float(np.clip(1.5 * heading_err, -self.safety_bounds.max_yaw_rate, self.safety_bounds.max_yaw_rate))

        # Reactive obstacle avoidance if clearance information is provided
        obstacle_status = "none"
        if sector_clearance is not None and getattr(sector_clearance, "obstacle_detected", False):
            req_vx, req_vy, req_yaw, obstacle_status = self.obstacle_avoidance.modulate_commands(
                req_vx, req_vy, req_yaw, sector_clearance
            )

        # Apply slew-rate limiter
        cmd_vx, cmd_vy, cmd_yaw = self.slew_limiter.step(req_vx, req_vy, req_yaw, dt)

        is_moving = (
            abs(cmd_vx) > MOTION_EPS_LINEAR
            or abs(cmd_vy) > MOTION_EPS_LINEAR
            or abs(cmd_yaw) > MOTION_EPS_YAW
        )

        # Apply requested gait if supported
        if hasattr(self.phase_generator, "set_gait") and getattr(clamped_cmd, "gait", None):
            self.phase_generator.set_gait(clamped_cmd.gait)

        # Update phase oscillator
        stance_phases, swing_phases = self.phase_generator.step(dt, is_moving=is_moving)

        if not is_moving:
            # Stand in neutral pose
            target_q = self.neutral_targets.copy()
            desired_feet = [
                (HIP_POSITIONS[i][0], HIP_POSITIONS[i][1] + (HIP_OFFSET if IS_LEFT[i] else -HIP_OFFSET), -self.gait_config.body_height)
                for i in range(4)
            ]
        else:
            # Compute omnidirectional foot trajectories
            target_q, desired_feet, step_lengths, step_rotations = self.foothold_planner.plan_leg_footholds(
                cmd_vx=cmd_vx,
                cmd_vy=cmd_vy,
                cmd_yaw_rate=cmd_yaw,
                meas_vx=float(base_linvel_body[0]),
                meas_vy=float(base_linvel_body[1]),
                meas_yaw_rate=float(base_angvel_body[2]),
                stance_phases=stance_phases,
                swing_phases=swing_phases,
                stride_period=self._live_stride_period(),
            )

        active_gait_name = getattr(self.phase_generator, "current_gait", "trot")
        if hasattr(active_gait_name, "value"):
            active_gait_name = active_gait_name.value

        info = {
            "cmd_vx": cmd_vx,
            "cmd_vy": cmd_vy,
            "cmd_yaw": cmd_yaw,
            "mode": self.mode.value,
            "fault": None,
            "is_moving": is_moving,
            "obstacle_status": obstacle_status,
            "gait": str(active_gait_name),
        }
        return target_q, desired_feet, stance_phases, swing_phases, info

    def compute_stance_torques(
        self,
        joint_positions,
        desired_feet: List[Tuple[float, float, float]],
        contacts: List[bool],
        accel_lin=None,
        accel_ang=None,
        return_diag: bool = False,
    ):
        """Computes optimal contact torques tau in R^12 via leg Jacobian transpose.

        Args:
            joint_positions: 12-element measured joint angle array [rad]
            desired_feet: List of 4 desired foot positions in body frame [m]
            contacts: List of 4 booleans indicating active stance contact
            accel_lin: Desired linear acceleration [ax, ay, az] in body frame
            accel_ang: Desired angular acceleration [alphax, alphay, alphaz] in body frame

        Returns:
            torques: 12-element joint torque array [N*m]

        Raises:
            ValueError: if no leg is in contact. There is then no contact-force
                solution to map to joint torques, and returning an all-zero vector
                would be indistinguishable from a legitimate inert command.

        Note:
            ``go2_description.xml`` declares these actuators as MuJoCo
            ``<position kp=... kv=...>`` servos, so ``data.ctrl`` holds a target
            *angle* and the live loops write ``target_q`` (a pose) to it — this
            torque vector is not what drives ``data.ctrl`` today. An all-zero
            return therefore does NOT mean "go limp"; the no-contact case is
            refused explicitly rather than silently fabricating a command.
        """
        from go2_balance_qp import (SingleRigidBodyQpSolver, QpBalanceConfig,
                                     compute_leg_jacobian)
        if not hasattr(self, "_qp_solver") or self._qp_solver is None:
            cfg = None
            raw_mass = getattr(self, "robot_mass", None)
            if raw_mass:
                cfg = QpBalanceConfig(mass=float(raw_mass))
            self._qp_solver = SingleRigidBodyQpSolver(cfg)

        active_contacts = [i for i, contact in enumerate(contacts) if contact]
        if not active_contacts:
            logger.warning(
                "compute_stance_torques called with no legs in contact; refusing to "
                "return an all-zero torque vector. contacts=%s", list(contacts),
            )
            raise ValueError(
                "compute_stance_torques requires at least one leg in contact; "
                f"got contacts={list(contacts)}"
            )

        feet_vecs = [np.array(f, dtype=float) for f in desired_feet]
        a_lin = accel_lin if accel_lin is not None else np.zeros(3, dtype=float)
        a_ang = accel_ang if accel_ang is not None else np.zeros(3, dtype=float)

        forces = self._qp_solver.solve(feet_vecs, contacts, a_lin, a_ang)

        torques = np.zeros(12, dtype=float)
        for i in range(4):
            if contacts[i]:
                th_hip = float(joint_positions[i * 3 + 0])
                th_thigh = float(joint_positions[i * 3 + 1])
                th_calf = float(joint_positions[i * 3 + 2])
                J = compute_leg_jacobian(th_hip, th_thigh, th_calf, is_left=IS_LEFT[i])
                tau_leg = J.T @ forces[i]
                torques[i * 3 : i * 3 + 3] = tau_leg

        if not return_diag:
            return torques

        # Primal residual, iteration count, and whether any bound was active are
        # exactly the quantities needed to show the solve is a solve rather than
        # a projection, so surface them alongside the torque vector. Callers that
        # ignore this (the default) keep the old single-value return.
        diag = {
            "qp_residual_norm": float(self._qp_solver.last_residual_norm),
            "qp_residual": np.asarray(self._qp_solver.last_residual, dtype=float).tolist(),
            "qp_iterations": int(self._qp_solver.last_iterations),
            "qp_bounds_active": bool(self._qp_solver.last_bounds_active),
            "qp_inputs_sanitized": bool(self._qp_solver.last_inputs_sanitized),
            # Friction-cone utilisation per active leg, sampled pre-projection:
            # > 1.0 means the stance asked for tangential force the cone cannot
            # supply and it had to be clipped. Not measured from the returned
            # forces, which are always feasible by construction.
            "qp_friction_utilisation": [
                float(v) for v in self._qp_solver.last_friction_utilisation
            ],
            "qp_friction_violated": [
                bool(v) for v in self._qp_solver.last_friction_violated
            ],
            "qp_active": True,
        }
        return torques, diag
