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

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


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

LEG_NAMES = ("FR", "FL", "RR", "RL")
IS_LEFT = (False, True, False, True)

# Hip roll joint positions relative to body center [m]
HIP_POSITIONS = np.array(
    [
        [0.1934, -0.0955, 0.0],   # FR
        [0.1934,  0.0955, 0.0],   # FL
        [-0.1934, -0.0955, 0.0],  # RR
        [-0.1934,  0.0955, 0.0],  # RL
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
    EMERGENCY_STAND = "emergency_stand"


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
    stop_after: bool = False
    gait: str = "trot"                      # Gait selection: trot | pace | bound | pronk | walk

    def clamped(self, bounds: "SafetyBounds") -> "MotionCommand":
        """Return a new MotionCommand clamped to configured safety bounds."""
        vx_clamped = float(np.clip(self.vx, bounds.min_vx, bounds.max_vx))
        vy_clamped = float(np.clip(self.vy, -bounds.max_vy, bounds.max_vy))
        yaw_clamped = float(np.clip(self.yaw_rate, -bounds.max_yaw_rate, bounds.max_yaw_rate))
        dur_clamped = float(np.clip(self.duration, 0.05, bounds.max_duration))
        return MotionCommand(
            vx=vx_clamped,
            vy=vy_clamped,
            yaw_rate=yaw_clamped,
            duration=dur_clamped,
            mode=self.mode,
            heading_target=self.heading_target,
            waypoint=self.waypoint,
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
            leg_clock = stride_time - offset * self.stride_period
            while leg_clock < -self.swing_duration:
                leg_clock += self.stride_period
            while leg_clock > self.stance_duration:
                leg_clock -= self.stride_period

            if 0.0 < leg_clock < self.stance_duration:
                stance_phases[index] = leg_clock / self.stance_duration
            elif -self.swing_duration < leg_clock <= 0.0:
                swing_phases[index] = (leg_clock + self.swing_duration) / self.swing_duration
            elif self.stance_duration < leg_clock <= self.stride_period:
                swing_phases[index] = (leg_clock - self.stance_duration) / self.swing_duration

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
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]], List[float], List[float]]:
        """Calculate joint angles and foot positions for all 4 legs.

        Returns:
            target_q: (12,) joint angle targets [rad]
            desired_feet_body: list of 4 foot positions in body frame [m]
            step_lengths: list of 4 step lengths [m]
            step_rotations: list of 4 trajectory rotations [rad]
        """
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
            step_len = v_hip_norm * self.config.stride_period
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

            # Foot position in body frame
            foot_pos_body = (
                hip_pos[0] + foot_x_hip,
                hip_pos[1] + foot_y_hip,
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
# Safety Supervisor
# -----------------------------------------------------------------------------

class SafetySupervisor:
    """Monitors robot attitude, height, joint limits, and numerical state."""

    def __init__(self, bounds: SafetyBounds):
        self.bounds = bounds
        self.faults: List[SafetyFault] = []

    def check_state(
        self,
        step_time: float,
        base_z: float,
        roll: float,
        pitch: float,
        joint_positions: np.ndarray,
    ) -> Optional[SafetyFault]:
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
    """Configure model gravity and actuator PD gains (Kp=60, Kd=5 by default)."""
    model.opt.gravity[:] = (0.0, 0.0, -gravity)
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kd


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
    ):
        self.gait_config = gait_config or config or GaitConfig()
        self.safety_bounds = safety_bounds or bounds or SafetyBounds()

        try:
            from go2_multi_gait import MultiGaitScheduler
            self.phase_generator = MultiGaitScheduler()
        except ImportError:
            self.phase_generator = ChampPhaseGenerator(
                stance_duration=self.gait_config.stance_duration,
                swing_duration=self.gait_config.swing_duration,
            )
        self.foothold_planner = FootholdPlanner(self.gait_config)
        self.slew_limiter = CommandSlewLimiter(self.safety_bounds)
        self.waypoint_navigator = WaypointNavigator()
        self.safety_supervisor = SafetySupervisor(self.safety_bounds)

        self.neutral_targets = neutral_joint_targets(self.gait_config.body_height)
        self.mode = MotionMode.STAND
        self.emergency_fault: Optional[SafetyFault] = None
        self._active_waypoint: Optional[Waypoint] = None
        self._obstacle_avoidance = None

    @property
    def obstacle_avoidance(self):
        if self._obstacle_avoidance is None:
            from go2_vision_perception import ReactiveObstacleAvoidance
            self._obstacle_avoidance = ReactiveObstacleAvoidance()
        return self._obstacle_avoidance

    def reset(self):
        self.slew_limiter.reset()
        self.mode = MotionMode.STAND
        self.emergency_fault = None
        self.safety_supervisor.faults.clear()
        self.phase_generator.reset()
        self._active_waypoint = None
        self.waypoint_navigator.reset()

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

        # If in emergency stand, smoothly decelerate to neutral pose
        if self.mode == MotionMode.EMERGENCY_STAND:
            cmd_vx, cmd_vy, cmd_yaw = self.slew_limiter.step(0.0, 0.0, 0.0, dt)
            target_q = self.neutral_targets.copy()
            desired_feet = [
                (HIP_POSITIONS[i][0], HIP_POSITIONS[i][1] + (HIP_OFFSET if IS_LEFT[i] else -HIP_OFFSET), -self.gait_config.body_height)
                for i in range(4)
            ]
            info = {
                "cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "cmd_yaw": cmd_yaw,
                "mode": self.mode.value, "fault": self.emergency_fault.fault_type if self.emergency_fault else None,
                "obstacle_status": "none",
            }
            return target_q, desired_feet, [0.0]*4, [0.0]*4, info

        # Clamp user command against safety limits
        clamped_cmd = command.clamped(self.safety_bounds)

        # Determine target velocities based on mode
        if command.mode == MotionMode.WAYPOINT and command.waypoint is not None:
            if self._active_waypoint != command.waypoint:
                self._active_waypoint = command.waypoint
                self.waypoint_navigator.reset([command.waypoint])

            wp_vx, wp_vy, wp_yaw, is_arrived = self.waypoint_navigator.compute_command(
                current_x=float(base_pos[0]),
                current_y=float(base_pos[1]),
                current_yaw=float(base_euler[2]),
            )
            req_vx, req_vy, req_yaw = wp_vx, wp_vy, wp_yaw
            self.mode = MotionMode.STAND if is_arrived else MotionMode.WAYPOINT
        elif command.mode == MotionMode.STAND:
            req_vx, req_vy, req_yaw = 0.0, 0.0, 0.0
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

        is_moving = (abs(cmd_vx) > 0.01 or abs(cmd_vy) > 0.01 or abs(cmd_yaw) > 0.02)

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
        joint_positions: np.ndarray,
        desired_feet: List[Tuple[float, float, float]],
        contacts: List[bool],
        accel_lin: Optional[np.ndarray] = None,
        accel_ang: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes optimal contact torques tau in R^12 via leg Jacobian transpose.

        Args:
            joint_positions: 12-element measured joint angle array [rad]
            desired_feet: List of 4 desired foot positions in body frame [m]
            contacts: List of 4 booleans indicating active stance contact
            accel_lin: Desired linear acceleration [ax, ay, az] in body frame
            accel_ang: Desired angular acceleration [alphax, alphay, alphaz] in body frame

        Returns:
            torques: 12-element joint torque array [N*m]
        """
        from go2_balance_qp import SingleRigidBodyQpSolver, compute_leg_jacobian
        if not hasattr(self, "_qp_solver") or self._qp_solver is None:
            self._qp_solver = SingleRigidBodyQpSolver()

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

        return torques
