"""
Go2 / MuJoCo — Bio-Inspired Gait Transitions (Humphreys et al., RA-L 2023)
============================================================================
Extends go2_mujoco_com_debug.py with the paper's Froude-number-governed
gait scheduler (Section III):

  1.5  Gait Representation      -> GAITS table (T, duty factor d, phase offsets)
  1.6  Gait Switching Trigger    -> BioInspiredGaitScheduler.update_trigger()
  1.7  Froude Number Bounds      -> FR_BOUNDS table
  1.8  Transition Mechanism      -> w1/w2 blending of (T, d), Eq. 1.4-1.6
  1.9  Adaptive Transition Dur.  -> D_trans / C, Eq. 1.7-1.8
       + Eq. 7 rapid-acceleration override (n_trans as fn of Fr)
       + Eq. 8 phase-offset "phasing in" (rather than blending)
       + Eq. 3  Cost of Transport, logged/plotted as a sanity check that
                gait choice is tracking the paper's minimal-CoT intent.

Everything unrelated to the gait scheduler (URDF/MJCF loading, leg IK,
standing pose init, IMU-based velocity estimate, debug plot/metrics
plumbing) is kept as close as possible to the original script so this
drops in as a direct upgrade.
"""

import math
import os
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer


# ============================================================
# 1. MUJOCO INITIALIZATION + GO2 MODEL PREP
# ============================================================

MODEL_PATH = r"unitree_go2\go2_description.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
SIM_DT = model.opt.timestep

# ============================================================
# 2. ROBOT GEOMETRY
# ============================================================

HIP_OFFSET = 0.0955
THIGH = 0.213
CALF = 0.213

BODY_HEIGHT = 0.3
STEP_HEIGHT = 0.15

BODY_L = 0.3762
BODY_W = 0.0935

G = 9.81
# Paper (Sec III-C): h is a geometric length, chosen as *maximum leg length*
# rather than hip height, because robots (unlike animals) have a
# configurable stance height. Max reach = thigh + calf fully extended.
H_LEG = THIGH + CALF

LEGS = ["FR", "FL", "RR", "RL"]
IS_LEFT = {"FR": False, "FL": True, "RR": False, "RL": True}

NOMINAL_FEET = {
    "FR": np.array([ BODY_L/2, -(BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "FL": np.array([ BODY_L/2,  (BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "RR": np.array([-BODY_L/2, -(BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "RL": np.array([-BODY_L/2,  (BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
}

HIP_BODY = {
    "FR": np.array([ BODY_L/2, -BODY_W/2]),
    "FL": np.array([ BODY_L/2,  BODY_W/2]),
    "RR": np.array([-BODY_L/2, -BODY_W/2]),
    "RL": np.array([-BODY_L/2,  BODY_W/2]),
}


def nominal_foot_hip(leg):
    return NOMINAL_FEET[leg] - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])


def leg_ik(x, y, z, is_left=False):
    l1 = HIP_OFFSET if is_left else -HIP_OFFSET
    yz = y * y + z * z
    if yz <= HIP_OFFSET ** 2:
        return 0.0, 0.0, 0.0

    theta1 = math.atan2(y, -z) - math.atan2(l1, math.sqrt(yz - HIP_OFFSET ** 2))
    d = x * x + yz - HIP_OFFSET ** 2
    D = (d - THIGH ** 2 - CALF ** 2) / (2 * THIGH * CALF)
    D = max(-1.0, min(1.0, D))
    theta3 = -math.acos(D)

    a = math.atan2(-x, math.sqrt(yz - HIP_OFFSET ** 2))
    b = math.atan2(CALF * math.sin(abs(theta3)), THIGH + CALF * math.cos(theta3))
    theta2 = -(a - b)
    return theta1, theta2, theta3


joint_order = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _name_to_id(name, kind='joint'):
    if kind == 'joint':
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    elif kind == 'actuator':
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    elif kind == 'body':
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    else:
        raise ValueError(kind)
    return idx if idx != -1 else None


qpos_adr = []
dof_adr = []
actuator_ids = []
for jn in joint_order:
    jid = _name_to_id(jn, 'joint')
    if jid is None:
        raise KeyError(f"Joint not found in MuJoCo model: {jn}")
    qpos_adr.append(model.jnt_qposadr[jid])
    dof_adr.append(model.jnt_dofadr[jid])

    aid = _name_to_id(jn, 'actuator')
    if aid is None:
        cand = _name_to_id(jn.replace('_joint', ''), 'actuator')
        if cand is not None:
            aid = cand
    actuator_ids.append(aid)

if any(a is None for a in actuator_ids):
    missing = [joint_order[i] for i, a in enumerate(actuator_ids) if a is None]
    raise KeyError(
        "Missing MuJoCo position actuators for joints: {}. "
        "Add actuators named like the joint or joint name without '_joint'.".format(missing)
    )

TOTAL_MASS = float(np.sum(model.body_mass))

# Standing joint targets (feet directly under hips at BODY_HEIGHT).
standing_q = []
for leg in LEGS:
    x, y, z = nominal_foot_hip(leg)
    standing_q += list(leg_ik(x, y, z, IS_LEFT[leg]))

if model.nq >= 7:
    data.qpos[0:3] = np.array([0.0, 0.0, BODY_HEIGHT + 0.01])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
for adr, val in zip(qpos_adr, standing_q):
    data.qpos[adr] = val
for aid, val in zip(actuator_ids, standing_q):
    data.ctrl[aid] = val
mujoco.mj_forward(model, data)


# ============================================================
# 3. BIO-INSPIRED GAIT SCHEDULER (paper Section III)
# ============================================================

@dataclass
class GaitParams:
    name: str
    T: float                 # gait period [s]                 (Table 1.1)
    d: float                 # duty factor                     (Table 1.1)
    offsets: dict            # per-leg phase offset in [0, 1)   (Table 1.1)


# --- Section 1.5 / Table 1.1: Gait Design Parameters -----------------------
# Phase offsets quoted in the paper as (FL, FR, RL, RR); stored as a dict
# here so leg lookups elsewhere in the file don't depend on ordering.
GAITS = {
    "static_walk":  GaitParams("Static Walk",  T=1.00, d=0.80,
                                offsets={"FL": 0.00, "FR": 0.75, "RL": 0.50, "RR": 0.25}),
    "walking_trot": GaitParams("Walking Trot", T=0.50, d=0.60,
                                offsets={"FL": 0.00, "FR": 0.50, "RL": 0.50, "RR": 0.00}),
    "trot":         GaitParams("Trot",         T=0.40, d=0.50,
                                offsets={"FL": 0.00, "FR": 0.50, "RL": 0.50, "RR": 0.00}),
    "flying_trot":  GaitParams("Flying Trot",  T=0.30, d=0.40,
                                offsets={"FL": 0.00, "FR": 0.50, "RL": 0.50, "RR": 0.00}),
}
# Slow -> fast ordering; neighbours in this list are the only legal
# transition targets (Section 1.6: "raise transition flag, select target
# gait" is always to the adjacent gait whose Fr band was entered).
GAIT_ORDER = ["static_walk", "walking_trot", "trot", "flying_trot"]

# --- Section 1.7 / Table 1.2: Froude Number Bounds --------------------------
FR_BOUNDS = {
    "static_walk":  (0.0009, 0.0024),
    "walking_trot": (0.0024, 0.1517),
    "trot":         (0.1517, 0.7400),
    "flying_trot":  (0.7400, 1.0000),
}

# --- Section III-D constants -------------------------------------------------
FR_CRIT = 1.0   # conservative stability limit of a walking Froude number
C_MAX = 2.0     # longest number of gait cycles observed for animal transitions


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_froude(vx_now, vy_now):
    """Eq. 1.3 / 2:  Fr = v^2 / (g h)."""
    v = math.hypot(vx_now, vy_now)
    return (v * v) / (G * H_LEG)


class BioInspiredGaitScheduler:
    """
    Implements Sections 1.6/1.8/1.9 (equivalently III-C/III-D of the paper):
    monitors the Froude number, triggers a transition to the neighbouring
    gait when Fr leaves the current gait's band, and smoothly blends the
    gait parameters (period, duty factor, phase offsets) over an adaptively
    sized transition window.
    """

    def __init__(self, start_gait="trot"):
        self.current_key = start_gait
        self.target_key = start_gait
        self.transitioning = False

        # Live (possibly blended) parameters that actually drive the legs.
        self.T = GAITS[start_gait].T
        self.d = GAITS[start_gait].d
        self.offsets = dict(GAITS[start_gait].offsets)

        # Single running master phase; per-leg phase = master + offset (mod 1).
        self.master_phase = 0.0

        # Transition bookkeeping.
        self.t_trans_start = None
        self.D_trans = 0.0        # seconds, Eq. 1.7
        self.fr_star = 0.0        # Fr at the instant the transition was raised
        self.fr_lb_from = self.fr_ub_from = 0.0
        self.fr_lb_to = self.fr_ub_to = 0.0
        self.w1 = 1.0
        self.w2 = 0.0
        self.n_progress = 0.0     # effective completion fraction in [0, 1]

        self.transition_log = []  # one entry per completed transition

    def _neighbor(self, key, direction):
        i = GAIT_ORDER.index(key)
        j = i + direction
        return GAIT_ORDER[j] if 0 <= j < len(GAIT_ORDER) else None

    def _raise_transition(self, new_key, fr_now, t_now):
        self.target_key = new_key
        self.transitioning = True
        self.t_trans_start = t_now
        self.fr_star = fr_now

        cur, new = GAITS[self.current_key], GAITS[new_key]
        # Eq. 1.8: C shrinks (=> faster transition) as Fr* approaches Fr_crit.
        # A floor is kept so the transition never fully collapses to a hard
        # switch even right at the stability limit.
        C = clamp(1.0 - self.fr_star / FR_CRIT, 0.15, 1.0) * C_MAX
        # Eq. 1.7: duration set from the mean of the two gait periods,
        # scaled by how many cycles (C) the transition should span.
        self.D_trans = max(2 * SIM_DT, ((cur.T + new.T) / 2.0) * C)

        self.fr_lb_from, self.fr_ub_from = FR_BOUNDS[self.current_key]
        self.fr_lb_to, self.fr_ub_to = FR_BOUNDS[new_key]
        self.n_progress = 0.0

        print(f"[gait] {GAITS[self.current_key].name} -> {new.name}  "
              f"(Fr*={fr_now:.4f}, D_trans={self.D_trans:.3f}s, C={C:.2f})")

    def update_trigger(self, fr_now, t_now):
        """Section 1.6: monitor Fr; raise a transition flag once it leaves
        the current gait's allowable range."""
        if self.transitioning:
            return
        lb, ub = FR_BOUNDS[self.current_key]
        if fr_now > ub:
            nxt = self._neighbor(self.current_key, +1)
        elif fr_now < lb:
            nxt = self._neighbor(self.current_key, -1)
        else:
            nxt = None
        if nxt is not None:
            self._raise_transition(nxt, fr_now, t_now)

    def step(self, dt, t_now, fr_now):
        self.update_trigger(fr_now, t_now)

        if self.transitioning:
            elapsed = t_now - self.t_trans_start
            time_progress = clamp(elapsed / self.D_trans, 0.0, 1.0)

            # Eq. 7: let progress additionally track how far Fr has moved
            # through the *target* band (shifting up) or out of the
            # *current* band (shifting down). This lets a rapid acceleration
            # finish the transition early instead of waiting out D_trans,
            # producing the inverse Fr-duration relationship the paper
            # describes for handling fast accelerations.
            if self.target_key == self._neighbor(self.current_key, +1):
                span = max(1e-9, self.fr_ub_to - self.fr_lb_to)
                fr_progress = clamp((fr_now - self.fr_lb_to) / span, 0.0, 1.0)
            else:
                span = max(1e-9, self.fr_ub_from - self.fr_lb_from)
                fr_progress = clamp((self.fr_ub_from - fr_now) / span, 0.0, 1.0)

            self.n_progress = max(time_progress, fr_progress)

            # Eq. 1.4 / 1.5
            self.w1 = 1.0 - self.n_progress
            self.w2 = self.n_progress

            cur, new = GAITS[self.current_key], GAITS[self.target_key]
            # Eq. 1.6: blend period & duty factor.
            self.T = self.w1 * cur.T + self.w2 * new.T
            self.d = self.w1 * cur.d + self.w2 * new.d
            # Eq. 8: phase offsets are "phased in" to the new gait's offset
            # rather than blended, at the same rate n_progress.
            for leg in LEGS:
                self.offsets[leg] = cur.offsets[leg] + \
                    (new.offsets[leg] - cur.offsets[leg]) * self.n_progress

            if self.n_progress >= 1.0:
                self.transition_log.append({
                    "from": cur.name, "to": new.name,
                    "fr_star": self.fr_star, "D_trans_s": self.D_trans,
                    "actual_duration_s": elapsed,
                })
                self.current_key = self.target_key
                self.transitioning = False
                self.T = GAITS[self.current_key].T
                self.d = GAITS[self.current_key].d
                self.offsets = dict(GAITS[self.current_key].offsets)
                self.w1, self.w2 = 1.0, 0.0
        else:
            self.T = GAITS[self.current_key].T
            self.d = GAITS[self.current_key].d
            self.offsets = dict(GAITS[self.current_key].offsets)
            self.w1, self.w2 = 1.0, 0.0

        self.master_phase = (self.master_phase + dt / self.T) % 1.0

    def leg_phase(self, leg):
        return (self.master_phase + self.offsets[leg]) % 1.0

    def leg_in_stance(self, leg):
        return self.leg_phase(leg) < self.d

    def swing_progress(self, leg):
        lp = self.leg_phase(leg)
        return clamp((lp - self.d) / max(1e-9, (1.0 - self.d)), 0.0, 1.0)

    def active_label(self):
        if self.transitioning:
            return f"{GAITS[self.current_key].name} -> {GAITS[self.target_key].name} " \
                   f"({100 * self.n_progress:.0f}%)"
        return GAITS[self.current_key].name


def compute_cost_of_transport(v_now):
    """Eq. 3:  CoT = sum(tau_i * thetadot_i) / (m g v)."""
    if abs(v_now) < 1e-3:
        return 0.0
    power = 0.0
    for aid, dadr in zip(actuator_ids, dof_adr):
        tau = data.actuator_force[aid]
        qd = data.qvel[dadr]
        power += tau * qd
    return power / (TOTAL_MASS * G * v_now)


# ============================================================
# 3A. VELOCITY COMMAND PROFILE
# ============================================================
# A ramp-up / ramp-down profile (mirrors the paper's Fig. 5/7 sweeps) so the
# Froude number actually crosses every band in Table 1.2 and every gait
# transition gets exercised during one run.

VEL_RAMP_ACCEL = 0.05   # m/s^2 (gentle: this PD position tracker was only
                         # validated at a constant 0.2 m/s in the source
                         # script; ramping faster than the WBC can track
                         # causes stride overreach and falls)
VEL_MAX = 1.0            # m/s peak commanded speed. Fr(1.0 m/s) ~= 0.24,
                         # comfortably into Trot. Raise this only after
                         # confirming Trot is stable, and only then push
                         # further (Fr=0.74 -> Flying Trot needs ~2.0 m/s,
                         # which likely needs stiffer/re-tuned PD gains).
vy = 0.0
omega = 0.0

# Fr can be computed from the *commanded* velocity (feedforward) or the
# *estimated* velocity (feedback). Feedback is what the paper describes,
# but it creates a lock-up failure mode with a weak low-level tracker: if
# the robot falls behind the velocity ramp, vx_est stays low, the scheduler
# never leaves the slow gait, yet foot placement (which plans off the
# commanded velocity) keeps taking strides sized for the fast command,
# overreaching and falling -- which then corrupts vx_est further. Using the
# commanded velocity for the trigger keeps gait selection and stride
# planning consistent with each other. Switch to "estimated" once the
# tracker is solid enough to follow the ramp closely.
FR_SOURCE = "commanded"   # "commanded" | "estimated"

# Simple fall guard so a bad run doesn't burn 30s logging garbage.
FALL_HEIGHT_MIN = 0.10     # m, base height below which we call it a fall
FALL_TILT_MAX = 1.0        # rad (~57 deg) roll/pitch magnitude


def velocity_command(t_loco):
    t_up = VEL_MAX / VEL_RAMP_ACCEL
    if t_loco < t_up:
        return VEL_RAMP_ACCEL * t_loco
    elif t_loco < 2 * t_up:
        return VEL_MAX - VEL_RAMP_ACCEL * (t_loco - t_up)
    else:
        return 0.0


# ============================================================
# 3B. DEBUG THREAD CONFIG + METRICS
# ============================================================

ENABLE_DEBUG_PLOT = True
DEBUG_PLOT_HZ = 30
DEBUG_WINDOW_SEC = 10.0
DEBUG_HISTORY = max(50, int(DEBUG_PLOT_HZ * DEBUG_WINDOW_SEC))

debug_lock = threading.Lock()
debug_stop_event = threading.Event()
latest_debug_sample = None

metrics_log = []
SETTLING_BAND_VX = 0.02
SETTLING_BAND_VY = 0.02
SETTLING_BAND_WZ = 0.10
SETTLING_HOLD_SEC = 0.50

# ============================================================
# 4. BASE BODY ID
# ============================================================

base_body_id = _name_to_id('base', 'body')
if base_body_id is None:
    for cand in ['trunk', 'base_link', 'body']:
        base_body_id = _name_to_id(cand, 'body')
        if base_body_id is not None:
            break
if base_body_id is None:
    base_body_id = 1 if model.nbody > 1 else 0


def Rz(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_to_world_xy(body_xy, body_pose):
    return Rz(body_pose[2]) @ body_xy + body_pose[:2]


def world_to_body_xy(world_xy, body_pose):
    return Rz(-body_pose[2]) @ (world_xy - body_pose[:2])


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_euler_wxyz(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def get_imu_state():
    q = data.qpos[3:7].copy()
    roll, pitch, yaw = quat_to_euler_wxyz(q)
    lin_world = data.qvel[0:3].copy()
    ang_world = data.qvel[3:6].copy()
    v_world_xy = np.array([lin_world[0], lin_world[1]])
    v_body_xy = Rz(-yaw) @ v_world_xy
    return {
        "roll": roll, "pitch": pitch, "yaw": yaw,
        "vx": v_body_xy[0], "vy": v_body_xy[1], "wz": ang_world[2],
    }


def get_estimated_body_velocities():
    imu = get_imu_state()
    return imu["vx"], imu["vy"], imu["wz"], imu["yaw"]


def publish_debug_sample(t_now, vx_c, vy_c, w_c, vx_e, vy_e, w_e, gait_label, cot):
    global latest_debug_sample
    sample = {
        "t": t_now, "vx_cmd": vx_c, "vy_cmd": vy_c, "w_cmd": w_c,
        "vx_est": vx_e, "vy_est": vy_e, "w_est": w_e,
        "fr_cmd": compute_froude(vx_c, vy_c), "fr_est": compute_froude(vx_e, vy_e),
        "gait_label": gait_label, "cot": cot,
    }
    with debug_lock:
        latest_debug_sample = sample


def append_metrics_sample(t_now, vx_cmd_now, vy_cmd_now, w_cmd_now, vx_est, vy_est, w_est,
                           yaw=None, roll=None, pitch=None, gait_key=None, transitioning=None, cot=None):
    if t_now < 0.0:
        return
    metrics_log.append({
        "t": float(t_now),
        "vx_cmd": float(vx_cmd_now), "vy_cmd": float(vy_cmd_now), "w_cmd": float(w_cmd_now),
        "vx_est": float(vx_est), "vy_est": float(vy_est), "w_est": float(w_est),
        "yaw": None if yaw is None else float(yaw),
        "roll": None if roll is None else float(roll),
        "pitch": None if pitch is None else float(pitch),
        "gait_key": gait_key, "transitioning": transitioning,
        "cot": None if cot is None else float(cot),
    })


def compute_rmse(cmd, est):
    cmd = np.asarray(cmd, dtype=float)
    est = np.asarray(est, dtype=float)
    if len(cmd) == 0:
        return None
    return float(np.sqrt(np.mean((cmd - est) ** 2)))


def integrate_abs(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.trapezoid(np.abs(values), times))


def compute_settling_time(times, err_vx, err_vy, err_wz):
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return None
    within = (
        (np.abs(err_vx) <= SETTLING_BAND_VX) &
        (np.abs(err_vy) <= SETTLING_BAND_VY) &
        (np.abs(err_wz) <= SETTLING_BAND_WZ)
    )
    for i in range(len(times)):
        if not within[i]:
            continue
        t0 = times[i]
        j = i
        while j + 1 < len(times) and within[j + 1]:
            j += 1
        if times[j] - t0 >= SETTLING_HOLD_SEC:
            return float(t0)
    return None


def build_metrics_summary(scheduler, include_attitude=False):
    if not metrics_log:
        return None
    times = np.array([s["t"] for s in metrics_log], dtype=float)
    vx_cmd_arr = np.array([s["vx_cmd"] for s in metrics_log], dtype=float)
    vy_cmd_arr = np.array([s["vy_cmd"] for s in metrics_log], dtype=float)
    w_cmd_arr = np.array([s["w_cmd"] for s in metrics_log], dtype=float)
    vx_est_arr = np.array([s["vx_est"] for s in metrics_log], dtype=float)
    vy_est_arr = np.array([s["vy_est"] for s in metrics_log], dtype=float)
    w_est_arr = np.array([s["w_est"] for s in metrics_log], dtype=float)
    yaw_arr = np.array([0.0 if s["yaw"] is None else s["yaw"] for s in metrics_log], dtype=float)
    roll_arr = np.array([0.0 if s["roll"] is None else s["roll"] for s in metrics_log], dtype=float)
    pitch_arr = np.array([0.0 if s["pitch"] is None else s["pitch"] for s in metrics_log], dtype=float)
    cot_arr = np.array([0.0 if s["cot"] is None else s["cot"] for s in metrics_log], dtype=float)

    err_vx = vx_cmd_arr - vx_est_arr
    err_vy = vy_cmd_arr - vy_est_arr
    err_wz = w_cmd_arr - w_est_arr
    yaw_err = yaw_arr - yaw_arr[0] if len(yaw_arr) else np.array([], dtype=float)

    time_in_gait = {}
    for s in metrics_log:
        if s["gait_key"] is None:
            continue
        time_in_gait[s["gait_key"]] = time_in_gait.get(s["gait_key"], 0) + 1
    n = max(1, len(metrics_log))
    time_in_gait_frac = {k: v / n for k, v in time_in_gait.items()}

    return {
        "samples": int(len(metrics_log)),
        "duration_sec": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "velocity_rmse": {
            "vx": compute_rmse(vx_cmd_arr, vx_est_arr),
            "vy": compute_rmse(vy_cmd_arr, vy_est_arr),
            "wz": compute_rmse(w_cmd_arr, w_est_arr),
        },
        "body_attitude_rms": {
            "roll": float(np.sqrt(np.mean(roll_arr ** 2))) if include_attitude and len(roll_arr) else None,
            "pitch": float(np.sqrt(np.mean(pitch_arr ** 2))) if include_attitude and len(pitch_arr) else None,
        },
        "peak_attitude_excursion": {
            "max_abs_roll": float(np.max(np.abs(roll_arr))) if include_attitude and len(roll_arr) else None,
            "max_abs_pitch": float(np.max(np.abs(pitch_arr))) if include_attitude and len(pitch_arr) else None,
        },
        "settling_time_after_start_sec": compute_settling_time(times, err_vx, err_vy, err_wz),
        "slip_heading_drift_proxy": {
            "sideways_velocity_bias_mps": float(np.mean(vy_est_arr)),
            "integrated_abs_vy_m": integrate_abs(vy_est_arr, times),
            "integrated_abs_yaw_rate_rad": integrate_abs(w_est_arr, times),
            "integrated_abs_yaw_error_rad_s": integrate_abs(yaw_err, times),
        },
        "cost_of_transport": {
            "mean": float(np.mean(cot_arr)),
            "max": float(np.max(cot_arr)),
        },
        "time_in_gait_fraction": time_in_gait_frac,
        "num_transitions": len(scheduler.transition_log),
        "transitions": scheduler.transition_log,
    }


def save_and_print_metrics(summary, tag):
    if summary is None:
        print(f"[{tag}] No locomotion samples captured; metrics not written.")
        return
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(logs_dir, f"{tag}_metrics_{stamp}.json")
    csv_path = os.path.join(logs_dir, f"{tag}_samples_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    headers = ["t", "vx_cmd", "vy_cmd", "w_cmd", "vx_est", "vy_est", "w_est",
               "yaw", "roll", "pitch", "gait_key", "transitioning", "cot"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for s in metrics_log:
            f.write(",".join("" if s.get(h) is None else str(s.get(h)) for h in headers) + "\n")

    print(f"\n[{tag}] Metrics summary")
    print(json.dumps(summary, indent=2))
    print(f"[{tag}] Wrote JSON: {json_path}")
    print(f"[{tag}] Wrote CSV: {csv_path}")


def debug_plot_worker():
    plt.ion()
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    try:
        fig.canvas.manager.set_window_title("Go2 Bio-Inspired Gait Scheduler")
    except Exception:
        pass

    hist_t = deque(maxlen=DEBUG_HISTORY)
    hist_vx_cmd = deque(maxlen=DEBUG_HISTORY)
    hist_vx_est = deque(maxlen=DEBUG_HISTORY)
    hist_vy_cmd = deque(maxlen=DEBUG_HISTORY)
    hist_vy_est = deque(maxlen=DEBUG_HISTORY)
    hist_w_cmd = deque(maxlen=DEBUG_HISTORY)
    hist_w_est = deque(maxlen=DEBUG_HISTORY)
    hist_fr_cmd = deque(maxlen=DEBUG_HISTORY)
    hist_fr_est = deque(maxlen=DEBUG_HISTORY)
    hist_cot = deque(maxlen=DEBUG_HISTORY)

    line_vx_cmd, = axs[0].plot([], [], "r--", lw=1.8, label="vx cmd")
    line_vx_est, = axs[0].plot([], [], "r", lw=1.8, label="vx est")
    axs[0].set_ylabel("vx [m/s]"); axs[0].legend(loc="upper right"); axs[0].grid(True)

    line_vy_cmd, = axs[1].plot([], [], "g--", lw=1.8, label="vy cmd")
    line_vy_est, = axs[1].plot([], [], "g", lw=1.8, label="vy est")
    axs[1].set_ylabel("vy [m/s]"); axs[1].legend(loc="upper right"); axs[1].grid(True)

    line_w_cmd, = axs[2].plot([], [], "b--", lw=1.8, label="w cmd")
    line_w_est, = axs[2].plot([], [], "b", lw=1.8, label="w est")
    axs[2].set_ylabel("w [rad/s]"); axs[2].legend(loc="upper right"); axs[2].grid(True)

    line_fr_cmd, = axs[3].plot([], [], "k--", lw=1.8, label="Fr cmd")
    line_fr_est, = axs[3].plot([], [], "k", lw=1.8, label="Fr est")
    for key, (lb, ub) in FR_BOUNDS.items():
        axs[3].axhspan(lb, ub, alpha=0.06)
    axs[3].set_ylabel("Froude"); axs[3].legend(loc="upper right"); axs[3].grid(True)

    line_cot, = axs[4].plot([], [], "m", lw=1.6, label="CoT")
    axs[4].set_ylabel("CoT"); axs[4].set_xlabel("time [s]")
    axs[4].legend(loc="upper right"); axs[4].grid(True)

    gait_text = axs[0].text(0.01, 0.92, "", transform=axs[0].transAxes,
                             fontsize=10, va="top", ha="left",
                             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    dt_plot = 1.0 / DEBUG_PLOT_HZ
    for ax in axs:
        ax.set_xlim(0.0, DEBUG_WINDOW_SEC)

    while not debug_stop_event.is_set():
        with debug_lock:
            sample = None if latest_debug_sample is None else latest_debug_sample.copy()

        if sample is not None:
            hist_t.append(sample["t"])
            hist_vx_cmd.append(sample["vx_cmd"]); hist_vx_est.append(sample["vx_est"])
            hist_vy_cmd.append(sample["vy_cmd"]); hist_vy_est.append(sample["vy_est"])
            hist_w_cmd.append(sample["w_cmd"]); hist_w_est.append(sample["w_est"])
            hist_fr_cmd.append(sample["fr_cmd"]); hist_fr_est.append(sample["fr_est"])
            hist_cot.append(sample["cot"])

            t_data = [t - hist_t[0] for t in hist_t] if len(hist_t) >= 2 else list(hist_t)
            line_vx_cmd.set_data(t_data, list(hist_vx_cmd))
            line_vx_est.set_data(t_data, list(hist_vx_est))
            line_vy_cmd.set_data(t_data, list(hist_vy_cmd))
            line_vy_est.set_data(t_data, list(hist_vy_est))
            line_w_cmd.set_data(t_data, list(hist_w_cmd))
            line_w_est.set_data(t_data, list(hist_w_est))
            line_fr_cmd.set_data(t_data, list(hist_fr_cmd))
            line_fr_est.set_data(t_data, list(hist_fr_est))
            line_cot.set_data(t_data, list(hist_cot))
            gait_text.set_text(sample["gait_label"])

            xmax = max(DEBUG_WINDOW_SEC, t_data[-1] if len(t_data) > 0 else DEBUG_WINDOW_SEC)
            xmin = max(0.0, xmax - DEBUG_WINDOW_SEC)
            for ax in axs:
                ax.set_xlim(xmin, xmin + DEBUG_WINDOW_SEC)
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        time.sleep(dt_plot)

    plt.close(fig)


# ============================================================
# 5. IK-DEPENDENT FOOT PLANNING HELPERS
# ============================================================

def stance_foot_in_hip_frame(leg, body_pose, stance_foot_world):
    foot_xy_body = world_to_body_xy(stance_foot_world[:2], body_pose)
    foot_body = np.array([foot_xy_body[0], foot_xy_body[1], stance_foot_world[2]])
    hip_body = np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
    return foot_body - hip_body


def swing_foot_in_hip_frame(leg, s, start_hip, target_hip):
    xy = (1.0 - s) * start_hip[:2] + s * target_hip[:2]
    z0 = start_hip[2]
    zf = target_hip[2]
    z_lin = (1.0 - s) * z0 + s * zf
    z = z_lin + STEP_HEIGHT * math.sin(math.pi * s)
    return np.array([xy[0], xy[1], z])


def compute_touchdown_body(leg, vx_now, vy_now, omega_now, t_stance):
    dx = vx_now * t_stance
    dy = vy_now * t_stance
    dtheta = omega_now * t_stance
    p_nom = NOMINAL_FEET[leg]
    td_xy = Rz(dtheta) @ p_nom[:2] + np.array([dx, dy])
    return np.array([td_xy[0], td_xy[1], p_nom[2]])


def command_joint_positions(q):
    for aid, angle in zip(actuator_ids, q):
        data.ctrl[aid] = angle


def step_with_viewer(viewer=None):
    mujoco.mj_step(model, data)
    if viewer is not None and viewer.is_running():
        viewer.sync()


# ============================================================
# 6. MAIN LOOP
# ============================================================

body_pose = np.array([0.0, 0.0, 0.0])
stance_feet = {}
swing_start_hip = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
touchdown_tgts_hip = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
prev_stance = {}

for leg in LEGS:
    world_xy = body_to_world_xy(NOMINAL_FEET[leg][:2], body_pose)
    stance_feet[leg] = np.array([world_xy[0], world_xy[1], NOMINAL_FEET[leg][2]])

# Start in Static Walk: at t=0 the commanded velocity (and hence Fr) is 0,
# so the scheduler will climb through Table 1.2's bands as the profile ramps.
scheduler = BioInspiredGaitScheduler(start_gait="static_walk")
for leg in LEGS:
    prev_stance[leg] = scheduler.leg_in_stance(leg)

STAND_DUR = 3
debug_thread = None
if ENABLE_DEBUG_PLOT:
    debug_thread = threading.Thread(target=debug_plot_worker, daemon=True)
    debug_thread.start()

wall_start = time.time()
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            elapsed = time.time() - wall_start

            if elapsed < STAND_DUR:
                command_joint_positions(standing_q)
                step_with_viewer(viewer)
                if ENABLE_DEBUG_PLOT:
                    vx_est, vy_est, w_est, _ = get_estimated_body_velocities()
                    publish_debug_sample(0.0, 0.0, 0.0, 0.0, vx_est, vy_est, w_est,
                                          scheduler.active_label(), 0.0)
                time.sleep(SIM_DT)
                continue

            t_loco = elapsed - STAND_DUR
            vx = velocity_command(t_loco)

            vx_est, vy_est, w_est, yaw_est = get_estimated_body_velocities()
            fr_now = compute_froude(vx, vy) if FR_SOURCE == "commanded" \
                else compute_froude(vx_est, vy_est)

            scheduler.step(SIM_DT, t_loco, fr_now)

            body_pose_k = body_pose.copy()
            t_stance_now = scheduler.T * scheduler.d
            q = []

            for leg in LEGS:
                is_st = scheduler.leg_in_stance(leg)

                if is_st and not prev_stance[leg]:
                    td_hip = touchdown_tgts_hip[leg]
                    td_body = td_hip + np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
                    td_world_xy = body_to_world_xy(td_body[:2], body_pose_k)
                    stance_feet[leg] = np.array([td_world_xy[0], td_world_xy[1], td_body[2]])

                if (not is_st) and prev_stance[leg]:
                    swing_start_hip[leg] = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
                    td_body = compute_touchdown_body(leg, vx, vy, omega, t_stance_now)
                    touchdown_tgts_hip[leg] = td_body - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

                if is_st:
                    foot_hip = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
                else:
                    s = scheduler.swing_progress(leg)
                    foot_hip = swing_foot_in_hip_frame(leg, s, swing_start_hip[leg], touchdown_tgts_hip[leg])

                angles = leg_ik(foot_hip[0], foot_hip[1], foot_hip[2], IS_LEFT[leg])
                q += list(angles)
                prev_stance[leg] = is_st

            command_joint_positions(q)
            step_with_viewer(viewer)

            cot_now = compute_cost_of_transport(vx_est if abs(vx_est) > 1e-3 else vx)
            append_metrics_sample(t_loco, vx, vy, omega, vx_est, vy_est, w_est, yaw=yaw_est,
                                   gait_key=scheduler.current_key, transitioning=scheduler.transitioning,
                                   cot=cot_now)
            if ENABLE_DEBUG_PLOT:
                publish_debug_sample(t_loco, vx, vy, omega, vx_est, vy_est, w_est,
                                      scheduler.active_label(), cot_now)

            body_pose[0] -= vx * SIM_DT
            body_pose[1] += vy * SIM_DT
            body_pose[2] = wrap_to_pi(body_pose[2] + omega * SIM_DT)

            base_height_now = float(data.qpos[2]) if model.nq >= 7 else None
            roll_now, pitch_now, _ = quat_to_euler_wxyz(data.qpos[3:7]) if model.nq >= 7 else (0.0, 0.0, 0.0)
            if base_height_now is not None and (
                base_height_now < FALL_HEIGHT_MIN or
                abs(roll_now) > FALL_TILT_MAX or abs(pitch_now) > FALL_TILT_MAX
            ):
                print(f"[fall] base_height={base_height_now:.3f} roll={roll_now:.2f} "
                      f"pitch={pitch_now:.2f} at t={t_loco:.2f}s, gait={scheduler.active_label()} "
                      f"-- stopping run.")
                break

            time.sleep(SIM_DT)

except KeyboardInterrupt:
    pass
finally:
    summary = build_metrics_summary(scheduler, include_attitude=False)
    save_and_print_metrics(summary, "go2_bioinspired_mujoco")
    if ENABLE_DEBUG_PLOT:
        debug_stop_event.set()
        if debug_thread is not None:
            debug_thread.join(timeout=1.0)
