import math
import os
import time
import json
import threading
from collections import deque

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
# 2. ROBOT GEOMETRY (moved up so we can compute the standing pose
#    before the very first mj_forward call)
# ============================================================

HIP_OFFSET = 0.0955
THIGH = 0.213
CALF = 0.213

BODY_HEIGHT = 0.3
STEP_HEIGHT = 0.15
GAIT_PERIOD = 0.4
STANCE_RATIO = 0.5
T_STANCE = GAIT_PERIOD * STANCE_RATIO

BODY_L = 0.3762
BODY_W = 0.0935
# ============================================================
# 3. DESIRED BODY COMMAND
# ============================================================

vx = 0.2
vy = 0.00
omega = 0.00
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

PHASE_OFFSET = {"FR": 0.0, "RL": 0.0, "FL": 0.0, "RR": 0.0}
IS_LEFT = {"FR": False, "FL": True, "RR": False, "RL": True}
LEGS = ["FR", "FL", "RR", "RL"]


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
actuator_ids = []
for jn in joint_order:
    jid = _name_to_id(jn, 'joint')
    if jid is None:
        raise KeyError(f"Joint not found in MuJoCo model: {jn}")
    qpos_adr.append(model.jnt_qposadr[jid])

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

# Standing joint targets (feet directly under hips at BODY_HEIGHT).
standing_q = []
for leg in LEGS:
    x, y, z = nominal_foot_hip(leg)
    standing_q += list(leg_ik(x, y, z, IS_LEFT[leg]))

# Spawn pose: base positioned at the height the standing legs actually
# support (not a free-fall drop from above), and every leg joint set to
# its standing target BEFORE the first physics step. Also pre-load ctrl
# with the same targets so the PD servo sees zero error at t=0 - this
# avoids the "unconstrained pendulum swing during free-fall" failure
# mode where legs would otherwise flail before the ramp catches up.
if model.nq >= 7:
    data.qpos[0:3] = np.array([0.0, 0.0, BODY_HEIGHT + 0.01])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
for adr, val in zip(qpos_adr, standing_q):
    data.qpos[adr] = val
for aid, val in zip(actuator_ids, standing_q):
    data.ctrl[aid] = val
mujoco.mj_forward(model, data)

# ============================================================
# 3A. DEBUG THREAD CONFIG + METRICS
# ============================================================

ENABLE_DEBUG_PLOT = True
DEBUG_PLOT_HZ = 30
DEBUG_WINDOW_SEC = 8.0
DEBUG_HISTORY = max(50, int(DEBUG_PLOT_HZ * DEBUG_WINDOW_SEC))

G = 9.81
L_CHAR = BODY_HEIGHT

debug_lock = threading.Lock()
debug_stop_event = threading.Event()
latest_debug_sample = None

metrics_log = []
SETTLING_BAND_VX = 0.02
SETTLING_BAND_VY = 0.02
SETTLING_BAND_WZ = 0.10
SETTLING_HOLD_SEC = 0.50

# ============================================================
# 4. BASE BODY ID (joint/actuator maps already built above)
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

def gait_phase(t):
    return (t % GAIT_PERIOD) / GAIT_PERIOD

def leg_phase(global_phase, offset):
    return (global_phase + offset) % 1.0

def in_stance(local_phase):
    return local_phase < STANCE_RATIO

def swing_progress(local_phase):
    return (local_phase - STANCE_RATIO) / (1.0 - STANCE_RATIO)

def body_to_world_xy(body_xy, body_pose):
    return Rz(body_pose[2]) @ body_xy + body_pose[:2]

def world_to_body_xy(world_xy, body_pose):
    return Rz(-body_pose[2]) @ (world_xy - body_pose[:2])

def nominal_foot_hip(leg):
    return NOMINAL_FEET[leg] - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

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
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vx": v_body_xy[0],
        "vy": v_body_xy[1],
        "wz": ang_world[2],
    }

def get_estimated_body_velocities():
    imu = get_imu_state()
    return imu["vx"], imu["vy"], imu["wz"], imu["yaw"]

def compute_froude(vx_now, vy_now):
    v = math.sqrt(vx_now * vx_now + vy_now * vy_now)
    return (v * v) / (G * L_CHAR)

def publish_debug_sample(t_now, vx_c, vy_c, w_c, vx_e, vy_e, w_e, imu=None):
    global latest_debug_sample
    sample = {
        "t": t_now,
        "vx_cmd": vx_c,
        "vy_cmd": vy_c,
        "w_cmd": w_c,
        "vx_est": vx_e,
        "vy_est": vy_e,
        "w_est": w_e,
        "fr_cmd": compute_froude(vx_c, vy_c),
        "fr_est": compute_froude(vx_e, vy_e),
    }
    if imu is not None:
        sample["roll"] = imu["roll"]
        sample["pitch"] = imu["pitch"]
        sample["yaw"] = imu["yaw"]
    with debug_lock:
        latest_debug_sample = sample

def append_metrics_sample(t_now, vx_cmd_now, vy_cmd_now, w_cmd_now, vx_est, vy_est, w_est, yaw=None, roll=None, pitch=None):
    if t_now < 0.0:
        return
    metrics_log.append({
        "t": float(t_now),
        "vx_cmd": float(vx_cmd_now),
        "vy_cmd": float(vy_cmd_now),
        "w_cmd": float(w_cmd_now),
        "vx_est": float(vx_est),
        "vy_est": float(vy_est),
        "w_est": float(w_est),
        "yaw": None if yaw is None else float(yaw),
        "roll": None if roll is None else float(roll),
        "pitch": None if pitch is None else float(pitch),
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

def build_metrics_summary(include_attitude=True):
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

    err_vx = vx_cmd_arr - vx_est_arr
    err_vy = vy_cmd_arr - vy_est_arr
    err_wz = w_cmd_arr - w_est_arr
    yaw_err = yaw_arr - yaw_arr[0] if len(yaw_arr) else np.array([], dtype=float)

    body_attitude_rms = {
        "roll": float(np.sqrt(np.mean(roll_arr ** 2))) if include_attitude and len(roll_arr) else None,
        "pitch": float(np.sqrt(np.mean(pitch_arr ** 2))) if include_attitude and len(pitch_arr) else None,
    }
    peak_attitude_excursion = {
        "max_abs_roll": float(np.max(np.abs(roll_arr))) if include_attitude and len(roll_arr) else None,
        "max_abs_pitch": float(np.max(np.abs(pitch_arr))) if include_attitude and len(pitch_arr) else None,
    }

    return {
        "samples": int(len(metrics_log)),
        "duration_sec": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "velocity_rmse": {
            "vx": compute_rmse(vx_cmd_arr, vx_est_arr),
            "vy": compute_rmse(vy_cmd_arr, vy_est_arr),
            "wz": compute_rmse(w_cmd_arr, w_est_arr),
        },
        "body_attitude_rms": body_attitude_rms,
        "peak_attitude_excursion": peak_attitude_excursion,
        "settling_time_after_start_sec": compute_settling_time(times, err_vx, err_vy, err_wz),
        "slip_heading_drift_proxy": {
            "sideways_velocity_bias_mps": float(np.mean(vy_est_arr)),
            "integrated_abs_vy_m": integrate_abs(vy_est_arr, times),
            "integrated_abs_yaw_rate_rad": integrate_abs(w_est_arr, times),
            "integrated_abs_yaw_error_rad_s": integrate_abs(yaw_err, times),
        },
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

    headers = ["t", "vx_cmd", "vy_cmd", "w_cmd", "vx_est", "vy_est", "w_est", "yaw", "roll", "pitch"]
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
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    try:
        fig.canvas.manager.set_window_title("Go2 Speed Debug MuJoCo")
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

    line_vx_cmd, = axs[0].plot([], [], "r--", lw=1.8, label="vx cmd")
    line_vx_est, = axs[0].plot([], [], "r", lw=1.8, label="vx est")
    axs[0].set_ylabel("vx [m/s]")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    line_vy_cmd, = axs[1].plot([], [], "g--", lw=1.8, label="vy cmd")
    line_vy_est, = axs[1].plot([], [], "g", lw=1.8, label="vy est")
    axs[1].set_ylabel("vy [m/s]")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    line_w_cmd, = axs[2].plot([], [], "b--", lw=1.8, label="w cmd")
    line_w_est, = axs[2].plot([], [], "b", lw=1.8, label="w est")
    axs[2].set_ylabel("w [rad/s]")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    line_fr_cmd, = axs[3].plot([], [], "k--", lw=1.8, label="Fr cmd")
    line_fr_est, = axs[3].plot([], [], "k", lw=1.8, label="Fr est")
    axs[3].set_ylabel("Froude")
    axs[3].set_xlabel("time [s]")
    axs[3].legend(loc="upper right")
    axs[3].grid(True)

    dt_plot = 1.0 / DEBUG_PLOT_HZ
    for ax in axs:
        ax.set_xlim(0.0, DEBUG_WINDOW_SEC)

    while not debug_stop_event.is_set():
        with debug_lock:
            sample = None if latest_debug_sample is None else latest_debug_sample.copy()

        if sample is not None:
            hist_t.append(sample["t"])
            hist_vx_cmd.append(sample["vx_cmd"])
            hist_vx_est.append(sample["vx_est"])
            hist_vy_cmd.append(sample["vy_cmd"])
            hist_vy_est.append(sample["vy_est"])
            hist_w_cmd.append(sample["w_cmd"])
            hist_w_est.append(sample["w_est"])
            hist_fr_cmd.append(sample["fr_cmd"])
            hist_fr_est.append(sample["fr_est"])

            t_data = [t - hist_t[0] for t in hist_t] if len(hist_t) >= 2 else list(hist_t)
            line_vx_cmd.set_data(t_data, list(hist_vx_cmd))
            line_vx_est.set_data(t_data, list(hist_vx_est))
            line_vy_cmd.set_data(t_data, list(hist_vy_cmd))
            line_vy_est.set_data(t_data, list(hist_vy_est))
            line_w_cmd.set_data(t_data, list(hist_w_cmd))
            line_w_est.set_data(t_data, list(hist_w_est))
            line_fr_cmd.set_data(t_data, list(hist_fr_cmd))
            line_fr_est.set_data(t_data, list(hist_fr_est))

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
# 6. IK-DEPENDENT HELPERS (leg_ik itself is defined earlier)
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


def command_joint_positions(q):
    for aid, angle in zip(actuator_ids, q):
        data.ctrl[aid] = angle

def step_with_viewer(viewer=None):
    mujoco.mj_step(model, data)
    if viewer is not None and viewer.is_running():
        viewer.sync()



def compute_touchdown_body(leg):
    # Raibert heuristic: place foot ahead to account for body travel
    # during the remaining half-stance plus the full swing phase
    T_swing = GAIT_PERIOD * (1.0 - STANCE_RATIO)
    T_horizon = T_STANCE / 1.2 + T_swing
    dx = vx * T_horizon
    dy = vy * T_horizon
    dtheta = omega * T_horizon
    p_nom = NOMINAL_FEET[leg]
    td_xy = Rz(dtheta) @ p_nom[:2] + np.array([dx, dy])
    return np.array([td_xy[0], td_xy[1], p_nom[2]])

body_pose = np.array([0.0, 0.0, 0.0])
stance_feet = {}
swing_start_hip = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
touchdown_tgts_hip = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
prev_stance = {}

for leg in LEGS:
    world_xy = body_to_world_xy(NOMINAL_FEET[leg][:2], body_pose)
    stance_feet[leg] = np.array([world_xy[0], world_xy[1], NOMINAL_FEET[leg][2]])

initial_global_phase = 0.0
for leg in LEGS:
    prev_stance[leg] = in_stance(leg_phase(initial_global_phase, PHASE_OFFSET[leg]))

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
                    publish_debug_sample(
                        0.0,
                        vx, vy, omega,
                        vx_est, vy_est, w_est
                    )

                time.sleep(SIM_DT)
                continue

            body_pose_k = body_pose.copy()
            t_loco = elapsed - STAND_DUR
            g_phase = gait_phase(t_loco)
            q = []

            for leg in LEGS:
                local = leg_phase(g_phase, PHASE_OFFSET[leg])
                is_st = in_stance(local)

                if is_st and not prev_stance[leg]:
                    td_hip = touchdown_tgts_hip[leg]
                    td_body = td_hip + np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
                    td_world_xy = body_to_world_xy(td_body[:2], body_pose_k)
                    stance_feet[leg] = np.array([td_world_xy[0], td_world_xy[1], td_body[2]])

                if (not is_st) and prev_stance[leg]:
                    swing_start_hip[leg] = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
                    td_body = compute_touchdown_body(leg)
                    touchdown_tgts_hip[leg] = td_body - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

                if is_st:
                    foot_hip = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
                else:
                    s = swing_progress(local)
                    foot_hip = swing_foot_in_hip_frame(leg, s, swing_start_hip[leg], touchdown_tgts_hip[leg])

                angles = leg_ik(foot_hip[0], foot_hip[1], foot_hip[2], IS_LEFT[leg])
                q += list(angles)
                prev_stance[leg] = is_st

            command_joint_positions(q)
            step_with_viewer(viewer)

            vx_est, vy_est, w_est, yaw_est = get_estimated_body_velocities()
            append_metrics_sample(t_loco, vx, vy, omega, vx_est, vy_est, w_est, yaw=yaw_est)
            if ENABLE_DEBUG_PLOT:
                publish_debug_sample(t_loco, vx, vy, omega, vx_est, vy_est, w_est)

            body_pose[0] -= vx * SIM_DT
            body_pose[1] += vy * SIM_DT
            body_pose[2] = wrap_to_pi(body_pose[2] + omega * SIM_DT)
            time.sleep(SIM_DT)

except KeyboardInterrupt:
    pass
finally:
    summary = build_metrics_summary(include_attitude=False)
    save_and_print_metrics(summary, "go2_nonimu_mujoco")
    if ENABLE_DEBUG_PLOT:
        debug_stop_event.set()
        if debug_thread is not None:
            debug_thread.join(timeout=1.0)