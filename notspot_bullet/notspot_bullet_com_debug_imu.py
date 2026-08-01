import math
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


# ============================================================
# 1. PYBULLET INITIALIZATION
# ============================================================

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

robot = p.loadURDF(
    "notspot_pybullet.urdf",
    [0, 0, 0.40],
    useFixedBase=False,
)


# ============================================================
# 2. ROBOT GEOMETRY
# ============================================================

HIP_OFFSET = 0.0469
THIGH = 0.1000
CALF = 0.094333

BODY_HEIGHT = 0.14
STEP_HEIGHT = 0.04
GAIT_PERIOD = 0.6
STANCE_RATIO = 0.5
T_STANCE = GAIT_PERIOD * STANCE_RATIO

BODY_L = 0.196
BODY_W = 0.065

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

PHASE_OFFSET = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
IS_LEFT = {"FR": False, "FL": True, "RR": False, "RL": True}
LEGS = ["FR", "FL", "RR", "RL"]


# ============================================================
# 3. DESIRED BODY COMMAND
# ============================================================

vx_cmd = 0.06
vy_cmd = 0.00
omega_cmd = 0.00

ENABLE_IMU_VX_FEEDBACK_ONLY = False
ENABLE_IMU_PLANAR_FEEDBACK = False

KVX = 0.0
KVY = 0.15
KW = 0.08

VX_LIM = 0.25
VY_LIM = 0.20
W_LIM = 1.20


# ============================================================
# 3A. DEBUG THREAD CONFIG
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


# ============================================================
# 4. JOINT MAP
# ============================================================

joint_map = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    if info[2] == p.JOINT_REVOLUTE:
        name = info[1].decode()
        joint_map[name] = i
        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, force=0)

joint_order = [
    "FR1_joint", "FR2_joint", "FR3_joint",
    "FL1_joint", "FL2_joint", "FL3_joint",
    "RR1_joint", "RR2_joint", "RR3_joint",
    "RL1_joint", "RL2_joint", "RL3_joint",
]
joint_ids = [joint_map[n] for n in joint_order]


# ============================================================
# 5. HELPERS
# ============================================================

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


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_imu_state(robot_id):
    _, quat = p.getBasePositionAndOrientation(robot_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(quat)

    lin_world, ang_world = p.getBaseVelocity(robot_id)
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


def compute_froude(vx_now, vy_now):
    v = math.sqrt(vx_now * vx_now + vy_now * vy_now)
    return (v * v) / (G * L_CHAR)


def compute_touchdown_body(leg, vx_use, vy_use, omega_use):
    alpha = 0.5
    dx = vx_use * (alpha * T_STANCE)
    dy = vy_use * (alpha * T_STANCE)
    dtheta = omega_use * (alpha * T_STANCE)
    p_nom = NOMINAL_FEET[leg]
    td_xy = Rz(dtheta) @ p_nom[:2] + np.array([dx, dy])
    return np.array([td_xy[0], td_xy[1], p_nom[2]])


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


def debug_plot_worker():
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("Notspot Speed Debug")

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

            if len(hist_t) >= 2:
                t0 = hist_t[0]
                t_data = [t - t0 for t in hist_t]
            else:
                t_data = list(hist_t)

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


debug_thread = None
if ENABLE_DEBUG_PLOT:
    debug_thread = threading.Thread(target=debug_plot_worker, daemon=True)
    debug_thread.start()


# ============================================================
# 6. IK : Hip to feet . Hip is fixed.
# ============================================================

def leg_ik(x, y, z, is_left=False):
    l1 = HIP_OFFSET if is_left else -HIP_OFFSET
    yz = y * y + z * z

    if yz <= HIP_OFFSET ** 2:
        return 0.0, 0.0, 0.0

    theta1 = (
        math.atan2(y, -z)
        - math.atan2(l1, math.sqrt(yz - HIP_OFFSET ** 2))
    )

    d = x * x + yz - HIP_OFFSET ** 2
    D = (d - THIGH ** 2 - CALF ** 2) / (2 * THIGH * CALF)
    D = max(-1.0, min(1.0, D))
    theta3 = -math.acos(D)

    a = math.atan2(-x, math.sqrt(yz - HIP_OFFSET ** 2))
    b = math.atan2(
        CALF * math.sin(abs(theta3)),
        THIGH + CALF * math.cos(theta3)
    )
    theta2 = -(a - b)

    return theta1, theta2, theta3


# ============================================================
# 7. STANCE CONTROLLER : foot to hip
# ============================================================

def stance_foot_in_hip_frame(leg, body_pose, stance_foot_world):
    foot_xy_body = world_to_body_xy(stance_foot_world[:2], body_pose)
    foot_body = np.array([foot_xy_body[0], foot_xy_body[1], stance_foot_world[2]])
    hip_body = np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
    return foot_body - hip_body


# ============================================================
# 8. SWING CONTROLLER
# ============================================================

def swing_foot_in_hip_frame(leg, s, start_hip, target_hip):
    xy = (1.0 - s) * start_hip[:2] + s * target_hip[:2]
    z0 = start_hip[2]
    zf = target_hip[2]
    z_lin = (1.0 - s) * z0 + s * zf
    z = z_lin + STEP_HEIGHT * math.sin(math.pi * s)
    return np.array([xy[0], xy[1], z])


# ============================================================
# 9. CONTROLLER STATE
# ============================================================

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


# ============================================================
# 10. LOW LEVEL JOINT COMMAND
# ============================================================

def command_joint_positions(q):
    for jid, angle in zip(joint_ids, q):
        p.setJointMotorControl2(
            robot,
            jid,
            p.POSITION_CONTROL,
            targetPosition=angle,
            force=25,
            positionGain=0.5,
            velocityGain=1.0,
        )


# ============================================================
# 11. INITIAL STAND-UP + MAIN LOOP
# ============================================================

SIM_DT = 1 / 240.0
STAND_DUR = 3.0
start = time.time()

try:
    while p.isConnected():
        elapsed = time.time() - start

        if elapsed < STAND_DUR:
            progress = (1 - math.cos(math.pi * elapsed / STAND_DUR)) / 2
            q = []
            for leg in LEGS:
                x, y, z = nominal_foot_hip(leg)
                angles = leg_ik(x, y, z, IS_LEFT[leg])
                q += [progress * a for a in angles]

            command_joint_positions(q)
            p.stepSimulation()

            if ENABLE_DEBUG_PLOT:
                imu = get_imu_state(robot)
                publish_debug_sample(
                    0.0,
                    vx_cmd, vy_cmd, omega_cmd,
                    imu["vx"], imu["vy"], imu["wz"],
                    imu
                )

            time.sleep(SIM_DT)
            continue

        imu = get_imu_state(robot)

        vx_est = imu["vx"]
        vy_est = imu["vy"]
        w_est = imu["wz"]

        if ENABLE_IMU_VX_FEEDBACK_ONLY:
            vx_use = vx_cmd + KVX * (vx_cmd - vx_est)
            vy_use = vy_cmd
            omega_use = omega_cmd
        elif ENABLE_IMU_PLANAR_FEEDBACK:
            vx_use = vx_cmd + KVX * (vx_cmd - vx_est)
            vy_use = vy_cmd + KVY * (vy_cmd - vy_est)
            omega_use = omega_cmd + KW * (omega_cmd - w_est)
        else:
            vx_use = vx_cmd
            vy_use = vy_cmd
            omega_use = omega_cmd

        vx_use = clamp(vx_use, -VX_LIM, VX_LIM)
        vy_use = clamp(vy_use, -VY_LIM, VY_LIM)
        omega_use = clamp(omega_use, -W_LIM, W_LIM)

        body_pose_k = body_pose.copy()
        body_pose_k[2] = imu["yaw"]

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
                td_body = compute_touchdown_body(leg, vx_use, vy_use, omega_use)
                touchdown_tgts_hip[leg] = td_body - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

            if is_st:
                foot_hip = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
            else:
                s = swing_progress(local)
                foot_hip = swing_foot_in_hip_frame(
                    leg,
                    s,
                    swing_start_hip[leg],
                    touchdown_tgts_hip[leg],
                )

            angles = leg_ik(foot_hip[0], foot_hip[1], foot_hip[2], IS_LEFT[leg])
            q += list(angles)
            prev_stance[leg] = is_st

        command_joint_positions(q)
        p.stepSimulation()

        if ENABLE_DEBUG_PLOT:
            publish_debug_sample(
                t_loco,
                vx_cmd, vy_cmd, omega_cmd,
                vx_est, vy_est, w_est,
                imu
            )

        body_pose[0] -= vx_use * SIM_DT
        body_pose[1] += vy_use * SIM_DT
        body_pose[2] = imu["yaw"]
        print(w_est)

        time.sleep(SIM_DT)

except KeyboardInterrupt:
    pass

finally:
    if ENABLE_DEBUG_PLOT:
        debug_stop_event.set()
        if debug_thread is not None:
            debug_thread.join(timeout=1.0)

    if p.isConnected():
        p.disconnect()
        