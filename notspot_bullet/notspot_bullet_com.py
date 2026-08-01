import math
import time
import numpy as np
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
    "FL": np.array([ BODY_L/2,   BODY_W/2 + HIP_OFFSET, -BODY_HEIGHT]),
    "RR": np.array([-BODY_L/2, -(BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "RL": np.array([-BODY_L/2,   BODY_W/2 + HIP_OFFSET, -BODY_HEIGHT]),
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

vx = 0.06
vy = 0.00
omega = 0.00

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
    "FR1_joint","FR2_joint","FR3_joint",
    "FL1_joint","FL2_joint","FL3_joint",
    "RR1_joint","RR2_joint","RR3_joint",
    "RL1_joint","RL2_joint","RL3_joint",
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

def stance_progress(local_phase):
    return local_phase / STANCE_RATIO

def swing_progress(local_phase):
    return (local_phase - STANCE_RATIO) / (1.0 - STANCE_RATIO)

def body_to_world_xy(body_xy, body_pose):
    return Rz(body_pose[2]) @ body_xy + body_pose[:2]

def world_to_body_xy(world_xy, body_pose):
    return Rz(-body_pose[2]) @ (world_xy - body_pose[:2])

def hip_world_xy(leg, body_pose):
    return body_to_world_xy(HIP_BODY[leg], body_pose)

def nominal_foot_hip(leg):
    return NOMINAL_FEET[leg] - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

# ============================================================
# 6. IK : Hip to feet . Hip is fixed.
# ============================================================

def leg_ik(x, y, z, is_left=False):
    l1 = HIP_OFFSET if is_left else -HIP_OFFSET
    yz = y*y + z*z

    if yz <= HIP_OFFSET**2:
        return 0.0, 0.0, 0.0

    theta1 = (
        math.atan2(y, -z)
        - math.atan2(l1, math.sqrt(yz - HIP_OFFSET**2))
    )

    d = x*x + yz - HIP_OFFSET**2
    D = (d - THIGH**2 - CALF**2) / (2 * THIGH * CALF)
    D = max(-1.0, min(1.0, D))
    theta3 = -math.acos(D)

    a = math.atan2(-x, math.sqrt(yz - HIP_OFFSET**2))
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
    hip_xy_world = hip_world_xy(leg, body_pose)
    foot_xy_body = world_to_body_xy(stance_foot_world[:2], body_pose)
    foot_body = np.array([foot_xy_body[0], foot_xy_body[1], stance_foot_world[2]])
    hip_body = np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
    return foot_body - hip_body

# ============================================================
# 8. SWING CONTROLLER
# ============================================================

def compute_touchdown_body(leg):
    dx = vx * T_STANCE
    dy = vy * T_STANCE
    dtheta = omega * T_STANCE
    p_nom = NOMINAL_FEET[leg]
    td_xy = Rz(dtheta) @ p_nom[:2] + np.array([dx, dy])
    return np.array([td_xy[0], td_xy[1], p_nom[2]])

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
# 11. INITIAL STAND-UP
# ============================================================

SIM_DT = 1 / 240.0
STAND_DUR = 3.0
start = time.time()

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
        time.sleep(SIM_DT)
        continue

    # Use pose at the START of this control step for all kinematics
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
    time.sleep(SIM_DT)

    # Integrate body pose AFTER generating this step's commands
    body_pose[0] -= vx * SIM_DT
    body_pose[1] += vy * SIM_DT
    body_pose[2] += omega * SIM_DT
