"""
go2_bullet.py
------------------------------------------------------------
PyBullet implementation of a trot gait for the Unitree Go2 robot.

Structure follows the accompanying IK report:

1. Imports and URDF Modification
2. PyBullet Initialization
3. Robot Parameters
4. Joint Interface
5. Locomotion Command Representation
6. Gait Scheduler
7. Stance Controller
8. Swing Controller
9. Foot Trajectory Generator
10. Closed-form Leg Inverse Kinematics
11. Joint Command Interface
12. Main Control Loop
------------------------------------------------------------
"""

import os
import math
import time
import pybullet as p
import pybullet_data

# ============================================================
# 1. URDF PREPARATION
# ============================================================

# Get local directory paths
here = os.path.dirname(os.path.abspath(__file__))
urdf_src = os.path.join(here, "unitree_go2", "urdf", "go2_description.urdf")
urdf_dst = os.path.join(here, "unitree_go2", "urdf", "go2_description_bullet.urdf")

# Convert package:// paths to relative paths for PyBullet
if not os.path.exists(urdf_dst):
    print("Generating modified URDF for PyBullet...")
    with open(urdf_src, "r", encoding="utf-8") as f:
        content = f.read()
    # Replace package paths with relative paths
    content = content.replace("package://go2_description/dae/", "../meshes/")
    with open(urdf_dst, "w", encoding="utf-8") as f:
        f.write(content)

# ============================================================
# 2. PYBULLET INITIALIZATION
# ============================================================

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.resetSimulation()
p.setGravity(0, 0, -9.81)

p.loadURDF("plane.urdf")

# Go2 robot starts slightly higher due to longer legs
robot = p.loadURDF(
    urdf_dst,
    [0, 0, 0.45],
    useFixedBase=False
)

# ============================================================
# 3. ROBOT GEOMETRY (URDF Ground Truth)
# ============================================================

# Go2 physical dimensions (from URDF)
HIP_OFFSET = 0.0955
THIGH = 0.213
CALF = 0.213

BODY_HEIGHT = 0.32
STEP_LENGTH = 0.08
STEP_HEIGHT = 0.05

GAIT_PERIOD = 1.0

# ============================================================
# 4. DESIRED BODY COMMAND
# ============================================================

vx = 0.15
vy = 0.0
omega = 0.0

# ============================================================
# 5. JOINT MAP
# ============================================================

joint_map = {}

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)

    if info[2] == p.JOINT_REVOLUTE:
        name = info[1].decode()
        joint_map[name] = i

        p.setJointMotorControl2(
            robot,
            i,
            p.VELOCITY_CONTROL,
            force=0
        )

# Go2 joint names
joint_order = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]

joint_ids = [joint_map[n] for n in joint_order]

# ============================================================
# 6. GAIT SCHEDULER
# ============================================================

def gait_phase(t):
    return (t % GAIT_PERIOD) / GAIT_PERIOD

# ============================================================
# 7. STANCE CONTROLLER (Corrected: moves foot backward to propel body forward)
# ============================================================

def stance_controller(local_phase):
    s = local_phase / 0.5
    x = STEP_LENGTH/2 - STEP_LENGTH*s
    return x, 0.0, -BODY_HEIGHT

# ============================================================
# 8. SWING CONTROLLER (Corrected: steps foot forward in the air)
# ============================================================

def swing_controller(local_phase):
    s = (local_phase - 0.5)/0.5
    x = -STEP_LENGTH/2 + STEP_LENGTH*s
    z = -BODY_HEIGHT + STEP_HEIGHT*math.sin(math.pi*s)
    return x, 0.0, z

# ============================================================
# 9. FOOT TRAJECTORY
# ============================================================

def foot_trajectory(global_phase, offset):

    phase = (global_phase + offset) % 1.0

    if phase < 0.5:
        return stance_controller(phase)

    return swing_controller(phase)

# ============================================================
# 10. CLOSED FORM LEG IK
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

    D = (d - THIGH**2 - CALF**2)/(2*THIGH*CALF)
    D = max(-1.0, min(1.0, D))

    theta3 = -math.acos(D)

    a = math.atan2(-x, math.sqrt(yz - HIP_OFFSET**2))
    b = math.atan2(
        CALF*math.sin(abs(theta3)),
        THIGH + CALF*math.cos(theta3)
    )

    theta2 = -(a - b)

    return theta1, theta2, theta3

# ============================================================
# 11. JOINT COMMAND INTERFACE
# ============================================================

def command_joint_positions(q):

    for jid, angle in zip(joint_ids, q):

        p.setJointMotorControl2(
            robot,
            jid,
            p.POSITION_CONTROL,
            targetPosition=angle,
            force=50,  # Torque suitable for Go2 weight
            positionGain=0.5,
            velocityGain=1.0
        )

# ============================================================
# 12. INITIAL STAND-UP AND MAIN CONTROL LOOP
# ============================================================

start = time.time()

while p.isConnected():

    elapsed = time.time() - start

    if elapsed < 3.0:
        # Interpolate from zero positions to standing height over 3 seconds
        progress = (1 - math.cos(math.pi*elapsed/3))/2

        fr = leg_ik(0, 0, -BODY_HEIGHT, False)
        fl = leg_ik(0, 0, -BODY_HEIGHT, True)
        rr = leg_ik(0, 0, -BODY_HEIGHT, False)
        rl = leg_ik(0, 0, -BODY_HEIGHT, True)

        q = [
            *(progress*x for x in fr),
            *(progress*x for x in fl),
            *(progress*x for x in rr),
            *(progress*x for x in rl),
        ]

    else:
        # Execute Trot Gait
        phase = gait_phase(elapsed-3.0)

        # Diagonal pairs
        fr_pos = foot_trajectory(phase, 0.0)
        rl_pos = foot_trajectory(phase, 0.0)

        fl_pos = foot_trajectory(phase, 0.5)
        rr_pos = foot_trajectory(phase, 0.5)

        # Calculate IK
        fr = leg_ik(*fr_pos, False)
        fl = leg_ik(*fl_pos, True)
        rr = leg_ik(*rr_pos, False)
        rl = leg_ik(*rl_pos, True) # Fixed bug: rl was not updated in stance/swing loop
       
        q = [*fr, *fl, *rr, *rl]

    command_joint_positions(q)

    p.stepSimulation()
    time.sleep(1/240)
