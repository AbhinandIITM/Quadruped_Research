"""Isaac Lab observation & action space constants for Go2 migration.

Maps all rl/contracts.py constants to Isaac Lab equivalents for the
Go2 ManagerBasedEnvCfg. Observation order matches the CPU stack's 45-d
obs: [ang_vel(3), gravity(3), commands(3), dq(12), qd(12), prev_action(12)].
"""

from __future__ import annotations

import numpy as np

# Simulation timing (identical to contracts.py)
SIM_DT = 0.002  # physics timestep
CONTROL_HZ = 50  # policy rate
DECIMATION = 10  # physics steps per control step
EPISODE_S = 20.0
MAX_EPISODE_STEPS = int(EPISODE_S * CONTROL_HZ)

# Action / observation dims
NUM_ACTIONS = 12
OBS_DIM = 45

# Joint / body names (Isaac Lab USD joint naming: ".*_hip_joint", etc.)
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

FOOT_BODIES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
BASE_BODY = "base"

# Torque limits (from contracts.py / go2.usd)
TORQUE_LIMIT_HIP_THIGH = 23.7
TORQUE_LIMIT_CALF = 45.43

# PD gains
KP = 60.0
KV = 3.0

# Default / home joint positions (from contracts.py DEFAULT_JOINT_POS)
DEFAULT_JOINT_POS = np.tile(np.array([0.0, 0.9, -1.8]), 4).astype(np.float64)

# Standing base height
HOME_BASE_HEIGHT = 0.27

# Froude-number gait regime thresholds (m/s)
GAIT_REGIME_THRESHOLDS_MS = (0.7, 1.7)
LEG_LENGTH_M = 0.41

# Gravity
GRAVITY = np.array([0.0, 0.0, -9.81])

# Termination thresholds
TERMINATE_BASE_HEIGHT = 0.16
TERMINATE_MAX_ROLL_PITCH = 1.0

# Domain randomization ranges
FRICTION_RANGE = (0.6, 1.4)
MASS_SCALE_RANGE = (0.9, 1.1)
MOTOR_STRENGTH_RANGE = (0.9, 1.1)

# Observation indices (45-d)
OBS_IDX_ANG_VEL = slice(0, 3)
OBS_IDX_GRAVITY = slice(3, 6)
OBS_IDX_COMMANDS = slice(6, 9)
OBS_IDX_DQ = slice(9, 21)
OBS_IDX_QD = slice(21, 33)
OBS_IDX_PREV_ACTION = slice(33, 45)
