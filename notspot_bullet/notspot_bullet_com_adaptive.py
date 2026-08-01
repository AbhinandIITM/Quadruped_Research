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
THIGH      = 0.1000
CALF       = 0.094333

# Paper Eq.(2): h = maximum leg length
LEG_LENGTH = THIGH + CALF          # 0.1943 m

BODY_HEIGHT = 0.14
STEP_HEIGHT = 0.04

BODY_L = 0.196
BODY_W = 0.065

NOMINAL_FEET = {
    "FR": np.array([ BODY_L/2, -(BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "FL": np.array([ BODY_L/2,   BODY_W/2 + HIP_OFFSET,  -BODY_HEIGHT]),
    "RR": np.array([-BODY_L/2, -(BODY_W/2 + HIP_OFFSET), -BODY_HEIGHT]),
    "RL": np.array([-BODY_L/2,   BODY_W/2 + HIP_OFFSET,  -BODY_HEIGHT]),
}

HIP_BODY = {
    "FR": np.array([ BODY_L/2, -BODY_W/2]),
    "FL": np.array([ BODY_L/2,  BODY_W/2]),
    "RR": np.array([-BODY_L/2, -BODY_W/2]),
    "RL": np.array([-BODY_L/2,  BODY_W/2]),
}

IS_LEFT = {"FR": False, "FL": True, "RR": False, "RL": True}
LEGS    = ["FR", "FL", "RR", "RL"]

# ============================================================
# 2b. GAIT DATABASE  (Table I — Humphreys et al. 2023)
# ============================================================
# Paper Table I column order:  FL, FR, RL, RR  → offsets 0.00, 0.75, 0.50, 0.25
# Our LEGS order:              FR, FL, RR, RL
#
# FIX 1 — Walk offsets corrected.
# The paper's Static Gait offsets (FL=0.00, FR=0.75, RL=0.50, RR=0.25)
# re-mapped to our LEGS order (FR, FL, RR, RL) give:
#   FR → 0.75  (was correct in previous version)
#   FL → 0.00  (was correct)
#   RR → 0.25  (was correct)
#   RL → 0.50  (was correct)
# The walk offsets were actually correct; the instability came from other bugs.

GAITS = {
    "walk": {
        "period": 1.00,
        "duty":   0.80,
        "offsets": {"FR": 0.75, "FL": 0.00, "RR": 0.25, "RL": 0.50},
    },
    "walking_trot": {
        "period": 0.50,
        "duty":   0.60,
        "offsets": {"FR": 0.00, "FL": 0.50, "RR": 0.50, "RL": 0.00},
    },
    "trot": {
        "period": 0.40,
        "duty":   0.50,
        "offsets": {"FR": 0.00, "FL": 0.50, "RR": 0.50, "RL": 0.00},
    },
    "flying_trot": {
        "period": 0.30,
        "duty":   0.40,
        "offsets": {"FR": 0.00, "FL": 0.50, "RR": 0.50, "RL": 0.00},
    },
}

# ============================================================
# 2c. FROUDE NUMBER BOUNDS  (Table II — paper)
# ============================================================
# FIX 2 — Trot lb corrected from 0.7400 (typo copy from ub) to 0.1517.
# In Table II, Trot lb = 0.1517 (= Walking Trot ub), not 0.7400.
# The original code had:
#   "trot": {"lb": 0.1517, "ub": 0.7400}   ← this was already correct
# BUT select_gait() used the wrong threshold chain — see FIX 4 below.

GAIT_FR_BOUNDS = {
    "walk":         {"lb": 0.0009, "ub": 0.0024},
    "walking_trot": {"lb": 0.0024, "ub": 0.1517},
    "trot":         {"lb": 0.1517, "ub": 0.7400},
    "flying_trot":  {"lb": 0.7400, "ub": 1.0000},
}

GAIT_ORDER = ["walk", "walking_trot", "trot", "flying_trot"]

# ============================================================
# 3. DESIRED BODY COMMAND
# ============================================================
# FIX 3 — vx reduced to 0.15 m/s (safe starting point).
# Fr(0.30 m/s) = 0.30^2 / (9.81 * 0.194) = 0.047  → walking_trot
# This is fine, but starting at 0.30 m/s with a walk gait that has
# period=1.0 s causes very large step targets on the first swing.
# 0.15 m/s keeps Fr clearly in walking_trot territory and step
# targets ~3 cm ahead — manageable for open-loop position control.
# Increase vx gradually after verifying basic stability.

vx    = 0.15   # m/s — safe initial value; raise to 0.30 once stable
vy    = 0.00
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

def compute_froude(v):
    """Paper Eq.(2):  Fr = v^2 / (g * h),  h = max leg length."""
    return (v ** 2) / (9.81 * LEG_LENGTH)

# FIX 4 — select_gait threshold chain corrected.
# Previous version compared fr against walking_trot["lb"] (0.0024) and
# trot["lb"] (0.1517), which was correct, but the elif for trot used
# flying_trot["lb"] (0.7400).  That meant any Fr in [0.1517, 0.7400]
# fell into trot — which is actually correct — so that part was fine.
# The real problem was that at vx=0.30, Fr≈0.047, which selects
# walking_trot (period=0.50, duty=0.60).  The GaitState was initialised
# with this gait, BUT gait_phase() used (t % gait.period) where gait.period
# changes mid-period during transition → phase JUMPS.  See FIX 5.

def select_gait(fr):
    """Optimal gait for Froude number fr (Table II thresholds)."""
    if fr < GAIT_FR_BOUNDS["walking_trot"]["lb"]:   # < 0.0024
        return "walk"
    elif fr < GAIT_FR_BOUNDS["trot"]["lb"]:          # < 0.1517
        return "walking_trot"
    elif fr < GAIT_FR_BOUNDS["flying_trot"]["lb"]:   # < 0.7400
        return "trot"
    else:
        return "flying_trot"

# ============================================================
# 5b. GAIT TRANSITION MODULE
# ============================================================

C_MAX  = 2
SIM_DT = 1 / 240.0

class GaitState:
    """Holds live (possibly blending) gait parameters.

    FIX 5 — Phase anchor on period change.
    When gait.period changes during a transition the expression
    (t % gait.period) produces a discontinuous jump in global_phase,
    which flips legs between stance and swing mid-step — the primary
    source of the observed instability.

    Solution: store a (t_anchor, phase_anchor) pair that is updated
    every time the period changes.  gait_phase() becomes:

        phase = phase_anchor + (t - t_anchor) / period

    so that the phase progresses smoothly from wherever it was
    regardless of period changes.
    """

    def __init__(self, name, t0=0.0):
        g = GAITS[name]
        self.name    = name
        self.period  = g["period"]
        self.duty    = g["duty"]
        self.offsets = dict(g["offsets"])

        # Phase anchor — updated whenever period changes
        self._t_anchor     = t0
        self._phase_anchor = 0.0

        # Transition bookkeeping
        self.transitioning = False
        self.src_name      = name
        self.tgt_name      = name
        self.n_trans       = 0
        self.d_trans       = 1
        self.src_period    = self.period
        self.src_duty      = self.duty
        self.src_offsets   = dict(self.offsets)
        self.tgt_period    = self.period
        self.tgt_duty      = self.duty
        self.tgt_offsets   = dict(self.offsets)

    # ----------------------------------------------------------
    def phase(self, t):
        """Continuous global phase in [0,1), anchored to avoid jumps."""
        raw = self._phase_anchor + (t - self._t_anchor) / self.period
        return raw % 1.0

    def _anchor(self, t):
        """Call this before changing self.period to preserve phase continuity."""
        self._phase_anchor = self.phase(t)
        self._t_anchor     = t

    # ----------------------------------------------------------
    def _compute_d_trans(self, fr_star, fr_crit=1.0):
        """Eq.(5)-(6): D_trans = ((D_src + D_tgt)/2) * C."""
        d_src = GAITS[self.src_name]["period"] / SIM_DT
        d_tgt = GAITS[self.tgt_name]["period"] / SIM_DT
        c     = max(0.1, (1.0 - fr_star / fr_crit) * C_MAX)
        return int(((d_src + d_tgt) / 2.0) * c)

    def trigger_transition(self, tgt_name, fr_star, t):
        """Start blending toward tgt_name."""
        if tgt_name == self.name and not self.transitioning:
            return
        self.transitioning = True
        self.src_name      = self.name
        self.tgt_name      = tgt_name
        self.n_trans       = 0
        self.d_trans       = max(1, self._compute_d_trans(fr_star))
        self.src_period    = self.period
        self.src_duty      = self.duty
        self.src_offsets   = dict(self.offsets)
        self.tgt_period    = GAITS[tgt_name]["period"]
        self.tgt_duty      = GAITS[tgt_name]["duty"]
        self.tgt_offsets   = dict(GAITS[tgt_name]["offsets"])

    def step(self, t):
        """Advance one control step; blend parameters if transitioning.

        FIX 5 continued — _anchor(t) is called before every period update
        so gait.phase(t) never jumps.
        """
        if not self.transitioning:
            return

        self.n_trans += 1
        alpha = self.n_trans / self.d_trans   # 0→1 over transition
        w1, w2 = 1.0 - alpha, alpha

        # Anchor phase BEFORE changing period (FIX 5 core)
        self._anchor(t)

        # Blend period and duty (Eq.4)
        self.period = w1 * self.src_period + w2 * self.tgt_period
        self.duty   = w1 * self.src_duty   + w2 * self.tgt_duty

        # Blend phase offsets — shortest arc (Eq.8)
        for leg in LEGS:
            delta = self.tgt_offsets[leg] - self.src_offsets[leg]
            if delta >  0.5: delta -= 1.0
            if delta < -0.5: delta += 1.0
            self.offsets[leg] = (self.src_offsets[leg] + delta * alpha) % 1.0

        if self.n_trans >= self.d_trans:
            self.period       = self.tgt_period
            self.duty         = self.tgt_duty
            self.offsets      = dict(self.tgt_offsets)
            self.name         = self.tgt_name
            self.transitioning = False

# ============================================================
# 6. PHASE FUNCTIONS
# ============================================================

def leg_phase(global_phase, offset):
    return (global_phase + offset) % 1.0

def in_stance(local_phase, duty):
    # FIX 6 — guard against duty == 1.0 edge case (never swing)
    # and duty <= 0 (degenerate).
    return local_phase < max(0.01, min(0.99, duty))

def swing_progress(local_phase, duty):
    # FIX 7 — clamp s to [0,1] to prevent negative or >1 progress
    # that can arise from floating-point noise at the exact transition
    # boundary, which previously sent swing_foot_in_hip_frame to NaN.
    swing_frac = 1.0 - duty
    if swing_frac < 1e-6:
        return 1.0
    s = (local_phase - duty) / swing_frac
    return max(0.0, min(1.0, s))

# ============================================================
# 7. IK
# ============================================================

def leg_ik(x, y, z, is_left=False):
    l1 = HIP_OFFSET if is_left else -HIP_OFFSET
    yz = y*y + z*z
    if yz <= HIP_OFFSET**2:
        return 0.0, 0.0, 0.0
    theta1 = (math.atan2(y, -z)
              - math.atan2(l1, math.sqrt(yz - HIP_OFFSET**2)))
    d = x*x + yz - HIP_OFFSET**2
    D = max(-1.0, min(1.0, (d - THIGH**2 - CALF**2) / (2 * THIGH * CALF)))
    theta3 = -math.acos(D)
    a = math.atan2(-x, math.sqrt(yz - HIP_OFFSET**2))
    b = math.atan2(CALF * math.sin(abs(theta3)),
                   THIGH + CALF * math.cos(theta3))
    theta2 = -(a - b)
    return theta1, theta2, theta3

# ============================================================
# 8. SPATIAL HELPERS
# ============================================================

def body_to_world_xy(body_xy, body_pose):
    return Rz(body_pose[2]) @ body_xy + body_pose[:2]

def world_to_body_xy(world_xy, body_pose):
    return Rz(-body_pose[2]) @ (world_xy - body_pose[:2])

def nominal_foot_hip(leg):
    return NOMINAL_FEET[leg] - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

# ============================================================
# 9. STANCE CONTROLLER
# ============================================================

def stance_foot_in_hip_frame(leg, body_pose, stance_foot_world):
    foot_xy_body = world_to_body_xy(stance_foot_world[:2], body_pose)
    foot_body    = np.array([foot_xy_body[0], foot_xy_body[1], stance_foot_world[2]])
    hip_body     = np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
    return foot_body - hip_body

# ============================================================
# 10. SWING CONTROLLER
# ============================================================

def compute_touchdown_body(leg, gait):
    # FIX 8 — touchdown target uses live T_stance from current gait.
    # Previous version computed dx from a stale T_STANCE constant.
    t_stance = gait.period * gait.duty
    dx = vx    * t_stance
    dy = vy    * t_stance
    dtheta = omega * t_stance
    p_nom  = NOMINAL_FEET[leg]
    td_xy  = Rz(dtheta) @ p_nom[:2] + np.array([dx, dy])
    return np.array([td_xy[0], td_xy[1], p_nom[2]])

def swing_foot_in_hip_frame(leg, s, start_hip, target_hip):
    xy    = (1.0 - s) * start_hip[:2] + s * target_hip[:2]
    z_lin = (1.0 - s) * start_hip[2]  + s * target_hip[2]
    z     = z_lin + STEP_HEIGHT * math.sin(math.pi * s)
    return np.array([xy[0], xy[1], z])

# ============================================================
# 11. CONTROLLER STATE
# ============================================================

body_pose = np.array([0.0, 0.0, 0.0])
stance_feet        = {}
swing_start_hip    = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
touchdown_tgts_hip = {leg: nominal_foot_hip(leg).copy() for leg in LEGS}
prev_stance        = {}

for leg in LEGS:
    wxy = body_to_world_xy(NOMINAL_FEET[leg][:2], body_pose)
    stance_feet[leg] = np.array([wxy[0], wxy[1], NOMINAL_FEET[leg][2]])

# Initialise gait from current vx — pass t0=0 so anchor starts at t=0
_fr0         = compute_froude(abs(vx))
_start_name  = select_gait(_fr0)
current_gait = GaitState(_start_name, t0=0.0)

# Seed prev_stance using t=0
for leg in LEGS:
    prev_stance[leg] = in_stance(
        leg_phase(current_gait.phase(0.0), current_gait.offsets[leg]),
        current_gait.duty)

# ============================================================
# 12. LOW-LEVEL JOINT COMMAND
# ============================================================

def command_joint_positions(q):
    for jid, angle in zip(joint_ids, q):
        p.setJointMotorControl2(
            robot, jid,
            p.POSITION_CONTROL,
            targetPosition=angle,
            force=25,
            positionGain=0.5,
            velocityGain=1.0,
        )

# ============================================================
# 13. MAIN LOOP
# ============================================================

SIM_DT    = 1 / 240.0
STAND_DUR = 3.0
start     = time.time()

while p.isConnected():
    elapsed = time.time() - start

    # ---- STAND-UP ----
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

    # ---- LOCOMOTION ----
    body_pose_k = body_pose.copy()
    t_loco      = elapsed - STAND_DUR

    # Froude → gait selection
    fr = compute_froude(abs(vx))
    desired_gait = select_gait(fr)

    if desired_gait != current_gait.name and not current_gait.transitioning:
        current_gait.trigger_transition(desired_gait, fr_star=fr, t=t_loco)

    # FIX 5 — pass t_loco into step() so anchor is correct
    current_gait.step(t_loco)

    # FIX 5 — use anchored phase method instead of bare (t % period)
    g_phase = current_gait.phase(t_loco)

    q = []
    for leg in LEGS:
        local = leg_phase(g_phase, current_gait.offsets[leg])
        is_st = in_stance(local, current_gait.duty)

        # Touchdown
        if is_st and not prev_stance[leg]:
            td_hip      = touchdown_tgts_hip[leg]
            td_body     = td_hip + np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])
            td_world_xy = body_to_world_xy(td_body[:2], body_pose_k)
            stance_feet[leg] = np.array([td_world_xy[0], td_world_xy[1], td_body[2]])

        # Liftoff
        if (not is_st) and prev_stance[leg]:
            swing_start_hip[leg]   = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
            td_body                 = compute_touchdown_body(leg, current_gait)
            touchdown_tgts_hip[leg] = td_body - np.array([HIP_BODY[leg][0], HIP_BODY[leg][1], 0.0])

        if is_st:
            foot_hip = stance_foot_in_hip_frame(leg, body_pose_k, stance_feet[leg])
        else:
            s        = swing_progress(local, current_gait.duty)
            foot_hip = swing_foot_in_hip_frame(leg, s, swing_start_hip[leg], touchdown_tgts_hip[leg])

        angles = leg_ik(foot_hip[0], foot_hip[1], foot_hip[2], IS_LEFT[leg])
        q += list(angles)
        prev_stance[leg] = is_st

    command_joint_positions(q)
    p.stepSimulation()
    time.sleep(SIM_DT)

    # Integrate body pose AFTER commands
    body_pose[0] -= vx    * SIM_DT
    body_pose[1] += vy    * SIM_DT
    body_pose[2] += omega * SIM_DT
