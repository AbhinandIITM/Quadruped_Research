"""Frozen interfaces for the LLM-vs-RL quadruped study.

Every module in ``rl`` imports from here and nothing here imports from
them. This file is authored once, before any parallel work starts, and is not edited
afterwards -- it is the mechanism that lets independent modules integrate without
runtime negotiation.

Verified facts encoded below (measured against
``third_party/unitree_mujoco/unitree_robots/go2/go2.xml``, Oct 2026):

* 12 actuators, ``nq=19`` (7 free-joint + 12 hinge), ``nv=18``, ``dt=0.002``, mass 15.21 kg.
* Actuator order is FR, FL, RR, RL but qpos joint order is FL, FR, RL, RR -- a real
  permutation, see ``ACTUATOR_TO_QPOS``. Getting this wrong silently mirrors the gait.
* The model ships a ``home`` keyframe: base height 0.27 m, joints ``0, 0.9, -1.8`` per leg.
  ``HOME_QPOS`` below is that keyframe verbatim, so ``DEFAULT_JOINT_POS`` matches the model
  rather than a guess.
* **The model has no ground geom** -- 56 geoms, no plane. A model compiled from ``go2.xml``
  alone has nothing to stand on, so every terrain kind (including ``flat``) must go through
  ``envs/terrain.py``, which supplies the floor. Standing height is 0.27 m, not 0.
* There is no base joint-position sensor; ``frame_pos``/``frame_vel`` provide base state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------------------
# Model constants
# --------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GO2_XML = os.path.join(
    REPO_ROOT, "third_party", "unitree_mujoco", "unitree_robots", "go2", "go2.xml"
)

SIM_DT = 0.002  # model physics timestep
CONTROL_HZ = 50  # policy rate (matches the surveyed papers)
DECIMATION = int(round((1.0 / CONTROL_HZ) / SIM_DT))  # 10 physics steps per control step
EPISODE_S = 20.0
MAX_EPISODE_STEPS = int(EPISODE_S * CONTROL_HZ)

NUM_ACTIONS = 12
OBS_DIM = 45  # 3 + 3 + 3 + 12 + 12 + 12

#: Actuator order as MuJoCo builds it, and the qpos index of each actuator's joint.
ACTUATOR_ORDER: tuple[str, ...] = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
#: For actuator ``i``, the qpos index of its joint. Derived from ``m.jnt_qposadr``.
ACTUATOR_TO_QPOS: tuple[int, ...] = (10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15)
#: For actuator ``i``, the qvel index of its joint.
ACTUATOR_TO_QVEL: tuple[int, ...] = (9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14)
#: For actuator ``i``, the index of its joint within the 12-d joint vector (hip, thigh, calf).
ACTUATOR_LEG_SLOT: tuple[int, ...] = (0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2)

QPOS_BASE = 0  # free joint occupies qpos[0:7]
QVEL_BASE = 0  # free joint occupies qvel[0:6]

LEGS: tuple[str, ...] = ("FR", "FL", "RR", "RL")  # actuator/leg order
FOOT_BODIES: tuple[str, ...] = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
BASE_BODY = "base_link"

#: The model's ``home`` keyframe, verbatim: free joint then 12 hinge joints.
HOME_QPOS = np.array(
    [0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.9, -1.8] * 4, dtype=np.float64
)
#: Standing base height from the ``home`` keyframe. Used to judge a terminated fall.
HOME_BASE_HEIGHT = 0.27

#: Go2 crouch (hip, thigh, calf) per leg, in actuator order. Equals the keyframe's joints.
DEFAULT_JOINT_POS = np.tile(np.array([0.0, 0.9, -1.8]), 4).astype(np.float64)

#: PD gains used by the surveyed setups and by the repo's existing ``go2_description.xml``.
KP = 60.0
KV = 3.0
#: Peak torque per actuator (hip/thigh, then calf), from the model's ``ctrlrange``.
#: Not ``forcerange``: the model's ``actuator_forcerange`` is ``[0, 0]`` for all 12 actuators
#: with ``actuator_forcelimited == False``, and the real limits live in ``actuator_ctrlrange``
#: (``[-23.7, 23.7]`` for hip/abduction, ``[-45.43, 45.43]`` for the knee, all
#: ``ctrllimited == True``). The values below are right; only the field they were documented as
#: coming from was wrong -- see INTEGRATION_NOTES finding 43 (B3).
TORQUE_LIMIT_HIP_THIGH = 23.7
TORQUE_LIMIT_CALF = 45.43

#: Leg link lengths (metres), from the model's body geometry.
#: Used by inline trot policies for analytic IK.
HIP_OFFSET = 0.0955  # abduction offset (hip-to-hip half-width)
THIGH = 0.213        # thigh link length
CALF = 0.213         # calf link length

#: Froude-number gait-regime thresholds (survey §Rank 1), leg length 0.41 m.
GAIT_REGIME_THRESHOLDS_MS = (0.7, 1.7)
LEG_LENGTH_M = 0.41

#: Speed bins for the evaluation sweep, m/s.
SPEED_BINS = (0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.1)

GRAVITY = np.array([0.0, 0.0, -9.81])


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomizationSpec:
    """Domain-randomization ranges, and -- once sampled -- one concrete draw.

    The same type serves both roles, which is deliberate but needs stating. As
    ``EnvSpec.randomization`` it is a **template**: every tuple is an inclusive ``(low, high)``
    range. ``Randomizer.sample`` reduces a template to a **draw**, collapsing each range to a
    point ``(x, x)`` and each latency range to a scalar step count. The env samples once per
    episode and applies the draw, so a draw is always self-describing and reproducible from
    its own values.
    """

    friction: tuple[float, float] = (0.6, 1.4)
    mass_scale: tuple[float, float] = (0.9, 1.1)
    motor_strength: tuple[float, float] = (0.9, 1.1)
    payload_kg: tuple[float, float] = (0.0, 0.0)
    latency_steps: tuple[int, int] = (0, 0)
    obs_noise: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "friction": self.friction,
            "mass_scale": self.mass_scale,
            "motor_strength": self.motor_strength,
            "payload_kg": self.payload_kg,
            "latency_steps": self.latency_steps,
            "obs_noise": self.obs_noise,
        }


@dataclass(frozen=True)
class EnvSpec:
    """One simulation configuration. Two envs with equal specs must behave identically."""

    model_path: str = str(GO2_XML)
    control_hz: int = CONTROL_HZ
    episode_s: float = EPISODE_S
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)
    seed: int = 0
    terrain: str = "flat"  # flat | slope | rough
    terminate_base_height: float = 0.16
    terminate_max_roll_pitch: float = 1.0  # rad

    def __post_init__(self) -> None:
        # Config files carry repo-relative paths so they stay portable; resolve here so
        # every consumer gets an absolute path regardless of the working directory.
        p = self.model_path
        if not os.path.isabs(p):
            object.__setattr__(self, "model_path", os.path.join(REPO_ROOT, p))


@dataclass(frozen=True)
class Command:
    """Body-frame velocity command."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.yaw_rate], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def gait_regime(self) -> str:
        """Froude-number regime label used by the expert dataset and the metrics."""
        s = self.speed
        if s < GAIT_REGIME_THRESHOLDS_MS[0]:
            return "walk"
        if s < GAIT_REGIME_THRESHOLDS_MS[1]:
            return "trot"
        return "bound"


# --------------------------------------------------------------------------------------
# Transitions and rollouts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Transition:
    """Everything a reward function may look at for a single control step.

    Deliberately rich: all three arms consume exactly this, so no arm needs a private
    channel into the environment. All arrays are float64 copies owned by the transition.
    """

    t: int
    qpos: np.ndarray
    qvel: np.ndarray
    action: np.ndarray  # 12-d raw policy action
    prev_action: np.ndarray
    command: Command
    base_pos: np.ndarray  # (3,) world position of base body
    base_quat: np.ndarray  # (4,) wxyz
    base_lin_vel: np.ndarray  # (3,) world frame
    base_ang_vel: np.ndarray  # (3,) body frame
    gravity_proj: np.ndarray  # (3,) gravity in body frame
    joint_pos: np.ndarray  # (12,) actuator order
    joint_vel: np.ndarray  # (12,) actuator order
    foot_pos_world: np.ndarray  # (4, 3) foot body positions
    foot_contact: np.ndarray  # (4,) bool
    joint_torque: np.ndarray  # (12,) applied torque, already clipped at the actuator limit
    default_joint_pos: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    #: Torque the PD law *asked* for, before the actuator limit clipped it. Applied torque
    #: saturates like a real motor, so a reward built only on ``joint_torque`` can never
    #: penalise saturation -- and the surveyed temporal-logic recipe lists torque limits as
    #: one of the formula families the LLM proposes. Without this, Arm B's torque formulas
    #: would be structurally inert, which would bias the study against the LLM arms.
    requested_torque: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class Rollout:
    """A recorded episode, sufficient to recompute any metric or reward offline."""

    transition: Transition  # final state
    qpos: np.ndarray  # (T, nq)
    qvel: np.ndarray  # (T, nv)
    action: np.ndarray  # (T, 12)
    reward: np.ndarray  # (T,)
    command: Command
    terminated: bool
    truncated: bool
    meta: dict[str, Any] = field(default_factory=dict)
    #: Per-step foot contact, ``(T, 4)`` bool in ``FOOT_BODIES`` order. Typed rather than
    #: left in ``meta`` because gait labelling -- the metric that detects a policy hacking a
    #: speed-only reward by bounding at every speed -- reads it on every episode.
    foot_contact_hist: np.ndarray | None = None
    #: Per-step applied torque ``(T, 12)``. Cost of transport is ``sum(tau * qvel)``, so
    #: without this every real rollout reports ``nan`` CoT -- the metric cannot be
    #: reconstructed from qpos/qvel/action without measuring something different.
    joint_torque_hist: np.ndarray | None = None
    #: Per-step *requested* torque ``(T, 12)``, before the actuator limit clipped it. Needed
    #: for an episode-wide saturation fraction; the terminal ``Transition`` alone only shows
    #: the last step.
    requested_torque_hist: np.ndarray | None = None
    #: Control rate the recording was made at, so a metric never has to assume one.
    control_hz: int = CONTROL_HZ

    @property
    def length(self) -> int:
        return int(self.reward.shape[0])


# --------------------------------------------------------------------------------------
# Arm B -- temporal-logic specifications
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class STLFormula:
    """A parsed probabilistic-temporal-logic node.

    ``op`` is one of ``"atom"`` (a predicate over named signals, with symbolic or grounded
    threshold), ``"G"`` (always, over ``args["window"]``), ``"F"`` (eventually),
    ``"and"``, ``"or"``. ``args`` carries ``window``, ``var``, ``cmp``, ``threshold``,
    ``threshold_symbol``, ``scale``.
    """

    op: str
    children: tuple["STLFormula", ...] = ()
    args: dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return self.op == "atom"

    def walk(self) -> Iterator["STLFormula"]:
        """Depth-first traversal, self first. A generator -- callers may iterate once."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class SpecBundle:
    """An LLM-proposed specification after grounding, ready to be turned into a reward.

    ``weights`` keys are ``"safe"``, ``"track"`` and ``"pattern"``. ``horizon`` is the
    trailing-window length H -- the survey's central negative result lives at H=30, so it
    is always carried explicitly rather than defaulted.
    """

    groups: dict[str, STLFormula]
    thresholds: dict[str, float]
    weights: dict[str, float]
    horizon: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def tau(self) -> float:
        """Soft-min temperature."""
        return float(self.provenance.get("tau", 0.1))


# --------------------------------------------------------------------------------------
# Arm C -- reference motions
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ReferenceMotion:
    """A kinematic reference produced by forward kinematics only (no physics).

    Shapes are ``(T, ...)`` at the control rate. ``qpos_ref`` is the full renderable
    configuration; ``meta["untrackable"]`` flags physically implausible references.
    """

    phase: np.ndarray  # (T,) in [0, 1)
    base_pos: np.ndarray  # (T, 3)
    base_quat: np.ndarray  # (T, 4) wxyz
    foot_pos: np.ndarray  # (T, 4, 3) body-frame target foot positions
    qpos_ref: np.ndarray  # (T, nq)
    joint_pos_ref: np.ndarray  # (T, 12) actuator order
    joint_vel_ref: np.ndarray  # (T, 12)
    skill: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.phase.shape[0])


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsResult:
    """Outcome of one evaluated episode. ``extra`` carries arm- and metric-specific detail.

    ``extra`` must not reuse a typed field's key -- ``as_row`` raises rather than let one
    silently shadow the other.
    """

    #: Keys ``as_row`` emits from typed fields; ``extra`` may not collide with them.
    _TYPED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "survival", "command_success", "vel_mae", "cot", "gait_label", "gait_match",
            "vx", "vy", "yaw_rate", "seed",
        }
    )

    survival: bool
    command_success: bool
    vel_mae: float
    cot: float
    gait_label: str
    gait_match: bool
    command: Command
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        colliding = set(self.extra) & set(self._TYPED_FIELDS)
        if colliding:
            # Silent shadowing is the failure mode: an ``extra`` key named ``cot`` would
            # replace the scalar cost of transport in every results table.
            raise ValueError(
                f"extra keys collide with typed MetricsResult fields: {sorted(colliding)}"
            )
        return {
            "survival": self.survival,
            "command_success": self.command_success,
            "vel_mae": self.vel_mae,
            "cot": self.cot,
            "gait_label": self.gait_label,
            "gait_match": self.gait_match,
            "vx": self.command.vx,
            "vy": self.command.vy,
            "yaw_rate": self.command.yaw_rate,
            "seed": self.seed,
            **self.extra,
        }


@dataclass(frozen=True)
class StudyConfig:
    """One cell of the study: an arm, a seed, a budget."""

    arm: str  # "A" | "B" | "C"
    seed: int
    env_spec: EnvSpec
    total_steps: int
    reward_cfg: dict[str, Any] = field(default_factory=dict)
    tag: str = ""

    @property
    def run_id(self) -> str:
        return self.tag or f"arm{self.arm}_seed{self.seed}"


# --------------------------------------------------------------------------------------
# Protocols -- the seams between modules owned by different agents
# --------------------------------------------------------------------------------------


@runtime_checkable
class RewardFn(Protocol):
    """An arm's reward.

    ``reset`` is called at the start of each episode so a stateful reward (Arm B's trailing
    window) can clear its buffer. ``__call__`` returns the scalar reward plus per-term
    components for logging. Implementations must be pure with respect to the transition.
    """

    def reset(self) -> None: ...

    def __call__(self, tr: Transition) -> tuple[float, dict[str, float]]: ...


@runtime_checkable
class PolicyFn(Protocol):
    """A hand-written controller (the expert) or a trained policy behind one interface."""

    def __call__(self, obs: np.ndarray, t: int, command: Command) -> np.ndarray: ...


@runtime_checkable
class Randomizer(Protocol):
    """Samples and applies domain randomization. Implemented in ``envs/randomization.py``."""

    def sample(self, rng: np.random.Generator) -> RandomizationSpec: ...

    def apply(self, model: Any, data: Any, spec: RandomizationSpec) -> None: ...


@runtime_checkable
class LLMClient(Protocol):
    """Text-in, text-out model access. Implemented in ``llm/client.py``."""

    def complete(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0
    ) -> str: ...


__all__ = [
    "REPO_ROOT", "GO2_XML", "SIM_DT", "CONTROL_HZ", "DECIMATION", "EPISODE_S",
    "MAX_EPISODE_STEPS", "NUM_ACTIONS", "OBS_DIM", "ACTUATOR_ORDER",
    "ACTUATOR_TO_QPOS", "ACTUATOR_TO_QVEL", "ACTUATOR_LEG_SLOT", "QPOS_BASE",
    "QVEL_BASE", "LEGS", "FOOT_BODIES", "BASE_BODY", "HOME_QPOS", "HOME_BASE_HEIGHT",
    "DEFAULT_JOINT_POS", "KP", "KV",
    "TORQUE_LIMIT_HIP_THIGH", "TORQUE_LIMIT_CALF", "GAIT_REGIME_THRESHOLDS_MS",
    "LEG_LENGTH_M", "SPEED_BINS", "GRAVITY",
    "HIP_OFFSET", "THIGH", "CALF",
    "RandomizationSpec", "EnvSpec", "Command", "Transition", "Rollout", "STLFormula",
    "SpecBundle", "ReferenceMotion", "MetricsResult", "StudyConfig",
    "RewardFn", "PolicyFn", "Randomizer", "LLMClient",
]
