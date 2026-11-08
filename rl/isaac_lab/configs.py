"""Stage configurations for the Isaac Lab Go2 migration.

Mirrors ``rl/gpu/env.py::_STAGE_CONFIGS`` exactly, including spawn heights,
base-height targets, reward-weight dictionaries, and curriculum flags. Stage
selection is the only thing that changes the environment's dynamics between
Stand -> Balance -> Transition -> Walk.

The configs live in two layers:

- :class:`IsaacStageConfig` (plain dataclass) holds the raw stage values in a
  form consumable from both Python and a Hydra/omegaconf layer.
- The ``@configclass`` Isaac Lab wrappers turn the stage into a
  ManagerBasedRLEnvCfg or DirectRLEnvCfg, with reward/observation/event terms
  wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from rl.contracts import Command  # noqa: E402


# -----------------------------------------------------------------------------
# Reward term name mapping: GPU stack keys -> Isaac Lab MDP term keys
# -----------------------------------------------------------------------------
# GPU: track_lin_vel, track_ang_vel, posture, orientation, smoothness, action_rate,
#      torque_limit, base_height, feet_air_time, forward_progress
# Isaac Lab: track_lin_vel_xy_exp, track_ang_vel_z_exp, joint_deviation_l2, flat_orientation_l2,
#            joint_acc_l2, action_rate_l2, joint_torques_l2, base_height_l2, feet_air_time, forward_progress

REWARD_KEY_MAP: Dict[str, str] = {
    "track_lin_vel": "track_lin_vel_xy_exp",
    "track_ang_vel": "track_ang_vel_z_exp",
    "posture": "joint_deviation_l2",
    "orientation": "flat_orientation_l2",
    "smoothness": "joint_acc_l2",
    "action_rate": "action_rate_l2",
    "torque_limit": "joint_torques_l2",
    "base_height": "base_height_l2",
    "feet_air_time": "feet_air_time",
    "forward_progress": "forward_progress",
}

# Shared sigma values (from rl/rewards/heuristic.py)
SIGMA_LIN_VEL = 0.25
SIGMA_ANG_VEL = 0.25

# Shared base height targets
BASE_HEIGHT_TARGET_DEFAULT = 0.3
AIR_TIME_TARGET_DUTY = 0.5

# Posture weights (from rl/rewards/heuristic.py)
# Pattern: [hip, thigh, calf] * 4 legs
_HIP_POSTURE_WEIGHT = 0.5
_SAGITTAL_POSTURE_WEIGHT = 1.0
_POSTURE_WEIGHT = [_HIP_POSTURE_WEIGHT, _SAGITTAL_POSTURE_WEIGHT, _SAGITTAL_POSTURE_WEIGHT] * 4


# -----------------------------------------------------------------------------
# Stage configuration dataclass
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class IsaacStageConfig:
    """Raw stage configuration consumed by @configclass wrappers.

    All values mirror rl/gpu/env.py::_STAGE_CONFIGS exactly.
    """

    stage_name: str
    spawn_height: float
    base_height_target: float
    use_curriculum: bool
    reward_weights: Dict[str, float]  # GPU stack keys; mapped via REWARD_KEY_MAP
    curriculum_cycle_steps: int = 2_000_000
    curriculum_resample_s: float = 10.0
    curriculum_min_speed: float = 0.3
    curriculum_spread_windows: int = 4

    @property
    def isaac_reward_weights(self) -> Dict[str, float]:
        """Weights with Isaac Lab MDP term keys."""
        return {
            REWARD_KEY_MAP[k]: v for k, v in self.reward_weights.items() if k in REWARD_KEY_MAP
        }

    @classmethod
    def from_gpu_stage(cls, stage: str) -> "IsaacStageConfig":
        """Get stage config by GPU stage name."""
        return get_stage_config(stage)


# -----------------------------------------------------------------------------
# Stage instances matching _STAGE_CONFIGS exactly
# -----------------------------------------------------------------------------

STAGE_CONFIGS: Dict[str, IsaacStageConfig] = {
    "stand": IsaacStageConfig(
        stage_name="stand",
        spawn_height=0.4,
        base_height_target=0.27,
        use_curriculum=False,
        reward_weights={
            "track_lin_vel": 0.0,
            "track_ang_vel": 0.0,
            "posture": 2.0,
            "orientation": 2.0,
            "smoothness": 0.02,
            "action_rate": 0.01,
            "torque_limit": 0.1,
            "base_height": 2.0,
            "feet_air_time": 0.0,
            "forward_progress": 0.0,
        },
    ),
    "balance": IsaacStageConfig(
        stage_name="balance",
        spawn_height=0.4,
        base_height_target=0.27,
        use_curriculum=False,
        reward_weights={
            "track_lin_vel": 0.0,
            "track_ang_vel": 0.0,
            "posture": 2.0,
            "orientation": 2.0,
            "smoothness": 0.02,
            "action_rate": 0.01,
            "torque_limit": 0.1,
            "base_height": 2.0,
            "feet_air_time": 0.0,
            "forward_progress": 0.0,
        },
    ),
    "transition": IsaacStageConfig(
        stage_name="transition",
        spawn_height=0.27,
        base_height_target=0.3,
        use_curriculum=True,
        reward_weights={
            "track_lin_vel": 0.1,
            "track_ang_vel": 0.05,
            "posture": 2.0,
            "orientation": 1.0,
            "smoothness": 0.02,
            "action_rate": 0.01,
            "torque_limit": 0.1,
            "base_height": 1.0,
            "feet_air_time": 1.0,
            "forward_progress": 0.5,
        },
    ),
    "walk": IsaacStageConfig(
        stage_name="walk",
        spawn_height=0.27,
        base_height_target=0.3,
        use_curriculum=True,
        reward_weights={
            "track_lin_vel": 1.0,
            "track_ang_vel": 0.5,
            "posture": 0.2,
            "orientation": 0.3,
            "smoothness": 0.02,
            "action_rate": 0.01,
            "torque_limit": 0.1,
            "base_height": 0.2,
            "feet_air_time": 0.0,
            "forward_progress": 0.0,
        },
    ),
}


def get_stage_config(stage: str) -> IsaacStageConfig:
    """Get stage config by name, with validation."""
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(STAGE_CONFIGS.keys())}")
    return STAGE_CONFIGS[stage]


# -----------------------------------------------------------------------------
# Curriculum stage specifications (mirroring rl/rl/curriculum.py)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumStageSpec:
    """One stage of the velocity ramp, matching rl/rl/curriculum.py::StageSpec."""

    name: str
    n_steps: int
    vx_range: tuple[float, float]
    vy_abs_max: float
    yaw_rate_abs_max: float


# Default stages from rl/rl/curriculum.py::DEFAULT_STAGES
DEFAULT_CURRICULUM_STAGES: tuple[CurriculumStageSpec, ...] = (
    CurriculumStageSpec("walk_warmup", 2000, (0.30, 0.55), 0.30, 0.80),
    CurriculumStageSpec("walk_steady", 2000, (0.55, 0.70), 0.30, 0.80),
    CurriculumStageSpec("trot_entry", 2000, (0.70, 0.95), 0.50, 1.00),
    CurriculumStageSpec("trot_steady", 3000, (0.95, 1.30), 0.50, 1.00),
    CurriculumStageSpec("trot_fast", 3000, (1.30, 1.70), 0.50, 1.00),
    CurriculumStageSpec("bound_entry", 2000, (1.70, 1.90), 0.45, 0.80),
    CurriculumStageSpec("bound_peak", 2000, (1.90, 2.10), 0.45, 0.80),
)

# Mix cycle from rl/rl/curriculum.py
MIX_PROPORTIONS: tuple[tuple[str, float], ...] = (
    ("forward", 0.40),
    ("lateral", 0.25),
    ("yaw", 0.25),
    ("combined", 0.10),
)

MIX_CYCLE_LENGTH: int = 20
MIX_CYCLE_STRIDE: int = 7
ZERO_COMMAND_PROB: float = 0.25
REQUIRED_TOP_SPEED_MS: float = 1.9
DRAWS_PER_WINDOW: int = 6
MAX_FLOOR_ATTEMPTS: int = 32

# Resolved mix cycle (8 forward, 5 lateral, 5 yaw, 2 combined)
import numpy as np


def _build_mix_cycle() -> tuple[str, ...]:
    counts = {}
    for name, prop in MIX_PROPORTIONS:
        exact = prop * MIX_CYCLE_LENGTH
        if not np.isclose(exact, round(exact)):
            raise ValueError(f"mix proportion {name}={prop} not whole number in {MIX_CYCLE_LENGTH}-window cycle")
        counts[name] = int(round(exact))
    flat = [name for name, _ in MIX_PROPORTIONS for _ in range(counts[name])]
    length = len(flat)
    if length != MIX_CYCLE_LENGTH or np.gcd(MIX_CYCLE_STRIDE, length) != 1:
        raise ValueError("mix cycle length mismatch or not coprime to stride")
    return tuple(flat[(i * MIX_CYCLE_STRIDE) % length] for i in range(length))


MIX_CYCLE: tuple[str, ...] = _build_mix_cycle()

_MODE_COMPONENTS: dict[str, tuple[bool, bool]] = {
    "forward": (False, False),
    "lateral": (True, False),
    "yaw": (False, True),
    "combined": (True, True),
}


def regime_for_speed(speed: float) -> str:
    """Gait regime from speed, delegating to contracts.Command."""
    return Command(vx=float(speed)).gait_regime


# -----------------------------------------------------------------------------
# Domain randomization spec (matching rl/contracts.py RandomizationSpec)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomizationSpec:
    """Domain randomization ranges, matching contracts.RandomizationSpec."""

    friction: tuple[float, float] = (0.6, 1.4)
    mass_scale: tuple[float, float] = (0.9, 1.1)
    motor_strength: tuple[float, float] = (0.9, 1.1)
    payload_kg: tuple[float, float] = (0.0, 0.0)
    latency_steps: tuple[int, int] = (0, 0)
    obs_noise: float = 0.0


DEFAULT_RANDOMIZATION = RandomizationSpec()


# -----------------------------------------------------------------------------
# Constants matching rl/contracts.py and rl/gpu/env.py
# -----------------------------------------------------------------------------

DECIMATION = 10  # 50 Hz control / 500 Hz physics (mjx) -> Isaac Lab uses 4 (200 Hz / 50 Hz)
CONTROL_HZ = 50
PHYSICS_HZ = 200  # Isaac Lab Newton default
MAX_EPISODE_STEPS = 1000  # 20.0s / 0.02s (control dt)


__all__ = [
    "IsaacStageConfig",
    "STAGE_CONFIGS",
    "get_stage_config",
    "CurriculumStageSpec",
    "DEFAULT_CURRICULUM_STAGES",
    "MIX_PROPORTIONS",
    "MIX_CYCLE",
    "MIX_CYCLE_LENGTH",
    "MIX_CYCLE_STRIDE",
    "ZERO_COMMAND_PROB",
    "REQUIRED_TOP_SPEED_MS",
    "DRAWS_PER_WINDOW",
    "MAX_FLOOR_ATTEMPTS",
    "MODE_COMPONENTS",
    "regime_for_speed",
    "RandomizationSpec",
    "DEFAULT_RANDOMIZATION",
    "REWARD_KEY_MAP",
    "SIGMA_LIN_VEL",
    "SIGMA_ANG_VEL",
    "BASE_HEIGHT_TARGET_DEFAULT",
    "AIR_TIME_TARGET_DUTY",
    "_HIP_POSTURE_WEIGHT",
    "_SAGITTAL_POSTURE_WEIGHT",
    "_POSTURE_WEIGHT",
    "DECIMATION",
    "CONTROL_HZ",
    "PHYSICS_HZ",
    "MAX_EPISODE_STEPS",
]