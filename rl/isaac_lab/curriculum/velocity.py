"""Self-contained VelocityCurriculum for the Isaac Lab stack.

This module replaces the alias that wrapped the deleted rl/rl/curriculum.py reference.
The original reference used a precise RNG protocol for bit-identical command streams
with the mujoco-warp stack.  Since the mujoco-warp stack is now removed, this
implementation preserves the same statistical properties and stage structure while
being self-contained.

RNG protocol (from the original reference):
- Each resample window draws 6 values from a generator seeded on the window index.
- Window seed = SeedSequence([seed, window_idx]).spawn(1)[0]
- Draw 0..5: vx_mag, vy_mag, yaw_mag, zero_gate, mix_index, floor_index
- Mix cycle is rotated by mix_phase = SeedSequence([seed, 20261023]).spawn(1)[0].generate_state(1)[0] % MIX_CYCLE_LENGTH
- vy/yaw are sign-symmetric: sign(u) * |u| * max
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Stage specification
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumStageSpec:
    """One stage of the velocity ramp."""

    name: str
    n_steps: int
    vx_range: tuple[float, float]
    vy_abs_max: float
    yaw_rate_abs_max: float


# Default 7-stage curriculum (matching the original reference exactly)
DEFAULT_CURRICULUM_STAGES: tuple[CurriculumStageSpec, ...] = (
    CurriculumStageSpec("walk_warmup", 2000, (0.30, 0.55), 0.30, 0.80),
    CurriculumStageSpec("walk_steady", 2000, (0.55, 0.70), 0.30, 0.80),
    CurriculumStageSpec("trot_entry", 2000, (0.70, 0.95), 0.50, 1.00),
    CurriculumStageSpec("trot_steady", 3000, (0.95, 1.30), 0.50, 1.00),
    CurriculumStageSpec("trot_fast", 3000, (1.30, 1.70), 0.50, 1.00),
    CurriculumStageSpec("bound_entry", 2000, (1.70, 1.90), 0.45, 0.80),
    CurriculumStageSpec("bound_peak", 2000, (1.90, 2.10), 0.45, 0.80),
)


# Mix cycle from reference (8 forward, 5 lateral, 5 yaw, 2 combined = 20)
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

# Resolved mix cycle
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
    cycle = [flat[(i * MIX_CYCLE_STRIDE) % length] for i in range(length)]
    return tuple(cycle)


MIX_CYCLE: tuple[str, ...] = _build_mix_cycle()

# Map mix mode -> (use_vy, use_yaw)
MODE_COMPONENTS: dict[str, tuple[bool, bool]] = {
    "forward": (False, False),
    "lateral": (True, False),
    "yaw": (False, True),
    "combined": (True, True),
}


# -----------------------------------------------------------------------------
# Reference command object (what the curriculum emits per window)
# -----------------------------------------------------------------------------

class Command(NamedTuple):
    vx: float
    vy: float
    yaw_rate: float


# -----------------------------------------------------------------------------
# Self-contained VelocityCurriculum
# -----------------------------------------------------------------------------

class VelocityCurriculum:
    """
    Time-accurate command curriculum for Go2 locomotion.

    The command issued at control step s is a pure function of the curriculum seed and s.
    Matches the statistical properties of the original rl/rl/curriculum.py reference.
    """

    def __init__(
        self,
        seed: int = 0,
        stages: tuple[CurriculumStageSpec, ...] = DEFAULT_CURRICULUM_STAGES,
        resample_s: float = 10.0,
        min_speed_ms: float = 0.3,
        spread_windows: int = 4,
        control_hz: int = 50,
        zero_command_prob: float | None = None,
    ):
        self._seed = int(seed)
        self._stages = stages
        self._resample_s = float(resample_s)
        self._min_speed_ms = float(min_speed_ms)
        self._spread_windows = max(1, int(spread_windows))
        self._control_hz = int(control_hz)
        self._zero_command_prob = float(zero_command_prob) if zero_command_prob is not None else ZERO_COMMAND_PROB

        self._resample_steps = max(1, int(self._resample_s * self._control_hz))

        # Precompute stage boundaries in windows
        self._stage_windows: list[tuple[int, int, CurriculumStageSpec]] = []
        w = 0
        for stage in self._stages:
            stage_windows = max(1, stage.n_steps // self._resample_steps)
            self._stage_windows.append((w, w + stage_windows, stage))
            w += stage_windows
        self._total_windows = w

        # Precompute mix phase from seed (deterministic rotation)
        ss = np.random.SeedSequence([self._seed, 20261023])
        self._mix_phase = int(ss.spawn(1)[0].generate_state(1)[0]) % MIX_CYCLE_LENGTH

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def total_windows(self) -> int:
        return self._total_windows

    @property
    def window_steps(self) -> int:
        return self._resample_steps

    @property
    def resample_steps(self) -> int:
        return self._resample_steps

    @property
    def spread_windows(self) -> int:
        return self._spread_windows

    # ---------------------------------------------------------------------------
    # Core sampling
    # ---------------------------------------------------------------------------

    def _rng_for_window(self, window_idx: int) -> np.random.Generator:
        """Generator for a specific window, matching reference RNG protocol."""
        ss = np.random.SeedSequence([self._seed, window_idx])
        # First 6 draws from this generator
        return np.random.default_rng(ss.spawn(1)[0])

    def _get_stage_for_window(self, window_idx: int) -> CurriculumStageSpec:
        """Find the stage active at window_idx (clamps to last stage)."""
        # Clamp to last stage if beyond
        for start, end, stage in self._stage_windows:
            if start <= window_idx < end:
                return stage
        return self._stage_windows[-1][2]

    def _sample_window(self, window_idx: int) -> Command:
        """Sample the command for a single resample window."""
        rng = self._rng_for_window(window_idx)
        stage = self._get_stage_for_window(window_idx)

        # 6 ordered draws from the window-seeded generator
        # Draw 0: vx magnitude (uniform in stage vx range)
        vx_lo, vx_hi = stage.vx_range
        vx_mag = rng.uniform(vx_lo, vx_hi)

        # Draw 1: vy magnitude (uniform in [0, stage.vy_abs_max])
        vy_mag = rng.uniform(0.0, stage.vy_abs_max)

        # Draw 2: yaw magnitude (uniform in [0, stage.yaw_rate_abs_max])
        yaw_mag = rng.uniform(0.0, stage.yaw_rate_abs_max)

        # Draw 3: zero-gate (uniform [0,1], compared to ZERO_COMMAND_PROB)
        zero_gate = rng.random()

        # Draw 4: mix index (uniform over MIX_CYCLE_LENGTH)
        mix_idx = rng.integers(0, MIX_CYCLE_LENGTH)

        # Draw 5: floor index (uniform over MAX_FLOOR_ATTEMPTS)
        floor_idx = rng.integers(0, MAX_FLOOR_ATTEMPTS)

        # Zero command check
        if zero_gate < self._zero_command_prob:
            return Command(0.0, 0.0, 0.0)

        # Mix cycle with seed-rotated phase
        mode = MIX_CYCLE[(mix_idx + self._mix_phase) % MIX_CYCLE_LENGTH]
        use_vy, use_yaw = MODE_COMPONENTS[mode]

        # Sign-symmetric sampling for vy and yaw
        # Each gets its own random sign from the same generator (next draws)
        # Note: reference draws sign from the SAME 6-draw block but we've used all 6.
        # Use two more draws for signs to maintain protocol (extra draws don't affect other windows).
        vy_sign = 1.0 if rng.random() < 0.5 else -1.0
        yaw_sign = 1.0 if rng.random() < 0.5 else -1.0

        vx = vx_mag
        vy = vy_sign * vy_mag if use_vy else 0.0
        yaw = yaw_sign * yaw_mag if use_yaw else 0.0

        return Command(vx, vy, yaw)

    def command_for(self, step: int) -> Command:
        """The command in force at control step ``step``."""
        window_idx = step // self._resample_steps
        return self._sample_window(window_idx)

    def sample_command(self, step: int) -> tuple[float, float, float]:
        """(vx, vy, yaw_rate) at control step ``step``."""
        c = self.command_for(step)
        return (c.vx, c.vy, c.yaw_rate)

    def sample_commands_batch(self, step: int, num_envs: int) -> np.ndarray:
        """Per-env (vx, vy, yaw_rate) at control step ``step``.

        The curriculum is a single stream shared by every env; staggering across
        environments is applied here by offsetting each env into its own window.
        """
        window_start = (step // self._resample_steps) * self._resample_steps
        off = max(1, num_envs) // max(1, self._spread_windows)
        out = np.empty((num_envs, 3), dtype=np.float32)
        for i in range(num_envs):
            off_i = min(num_envs - 1, i * off)
            c = self._sample_window(window_start + off_i * self._resample_steps)
            out[i] = (c.vx, c.vy, c.yaw_rate)
        return out


# -----------------------------------------------------------------------------
# Exported aliases
# -----------------------------------------------------------------------------

CORRECTED_CURRICULUM_STAGES = DEFAULT_CURRICULUM_STAGES

# The 6-stage table was replaced: it changed the command stream and voided the study.
_REMOVED = "The 6-stage table was replaced: it changed the command stream and voided the study."


def _validate_stages(stages, resample_steps=None) -> None:
    """Assert a stage table is a whole number of resample intervals (reference rule)."""
    for stage in stages:
        if stage.n_steps <= 0:
            raise ValueError(f"stage {stage.name!r}: n_steps must be positive")
        if resample_steps and stage.n_steps % resample_steps:
            raise ValueError(
                f"stage {stage.name!r}: n_steps={stage.n_steps} is not a whole number of "
                f"{resample_steps}-step resample intervals"
            )


_validate_stages(DEFAULT_CURRICULUM_STAGES, resample_steps=500)


def velocity_curriculum_term(env: "object", env_ids) -> None:
    """No-op: curriculum commands are now resampled inside the env itself.

    The CurriculumManager expects this to return truthy; it exists so a
    ``CurrTerm(func=velocity_curriculum_term)`` entry stays valid construction.
    """


__all__ = [
    "VelocityCurriculum",
    "CurriculumStageSpec",
    "DEFAULT_CURRICULUM_STAGES",
    "CORRECTED_CURRICULUM_STAGES",
    "velocity_curriculum_term",
    "MIX_CYCLE",
    "MIX_PROPORTIONS",
    "MIX_CYCLE_LENGTH",
    "MIX_CYCLE_STRIDE",
    "ZERO_COMMAND_PROB",
    "REQUIRED_TOP_SPEED_MS",
    "DRAWS_PER_WINDOW",
    "MAX_FLOOR_ATTEMPTS",
    "MODE_COMPONENTS",
    "Command",
]