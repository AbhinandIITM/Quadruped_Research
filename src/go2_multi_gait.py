"""
go2_multi_gait.py
=================
Unified Multi-Gait Phase Scheduler and In-Flight Transition Engine
for Unitree Go2 Quadruped.

Supports canonical quadruped gaits:
  - TROT: Diagonal leg pairs (FR+RL / FL+RR)
  - PACE: Lateral leg pairs (FR+RR / FL+RL)
  - BOUND: Sagittal leg pairs (FR+FL / RR+RL)
  - PRONK: Synchronized 4-leg spring
  - WALK: 4-beat crawl with >= 3 legs in stance for static stability
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class GaitType(str, Enum):
    TROT = "trot"
    PACE = "pace"
    BOUND = "bound"
    PRONK = "pronk"
    WALK = "walk"


@dataclass(frozen=True)
class GaitProfile:
    name: str
    phase_offsets: Tuple[float, float, float, float]  # [FR, FL, RR, RL] in [0, 1)
    stance_duration: float                           # seconds
    swing_duration: float                            # seconds
    swing_height: float = 0.06                       # meters ground clearance

    @property
    def stride_period(self) -> float:
        return self.stance_duration + self.swing_duration

    @property
    def duty_factor(self) -> float:
        return self.stance_duration / self.stride_period


# Canonical Quadruped Gait Presets
GAIT_PRESETS: Dict[GaitType, GaitProfile] = {
    GaitType.TROT: GaitProfile(
        name="trot",
        phase_offsets=(0.0, 0.5, 0.5, 0.0),
        stance_duration=0.22,
        swing_duration=0.22,
        swing_height=0.06,
    ),
    GaitType.PACE: GaitProfile(
        name="pace",
        phase_offsets=(0.0, 0.5, 0.0, 0.5),
        stance_duration=0.22,
        swing_duration=0.22,
        swing_height=0.06,
    ),
    GaitType.BOUND: GaitProfile(
        name="bound",
        phase_offsets=(0.0, 0.0, 0.5, 0.5),
        stance_duration=0.20,
        swing_duration=0.24,
        swing_height=0.07,
    ),
    GaitType.PRONK: GaitProfile(
        name="pronk",
        phase_offsets=(0.0, 0.0, 0.0, 0.0),
        stance_duration=0.18,
        swing_duration=0.26,
        swing_height=0.08,
    ),
    GaitType.WALK: GaitProfile(
        name="walk",
        phase_offsets=(0.0, 0.5, 0.75, 0.25),
        stance_duration=0.60,
        swing_duration=0.20,
        swing_height=0.05,
    ),
}


class MultiGaitScheduler:
    """Manages continuous phase integration and smooth in-flight gait transitions."""

    def _duty_split_for(self, gait: GaitType) -> Tuple[float, float]:
        """Stance/swing split in force for ``gait``: override if it applies, else preset.

        The override is honoured only for the gait it was scoped to at construction,
        so tuning one gait's duty factor never silently rewrites another's profile.
        """
        prof = GAIT_PRESETS[gait]
        if gait != self._override_gait:
            return prof.stance_duration, prof.swing_duration
        stance = prof.stance_duration if self._override_stance is None else float(self._override_stance)
        swing = prof.swing_duration if self._override_swing is None else float(self._override_swing)
        return stance, swing

    def __init__(
        self,
        initial_gait: GaitType = GaitType.TROT,
        stance_duration: Optional[float] = None,
        swing_duration: Optional[float] = None,
        override_gait: Optional[GaitType] = None,
    ):
        self.current_gait = initial_gait
        self.target_gait = initial_gait
        self.profile = GAIT_PRESETS[initial_gait]
        self.initial_gait = initial_gait  # Store initial for reset

        # Optional duty split override, *scoped to the gait it was configured for*.
        # A caller tuning duty factor tunes the gait it is actually running (TROT
        # for the CHAMP controller); without this the preset wins and the caller's
        # GaitConfig fields are silently dead. The override is applied only while
        # the effective gait matches _override_gait, so a later set_gait() keeps
        # that gait's own profile (WALK's 0.60/0.20, etc.) instead of inheriting
        # the split it was never meant to have.
        self._override_gait = initial_gait if override_gait is None else override_gait
        self._override_stance = stance_duration
        self._override_swing = swing_duration

        # Effective split for the initial gait: the override if it applies here,
        # else the preset.
        eff_stance, eff_swing = self._duty_split_for(initial_gait)

        # Snapshot initial transition source state
        self._init_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self._init_stance_dur = eff_stance
        self._init_swing_dur = eff_swing

        self.current_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.target_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.current_stance_dur = eff_stance
        self.current_swing_dur = eff_swing

        self.phase_time = 0.0
        self.transition_progress = 1.0  # 1.0 means fully reached target gait
        self.transition_rate = 2.0      # Transition speed (1/s)
        # Frozen source state for the in-flight transition: the current *blended*
        # durations/offsets, snapshotted when the transition starts.
        self._transition_source_stance_dur = self.current_stance_dur
        self._transition_source_swing_dur = self.current_swing_dur
        self._transition_source_offsets = self.current_offsets.copy()

        # Live target split for in-flight blending, kept in sync with the override
        # (set_gait refreshes these; both are seeded here so the blending path never
        # has to fall back to getattr defaults).
        self._target_stance_dur = eff_stance
        self._target_swing_dur = eff_swing

    @property
    def stance_duration(self) -> float:
        return float(self.current_stance_dur)

    @property
    def swing_duration(self) -> float:
        return float(self.current_swing_dur)

    @property
    def stride_period(self) -> float:
        return float(self.current_stance_dur + self.current_swing_dur)

    def set_gait(self, target: str | GaitType, transition_duration: float = 0.5) -> bool:
        """Requests a gait transition to target gait.

        Repeated requests for the same target gait while a transition is in progress
        are idempotent and do not restart the transition.
        """
        if isinstance(target, str):
            target_clean = target.strip().lower()
            try:
                gait_enum = GaitType(target_clean)
            except ValueError:
                return False
        else:
            gait_enum = target

        if gait_enum not in GAIT_PRESETS:
            return False

        # Already at target gait and transition complete
        if gait_enum == self.current_gait and self.transition_progress >= 1.0:
            return True

        # Already transitioning to this gait - do not restart
        if gait_enum == self.target_gait and self.transition_progress < 1.0:
            return True

        # New transition requested - freeze the *current blended state* as the
        # source, so interrupting an in-flight transition stays continuous
        # instead of restarting from the last completed gait.
        self.target_gait = gait_enum
        self.target_offsets = np.array(GAIT_PRESETS[gait_enum].phase_offsets, dtype=float)
        # Target split for the gait being transitioned into. _duty_split_for applies
        # the override only when it is scoped to this gait, so a configured trot
        # split survives a redundant set_gait('trot') but WALK keeps its own timing.
        self._target_stance_dur, self._target_swing_dur = self._duty_split_for(gait_enum)
        self._transition_source_offsets = self.current_offsets.copy()
        self._transition_source_stance_dur = float(self.current_stance_dur)
        self._transition_source_swing_dur = float(self.current_swing_dur)
        self.transition_progress = 0.0
        self.transition_rate = 1.0 / max(0.05, transition_duration)
        return True

    def reset(self):
        """Reset to the original initial gait set at construction."""
        self.phase_time = 0.0
        self.transition_progress = 1.0
        self.current_gait = self.initial_gait
        self.profile = GAIT_PRESETS[self.initial_gait]
        self.current_offsets = self._init_offsets.copy()
        self.target_offsets = self._init_offsets.copy()
        self.current_stance_dur = self._init_stance_dur
        self.current_swing_dur = self._init_swing_dur
        self._transition_source_stance_dur = self._init_stance_dur
        self._transition_source_swing_dur = self._init_swing_dur
        self._transition_source_offsets = self._init_offsets.copy()
        self._target_stance_dur = self._init_stance_dur
        self._target_swing_dur = self._init_swing_dur

    def step(self, dt: float, is_moving: bool = True) -> Tuple[List[float], List[float]]:
        """Advances phase clock and smoothly blends parameters during active transitions."""
        if not is_moving:
            return [0.0] * 4, [0.0] * 4

        # Update transition blending if in progress
        if self.transition_progress < 1.0:
            self.transition_progress = min(1.0, self.transition_progress + self.transition_rate * dt)
            alpha = self.transition_progress

            target_prof = GAIT_PRESETS[self.target_gait]

            # Blend stance/swing durations from source to target
            # Use the frozen source snapshot to prevent compounding; it holds the
            # blended values in force when the transition (re)started, so an
            # interrupted transition continues from where it actually was.
            src_stance = self._transition_source_stance_dur
            src_swing = self._transition_source_swing_dur
            tgt_stance = self._target_stance_dur
            tgt_swing = self._target_swing_dur

            self.current_stance_dur = (1.0 - alpha) * src_stance + alpha * tgt_stance
            self.current_swing_dur = (1.0 - alpha) * src_swing + alpha * tgt_swing

            # Phase offsets blend from frozen source snapshot
            src_offsets = self._transition_source_offsets
            diff = (self.target_offsets - src_offsets + 0.5) % 1.0 - 0.5
            blended_offsets = (src_offsets + alpha * diff) % 1.0

            if self.transition_progress >= 1.0:
                self.current_gait = self.target_gait
                self.profile = target_prof
                self.current_offsets = self.target_offsets.copy()
            else:
                self.current_offsets = blended_offsets

        stride_period = self.current_stance_dur + self.current_swing_dur
        self.phase_time += dt
        stride_time = self.phase_time % stride_period

        stance_phases = [0.0] * 4
        swing_phases = [0.0] * 4

        for index, offset in enumerate(self.current_offsets):
            # Wrap the leg clock into (-swing_duration, stance_duration] so every
            # sample falls in exactly one of the two branches below: swing covers
            # (-swing, 0] and stance covers (0, stance]. Without this, the clock
            # could land exactly on the wrap point -swing, where the old strict
            # branch conditions all evaluated False and both phases came out 0.0
            # (a "dead tick" that snaps a foot back to the nominal centre).
            #
            # The wrap point leg_clock == -swing is deliberately mapped onto the
            # end of stance (stance_phase == 1.0): the planner renders that as the
            # same rear extreme (x = -L/2) as the first swing tick, so the output
            # stays continuous. Mapping it into swing instead would give
            # swing_phase == 0.0 == stance_phase, which is exactly the planner's
            # degenerate (0, 0) snap branch. Do not "simplify" this back.
            leg_clock = (stride_time - offset * stride_period) % stride_period
            if leg_clock > self.current_stance_dur:
                leg_clock -= stride_period

            if leg_clock > 0.0:
                stance_phases[index] = leg_clock / self.current_stance_dur
            else:
                swing_phases[index] = (leg_clock + self.current_swing_dur) / self.current_swing_dur

        return stance_phases, swing_phases
