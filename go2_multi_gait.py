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

    def __init__(self, initial_gait: GaitType = GaitType.TROT):
        self.current_gait = initial_gait
        self.target_gait = initial_gait
        self.profile = GAIT_PRESETS[initial_gait]

        self.current_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.target_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.current_stance_dur = self.profile.stance_duration
        self.current_swing_dur = self.profile.swing_duration

        self.phase_time = 0.0
        self.transition_progress = 1.0  # 1.0 means fully reached target gait
        self.transition_rate = 2.0      # Transition speed (1/s)

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
        """Requests a gait transition to target gait."""
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

        if gait_enum == self.current_gait and self.transition_progress >= 1.0:
            return True

        self.target_gait = gait_enum
        self.target_offsets = np.array(GAIT_PRESETS[gait_enum].phase_offsets, dtype=float)
        self.transition_progress = 0.0
        self.transition_rate = 1.0 / max(0.05, transition_duration)
        return True

    def reset(self):
        self.phase_time = 0.0
        self.transition_progress = 1.0
        self.current_gait = self.target_gait
        self.profile = GAIT_PRESETS[self.current_gait]
        self.current_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.target_offsets = np.array(self.profile.phase_offsets, dtype=float)
        self.current_stance_dur = self.profile.stance_duration
        self.current_swing_dur = self.profile.swing_duration

    def step(self, dt: float, is_moving: bool = True) -> Tuple[List[float], List[float]]:
        """Advances phase clock and smoothly blends parameters during active transitions."""
        if not is_moving:
            return [0.0] * 4, [0.0] * 4

        # Update transition blending if in progress
        if self.transition_progress < 1.0:
            self.transition_progress = min(1.0, self.transition_progress + self.transition_rate * dt)
            alpha = self.transition_progress

            target_prof = GAIT_PRESETS[self.target_gait]
            self.current_stance_dur = (1.0 - alpha) * self.profile.stance_duration + alpha * target_prof.stance_duration
            self.current_swing_dur = (1.0 - alpha) * self.profile.swing_duration + alpha * target_prof.swing_duration

            # Phase offsets wrap circularly in [0, 1)
            diff = (self.target_offsets - self.current_offsets + 0.5) % 1.0 - 0.5
            blended_offsets = (self.current_offsets + alpha * diff) % 1.0

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
            leg_clock = stride_time - offset * stride_period
            while leg_clock < -self.current_swing_dur:
                leg_clock += stride_period
            while leg_clock > self.current_stance_dur:
                leg_clock -= stride_period

            if 0.0 < leg_clock < self.current_stance_dur:
                stance_phases[index] = leg_clock / self.current_stance_dur
            elif -self.current_swing_dur < leg_clock <= 0.0:
                swing_phases[index] = (leg_clock + self.current_swing_dur) / self.current_swing_dur
            elif self.current_stance_dur < leg_clock <= stride_period:
                swing_phases[index] = (leg_clock - self.current_stance_dur) / self.current_swing_dur

        return stance_phases, swing_phases
