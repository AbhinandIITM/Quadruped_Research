"""Seed derivation for parallel environments.

Minimal module providing only derive_env_seeds for Isaac Lab runner.
"""

from __future__ import annotations

import numpy as np


def derive_env_seeds(base_seed: int, num_envs: int) -> np.ndarray:
    """
    Derive per-environment seeds from a base seed.

    Uses NumPy's SeedSequence.spawn to generate independent streams.
    """
    root = np.random.SeedSequence(int(base_seed) & 0xFFFFFFFF)
    children = root.spawn(num_envs)
    return np.array([int(c.generate_state(1)[0]) for c in children], dtype=np.uint32)


__all__ = ["derive_env_seeds"]