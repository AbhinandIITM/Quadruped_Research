"""LLM-RL Quadruped -- does the LLM write the reward, or the reference motion?

This package implements one controlled study. The same Unitree Go2 policy is trained in
MuJoCo under three reward sources that differ *only* in where the reward comes from:

* arm ``A`` -- a hand-engineered reward (velocity tracking, posture, smoothness, torque),
* arm ``B`` -- a reward compiled from an LLM-written probabilistic-temporal-logic spec,
* arm ``C`` -- a DeepMimic-style imitation reward against an LLM-written reference motion.

All three are compared on survival, command tracking, cost of transport and gait match at a
matched step budget, which is what separates "the LLM can write a reward" from "the LLM can
write the thing that makes a reward work".

Every module in the package depends on :mod:`rl.contracts` and nothing in that
module depends on them. Its symbols are re-exported here, so callers can write
``from rl import EnvSpec, Command, Transition`` without reaching into the
submodule. Importing this package is deliberately cheap: it pulls in ``contracts`` (which
needs only NumPy) and nothing else, so the heavy stacks -- torch, mujoco, gymnasium -- stay
unimported until a module that genuinely needs them is used, and importing the package cannot
fail just because they are absent.
"""

from __future__ import annotations

from . import contracts as _contracts
from .contracts import *  # noqa: F401,F403

__version__ = "0.1.0"

__all__ = [*_contracts.__all__, "__version__"]
