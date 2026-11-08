"""RewardManager wiring for Isaac Lab Go2 environment.

Wires all MDP reward terms from rl/isaac_lab/mdp/__init__.py with
stage-specific weights matching rl/gpu/env.py::_STAGE_CONFIGS exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import RewardTermCfg as RewTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from rl.isaac_lab.mdp import (
    track_lin_vel_xy_exp,
    track_ang_vel_z_exp,
    lin_vel_z_l2,
    ang_vel_xy_l2,
    base_height_l2,
    flat_orientation_l2,
    joint_torques_l2,
    joint_acc_l2,
    action_rate_l2,
    joint_deviation_l2,
    feet_air_time,
    forward_progress,
)


# -----------------------------------------------------------------------------
# Stage reward configurations (mapping from rl/gpu/env.py::_STAGE_CONFIGS)
# Keys are MDP function names from rl/isaac_lab/mdp/__init__.py
# -----------------------------------------------------------------------------

STAGE_REWARD_WEIGHTS = {
    "stand": {
        "track_lin_vel_xy_exp": 0.0,
        "track_ang_vel_z_exp": 0.0,
        "joint_deviation_l2": -2.0,      # posture
        "flat_orientation_l2": -2.0,      # orientation
        "joint_acc_l2": -0.02,            # smoothness
        "action_rate_l2": -0.01,          # action_rate
        "joint_torques_l2": -0.1,         # torque_limit
        "base_height_l2": -2.0,           # base_height (target 0.27)
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
    "balance": {
        "track_lin_vel_xy_exp": 0.0,
        "track_ang_vel_z_exp": 0.0,
        "joint_deviation_l2": -2.0,
        "flat_orientation_l2": -2.0,
        "joint_acc_l2": -0.02,
        "action_rate_l2": -0.01,
        "joint_torques_l2": -0.1,
        "base_height_l2": -2.0,
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
    "transition": {
        "track_lin_vel_xy_exp": 0.1,
        "track_ang_vel_z_exp": 0.05,
        "joint_deviation_l2": -2.0,
        "flat_orientation_l2": -1.0,
        "joint_acc_l2": -0.02,
        "action_rate_l2": -0.01,
        "joint_torques_l2": -0.1,
        "base_height_l2": -1.0,
        "feet_air_time": 1.0,
        "forward_progress": 0.5,
    },
    "walk": {
        "track_lin_vel_xy_exp": 1.0,
        "track_ang_vel_z_exp": 0.5,
        "joint_deviation_l2": -0.2,
        "flat_orientation_l2": -0.3,
        "joint_acc_l2": -0.02,
        "action_rate_l2": -0.01,
        "joint_torques_l2": -0.1,
        "base_height_l2": -0.2,
        "feet_air_time": 0.0,
        "forward_progress": 0.0,
    },
}


# Shared parameters for tracking terms (from rl/rewards/heuristic.py)
TRACK_LIN_VEL_PARAMS = {
    "command_name": "base_velocity",
    "std": 0.25,  # SIGMA_LIN_VEL
}

TRACK_ANG_VEL_PARAMS = {
    "command_name": "base_velocity",
    "std": 0.25,  # SIGMA_ANG_VEL
}

BASE_HEIGHT_PARAMS = {
    "target_height": 0.27,  # BASE_HEIGHT_TARGET_DEFAULT
    "sensor_cfg": "robot",  # SceneEntityCfg("robot")
}


def build_reward_cfg(stage: str) -> dict[str, RewTerm]:
    """Build reward configuration dict for a given stage.

    Returns dict mapping term name -> RewardTermCfg for Isaac Lab RewardManager.
    """
    if stage not in STAGE_REWARD_WEIGHTS:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(STAGE_REWARD_WEIGHTS.keys())}")

    weights = STAGE_REWARD_WEIGHTS[stage]
    reward_cfg = {}

    # Tracking terms (exponential kernels, positive weights)
    if weights.get("track_lin_vel_xy_exp", 0) != 0:
        reward_cfg["track_lin_vel_xy"] = RewTerm(
            func=track_lin_vel_xy_exp,
            weight=weights["track_lin_vel_xy_exp"],
            params=TRACK_LIN_VEL_PARAMS,
        )

    if weights.get("track_ang_vel_z_exp", 0) != 0:
        reward_cfg["track_ang_vel_z"] = RewTerm(
            func=track_ang_vel_z_exp,
            weight=weights["track_ang_vel_z_exp"],
            params=TRACK_ANG_VEL_PARAMS,
        )

    # Penalty terms (L2, negative weights)
    penalty_terms = {
        "lin_vel_z": (lin_vel_z_l2, {}),
        "ang_vel_xy": (ang_vel_xy_l2, {}),
        "joint_deviation": (joint_deviation_l2, {}),
        "flat_orientation": (flat_orientation_l2, {}),
        "joint_torques": (joint_torques_l2, {}),
        "joint_acc": (joint_acc_l2, {}),
        "action_rate": (action_rate_l2, {}),
        "base_height": (base_height_l2, BASE_HEIGHT_PARAMS),
    }

    for term_name, (func, params) in penalty_terms.items():
        weight = weights.get(term_name, 0)
        if weight != 0:
            reward_cfg[term_name] = RewTerm(
                func=func,
                weight=weight,
                params=params,
            )

    # Special terms with custom params
    if weights.get("feet_air_time", 0) != 0:
        reward_cfg["feet_air_time"] = RewTerm(
            func=feet_air_time,
            weight=weights["feet_air_time"],
            params={
                "command_name": "base_velocity",
                "threshold": 0.5,  # minimum command speed for air time reward
                "sensor_cfg": "robot",
            },
        )

    if weights.get("forward_progress", 0) != 0:
        reward_cfg["forward_progress"] = RewTerm(
            func=forward_progress,
            weight=weights["forward_progress"],
            params={
                "command_name": "base_velocity",
                "sensor_cfg": "robot",
            },
        )

    return reward_cfg


def get_reward_weights(stage: str) -> dict[str, float]:
    """Get flat dict of reward weights for a stage (for logging/debugging)."""
    return STAGE_REWARD_WEIGHTS.get(stage, {}).copy()


__all__ = [
    "STAGE_REWARD_WEIGHTS",
    "TRACK_LIN_VEL_PARAMS",
    "TRACK_ANG_VEL_PARAMS",
    "BASE_HEIGHT_PARAMS",
    "build_reward_cfg",
    "get_reward_weights",
]
