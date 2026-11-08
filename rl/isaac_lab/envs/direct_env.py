"""DirectRLEnv for Go2 matching the mujoco-warp Go2WarpEnv.

DirectRLEnv gives us low-level step control needed to match the exact PD
controller + torque computation + reward terms of the GPU warp stack, while
using Newton's MJWarp physics backend inside Isaac Lab for maximum parity.

Layout matches :class:`Go2WarpEnv` one-for-one:
- obs_buf: (num_envs, 45)  = ang_vel(3) + grav(3) + cmd(3) + dq(12) + qd(12) + prev_action(12)
- actions: (num_envs, 12)  — policy targets clamped to [-1, 1], scaled by ACTION_SCALE
- reward: 10 terms summed with stage weights
- termination: base_height < 0.16, |projected_gravity_xy| > sin(1.0), NaNs, timeout 1000 steps
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.envs.direct_rl_env import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from rl.contracts import (
    SIM_DT,
    CONTROL_HZ,
    DECIMATION,
    MAX_EPISODE_STEPS,
    NUM_ACTIONS,
    OBS_DIM,
    KP as KP_DEFAULT,
    KV as KV_DEFAULT,
    TORQUE_LIMIT_HIP_THIGH,
    TORQUE_LIMIT_CALF,
    DEFAULT_JOINT_POS,
    ACTUATOR_TO_QPOS,
    ACTUATOR_TO_QVEL,
)

from rl.isaac_lab.configs import IsaacStageConfig
from rl.isaac_lab.curriculum.velocity import VelocityCurriculum, CORRECTED_CURRICULUM_STAGES


# Build torque limits array: [hip, thigh, calf] * 4 legs
TORQUE_LIMITS = torch.tensor(
    [TORQUE_LIMIT_HIP_THIGH, TORQUE_LIMIT_HIP_THIGH, TORQUE_LIMIT_CALF] * 4,
    dtype=torch.float32
)

# Isaac Lab joint reordering: contracts ACTUATOR_TO_QPOS/QVEL are DOF indices (range 7..18 / 6..17);
# Isaac Lab's `robot.data.joint_pos/vel` only holds the 12 joints, so subtract the base offset.
_ISAAC_FROM_CONTRACTS = {
    "qpos": [int(x) - 7 for x in ACTUATOR_TO_QPOS],
    "qvel": [int(x) - 6 for x in ACTUATOR_TO_QVEL],
    # tau is written back into joint space: the DOF indices collapse to joint indices uniformly.
    "tau_write": [int(x) - 7 for x in ACTUATOR_TO_QPOS],
}

# Default joint pos in ISAAC joint order (already matches).
ACTUATOR_TO_QPOS_ISAAC = _ISAAC_FROM_CONTRACTS["qpos"]
ACTUATOR_TO_QVEL_ISAAC = _ISAAC_FROM_CONTRACTS["qvel"]
TAU_WRITE_ISAAC = _ISAAC_FROM_CONTRACTS["tau_write"]

# Sanity: all indices must be in [0, 11]
assert all(0 <= i < 12 for i in ACTUATOR_TO_QPOS_ISAAC), ACTUATOR_TO_QPOS_ISAAC
assert all(0 <= i < 12 for i in ACTUATOR_TO_QVEL_ISAAC), ACTUATOR_TO_QVEL_ISAAC
assert all(0 <= i < 12 for i in TAU_WRITE_ISAAC), TAU_WRITE_ISAAC

# Action scale
ACTION_SCALE = 0.25


def _get_go2_cfg(spawn_height: float = 0.27) -> ArticulationCfg:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    # Joint init must match the warp stack's spawn stance: DEFAULT_JOINT_POS
    # (the model's `home` keyframe: 0, 0.9, -1.8 per leg) in Isaac joint order
    # (FL, FR, RL, RR).  The Unitree stock init (hip=+-0.1, thigh=0.8/1.0,
    # calf=-1.5) is a different stance and breaks obs/reward parity at reset.
    default_pos = {
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.9,
        ".*_calf_joint": -1.8,
    }
    cfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UNITREE_GO2_CFG.init_state.replace(
            pos=(0.0, 0.0, spawn_height),
            joint_pos=default_pos,
        ),
    )
    # Actuator effort limits — must match TORQUE_LIMITS exactly
    for actuator_name, actuator in cfg.actuators.items():
        if actuator_name == "base_legs":
            actuator.effort_limit = {
                ".*_hip_joint": 23.7,
                ".*_thigh_joint": 23.7,
                ".*_calf_joint": 45.43,
            }
            actuator.stiffness = KP_DEFAULT  # 60.0
            actuator.damping = KV_DEFAULT    # 3.0
    return cfg


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Scene configuration with Go2 robot."""
    robot: ArticulationCfg = _get_go2_cfg(spawn_height=0.27)
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class Go2DirectRLEnvCfg(DirectRLEnvCfg):
    """Config for Go2 DirectRLEnv matching warp stack.

    Fields mirroring Go2WarpEnvConfig:
    - stage, spawn_height, base_height_target, weights, use_curriculum
    - num_envs, seed, device, physics parameters
    """
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=65,
                nconmax=35,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        ),
    )
    scene: Go2SceneCfg = Go2SceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    # env
    decimation: int = DECIMATION
    episode_length_s: float = MAX_EPISODE_STEPS / CONTROL_HZ  # 20.0
    seed: int | None = None
    observation_space: int = OBS_DIM
    action_space: int = NUM_ACTIONS

    # stage
    stage: str = "stand"
    spawn_height: float = 0.4
    base_height_target: float = 0.27
    base_height_term: float = 0.16
    orientation_term: float = 1.0  # rad limit
    obs_noise: float = 0.0

    # Please note: action clip and torque limits are applied in _apply_action
    # obs noise is added in _get_observations


class Go2DirectRLEnv(DirectRLEnv):
    """DirectRLEnv for Go2 → parity target of mujoco-warp Go2WarpEnv."""

    cfg: Go2DirectRLEnvCfg

    def __init__(self, cfg: Go2DirectRLEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Stage config
        self._stage_cfg = IsaacStageConfig.from_gpu_stage(cfg.stage)
        self._num_envs = cfg.scene.num_envs
        self._device = self.device

        # Action & obs dims
        self._num_actions = NUM_ACTIONS
        self._num_obs = OBS_DIM

        # Robot asset (looked up from scene after construction)
        # Note: _setup_scene() is called by DirectRLEnv.__init__ via _init_sim(),
        # so self._robot is already set here. Do NOT re-assign None.

        # Buffering (CPU/GPU)
        self._obs_buf = torch.zeros(self._num_envs, self._num_obs, device=self._device, dtype=torch.float32)
        self._actions = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float32)
        self._prev_action = torch.zeros_like(self._actions)

        # Use curriculum flag resolved from the stage config (authoritative).
        self._use_curriculum = bool(self._stage_cfg.use_curriculum)
        cfg.use_curriculum = self._use_curriculum

        # Curriculum state (match GPU stack / VelocityCurriculum).
        self._global_step = 0
        CONTROL_HZ_L = CONTROL_HZ
        resample_s = float(self._stage_cfg.curriculum_resample_s)
        spread_windows = int(self._stage_cfg.curriculum_spread_windows)
        self._resample_steps = max(1, int(resample_s * CONTROL_HZ_L))
        cycle_steps = int(self._stage_cfg.curriculum_cycle_steps)

        if self._use_curriculum:
            self._curriculum = VelocityCurriculum(
                seed=cfg.seed if cfg.seed is not None else 0,
                stages=CORRECTED_CURRICULUM_STAGES,
                resample_s=resample_s,
                min_speed_ms=float(self._stage_cfg.curriculum_min_speed),
                spread_windows=spread_windows,
            )
            total_windows = int(self._curriculum.total_windows)
            cycle_steps = max(cycle_steps, total_windows)
            self._cmd_steps_per_window = cycle_steps // total_windows
            self._cmd_step_offset = (
                (spread_windows - 1) * self._cmd_steps_per_window / max(1, self._num_envs)
            )
            self._cmd_total_windows = total_windows
            self._cmd_spread_windows = spread_windows
        else:
            self._curriculum = None
            self._cmd_steps_per_window = 0
            self._cmd_step_offset = 0.0
            self._cmd_total_windows = 0
            self._cmd_spread_windows = 1

        # Commands buffer
        self._commands = torch.zeros(self._num_envs, 3, device=self._device)

    def _setup_scene(self):
        # Robot is already in scene via Go2SceneCfg; just need to look it up
        self._robot = self.scene.articulations["robot"]

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step.

        Clamp actions to [-1, 1] (policy output) then store for PD controller.
        Resample commands at curriculum boundary (matches GPU step() ordering).
        """
        # Resample commands at the start of the step (matches GPU stack)
        if self._use_curriculum and (self._global_step % self._resample_steps == 0):
            self._resample_commands()

        actions = actions.to(self._device).clamp(-1.0, 1.0)
        self._actions = actions

    def _apply_action(self):
        """Compute PD torques from actions and write to articulation.

        Matches gpu/env.py:_physics_substep PD kernel exactly:
        q_des = DEFAULT_JOINT_POS + ACTION_SCALE * action
        tau = KP * (q_des - q) - KV * qd
        tau = clamp(tau, -torque_limit, +torque_limit)
        """
        robot: Articulation = self._robot
        joint_pos = robot.data.joint_pos  # (N, 12) in q order
        joint_vel = robot.data.joint_vel  # (N, 12) in q order

        # Reorder joint_pos -> actuator order for qpos, and qvel for qvel
        # Use ISAAC-specific indices (0..11) instead of contract DOF indices (7..18 / 6..17)
        qpos_actuator = joint_pos[:, ACTUATOR_TO_QPOS_ISAAC]
        qvel_actuator = joint_vel[:, ACTUATOR_TO_QVEL_ISAAC]

        default_q = torch.as_tensor(DEFAULT_JOINT_POS, device=self._device, dtype=torch.float32)

        q_des = default_q + ACTION_SCALE * self._actions
        tau = KP_DEFAULT * (q_des - qpos_actuator) - KV_DEFAULT * qvel_actuator

        # Clamp to torque limits
        torque_limits = torch.as_tensor(TORQUE_LIMITS, device=self._device, dtype=torch.float32)
        tau = torch.clamp(tau, -torque_limits, torque_limits)

        # Reorder tau -> qpos order before writing
        applied_tau = torch.zeros_like(joint_pos)
        for act_idx, qpos_idx in enumerate(TAU_WRITE_ISAAC):
            applied_tau[:, qpos_idx] = tau[:, act_idx]

        robot.set_joint_effort_target(applied_tau)

    def _get_commands(self) -> torch.Tensor:
        """Return velocity commands (vx, vy, yaw_rate)."""
        return self._commands

    def _resample_commands(self):
        """Resample commands from VelocityCurriculum with per-env staggering.

        Mirrors rl/gpu/env.py::_resample_commands exactly.
        - global_step advances each step
        - At global_step % resample_steps == 0: resample
        - Stagger: effective_step = global_step + env_idx * step_offset
        - step_offset = (spread_windows - 1) * cmd_steps_per_window / num_envs
        - window = floor(effective_step / cmd_steps_per_window) % total_windows
        - command = curriculum.sample_command(window * resample_steps)
        """
        if self._curriculum is None:
            self._commands.zero_()
            return

        N = self._num_envs
        dev = self._device
        spw = self._cmd_steps_per_window
        off = self._cmd_step_offset
        tw = self._cmd_total_windows
        rs = self._resample_steps

        # per-env effective step → window index (matches GPU vectorized logic)
        env_idx = torch.arange(N, device=dev, dtype=torch.float32)
        effective = self._global_step + env_idx * off
        windows = torch.div(effective, float(spw), rounding_mode="floor").long() % tw  # (N,)

        # Group envs by their distinct window and issue one sample_command each.
        distinct = torch.unique(windows)
        for w in distinct.tolist():
            # sample_command expects a control step at the window boundary
            step_for_cmd = int(w) * rs
            vx, vy, yaw_rate = self._curriculum.sample_command(step_for_cmd)
            col = torch.as_tensor(
                (vx, vy, yaw_rate), dtype=torch.float32, device=dev
            )
            mask = windows == w
            self._commands[mask] = col

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Build the 45-dim observation matching contract.

        obs = [ang_vel(3), grav_proj(3), cmd(3), dq(12), qd(12), prev_action(12)]

        The order matches gpu/env.py:_refresh_obs exactly.
        """
        robot: Articulation = self._robot

        # base angular velocity body frame
        ang_vel = robot.data.root_ang_vel_b  # (N, 3)

        # projected gravity body frame
        grav_proj = robot.data.projected_gravity_b  # (N, 3)

        # velocity command (N, 3)
        cmd = self._commands

        # joint pos rel to default (actuator order)
        joint_pos = robot.data.joint_pos  # (N, 12) in q order
        qpos_act = joint_pos[:, ACTUATOR_TO_QPOS_ISAAC]
        default_q = torch.as_tensor(DEFAULT_JOINT_POS, device=self._device, dtype=torch.float32)
        dq = qpos_act - default_q

        # joint vel (actuator order)
        joint_vel = robot.data.joint_vel
        qd = joint_vel[:, ACTUATOR_TO_QVEL_ISAAC]

        # prev action
        prev_action = self._prev_action

        obs = torch.cat([
            ang_vel,
            grav_proj,
            cmd,
            dq,
            qd,
            prev_action,
        ], dim=1)  # (N, 45)

        # Add obs noise if configured
        if self.cfg.obs_noise > 0.0:
            obs = obs + self.cfg.obs_noise * torch.randn_like(obs)

        # Update prev_action for next step
        self._prev_action = self._actions.clone()

        # Advance global step at end of step (matches GPU stack step() ordering)
        self._global_step += 1

        self._obs_buf.copy_(obs)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward as weighted sum of 10 terms.

        Mirrors gpu/env.py:_compute_reward exactly.
        Uses DirectRLEnv-compatible reward functions from rl.isaac_lab.rewards_direct.
        """
        from rl.isaac_lab.rewards_direct import (
            track_lin_vel,
            track_ang_vel,
            lin_vel_z_l2,
            ang_vel_xy_l2,
            base_height_l2,
            flat_orientation_l2,
            joint_torques_l2,
            joint_acc_l2,
            action_rate_l2,
            joint_deviation_l2,
            joint_vel_l2,        # NEW: joint velocity L2 for "smoothness"
            normalized_torques_l2,  # NEW: normalized torque for "torque_limit"
            feet_air_time,
            forward_progress,
            STAGE_WEIGHTS,
        )

        stage = self._stage_cfg.stage_name
        weights = STAGE_WEIGHTS[stage]

        terms = {
            "track_lin_vel": track_lin_vel,
            "track_ang_vel": track_ang_vel,
            "posture": joint_deviation_l2,
            "orientation": flat_orientation_l2,
            "smoothness": joint_vel_l2,
            "action_rate": action_rate_l2,
            "torque_limit": normalized_torques_l2,
            "base_height": base_height_l2,
            "feet_air_time": feet_air_time,
            "forward_progress": forward_progress,
        }

        total = torch.zeros(self._num_envs, device=self._device)
        for name, fn in terms.items():
            w = float(weights.get(name, 0.0))
            if w == 0.0:
                continue
            try:
                if name == "base_height":
                    target = self._stage_cfg.base_height_target
                    val = fn(self, target_height=target)
                elif name == "feet_air_time":
                    val = fn(self, threshold=0.5)
                else:
                    val = fn(self)
                total = total + w * val
            except Exception:
                continue

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation.

        Mirrors gpu/env.py:_check_termination:
        - terminated: base height < 0.16 | |proj_grav_xy| > sin(1.0) | NaN qpos | NaN qvel
        - truncated: episode_length >= MAX_EPISODE_STEPS
        """
        robot: Articulation = self._robot

        # base_height termination
        base_z = robot.data.root_pos_w[:, 2]
        fall_height = base_z < self.cfg.base_height_term

        # orientation termination
        grav_proj = robot.data.projected_gravity_b
        fall_attitude = torch.sqrt(grav_proj[:, 0] ** 2 + grav_proj[:, 1] ** 2) > torch.sin(
            torch.tensor(self.cfg.orientation_term, device=self._device)
        )

        # NaN checks
        nan_q = ~torch.isfinite(robot.data.joint_pos).all(dim=1)
        nan_v = ~torch.isfinite(robot.data.joint_vel).all(dim=1)

        terminated = (fall_height | fall_attitude | nan_q | nan_v)

        # timeout
        truncated = self.episode_length_buf >= MAX_EPISODE_STEPS

        return (terminated, truncated)

    def _reset_idx(self, env_ids: torch.Tensor | Sequence[int]):
        """Reset specific env indices.

        Newton's articulation.reset() only resets actuators and wrenches; it does NOT
        re-apply init_state.joint_pos.  Write the nominal crouch stance explicitly
        (matching warp stack's _nominal_qpos) for the reset environments.
        """
        super()._reset_idx(env_ids)

        # Write default joint positions to sim (shape: (num_reset, 12))
        # robot.data.default_joint_pos already holds the correct stance in Isaac joint order
        pos = self._robot.data.default_joint_pos[env_ids]
        vel = torch.zeros_like(pos)
        self._robot.write_joint_state_to_sim_index(
            position=pos,
            velocity=vel,
            env_ids=env_ids,
        )

        # Reset prev_action buffer
        self._prev_action[env_ids] = 0.0


# -----------------------------------------------------------------------------
# Stage-specific config classes
# -----------------------------------------------------------------------------

@configclass
class Go2StandEnvCfg(Go2DirectRLEnvCfg):
    """Stage 0: Stand from rest."""
    def __post_init__(self):
        super().__post_init__()
        self.stage = "stand"
        self.spawn_height = 0.4
        self.base_height_target = 0.27
        # Update robot spawn height
        self.scene.robot = _get_go2_cfg(spawn_height=0.4)


@configclass
class Go2BalanceEnvCfg(Go2DirectRLEnvCfg):
    """Stage 1: Balance (stand with small perturbations)."""
    def __post_init__(self):
        super().__post_init__()
        self.stage = "balance"
        self.spawn_height = 0.4
        self.base_height_target = 0.27
        self.scene.robot = _get_go2_cfg(spawn_height=0.4)


@configclass
class Go2TransitionEnvCfg(Go2DirectRLEnvCfg):
    """Stage 2: Transition (low velocity tracking + locomotion terms)."""
    def __post_init__(self):
        super().__post_init__()
        self.stage = "transition"
        self.spawn_height = 0.27
        self.base_height_target = 0.3
        self.scene.robot = _get_go2_cfg(spawn_height=0.27)


@configclass
class Go2WalkEnvCfg(Go2DirectRLEnvCfg):
    """Stage 3: Walk (full velocity tracking + gait reward)."""
    def __post_init__(self):
        super().__post_init__()
        self.stage = "walk"
        self.spawn_height = 0.27
        self.base_height_target = 0.3
        self.scene.robot = _get_go2_cfg(spawn_height=0.27)