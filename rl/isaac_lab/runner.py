"""Isaac Lab Runner: RSL-RL OnPolicyRunner wrapper with run.json logging.

Mirrors the mujoco-warp GPURunner API for parity with existing training scripts.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from rl.envs.vec import derive_env_seeds
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version

from rl.isaac_lab.envs.go2_env_cfg import (
    Go2StandEnvCfg,
    Go2BalanceEnvCfg,
    Go2TransitionEnvCfg,
    Go2WalkEnvCfg,
    Go2FlatEnvCfg,
)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _load_yaml_config(config_name: str) -> dict:
    """Load YAML configuration file."""
    config_path = Path(__file__).parent / "config" / f"{config_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_runner_cfg(config_name: str, num_envs: int, device: str) -> RslRlOnPolicyRunnerCfg:
    """Build RSL-RL runner configuration matching official UnitreeGo2FlatPPORunnerCfg."""
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

    # Match official flat config exactly
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=300,
        save_interval=50,
        experiment_name="go2_stand",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        init_at_random_ep_len=True,  # Match official default
        actor=RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,  # Official uses False
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=1.0,
                std_type="scalar",  # Official uses scalar, not log
            ),
        ),
        critic=RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,  # Official uses False
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,  # mild entropy keeps exploration alive so gait can be discovered
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        seed=0,
        device=device,
    )


def _build_env_cfg(config_name: str, num_envs: int, device: str):
    """Build environment configuration for the given stage using our custom configs."""
    configs = {
        "stand": Go2StandEnvCfg,
        "balance": Go2BalanceEnvCfg,
        "transition": Go2TransitionEnvCfg,
        "walk": Go2WalkEnvCfg,
        "flat": Go2FlatEnvCfg,
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    # Use our custom stage config directly
    stage_cfg_class = configs[config_name]
    env_cfg = stage_cfg_class()

    # Set num_envs and device
    env_cfg.scene.num_envs = num_envs
    env_cfg.device = device

    return env_cfg


# ---------------------------------------------------------------------------
# IsaacLabRunner
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Per-iteration training metrics matching run.json schema."""

    steps: int
    wall_clock_s: float
    ep_rew_mean: float
    ep_len_mean: float
    value_loss: float
    policy_gradient_loss: float
    entropy_loss: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    n_updates: int
    steps_per_s: float
    terms: dict[str, float] = field(default_factory=dict)


class IsaacLabRunner:
    """Thin wrapper around RSL-RL OnPolicyRunner with run.json logging.

    Responsibilities:
    - Build the RSL-RL training config and instantiate OnPolicyRunner.
    - Override learn() to capture per-iteration metrics and write run.json.
    - Handle checkpoint resume via OnPolicyRunner.load().
    """

    def __init__(
        self,
        stage: str = "walk",
        seed: int = 0,
        device: str = "cuda",
        num_envs: int = 4096,
        config_name: str | None = None,
        verbose: int = 1,
        log_dir: str | None = None,
    ):
        """Initialize the runner.

        Args:
            stage: Stage name (stand/balance/transition/walk).
            seed: Random seed.
            device: Training device (cuda/cpu).
            num_envs: Number of parallel environments.
            config_name: YAML config name (defaults to stage).
            verbose: Verbosity level.
        """
        self.stage = stage
        self.seed = seed
        self.device = device
        self.num_envs = num_envs
        self.config_name = config_name or stage
        self.verbose = verbose
        self.log_dir = log_dir or "logs/rsl_rl"

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build configs
        self.env_cfg = _build_env_cfg(self.config_name, num_envs, device)
        self.runner_cfg = _build_runner_cfg(self.config_name, num_envs, device)

        # Set env_cfg.seed to match runner seed (like official entrypoint)
        self.env_cfg.seed = seed
        self.runner_cfg.seed = seed

        if verbose:
            print(f"[IsaacLabRunner] stage={stage} seed={seed} envs={num_envs} "
                  f"config={self.config_name} device={device}")

        # Create environment
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        self.env = ManagerBasedRLEnv(self.env_cfg)
        self.env = RslRlVecEnvWrapper(self.env, clip_actions=self.runner_cfg.clip_actions)
        self.env.seed(seed)

        # Migrate deprecated RSL-RL config (rsl-rl 5.5.1)
        installed_version = check_rsl_rl_version()
        self.runner_cfg = handle_deprecated_rsl_rl_cfg(self.runner_cfg, installed_version)
        # Create runner with log_dir (official does this)
        self._runner = create_rsl_rl_runner(self.env, self.runner_cfg, log_dir=self.log_dir)

        # Configure seed AFTER runner construction (matches official entrypoint)
        from isaaclab.utils.seed import configure_seed
        configure_seed(self.env_cfg.seed, torch_deterministic=False)

        # Metrics history
        self._metrics_history: list[TrainingMetrics] = []
        self._start_time = time.time()
        self._global_step = 0

    @property
    def env(self) -> ManagerBasedRLEnv:
        return self._env

    @env.setter
    def env(self, value: ManagerBasedRLEnv):
        self._env = value

    @property
    def runner(self):
        return self._runner

    @property
    def run_json_path(self) -> Path:
        """Path of the run.json written by the most recent train() call."""
        return getattr(self, "_run_json_path", None)

    # -------------------------------------------------------------------------
    # Training entry point
    # -------------------------------------------------------------------------

    def train(
        self,
        total_steps: int,
        resume_path: str | Path | None = None,
        arm: str = "A",
    ) -> Path:
        """Run PPO for total_steps env steps and return the path to run.json.

        Args:
            total_steps: Total env steps to train for (absolute budget).
            resume_path: Path to a .pt checkpoint to resume from.
            arm: Study arm written to run.json (always "A" for heuristic reward).

        Returns:
            Path to the written run.json file.
        """
        n_iters_per_step = self._runner.cfg["num_steps_per_env"] * self.num_envs
        num_iterations = max(1, total_steps // n_iters_per_step)

        # Resume
        resumed_from: str | None = None
        if resume_path is not None:
            rp = Path(resume_path)
            if rp.exists():
                self._runner.load(str(rp))
                resumed_from = str(rp)
                if self.verbose:
                    print(f"[IsaacLabRunner] Resumed from {rp}")

        if self.verbose:
            print(
                f"[IsaacLabRunner] stage={self.stage} seed={self.seed} "
                f"envs={self.num_envs} iterations={num_iterations} "
                f"steps_per_iter={n_iters_per_step}"
            )

        # Training loop - manual iteration to capture metrics
        self._metrics_history.clear()
        self._start_time = time.time()
        self._global_step = 0
        self._last_loss_dict: dict[str, float] = {}

        # Capture per-update losses: RSL-RL's PPO.update() returns a loss_dict that
        # learn() discards; wrap it so we can log the real values per iteration.
        _orig_update = self._runner.alg.update

        def _update_with_capture(*args, **kwargs):
            loss_dict = _orig_update(*args, **kwargs)
            self._last_loss_dict = {k: float(v) for k, v in loss_dict.items()}
            return loss_dict

        self._runner.alg.update = _update_with_capture

        for iteration in range(num_iterations):
            # CRITICAL: RSL-RL's learn() uses current_learning_iteration as the start of its
            # internal range(). Each learn(1) call would reset to it=0 unless we advance it.
            self._runner.current_learning_iteration = iteration

            # Step the runner - match official entrypoint (NO inference_mode!)
            self._runner.learn(num_learning_iterations=1, init_at_random_ep_len=self.runner_cfg.init_at_random_ep_len)

            # Get metrics from runner
            metrics = self._extract_metrics(iteration)
            self._metrics_history.append(metrics)
            self._global_step += n_iters_per_step

            # Log progress
            if self.verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                print(
                    f"  iter {iteration:4d} | step {self._global_step:8d} | "
                    f"rew {metrics.ep_rew_mean:6.3f} | len {metrics.ep_len_mean:5.1f} | "
                    f"vl {metrics.value_loss:.4f} | pg {metrics.policy_gradient_loss:.4f} | "
                    f"ent {metrics.entropy_loss:.4f} | kl {metrics.approx_kl:.4f} | "
                    f"ev {metrics.explained_variance:.4f} | "
                    f"{metrics.steps_per_s:.0f} steps/s"
                )

            # Save checkpoint
            if (iteration + 1) % self._runner.cfg["save_interval"] == 0:
                self._save_checkpoint(iteration + 1)

        # Final save (runner.current_learning_iteration now equals num_iterations-1,
        # learn() will have set it to num_iterations-1, so final save uses that iteration)
        self._runner.current_learning_iteration = num_iterations
        self._save_checkpoint(num_iterations, final=True)

        # Write run.json
        run_json_path = self._write_run_json(resumed_from, arm)
        return run_json_path

    def _extract_metrics(self, iteration: int) -> TrainingMetrics:
        """Extract metrics from runner after an iteration."""
        # Get training stats from runner
        # RSL-RL stores these in the algorithm and logger
        algo = self._runner.alg
        logger = self._runner.logger

        # Episode stats - from logger's buffers (matches official entrypoint)
        ep_rew_mean = logger.rewbuffer[-1] if logger.rewbuffer else 0.0
        ep_len_mean = logger.lenbuffer[-1] if logger.lenbuffer else 0.0

        # Algorithm stats - RSL-RL 5.x returns loss_dict from PPO.update() with keys
        # value / surrogate / entropy. KL and clip fraction are not computed in this
        # version, so they stay 0.0 (they were holdover fields from SB3).
        loss_dict = self._last_loss_dict
        value_loss = loss_dict.get("value", 0.0)
        pg_loss = loss_dict.get("surrogate", 0.0)
        entropy_loss = loss_dict.get("entropy", 0.0)
        approx_kl = 0.0
        clip_fraction = 0.0

        # Explained variance - compute from storage
        storage = algo.storage
        if hasattr(storage, "values") and hasattr(storage, "returns"):
            values = storage.values.flatten()
            returns = storage.returns.flatten()
            if values.numel() > 1 and returns.numel() > 1:
                var_returns = torch.var(returns)
                if var_returns > 1e-8:
                    explained_var = 1.0 - torch.var(returns - values) / var_returns
                else:
                    explained_var = torch.tensor(0.0)
            else:
                explained_var = torch.tensor(0.0)
        else:
            explained_var = torch.tensor(0.0)

        # Reward terms - from episode extras
        terms = {}
        env = self._runner.env.unwrapped
        if hasattr(env, "extras") and "reward_terms" in env.extras:
            reward_terms = env.extras["reward_terms"]
            for k, v in reward_terms.items():
                terms[k] = v.mean().item()

        return TrainingMetrics(
            steps=self._global_step,
            wall_clock_s=time.time() - self._start_time,
            ep_rew_mean=ep_rew_mean,
            ep_len_mean=ep_len_mean,
            value_loss=value_loss,
            policy_gradient_loss=pg_loss,
            entropy_loss=entropy_loss,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            explained_variance=explained_var.item(),
            n_updates=iteration + 1,
            steps_per_s=self._global_step / (time.time() - self._start_time),
            terms=terms,
        )

    def _save_checkpoint(self, iteration: int, final: bool = False):
        """Save model checkpoint."""
        run_dir = Path(self._runner.logger.log_dir) if self._runner.logger.log_dir else Path("logs/isaac_lab") / f"run_{self.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = run_dir / "model_final.pt"
        else:
            path = run_dir / f"model_{iteration:04d}.pt"

        self._runner.save(str(path))
        if self.verbose:
            print(f"[IsaacLabRunner] Saved checkpoint: {path}")

    def _write_run_json(self, resumed_from: str | None, arm: str) -> Path:
        """Write run.json with metrics history."""
        run_dir = Path(self._runner.logger.log_dir) if self._runner.logger.log_dir else Path("logs/isaac_lab") / f"run_{self.seed}"
        run_json_path = run_dir / "run.json"

        # Build run.json matching the study schema
        run_data = {
            "stage": self.stage,
            "arm": arm,
            "seed": self.seed,
            "num_envs": self.num_envs,
            "total_steps": self._global_step,
            "config_name": self.config_name,
            "resumed_from": resumed_from,
            "metrics_history": [asdict(m) for m in self._metrics_history],
        }

        with open(run_json_path, "w") as f:
            json.dump(run_data, f, indent=2)

        if self.verbose:
            print(f"[IsaacLabRunner] Wrote run.json: {run_json_path}")

        self._run_json_path = run_json_path
        return run_json_path


# ---------------------------------------------------------------------------
# Convenience function matching GPURunner API
# ---------------------------------------------------------------------------


def train_stage(
    stage: str,
    seed: int = 0,
    total_steps: int = 5_000_000,
    run_dir: Path | None = None,
    resume: Path | None = None,
    device: str = "cuda",
    num_envs: int = 4096,
    verbose: int = 1,
    config_name: str | None = None,
) -> 'IsaacLabRunner':
    """Train a stage and return run.json path.

    Matches the GPURunner.train_stage() signature for drop-in replacement.
    """
    log_dir = str(run_dir) if run_dir is not None else None
    runner = IsaacLabRunner(
        stage=stage,
        seed=seed,
        device=device,
        num_envs=num_envs,
        config_name=config_name,
        verbose=verbose,
        log_dir=log_dir,
    )

    try:
        runner.train(total_steps=total_steps, resume_path=resume, arm="A")
    finally:
        # Ensure environment is closed to clear SimulationContext singleton
        # This is critical for ProcessPoolExecutor worker reuse
        try:
            runner.env.close()
        except Exception:
            pass
    return runner