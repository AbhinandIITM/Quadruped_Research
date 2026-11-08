# Isaac Lab Migration Module (rl/isaac_lab/)

This module implements the Unitree Go2 quadruped RL training stack on **Isaac Lab 3.x** with the **Newton** physics engine. It maintains exact numerical parity with the mujoco-warp GPU stack (`rl/gpu/`) for:

- Observation space (45 dims per `rl/contracts.py`)
- Action space (12 dims, joint order FR/FL/RR/RL hip-thigh-calf)
- Reward term weights per stage (stand/balance/transition/walk)
- Deterministic velocity curriculum (0.3 → 2.1 m/s)
- Domain randomization schedule

---

## Module Layout

```
rl/isaac_lab/
├── __init__.py           # Package exports
├── configs.py            # Stage dataclasses (IsaacStageConfig) + reward mapping
├── runner.py             # IsaacLabRunner (RSL-RL OnPolicyRunner wrapper)
├── env.py                # DirectRLEnv parity env (Go2IsaacEnv)
├── envs/
│   ├── go2_env_cfg.py    # ManagerBasedRLEnvCfg configs (Go2*EnvCfg subclasses)
│   └── direct_env.py     # DirectRLEnv configs (legacy/alternative path)
├── mdp/                  # Observation/reward/termination MDP terms
├── rewards.py            # Legacy reward helpers
├── rewards_direct.py     # Reward MDP terms used by ManagerBasedRLEnv
├── events.py             # EventManager domain randomization terms
├── terminations.py       # Termination MDP terms
├── commands.py           # Velocity command observation term
├── curriculum/
│   └── velocity.py       # VelocityCurriculum (matches rl/rl/curriculum.py)
├── eval_adapter.py       # load_isaac_policy(checkpoint, stage) -> callable
├── visualize.py          # Standalone viewer for trained policies
├── checkpoint_bridge.py  # RSL-RL .pt <-> SB3 .zip conversion
├── config/               # YAML stage configs (stand.yaml, walk.yaml, smoke.yaml)
└── scripts/
    ├── train_isaac.py          # Single-stage single-seed CLI (no --stage all)
    ├── train_isaac_stand.py    # Stand stage wrapper
    ├── train_isaac_balance.py  # Balance stage wrapper
    ├── train_isaac_transition.py
    ├── train_isaac_walk.py     # Walk stage wrapper
    ├── train_isaac_multiseed.py
    ├── train_multiseed.py      # Subprocess-based multi-seed (per-stage)
    ├── smoke_test.py           # Fast env sanity check (all 4 stages)
    └── verify_parity.py        # GPU vs Isaac Lab parity verification
```

---

## Two Environment Paths

| Path | Class | Use case |
|------|-------|----------|
| **Primary (ManagerBased)** | `Go2StandEnvCfg`, `Go2BalanceEnvCfg`, `Go2TransitionEnvCfg`, `Go2WalkEnvCfg` in `envs/go2_env_cfg.py` | Production training via `runner.py` + `train_isaac` |
| **Legacy (DirectRL)** | `Go2IsaacEnv` in `env.py` + `envs/direct_env.py` | Parity reference; same obs/action but DirectRLEnv API |

The training CLI (`rl/train_isaac.py`, `rl/isaac_lab/scripts/train_isaac.py`) uses the **ManagerBased** path via `runner.py:_build_env_cfg()`.

---

## Key Configurations

### Stage Definitions (`configs.py`)

`IsaacStageConfig` dataclass holds per-stage params:
- `spawn_height`: 0.4 (stand/balance), 0.27 (transition/walk)
- `reward_weights`: dict matching `rl/gpu/env.py::_STAGE_CONFIGS` (keys mapped via `REWARD_KEY_MAP`)
- `curriculum`: `True` for transition/walk, `False` otherwise

### Reward Weights (GPU keys → Isaac Lab MDP terms)

| GPU term | Isaac Lab MDP term | Stand | Balance | Transition | Walk |
|----------|--------------------|-------|---------|------------|------|
| track_lin_vel | track_lin_vel_xy_exp | 0.0 | 0.0 | 0.1 | 1.0 |
| track_ang_vel | track_ang_vel_z_exp | 0.0 | 0.0 | 0.05 | 0.5 |
| posture | joint_deviation_l2 | -2.0 | -2.0 | -2.0 | -0.2 |
| orientation | flat_orientation_l2 | -2.0 | -2.0 | -1.0 | -0.3 |
| base_height | base_height_l2 | -2.0 | -2.0 | -1.0 | -0.2 |

### Curriculum (`curriculum/velocity.py`)

Deterministic velocity curriculum mirroring `rl/rl/curriculum.py`:
- 7 stages, 0.3 → 2.1 m/s (`REQUIRED_TOP_SPEED_MS`)
- Mix modes: forward / lateral / yaw / combined
- Resample every 10 s (`resample_s`)
- Spread windows = 4 (staggered per-env resampling)
- Deterministic per-step RNG via `SeedSequence([seed, step])`

### Domain Randomization (`events.py`)

EventManager terms porting `MuJoCoRandomizer`:
- `randomize_rigid_body_mass`
- `randomize_rigid_body_com`
- `randomize_joint_friction`
- `randomize_friction` (terrain)
- `randomize_motor_strength`
- `push_robot` (external force)

---

## Training Entrypoints

| Command | Module | Scope |
|---------|--------|-------|
| `python -m rl.train_isaac` | `rl/train_isaac.py` | **Primary**: `--stage all`, `--num-seeds`, `--seeds` |
| `python -m rl.isaac_lab.scripts.train_isaac` | `scripts/train_isaac.py` | Single stage, single seed only |
| `python -m rl.isaac_lab.scripts.train_isaac_<stage>` | `scripts/train_isaac_*.py` | Per-stage convenience wrappers |

All entry points use `runner.py:train_stage()` → `IsaacLabRunner` → RSL-RL `OnPolicyRunner`.

### PPO Settings (fixed, in `runner.py:_build_runner_cfg`)

- `num_steps_per_env=24`, `max_iterations` derived from `total_steps`
- `entropy_coef=0.0` (critical: prevents entropy explosion / action rail saturation)
- `clip_param=0.2`, `gamma=0.99`, `lam=0.95`
- LR `1e-3`, 5 epochs, 4 minibatches, adaptive schedule
- Actor/Critic: MLP [128, 128, 128], ELU, obs_normalization=False

---

## Evaluation & Utilities

### Load Policy for Inference

```python
from rl.isaac_lab.eval_adapter import load_isaac_policy

policy = load_isaac_policy("logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt", stage="walk")
action = policy(obs)  # obs: np.ndarray[45] -> action: np.ndarray[12]
```

### Visualize

```bash
python -m rl.isaac_lab.visualize \
    --checkpoint logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt \
    --stage walk --episodes 5  # add --headless to run without GUI
```

### Checkpoint Conversion

```python
from rl.isaac_lab.checkpoint_bridge import convert_rsl_to_sb3, convert_sb3_to_rsl

convert_rsl_to_sb3("logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt", "walk_sb3.zip")
convert_sb3_to_rsl("walk_sb3.zip", "walk_rsl.pt")
```

### Parity Verification

```bash
python -m rl.isaac_lab.scripts.verify_parity
```

Trains both stacks for 100k steps (seed 0, 256 envs) and compares distributions.

---

## Migration Status

See [`docs/ISAAC_LAB_MIGRATION_GUIDE.md`](../ISAAC_LAB_MIGRATION_GUIDE.md#migration-checklist) for the full checklist.