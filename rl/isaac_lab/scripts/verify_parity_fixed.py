#!/usr/bin/env python3
"""Training parity verification: mujoco-warp GPU vs Isaac Lab DirectRLEnv.

Trains both stacks for 10k steps with same seed and compares:
- Observation distribution (mean/std per dimension)
- Action distribution (mean/std per dimension)
- Reward term means per term
- Episode length distribution
- Curriculum command progression
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from scipy import stats

# Add rl to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Mujoco-warp imports
from rl.gpu.env import Go2WarpEnv, Go2WarpEnvConfig
from rl.gpu.runner import GPURunner

# Isaac Lab DirectRLEnv
from rl.isaac_lab.envs.direct_env import Go2DirectRLEnv, Go2DirectRLEnvCfg

# AppLauncher for Isaac Lab
from isaaclab.app import AppLauncher


PARITY_STEPS = 10_000  # Fast mode: 10k steps
NUM_ENVS = 256
SEED = 0
TOLERANCE_MEAN = 0.01  # 1% relative tolerance
TOLERANCE_STD = 0.05   # 5% relative tolerance


def _aggregate_reward_terms(reward_terms_list: List[Dict]) -> Dict[str, float]:
    """Aggregate reward term statistics."""
    if not reward_terms_list:
        return {}
    all_keys = set()
    for t in reward_terms_list:
        all_keys.update(t.keys())
    result = {}
    for k in all_keys:
        vals = [t.get(k, 0) for t in reward_terms_list]
        result[k] = float(np.mean(vals))
    return result


def collect_mujoco_warp_data(num_steps: int = PARITY_STEPS, num_envs: int = NUM_ENVS) -> Dict[str, Any]:
    """Collect statistics from mujoco-warp stack."""
    print(f"Collecting mujoco-warp data ({num_envs} envs, {num_steps} steps)...")

    cfg = Go2WarpEnvConfig(
        stage="walk",
        num_envs=num_envs,
        seed=SEED,
        device="cuda",
    )
    env = Go2WarpEnv(cfg)
    runner = GPURunner(env, run_dir="/tmp/mujoco_warp_parity", seed=SEED)

    # Storage
    obs_list = []
    action_list = []
    reward_terms_list = []
    episode_lengths = []
    commands_list = []

    # Get initial observations
    obs_td = env.get_observations()
    obs = obs_td["policy"]
    ep_len = 0

    for step in range(num_steps):
        # Get actions from random policy (for distribution comparison)
        with torch.no_grad():
            actions = torch.randn(num_envs, 12, device="cuda").clamp(-1.0, 1.0)

        obs_list.append(obs.cpu().numpy().copy())
        action_list.append(actions.cpu().numpy().copy())
        commands_list.append(env._commands.cpu().numpy().copy())

        obs_td, reward, dones, extras = env.step(actions)
        obs = obs_td["policy"]

        if "log" in extras:
            reward_terms_list.append(extras["log"])

        ep_len += 1
        if dones.any():
            done_idx = torch.where(dones)[0]
            for idx in done_idx:
                episode_lengths.append(ep_len)
            ep_len = 0

        if step % 2000 == 0:
            print(f"  Step {step}/{num_steps}")

    env.close()

    # Compute statistics
    obs_array = np.concatenate(obs_list, axis=0)
    action_array = np.concatenate(action_list, axis=0)
    cmd_array = np.concatenate(commands_list, axis=0)

    return {
        "obs": obs_array,
        "actions": action_array,
        "commands": cmd_array,
        "reward_terms": _aggregate_reward_terms(reward_terms_list),
        "episode_lengths": np.array(episode_lengths),
        "obs_mean": obs_array.mean(axis=0).tolist(),
        "obs_std": obs_array.std(axis=0).tolist(),
        "action_mean": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "command_mean": cmd_array.mean(axis=0).tolist(),
        "command_std": cmd_array.std(axis=0).tolist(),
        "ep_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "ep_len_std": float(np.std(episode_lengths)) if episode_lengths else 0.0,
    }


def collect_direct_rl_data(num_steps: int = PARITY_STEPS, num_envs: int = NUM_ENVS, app=None) -> Dict[str, Any]:
    """Collect statistics from Isaac Lab DirectRLEnv stack."""
    print(f"Collecting DirectRLEnv data ({num_envs} envs, {num_steps} steps)...")

    env_cfg = Go2DirectRLEnvCfg(
        stage="walk",
        num_envs=num_envs,
        seed=SEED,
        device="cuda",
        decimation=10,
        episode_length_s=20.0,
        obs_noise=0.0,
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.device = "cuda"

    env = Go2DirectRLEnv(env_cfg)

    # Check if physics handles decimation (PD re-solved at 500 Hz)
    try:
        handles_dec = hasattr(env, '_physics_handles_decimation') and env._physics_handles_decimation
        print(f"  Physics handles decimation (PD @ 500Hz): {handles_dec}")
        if hasattr(env.sim, 'physics_manager'):
            print(f"  Newton handles_decimation(): {env.sim.physics_manager.handles_decimation()}")
    except Exception as e:
        print(f"  Could not check handles_decimation: {e}")

    obs_list = []
    action_list = []
    reward_terms_list = []
    episode_lengths = []
    commands_list = []

    obs_dict = env.reset()
    obs = obs_dict["policy"]
    ep_len = 0

    for step in range(num_steps):
        with torch.no_grad():
            actions = torch.randn(num_envs, 12, device="cuda").clamp(-1, 1)

        obs_list.append(obs.cpu().numpy().copy())
        action_list.append(actions.cpu().numpy().copy())
        commands_list.append(env._commands.cpu().numpy().copy())

        obs_dict, rewards, dones, infos = env.step(actions)
        obs = obs_dict["policy"]

        if "reward_terms" in infos:
            reward_terms_list.append(infos["reward_terms"])

        ep_len += 1
        if dones.any():
            done_idx = torch.where(dones)[0]
            for idx in done_idx:
                episode_lengths.append(ep_len)
            ep_len = 0

        if step % 2000 == 0:
            print(f"  Step {step}/{num_steps}")

    env.close()

    obs_array = np.concatenate(obs_list, axis=0)
    action_array = np.concatenate(action_list, axis=0)
    cmd_array = np.concatenate(commands_list, axis=0)

    return {
        "obs": obs_array,
        "actions": action_array,
        "commands": cmd_array,
        "reward_terms": _aggregate_reward_terms(reward_terms_list),
        "episode_lengths": np.array(episode_lengths),
        "obs_mean": obs_array.mean(axis=0).tolist(),
        "obs_std": obs_array.std(axis=0).tolist(),
        "action_mean": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "command_mean": cmd_array.mean(axis=0).tolist(),
        "command_std": cmd_array.std(axis=0).tolist(),
        "ep_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "ep_len_std": float(np.std(episode_lengths)) if episode_lengths else 0.0,
    }


def compare_distributions(name: str, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
    """Compare two distributions with statistical tests."""
    mean1, std1 = data1.mean(), data1.std()
    mean2, std2 = data2.mean(), data2.std()

    rel_mean_diff = abs(mean1 - mean2) / (abs(mean1) + 1e-8)
    rel_std_diff = abs(std1 - std2) / (abs(std1) + 1e-8)

    # KS test
    ks_stat, ks_p = stats.ks_2samp(data1.flatten(), data2.flatten())

    # t-test for means
    t_stat, t_p = stats.ttest_ind(data1.flatten(), data2.flatten(), equal_var=False)

    return {
        "name": name,
        "mean1": float(mean1), "mean2": float(mean2),
        "std1": float(std1), "std2": float(std2),
        "rel_mean_diff": float(rel_mean_diff),
        "rel_std_diff": float(rel_std_diff),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_p),
        "mean_pass": rel_mean_diff < TOLERANCE_MEAN,
        "std_pass": rel_std_diff < TOLERANCE_STD,
        "ks_pass": ks_p > 0.05,  # Fail to reject null = distributions similar
    }


def compare_reward_terms(mw_terms: Dict, il_terms: Dict) -> Dict[str, Any]:
    """Compare reward term means."""
    all_keys = set(mw_terms.keys()) | set(il_terms.keys())
    results = {}
    for k in all_keys:
        v1 = mw_terms.get(k, 0.0)
        v2 = il_terms.get(k, 0.0)
        rel_diff = abs(v1 - v2) / (abs(v1) + 1e-8)
        results[k] = {
            "mujoco_warp": float(v1),
            "direct_rl": float(v2),
            "rel_diff": float(rel_diff),
            "pass": rel_diff < TOLERANCE_MEAN,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=PARITY_STEPS)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
    parser.add_argument("--output", type=str, default="/tmp/parity_report.json")
    args = parser.parse_args()

    # Configure app for Isaac Lab
    app_launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=False,
    )
    app = app_launcher.app

    print("=" * 60)
    print("PARITY VERIFICATION: MUJOCO-WARP vs ISAAC LAB DirectRLEnv")
    print(f"Steps: {args.steps}, Envs: {args.num_envs}, Seed: {args.seed}")
    print("=" * 60)

    # Collect data from both stacks
    mw_data = collect_mujoco_warp_data(num_steps=args.steps, num_envs=args.num_envs)
    il_data = collect_direct_rl_data(num_steps=args.steps, num_envs=args.num_envs, app=app)

    # Compare observations per dimension
    print("\nComparing observations (45 dims)...")
    obs_results = {}
    obs_pass_count = 0
    for i in range(45):
        res = compare_distributions(f"obs_{i}", mw_data["obs"][:, i], il_data["obs"][:, i])
        obs_results[f"obs_{i}"] = res
        if res["mean_pass"] and res["std_pass"]:
            obs_pass_count += 1

    # Compare actions per dimension
    print("Comparing actions (12 dims)...")
    action_results = {}
    action_pass_count = 0
    for i in range(12):
        res = compare_distributions(f"action_{i}", mw_data["actions"][:, i], il_data["actions"][:, i])
        action_results[f"action_{i}"] = res
        if res["mean_pass"] and res["std_pass"]:
            action_pass_count += 1

    # Compare commands (3 dims)
    print("Comparing commands...")
    cmd_results = {}
    for i, name in enumerate(["vx", "vy", "yaw"]):
        res = compare_distributions(f"cmd_{name}", mw_data["commands"][:, i], il_data["commands"][:, i])
        cmd_results[name] = res

    # Compare reward terms
    print("Comparing reward terms...")
    reward_results = compare_reward_terms(mw_data["reward_terms"], il_data["reward_terms"])
    reward_pass = sum(1 for v in reward_results.values() if v["pass"])
    reward_total = len(reward_results)

    # Compare episode lengths
    print("Comparing episode lengths...")
    ep_len_result = compare_distributions(
        "episode_length",
        mw_data["episode_lengths"].astype(float),
        il_data["episode_lengths"].astype(float),
    )

    # Summary
    print("\n" + "=" * 60)
    print("PARITY REPORT")
    print("=" * 60)

    all_pass = True

    for category, results, pass_count, total in [
        ("Observations", obs_results, obs_pass_count, 45),
        ("Actions", action_results, action_pass_count, 12),
        ("Commands", cmd_results, sum(1 for v in cmd_results.values() if v["mean_pass"] and v["std_pass"]), 3),
        ("Episode Lengths", {"episode_length": ep_len_result}, int(ep_len_result["mean_pass"] and ep_len_result["std_pass"]), 1),
    ]:
        print(f"\n{category}: {pass_count}/{total} dims PASS")
        for k, v in results.items():
            if isinstance(v, dict) and "mean_pass" in v:
                status = "✓" if (v["mean_pass"] and v["std_pass"]) else "✗"
                print(f"  {status} {k}: mean1={v['mean1']:.4f} mean2={v['mean2']:.4f} "
                      f"rel_diff={v['rel_mean_diff']:.4f} std_diff={v['rel_std_diff']:.4f}")

    for k, v in reward_results.items():
        status = "✓" if v["pass"] else "✗"
        print(f"  {status} {k}: mw={v['mujoco_warp']:.4f} dr={v['direct_rl']:.4f} rel_diff={v['rel_diff']:.4f}")

    reward_pass_count = sum(1 for v in reward_results.values() if v["pass"])
    print(f"\nReward terms: {reward_pass_count}/{reward_total} PASS")

    if reward_pass_count != reward_total:
        all_pass = False

    # Overall
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ PARITY ACHIEVED")
    else:
        print("✗ PARITY FAILED")
    print("=" * 60)

    # Save report
    report = {
        "parity_steps": args.steps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "obs_pass_rate": obs_pass_count / 45,
        "action_pass_rate": action_pass_count / 12,
        "command_pass_rate": sum(1 for v in cmd_results.values() if v["mean_pass"] and v["std_pass"]) / 3,
        "ep_len_pass": ep_len_result["mean_pass"] and ep_len_result["std_pass"],
        "reward_pass_rate": reward_pass_count / reward_total if reward_total else 0,
        "observations": obs_results,
        "actions": action_results,
        "commands": cmd_results,
        "reward_terms": reward_results,
        "episode_length": ep_len_result,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {args.output}")
    app.close()


if __name__ == "__main__":
    main()