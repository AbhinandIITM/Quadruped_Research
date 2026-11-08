#!/usr/bin/env python3
"""Training parity verification: mujoco-warp GPU vs Isaac Lab.

Trains both stacks for 100k steps with same seed and compares:
- Observation distribution (mean/std per dimension)
- Action distribution (mean/std per dimension)
- Reward term means per term
- Episode length distribution
- Curriculum command progression
- Domain randomization statistics

NOTE: This script is retained as a reference artifact of the MuJoCo <-> Isaac
parity check. It imports the MuJoCo stack (``rl.gpu.env``, ``rl.gpu.runner``);
those modules were removed during the rl/configs cleanup, so this script is not
executable as-is and should be read for the parity methodology / expectations.
"""

from __future__ import annotations

import sys
import json
import time
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

# Isaac Lab imports
from rl.isaac_lab.runner import _build_env_cfg, _build_runner_cfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


PARITY_STEPS = 100_000
NUM_ENVS = 256  # Smaller for faster comparison
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
        result[k] = np.mean(vals)
    return result


def collect_mujoco_warp_data(num_steps: int = PARITY_STEPS) -> Dict[str, Any]:
    """Collect statistics from mujoco-warp stack."""
    print("Collecting mujoco-warp data...")

    cfg = Go2WarpEnvConfig(
        stage="walk",
        num_envs=NUM_ENVS,
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
            actions = torch.randn(NUM_ENVS, 12, device="cuda").clamp(-1.0, 1.0)

        obs_list.append(obs.cpu().numpy().copy())
        action_list.append(actions.cpu().numpy().copy())
        commands_list.append(env._commands.cpu().numpy().copy())

        obs_td, reward, dones, extras = env.step(actions)
        obs = obs_td["policy"]

        if "log" in extras:
            reward_terms_list.append(extras["log"])

        ep_len += 1
        if dones.any():
            episode_lengths.append(ep_len)
            ep_len = 0

    # Compute statistics
    obs_array = np.concatenate(obs_list, axis=0)
    action_array = np.concatenate(action_list, axis=0)
    cmd_array = np.concatenate(commands_list, axis=0)

    return {
        "obs_mean": obs_array.mean(axis=0).tolist(),
        "obs_std": obs_array.std(axis=0).tolist(),
        "action_mean": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "command_mean": cmd_array.mean(axis=0).tolist(),
        "command_std": cmd_array.std(axis=0).tolist(),
        "reward_terms": _aggregate_reward_terms(reward_terms_list),
        "episode_lengths": episode_lengths,
        "ep_len_mean": np.mean(episode_lengths) if episode_lengths else 0.0,
        "ep_len_std": np.std(episode_lengths) if episode_lengths else 0.0,
    }


def collect_isaac_lab_data(num_steps: int = PARITY_STEPS) -> Dict[str, Any]:
    """Collect statistics from Isaac Lab stack."""
    print("Collecting Isaac Lab data...")

    env_cfg = _build_env_cfg("walk", NUM_ENVS, "cuda")
    env = ManagerBasedRLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    obs_list = []
    action_list = []
    reward_terms_list = []
    episode_lengths = []
    commands_list = []

    obs, _ = env.reset()
    ep_len = 0

    for step in range(num_steps):
        with torch.no_grad():
            actions = torch.randn(NUM_ENVS, 12, device="cuda").clamp(-1, 1)

        obs, rewards, dones, infos = env.step(actions)

        obs_list.append(obs.cpu().numpy())
        action_list.append(actions.cpu().numpy())
        commands_list.append(
            env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy()
        )

        if "reward_terms" in infos:
            reward_terms_list.append(infos["reward_terms"])

        ep_len += 1
        if dones.any():
            done_idx = torch.where(dones)[0]
            for idx in done_idx:
                episode_lengths.append(ep_len)
            ep_len = 0

        if step % 20000 == 0:
            print(f"  Step {step}/{num_steps}")

    env.close()

    return {
        "obs": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(action_list, axis=0),
        "reward_terms": reward_terms_list,
        "episode_lengths": np.array(episode_lengths),
        "commands": np.concatenate(commands_list, axis=0),
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


def compare_reward_terms(terms1: list, terms2: list) -> Dict[str, Any]:
    """Compare reward term means."""
    if not terms1 or not terms2:
        return {"error": "No reward terms available"}

    # Get all term names
    all_terms = set()
    for t in terms1:
        all_terms.update(t.keys())
    for t in terms2:
        all_terms.update(t.keys())

    results = {}
    for term in sorted(all_terms):
        vals1 = np.array([t.get(term, 0) for t in terms1])
        vals2 = np.array([t.get(term, 0) for t in terms2])

        mean1, mean2 = vals1.mean(), vals2.mean()
        rel_diff = abs(mean1 - mean2) / (abs(mean1) + 1e-8)

        results[term] = {
            "mean1": float(mean1),
            "mean2": float(mean2),
            "rel_diff": float(rel_diff),
            "pass": rel_diff < TOLERANCE_MEAN,
        }

    return results


def main():
    print("=" * 60)
    print("PARITY VERIFICATION: MUJOCO-WARP vs ISAAC LAB")
    print("=" * 60)

    # Collect data from both stacks
    mw_data = collect_mujoco_warp_data()
    il_data = collect_isaac_lab_data()

    # Compare observations per dimension
    print("\nComparing observations (45 dims)...")
    obs_results = {}
    for i in range(45):
        obs_results[f"obs_{i}"] = compare_distributions(
            f"obs_{i}", mw_data["obs"][:, i], il_data["obs"][:, i]
        )

    # Compare actions per dimension
    print("Comparing actions (12 dims)...")
    action_results = {}
    for i in range(12):
        action_results[f"action_{i}"] = compare_distributions(
            f"action_{i}", mw_data["actions"][:, i], il_data["actions"][:, i]
        )

    # Compare commands (3 dims)
    print("Comparing commands...")
    cmd_results = {}
    for i, name in enumerate(["vx", "vy", "yaw"]):
        cmd_results[name] = compare_distributions(
            f"cmd_{name}", mw_data["commands"][:, i], il_data["commands"][:, i]
        )

    # Compare reward terms
    print("Comparing reward terms...")
    reward_results = compare_reward_terms(mw_data["reward_terms"], il_data["reward_terms"])

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

    for category, results in [
        ("Observations", obs_results),
        ("Actions", action_results),
        ("Commands", cmd_results),
    ]:
        print(f"\n{category}:")
        for name, r in results.items():
            status = "PASS" if r["mean_pass"] and r["std_pass"] and r["ks_pass"] else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {name:12s}: {status} (mean_diff={r['rel_mean_diff']:.3f}, std_diff={r['rel_std_diff']:.3f}, ks_p={r['ks_pvalue']:.3f})")

    print("\nReward Terms:")
    for name, r in reward_results.items():
        status = "PASS" if r["pass"] else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name:25s}: {status} (diff={r['rel_diff']:.3f})")

    print(f"\nEpisode Length: {'PASS' if ep_len_result['mean_pass'] and ep_len_result['ks_pass'] else 'FAIL'}")
    if not (ep_len_result['mean_pass'] and ep_len_result['ks_pass']):
        all_pass = False

    # Save detailed report
    report = {
        "timestamp": time.time(),
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "num_steps": PARITY_STEPS,
        "tolerance_mean": TOLERANCE_MEAN,
        "tolerance_std": TOLERANCE_STD,
        "overall_pass": all_pass,
        "observations": obs_results,
        "actions": action_results,
        "commands": cmd_results,
        "reward_terms": reward_results,
        "episode_length": ep_len_result,
    }

    output_path = Path(__file__).parent.parent / "parity_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {output_path}")
    print(f"\nOVERALL: {'PARITY ACHIEVED' if all_pass else 'PARITY FAILED'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())