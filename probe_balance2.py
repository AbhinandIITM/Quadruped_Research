import sys
sys.path.insert(0, "/d/Projects/Quadruped")
import torch
from rl.isaac_lab.envs.go2_env_cfg import Go2BalanceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

cfg = Go2BalanceEnvCfg()
cfg.scene.num_envs = 4
cfg.device = "cuda"
env = ManagerBasedRLEnv(cfg)

# First reset (happens during env creation)
print("=== Episode 1 ===")
obs, _ = env.reset()
print("After reset z:", env.scene['robot'].data.root_pos_w[:, 2].tolist())
print("root_lin_vel_b:", env.scene['robot'].data.root_lin_vel_b.tolist())
print("joint_pos:", env.scene['robot'].data.joint_pos[0].tolist())

for i in range(5):
    action = torch.zeros((4, 12), device="cuda")
    next_obs, rew, terminated, truncated, info = env.step(action)
    print(f"Step {i}: z={env.scene['robot'].data.root_pos_w[:, 2].tolist()}, term={terminated.tolist()}, rew={rew.tolist()[:2]}")

print("\n=== Episode 2 (reset after timeout) ===")
# Force a reset for next episode - timeout after episode_length_s=20s / decimation=10 = 200 steps
# Actually let's just call reset() which happens automatically on termination
obs, _ = env.reset()
print("After reset z:", env.scene['robot'].data.root_pos_w[:, 2].tolist())
print("root_lin_vel_b:", env.scene['robot'].data.root_lin_vel_b.tolist())
print("joint_pos:", env.scene['robot'].data.joint_pos[0].tolist())

for i in range(5):
    action = torch.zeros((4, 12), device="cuda")
    next_obs, rew, terminated, truncated, info = env.step(action)
    print(f"Step {i}: z={env.scene['robot'].data.root_pos_w[:, 2].tolist()}, term={terminated.tolist()}, rew={rew.tolist()[:2]}")

env.close()
