import sys
sys.path.insert(0, "/d/Projects/Quadruped")
import torch
from rl.isaac_lab.envs.go2_env_cfg import Go2BalanceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

cfg = Go2BalanceEnvCfg()
cfg.scene.num_envs = 4
cfg.device = "cuda"
env = ManagerBasedRLEnv(cfg)

obs, _ = env.reset()
print("After reset z:", env.scene['robot'].data.root_pos_w[:, 2].tolist())
print("root_lin_vel_b:", env.scene['robot'].data.root_lin_vel_b.tolist())

action = torch.zeros((4, 12), device="cuda")
next_obs, rew, terminated, truncated, info = env.step(action)

print("After step z:", env.scene['robot'].data.root_pos_w[:, 2].tolist())
print("terminated:", terminated.tolist())
print("truncated:", truncated.tolist())
print("episode_len:", env.episode_length_buf.tolist())
print("reward:", rew.tolist())
env.close()
