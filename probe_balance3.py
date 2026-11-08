import sys
sys.path.insert(0, "/d/Projects/Quadruped")
import torch
from rl.isaac_lab.envs.go2_env_cfg import Go2BalanceEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

cfg = Go2BalanceEnvCfg()
cfg.scene.num_envs = 4
cfg.device = "cuda"
env = ManagerBasedRLEnv(cfg)

# Before reset, check the init_state
print("Scene robot init_state:", env.scene['robot'].cfg.init_state)
print("Scene robot init_state.joint_pos:", env.scene['robot'].cfg.init_state.joint_pos)

# After reset
obs, _ = env.reset()
print("\nAfter reset:")
print("  root_pos_w[:,2]:", env.scene['robot'].data.root_pos_w[:, 2].tolist())
print("  joint_pos[0]:", env.scene['robot'].data.joint_pos[0].tolist())
print("  joint_vel[0]:", env.scene['robot'].data.joint_vel[0].tolist())

# Check joint names
print("  joint_names:", env.scene['robot'].joint_names)

# What's the target joint position (PD target)?
print("  default_joint_pos:", env.scene['robot'].data.default_joint_pos[0].tolist())

env.close()
