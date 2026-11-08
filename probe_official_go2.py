import sys
sys.path.insert(0, "/d/Projects/Quadruped")
import torch
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.mdp.events import reset_joints_by_scale, reset_root_state_uniform
from isaaclab.utils import configclass

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class TestEnvCfg(ManagerBasedRLEnv):
    scene: TestSceneCfg = TestSceneCfg(num_envs=4, env_spacing=2.5)
    sim = None
    decimation = 10
    episode_length_s = 20.0
    seed = 0
    
    @configclass
    class EventsCfg:
        reset_robot = EventTerm(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": "robot",
            },
        )
        reset_joints = EventTerm(
            func=reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": "robot",
            },
        )
    events: EventsCfg = EventsCfg()

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg:
            pass
        policy: PolicyCfg = PolicyCfg()

    observations: ObservationsCfg = ObservationsCfg()

    @configclass
    class ActionsCfg:
        pass
    actions: ActionsCfg = ActionsCfg()

    @configclass
    class CommandsCfg:
        pass
    commands: CommandsCfg = CommandsCfg()

    @configclass
    class RewardsCfg:
        pass
    rewards: RewardsCfg = RewardsCfg()

    @configclass
    class TerminationsCfg:
        pass
    terminations: TerminationsCfg = TerminationsCfg()

    @configclass
    class CurriculumCfg:
        pass
    curriculum: CurriculumCfg = CurriculumCfg()

env = ManagerBasedRLEnv(TestEnvCfg)

print("Before reset:")
print("  default_joint_pos:", env.scene['robot'].data.default_joint_pos[0].tolist())

obs, _ = env.reset()
print("\nAfter reset:")
print("  joint_pos:", env.scene['robot'].data.joint_pos[0].tolist())
print("  root_pos_w z:", env.scene['robot'].data.root_pos_w[:, 2].tolist())

for i in range(3):
    action = torch.zeros((4, 12), device="cuda")
    next_obs, rew, terminated, truncated, info = env.step(action)
    print(f"Step {i}: z={env.scene['robot'].data.root_pos_w[:, 2].tolist()}, term={terminated.tolist()}, rew={rew.tolist()[:2]}")

env.close()
