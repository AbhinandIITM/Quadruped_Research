import sys, torch, numpy as np, math, gc
sys.path.insert(0, r"D:\Projects\Quadruped")
from rl.isaac_lab.envs.go2_env_cfg import Go2StandEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


def trial(thigh, calf, z=0.34, steps=110, push=False):
    cfg = Go2StandEnvCfg()
    cfg.scene.robot.init_state.pos = (0.0, 0.0, z)
    jp = dict(cfg.scene.robot.init_state.joint_pos)
    for n in jp:
        if "thigh" in n:
            jp[n] = thigh
        elif "calf" in n:
            jp[n] = calf
    cfg.scene.robot.init_state.joint_pos = jp
    cfg.scene.num_envs = 1
    act = cfg.scene.robot.actuators["legs"]
    act.stiffness = 400.0
    act.damping = 8.0
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene.articulations["robot"]
    env.reset()
    zero = torch.zeros(1, 12, device="cuda")
    maxpitch = 0.0
    maxroll = 0.0
    last = None
    for i in range(steps):
        if push and i == 40:
            vel = torch.zeros(1, 6, device="cuda")
            vel[0, 0] = 0.3
            vel[0, 1] = -0.3
            robot.write_root_velocity_to_sim(vel)
        env.step(zero)
        q = robot.data.root_quat_w[0]
        w, x, y, zq = [float(v.item()) for v in q]
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - zq * x)))))
        roll = math.degrees(math.atan2(2 * (w * x + y * zq), 1 - 2 * (x * x + y * y)))
        zm = float(robot.data.root_pos_w[0, 2].item())
        maxpitch = max(maxpitch, abs(pitch))
        maxroll = max(maxroll, abs(roll))
        last = (pitch, roll, zm)
    print(
        f"thigh={thigh} calf={calf} push={push}: max|pitch|={maxpitch:.1f} max|roll|={maxroll:.1f} "
        f"final=(p={last[0]:.1f}, r={last[1]:.1f}, z={last[2]:.3f})"
    )
    del env
    gc.collect()
    torch.cuda.empty_cache()


for t, c in [(0.9, -1.8), (0.8, -1.7), (0.7, -1.6), (0.6, -1.5), (0.75, -1.65)]:
    trial(t, c, push=False)
print("--- with push ---")
for t, c in [(0.8, -1.7), (0.7, -1.6), (0.75, -1.65)]:
    trial(t, c, push=True)
