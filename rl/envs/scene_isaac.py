"""Isaac Lab scene configuration for Go2 robot.

Defines the InteractiveSceneCfg with:
- Flat terrain plane (physics material with friction)
- Go2 robot articulation (from isaaclab_assets.robots.unitree.UNITREE_GO2_CFG)
- Lights for rendering
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from rl.envs.constants_isaac import (
    JOINT_NAMES,
    FOOT_BODIES,
    BASE_BODY,
    HOME_BASE_HEIGHT,
    DEFAULT_JOINT_POS,
    KP,
    KV,
)

# Import the built-in Go2 config from isaaclab_assets
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    """Scene configuration for Go2 quadruped locomotion."""

    # Required by InteractiveSceneCfg
    env_spacing: float = 2.5

    # Terrain - flat ground plane with physics material
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        debug_vis=False,
    )

    # Go2 robot articulation - use the built-in config with minor overrides
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, HOME_BASE_HEIGHT),
            joint_pos={name: pos for name, pos in zip(JOINT_NAMES, DEFAULT_JOINT_POS)},
            joint_vel={".*": 0.0},
        ),
        # Override actuator gains to match contracts.py (KP=60, KV=3)
        actuators={
            "base_legs": UNITREE_GO2_CFG.actuators["base_legs"].replace(
                stiffness=KP,
                damping=KV,
            ),
        },
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


__all__ = ["Go2SceneCfg"]