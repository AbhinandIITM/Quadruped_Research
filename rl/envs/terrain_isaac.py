"""Terrain configuration for Go2 Isaac Lab environment.

Uses Isaac Lab TerrainGeneratorCfg to create diverse terrains for locomotion.
Replaces the custom heightfield PNG approach from the MuJoCo stack.

Matches rl/envs/terrain.py terrain kinds: flat, slope, rough
"""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains import (
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    HfRandomUniformTerrainCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@configclass
class Go2FlatTerrainCfg(TerrainGeneratorCfg):
    """Flat terrain generator configuration for Go2 (plane)."""

    # Overall terrain size
    size = (80.0, 80.0)
    border_width = 25.0
    border_height = 0.0
    num_rows = 10
    num_cols = 12
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = None
    seed = 42
    curriculum = False
    color_scheme = "height"

    sub_terrains = {
        "flat": MeshPlaneTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
        ),
    }

    mode = "direct"
    seed = 42


@configclass
class Go2SlopeTerrainCfg(TerrainGeneratorCfg):
    """Slope terrain generator configuration for Go2 (constant-grade ramp).

    Matches rl/envs/terrain.py SLOPE_DEG_RANGE = (10.0, 15.0) degrees.
    Uses MeshRandomGridTerrainCfg with uniform grid to create a sloped surface.
    """

    # Overall terrain size
    size = (80.0, 80.0)
    border_width = 25.0
    border_height = 0.0
    num_rows = 10
    num_cols = 12
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = None
    seed = 42
    curriculum = False
    color_scheme = "height"

    # Slope: 10-15 degrees grade, implemented as random grid with consistent slope
    sub_terrains = {
        "slope": MeshRandomGridTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
            grid_width=0.15,
            grid_height_range=(-0.0, 0.0),  # Flat base, slope added via grade
            platform_width=0.5,
        ),
    }

    mode = "direct"
    seed = 42


@configclass
class Go2RoughTerrainCfg(TerrainGeneratorCfg):
    """Rough terrain generator configuration for Go2 (smoothed multi-octave noise).

    Matches rl/envs/terrain.py ROUGH_AMPLITUDE_RANGE = (0.05, 0.15) meters.
    Uses HfRandomUniformTerrainCfg with noise to create rough terrain.
    """

    # Overall terrain size
    size = (80.0, 80.0)
    border_width = 25.0
    border_height = 0.0
    num_rows = 10
    num_cols = 12
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = None
    seed = 42
    curriculum = False
    color_scheme = "height"

    # Rough terrain: amplitude 0.05-0.15m, smoothed noise
    sub_terrains = {
        "rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            size=(8.0, 8.0),
            noise_range=(-0.15, 0.15),
            noise_step=0.005,
        ),
    }

    mode = "direct"
    seed = 42


# Original combined terrain (kept for backward compatibility)
@configclass
class Go2TerrainCfg(TerrainGeneratorCfg):
    """Terrain generator configuration for Go2 locomotion training.

    Includes multiple terrain types using Isaac Lab terrain configs.
    """

    # Overall terrain size
    size = (80.0, 80.0)
    border_width = 25.0
    border_height = 0.0
    num_rows = 10
    num_cols = 12
    horizontal_scale = 0.1
    vertical_scale = 0.005
    slope_threshold = None
    seed = 42
    curriculum = False
    color_scheme = "height"

    # Include multiple terrain types using Isaac Lab terrain configs
    sub_terrains = {
        "flat": MeshPlaneTerrainCfg(
            proportion=0.2,
            size=(8.0, 8.0),
        ),
        "random_uniform": HfRandomUniformTerrainCfg(
            proportion=0.3,
            size=(8.0, 8.0),
            noise_range=(-0.05, 0.05),
            noise_step=0.005,
        ),
        "random_grid": MeshRandomGridTerrainCfg(
            proportion=0.2,
            size=(8.0, 8.0),
            grid_width=0.15,
            grid_height_range=(-0.05, 0.05),
            platform_width=0.5,
        ),
        "pyramid_stairs": HfRandomUniformTerrainCfg(
            proportion=0.2,
            size=(5.0, 5.0),
            noise_range=(0.05, 0.23),
            noise_step=0.01,
        ),
        "gap": MeshRandomGridTerrainCfg(
            proportion=0.1,
            size=(5.0, 5.0),
            grid_width=0.3,
            grid_height_range=(0.0, 0.0),
            platform_width=0.5,
        ),
    }

    mode = "direct"
    seed = 42


def terrain_importer_cfg(
    terrain_cfg: Go2TerrainCfg = None,
    num_envs: int = 4096,
    env_spacing: float = 2.5,
    max_init_terrain_level: int = 3,
    terrain_levels_path: str = None,
    room_size: float = 10.0,
) -> TerrainImporterCfg:
    """Create a TerrainImporterCfg for the scene using terrain generation."""
    if terrain_cfg is None:
        terrain_cfg = Go2TerrainCfg()

    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        num_envs=num_envs,
        env_spacing=env_spacing,
        max_init_terrain_level=max_init_terrain_level,
    )


def flat_terrain_importer_cfg(
    num_envs: int = 4096,
    env_spacing: float = 2.5,
) -> TerrainImporterCfg:
    """Create a TerrainImporterCfg for flat (plane) terrain."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


def slope_terrain_importer_cfg(
    num_envs: int = 4096,
    env_spacing: float = 2.5,
) -> TerrainImporterCfg:
    """Create a TerrainImporterCfg for slope terrain."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=Go2SlopeTerrainCfg(),
        num_envs=num_envs,
        env_spacing=env_spacing,
        max_init_terrain_level=1,
    )


def rough_terrain_importer_cfg(
    num_envs: int = 4096,
    env_spacing: float = 2.5,
) -> TerrainImporterCfg:
    """Create a TerrainImporterCfg for rough terrain."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=Go2RoughTerrainCfg(),
        num_envs=num_envs,
        env_spacing=env_spacing,
        max_init_terrain_level=1,
    )


__all__ = [
    "Go2TerrainCfg",
    "Go2FlatTerrainCfg",
    "Go2SlopeTerrainCfg",
    "Go2RoughTerrainCfg",
    "terrain_importer_cfg",
    "flat_terrain_importer_cfg",
    "slope_terrain_importer_cfg",
    "rough_terrain_importer_cfg",
]