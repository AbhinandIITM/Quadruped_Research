"""
go2_terrain_generator.py
========================
Procedural MuJoCo Terrain & Obstacle Generator for Unitree Go2.

Generates dynamic MuJoCo XML assets including:
  - Inclined slopes / ramps (5 to 20 degrees)
  - Multi-step staircases (step height 0.04 to 0.08 m)
  - Uneven box hurdles and rough terrain patches
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

# Terrain friction MUST match the friction coefficient the contact QP constrains
# against (QpBalanceConfig.mu in src/go2_balance_qp.py). If the generated terrain
# is more slip-resistant than the QP assumes, a "no friction pyramid violations"
# result is unfalsifiable: the solver would be enforcing a cone the simulator
# never lets it break. Keep this in sync with that config.
try:
    from go2_balance_qp import QpBalanceConfig as _QpBalanceConfig

    TERRAIN_MU = _QpBalanceConfig().mu
except Exception:  # pragma: no cover - fall back if the solver is unavailable
    TERRAIN_MU = 0.6

TERRAIN_FRICTION = f"{TERRAIN_MU} 0.005 0.0001"


def generate_stairs_xml(
    num_steps: int = 8,
    step_height: float = 0.05,
    step_depth: float = 0.28,
    stair_width: float = 1.6,
    start_x: float = 1.0,
    friction: str = TERRAIN_FRICTION,
) -> str:
    """Generates MuJoCo geoms for a straight staircase."""
    lines = ["<!-- Procedural Staircase -->"]
    for i in range(num_steps):
        x = start_x + (i + 0.5) * step_depth
        size_x = step_depth / 2.0
        size_y = stair_width / 2.0
        size_z = (i + 1) * step_height / 2.0
        geom_z = size_z
        lines.append(
            f'    <geom name="step_{i}" type="box" size="{size_x:.3f} {size_y:.3f} {size_z:.3f}" '
            f'pos="{x:.3f} 0.0 {geom_z:.3f}" rgba="0.45 0.50 0.55 1" friction="{friction}"/>'
        )
    return "\n".join(lines)


def generate_slope_xml(
    incline_deg: float = 12.0,
    ramp_length: float = 4.0,
    ramp_width: float = 1.8,
    start_x: float = 1.2,
    friction: str = TERRAIN_FRICTION,
) -> str:
    """Generates MuJoCo geoms for an inclined ramp."""
    angle_rad = math.radians(incline_deg)
    # Box tilted about Y axis
    # Pitch rotation about Y by -angle_rad tilts up along +X
    w = math.cos(-angle_rad / 2.0)
    y_rot = math.sin(-angle_rad / 2.0)
    quat = f"{w:.5f} 0.0 {y_rot:.5f} 0.0"

    thick = 0.05
    dx = (ramp_length / 2.0) * math.cos(angle_rad)
    dz = (ramp_length / 2.0) * math.sin(angle_rad)
    center_x = start_x + dx
    center_z = dz

    return (
        f'    <!-- Inclined Ramp ({incline_deg} deg) -->\n'
        f'    <geom name="ramp_incline" type="box" size="{ramp_length/2.0:.3f} {ramp_width/2.0:.3f} {thick:.3f}" '
        f'pos="{center_x:.3f} 0.0 {center_z:.3f}" quat="{quat}" rgba="0.50 0.55 0.60 1" friction="{friction}"/>'
    )


def build_terrain_world_xml(
    terrain_type: str = "stairs",
    output_path: Optional[Path] = None,
    **kwargs,
) -> str:
    """Builds a complete MuJoCo XML string with Go2 model and procedural terrain."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unitree_go2")) else os.getcwd()
    base_go2_xml_path = os.path.join(repo_root, "unitree_go2", "go2_description.xml")
    if not os.path.exists(base_go2_xml_path):
        base_go2_xml_path = "unitree_go2/go2_description.xml"
    if not os.path.exists(base_go2_xml_path):
        raise FileNotFoundError(f"Base Go2 description not found: {base_go2_xml_path}")

    base_xml = open(base_go2_xml_path, "r", encoding="utf-8").read()

    if terrain_type == "stairs":
        terrain_geoms = generate_stairs_xml(**kwargs)
    elif terrain_type == "slope":
        terrain_geoms = generate_slope_xml(**kwargs)
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")

    # Insert terrain geoms right before </worldbody>
    insertion_point = "</worldbody>"
    if insertion_point not in base_xml:
        raise ValueError("Could not locate </worldbody> in base Go2 description.")

    combined_xml = base_xml.replace(insertion_point, f"{terrain_geoms}\n  {insertion_point}")

    if output_path is not None:
        output_path = str(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_xml)

    return combined_xml
