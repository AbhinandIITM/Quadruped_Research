"""
go2_scene_obstacles.py
======================
Runtime injection of static obstacles into the Go2 MuJoCo scene.

The vendor MJCF (``unitree_go2/go2_description.xml``) is load-bearing: its
contents are quoted in the manuscript, so the scene must not be edited to add
geometry.  Instead this module loads the XML into a :class:`mujoco.MjSpec`,
appends the declared obstacle geoms to the *world* body, and compiles a
derived model.  The derived model is the vendor model plus N extra world
geoms -- robot bodies, the ``front_cam`` camera, actuators, and masses are
untouched -- so anything that reads the model by name (the controller, the
vision pipeline) keeps working unchanged.

Obstacles are attached directly to the world body, so they are:

* **static** -- no joint, no free joint, nothing to simulate;
* **collidable** -- ``contype = conaffinity = 1``, so the robot's feet and
  head can strike them;
* **rendered** -- ``group = 0`` so they appear in both the RGB and the depth
  pass of the offscreen renderer, which is what makes the camera path
  testable at all.

Two derived quantities are exported for the acceptance gate:

* :func:`obstacle_geom_distance` -- the signed distance between an obstacle
  geom and the robot's collidable geoms.  It is *signed*: it goes negative when
  the robot penetrates the obstacle, so "did the robot hit it" is a geometric
  question with a geometric answer rather than a self-report.
* :func:`obstacle_contact` -- whether the simulator reports an active contact
  between a robot geom and an obstacle geom.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

#: Obstacle primitives the scenario schema accepts.
OBSTACLE_TYPES = ("box", "cylinder")

#: Default visual material per primitive.  Bright and saturated so the obstacle
#: is unambiguous in an RGB render and so it is *not* mistakable for the floor
#: or the robot in a debug image.
DEFAULT_RGBA: Dict[str, Tuple[float, float, float, float]] = {
    "box": (0.85, 0.25, 0.12, 1.0),
    "cylinder": (0.15, 0.45, 0.90, 1.0),
}

#: Friction of the injected geometry.  Matched to the floor's tangential
#: friction so a robot that brushes an obstacle does not gain or lose grip
#: relative to the ground it is already standing on.
OBSTACLE_FRICTION = (0.6, 0.005, 0.0001)

#: Upper bound used for the signed-distance query.  Anything further away than
#: this is reported as "this far away", which is all the gate needs.
DISTANCE_QUERY_MAX_M = 10.0


class ObstacleSpecError(ValueError):
    """Raised when a scenario's obstacle declaration is malformed."""


@dataclass(frozen=True)
class ObstacleSpec:
    """One static obstacle, in world coordinates.

    ``pos`` is the (x, y) position of the obstacle's *base centre* on the
    floor, and ``size`` holds MuJoCo half-extents: ``(hx, hy, hz)`` for a box
    and ``(radius, half_height)`` for a cylinder.
    """

    kind: str
    pos: Tuple[float, float]
    size: Tuple[float, ...]
    yaw: float = 0.0
    rgba: Tuple[float, float, float, float] = (0.85, 0.25, 0.12, 1.0)
    name: str = ""

    @property
    def half_height(self) -> float:
        """Half-height of the obstacle along the world z axis."""
        return float(self.size[-1])

    @property
    def centre_z(self) -> float:
        """World z of the obstacle's centre, resting on the floor."""
        return self.half_height

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": self.kind,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "size": [float(s) for s in self.size],
            "yaw": float(self.yaw),
        }


def _as_floats(value: Any, label: str, count: Optional[int] = None) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ObstacleSpecError(f"{label} must be a list of numbers, got {type(value).__name__}")
    if count is not None and len(value) != count:
        raise ObstacleSpecError(f"{label} must have exactly {count} entries, got {len(value)}")
    out: List[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ObstacleSpecError(
                f"{label}[{i}] must be a number, got {type(item).__name__}"
            )
        f = float(item)
        if not math.isfinite(f):
            raise ObstacleSpecError(f"{label}[{i}] must be finite, got {f!r}")
        out.append(f)
    return tuple(out)


def parse_obstacle(entry: Any, index: int = 0) -> ObstacleSpec:
    """Parse one entry of a scenario's ``obstacles`` list.

    Raises :class:`ObstacleSpecError` with an actionable message, mirroring the
    scenario validator's style: a malformed obstacle must fail loudly at load
    time, not silently disappear from the scene.
    """
    label = f"obstacles[{index}]"
    if not isinstance(entry, dict):
        raise ObstacleSpecError(f"{label} must be an object, got {type(entry).__name__}")

    kind = entry.get("type", "box")
    if not isinstance(kind, str) or kind not in OBSTACLE_TYPES:
        raise ObstacleSpecError(
            f"{label}.type must be one of {list(OBSTACLE_TYPES)}, got {kind!r}"
        )

    if "pos" not in entry:
        raise ObstacleSpecError(f"{label}.pos is required but missing")
    pos = _as_floats(entry["pos"], f"{label}.pos")
    if len(pos) != 2:
        raise ObstacleSpecError(f"{label}.pos must have exactly 2 entries [x, y], got {len(pos)}")

    if "size" not in entry:
        raise ObstacleSpecError(f"{label}.size is required but missing")
    size = _as_floats(entry["size"], f"{label}.size")
    expected = 3 if kind == "box" else 2
    if len(size) != expected:
        raise ObstacleSpecError(
            f"{label}.size must have exactly {expected} entries for a {kind} "
            f"({'hx, hy, hz' if kind == 'box' else 'radius, half_height'}), got {len(size)}"
        )
    if any(s <= 0.0 for s in size):
        raise ObstacleSpecError(f"{label}.size entries must all be positive, got {list(size)}")

    yaw_raw = entry.get("yaw", 0.0)
    if isinstance(yaw_raw, bool) or not isinstance(yaw_raw, (int, float)):
        raise ObstacleSpecError(f"{label}.yaw must be a number, got {type(yaw_raw).__name__}")
    yaw = float(yaw_raw)
    if not math.isfinite(yaw):
        raise ObstacleSpecError(f"{label}.yaw must be finite, got {yaw!r}")

    rgba = DEFAULT_RGBA[kind]
    if "rgba" in entry:
        parsed = _as_floats(entry["rgba"], f"{label}.rgba")
        if len(parsed) != 4:
            raise ObstacleSpecError(
                f"{label}.rgba must have exactly 4 entries [r, g, b, a], got {len(parsed)}"
            )
        if any(c < 0.0 or c > 1.0 for c in parsed):
            raise ObstacleSpecError(
                f"{label}.rgba entries must lie in [0, 1], got {list(parsed)}"
            )
        rgba = parsed  # type: ignore[assignment]

    name = entry.get("name", f"obstacle_{index}")
    if not isinstance(name, str) or not name:
        raise ObstacleSpecError(f"{label}.name must be a non-empty string")
    if ".." in name or "/" in name or "\\" in name or " " in name:
        raise ObstacleSpecError(
            f"{label}.name must be a simple identifier (no spaces, path separators or '..'), got {name!r}"
        )

    return ObstacleSpec(
        kind=kind,
        pos=(float(pos[0]), float(pos[1])),
        size=tuple(float(s) for s in size),
        yaw=yaw,
        rgba=rgba,
        name=name,
    )


def parse_obstacles(entries: Any) -> List[ObstacleSpec]:
    """Parse a scenario's ``obstacles`` list into :class:`ObstacleSpec` objects."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ObstacleSpecError(f"obstacles must be a list, got {type(entries).__name__}")
    specs = [parse_obstacle(entry, i) for i, entry in enumerate(entries)]
    names = [s.name for s in specs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ObstacleSpecError(f"obstacle names must be unique, got duplicates {sorted(duplicates)}")
    return specs


def build_model_with_obstacles(
    model_path, obstacles: Sequence[ObstacleSpec]
) -> Tuple[mujoco.MjModel, List[int]]:
    """Compile the vendor model with the given obstacles added to the world.

    Returns the compiled model and the list of geom ids of the injected
    obstacles, in the order the specs were given.

    The vendor XML is read, never written.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    spec = mujoco.MjSpec.from_file(model_path)
    for obstacle in obstacles:
        geom = spec.worldbody.add_geom()
        geom.name = obstacle.name
        geom.type = (
            mujoco.mjtGeom.mjGEOM_BOX
            if obstacle.kind == "box"
            else mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        # MjGeom.size is a fixed 3-vector even for a cylinder, whose third
        # entry is ignored.  A plain list of the wrong length is rejected by
        # the bindings, so pad explicitly.
        size_vec = np.zeros(3, dtype=np.float64)
        for i, s in enumerate(obstacle.size[:3]):
            size_vec[i] = s
        geom.size = size_vec
        geom.pos = [obstacle.pos[0], obstacle.pos[1], obstacle.centre_z]
        # Yaw about the world z axis.
        geom.quat = [math.cos(obstacle.yaw / 2.0), 0.0, 0.0, math.sin(obstacle.yaw / 2.0)]
        geom.rgba = list(obstacle.rgba)
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 0
        geom.friction = list(OBSTACLE_FRICTION)
        geom.density = 0.0

    model = spec.compile()

    geom_ids: List[int] = []
    for obstacle in obstacles:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle.name)
        if gid < 0:
            raise RuntimeError(f"Obstacle geom {obstacle.name!r} vanished during compilation")
        geom_ids.append(gid)
    return model, geom_ids


def robot_collidable_geom_ids(model: mujoco.MjModel) -> List[int]:
    """Ids of the robot's own collidable geoms (everything not in the world body)."""
    ids: List[int] = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0:
            continue
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        ids.append(gid)
    return ids


def obstacle_geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obstacle_geom_ids: Sequence[int],
    robot_geom_ids: Sequence[int],
) -> float:
    """Smallest signed distance between any obstacle geom and any robot geom.

    Negative means the two are interpenetrating.  Returns ``inf`` when there
    are no obstacles to measure against.
    """
    if not obstacle_geom_ids:
        return float("inf")
    best = float("inf")
    for ogid in obstacle_geom_ids:
        for rgid in robot_geom_ids:
            dist = mujoco.mj_geomDistance(
                model, data, int(ogid), int(rgid), DISTANCE_QUERY_MAX_M, None
            )
            if dist < best:
                best = float(dist)
                if best < 0.0:
                    return best
    return best


def obstacle_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obstacle_geom_ids: Sequence[int],
) -> bool:
    """True when the simulator reports an active contact touching an obstacle."""
    if not obstacle_geom_ids:
        return False
    obstacle_set = set(int(g) for g in obstacle_geom_ids)
    for c_idx in range(data.ncon):
        contact = data.contact[c_idx]
        if int(contact.geom1) in obstacle_set or int(contact.geom2) in obstacle_set:
            return True
    return False


def line_lateral_deviation(
    point_xy: Tuple[float, float],
    start_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
) -> float:
    """Perpendicular distance from ``point_xy`` to the line start -> target.

    This is the "did the robot swerve" measure.  The reference line joins the
    robot's start position to the obstacle's centre, so a robot that drives
    straight at the obstacle has a deviation of about zero the whole way, while
    a robot that steers around it accumulates a growing deviation.  A robot
    that simply stops in front of the obstacle also has a deviation of about
    zero, which is exactly what we want: stopping is not avoiding.
    """
    ax, ay = float(start_xy[0]), float(start_xy[1])
    bx, by = float(target_xy[0]), float(target_xy[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        # Degenerate reference line (obstacle sits on the start point): fall
        # back to the raw distance to the target.
        return math.hypot(float(point_xy[0]) - bx, float(point_xy[1]) - by)
    # |cross product| / |direction| == perpendicular distance to the line.
    cross = abs(dx * (float(point_xy[1]) - ay) - dy * (float(point_xy[0]) - ax))
    return cross / length
