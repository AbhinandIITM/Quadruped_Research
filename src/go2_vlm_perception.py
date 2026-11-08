"""
go2_vlm_perception.py
======================
Vision-Language Model (VLM) Spatial Grounding & Multimodal Goal Reasoning
for the Unitree Go2 Quadruped in MuJoCo physics.

Capabilities:
  1. Capture synchronized RGB and metric depth frames from `<camera name="front_cam">`.
  2. Query Google Gemini 2.5 Flash Multimodal (Google AI Studio) via REST API with inline JPEG.
  3. Detect semantic target objects described in natural language (e.g. "red cylinder", "cone").
  4. Pinhole depth back-projection from 2D normalized bounding box [ymin, xmin, ymax, xmax]
     to 3D metric coordinates (X, Y, Z) in camera, world, and robot base body frames.
  5. Automatic synthesis of safe closed-loop 2D waypoint motion commands
     (MotionCommand with MotionMode.WAYPOINT) with clearance buffering.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
import numpy as np
from PIL import Image

from go2_llm_commands import (
    DEFAULT_GEMINI_API_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TIMEOUT_S,
    LLMCommandResult,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    Waypoint,
    _reject_nonfinite_json_constant,
    load_dotenv,
)

logger = logging.getLogger("go2_vlm")
load_dotenv()

#: Workspace bound for waypoint coordinates [m].  Matches the bound the action
#: contract's validator applies to LLM-supplied waypoints.  The VLM branch does
#: not pass through that validator, so the bound is applied here as well.
MAX_WAYPOINT_COORD_M = 10.0


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class BoundingBox2D:
    """Normalized bounding box coordinates."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float

    @classmethod
    def from_gemini_box(cls, box: List[Union[int, float]]) -> "BoundingBox2D":
        """Convert [ymin, xmin, ymax, xmax] from Gemini (either 0..1000 or 0..1)."""
        if len(box) != 4:
            raise ValueError(f"Expected 4 bounding box elements, got {len(box)}")
        y0, x0, y1, x1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # If coordinates are in 0..1000 scale, normalize to 0..1000 standard
        if max(y0, x0, y1, x1) <= 1.0:
            y0 *= 1000.0
            x0 *= 1000.0
            y1 *= 1000.0
            x1 *= 1000.0
        return cls(
            ymin=max(0.0, min(1000.0, y0)),
            xmin=max(0.0, min(1000.0, x0)),
            ymax=max(0.0, min(1000.0, y1)),
            xmax=max(0.0, min(1000.0, x1)),
        )

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized [0..1000] box to integer pixel coordinates (ymin, xmin, ymax, xmax)."""
        ymin_px = int(np.clip(self.ymin * height / 1000.0, 0, height - 1))
        xmin_px = int(np.clip(self.xmin * width / 1000.0, 0, width - 1))
        ymax_px = int(np.clip(self.ymax * height / 1000.0, ymin_px + 1, height))
        xmax_px = int(np.clip(self.xmax * width / 1000.0, xmin_px + 1, width))
        return ymin_px, xmin_px, ymax_px, xmax_px

    def center_pixel(self, width: int, height: int) -> Tuple[float, float]:
        """Returns (u_center, v_center) in pixel coordinates."""
        ymin_px, xmin_px, ymax_px, xmax_px = self.to_pixel_coords(width, height)
        return (xmin_px + xmax_px) / 2.0, (ymin_px + ymax_px) / 2.0


@dataclass
class Grounding3DTarget:
    """Fully grounded 3D target coordinates and navigation goals."""
    label: str
    box_2d: BoundingBox2D
    confidence: float
    camera_coords: Tuple[float, float, float]  # (X_cam, Y_cam, Z_cam)
    world_coords: Tuple[float, float, float]   # (X_world, Y_world, Z_world)
    body_coords: Tuple[float, float, float]    # (X_body, Y_body, Z_body)
    target_waypoint: Tuple[float, float]       # (X_world, Y_world) navigation waypoint
    depth_m: float
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "box_2d": [self.box_2d.ymin, self.box_2d.xmin, self.box_2d.ymax, self.box_2d.xmax],
            "depth_m": round(self.depth_m, 3),
            "body_coords": [round(c, 3) for c in self.body_coords],
            "world_coords": [round(c, 3) for c in self.world_coords],
            "target_waypoint": [round(c, 3) for c in self.target_waypoint],
        }


@dataclass
class VLMGroundingResult:
    """Outcome of VLM multimodal goal reasoning."""
    target_found: bool
    target: Optional[Grounding3DTarget]
    explanation: str
    motion_commands: List[MotionCommand]
    wall_time_s: float
    model_name: str
    raw_response: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_found": self.target_found,
            "target": self.target.to_dict() if self.target else None,
            "explanation": self.explanation,
            "model_name": self.model_name,
            "wall_time_s": round(self.wall_time_s, 3),
            "error_message": self.error_message,
            "commands": [c.to_dict() for c in self.motion_commands],
        }


# -----------------------------------------------------------------------------
# Core Sensing & Geometry Functions
# -----------------------------------------------------------------------------

def capture_rgb_and_depth(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str = "front_cam",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders synchronized RGB (uint8, HxWx3) and metric Depth (float32, HxW in meters)
    from the specified camera.
    """
    renderer.update_scene(data, camera=camera_name)

    # 1. Capture RGB frame
    renderer.disable_depth_rendering()
    rgb = renderer.render()

    # 2. Capture metric depth buffer
    renderer.enable_depth_rendering()
    depth = renderer.render()

    return rgb, depth


def encode_image_to_jpeg_base64(rgb: np.ndarray, quality: int = 85) -> str:
    """Encode RGB numpy array to base64 JPEG string for inline multimodal API transmission."""
    buf = io.BytesIO()
    img = Image.fromarray(rgb)
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _base_pose(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
    """Return the robot base (position, rotation matrix) in world coordinates.

    Falls back to the root free-joint pose when the model has no body named
    ``base``.  Shared by the back-projection and the waypoint construction so
    that both use exactly the same frame convention.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id >= 0:
        base_pos = np.array(data.xpos[base_id], dtype=np.float64)
        base_mat = np.array(data.xmat[base_id], dtype=np.float64).reshape(3, 3)
    else:
        # Fallback to root qpos
        base_pos = np.array(data.qpos[:3], dtype=np.float64)
        base_mat = np.eye(3, dtype=np.float64)
    return base_pos, base_mat


def backproject_2d_to_3d_body(
    u: float,
    v: float,
    depth_val: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "front_cam",
    image_width: int = 320,
    image_height: int = 240,
    fovy_deg: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Back-projects pixel coordinates (u, v) and metric depth z into 3D metric coordinates:
      - p_cam: Camera optical frame [X_cam (right), Y_cam (up), Z_cam (-forward)]
      - p_world: MuJoCo world frame
      - p_body: Robot base frame (forward +X, left +Y, up +Z)
    """
    # 1. Pinhole intrinsics calculation from field-of-view
    fovy_rad = math.radians(fovy_deg)
    fy = (image_height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy  # Square pixels
    cx = image_width / 2.0
    cy = image_height / 2.0

    # Ray in OpenGL camera frame:
    # +X_cam is right, +Y_cam is up, -Z_cam is optical forward direction
    p_cam = np.array([
        (u - cx) / fx * depth_val,
        -(v - cy) / fy * depth_val,
        -depth_val,
    ], dtype=np.float64)

    # 2. Camera pose from MuJoCo kinematics
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

    cam_pos = np.array(data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.array(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

    p_world = cam_pos + cam_mat @ p_cam

    # 3. Transform to Robot Base coordinate frame
    base_pos, base_mat = _base_pose(model, data)

    p_body = base_mat.T @ (p_world - base_pos)

    return p_cam, p_world, p_body


def ground_bounding_box(
    box: BoundingBox2D,
    depth_map: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    label: str = "target",
    confidence: float = 1.0,
    description: str = "",
    camera_name: str = "front_cam",
    clearance_m: float = 0.5,
    fovy_deg: float = 60.0,
) -> Grounding3DTarget:
    """
    Extracts median/surface depth from bounding box and projects to 3D body position.
    Also calculates safe navigation waypoint with clearance offset.
    """
    h, w = depth_map.shape
    ymin_px, xmin_px, ymax_px, xmax_px = box.to_pixel_coords(w, h)

    # Extract depth subregion
    sub_depth = depth_map[ymin_px:ymax_px, xmin_px:xmax_px]
    valid_depth = sub_depth[(sub_depth > 0.15) & (sub_depth < 12.0) & np.isfinite(sub_depth)]

    if len(valid_depth) > 0:
        # Use 20th percentile to sample the front surface of the target object
        z_val = float(np.percentile(valid_depth, 20))
    else:
        z_val = 2.0  # Fallback default nominal distance

    u_center, v_center = box.center_pixel(w, h)

    p_cam, p_world, p_body = backproject_2d_to_3d_body(
        u=u_center,
        v=v_center,
        depth_val=z_val,
        model=model,
        data=data,
        camera_name=camera_name,
        image_width=w,
        image_height=h,
        fovy_deg=fovy_deg,
    )

    # Compute planar waypoint (offset back by clearance_m along line-of-sight).
    #
    # The back-off is computed in the robot BODY frame, where the line of sight
    # is simply the target's body coordinates, and then rotated into the WORLD
    # frame.  The waypoint has to be world-frame: MotionCommand.waypoint is
    # consumed by WaypointNavigator as a world coordinate and compared against
    # the robot's world position.  Expressing it in the body frame worked only
    # because the robot started at the origin with zero yaw.
    x_body, y_body = float(p_body[0]), float(p_body[1])
    dist_2d = math.hypot(x_body, y_body)

    if dist_2d > clearance_m:
        scale = (dist_2d - clearance_m) / dist_2d
        x_waypoint_body = x_body * scale
        y_waypoint_body = y_body * scale
    else:
        x_waypoint_body = 0.0
        y_waypoint_body = 0.0

    base_pos, base_mat = _base_pose(model, data)
    waypoint_world = base_pos + base_mat @ np.array(
        [x_waypoint_body, y_waypoint_body, 0.0], dtype=np.float64
    )
    x_waypoint = float(waypoint_world[0])
    y_waypoint = float(waypoint_world[1])

    # The waypoint bypasses the shared action-contract validator (the VLM branch
    # is a separate code path), so the workspace bound and finiteness that the
    # validator would enforce are applied here instead.
    if not (math.isfinite(x_waypoint) and math.isfinite(y_waypoint)):
        x_waypoint = 0.0
        y_waypoint = 0.0
    x_waypoint = float(np.clip(x_waypoint, -MAX_WAYPOINT_COORD_M, MAX_WAYPOINT_COORD_M))
    y_waypoint = float(np.clip(y_waypoint, -MAX_WAYPOINT_COORD_M, MAX_WAYPOINT_COORD_M))

    return Grounding3DTarget(
        label=label,
        box_2d=box,
        confidence=confidence,
        camera_coords=(float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
        world_coords=(float(p_world[0]), float(p_world[1]), float(p_world[2])),
        body_coords=(x_body, y_body, float(p_body[2])),
        target_waypoint=(x_waypoint, y_waypoint),
        depth_m=z_val,
        description=description,
    )


# -----------------------------------------------------------------------------
# VLM Goal Grounder Class
# -----------------------------------------------------------------------------

class VLMGoalGrounder:
    """Multimodal Vision-Language Model Grounding Agent using Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        bounds: Optional[SafetyBounds] = None,
    ):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or ""
            ).strip()
        self.model = model
        self.timeout_s = timeout_s
        self.bounds = bounds or SafetyBounds()

    def check_availability(self) -> Tuple[bool, str]:
        """Check if Gemini API key is configured."""
        if not self.api_key:
            return (
                False,
                "GEMINI_API_KEY not configured. Set GEMINI_API_KEY in .env or pass --gemini-api-key.",
            )
        return True, f"Gemini API key detected. Model: '{self.model}'."

    def ground_instruction(
        self,
        renderer: mujoco.Renderer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        instruction: str,
        camera_name: str = "front_cam",
        clearance_m: float = 0.5,
        target_speed: float = 0.35,
    ) -> VLMGroundingResult:
        """
        Renders the camera view, sends to Gemini 2.5 Flash Multimodal,
        parses the bounding box, back-projects to 3D, and produces safe waypoint motion commands.
        """
        t_start = time.perf_counter()

        if not self.api_key:
            return VLMGroundingResult(
                target_found=False,
                target=None,
                explanation="Missing GEMINI_API_KEY. VLM multimodal grounding requires an active Gemini API key.",
                motion_commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                wall_time_s=0.0,
                model_name=self.model,
                raw_response="",
                error_message="Missing GEMINI_API_KEY",
            )

        # 1. Capture RGB and Depth
        rgb, depth = capture_rgb_and_depth(renderer, data, camera_name=camera_name)
        b64_img = encode_image_to_jpeg_base64(rgb)

        # 2. Formulate Prompt with Strict JSON Schema
        prompt_text = (
            f"You are the visual grounding system for an autonomous Unitree Go2 quadruped robot.\n"
            f"User Goal: \"{instruction}\"\n\n"
            "Analyze the camera image captured from the robot's front camera.\n"
            "Locate the target object/landmark mentioned in the user's goal.\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "target_found": true,\n'
            '  "target_label": "name of target object",\n'
            '  "box_2d": [ymin, xmin, ymax, xmax],\n'
            '  "confidence": 0.95,\n'
            '  "description": "concise visual observation of target location"\n'
            "}\n"
            "Coordinates in box_2d MUST be normalized integers from 0 to 1000 [ymin, xmin, ymax, xmax].\n"
            "If the requested target is not visible, set target_found: false, box_2d: null, confidence: 0.0."
        )

        models_to_try = [self.model, "gemini-2.5-flash", "gemini-flash-latest"]
        last_error = ""
        parsed_json: Optional[Dict[str, Any]] = None
        raw_text = ""
        active_model = self.model

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}},
                    {"text": prompt_text},
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        req_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Go2-VLM-Grounding/1.0",
            # Credentials belong in a header, never in the URL query string (proxies log URLs).
            "x-goog-api-key": self.api_key,
        }

        for model_name in models_to_try:
            url = f"{DEFAULT_GEMINI_API_URL}/models/{model_name}:generateContent"
            for attempt in range(2):
                try:
                    req = urllib.request.Request(url, data=req_body, headers=headers)
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        res = json.loads(
                            resp.read().decode("utf-8"),
                            parse_constant=_reject_nonfinite_json_constant,
                        )
                        active_model = model_name
                        candidates = res.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            raw_text = "".join(p.get("text", "") for p in parts)
                            # parse_constant rejects the non-standard NaN /
                            # Infinity tokens that json.loads accepts by default.
                            # A bounding-box coordinate that is not finite would
                            # otherwise be clamped into a plausible-looking value
                            # and silently point the robot at the wrong place.
                            parsed_json = json.loads(
                                raw_text,
                                parse_constant=_reject_nonfinite_json_constant,
                            )
                            break
                except urllib.error.HTTPError as err:
                    err_msg = err.read().decode("utf-8", errors="replace") if hasattr(err, "read") else str(err)
                    last_error = f"HTTP {err.code}: {err_msg}"
                    time.sleep(1.0)
                except Exception as err:
                    last_error = str(err)
                    time.sleep(1.0)
            if parsed_json is not None:
                break

        t_elapsed = time.perf_counter() - t_start

        if parsed_json is None:
            return VLMGroundingResult(
                target_found=False,
                target=None,
                explanation=f"VLM grounding failed to query Gemini ({last_error}). Falling closed to safe stand.",
                motion_commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                wall_time_s=t_elapsed,
                model_name=active_model,
                raw_response=raw_text,
                error_message=last_error,
            )

        target_found = bool(parsed_json.get("target_found", False))
        box_data = parsed_json.get("box_2d")

        if not target_found or not box_data:
            desc = parsed_json.get("description", "Target object not detected in camera frame.")
            return VLMGroundingResult(
                target_found=False,
                target=None,
                explanation=f"VLM scanned scene: {desc}",
                motion_commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                wall_time_s=t_elapsed,
                model_name=active_model,
                raw_response=raw_text,
            )

        # 3. Ground 2D Bounding Box to 3D Coordinates
        try:
            box_2d = BoundingBox2D.from_gemini_box(box_data)
            label = str(parsed_json.get("target_label", "target"))
            conf = float(parsed_json.get("confidence", 0.9))
            desc = str(parsed_json.get("description", ""))

            grounded_target = ground_bounding_box(
                box=box_2d,
                depth_map=depth,
                model=model,
                data=data,
                label=label,
                confidence=conf,
                description=desc,
                camera_name=camera_name,
                clearance_m=clearance_m,
            )
        except Exception as e:
            return VLMGroundingResult(
                target_found=False,
                target=None,
                explanation=f"Error back-projecting target bounding box: {e}",
                motion_commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                wall_time_s=t_elapsed,
                model_name=active_model,
                raw_response=raw_text,
                error_message=str(e),
            )

        # 4. Generate Waypoint Navigation Commands
        x_nav, y_nav = grounded_target.target_waypoint
        nav_dist = math.hypot(x_nav, y_nav)
        nav_duration = max(1.5, min(10.0, (nav_dist / max(0.1, target_speed)) + 1.0))

        waypoint_cmd = MotionCommand(
            mode=MotionMode.WAYPOINT,
            vx=min(0.5, target_speed),
            vy=0.0,
            yaw_rate=0.0,
            duration=nav_duration,
            waypoint=Waypoint(x=x_nav, y=y_nav, speed=target_speed),
        )
        stand_cmd = MotionCommand(
            mode=MotionMode.STAND,
            duration=1.0,
        )

        explanation = (
            f"VLM Grounding: Located '{grounded_target.label}' at "
            f"X={grounded_target.body_coords[0]:.2f}m, Y={grounded_target.body_coords[1]:.2f}m. "
            f"Navigating to waypoint ({x_nav:.2f}, {y_nav:.2f})m with {clearance_m:.2f}m clearance buffer."
        )

        return VLMGroundingResult(
            target_found=True,
            target=grounded_target,
            explanation=explanation,
            motion_commands=[waypoint_cmd, stand_cmd],
            wall_time_s=t_elapsed,
            model_name=active_model,
            raw_response=raw_text,
        )

    def offline_ground_instruction(
        self,
        renderer: mujoco.Renderer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        target_box: BoundingBox2D,
        label: str = "synthetic_target",
        camera_name: str = "front_cam",
        clearance_m: float = 0.5,
        target_speed: float = 0.35,
    ) -> VLMGroundingResult:
        """
        Deterministic offline grounding for automated tests without network access.
        """
        t_start = time.perf_counter()
        rgb, depth = capture_rgb_and_depth(renderer, data, camera_name=camera_name)

        grounded_target = ground_bounding_box(
            box=target_box,
            depth_map=depth,
            model=model,
            data=data,
            label=label,
            confidence=1.0,
            description="Synthetic offline target detection.",
            camera_name=camera_name,
            clearance_m=clearance_m,
        )

        x_nav, y_nav = grounded_target.target_waypoint
        nav_dist = math.hypot(x_nav, y_nav)
        nav_duration = max(1.5, min(10.0, (nav_dist / max(0.1, target_speed)) + 1.0))

        waypoint_cmd = MotionCommand(
            mode=MotionMode.WAYPOINT,
            vx=min(0.5, target_speed),
            vy=0.0,
            yaw_rate=0.0,
            duration=nav_duration,
            waypoint=Waypoint(x=x_nav, y=y_nav, speed=target_speed),
        )
        stand_cmd = MotionCommand(mode=MotionMode.STAND, duration=1.0)

        explanation = (
            f"Offline Grounding: Located '{label}' at "
            f"X={grounded_target.body_coords[0]:.2f}m, Y={grounded_target.body_coords[1]:.2f}m."
        )

        return VLMGroundingResult(
            target_found=True,
            target=grounded_target,
            explanation=explanation,
            motion_commands=[waypoint_cmd, stand_cmd],
            wall_time_s=time.perf_counter() - t_start,
            model_name="offline-grounder",
            raw_response="{}",
        )


def create_vlm_scene_xml(
    target_pos: Tuple[float, float, float] = (2.0, 0.3, 0.4),
    target_geom_type: str = "cylinder",
    target_size: str = "0.15 0.4",
    target_rgba: str = "0.9 0.1 0.1 1",
    target_name: str = "target_cylinder",
) -> str:
    """
    Generate a MuJoCo XML string combining the Go2 quadruped robot
    and an interactive benchmark target object for visual grounding tests.
    """
    return f"""<mujoco model="go2_vlm_scene">
  <include file="unitree_go2/go2_description.xml"/>
  <worldbody>
    <geom name="{target_name}" type="{target_geom_type}" size="{target_size}" pos="{target_pos[0]} {target_pos[1]} {target_pos[2]}" rgba="{target_rgba}"/>
  </worldbody>
</mujoco>"""

