"""
go2_vision_perception.py
========================
Closed-loop vision, depth sensing, and reactive obstacle avoidance
for the Unitree Go2 in MuJoCo physics.

Features:
  - Metric depth map extraction from MuJoCo Renderer via `front_cam`.
  - Tri-sector clearance analysis (Left, Center, Right) with ground-plane filtering.
  - Potential-field / reactive steering deflection to bypass obstacles.
  - Fail-safe fallback when renderer or camera is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import mujoco

logger = logging.getLogger("go2_vision")


@dataclass(frozen=True)
class DepthPerceptionConfig:
    image_width: int = 160
    image_height: int = 120
    camera_name: str = "front_cam"
    min_range_m: float = 0.15
    max_range_m: float = 6.0
    danger_distance_m: float = 0.70    # Critical proximity: halt forward translation
    warning_distance_m: float = 1.60   # Proximity threshold: initiate deceleration & yaw deflection
    ground_cutoff_row: int = 55        # Ignore lower image rows seeing flat floor ahead
    sector_split_left: float = 0.33    # Left column fraction (0 .. 33%)
    sector_split_right: float = 0.67   # Right column fraction (67% .. 100%)
    avoidance_yaw_gain: float = 0.85   # Deflection gain (rad/s)
    min_speed_scale: float = 0.15      # Floor speed scaling factor under active evasion


@dataclass(frozen=True)
class SectorClearance:
    left_dist: float
    center_dist: float
    right_dist: float
    min_dist: float
    closest_sector: str
    obstacle_detected: bool
    is_critical: bool


class DepthPerceptionPipeline:
    """Manages offscreen rendering and metric depth analysis."""

    def __init__(self, model: mujoco.MjModel, config: Optional[DepthPerceptionConfig] = None):
        self.model = model
        self.config = config or DepthPerceptionConfig()
        self._renderer: Optional[mujoco.Renderer] = None
        self._has_camera = False

        # Verify camera exists in model
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name)
        if cam_id >= 0:
            self._has_camera = True
            try:
                self._renderer = mujoco.Renderer(
                    model, height=self.config.image_height, width=self.config.image_width
                )
            except Exception as e:
                logger.warning("Failed to initialize MuJoCo Renderer for vision: %s", e)
                self._renderer = None
        else:
            logger.warning("Camera '%s' not found in model.", self.config.camera_name)

    @property
    def is_available(self) -> bool:
        return self._renderer is not None and self._has_camera

    def capture_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Captures a metric depth array (H, W) in meters."""
        if not self.is_available or self._renderer is None:
            # Fallback: infinite clearance
            return np.full((self.config.image_height, self.config.image_width), self.config.max_range_m)

        self._renderer.update_scene(data, camera=self.config.camera_name)
        self._renderer.enable_depth_rendering()
        depth_metric = self._renderer.render()
        # Ensure clipping within configured range
        return np.clip(depth_metric, self.config.min_range_m, self.config.max_range_m)

    def analyze_clearance(self, depth_map: np.ndarray) -> SectorClearance:
        """Computes obstacle distances across Left, Center, and Right sectors."""
        cfg = self.config
        h, w = depth_map.shape

        # Filter out bottom rows seeing floor
        valid_rows = min(cfg.ground_cutoff_row, h)
        roi = depth_map[:valid_rows, :]

        col_left_end = int(w * cfg.sector_split_left)
        col_right_start = int(w * cfg.sector_split_right)

        left_sector = roi[:, :col_left_end]
        center_sector = roi[:, col_left_end:col_right_start]
        right_sector = roi[:, col_right_start:]

        # Use 5th percentile to reject single-pixel depth artifacts
        left_dist = float(np.percentile(left_sector, 5)) if left_sector.size > 0 else cfg.max_range_m
        center_dist = float(np.percentile(center_sector, 5)) if center_sector.size > 0 else cfg.max_range_m
        right_dist = float(np.percentile(right_sector, 5)) if right_sector.size > 0 else cfg.max_range_m

        dists = {"left": left_dist, "center": center_dist, "right": right_dist}
        closest_sector = min(dists, key=dists.get)
        min_dist = dists[closest_sector]

        obstacle_detected = min_dist < cfg.warning_distance_m
        is_critical = min_dist <= cfg.danger_distance_m

        return SectorClearance(
            left_dist=left_dist,
            center_dist=center_dist,
            right_dist=right_dist,
            min_dist=min_dist,
            closest_sector=closest_sector,
            obstacle_detected=obstacle_detected,
            is_critical=is_critical,
        )


class ReactiveObstacleAvoidance:
    """Modulates commanded velocities to avoid frontal obstacles."""

    def __init__(self, config: Optional[DepthPerceptionConfig] = None):
        self.config = config or DepthPerceptionConfig()

    def modulate_commands(
        self,
        cmd_vx: float,
        cmd_vy: float,
        cmd_yaw_rate: float,
        clearance: SectorClearance,
    ) -> Tuple[float, float, float, str]:
        """
        Adjusts linear and yaw velocity targets based on sectoral obstacle proximity.

        Returns:
            (safe_vx, safe_vy, safe_yaw_rate, status_message)
        """
        cfg = self.config

        # If only moving backward or standing, or no obstacle ahead: pass through
        if cmd_vx <= 0.0 or not clearance.obstacle_detected:
            return cmd_vx, cmd_vy, cmd_yaw_rate, "clear"

        # Calculate proximity severity: 0.0 at warning boundary, 1.0 at danger threshold
        severity = np.clip(
            (cfg.warning_distance_m - clearance.min_dist) / (cfg.warning_distance_m - cfg.danger_distance_m),
            0.0,
            1.0,
        )

        # Critical proximity: halt forward motion and turn away
        if clearance.is_critical:
            safe_vx = 0.0
            safe_vy = 0.0
            # Steer toward the clearer side
            if clearance.left_dist > clearance.right_dist:
                safe_yaw = +cfg.avoidance_yaw_gain  # Turn left
                status = f"CRITICAL_HALT: obstacle at {clearance.min_dist:.2f}m in {clearance.closest_sector}; turning left"
            else:
                safe_yaw = -cfg.avoidance_yaw_gain  # Turn right
                status = f"CRITICAL_HALT: obstacle at {clearance.min_dist:.2f}m in {clearance.closest_sector}; turning right"
            return safe_vx, safe_vy, safe_yaw, status

        # Warning zone: scale forward speed and inject yaw deflection
        speed_scale = max(cfg.min_speed_scale, 1.0 - 0.75 * severity)
        safe_vx = cmd_vx * speed_scale
        safe_vy = cmd_vy * speed_scale

        # Determine deflection direction
        if clearance.closest_sector == "center":
            # Obstacle directly ahead: bias toward side with greater clearance
            if clearance.left_dist >= clearance.right_dist:
                yaw_deflection = +cfg.avoidance_yaw_gain * severity  # Turn left
            else:
                yaw_deflection = -cfg.avoidance_yaw_gain * severity  # Turn right
        elif clearance.closest_sector == "left":
            # Obstacle to the left: turn right
            yaw_deflection = -cfg.avoidance_yaw_gain * severity
        else:
            # Obstacle to the right: turn left
            yaw_deflection = +cfg.avoidance_yaw_gain * severity

        safe_yaw = cmd_yaw_rate + yaw_deflection
        # Clamp yaw rate to reasonable bound
        safe_yaw = float(np.clip(safe_yaw, -1.2, 1.2))

        status = f"AVOIDING: min_dist={clearance.min_dist:.2f}m ({clearance.closest_sector}), vx_scale={speed_scale:.2f}"
        return safe_vx, safe_vy, safe_yaw, status
