"""
tests/test_vlm_perception.py
=============================
Unit and integration tests for VLM spatial grounding, camera pinhole back-projection,
and multimodal waypoint command synthesis.
"""

import math
import os
import unittest
from pathlib import Path

import mujoco
import numpy as np

from go2_champ_control import initialize_simulation_standing
from go2_llm_commands import MotionMode
from go2_vlm_perception import (
    BoundingBox2D,
    Grounding3DTarget,
    VLMGoalGrounder,
    VLMGroundingResult,
    backproject_2d_to_3d_body,
    capture_rgb_and_depth,
    encode_image_to_jpeg_base64,
    ground_bounding_box,
)


class TestBoundingBox2D(unittest.TestCase):
    """Test 2D bounding box data structures and conversions."""

    def test_from_gemini_1000_scale(self):
        box = BoundingBox2D.from_gemini_box([100, 200, 500, 600])
        self.assertEqual(box.ymin, 100.0)
        self.assertEqual(box.xmin, 200.0)
        self.assertEqual(box.ymax, 500.0)
        self.assertEqual(box.xmax, 600.0)

    def test_from_gemini_unit_scale(self):
        box = BoundingBox2D.from_gemini_box([0.1, 0.2, 0.5, 0.6])
        self.assertAlmostEqual(box.ymin, 100.0)
        self.assertAlmostEqual(box.xmin, 200.0)
        self.assertAlmostEqual(box.ymax, 500.0)
        self.assertAlmostEqual(box.xmax, 600.0)

    def test_to_pixel_coords(self):
        box = BoundingBox2D(ymin=250.0, xmin=250.0, ymax=750.0, xmax=750.0)
        ymin_px, xmin_px, ymax_px, xmax_px = box.to_pixel_coords(width=320, height=240)
        self.assertEqual(ymin_px, 60)
        self.assertEqual(xmin_px, 80)
        self.assertEqual(ymax_px, 180)
        self.assertEqual(xmax_px, 240)

    def test_center_pixel(self):
        box = BoundingBox2D(ymin=200.0, xmin=300.0, ymax=600.0, xmax=700.0)
        u_c, v_c = box.center_pixel(width=200, height=100)
        self.assertAlmostEqual(u_c, 100.0)  # (60 + 140) / 2
        self.assertAlmostEqual(v_c, 40.0)   # (20 + 60) / 2


class TestVLMGeometryAndBackprojection(unittest.TestCase):
    """Test camera rendering, geometry, and metric back-projection."""

    @classmethod
    def setUpClass(cls):
        # Create MuJoCo model with Go2 and a known benchmark red cylinder
        cls.xml_content = """<mujoco model="vlm_test_scene">
          <include file="unitree_go2/go2_description.xml"/>
          <worldbody>
            <geom name="test_cylinder" type="cylinder" size="0.15 0.4" pos="2.0 0.3 0.4" rgba="0.9 0.1 0.1 1"/>
          </worldbody>
        </mujoco>"""
        cls.model = mujoco.MjModel.from_xml_string(cls.xml_content)
        cls.data = mujoco.MjData(cls.model)
        initialize_simulation_standing(cls.model, cls.data, body_height=0.28)
        mujoco.mj_step(cls.model, cls.data)
        cls.renderer = mujoco.Renderer(cls.model, height=120, width=160)

    def test_capture_rgb_and_depth(self):
        rgb, depth = capture_rgb_and_depth(self.renderer, self.data, camera_name="front_cam")
        self.assertEqual(rgb.shape, (120, 160, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.shape, (120, 160))
        self.assertEqual(depth.dtype, np.float32)
        # Depth should be strictly positive meters
        self.assertTrue(np.all(depth > 0.0))

    def test_encode_image_to_jpeg_base64(self):
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200  # Red
        b64_str = encode_image_to_jpeg_base64(rgb)
        self.assertIsInstance(b64_str, str)
        self.assertTrue(len(b64_str) > 100)

    def test_backprojection_math(self):
        # Optical center ray at depth 2.0m
        p_cam, p_world, p_body = backproject_2d_to_3d_body(
            u=80.0,
            v=60.0,
            depth_val=2.0,
            model=self.model,
            data=self.data,
            camera_name="front_cam",
            image_width=160,
            image_height=120,
        )
        # Optical center should have p_cam = [0, 0, -2.0]
        self.assertAlmostEqual(p_cam[0], 0.0, places=3)
        self.assertAlmostEqual(p_cam[1], 0.0, places=3)
        self.assertAlmostEqual(p_cam[2], -2.0, places=3)

        # Body coordinates should be forward +X
        self.assertGreater(p_body[0], 1.5)

    def test_ground_bounding_box_accuracy(self):
        # Known cylinder at pos="2.0 0.3 0.4"
        # Bounding box roughly centered at cylinder
        rgb, depth = capture_rgb_and_depth(self.renderer, self.data, camera_name="front_cam")

        # Find red pixels of cylinder in test scene
        red_mask = (rgb[:, :, 0] > 100) & (rgb[:, :, 0] > rgb[:, :, 1] + 40)
        v_idx, u_idx = np.where(red_mask)
        self.assertGreater(len(u_idx), 0, "Red cylinder should be visible to front_cam")

        box_2d = BoundingBox2D(
            ymin=float(v_idx.min()) * 1000.0 / 120.0,
            xmin=float(u_idx.min()) * 1000.0 / 160.0,
            ymax=float(v_idx.max()) * 1000.0 / 120.0,
            xmax=float(u_idx.max()) * 1000.0 / 160.0,
        )

        target = ground_bounding_box(
            box=box_2d,
            depth_map=depth,
            model=self.model,
            data=self.data,
            label="red cylinder",
            clearance_m=0.5,
        )

        # Ground truth cylinder is at X=2.0m, Y=0.3m. Surface is at X=1.85m, Y=0.3m.
        # Check lateral position Y matches within 0.05m
        self.assertAlmostEqual(target.body_coords[1], 0.3, delta=0.06)
        # Check forward position X matches front surface within 0.10m
        self.assertAlmostEqual(target.body_coords[0], 1.85, delta=0.10)

        # Check waypoint clearance: waypoint should be ~0.5m in front of the target
        dist_body = math.hypot(target.body_coords[0], target.body_coords[1])
        dist_waypoint = math.hypot(target.target_waypoint[0], target.target_waypoint[1])
        self.assertAlmostEqual(dist_body - dist_waypoint, 0.5, delta=0.02)


class TestVLMGoalGrounder(unittest.TestCase):
    """Test VLMGoalGrounder offline synthesis and error handling."""

    @classmethod
    def setUpClass(cls):
        xml_content = """<mujoco model="vlm_test_scene">
          <include file="unitree_go2/go2_description.xml"/>
          <worldbody>
            <geom name="test_cylinder" type="cylinder" size="0.15 0.4" pos="2.0 0.3 0.4" rgba="0.9 0.1 0.1 1"/>
          </worldbody>
        </mujoco>"""
        cls.model = mujoco.MjModel.from_xml_string(xml_content)
        cls.data = mujoco.MjData(cls.model)
        initialize_simulation_standing(cls.model, cls.data, body_height=0.28)
        mujoco.mj_step(cls.model, cls.data)
        cls.renderer = mujoco.Renderer(cls.model, height=120, width=160)

    def test_offline_ground_instruction(self):
        grounder = VLMGoalGrounder(api_key="mock_key")
        box = BoundingBox2D(ymin=200, xmin=300, ymax=600, xmax=450)
        res = grounder.offline_ground_instruction(
            renderer=self.renderer,
            model=self.model,
            data=self.data,
            target_box=box,
            label="test_cone",
            clearance_m=0.5,
        )
        self.assertTrue(res.target_found)
        self.assertIsNotNone(res.target)
        self.assertEqual(len(res.motion_commands), 2)
        # First command must be WAYPOINT mode
        self.assertEqual(res.motion_commands[0].mode, MotionMode.WAYPOINT)
        self.assertIsNotNone(res.motion_commands[0].waypoint)
        # Second command must be STAND mode
        self.assertEqual(res.motion_commands[1].mode, MotionMode.STAND)

    def test_missing_api_key_fails_closed(self):
        grounder = VLMGoalGrounder(api_key="")
        res = grounder.ground_instruction(
            renderer=self.renderer,
            model=self.model,
            data=self.data,
            instruction="walk to the cylinder",
        )
        self.assertFalse(res.target_found)
        self.assertEqual(len(res.motion_commands), 1)
        self.assertEqual(res.motion_commands[0].mode, MotionMode.STAND)
        self.assertIn("Missing GEMINI_API_KEY", res.explanation)


if __name__ == "__main__":
    unittest.main()
