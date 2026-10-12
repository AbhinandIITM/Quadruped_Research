"""
test_vision_perception.py
=========================
Unit tests for the Go2 vision, depth perception, and reactive obstacle avoidance stack.
"""

import unittest
import numpy as np

from go2_vision_perception import (
    DepthPerceptionConfig,
    DepthPerceptionPipeline,
    ReactiveObstacleAvoidance,
    SectorClearance,
)


class TestVisionPerception(unittest.TestCase):

    def setUp(self):
        self.config = DepthPerceptionConfig(
            image_width=160,
            image_height=120,
            danger_distance_m=0.70,
            warning_distance_m=1.60,
            ground_cutoff_row=55,
            max_range_m=6.0,
        )
        self.avoidance = ReactiveObstacleAvoidance(self.config)

    def test_clear_depth_map_has_no_obstacles(self):
        # Empty space: depth everywhere is max_range_m
        depth = np.full((120, 160), 5.5, dtype=np.float32)

        # Mock pipeline without needing a live MuJoCo renderer for unit test
        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config

        clearance = pipeline.analyze_clearance(depth)
        self.assertFalse(clearance.obstacle_detected)
        self.assertFalse(clearance.is_critical)
        self.assertAlmostEqual(clearance.min_dist, 5.5, places=2)

        # Test command modulation
        vx, vy, yaw, status = self.avoidance.modulate_commands(0.4, 0.0, 0.0, clearance)
        self.assertEqual(vx, 0.4)
        self.assertEqual(vy, 0.0)
        self.assertEqual(yaw, 0.0)
        self.assertEqual(status, "clear")

    def test_center_obstacle_slows_down_and_steers(self):
        depth = np.full((120, 160), 5.0, dtype=np.float32)
        # Put an obstacle at 1.1 meters in the center sector (cols 53 to 107, rows 20 to 50)
        depth[20:50, 60:90] = 1.1

        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config
        clearance = pipeline.analyze_clearance(depth)

        self.assertTrue(clearance.obstacle_detected)
        self.assertFalse(clearance.is_critical)
        self.assertEqual(clearance.closest_sector, "center")
        self.assertAlmostEqual(clearance.min_dist, 1.1, delta=0.05)

        # Reactive avoidance should slow down forward speed and turn
        vx, vy, yaw, status = self.avoidance.modulate_commands(0.5, 0.0, 0.0, clearance)
        self.assertLess(vx, 0.5)
        self.assertGreater(vx, 0.0)
        self.assertNotEqual(yaw, 0.0)
        self.assertIn("AVOIDING", status)

    def test_left_obstacle_steers_right(self):
        depth = np.full((120, 160), 4.5, dtype=np.float32)
        # Obstacle on the left (col < 50) at 1.0 m, rows 20 to 50
        depth[20:50, 10:40] = 1.0

        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config
        clearance = pipeline.analyze_clearance(depth)

        self.assertTrue(clearance.obstacle_detected)
        self.assertEqual(clearance.closest_sector, "left")

        # Steering away from left means negative yaw_rate (turn right)
        vx, vy, yaw, status = self.avoidance.modulate_commands(0.4, 0.0, 0.0, clearance)
        self.assertLess(yaw, 0.0)
        self.assertLess(vx, 0.4)

    def test_right_obstacle_steers_left(self):
        depth = np.full((120, 160), 4.5, dtype=np.float32)
        # Obstacle on the right (col > 110) at 1.0 m, rows 20 to 50
        depth[20:50, 120:150] = 1.0

        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config
        clearance = pipeline.analyze_clearance(depth)

        self.assertTrue(clearance.obstacle_detected)
        self.assertEqual(clearance.closest_sector, "right")

        # Steering away from right means positive yaw_rate (turn left)
        vx, vy, yaw, status = self.avoidance.modulate_commands(0.4, 0.0, 0.0, clearance)
        self.assertGreater(yaw, 0.0)
        self.assertLess(vx, 0.4)

    def test_critical_proximity_halts_forward_motion(self):
        depth = np.full((120, 160), 4.0, dtype=np.float32)
        # Dangerously close obstacle at 0.5m (rows 20 to 50)
        depth[20:50, 60:100] = 0.5

        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config
        clearance = pipeline.analyze_clearance(depth)

        self.assertTrue(clearance.obstacle_detected)
        self.assertTrue(clearance.is_critical)

        # Critical halt must set forward speed to exactly 0.0
        vx, vy, yaw, status = self.avoidance.modulate_commands(0.5, 0.1, 0.0, clearance)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)
        self.assertNotEqual(yaw, 0.0)
        self.assertIn("CRITICAL_HALT", status)

    def test_ground_cutoff_ignores_floor_depth(self):
        depth = np.full((120, 160), 4.0, dtype=np.float32)
        # Put close pixels ONLY in the bottom rows (e.g. row 65 > cutoff 55)
        depth[65:120, :] = 0.3

        pipeline = DepthPerceptionPipeline.__new__(DepthPerceptionPipeline)
        pipeline.config = self.config
        clearance = pipeline.analyze_clearance(depth)

        # Floor pixels below cutoff should be ignored
        self.assertFalse(clearance.obstacle_detected)
        self.assertAlmostEqual(clearance.min_dist, 4.0, places=1)

    def test_reverse_motion_not_impeded_by_front_obstacle(self):
        # Even if obstacle is right in front, reversing (vx < 0) is allowed
        clearance = SectorClearance(
            left_dist=0.5,
            center_dist=0.5,
            right_dist=0.5,
            min_dist=0.5,
            closest_sector="center",
            obstacle_detected=True,
            is_critical=True,
        )
        vx, vy, yaw, status = self.avoidance.modulate_commands(-0.2, 0.0, 0.0, clearance)
        self.assertEqual(vx, -0.2)
        self.assertEqual(status, "clear")


if __name__ == "__main__":
    unittest.main()
