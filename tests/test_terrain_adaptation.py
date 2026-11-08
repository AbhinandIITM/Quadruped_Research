"""
test_terrain_adaptation.py
==========================
Unit tests for procedural terrain generation and elevation adaptation.
"""

import unittest
import mujoco

from go2_terrain_generator import (
    generate_stairs_xml,
    generate_slope_xml,
    build_terrain_world_xml,
)


class TestTerrainGeneration(unittest.TestCase):

    def test_generate_stairs_xml_structure(self):
        stairs_xml = generate_stairs_xml(num_steps=6, step_height=0.05, step_depth=0.25)
        self.assertIn("step_0", stairs_xml)
        self.assertIn("step_5", stairs_xml)
        self.assertEqual(stairs_xml.count("<geom name=\"step_"), 6)

    def test_generate_slope_xml_structure(self):
        slope_xml = generate_slope_xml(incline_deg=10.0, ramp_length=3.0)
        self.assertIn("ramp_incline", slope_xml)
        self.assertIn("quat=", slope_xml)

    def test_build_terrain_world_xml_parses_in_mujoco(self):
        """Verifies that generated stair world XML is fully valid in MuJoCo physics."""
        stairs_xml = build_terrain_world_xml(terrain_type="stairs", num_steps=5, step_height=0.04)
        # MuJoCo meshdir in go2_description.xml is '.' so supply directory
        try:
            model = mujoco.MjModel.from_xml_string(stairs_xml.replace('meshdir="."', 'meshdir="unitree_go2"'))
            self.assertIsNotNone(model)
            self.assertGreater(model.ngeom, 20)
        except Exception as e:
            self.fail(f"MuJoCo failed to parse procedural stairs XML: {e}")

    def test_build_slope_world_xml_parses_in_mujoco(self):
        """Verifies that generated slope world XML is fully valid in MuJoCo physics."""
        slope_xml = build_terrain_world_xml(terrain_type="slope", incline_deg=8.0)
        try:
            model = mujoco.MjModel.from_xml_string(slope_xml.replace('meshdir="."', 'meshdir="unitree_go2"'))
            self.assertIsNotNone(model)
            self.assertGreater(model.ngeom, 20)
        except Exception as e:
            self.fail(f"MuJoCo failed to parse procedural slope XML: {e}")


if __name__ == "__main__":
    unittest.main()
