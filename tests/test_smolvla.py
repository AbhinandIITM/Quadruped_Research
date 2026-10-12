"""
tests/test_smolvla.py
=====================
Unit tests for Hugging Face SmolVLA (Vision-Language-Action) integration,
observation preprocessing, action decoding, and policy execution for the Unitree Go2.
"""

import unittest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from go2_champ_control import MotionMode, SafetyBounds
from go2_smolvla import (
    SmolVLAAction,
    SmolVLAActionDecoder,
    SmolVLAObservation,
    SmolVLAObservationPreprocessor,
    SmolVLAQuadrupedAgent,
)


class TestSmolVLAObservationPreprocessor(unittest.TestCase):
    """Test observation preprocessing into PyTorch tensor batches."""

    def setUp(self):
        self.preprocessor = SmolVLAObservationPreprocessor(device="cpu")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for preprocessor tests")
    def test_preprocess_image_shape_and_range(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Solid red
        tensor = self.preprocessor.preprocess_image(rgb)

        self.assertEqual(tensor.shape, (1, 3, 256, 256))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(tensor[0, 1, 0, 0]), 0.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for preprocessor tests")
    def test_preprocess_state_dimension(self):
        state_4d = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        tensor = self.preprocessor.preprocess_state(state_4d)
        self.assertEqual(tensor.shape, (1, 6))

        state_8d = np.arange(8, dtype=np.float32)
        tensor2 = self.preprocessor.preprocess_state(state_8d)
        self.assertEqual(tensor2.shape, (1, 6))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for preprocessor tests")
    def test_tokenize_prompt(self):
        tokens, mask = self.preprocessor.tokenize_prompt("Walk forward at 0.3 m/s")
        self.assertEqual(tokens.shape, (1, 48))
        self.assertEqual(mask.shape, (1, 48))
        self.assertTrue(mask.any())

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for preprocessor tests")
    def test_build_batch_keys(self):
        obs = SmolVLAObservation(
            rgb_image=np.zeros((60, 80, 3), dtype=np.uint8),
            state_vector=np.zeros(6, dtype=np.float32),
            prompt="trot forward",
        )
        batch = self.preprocessor.build_batch(obs)
        expected_keys = [
            "observation.images.camera1",
            "observation.state",
            "observation.language.tokens",
            "observation.language.attention_mask",
        ]
        for k in expected_keys:
            self.assertIn(k, batch, f"Missing key '{k}' in SmolVLA batch dictionary")


class TestSmolVLAActionDecoder(unittest.TestCase):
    """Test mapping raw action tensors to safe MotionCommand primitives."""

    def setUp(self):
        self.bounds = SafetyBounds(
            max_vx=0.50,
            min_vx=-0.25,
            max_vy=0.30,
            max_yaw_rate=0.60,
        )
        self.decoder = SmolVLAActionDecoder(bounds=self.bounds)

    def test_clamping_to_safety_bounds(self):
        # Extreme raw actions exceeding limits
        extreme_raw = np.array([10.0, -10.0, 10.0, 5.0, 0.0, 0.0], dtype=np.float32)
        action = self.decoder.decode(extreme_raw, prompt="test")

        self.assertLessEqual(action.vx, self.bounds.max_vx)
        self.assertGreaterEqual(action.vx, self.bounds.min_vx)
        self.assertLessEqual(action.vy, self.bounds.max_vy)
        self.assertGreaterEqual(action.vy, -self.bounds.max_vy)
        self.assertLessEqual(action.yaw_rate, self.bounds.max_yaw_rate)
        self.assertGreaterEqual(action.yaw_rate, -self.bounds.max_yaw_rate)

    def test_stop_stand_keyword(self):
        raw = np.array([2.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action = self.decoder.decode(raw, prompt="stop now and stand")

        self.assertEqual(action.vx, 0.0)
        self.assertEqual(action.vy, 0.0)
        self.assertEqual(action.yaw_rate, 0.0)
        self.assertEqual(action.motion_commands[0].mode, MotionMode.STAND)

    def test_commands_structure(self):
        raw = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action = self.decoder.decode(raw, prompt="walk forward")

        self.assertEqual(len(action.motion_commands), 2)
        self.assertEqual(action.motion_commands[0].mode, MotionMode.VELOCITY)
        self.assertEqual(action.motion_commands[1].mode, MotionMode.STAND)

    def test_explicit_velocity_and_duration_matching(self):
        # Prompt explicitly specifies 0.3 m/s for 6 seconds
        raw = np.array([0.1, 0.9, -0.8, -0.5, 0.0, 0.0], dtype=np.float32)
        action = self.decoder.decode(raw, prompt="Walk forward at 0.3 m/s for 6 seconds")

        self.assertAlmostEqual(action.vx, 0.30, places=2)
        self.assertAlmostEqual(action.vy, 0.00, places=2)
        self.assertAlmostEqual(action.yaw_rate, 0.00, places=2)
        self.assertAlmostEqual(action.duration, 6.0, places=1)
        self.assertEqual(action.motion_commands[0].vx, 0.30)
        self.assertEqual(action.motion_commands[0].duration, 6.0)

    def test_explicit_turn_matching(self):
        # Prompt explicitly specifies turn left at 0.4 rad/s for 2 seconds
        raw = np.array([0.9, 0.0, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        action = self.decoder.decode(raw, prompt="Turn left at 0.4 rad/s for 2 seconds")

        self.assertAlmostEqual(action.vx, 0.00, places=2)
        self.assertAlmostEqual(action.yaw_rate, 0.40, places=2)
        self.assertAlmostEqual(action.duration, 2.0, places=1)


class TestSmolVLAQuadrupedAgent(unittest.TestCase):
    """Test end-to-end SmolVLA quadruped agent inference."""

    @classmethod
    def setUpClass(cls):
        # Mock policy initialization to allow ultra-fast offline unit testing (<0.05s)
        with unittest.mock.patch.object(SmolVLAQuadrupedAgent, "_init_policy"):
            cls.agent = SmolVLAQuadrupedAgent(device="cpu", load_pretrained=False)
            mock_policy = unittest.mock.MagicMock()
            if HAS_TORCH:
                mock_policy.select_action.return_value = torch.tensor([[0.5, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
            else:
                mock_policy.select_action.return_value = np.array([[0.5, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
            cls.agent.policy = mock_policy
            cls.agent.is_loaded = True

    def test_agent_availability(self):
        avail, msg = self.agent.check_availability()
        self.assertTrue(avail)
        self.assertIsInstance(msg, str)

    def test_agent_predict_action(self):
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        state = np.zeros(6, dtype=np.float32)
        action = self.agent.predict_action(rgb=rgb, state=state, prompt="walk forward at 0.3 m/s")

        self.assertIsInstance(action, SmolVLAAction)
        self.assertGreater(action.duration, 0.0)
        self.assertGreaterEqual(len(action.motion_commands), 2)

    def test_agent_query_interface(self):
        res = self.agent.query("Turn left 0.4 rad/s")
        self.assertEqual(res.status, "success")
        self.assertIn("hf-smolvla", res.model_name)
        self.assertGreater(len(res.commands), 0)
        self.assertIsNotNone(res.parsed_json)


if __name__ == "__main__":
    unittest.main()
