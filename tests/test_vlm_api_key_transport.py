"""Credential transport regression tests for the Gemini-backed VLM grounding client.

The Gemini API key must be transmitted in the `x-goog-api-key` header.  Putting it in
the URL query string leaks it to proxies, gateways and HTTP debug logs.

All HTTP traffic is mocked; no network call is made.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_llm_commands import MotionMode
from go2_vlm_perception import VLMGoalGrounder


class TestVLMGeminiApiKeyTransport(unittest.TestCase):
    """Test that VLMGoalGrounder keeps the API key out of the request URL."""

    @patch("urllib.request.urlopen")
    def test_vlm_api_key_sent_as_header_not_in_url(self, mock_urlopen):
        grounder = VLMGoalGrounder(api_key="fake-vlm-secret-key", model="gemini-2.5-flash")

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = np.zeros((48, 64, 3), dtype=np.uint8)

        mock_cm = MagicMock()
        mock_cm.status = 200
        mock_cm.read.return_value = json.dumps({
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({
                    "target_found": False,
                    "target_label": "",
                    "box_2d": None,
                    "confidence": 0.0,
                    "description": "No target visible.",
                })}]}
            }]
        }).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm
        mock_urlopen.return_value = mock_cm

        res = grounder.ground_instruction(
            renderer=mock_renderer,
            model=MagicMock(),
            data=MagicMock(),
            instruction="walk to the red cylinder",
        )

        request = mock_urlopen.call_args[0][0]
        self.assertNotIn("fake-vlm-secret-key", request.full_url)
        self.assertNotIn("key=", request.full_url)
        headers = {k.lower(): v for k, v in request.header_items()}
        self.assertEqual(headers.get("x-goog-api-key"), "fake-vlm-secret-key")
        self.assertEqual(headers.get("content-type"), "application/json")
        self.assertEqual(headers.get("user-agent"), "Go2-VLM-Grounding/1.0")

        # A missing target still fails closed to a safe stand.
        self.assertFalse(res.target_found)
        self.assertEqual(res.motion_commands[0].mode, MotionMode.STAND)


if __name__ == "__main__":
    unittest.main()
