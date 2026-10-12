"""Unit tests for Local DeepSeek-R1 (Ollama) Natural Language Interface and Safety Guardrails.

Tests cover:
- JSON schema structure and validation
- Markdown block and <think> token extraction
- Parsing valid velocity, waypoint, and multi-step command sequences
- Adversarial prompt handling (prompt injection, out-of-bounds parameters, non-finite values)
- Mock transport testing for Ollama client (success, network error, timeout)
- Immutable JSONL audit logging
"""

import io
import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_champ_control import MotionCommand, MotionMode, SafetyBounds
from go2_llm_commands import (
    DEFAULT_MODEL_NAME,
    GeminiClient,
    LLMCommandResult,
    LiveModelServiceClient,
    OllamaDeepSeekClient,
    OmniRouteClient,
    extract_json_block,
    load_dotenv,
    parse_and_validate_response,
    strip_think_tags,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestResponseParsingAndSanitization(unittest.TestCase):
    """Test text processing, think-tag stripping, and JSON extraction."""

    def test_strip_think_tags(self):
        raw = "<think>\nThinking about walking forward.\n</think>\n{\"commands\": []}"
        cleaned, thinking = strip_think_tags(raw)
        self.assertEqual(cleaned, '{"commands": []}')
        self.assertEqual(thinking, "Thinking about walking forward.")

    def test_extract_json_from_markdown_fence(self):
        fenced = "Here is the plan:\n```json\n{\n  \"explanation\": \"ok\",\n  \"commands\": []\n}\n```\nEnjoy!"
        extracted = extract_json_block(fenced)
        data = json.loads(extracted)
        self.assertEqual(data["explanation"], "ok")

    def test_extract_json_from_raw_braces(self):
        raw = "Some introductory text { \"mode\": \"stand\" } trailing text"
        extracted = extract_json_block(raw)
        self.assertEqual(extracted, '{ "mode": "stand" }')

    def test_extract_json_from_array(self):
        raw = 'Here are commands: [{"mode": "velocity", "vx": 0.3, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}]'
        extracted = extract_json_block(raw)
        self.assertEqual(extracted, '[{"mode": "velocity", "vx": 0.3, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}]')


class TestCommandValidation(unittest.TestCase):
    """Test command parsing, validation, and safety clamping."""

    def test_parse_valid_json_array_response(self):
        """Top-level JSON arrays returned by Gemini/Gemma must parse successfully."""
        raw_array = json.dumps([
            {"mode": "velocity", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.4, "duration": 3.0, "explanation": "Turning left"},
            {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
        ])
        bounds = SafetyBounds()
        res = parse_and_validate_response(
            raw_content=raw_array,
            thinking=None,
            user_prompt="Turn left for 3 seconds",
            bounds=bounds,
        )
        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.commands), 2)
        self.assertAlmostEqual(res.commands[0].yaw_rate, 0.4)
        self.assertAlmostEqual(res.commands[0].duration, 3.0)
        self.assertEqual(res.commands[1].mode, MotionMode.STAND)

    def test_parse_valid_velocity_fixture(self):
        content = (FIXTURES_DIR / "valid_velocity_response.json").read_text(encoding="utf-8")
        bounds = SafetyBounds()
        res = parse_and_validate_response(
            raw_content=content,
            thinking="Test thinking",
            user_prompt="Walk forward at 0.35 m/s",
            bounds=bounds,
        )

        self.assertEqual(res.status, "success")
        self.assertEqual(len(res.commands), 2)
        # Command 1: Velocity
        self.assertEqual(res.commands[0].mode, MotionMode.VELOCITY)
        self.assertAlmostEqual(res.commands[0].vx, 0.35, places=4)
        self.assertAlmostEqual(res.commands[0].duration, 2.5, places=4)
        # Command 2: Stand
        self.assertEqual(res.commands[1].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[1].duration, 1.0, places=4)

    def test_parse_valid_waypoint_fixture(self):
        content = (FIXTURES_DIR / "valid_waypoint_response.json").read_text(encoding="utf-8")
        bounds = SafetyBounds()
        res = parse_and_validate_response(
            raw_content=content,
            thinking=None,
            user_prompt="Go to (1.5, 0.8)",
            bounds=bounds,
        )

        self.assertEqual(res.status, "success")
        # Movement sequence safely concludes with a 1.0s stand command
        self.assertEqual(len(res.commands), 2)
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.WAYPOINT)
        self.assertIsNotNone(cmd.waypoint)
        self.assertAlmostEqual(cmd.waypoint.x, 1.5, places=4)
        self.assertAlmostEqual(cmd.waypoint.y, 0.8, places=4)
        self.assertAlmostEqual(cmd.waypoint.speed, 0.35, places=4)
        self.assertAlmostEqual(cmd.waypoint.tolerance, 0.15, places=4)

        # Safety stand at end
        self.assertEqual(res.commands[1].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[1].duration, 1.0, places=4)

    def test_parse_out_of_bounds_clamped(self):
        """Excessive velocities and durations must be strictly clamped to SafetyBounds."""
        content = (FIXTURES_DIR / "out_of_bounds_response.json").read_text(encoding="utf-8")
        bounds = SafetyBounds(
            max_vx=0.60,
            max_vy=0.35,
            max_yaw_rate=1.20,
            max_duration=15.0,
        )
        res = parse_and_validate_response(
            raw_content=content,
            thinking=None,
            user_prompt="Run at 120 m/s!",
            bounds=bounds,
        )

        self.assertEqual(res.status, "clamped")
        # Command 0 is clamped velocity, Command 1 is the safety stand
        self.assertGreaterEqual(len(res.commands), 1)
        cmd = res.commands[0]
        self.assertLessEqual(cmd.vx, bounds.max_vx)
        self.assertLessEqual(abs(cmd.vy), bounds.max_vy)
        self.assertLessEqual(cmd.yaw_rate, bounds.max_yaw_rate)
        # Duration clamped to max single command duration (15.0 s)
        self.assertLessEqual(cmd.duration, 15.0)

    def test_parse_malformed_fails_closed_to_stand(self):
        """Malformed or non-JSON output must fail closed to a safe 1.0s stand command."""
        content = (FIXTURES_DIR / "malformed_response.json").read_text(encoding="utf-8")
        bounds = SafetyBounds()
        res = parse_and_validate_response(
            raw_content=content,
            thinking=None,
            user_prompt="Dance!",
            bounds=bounds,
        )

        self.assertEqual(res.status, "error")
        self.assertEqual(len(res.commands), 1)
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].duration, 1.0)

    def test_adversarial_nan_inf_sanitized(self):
        """Commands containing NaN or Inf must default to safe STAND."""
        adversarial_json = json.dumps({
            "explanation": "NaN attack",
            "commands": [
                {"mode": "velocity", "vx": float("nan"), "vy": 0.0, "yaw_rate": float("inf"), "duration": 2.0}
            ]
        })
        bounds = SafetyBounds()
        res = parse_and_validate_response(
            raw_content=adversarial_json,
            thinking=None,
            user_prompt="NaN attack",
            bounds=bounds,
        )

        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].vx, 0.0)
        self.assertAlmostEqual(res.commands[0].yaw_rate, 0.0)


class TestOllamaClientMockTransport(unittest.TestCase):
    """Test client behavior using mocked HTTP transports without requiring a running Ollama server."""

    @patch("urllib.request.urlopen")
    def test_mock_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_audit.jsonl"
            client = OllamaDeepSeekClient(audit_file=audit_file)

            mock_response_data = {
                "model": "deepseek-r1:8b",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "explanation": "Moving ahead",
                        "commands": [{"mode": "velocity", "vx": 0.3, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}]
                    }),
                    "thinking": "Step by step plan",
                }
            }

            mock_cm = MagicMock()
            mock_cm.status = 200
            mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
            mock_cm.__enter__.return_value = mock_cm
            mock_urlopen.return_value = mock_cm

            res = client.query("Walk forward for 2s")

            self.assertEqual(res.status, "success")
            # Movement command + auto-safety stand
            self.assertEqual(len(res.commands), 2)
            self.assertAlmostEqual(res.commands[0].vx, 0.3, places=4)
            self.assertTrue(audit_file.exists())
            # Verify line in audit file
            lines = audit_file.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            audit_record = json.loads(lines[0])
            self.assertEqual(audit_record["status"], "success")
            self.assertEqual(audit_record["user_prompt"], "Walk forward for 2s")

    @patch("urllib.request.urlopen")
    def test_mock_query_network_error_fails_closed(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_audit_err.jsonl"
            client = OllamaDeepSeekClient(audit_file=audit_file)

            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            res = client.query("Move")

            self.assertEqual(res.status, "error")
            self.assertIn("Connection refused", res.error_message or "")
            # Returns safe fallback stand
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertTrue(audit_file.exists())


class TestGeminiClientMockTransport(unittest.TestCase):
    """Test Gemini / Google AI Studio client with mock transport."""

    def test_missing_api_key_fails_safely(self):
        client = GeminiClient(api_key="")
        available, msg = client.check_availability()
        self.assertFalse(available)
        res = client.query("Walk forward")
        self.assertEqual(res.status, "error")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)

    @patch("urllib.request.urlopen")
    def test_gemini_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_gemini_audit.jsonl"
            client = GeminiClient(api_key="fake-test-key", model="gemma-2-27b-it", audit_file=audit_file)

            mock_response_data = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps({
                                        "explanation": "Trotting forward safely.",
                                        "commands": [
                                            {"mode": "velocity", "vx": 0.35, "vy": 0.0, "yaw_rate": 0.0, "duration": 3.0}
                                        ]
                                    })
                                }
                            ]
                        }
                    }
                ]
            }

            mock_cm = MagicMock()
            mock_cm.status = 200
            mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
            mock_cm.__enter__.return_value = mock_cm
            mock_urlopen.return_value = mock_cm

            res = client.query("Walk forward at 0.35 m/s for 3s")

            self.assertEqual(res.status, "success")
            self.assertEqual(len(res.commands), 2)
            self.assertAlmostEqual(res.commands[0].vx, 0.35)
            self.assertEqual(res.commands[1].mode, MotionMode.STAND)

    def test_load_dotenv_parses_file(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("TEST_DOTENV_KEY=sample_secret_value\n# Comment line\nANOTHER_KEY='quoted_val'", encoding="utf-8")
            load_dotenv(env_file)
            self.assertEqual(os.environ.get("TEST_DOTENV_KEY"), "sample_secret_value")
            self.assertEqual(os.environ.get("ANOTHER_KEY"), "quoted_val")


class TestOmniRouteClientMockTransport(unittest.TestCase):
    """Test OmniRoute / OpenAI-compatible client with mock transport."""

    @patch("urllib.request.urlopen")
    def test_omniroute_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_omniroute_audit.jsonl"
            client = OmniRouteClient(
                base_url="http://localhost:8000/v1",
                model="deepseek-r1:8b",
                api_key="sk-test-key",
                audit_file=audit_file,
            )

            mock_response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({
                                "explanation": "Trotting forward at 0.35 m/s for 2.5s.",
                                "commands": [
                                    {"mode": "velocity", "vx": 0.35, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.5}
                                ]
                            }),
                            "reasoning_content": "Planning the forward trot trajectory.",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

            mock_cm = MagicMock()
            mock_cm.status = 200
            mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
            mock_cm.__enter__.return_value = mock_cm
            mock_urlopen.return_value = mock_cm

            res = client.query("Walk forward at 0.35 m/s for 2.5s")

            self.assertEqual(res.status, "success")
            self.assertEqual(len(res.commands), 2)  # Velocity + auto safety stand
            self.assertAlmostEqual(res.commands[0].vx, 0.35)
            self.assertEqual(res.commands[0].mode, MotionMode.VELOCITY)
            self.assertEqual(res.commands[1].mode, MotionMode.STAND)
            self.assertEqual(res.thinking, "Planning the forward trot trajectory.")
            self.assertTrue(audit_file.exists())

    @patch("urllib.request.urlopen")
    def test_omniroute_network_error_fails_closed(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_omniroute_err.jsonl"
            client = OmniRouteClient(base_url="http://localhost:8000/v1", audit_file=audit_file)

            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            res = client.query("Turn left")
            self.assertEqual(res.status, "error")
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)


class TestLiveModelServiceClientMockTransport(unittest.TestCase):
    """Test LiveModelServiceClient communicating with start_model_server.py."""

    @patch("urllib.request.urlopen")
    def test_live_server_health_check(self, mock_urlopen):
        client = LiveModelServiceClient(server_url="http://127.0.0.1:5005")

        mock_health_data = {
            "status": "ok",
            "backend": "OmniRoute (OpenAI-Compatible Gateway)",
            "model": "deepseek-r1:8b",
            "endpoint": "http://localhost:8000/v1",
        }

        mock_cm = MagicMock()
        mock_cm.status = 200
        mock_cm.read.return_value = json.dumps(mock_health_data).encode("utf-8")
        mock_cm.__enter__.return_value = mock_cm
        mock_urlopen.return_value = mock_cm

        ok, msg = client.check_availability()
        self.assertTrue(ok)
        self.assertIn("deepseek-r1:8b", msg)

    @patch("urllib.request.urlopen")
    def test_live_server_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = Path(tmp_dir) / "test_server_audit.jsonl"
            client = LiveModelServiceClient(server_url="http://127.0.0.1:5005", audit_file=audit_file)

            mock_query_response = {
                "user_prompt": "Walk forward 2 meters",
                "status": "success",
                "explanation": "Trotting forward to target distance.",
                "model_name": "deepseek-r1:8b",
                "wall_time_s": 0.85,
                "commands": [
                    {"mode": "velocity", "vx": 0.40, "vy": 0.0, "yaw_rate": 0.0, "duration": 5.0},
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            }

            mock_cm = MagicMock()
            mock_cm.status = 200
            mock_cm.read.return_value = json.dumps(mock_query_response).encode("utf-8")
            mock_cm.__enter__.return_value = mock_cm
            mock_urlopen.return_value = mock_cm

            res = client.query("Walk forward 2 meters")
            self.assertEqual(res.status, "success")
            self.assertEqual(len(res.commands), 2)
            self.assertAlmostEqual(res.commands[0].vx, 0.40)
            self.assertAlmostEqual(res.commands[0].duration, 5.0)
            self.assertEqual(res.commands[1].mode, MotionMode.STAND)

    @patch("urllib.request.urlopen")
    def test_live_server_offline_fails_closed(self, mock_urlopen):
        client = LiveModelServiceClient(server_url="http://127.0.0.1:5005")
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        ok, msg = client.check_availability()
        self.assertFalse(ok)
        self.assertIn("Could not connect", msg)

        res = client.query("Move")
        self.assertEqual(res.status, "error")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)


if __name__ == "__main__":
    unittest.main()
