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
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

# Ensure root directory is on PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from go2_champ_control import MotionCommand, MotionMode, SafetyBounds
from go2_llm_commands import (
    AUDIT_FAILURE_WARNING_INTERVAL,
    DEFAULT_MODEL_NAME,
    AuditTrail,
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

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def make_json_http_response(body: str) -> MagicMock:
    """Build a MagicMock standing in for an HTTP 200 response carrying `body`."""
    mock_cm = MagicMock()
    mock_cm.status = 200
    mock_cm.read.return_value = body.encode("utf-8")
    mock_cm.__enter__.return_value = mock_cm
    return mock_cm


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
        content = open(os.path.join(FIXTURES_DIR, "valid_velocity_response.json"), encoding="utf-8").read()
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
        content = open(os.path.join(FIXTURES_DIR, "valid_waypoint_response.json"), encoding="utf-8").read()
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
        content = open(os.path.join(FIXTURES_DIR, "out_of_bounds_response.json"), encoding="utf-8").read()
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
        content = open(os.path.join(FIXTURES_DIR, "malformed_response.json"), encoding="utf-8").read()
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
        """A response containing a non-finite constant must default to safe STAND.

        Contract rule: a non-finite JSON constant makes the response inadmissible
        as a whole.  The validator's own ``json.loads`` installs
        ``parse_constant=_reject_nonfinite_json_constant``, so the ``NaN`` /
        ``Infinity`` literal is never constructed as a float and the document is
        rejected before any field is inspected.  The safety property this test
        has always asserted -- the robot ends up stationary -- still holds; what
        changed is the reported status, from ``clamped`` to ``error``, because
        the rejection now happens at parse time rather than at field level.
        """
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

        self.assertEqual(res.status, "error")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].vx, 0.0)
        self.assertAlmostEqual(res.commands[0].yaw_rate, 0.0)

    def test_heading_target_normalized_to_pi_range(self):
        """heading_target is an angle: it must be wrapped into [-pi, pi]."""
        raw = (
            '{"explanation": "heading", "commands": ['
            '{"mode": "velocity", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "heading_target": 7.5}]}'
        )
        res = parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="Face 7.5 rad",
            bounds=SafetyBounds(),
        )

        self.assertEqual(res.status, "clamped")
        cmd = res.commands[0]
        self.assertIsNotNone(cmd.heading_target)
        self.assertLessEqual(abs(cmd.heading_target), math.pi)
        self.assertAlmostEqual(cmd.heading_target, 7.5 - 2.0 * math.pi, places=6)

    def test_nonfinite_heading_target_dropped(self):
        """A non-finite heading_target must never be forwarded.

        Contract rule: a non-finite JSON constant makes the response inadmissible
        as a whole, so the document is rejected at parse time and the validator
        returns the single fail-closed stand.  The property under test -- a
        non-finite heading is never forwarded to the controller -- holds a
        fortiori: the heading is not merely dropped, the whole command is.
        """
        raw = (
            '{"explanation": "heading", "commands": ['
            '{"mode": "velocity", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "heading_target": NaN}]}'
        )
        res = parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="Face NaN",
            bounds=SafetyBounds(),
        )

        self.assertEqual(res.status, "error")
        self.assertIsNone(res.commands[0].heading_target)
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].vx, 0.0)

    def test_unknown_mode_stand_carries_no_velocity(self):
        """An unrecognized mode must fail closed to a STAND that is truly stationary.

        The command has to be internally consistent with its own mode: a STAND must never
        carry a non-zero velocity that a consumer reading vx without re-checking the mode
        would act on.
        """
        raw = (
            '{"explanation": "backflip", "commands": ['
            '{"mode": "backflip", "vx": 0.5, "vy": -0.4, "yaw_rate": 0.9, "duration": 2.0}]}'
        )
        res = parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="Do a backflip",
            bounds=SafetyBounds(),
        )

        self.assertEqual(res.status, "clamped")
        self.assertEqual(len(res.commands), 1)
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.STAND)
        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vy, 0.0)
        self.assertAlmostEqual(cmd.yaw_rate, 0.0)

    def test_explicit_stand_keeps_its_finite_velocities(self):
        """A STAND command must be self-consistent: no velocity triple.

        This inverts an earlier expectation, deliberately.  The previous rule
        was "an explicitly requested STAND keeps its velocities; only the
        unknown-mode fallback zeroes them", which left the contract admitting
        a command labelled ``stand`` that advertises ``vx=0.25``.  Such a
        command is internally contradictory, and a consumer that reads
        ``cmd.vx`` without re-checking ``cmd.mode`` -- the ``run_meta.json``
        writer in the CLI is exactly such a consumer -- would act on a
        velocity the command does not authorise.  The controller already
        discards the velocities of a STAND command at the actuation site, so
        zeroing them here changes no actuation; it only makes the command
        object safe to read from either field.
        """
        raw = (
            '{"explanation": "stand", "commands": ['
            '{"mode": "stand", "vx": 0.25, "vy": -0.1, "yaw_rate": 0.05, "duration": 1.5}]}'
        )
        res = parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="Stand still",
            bounds=SafetyBounds(),
        )

        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.STAND)
        self.assertAlmostEqual(cmd.vx, 0.0)
        self.assertAlmostEqual(cmd.vy, 0.0)
        self.assertAlmostEqual(cmd.yaw_rate, 0.0)
        # The duration is a genuine, finite request and is preserved.
        self.assertAlmostEqual(cmd.duration, 1.5)
        # The command was modified, so the result must not report plain success.
        self.assertEqual(res.status, "clamped")


class TestContractHoleRegressions(unittest.TestCase):
    """Regression tests for three holes the adversarial audit found in the contract.

    Each of these was a *silent* failure: the validator reported a successful
    parse and returned a command that violated the contract's own invariants.
    The 300-odd unit tests in this suite did not cover them, because they only
    exercised the field types the tests happened to use.  The adversarial
    evaluation in ``experiments/adversarial_llm_eval.py`` found all three; these
    tests are the durable form of that finding.
    """

    def _parse(self, raw: str):
        return parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="regression",
            bounds=SafetyBounds(),
        )

    def test_nonfinite_waypoint_coordinate_fails_closed(self):
        """A NaN waypoint coordinate must not be admitted as a successful command.

        Before the fix the validator reported ``status="success"`` and returned
        a waypoint with ``x=nan``: the waypoint scalars were not in the tuple
        tested for finiteness, and the comparison-based clip passes NaN through
        unchanged because neither ``nan < low`` nor ``nan > high`` holds.  A NaN
        that reaches the navigator also poisons the slew limiter's internal
        state permanently, so a subsequent perfectly valid command would still
        yield NaN.
        """
        res = self._parse(
            '{"explanation": "wp", "commands": ['
            '{"mode": "waypoint", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "waypoint_x": NaN, "waypoint_y": 1.0}]}'
        )
        self.assertNotEqual(res.status, "success")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].vx, 0.0)
        # No command in the list may carry a non-finite waypoint.
        for cmd in res.commands:
            if cmd.waypoint is not None:
                self.assertTrue(math.isfinite(cmd.waypoint.x))
                self.assertTrue(math.isfinite(cmd.waypoint.y))

    def test_infinite_waypoint_coordinate_fails_closed(self):
        """The same rule applies to the Infinity constant."""
        res = self._parse(
            '{"explanation": "wp", "commands": ['
            '{"mode": "waypoint", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "waypoint_x": 1.0, "waypoint_y": Infinity}]}'
        )
        self.assertNotEqual(res.status, "success")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)

    def test_waypoint_command_is_followed_by_a_trailing_stand(self):
        """The trailing-stand invariant must cover waypoint-mode motion.

        The invariant used to test the *velocity triple*, but a WAYPOINT command
        carries zero velocity fields -- its motion comes from the waypoint
        navigator -- so a waypoint plan could end with no stand, contradicting
        the invariant the system prompt promises the model.  The test is now on
        the mode.
        """
        res = self._parse(
            '{"explanation": "wp", "commands": ['
            '{"mode": "waypoint", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "waypoint_x": 2.0, "waypoint_y": 0.3}]}'
        )
        self.assertGreaterEqual(len(res.commands), 2)
        self.assertEqual(res.commands[-1].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[-1].duration, 1.0)
        self.assertAlmostEqual(res.commands[-1].vx, 0.0)

    def test_velocity_command_still_gets_a_trailing_stand(self):
        """The pre-existing trailing-stand behaviour must be preserved."""
        res = self._parse(
            '{"explanation": "v", "commands": ['
            '{"mode": "velocity", "vx": 0.3, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}]}'
        )
        self.assertGreaterEqual(len(res.commands), 2)
        self.assertEqual(res.commands[-1].mode, MotionMode.STAND)

    def test_a_pure_stand_command_is_not_given_a_redundant_trailing_stand(self):
        """A list that already ends in a stand must not grow a second one."""
        res = self._parse(
            '{"explanation": "s", "commands": ['
            '{"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}]}'
        )
        self.assertEqual(len(res.commands), 1)
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)

    def test_no_admitted_command_ever_carries_a_non_finite_value(self):
        """The blanket property: every field of every returned command is finite."""
        payloads = [
            '{"mode": "velocity", "vx": NaN, "vy": 0, "yaw_rate": 0, "duration": 1}',
            '{"mode": "velocity", "vx": Infinity, "vy": 0, "yaw_rate": 0, "duration": 1}',
            '{"mode": "velocity", "vx": 0.1, "vy": NaN, "yaw_rate": 0, "duration": 1}',
            '{"mode": "velocity", "vx": 0.1, "vy": 0, "yaw_rate": NaN, "duration": 1}',
            '{"mode": "velocity", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration": NaN}',
            '{"mode": "waypoint", "vx": 0, "vy": 0, "yaw_rate": 0, "duration": 1,'
            ' "waypoint_x": NaN, "waypoint_y": 0}',
            '{"mode": "waypoint", "vx": 0, "vy": 0, "yaw_rate": 0, "duration": 1,'
            ' "waypoint_x": 0, "waypoint_y": NaN}',
            '{"mode": "velocity", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration": 1,'
            ' "heading_target": NaN}',
        ]
        for raw in payloads:
            res = self._parse(raw)
            for cmd in res.commands:
                for value in (cmd.vx, cmd.vy, cmd.yaw_rate, cmd.duration):
                    self.assertTrue(
                        math.isfinite(value),
                        f"non-finite value {value!r} admitted from {raw!r}",
                    )
                if cmd.waypoint is not None:
                    self.assertTrue(math.isfinite(cmd.waypoint.x))
                    self.assertTrue(math.isfinite(cmd.waypoint.y))
                    self.assertTrue(math.isfinite(cmd.waypoint.speed))
                    self.assertTrue(math.isfinite(cmd.waypoint.tolerance))

    def test_no_admitted_stand_command_carries_a_velocity(self):
        """A STAND command must be stationary under either reading of the object."""
        payloads = [
            '{"mode": "stand", "vx": 0.25, "vy": -0.1, "yaw_rate": 0.05, "duration": 1.5}',
            '{"mode": "backflip", "vx": 0.5, "vy": -0.4, "yaw_rate": 0.9, "duration": 2.0}',
            '{"mode": "waypoint", "vx": 0.4, "vy": 0.2, "yaw_rate": 0.3, "duration": 1.0}',
        ]
        for raw in payloads:
            res = self._parse(raw)
            for cmd in res.commands:
                if cmd.mode == MotionMode.STAND:
                    self.assertAlmostEqual(cmd.vx, 0.0)
                    self.assertAlmostEqual(cmd.vy, 0.0)
                    self.assertAlmostEqual(cmd.yaw_rate, 0.0)


    def test_oversized_integer_literal_does_not_escape_the_validator(self):
        """A 400-digit integer literal must not make the validator raise.

        ``json.loads`` produces a Python ``int`` whose magnitude exceeds the
        float range, and ``float()`` on it raises ``OverflowError`` -- which
        the original coercion handler did not catch.  The exception therefore
        escaped ``parse_and_validate_response`` entirely, which broke the
        contract's totality: the function was supposed to map *every* input to
        an admissible command list, and for this family of inputs it raised
        instead.  Containment had been left to the broad ``except Exception``
        in the HTTP clients, so a caller that used the validator directly got
        an exception rather than a stationary hold.
        """
        huge = "9" * 400
        payloads = [
            '{"mode": "velocity", "vx": ' + huge + ', "vy": 0, "yaw_rate": 0, "duration": 2}',
            '{"mode": "velocity", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration": ' + huge + '}',
            '{"mode": "waypoint", "vx": 0, "vy": 0, "yaw_rate": 0, "duration": 2,'
            ' "waypoint_x": ' + huge + ', "waypoint_y": 1}',
            '{"mode": "velocity", "vx": 0.1, "vy": 0, "yaw_rate": 0, "duration": 2,'
            ' "heading_target": ' + huge + '}',
        ]
        for raw in payloads:
            try:
                res = self._parse(raw)
            except Exception as exc:  # pragma: no cover - the point of the test
                self.fail(f"validator raised {type(exc).__name__} on {raw[:60]}: {exc}")
            self.assertTrue(res.commands, "the validator must always return a command list")
            for cmd in res.commands:
                for value in (cmd.vx, cmd.vy, cmd.yaw_rate, cmd.duration):
                    self.assertTrue(math.isfinite(value))
                if cmd.waypoint is not None:
                    self.assertTrue(math.isfinite(cmd.waypoint.x))
                    self.assertTrue(math.isfinite(cmd.waypoint.y))

    def test_a_boolean_velocity_is_not_read_as_one_metre_per_second(self):
        """``float(True)`` is 1.0, so a boolean must be rejected, not coerced."""
        res = self._parse(
            '{"mode": "velocity", "vx": true, "vy": 0, "yaw_rate": 0, "duration": 2}'
        )
        for cmd in res.commands:
            self.assertAlmostEqual(cmd.vx, 0.0)
            self.assertAlmostEqual(cmd.vy, 0.0)
            self.assertAlmostEqual(cmd.yaw_rate, 0.0)
        self.assertNotEqual(res.status, "success")

    def test_a_clipped_waypoint_is_reported_as_modified(self):
        """A waypoint outside the workspace must not be reported as plain success."""
        res = self._parse(
            '{"mode": "waypoint", "vx": 0, "vy": 0, "yaw_rate": 0, "duration": 2,'
            ' "waypoint_x": 500, "waypoint_y": 1}'
        )
        self.assertEqual(res.status, "clamped")
        self.assertIsNotNone(res.commands[0].waypoint)
        self.assertLessEqual(abs(res.commands[0].waypoint.x), 10.0)

    def test_the_command_count_is_bounded(self):
        """An unbounded command list is a denial-of-service surface."""
        from go2_llm_commands import MAX_COMMANDS_PER_RESPONSE

        many = ",".join(
            ['{"mode": "velocity", "vx": 0.1, "duration": 1}'] * (MAX_COMMANDS_PER_RESPONSE + 50)
        )
        res = self._parse('{"commands": [' + many + ']}')
        self.assertLessEqual(
            len(res.commands), MAX_COMMANDS_PER_RESPONSE + 1, "the trailing stand may add one"
        )
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[-1].mode, MotionMode.STAND)


class TestTrajectoryAndGaitContract(unittest.TestCase):
    """Tests for the trajectory mode and the gait field of the action contract.

    ``MotionMode.TRAJECTORY`` and ``MotionCommand.gait`` were both implemented
    in the control module long before the language interface could reach
    them: the schema's mode enum admitted only ``stand``/``velocity``/
    ``waypoint``, and the validator never read the ``gait`` field, so every
    run in this repository was a trot run.  These tests pin the widened
    contract, and above all its *fail-closed* behaviour: the whole point of
    the contract is that a payload the stack cannot execute is refused, not
    approximated.
    """

    def _parse(self, raw: str):
        return parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="trajectory/gait",
            bounds=SafetyBounds(),
        )

    def _trajectory(self, path_type: str, params: str, extra: str = "") -> str:
        return (
            '{"explanation": "follow a path", "commands": ['
            '{"mode": "trajectory", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 6.0, "path_type": "' + path_type + '", '
            '"trajectory_params": {' + params + '}' + extra + '}]}'
        )

    def test_valid_trajectory_command_is_admitted(self):
        """A well-formed trajectory request produces a usable TrajectorySpec."""
        res = self._parse(self._trajectory(
            "circle",
            '"cx": 1.0, "cy": -0.5, "radius": 1.5, "speed": 0.3',
            ', "trajectory_duration": 8.0',
        ))
        self.assertEqual(res.status, "success")
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.TRAJECTORY)
        self.assertIsNotNone(cmd.trajectory)
        self.assertEqual(cmd.trajectory.path_type, "circle")
        self.assertEqual(cmd.trajectory.params["cx"], 1.0)
        self.assertEqual(cmd.trajectory.params["cy"], -0.5)
        self.assertEqual(cmd.trajectory.params["radius"], 1.5)
        self.assertEqual(cmd.trajectory.params["speed"], 0.3)
        self.assertEqual(cmd.trajectory.duration, 8.0)

    def test_figure_8_trajectory_is_admitted(self):
        """The second supported path type is reachable too."""
        res = self._parse(self._trajectory(
            "figure_8",
            '"cx": 0.0, "cy": 0.0, "scale_x": 2.0, "scale_y": 1.0, "omega": 0.8',
        ))
        self.assertEqual(res.status, "success")
        self.assertEqual(res.commands[0].trajectory.path_type, "figure_8")
        self.assertEqual(res.commands[0].trajectory.params["omega"], 0.8)

    def test_unsupported_path_type_fails_closed_to_stand(self):
        """A path the tracker does not implement must not reach the tracker.

        ``TrajectoryTracker.evaluate_path`` raises on an unknown path type, so
        admitting one would turn a model mistake into an exception inside the
        control loop.  The command fails closed to a stationary hold instead.
        """
        res = self._parse(self._trajectory("spiral", '"cx": 0.0'))
        self.assertEqual(res.status, "clamped")
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.STAND)
        self.assertEqual(cmd.vx, 0.0)
        self.assertEqual(cmd.vy, 0.0)
        self.assertEqual(cmd.yaw_rate, 0.0)
        self.assertIsNone(cmd.trajectory)

    def test_missing_path_type_fails_closed_to_stand(self):
        """A trajectory command with no path at all has nothing to execute."""
        res = self._parse(
            '{"explanation": "follow a path", "commands": ['
            '{"mode": "trajectory", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 6.0}]}'
        )
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertIsNone(res.commands[0].trajectory)

    def test_nonfinite_trajectory_parameter_fails_closed_to_stand(self):
        """A NaN path parameter must not be clamped into range.

        The comparison-based clip passes a NaN through unchanged, and a NaN
        inside the tracker would poison the command it produces -- the same
        hazard the waypoint coordinates have, and the same treatment.
        """
        res = self._parse(
            '{"explanation": "follow a path", "commands": ['
            '{"mode": "trajectory", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 6.0, "path_type": "circle", '
            '"trajectory_params": {"radius": NaN}}]}'
        )
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.STAND)
        self.assertEqual(cmd.vx, 0.0)
        self.assertIsNone(cmd.trajectory)

    def test_nonfinite_trajectory_parameter_from_overflow_fails_closed(self):
        """A parameter that overflows float64 to inf is non-finite too."""
        res = self._parse(self._trajectory("circle", '"radius": 1e400'))
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.STAND)
        self.assertIsNone(cmd.trajectory)

    def test_out_of_range_trajectory_parameter_is_clamped(self):
        """An out-of-range parameter is moved into range and reported as clamped.

        The audit trail has to be able to tell "the model asked for a legal
        path" from "the model asked for an illegal one and we moved it".
        """
        res = self._parse(self._trajectory("circle", '"radius": 99.0'))
        self.assertEqual(res.status, "clamped")
        cmd = res.commands[0]
        self.assertEqual(cmd.mode, MotionMode.TRAJECTORY)
        self.assertEqual(cmd.trajectory.params["radius"], 10.0)

    def test_out_of_range_figure_8_omega_is_clamped(self):
        """The figure-8 angular rate has its own declared range."""
        res = self._parse(self._trajectory("figure_8", '"omega": 25.0'))
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].trajectory.params["omega"], 2.0)

    def test_trajectory_duration_is_clamped_to_the_command_duration_range(self):
        """A trajectory duration is a command duration and is bounded as one."""
        res = self._parse(self._trajectory("circle", '"cx": 0.0', ', "trajectory_duration": 9999.0'))
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].trajectory.duration, SafetyBounds().max_duration)

    def test_unsupported_gait_falls_back_to_trot(self):
        """An unknown gait is a default, not a safety failure.

        A gait names a locomotion pattern; the stack's own default (trot) is a
        legal, well-tested one, so the command is still executed with a sane
        gait rather than being thrown away.  The substitution is reported, so
        the audit trail records that the request was not honoured as made.
        """
        res = self._parse(
            '{"explanation": "gallop", "commands": ['
            '{"mode": "velocity", "vx": 0.2, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "gait": "gallop"}]}'
        )
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].gait, "trot")
        # The command itself is still executed: an unknown gait is a default.
        self.assertEqual(res.commands[0].mode, MotionMode.VELOCITY)
        self.assertAlmostEqual(res.commands[0].vx, 0.2)

    def test_valid_gait_is_honoured(self):
        """A supported gait reaches the command and survives to the stand."""
        res = self._parse(
            '{"explanation": "pace", "commands": ['
            '{"mode": "velocity", "vx": 0.2, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0, "gait": "pace"}]}'
        )
        self.assertEqual(res.status, "success")
        self.assertEqual(res.commands[0].gait, "pace")
        # The trailing stand inherits the gait, so stopping does not silently
        # cancel an in-flight gait transition back to the trot profile.
        self.assertEqual(res.commands[-1].mode, MotionMode.STAND)
        self.assertEqual(res.commands[-1].gait, "pace")

    def test_omitted_gait_defaults_to_trot(self):
        """A command that names no gait uses the stack's default."""
        res = self._parse(
            '{"explanation": "walk", "commands": ['
            '{"mode": "velocity", "vx": 0.2, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 2.0}]}'
        )
        self.assertEqual(res.status, "success")
        self.assertEqual(res.commands[0].gait, "trot")

    def test_trajectory_command_gets_a_trailing_stand(self):
        """A trajectory plan must end with a hold.

        The trailing-stand test is on the MODE, not on the velocity triple: a
        trajectory command carries zero velocities -- its motion comes from
        the trajectory tracker -- so a velocity-magnitude test would let a
        trajectory plan end without a stand.
        """
        res = self._parse(self._trajectory("circle", '"cx": 0.0'))
        self.assertEqual(res.commands[-1].mode, MotionMode.STAND)
        self.assertEqual(res.commands[-1].vx, 0.0)
        self.assertGreaterEqual(len(res.commands), 2)

    def test_trajectory_params_must_be_an_object(self):
        """A parameter block that is not an object cannot be read at all."""
        res = self._parse(
            '{"explanation": "follow a path", "commands": ['
            '{"mode": "trajectory", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, '
            '"duration": 6.0, "path_type": "circle", "trajectory_params": [1, 2]}]}'
        )
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertIsNone(res.commands[0].trajectory)

    def test_no_admitted_trajectory_ever_carries_a_non_finite_parameter(self):
        """The blanket property: every admitted path parameter is finite."""
        payloads = [
            self._trajectory("circle", '"radius": NaN'),
            self._trajectory("circle", '"radius": Infinity'),
            self._trajectory("circle", '"radius": -Infinity'),
            self._trajectory("circle", '"radius": 1e400'),
            self._trajectory("circle", '"cx": NaN'),
            self._trajectory("figure_8", '"omega": NaN'),
            self._trajectory("figure_8", '"scale_x": 1e400'),
        ]
        for raw in payloads:
            res = self._parse(raw)
            for cmd in res.commands:
                if cmd.trajectory is not None:
                    for name, value in cmd.trajectory.params.items():
                        self.assertTrue(
                            math.isfinite(value),
                            f"non-finite trajectory parameter {name}={value!r} "
                            f"admitted from {raw!r}",
                        )

    def test_trajectory_speed_is_bounded_by_the_runtime_velocity_bound(self):
        """The circle's traversal speed follows the SafetyBounds under test.

        The bound is read from the ``SafetyBounds`` the caller passed in, not
        from a value frozen at import time, so a run with tighter bounds
        cannot be talked into a faster path than the run allows.
        """
        bounds = SafetyBounds()
        bounds.max_vx = 0.25
        res = parse_and_validate_response(
            raw_content=self._trajectory("circle", '"speed": 0.5'),
            thinking=None,
            user_prompt="trajectory/gait",
            bounds=bounds,
        )
        self.assertEqual(res.status, "clamped")
        self.assertEqual(res.commands[0].trajectory.params["speed"], 0.25)

    def test_schema_advertises_the_trajectory_mode_and_the_gait_field(self):
        """The schema is the model-facing contract; it must advertise both."""
        from go2_llm_commands import MOTION_COMMAND_SCHEMA

        item = MOTION_COMMAND_SCHEMA["properties"]["commands"]["items"]
        props = item["properties"]
        self.assertIn("trajectory", props["mode"]["enum"])
        self.assertEqual(props["path_type"]["enum"], ["circle", "figure_8"])
        self.assertIn("trajectory_params", props)
        self.assertIn("trajectory_duration", props)
        self.assertEqual(
            sorted(props["gait"]["enum"]),
            ["bound", "pace", "pronk", "trot", "walk"],
        )


class TestContractRevisionMarker(unittest.TestCase):
    """Guard on the contract revision that is actually installed.

    The adversarial experiment in ``experiments/adversarial_llm_eval.py``
    measures the contract by swapping ``src/go2_llm_commands.py`` against the
    pre-audit snapshot in ``report/baseline/``.  A swap that is not undone
    would silently put the vulnerable validator back into the working tree,
    and because the vulnerable validator is the one the *pre-fix* result was
    measured with, nothing in the ordinary test suite would notice: the
    304 pre-audit tests pass against it too.

    These two tests make such a swap loud.  The first asserts the marker;
    the second asserts the behaviour the marker is supposed to describe, so
    the check does not depend on the marker alone.
    """

    def test_installed_contract_is_the_current_revision(self):
        import go2_llm_commands

        version = getattr(go2_llm_commands, "CONTRACT_VERSION", None)
        self.assertIsNotNone(
            version,
            "go2_llm_commands does not export CONTRACT_VERSION; the adversarial "
            "experiment cannot select the expected behaviour of an unidentified "
            "contract revision",
        )
        self.assertEqual(
            version,
            "2",
            "the installed validator is not the current contract revision; the "
            "pre-audit snapshot may have been swapped into src/ by mistake",
        )

    def test_installed_contract_behaves_as_the_current_revision(self):
        """Behavioural check, independent of the marker."""
        from go2_champ_control import SafetyBounds

        huge = "9" * 400
        raw = '{"mode": "velocity", "vx": ' + huge + ', "vy": 0, "yaw_rate": 0, "duration": 2}'
        res = parse_and_validate_response(
            raw_content=raw,
            thinking=None,
            user_prompt="overflow probe",
            bounds=SafetyBounds(),
        )
        self.assertTrue(res.commands)
        self.assertEqual(res.commands[0].mode, MotionMode.STAND)
        self.assertAlmostEqual(res.commands[0].vx, 0.0)


class TestOllamaClientMockTransport(unittest.TestCase):
    """Test client behavior using mocked HTTP transports without requiring a running Ollama server."""

    @patch("urllib.request.urlopen")
    def test_mock_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_audit.jsonl")
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
            self.assertTrue(os.path.exists(audit_file))
            # Verify line in audit file
            lines = open(audit_file, encoding="utf-8").read().strip().split("\n")
            self.assertEqual(len(lines), 1)
            audit_record = json.loads(lines[0])
            self.assertEqual(audit_record["status"], "success")
            self.assertEqual(audit_record["user_prompt"], "Walk forward for 2s")

    @patch("urllib.request.urlopen")
    def test_mock_query_network_error_fails_closed(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_audit_err.jsonl")
            client = OllamaDeepSeekClient(audit_file=audit_file)

            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            res = client.query("Move")

            self.assertEqual(res.status, "error")
            self.assertIn("Connection refused", res.error_message or "")
            # Returns safe fallback stand
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertTrue(os.path.exists(audit_file))


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
            audit_file = os.path.join(tmp_dir, "test_gemini_audit.jsonl")
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
            env_file = os.path.join(tmp_dir, ".env")
            open(env_file, "w", encoding="utf-8").write("TEST_DOTENV_KEY=sample_secret_value\n# Comment line\nANOTHER_KEY='quoted_val'")
            load_dotenv(env_file)
            self.assertEqual(os.environ.get("TEST_DOTENV_KEY"), "sample_secret_value")
            self.assertEqual(os.environ.get("ANOTHER_KEY"), "quoted_val")

    @patch("urllib.request.urlopen")
    def test_gemini_api_key_sent_as_header_not_in_url(self, mock_urlopen):
        """The API key must travel in the x-goog-api-key header, never in the URL query string."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_gemini_header_audit.jsonl")
            client = GeminiClient(
                api_key="fake-secret-key",
                model="gemini-2.5-flash",
                audit_file=audit_file,
            )

            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "candidates": [{
                    "content": {"parts": [{"text": json.dumps({
                        "explanation": "Standing still.",
                        "commands": [
                            {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                        ]
                    })}]}
                }]
            }))

            res = client.query("Stand still")
            self.assertEqual(res.status, "success")

            request = mock_urlopen.call_args[0][0]
            self.assertNotIn("fake-secret-key", request.full_url)
            self.assertNotIn("key=", request.full_url)
            headers = {k.lower(): v for k, v in request.header_items()}
            self.assertEqual(headers.get("x-goog-api-key"), "fake-secret-key")
            self.assertEqual(headers.get("content-type"), "application/json")
            self.assertEqual(headers.get("user-agent"), "Go2-CHAMP-Gemini/1.0")


class TestOmniRouteClientMockTransport(unittest.TestCase):
    """Test OmniRoute / OpenAI-compatible client with mock transport."""

    @patch("urllib.request.urlopen")
    def test_omniroute_query_success(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_omniroute_audit.jsonl")
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
            self.assertTrue(os.path.exists(audit_file))

    @patch("urllib.request.urlopen")
    def test_omniroute_network_error_fails_closed(self, mock_urlopen):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_omniroute_err.jsonl")
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
            audit_file = os.path.join(tmp_dir, "test_server_audit.jsonl")
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

    @patch("urllib.request.urlopen")
    def test_live_server_nonfinite_values_fail_closed(self, mock_urlopen):
        """Bare NaN/Infinity tokens in the server response must never reach the controller."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_nan_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )

            for token in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(token=token):
                    body = (
                        '{"status": "success", "explanation": "hostile payload", '
                        '"commands": [{"mode": "velocity", "vx": ' + token + ', '
                        '"vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}]}'
                    )
                    mock_urlopen.return_value = make_json_http_response(body)

                    res = client.query("Walk forward at maximum speed")

                    self.assertEqual(res.status, "error")
                    self.assertEqual(len(res.commands), 1)
                    self.assertEqual(res.commands[0].mode, MotionMode.STAND)
                    self.assertTrue(math.isfinite(res.commands[0].vx))
                    self.assertTrue(math.isfinite(res.commands[0].yaw_rate))
                    self.assertAlmostEqual(res.commands[0].vx, 0.0)
                    self.assertAlmostEqual(res.commands[0].duration, 1.0)

    @patch("urllib.request.urlopen")
    def test_live_server_nonfinite_string_content_fails_closed(self, mock_urlopen):
        """A quoted `"NaN"` bypassing the JSON guard is still caught by the validator."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_nan_str_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )
            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "status": "success",
                "explanation": "quoted NaN",
                "commands": [
                    {"mode": "velocity", "vx": "NaN", "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}
                ],
            }))

            res = client.query("Walk forward")

            self.assertEqual(res.status, "clamped")
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertAlmostEqual(res.commands[0].vx, 0.0)

    @patch("urllib.request.urlopen")
    def test_live_server_out_of_bounds_clamped_to_bounds(self, mock_urlopen):
        """Out-of-bounds server velocities must be clamped to the client's SafetyBounds."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_clamp_audit.jsonl")
            bounds = SafetyBounds(max_vx=0.50, max_vy=0.30, max_yaw_rate=0.70, max_duration=20.0)
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file, bounds=bounds
            )

            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "status": "success",
                "explanation": "Running at 120 m/s.",
                "commands": [
                    {"mode": "velocity", "vx": 120.0, "vy": -50.0, "yaw_rate": 45.0, "duration": 500.0}
                ],
            }))

            res = client.query("Run at 120 m/s")

            self.assertEqual(res.status, "clamped")
            cmd = res.commands[0]
            self.assertEqual(cmd.mode, MotionMode.VELOCITY)
            self.assertAlmostEqual(cmd.vx, bounds.max_vx)
            self.assertAlmostEqual(cmd.vy, -bounds.max_vy)
            self.assertAlmostEqual(cmd.yaw_rate, bounds.max_yaw_rate)
            self.assertAlmostEqual(cmd.duration, bounds.max_duration)
            # A moving command sequence must still terminate in a safe stand
            self.assertEqual(res.commands[-1].mode, MotionMode.STAND)

    @patch("urllib.request.urlopen")
    def test_live_server_waypoint_payload_reconstructed(self, mock_urlopen):
        """The server's nested waypoint schema must survive validation intact."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_waypoint_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )

            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "status": "success",
                "explanation": "Navigating to the target.",
                "commands": [
                    {
                        "mode": "waypoint",
                        "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 8.0,
                        "heading_target": None,
                        "waypoint": {"x": 1.5, "y": -0.8, "heading": None, "tolerance": 0.15, "speed": 0.35},
                    },
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0},
                ],
            }))

            res = client.query("Go to (1.5, -0.8)")

            self.assertEqual(res.status, "success")
            self.assertEqual(len(res.commands), 2)
            cmd = res.commands[0]
            self.assertEqual(cmd.mode, MotionMode.WAYPOINT)
            self.assertIsNotNone(cmd.waypoint)
            self.assertAlmostEqual(cmd.waypoint.x, 1.5)
            self.assertAlmostEqual(cmd.waypoint.y, -0.8)
            self.assertAlmostEqual(cmd.waypoint.speed, 0.35)
            self.assertAlmostEqual(cmd.waypoint.tolerance, 0.15)
            self.assertAlmostEqual(cmd.duration, 8.0)
            self.assertEqual(res.commands[1].mode, MotionMode.STAND)

    @patch("urllib.request.urlopen")
    def test_live_server_unknown_mode_falls_back_to_stand(self, mock_urlopen):
        """An unrecognized mode from the server must fall back to STAND, not a moving mode."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_mode_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )
            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "status": "success",
                "explanation": "Nonsense mode.",
                "commands": [
                    {"mode": "backflip", "vx": 0.5, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}
                ],
            }))

            res = client.query("Backflip")

            self.assertEqual(res.status, "clamped")
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertAlmostEqual(res.commands[0].vx, 0.0)

    @patch("urllib.request.urlopen")
    def test_live_server_response_body_not_a_dict_fails_closed(self, mock_urlopen):
        """A structurally invalid server payload must fail closed rather than raise."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_bad_body_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )
            mock_urlopen.return_value = make_json_http_response('["not", "an", "object"]')

            res = client.query("Move")

            self.assertEqual(res.status, "error")
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)

    @patch("urllib.request.urlopen")
    def test_live_server_auth_token_from_env_sent_as_header(self, mock_urlopen):
        """MODEL_SERVER_TOKEN (shared with the server) must be sent as an auth header."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_auth_env_audit.jsonl")
            with patch.dict(os.environ, {"MODEL_SERVER_TOKEN": "shared-secret-from-env"}):
                client = LiveModelServiceClient(
                    server_url="http://127.0.0.1:5005", audit_file=audit_file
                )
                self.assertEqual(client.auth_token, "shared-secret-from-env")

                mock_urlopen.return_value = make_json_http_response(json.dumps({
                    "status": "success",
                    "explanation": "Standing.",
                    "commands": [
                        {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                    ],
                }))
                res = client.query("Stand still")

            self.assertEqual(res.status, "success")
            request = mock_urlopen.call_args[0][0]
            headers = {k.lower(): v for k, v in request.header_items()}
            self.assertEqual(headers.get("authorization"), "Bearer shared-secret-from-env")
            # The secret must never appear in the URL (proxies and gateways log URLs).
            self.assertNotIn("shared-secret-from-env", request.full_url)
            self.assertNotIn("token=", request.full_url)

    @patch("urllib.request.urlopen")
    def test_live_server_auth_token_argument_overrides_env(self, mock_urlopen):
        """An explicit auth_token argument takes precedence over the environment variable."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_auth_arg_audit.jsonl")
            with patch.dict(os.environ, {"MODEL_SERVER_TOKEN": "env-token"}):
                client = LiveModelServiceClient(
                    server_url="http://127.0.0.1:5005", audit_file=audit_file,
                    auth_token="explicit-token",
                )
                mock_urlopen.return_value = make_json_http_response(json.dumps({
                    "status": "success",
                    "explanation": "Standing.",
                    "commands": [
                        {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                    ],
                }))
                client.query("Stand still")

            request = mock_urlopen.call_args[0][0]
            headers = {k.lower(): v for k, v in request.header_items()}
            self.assertEqual(headers.get("authorization"), "Bearer explicit-token")

    @patch("urllib.request.urlopen")
    def test_live_server_without_token_sends_no_auth_header(self, mock_urlopen):
        """No configured token => no auth header at all (tokenless local flow unchanged)."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_auth_none_audit.jsonl")
            with patch.dict(os.environ):
                os.environ.pop("MODEL_SERVER_TOKEN", None)
                client = LiveModelServiceClient(
                    server_url="http://127.0.0.1:5005", audit_file=audit_file
                )
                self.assertIsNone(client.auth_token)

                mock_urlopen.return_value = make_json_http_response(json.dumps({
                    "status": "success",
                    "explanation": "Standing.",
                    "commands": [
                        {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                    ],
                }))
                res = client.query("Stand still")

            self.assertEqual(res.status, "success")
            request = mock_urlopen.call_args[0][0]
            headers = {k.lower(): v for k, v in request.header_items()}
            self.assertNotIn("authorization", headers)
            self.assertNotIn("x-auth-token", headers)
            self.assertEqual(headers.get("user-agent"), "Go2-CHAMP-Client/1.0")

    @patch("urllib.request.urlopen")
    def test_live_server_401_fails_closed_with_auth_error(self, mock_urlopen):
        """A 401 must surface a clear auth error, fail closed, and never retry unauthenticated."""
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_auth_401_audit.jsonl")
            with patch.dict(os.environ, {"MODEL_SERVER_TOKEN": "wrong-token"}):
                client = LiveModelServiceClient(
                    server_url="http://127.0.0.1:5005", audit_file=audit_file
                )
                mock_urlopen.side_effect = urllib.error.HTTPError(
                    "http://127.0.0.1:5005/query", 401, "Unauthorized", {}, None
                )

                res = client.query("Walk forward")

            # Exactly one attempt: no retry loop and no unauthenticated fallback.
            self.assertEqual(mock_urlopen.call_count, 1)
            self.assertEqual(res.status, "error")
            self.assertEqual(len(res.commands), 1)
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertAlmostEqual(res.commands[0].vx, 0.0)
            self.assertIn("401", res.explanation)
            self.assertIn("unauthorized", res.explanation.lower())
            self.assertIn("MODEL_SERVER_TOKEN", res.explanation)
            self.assertIn("MODEL_SERVER_TOKEN", res.error_message or "")
            # The rejected request still carried the token as a header, not in the URL.
            request = mock_urlopen.call_args[0][0]
            self.assertEqual(
                {k.lower(): v for k, v in request.header_items()}.get("authorization"),
                "Bearer wrong-token",
            )
            self.assertNotIn("wrong-token", request.full_url)

    @patch("urllib.request.urlopen")
    def test_live_server_500_fails_closed(self, mock_urlopen):
        """A non-401 HTTP failure also fails closed, with the status named."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_live_500_audit.jsonl")
            client = LiveModelServiceClient(
                server_url="http://127.0.0.1:5005", audit_file=audit_file
            )
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://127.0.0.1:5005/query", 500, "Internal Server Error", {}, None
            )

            res = client.query("Walk forward")

            self.assertEqual(res.status, "error")
            self.assertEqual(res.commands[0].mode, MotionMode.STAND)
            self.assertIn("500", res.explanation)


class TestAuditTrailFailureVisibility(unittest.TestCase):
    """A dropped safety-audit record must be counted and reported, never silently lost."""

    @staticmethod
    def _sample_result() -> LLMCommandResult:
        return LLMCommandResult(
            user_prompt="Walk forward",
            raw_response="{}",
            thinking=None,
            parsed_json=None,
            commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
            status="success",
            explanation="ok",
        )

    def test_write_success_leaves_failure_count_at_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_file = os.path.join(tmp_dir, "test_audit_ok.jsonl")
            trail = AuditTrail(audit_file)

            self.assertTrue(trail.write(self._sample_result()))

            self.assertEqual(trail.failure_count, 0)
            self.assertIsNone(trail.last_error)
            self.assertEqual(len(open(audit_file, encoding="utf-8").read().strip().split("\n")), 1)

    def test_write_failure_is_counted_and_warned(self):
        """Inject a failing sink (a directory) and assert the failure is visible, not swallowed."""
        import contextlib
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Appending to a directory always fails, so no real log path is touched.
            trail = AuditTrail(tmp_dir)

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                first = trail.write(self._sample_result())
                second = trail.write(self._sample_result())

            self.assertFalse(first)
            self.assertFalse(second)
            self.assertEqual(trail.failure_count, 2)
            self.assertIsNotNone(trail.last_error)
            output = captured.getvalue()
            self.assertIn("Failed to write LLM audit log", output)
            self.assertIn("1 failure(s) so far", output)
            # Rate-limited: the immediate repeat is not re-reported.
            self.assertEqual(output.count("Failed to write LLM audit log"), 1)

    def test_warnings_are_rate_limited_but_never_silent_forever(self):
        """Every failure is counted; warnings resume at each interval boundary."""
        import contextlib
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            trail = AuditTrail(tmp_dir)

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                for _ in range(AUDIT_FAILURE_WARNING_INTERVAL + 2):
                    trail.write(self._sample_result())

            self.assertEqual(trail.failure_count, AUDIT_FAILURE_WARNING_INTERVAL + 2)
            # Warned on the first failure and again at the interval boundary.
            self.assertEqual(captured.getvalue().count("Failed to write LLM audit log"), 2)

    @patch("urllib.request.urlopen")
    def test_client_query_does_not_raise_when_audit_sink_breaks(self, mock_urlopen):
        """A broken audit sink must not corrupt or block the query result."""
        import contextlib
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            client = OllamaDeepSeekClient(audit_file=tmp_dir)
            mock_urlopen.return_value = make_json_http_response(json.dumps({
                "model": "deepseek-r1:8b",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "explanation": "Moving ahead",
                        "commands": [
                            {"mode": "velocity", "vx": 0.3, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}
                        ],
                    }),
                },
            }))

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                res = client.query("Walk forward for 2s")

            self.assertEqual(res.status, "success")
            self.assertAlmostEqual(res.commands[0].vx, 0.3, places=4)
            self.assertEqual(client.audit_trail.failure_count, 1)
            self.assertIn("Failed to write LLM audit log", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
