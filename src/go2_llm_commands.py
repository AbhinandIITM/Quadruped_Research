"""Local DeepSeek-R1 (Ollama) Natural Language Interface for Go2 CHAMP.

This module provides a strictly constrained, simulation-only natural language
command interface.  It translates user prompts into typed, bounded
MotionCommand sequences using local Ollama (`deepseek-r1:8b`) and standard-library
urllib.request.

Safety Guarantees (Simulation-Only):
1. The LLM NEVER produces actuator torques or joint angles directly.
2. The LLM CANNOT execute shell commands, file modifications, or network calls.
3. Every command is strictly parsed against a typed JSON schema and clamped
   to conservative SafetyBounds before execution in MuJoCo.
4. Malformed, out-of-range, ambiguous, or dangerous requests fail closed to
   a stationary STAND command.
5. All queries, raw LLM outputs, parsed commands, and safety interventions
   are immutably logged to an audit JSONL file.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

from go2_champ_control import (
    TRAJECTORY_PATH_TYPES,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    TrajectorySpec,
    Waypoint,
    wrap_to_pi,
)

#: Revision of the action contract implemented by this module.  The marker lets
#: an out-of-tree measurement (experiments/adversarial_llm_eval.py) pin its
#: expectations to the revision under test instead of to one hard-coded
#: behaviour, so the same experiment can be pointed at two revisions.
#:
#: Revision "2" is unchanged by the Phase-1 widening (trajectory mode + gait
#: selection).  Those additions are *additive*: every payload the revision-2
#: contract already handled is still handled identically, so the revision
#: marker -- which the adversarial experiment uses to select the behaviour it
#: expects from the shadow classifier -- still describes the same contract.
#: Bumping it to "3" would require modelling a third revision in that script
#: (and would break the pre-fix snapshot workflow documented in
#: report/notes/06_adversarial_experiment.md) for no behavioural reason.
CONTRACT_VERSION = "2"

#: Gaits the deterministic stack can execute.  These are the ``GaitType``
#: members of ``go2_multi_gait``; the controller's ``set_gait`` accepts the
#: same five strings and rejects anything else.
GAIT_TYPES: Tuple[str, ...] = ("trot", "pace", "bound", "pronk", "walk")

#: Gait used when the requested one is not one of :data:`GAIT_TYPES`.  An
#: unknown gait is a *default*, not a hazard: the request is for a locomotion
#: pattern, and the stack's own default (trot) is a legal, well-tested one, so
#: the command is still executed with a sane gait rather than discarded.
DEFAULT_GAIT = "trot"

#: Bounds on the parametric-trajectory parameters, per path type.  These are
#: the closed parameter sets of the two paths implemented by
#: ``TrajectoryTracker.evaluate_path`` (src/go2_champ_control.py:718).
#: The circle's traversal speed is bounded by the forward-velocity bound, so
#: the validator overrides that one entry with the *runtime* ``bounds.max_vx``
#: rather than trusting the value recorded here.
TRAJECTORY_PARAM_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "circle": {
        "cx": (-10.0, 10.0),
        "cy": (-10.0, 10.0),
        "radius": (0.05, 10.0),
        "speed": (0.0, 0.60),
    },
    "figure_8": {
        "cx": (-10.0, 10.0),
        "cy": (-10.0, 10.0),
        "scale_x": (0.05, 10.0),
        "scale_y": (0.05, 10.0),
        "omega": (0.0, 2.0),
    },
}

#: Default parameter values for each path type, used when the model omits a
#: parameter.  They mirror the defaults inside ``evaluate_path`` so that an
#: omitted parameter means the same thing to the validator and to the tracker.
TRAJECTORY_PARAM_DEFAULTS: Dict[str, Dict[str, float]] = {
    "circle": {"cx": 0.0, "cy": 0.0, "radius": 1.0, "speed": 0.30},
    "figure_8": {"cx": 0.0, "cy": 0.0, "scale_x": 1.5, "scale_y": 0.75, "omega": 1.0},
}

#: Default duration of a trajectory command when the model omits
#: ``trajectory_duration``.  It matches the ``TrajectorySpec`` default.
TRAJECTORY_DEFAULT_DURATION = 10.0


def load_dotenv(dotenv_path: Optional[Union[str, pathlib.Path]] = None) -> None:
    """Load key-value pairs from a .env file into os.environ (if not already set)."""
    if dotenv_path is None:
        # Check current working directory first, then root directory of the project
        candidates = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        ]
    else:
        candidates = [dotenv_path]

    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                break
            except Exception as err:
                print(f"[Warning] Failed to parse .env file at {p}: {err}")


# Automatically load .env on module import
load_dotenv()


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
_raw_omni_url = os.environ.get("OMNIROUTE_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or "http://localhost:20128/v1"
if not _raw_omni_url.rstrip("/").endswith("/v1"):
    DEFAULT_OMNIROUTE_URL = f"{_raw_omni_url.rstrip('/')}/v1"
else:
    DEFAULT_OMNIROUTE_URL = _raw_omni_url.rstrip("/")

DEFAULT_OMNIROUTE_MODEL = os.environ.get("OMNIROUTE_MODEL", "gemini_studio")
DEFAULT_OMNIROUTE_API_KEY = (
    os.environ.get("OMNIROUTE_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
DEFAULT_MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:5005")
# Shared secret for the local model server. This is the SAME env var start_model_server.py
# reads, so client and server stay in sync without any extra configuration.
MODEL_SERVER_TOKEN_ENV_VAR = "MODEL_SERVER_TOKEN"
DEFAULT_TIMEOUT_S = 90.0
# Availability probes (`check_availability`) are liveness checks against a local process,
# not inference calls, so they deliberately use a much shorter budget than
# DEFAULT_TIMEOUT_S: an unresponsive backend should be reported in seconds instead of
# stalling the caller for the full inference budget before the real query is attempted.
AVAILABILITY_PROBE_TIMEOUT_S = 5.0
DEFAULT_AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "llm_audit.jsonl")

# A failed audit write is a safety event, so it is always counted and reported. After the
# first failure, warnings are rate-limited to at most one per this many failures so a
# persistently broken audit sink stays visible without flooding stdout forever.
AUDIT_FAILURE_WARNING_INTERVAL = 10


# -----------------------------------------------------------------------------
# Structured JSON Schema for Ollama DeepSeek-R1
# -----------------------------------------------------------------------------

MOTION_COMMAND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "Brief explanation of the commanded action and safety considerations."
        },
        "commands": {
            "type": "array",
            "description": "Sequential motion commands to execute.",
            "items": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["stand", "velocity", "waypoint", "trajectory"],
                        "description": "Motion mode: 'stand' (stop/stay), 'velocity' (trot at vx, vy, yaw_rate), 'waypoint' (navigate to x, y), 'trajectory' (follow a parametric path)."
                    },
                    "vx": {
                        "type": "number",
                        "description": "Forward linear velocity in m/s (+forward 0.0 to 0.5, -backward -0.25 to 0.0)."
                    },
                    "vy": {
                        "type": "number",
                        "description": "Lateral linear velocity in m/s (+left 0.0 to 0.3, -right -0.3 to 0.0)."
                    },
                    "yaw_rate": {
                        "type": "number",
                        "description": "Body heading rate in rad/s (+left/CCW 0.0 to 0.6, -right/CW -0.6 to 0.0)."
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration for this command in seconds (0.2 to 10.0 s)."
                    },
                    "heading_target": {
                        "type": ["number", "null"],
                        "description": "Optional world heading angle target in radians."
                    },
                    "waypoint_x": {
                        "type": ["number", "null"],
                        "description": "Target X coordinate in meters for waypoint mode."
                    },
                    "waypoint_y": {
                        "type": ["number", "null"],
                        "description": "Target Y coordinate in meters for waypoint mode."
                    },
                    "waypoint_speed": {
                        "type": ["number", "null"],
                        "description": "Navigation speed in m/s (0.15 to 0.40 m/s)."
                    },
                    "waypoint_tolerance": {
                        "type": ["number", "null"],
                        "description": "Acceptance radius in meters (typically 0.10 to 0.20 m)."
                    },
                    "path_type": {
                        "type": "string",
                        "enum": ["circle", "figure_8"],
                        "description": "Trajectory path shape for trajectory mode: 'circle' "
                                       "(params: cx, cy, radius, speed) or 'figure_8' "
                                       "(params: cx, cy, scale_x, scale_y, omega)."
                    },
                    "trajectory_params": {
                        "type": ["object", "null"],
                        "description": "Parameters of the parametric path. For 'circle': cx, cy "
                                       "(metres), radius (metres), speed (m/s). For 'figure_8': "
                                       "cx, cy (metres), scale_x, scale_y (metres), omega (rad/s). "
                                       "Omitted parameters take their defaults.",
                        "properties": {
                            "cx": {"type": ["number", "null"], "description": "Path centre X in metres."},
                            "cy": {"type": ["number", "null"], "description": "Path centre Y in metres."},
                            "radius": {"type": ["number", "null"], "description": "Circle radius in metres."},
                            "speed": {"type": ["number", "null"], "description": "Circle traversal speed in m/s."},
                            "scale_x": {"type": ["number", "null"], "description": "Figure-8 X scale in metres."},
                            "scale_y": {"type": ["number", "null"], "description": "Figure-8 Y scale in metres."},
                            "omega": {"type": ["number", "null"], "description": "Figure-8 angular rate in rad/s."},
                        },
                    },
                    "trajectory_duration": {
                        "type": ["number", "null"],
                        "description": "How long to follow the path in seconds (0.1 to 60.0 s)."
                    },
                    "gait": {
                        "type": "string",
                        "enum": ["trot", "pace", "bound", "pronk", "walk"],
                        "description": "Locomotion pattern to use for this command. Defaults to 'trot'."
                    }
                },
                "required": ["mode", "vx", "vy", "yaw_rate", "duration"]
            }
        }
    },
    "required": ["explanation", "commands"]
}


SYSTEM_PROMPT = """You are an expert quadruped motion planner for a Unitree Go2 robot in MuJoCo simulation.
Your job is to translate human natural language instructions into a sequence of safe, bounded robot motion commands.

COORDINATE CONVENTIONS (Robot Body Frame):
- +X is FORWARD, -X is BACKWARD
- +Y is LEFT, -Y is RIGHT
- +Yaw is COUNTER-CLOCKWISE (turning left), -Yaw is CLOCKWISE (turning right)

ALLOWED RANGES & SAFE LIMITS:
- vx: -0.25 to +0.50 m/s (default trot is ~0.30 m/s)
- vy: -0.30 to +0.30 m/s
- yaw_rate: -0.60 to +0.60 rad/s (default turn is ~0.40 rad/s)
- duration: 0.5 to 10.0 seconds per command
- waypoint coordinates: -10.0 to +10.0 meters

MODES:
- "velocity": Continuous trotting using vx, vy, and yaw_rate
- "stand": Stationary neutral standing (vx=0, vy=0, yaw_rate=0)
- "waypoint": 2D target waypoint navigation (waypoint_x, waypoint_y, waypoint_speed)
- "trajectory": Follow a parametric path (path_type, trajectory_params, trajectory_duration)

TRAJECTORY MODE:
- path_type is either "circle" or "figure_8"; any other value is rejected.
- For a "circle", trajectory_params may contain cx, cy (metres), radius (metres) and
  speed (m/s). For a "figure_8", it may contain cx, cy (metres), scale_x, scale_y
  (metres) and omega (rad/s). Any parameter you omit takes its default value.
- trajectory_duration is how long the path is followed, in seconds.
- The path is tracked by the deterministic trajectory tracker; you never specify
  joint angles, foot placements or the velocity sequence along the path.

GAIT:
- The optional "gait" field selects the locomotion pattern: "trot", "pace",
  "bound", "pronk" or "walk". Use "trot" for ordinary forward motion, "walk"
  for slow or careful motion. If you omit it, the robot uses "trot".

SAFETY RULES (MANDATORY):
1. Never produce actuator torques, joint angles, or low-level controller gains.
2. If asked to do something dangerous (e.g., jump off cliff, flip, run 100 m/s, damage hardware), reject it by returning mode="stand" and explain why in "explanation".
3. If the user prompt is ambiguous or empty, return mode="stand".
4. Always conclude any moving command sequence with a final "stand" command (duration=1.0) so the robot comes to a stable stop.
5. Emit valid JSON matching the requested schema.
"""


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class LLMCommandResult:
    """Result of querying the LLM and validating its motion plan."""
    user_prompt: str
    raw_response: str
    thinking: Optional[str]
    parsed_json: Optional[Dict[str, Any]]
    commands: List[MotionCommand]
    status: str                         # "success", "clamped", "rejected", "error"
    explanation: str
    error_message: Optional[str] = None
    wall_time_s: float = 0.0
    model_name: str = DEFAULT_MODEL_NAME

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_prompt": self.user_prompt,
            "model_name": self.model_name,
            "status": self.status,
            "wall_time_s": round(self.wall_time_s, 3),
            "explanation": self.explanation,
            "error_message": self.error_message,
            "thinking": self.thinking[:300] if self.thinking else None,
            "raw_response": self.raw_response,
            "commands": [cmd.to_dict() for cmd in self.commands],
        }


class AuditTrail:
    """Append-only JSONL audit writer shared by every LLM client backend.

    The audit log is a safety control, so a dropped record must never be silent:
    failures are counted (``failure_count``, assertable by callers and tests) and
    reported through a rate-limited warning instead of being swallowed.
    """

    def __init__(self, audit_file: Optional[Union[str, pathlib.Path]] = DEFAULT_AUDIT_PATH):
        self.audit_file: Optional[str] = audit_file if audit_file else None
        self.failure_count: int = 0
        self.last_error: Optional[str] = None

        if self.audit_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.audit_file)), exist_ok=True)

    def write(self, result: LLMCommandResult) -> bool:
        """Append one record to the JSONL audit trail.

        Returns True when the record was written (or auditing is disabled), False when
        the write failed.  Never raises: the query result itself is unaffected.
        """
        if not self.audit_file:
            return True
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_audit_dict()) + "\n")
            return True
        except Exception as err:
            self.failure_count += 1
            self.last_error = str(err)
            if self.failure_count == 1 or self.failure_count % AUDIT_FAILURE_WARNING_INTERVAL == 0:
                print(
                    f"[Warning] Failed to write LLM audit log "
                    f"({self.failure_count} failure(s) so far): {err}"
                )
            return False


# -----------------------------------------------------------------------------
# Text Cleaning & JSON Extraction
# -----------------------------------------------------------------------------

def strip_think_tags(text: str) -> Tuple[str, Optional[str]]:
    """Extract <think>...</think> reasoning tokens from DeepSeek-R1 responses."""
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text)
    if match:
        thinking = match.group(1).strip()
        cleaned_text = think_pattern.sub("", text).strip()
        return cleaned_text, thinking
    return text.strip(), None


def extract_json_block(text: str) -> str:
    """Extract valid JSON from markdown code fences or raw text."""
    trimmed = text.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (
        trimmed.startswith("[") and trimmed.endswith("]")
    ):
        return trimmed

    # Check for markdown ```json ... ``` code fence
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    # Find the outer-most JSON structure (either array [ ... ] or object { ... })
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    candidates = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(
            (first_brace, last_brace, text[first_brace : last_brace + 1].strip())
        )
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append(
            (first_bracket, last_bracket, text[first_bracket : last_bracket + 1].strip())
        )

    if candidates:
        # Pick the one that starts earliest
        candidates.sort(key=lambda c: c[0])
        return candidates[0][2]

    return text.strip()


# -----------------------------------------------------------------------------
# Parser and Safety Validator
# -----------------------------------------------------------------------------

def parse_and_validate_response(
    raw_content: str,
    thinking: Optional[str],
    user_prompt: str,
    bounds: SafetyBounds,
    model_name: str = DEFAULT_MODEL_NAME,
    wall_time: float = 0.0,
) -> LLMCommandResult:
    """Strictly parse and validate the LLM output into bounded MotionCommands.

    Fails closed: if any unrecoverable error occurs, returns a safe STAND command.
    """
    cleaned_content, embedded_think = strip_think_tags(raw_content)
    final_thinking = thinking or embedded_think

    json_str = extract_json_block(cleaned_content)
    if not json_str:
        return LLMCommandResult(
            user_prompt=user_prompt,
            raw_response=raw_content,
            thinking=final_thinking,
            parsed_json=None,
            commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
            status="error",
            explanation="Failed to extract JSON from LLM response. Defaulting to safe stand.",
            error_message="No JSON structure found",
            wall_time_s=wall_time,
            model_name=model_name,
        )

    try:
        # parse_constant rejects the non-standard NaN / Infinity / -Infinity
        # tokens that json.loads accepts by default.  The transport envelopes
        # already use this hook, but they cannot see *inside* the model's
        # content string, so a literal `NaN` emitted as payload text reaches
        # this parse unmolested.  Rejecting the token here means the
        # non-finite value is never constructed as a float at all, which is
        # strictly stronger than testing for finiteness downstream: there is
        # no window in which a NaN exists in a command field.
        data = json.loads(json_str, parse_constant=_reject_nonfinite_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        return LLMCommandResult(
            user_prompt=user_prompt,
            raw_response=raw_content,
            thinking=final_thinking,
            parsed_json=None,
            commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
            status="error",
            explanation=f"Malformed JSON: {exc}. Defaulting to safe stand.",
            error_message=f"JSONDecodeError: {exc}",
            wall_time_s=wall_time,
            model_name=model_name,
        )

    explanation = ""
    commands_raw = []

    if isinstance(data, list):
        commands_raw = data
        for item in data:
            if isinstance(item, dict) and "explanation" in item:
                explanation = str(item["explanation"])
                break
    elif isinstance(data, dict):
        explanation = str(data.get("explanation", ""))
        commands_raw = data.get("commands", [])
        if not isinstance(commands_raw, list) or len(commands_raw) == 0:
            if "mode" in data:
                commands_raw = [data]
            else:
                return LLMCommandResult(
                    user_prompt=user_prompt,
                    raw_response=raw_content,
                    thinking=final_thinking,
                    parsed_json=data,
                    commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                    status="rejected",
                    explanation=explanation or "No motion commands provided in response.",
                    error_message="Empty commands list",
                    wall_time_s=wall_time,
                    model_name=model_name,
                )
    else:
        return LLMCommandResult(
            user_prompt=user_prompt,
            raw_response=raw_content,
            thinking=final_thinking,
            parsed_json=None,
            commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
            status="error",
            explanation="Invalid JSON root type. Expected object or array.",
            error_message="Invalid JSON root type",
            wall_time_s=wall_time,
            model_name=model_name,
        )

    parsed_commands: List[MotionCommand] = []
    was_clamped = False

    mode_map = {
        "stand": MotionMode.STAND,
        "stop": MotionMode.STAND,
        "halt": MotionMode.STAND,
        "stay": MotionMode.STAND,
        "sit": MotionMode.STAND,
        "velocity": MotionMode.VELOCITY,
        "trot": MotionMode.VELOCITY,
        "walk": MotionMode.VELOCITY,
        "forward": MotionMode.VELOCITY,
        "backward": MotionMode.VELOCITY,
        "run": MotionMode.VELOCITY,
        "turn": MotionMode.VELOCITY,
        "waypoint": MotionMode.WAYPOINT,
        "navigate": MotionMode.WAYPOINT,
        "goto": MotionMode.WAYPOINT,
        "trajectory": MotionMode.TRAJECTORY,
    }

    # Bound the number of commands in one response.  The parse cost is linear
    # in the attacker-controlled payload, so an unbounded command list is a
    # denial-of-service surface on the control path: 20 001 commands cost
    # 67.5 ms of parsing, which is 34 control ticks at 500 Hz.  The excess is
    # truncated and the result is marked as modified, so the audit trail
    # records that the plan was not executed in full.
    if len(commands_raw) > MAX_COMMANDS_PER_RESPONSE:
        commands_raw = commands_raw[:MAX_COMMANDS_PER_RESPONSE]
        was_clamped = True

    for idx, cmd_dict in enumerate(commands_raw):
        if not isinstance(cmd_dict, dict):
            continue

        raw_mode_str = str(cmd_dict.get("mode", "stand")).lower().strip()
        requested_mode = mode_map.get(raw_mode_str)
        mode = requested_mode if requested_mode is not None else MotionMode.STAND

        try:
            vx_val = _coerce_number(cmd_dict.get("vx", 0.0))
            vy_val = _coerce_number(cmd_dict.get("vy", 0.0))
            yaw_val = _coerce_number(cmd_dict.get("yaw_rate", 0.0))
            dur_val = _coerce_number(cmd_dict.get("duration", 1.0))
        except (ValueError, TypeError, OverflowError):
            # A field that cannot be read as a number at all -- a boolean, a
            # list, an integer literal too large for a float -- makes the whole
            # command unreadable.  Fail closed to a stationary hold rather than
            # guessing a value for the offending field.
            vx_val, vy_val, yaw_val, dur_val = 0.0, 0.0, 0.0, 1.0
            mode = MotionMode.STAND
            was_clamped = True

        if requested_mode is None:
            # Unrecognized mode: fail closed to STAND *and* zero the velocities, so the
            # command is internally consistent with its own mode.  Without this a STAND
            # command could still carry vx=0.5, and any consumer that reads a velocity
            # without re-checking the mode would start moving.
            vx_val, vy_val, yaw_val = 0.0, 0.0, 0.0
            was_clamped = True

        # Check for NaN / Inf
        if any(not math.isfinite(v) for v in (vx_val, vy_val, yaw_val, dur_val)):
            vx_val, vy_val, yaw_val, dur_val = 0.0, 0.0, 0.0, 1.0
            mode = MotionMode.STAND
            was_clamped = True

        # Clamp against safety bounds
        clamped_vx = float(np_clip(vx_val, bounds.min_vx, bounds.max_vx))
        clamped_vy = float(np_clip(vy_val, -bounds.max_vy, bounds.max_vy))
        clamped_yaw = float(np_clip(yaw_val, -bounds.max_yaw_rate, bounds.max_yaw_rate))
        clamped_dur = float(np_clip(dur_val, 0.1, bounds.max_duration))

        if (
            clamped_vx != vx_val
            or clamped_vy != vy_val
            or clamped_yaw != yaw_val
            or clamped_dur != dur_val
        ):
            was_clamped = True

        # Optional waypoint extraction
        waypoint_obj: Optional[Waypoint] = None
        if mode == MotionMode.WAYPOINT:
            wp_x = cmd_dict.get("waypoint_x")
            wp_y = cmd_dict.get("waypoint_y")
            if wp_x is not None and wp_y is not None:
                try:
                    wx_raw = _coerce_number(wp_x)
                    wy_raw = _coerce_number(wp_y)
                    speed_raw = _coerce_number(cmd_dict.get("waypoint_speed", 0.30))
                    tol_raw = _coerce_number(cmd_dict.get("waypoint_tolerance", 0.15))
                except (ValueError, TypeError, OverflowError):
                    mode = MotionMode.STAND
                    was_clamped = True
                else:
                    # A non-finite waypoint coordinate is not something to clamp
                    # into range -- there is no meaningful "nearest legal
                    # waypoint" for a NaN, and a NaN that reaches the navigator
                    # poisons the slew limiter's internal state permanently.
                    # Fail closed instead: the whole command becomes a stand.
                    if not all(
                        math.isfinite(v) for v in (wx_raw, wy_raw, speed_raw, tol_raw)
                    ):
                        mode = MotionMode.STAND
                        was_clamped = True
                    else:
                        wx = float(np_clip(wx_raw, -10.0, 10.0))
                        wy = float(np_clip(wy_raw, -10.0, 10.0))
                        speed = float(np_clip(speed_raw, 0.1, bounds.max_vx))
                        tol = float(np_clip(tol_raw, 0.05, 0.50))
                        # The clip is a modification, so the result must not
                        # report plain success: the audit trail has to be able
                        # to tell "the model asked for a legal waypoint" from
                        # "the model asked for an illegal one and we moved it".
                        if (wx != wx_raw or wy != wy_raw or speed != speed_raw
                                or tol != tol_raw):
                            was_clamped = True
                        waypoint_obj = Waypoint(x=wx, y=wy, speed=speed, tolerance=tol)
            else:
                mode = MotionMode.STAND
                was_clamped = True

        # Optional parametric-trajectory extraction.
        trajectory_obj: Optional[TrajectorySpec] = None
        if mode == MotionMode.TRAJECTORY:
            raw_path_type = cmd_dict.get("path_type")
            path_type = str(raw_path_type).lower().strip() if raw_path_type is not None else ""
            if path_type not in TRAJECTORY_PATH_TYPES:
                # An unsupported path is not something to clamp into range: the
                # tracker's evaluate_path raises on a path it does not know, so a
                # request for one must never reach it.  Fail closed to a hold, the
                # same treatment an unnameable mode gets.
                mode = MotionMode.STAND
                was_clamped = True
            else:
                raw_params = cmd_dict.get("trajectory_params")
                if raw_params is None:
                    raw_params = {}
                if not isinstance(raw_params, dict):
                    # A non-object parameter block cannot be read at all, so
                    # there is no parameter to keep and none to clamp.
                    mode = MotionMode.STAND
                    was_clamped = True
                else:
                    # The circle's traversal speed is bounded by the *runtime*
                    # forward-velocity bound, so the limit follows the SafetyBounds
                    # the caller passed in rather than a module-level snapshot.
                    param_bounds = dict(TRAJECTORY_PARAM_BOUNDS[path_type])
                    if path_type == "circle":
                        param_bounds["speed"] = (0.0, bounds.max_vx)
                    defaults = TRAJECTORY_PARAM_DEFAULTS[path_type]
                    params: Dict[str, float] = {}
                    param_ok = True
                    for name, (low, high) in param_bounds.items():
                        raw_value = raw_params.get(name, defaults[name])
                        try:
                            value_raw = _coerce_number(raw_value)
                        except (ValueError, TypeError, OverflowError):
                            param_ok = False
                            break
                        if not math.isfinite(value_raw):
                            # A non-finite path parameter is not something to clamp
                            # into range: the comparison-based clip passes a NaN
                            # through unchanged, and a NaN inside the tracker would
                            # poison the command it produces.  Fail closed, exactly
                            # as for a non-finite waypoint coordinate.
                            param_ok = False
                            break
                        value = float(np_clip(value_raw, low, high))
                        if value != value_raw:
                            # The clip is a modification, so the result must not
                            # report plain success: the audit trail has to be able
                            # to tell "the model asked for a legal path" from "the
                            # model asked for an illegal one and we moved it".
                            was_clamped = True
                        params[name] = value
                    if not param_ok:
                        mode = MotionMode.STAND
                        was_clamped = True
                    else:
                        # The trajectory duration is a command duration like any
                        # other, so it is bounded by the same two limits.
                        try:
                            traj_dur_raw = _coerce_number(
                                cmd_dict.get("trajectory_duration", TRAJECTORY_DEFAULT_DURATION)
                            )
                        except (ValueError, TypeError, OverflowError):
                            traj_dur_raw = None
                        if traj_dur_raw is None or not math.isfinite(traj_dur_raw):
                            mode = MotionMode.STAND
                            was_clamped = True
                        else:
                            traj_dur = float(np_clip(traj_dur_raw, 0.1, bounds.max_duration))
                            if traj_dur != traj_dur_raw:
                                was_clamped = True
                            trajectory_obj = TrajectorySpec(
                                path_type=path_type,
                                params=params,
                                duration=traj_dur,
                            )

        # Gait selection.  An unknown gait is a *default*, not a hazard: the
        # request names a locomotion pattern, and the stack's own default (trot)
        # is a legal, well-tested pattern, so the command is still executed with a
        # sane gait instead of being discarded.  It is therefore reported as a
        # modification (status "clamped") rather than failing the command to a
        # stand -- unlike the trajectory path type above, where an unknown path
        # leaves the command with nothing to execute.
        gait = DEFAULT_GAIT
        raw_gait = cmd_dict.get("gait")
        if raw_gait is not None:
            gait_candidate = str(raw_gait).lower().strip()
            if gait_candidate in GAIT_TYPES:
                gait = gait_candidate
            else:
                gait = DEFAULT_GAIT
                was_clamped = True

        heading_target = cmd_dict.get("heading_target")
        if heading_target is not None:
            try:
                heading_target = _coerce_number(heading_target)
            except (ValueError, TypeError, OverflowError):
                heading_target = None
                was_clamped = True
            else:
                if math.isfinite(heading_target):
                    # Headings are angles: normalize to [-pi, pi] as the controller does.
                    wrapped_heading = wrap_to_pi(heading_target)
                    if wrapped_heading != heading_target:
                        was_clamped = True
                    heading_target = wrapped_heading
                else:
                    heading_target = None
                    was_clamped = True

        # Mode/velocity consistency: a STAND command must not carry a velocity
        # triple.  The unknown-mode branch above already enforces this, but the
        # waypoint-demotion branches did not, so a command labelled "stand"
        # could still advertise vx=0.3.  The controller happens to override the
        # velocities for STAND mode at the actuation site, but any other
        # consumer that reads cmd.vx without re-checking cmd.mode (the
        # run_meta.json writer is exactly such a consumer) would act on a
        # velocity the command does not authorise.  Enforced here, once, so
        # every path that can produce a STAND mode is covered.
        if mode == MotionMode.STAND and (
            clamped_vx != 0.0 or clamped_vy != 0.0 or clamped_yaw != 0.0
        ):
            clamped_vx, clamped_vy, clamped_yaw = 0.0, 0.0, 0.0
            was_clamped = True

        parsed_cmd = MotionCommand(
            vx=clamped_vx,
            vy=clamped_vy,
            yaw_rate=clamped_yaw,
            duration=clamped_dur,
            mode=mode,
            heading_target=heading_target,
            waypoint=waypoint_obj,
            trajectory=trajectory_obj,
            gait=gait,
        )
        parsed_commands.append(parsed_cmd)

    if not parsed_commands:
        parsed_commands.append(MotionCommand(mode=MotionMode.STAND, duration=1.0))

    # Safety convention: if the sequence moves, ensure it ends with a safe
    # stand.  The test is on the MODE, not on the velocity triple: a WAYPOINT
    # command carries zero velocity fields -- its motion is produced by the
    # waypoint navigator, not by vx/vy/yaw_rate -- so a velocity-magnitude test
    # let a waypoint plan end without a stand, contradicting the invariant the
    # system prompt promises the model.  The same holds for a TRAJECTORY
    # command, whose motion comes from the trajectory tracker.
    last_cmd = parsed_commands[-1]
    if last_cmd.mode != MotionMode.STAND:
        # The trailing stand inherits the last command's gait.  The controller
        # applies the gait of whichever command is active, and a stand that
        # defaulted to "trot" would cancel an in-flight gait transition back to
        # the trot profile the moment the robot stopped.
        parsed_commands.append(
            MotionCommand(mode=MotionMode.STAND, duration=1.0, gait=last_cmd.gait)
        )

    status = "clamped" if was_clamped else "success"
    return LLMCommandResult(
        user_prompt=user_prompt,
        raw_response=raw_content,
        thinking=final_thinking,
        parsed_json=data,
        commands=parsed_commands,
        status=status,
        explanation=explanation,
        error_message=None,
        wall_time_s=wall_time,
        model_name=model_name,
    )


#: Maximum number of commands accepted from a single model response.  The
#: parse cost is linear in the payload, so an unbounded command list is a
#: denial-of-service surface on the control path.
MAX_COMMANDS_PER_RESPONSE = 64


def _coerce_number(value: Any) -> float:
    """Coerce a JSON scalar to a float, rejecting the cases that must fail closed.

    Three hazards are handled here rather than at the call sites, because each
    of them turns an unreadable field into a *plausible* number:

    * A JSON boolean is not a number.  ``float(True)`` is ``1.0``, so a payload
      of ``"vx": true`` would otherwise be read as a request for 1.0\\,m/s --
      and then clamped to a legal-looking value -- rather than rejected.
    * An integer literal too large for a float raises ``OverflowError`` inside
      ``float()``.  ``json.loads`` happily produces a 400-digit Python ``int``,
      whose magnitude exceeds the float range, so the conversion can raise.
      Left uncaught, that exception escapes the validator entirely and the
      contract is no longer total.
    * A numeric *string* is accepted, because providers sometimes quote
      numbers.  The magnitude is still bounded: an out-of-range string
      saturates to ``inf`` and is caught by the caller's finite check.

    Non-finite results are returned rather than raised, so that the caller's
    single finite check can apply its one fail-closed policy (demote the
    command to a stationary hold) to every field uniformly.
    """
    if isinstance(value, bool):
        raise ValueError("a JSON boolean is not a number")
    return float(value)


def np_clip(val: float, low: float, high: float) -> float:
    """Clips a value to [low, high] without requiring heavy numpy imports.

    Non-finite inputs are mapped to ``low`` rather than passed through.  The
    comparison-based clip is not NaN-safe on its own: both ``nan < low`` and
    ``nan > high`` are False, so a NaN would survive the clip unchanged and
    poison every downstream consumer (the waypoint navigator, the slew
    limiter's internal state).  Callers that need to *reject* a non-finite
    value rather than saturate it must test finiteness explicitly; the
    validator does so for the waypoint scalars.
    """
    if not math.isfinite(val):
        return low
    if val < low:
        return low
    if val > high:
        return high
    return val


def _reject_nonfinite_json_constant(token: str) -> float:
    """`json.loads` hook rejecting the non-standard NaN/Infinity tokens it accepts by default."""
    raise ValueError(f"Non-finite JSON constant '{token}' is not permitted")


def _server_response_to_validator_json(resp_json: Any) -> str:
    """Translate a Live Model Server response into the shared validator's flat schema.

    The server serializes commands with ``MotionCommand.to_dict()`` (a nested ``waypoint``
    object), while ``parse_and_validate_response`` expects flat ``waypoint_*`` keys.
    Flattening here lets the server response reuse the exact same validation pipeline as
    every other backend instead of hand-building MotionCommands from untrusted JSON.
    """
    if not isinstance(resp_json, dict):
        return json.dumps(resp_json)

    normalized: Dict[str, Any] = dict(resp_json)
    commands = normalized.get("commands")
    if isinstance(commands, list):
        flattened: List[Any] = []
        for cmd in commands:
            if not isinstance(cmd, dict):
                flattened.append(cmd)
                continue
            cmd = dict(cmd)
            waypoint = cmd.pop("waypoint", None)
            if isinstance(waypoint, dict):
                cmd.setdefault("waypoint_x", waypoint.get("x"))
                cmd.setdefault("waypoint_y", waypoint.get("y"))
                cmd.setdefault("waypoint_speed", waypoint.get("speed"))
                cmd.setdefault("waypoint_tolerance", waypoint.get("tolerance"))
                if cmd.get("heading_target") is None and waypoint.get("heading") is not None:
                    cmd["heading_target"] = waypoint["heading"]
            flattened.append(cmd)
        normalized["commands"] = flattened

    # allow_nan=False makes any non-finite value a hard error rather than invalid JSON output.
    return json.dumps(normalized, allow_nan=False)


# -----------------------------------------------------------------------------
# Ollama DeepSeek-R1 Client
# -----------------------------------------------------------------------------

class OllamaDeepSeekClient:
    """Client for local Ollama running deepseek-r1:8b using standard library."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL_NAME,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        audit_file: Optional[Union[str, pathlib.Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.bounds = bounds or SafetyBounds()
        self.audit_trail = AuditTrail(audit_file)
        self.audit_file = self.audit_trail.audit_file

    def check_availability(self) -> Tuple[bool, str]:
        """Check if local Ollama is reachable and deepseek-r1:8b is loaded."""
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Go2-CHAMP/1.0"})
            with urllib.request.urlopen(req, timeout=AVAILABILITY_PROBE_TIMEOUT_S) as resp:
                if resp.status != 200:
                    return False, f"Ollama returned HTTP status {resp.status}"
                data = json.loads(
                    resp.read().decode("utf-8"),
                    parse_constant=_reject_nonfinite_json_constant,
                )
                models = [m.get("name", "") for m in data.get("models", [])]
                # Check exact or prefix match (e.g., deepseek-r1:8b or deepseek-r1:8b-q4)
                if any(self.model in m or m.startswith(self.model) for m in models):
                    return True, f"Model '{self.model}' is available."
                return False, f"Model '{self.model}' not found in installed models: {models}"
        except urllib.error.URLError as err:
            return False, f"Could not connect to Ollama at {self.host}: {err.reason}"
        except Exception as err:
            return False, f"Error checking Ollama availability: {err}"

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMCommandResult:
        """Send a natural language prompt to Ollama DeepSeek-R1 and return parsed commands."""
        sys_prompt = system_prompt or SYSTEM_PROMPT
        url = f"{self.host}/api/chat"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": MOTION_COMMAND_SCHEMA,
            "options": {
                "temperature": 0.0,
            }
        }

        req_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Go2-CHAMP-Client/1.0",
        }

        t_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                t_elapsed = time.perf_counter() - t_start
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP Error {resp.status}", resp.headers, None
                    )
                resp_json = json.loads(
                    resp.read().decode("utf-8"),
                    parse_constant=_reject_nonfinite_json_constant,
                )
                message = resp_json.get("message", {})
                raw_content = message.get("content", "")
                thinking = message.get("thinking", None)

                result = parse_and_validate_response(
                    raw_content=raw_content,
                    thinking=thinking,
                    user_prompt=user_prompt,
                    bounds=self.bounds,
                    model_name=self.model,
                    wall_time=t_elapsed,
                )
        except urllib.error.URLError as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Ollama network error: {err.reason}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )
        except Exception as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Unexpected error: {err}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )

        self.audit_trail.write(result)
        return result


# -----------------------------------------------------------------------------
# Google Gemini / Gemma Client (Google AI Studio REST API)
# -----------------------------------------------------------------------------

class GeminiClient:
    """Client for Google AI Studio / Gemini / Gemma models using standard library."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        audit_file: Optional[Union[str, pathlib.Path]] = DEFAULT_AUDIT_PATH,
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
        self.audit_trail = AuditTrail(audit_file)
        self.audit_file = self.audit_trail.audit_file

    def check_availability(self) -> Tuple[bool, str]:
        """Check if Gemini API key is configured."""
        if not self.api_key:
            return (
                False,
                "GEMINI_API_KEY not set. Add `GEMINI_API_KEY=your_key` to a `.env` file, set the environment variable, or pass `--gemini-api-key <key>`.",
            )
        return True, f"Gemini API key detected. Using model '{self.model}'."

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMCommandResult:
        """Send prompt to Google AI Studio REST API and return parsed commands."""
        if not self.api_key:
            res = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation="Gemini API key is missing. Set GEMINI_API_KEY or use --gemini-api-key.",
                error_message="Missing GEMINI_API_KEY",
                wall_time_s=0.0,
                model_name=self.model,
            )
            self.audit_trail.write(res)
            return res

        sys_prompt = system_prompt or SYSTEM_PROMPT
        url = f"{DEFAULT_GEMINI_API_URL}/models/{self.model}:generateContent"

        payload = {
            "systemInstruction": {
                "parts": [{"text": sys_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            }
        }

        req_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Go2-CHAMP-Gemini/1.0",
            # Credentials belong in a header, never in the URL query string (proxies log URLs).
            "x-goog-api-key": self.api_key,
        }

        t_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                t_elapsed = time.perf_counter() - t_start
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP Error {resp.status}", resp.headers, None
                    )
                resp_json = json.loads(
                    resp.read().decode("utf-8"),
                    parse_constant=_reject_nonfinite_json_constant,
                )

                candidates = resp_json.get("candidates", [])
                raw_content = ""
                thinking = None
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    raw_content = "".join(p.get("text", "") for p in parts)

                result = parse_and_validate_response(
                    raw_content=raw_content,
                    thinking=thinking,
                    user_prompt=user_prompt,
                    bounds=self.bounds,
                    model_name=self.model,
                    wall_time=t_elapsed,
                )
        except urllib.error.HTTPError as err:
            t_elapsed = time.perf_counter() - t_start
            err_body = (
                err.read().decode("utf-8", errors="replace")
                if hasattr(err, "read")
                else str(err)
            )
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response=err_body,
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Gemini API HTTP {err.code}: {err.reason}. Failing closed to safe stand.",
                error_message=f"HTTPError {err.code}: {err_body}",
                wall_time_s=t_elapsed,
                model_name=self.model,
            )
        except urllib.error.URLError as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Gemini network error: {err.reason}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )
        except Exception as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Unexpected error: {err}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )

        self.audit_trail.write(result)
        return result


# -----------------------------------------------------------------------------
# OmniRoute / OpenAI-Compatible Gateway Client
# -----------------------------------------------------------------------------

class OmniRouteClient:
    """Client for OmniRoute / LiteLLM / OpenAI-compatible model routing proxies."""

    def __init__(
        self,
        base_url: str = DEFAULT_OMNIROUTE_URL,
        model: str = DEFAULT_OMNIROUTE_MODEL,
        api_key: str = DEFAULT_OMNIROUTE_API_KEY,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        audit_file: Optional[Union[str, pathlib.Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
    ):
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/v1"):
            clean_url = f"{clean_url}/v1"
        self.base_url = clean_url
        self.model = model
        self.api_key = (api_key or DEFAULT_OMNIROUTE_API_KEY).strip()
        self.timeout_s = timeout_s
        self.bounds = bounds or SafetyBounds()
        self.audit_trail = AuditTrail(audit_file)
        self.audit_file = self.audit_trail.audit_file

    def check_availability(self) -> Tuple[bool, str]:
        """Check if the OmniRoute / OpenAI-compatible endpoint is reachable."""
        models_url = f"{self.base_url}/models"
        try:
            headers = {"User-Agent": "Go2-CHAMP/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(models_url, headers=headers)
            with urllib.request.urlopen(req, timeout=AVAILABILITY_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200:
                    return True, f"OmniRoute endpoint reachable at {self.base_url}. Active model: '{self.model}'."
                return False, f"OmniRoute endpoint returned HTTP {resp.status}"
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                return True, f"OmniRoute endpoint detected at {self.base_url} (HTTP {err.code}). Using model '{self.model}'."
            return False, f"OmniRoute HTTP {err.code}: {err.reason}"
        except urllib.error.URLError as err:
            return False, f"Could not connect to OmniRoute at {self.base_url}: {err.reason}"
        except Exception as err:
            return False, f"Error checking OmniRoute availability: {err}"

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMCommandResult:
        """Send prompt to OmniRoute /chat/completions and return parsed commands."""
        sys_prompt = system_prompt or SYSTEM_PROMPT
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        req_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Go2-CHAMP-OmniRoute/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                t_elapsed = time.perf_counter() - t_start
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP Error {resp.status}", resp.headers, None
                    )
                resp_json = json.loads(
                    resp.read().decode("utf-8"),
                    parse_constant=_reject_nonfinite_json_constant,
                )
                choices = resp_json.get("choices", [])
                raw_content = ""
                thinking = None
                if choices:
                    msg = choices[0].get("message", {})
                    raw_content = msg.get("content", "")
                    thinking = msg.get("reasoning_content") or msg.get("reasoning")

                result = parse_and_validate_response(
                    raw_content=raw_content,
                    thinking=thinking,
                    user_prompt=user_prompt,
                    bounds=self.bounds,
                    model_name=self.model,
                    wall_time=t_elapsed,
                )
        except urllib.error.HTTPError as err:
            t_elapsed = time.perf_counter() - t_start
            err_body = (
                err.read().decode("utf-8", errors="replace")
                if hasattr(err, "read")
                else str(err)
            )
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response=err_body,
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"OmniRoute HTTP {err.code}: {err.reason}. Failing closed to safe stand.",
                error_message=f"HTTPError {err.code}: {err_body}",
                wall_time_s=t_elapsed,
                model_name=self.model,
            )
        except urllib.error.URLError as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"OmniRoute network error: {err.reason}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )
        except Exception as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Unexpected error: {err}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name=self.model,
            )

        self.audit_trail.write(result)
        return result


# -----------------------------------------------------------------------------
# Live Model Service Client (Connects to start_model_server.py)
# -----------------------------------------------------------------------------

class LiveModelServiceClient:
    """Client that communicates with the dedicated local start_model_server.py process."""

    def __init__(
        self,
        server_url: str = DEFAULT_MODEL_SERVER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        audit_file: Optional[Union[str, pathlib.Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
        auth_token: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.bounds = bounds or SafetyBounds()
        # Shared secret for POST /query. An explicit argument overrides; otherwise mirror
        # the server's own env var so both ends of the local link agree without extra
        # configuration. No token configured => no auth header is sent at all.
        token = auth_token if auth_token is not None else os.environ.get(MODEL_SERVER_TOKEN_ENV_VAR, "")
        self.auth_token: Optional[str] = (token or "").strip() or None
        self.audit_trail = AuditTrail(audit_file)
        self.audit_file = self.audit_trail.audit_file

    def check_availability(self) -> Tuple[bool, str]:
        """Check if the live model server process is active and ready."""
        url = f"{self.server_url}/health"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Go2-CHAMP/1.0"})
            with urllib.request.urlopen(req, timeout=AVAILABILITY_PROBE_TIMEOUT_S) as resp:
                if resp.status == 200:
                    data = json.loads(
                        resp.read().decode("utf-8"),
                        parse_constant=_reject_nonfinite_json_constant,
                    )
                    model = data.get("model", "unknown")
                    backend = data.get("backend", "unknown")
                    return True, f"Connected to Live Model Server at {self.server_url} (Model: '{model}' via {backend})."
                return False, f"Live Model Server returned HTTP {resp.status}"
        except urllib.error.URLError as err:
            return False, f"Could not connect to Live Model Server at {self.server_url}: {err.reason}"
        except Exception as err:
            return False, f"Error checking Live Model Server availability: {err}"

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMCommandResult:
        """Send prompt to the live model server /query endpoint."""
        url = f"{self.server_url}/query"
        payload = {
            "prompt": user_prompt,
            "system_prompt": system_prompt,
        }
        req_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Go2-CHAMP-Client/1.0",
        }
        if self.auth_token:
            # Header, never a query string: URLs are logged by proxies and gateways.
            headers["Authorization"] = f"Bearer {self.auth_token}"

        t_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                t_elapsed = time.perf_counter() - t_start
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP Error {resp.status}", resp.headers, None
                    )
                resp_json = json.loads(
                    resp.read().decode("utf-8"),
                    parse_constant=_reject_nonfinite_json_constant,
                )

                try:
                    server_wall_time = float(resp_json.get("wall_time_s", t_elapsed))
                except (TypeError, ValueError):
                    server_wall_time = t_elapsed

                # Route the server payload through the shared validator: the server is
                # untrusted input, so every command is bounds-checked and finite-checked
                # exactly like the direct Ollama / Gemini / OmniRoute backends.
                result = parse_and_validate_response(
                    raw_content=_server_response_to_validator_json(resp_json),
                    thinking=resp_json.get("thinking"),
                    user_prompt=user_prompt,
                    bounds=self.bounds,
                    model_name=str(resp_json.get("model_name") or "live-model-server"),
                    wall_time=server_wall_time,
                )
                if isinstance(resp_json.get("raw_response"), str) and resp_json["raw_response"]:
                    # Preserve the server's raw model output for the audit trail.
                    result.raw_response = resp_json["raw_response"]
        except urllib.error.HTTPError as err:
            # A 401 arrives as a raised HTTPError, so it never reaches the status check
            # above.  Name it explicitly instead of reporting a generic connection error,
            # and fail closed without retrying or re-sending the request unauthenticated.
            t_elapsed = time.perf_counter() - t_start
            if err.code == 401:
                reason = (
                    "Live Model Server rejected the request: HTTP 401 unauthorized "
                    f"(missing or invalid shared secret). Set {MODEL_SERVER_TOKEN_ENV_VAR} "
                    "or pass auth_token=... to match the server's --token."
                )
            else:
                reason = f"Live Model Server returned HTTP {err.code}: {err.reason}."
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"{reason} Failing closed to safe stand.",
                error_message=reason,
                wall_time_s=t_elapsed,
                model_name="live-model-server",
            )
        except urllib.error.URLError as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Live Model Server connection error: {err.reason}. Failing closed to safe stand.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name="live-model-server",
            )
        except Exception as err:
            t_elapsed = time.perf_counter() - t_start
            result = LLMCommandResult(
                user_prompt=user_prompt,
                raw_response="",
                thinking=None,
                parsed_json=None,
                commands=[MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                status="error",
                explanation=f"Unexpected error communicating with Live Model Server: {err}. Failing closed.",
                error_message=str(err),
                wall_time_s=t_elapsed,
                model_name="live-model-server",
            )

        self.audit_trail.write(result)
        return result


# -----------------------------------------------------------------------------
# Mock Client for Offline Testing and CI
# -----------------------------------------------------------------------------

class MockDeepSeekClient:
    """Mock DeepSeek client that produces deterministic responses without Ollama."""

    def __init__(
        self,
        audit_file: Optional[Union[str, pathlib.Path]] = None,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.audit_file = audit_file if audit_file else None
        self.bounds = bounds or SafetyBounds()
        self.custom_responses: Dict[str, str] = {}

    def check_availability(self) -> Tuple[bool, str]:
        return True, "Mock DeepSeek client is ready (offline)."

    def register_response(self, prompt: str, json_response: str):
        self.custom_responses[prompt.strip().lower()] = json_response

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMCommandResult:
        p_clean = user_prompt.strip().lower()

        # Check explicit custom responses
        if p_clean in self.custom_responses:
            raw = self.custom_responses[p_clean]
            return parse_and_validate_response(
                raw_content=raw,
                thinking="[Mock reasoning]",
                user_prompt=user_prompt,
                bounds=self.bounds,
                model_name="mock-deepseek-r1",
                wall_time=0.01,
            )

        # Rule-based heuristics for common prompts
        if "forward" in p_clean or "walk" in p_clean or "trot" in p_clean:
            # Parse speed if given
            speed = 0.30
            m_spd = re.search(r"(\d+\.?\d*)\s*m/s", p_clean)
            if m_spd:
                speed = float(m_spd.group(1))

            duration = 2.0
            m_dur = re.search(r"(\d+\.?\d*)\s*(?:s|sec|seconds)", p_clean)
            if m_dur:
                duration = float(m_dur.group(1))

            raw = json.dumps({
                "explanation": f"Trotting forward at {speed} m/s for {duration} s, then stopping.",
                "commands": [
                    {"mode": "velocity", "vx": speed, "vy": 0.0, "yaw_rate": 0.0, "duration": duration},
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })
        elif "backward" in p_clean or "reverse" in p_clean:
            raw = json.dumps({
                "explanation": "Trotting backward at -0.20 m/s for 2 s, then stopping.",
                "commands": [
                    {"mode": "velocity", "vx": -0.20, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0},
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })
        elif "turn left" in p_clean:
            raw = json.dumps({
                "explanation": "Turning left in place at +0.40 rad/s for 2.5 s.",
                "commands": [
                    {"mode": "velocity", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.40, "duration": 2.5},
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })
        elif "turn right" in p_clean:
            raw = json.dumps({
                "explanation": "Turning right in place at -0.40 rad/s for 2.5 s.",
                "commands": [
                    {"mode": "velocity", "vx": 0.0, "vy": 0.0, "yaw_rate": -0.40, "duration": 2.5},
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })
        elif "stop" in p_clean or "stand" in p_clean or "halt" in p_clean:
            raw = json.dumps({
                "explanation": "Stationary stand hold.",
                "commands": [
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}
                ]
            })
        elif "waypoint" in p_clean or "goto" in p_clean or "navigate" in p_clean:
            # Look for coordinates (x, y)
            coords = re.findall(r"[-+]?\d*\.?\d+", p_clean)
            wx = float(coords[0]) if len(coords) > 0 else 1.0
            wy = float(coords[1]) if len(coords) > 1 else 1.0
            raw = json.dumps({
                "explanation": f"Navigating to waypoint ({wx}, {wy}).",
                "commands": [
                    {
                        "mode": "waypoint",
                        "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0,
                        "duration": 15.0,
                        "waypoint_x": wx,
                        "waypoint_y": wy,
                        "waypoint_speed": 0.30,
                        "waypoint_tolerance": 0.15
                    },
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })
        elif "flip" in p_clean or "jump" in p_clean or "hack" in p_clean:
            # Adversarial / unsafe request
            raw = json.dumps({
                "explanation": "Request rejected: Action violates safety boundaries. Staying in safe stand.",
                "commands": [
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 2.0}
                ]
            })
        else:
            raw = json.dumps({
                "explanation": "Unrecognized motion prompt. Safely standing.",
                "commands": [
                    {"mode": "stand", "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0, "duration": 1.0}
                ]
            })

        return parse_and_validate_response(
            raw_content=raw,
            thinking="[Mock reasoning]",
            user_prompt=user_prompt,
            bounds=self.bounds,
            model_name="mock-deepseek-r1",
            wall_time=0.01,
        )


# -----------------------------------------------------------------------------
# Module CLI Smoke Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Checking Ollama availability...")
    client = OllamaDeepSeekClient()
    available, msg = client.check_availability()
    print(f"Status: {msg}")

    test_prompt = "Walk forward at 0.35 m/s for 2.5 seconds, then stop"
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])

    print(f"\nQuerying: '{test_prompt}'")
    if available:
        res = client.query(test_prompt)
    else:
        print("Using MockDeepSeekClient fallback for testing...")
        mock_client = MockDeepSeekClient()
        res = mock_client.query(test_prompt)

    print(f"Status: {res.status}")
    print(f"Explanation: {res.explanation}")
    print(f"Commands ({len(res.commands)}):")
    for i, c in enumerate(res.commands):
        print(f"  [{i}] mode={c.mode} vx={c.vx:.2f} vy={c.vy:.2f} yaw={c.yaw_rate:.2f} dur={c.duration:.1f}s")
