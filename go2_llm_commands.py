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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from go2_champ_control import (
    MotionCommand,
    MotionMode,
    SafetyBounds,
    Waypoint,
)


def load_dotenv(dotenv_path: Optional[Union[str, Path]] = None) -> None:
    """Load key-value pairs from a .env file into os.environ (if not already set)."""
    if dotenv_path is None:
        # Check current working directory first, then root directory of the project
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent / ".env",
        ]
    else:
        candidates = [Path(dotenv_path)]

    for p in candidates:
        if p.exists() and p.is_file():
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
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_AUDIT_PATH = Path(__file__).resolve().parent / "logs" / "llm_audit.jsonl"


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
                        "enum": ["stand", "velocity", "waypoint"],
                        "description": "Motion mode: 'stand' (stop/stay), 'velocity' (trot at vx, vy, yaw_rate), 'waypoint' (navigate to x, y)."
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
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
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
    }

    for idx, cmd_dict in enumerate(commands_raw):
        if not isinstance(cmd_dict, dict):
            continue

        raw_mode_str = str(cmd_dict.get("mode", "stand")).lower().strip()
        mode = mode_map.get(raw_mode_str, MotionMode.STAND)

        try:
            vx_val = float(cmd_dict.get("vx", 0.0))
            vy_val = float(cmd_dict.get("vy", 0.0))
            yaw_val = float(cmd_dict.get("yaw_rate", 0.0))
            dur_val = float(cmd_dict.get("duration", 1.0))
        except (ValueError, TypeError):
            vx_val, vy_val, yaw_val, dur_val = 0.0, 0.0, 0.0, 1.0
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
                    wx = float(np_clip(float(wp_x), -10.0, 10.0))
                    wy = float(np_clip(float(wp_y), -10.0, 10.0))
                    speed = float(np_clip(float(cmd_dict.get("waypoint_speed", 0.30)), 0.1, bounds.max_vx))
                    tol = float(np_clip(float(cmd_dict.get("waypoint_tolerance", 0.15)), 0.05, 0.50))
                    waypoint_obj = Waypoint(x=wx, y=wy, speed=speed, tolerance=tol)
                except (ValueError, TypeError):
                    mode = MotionMode.STAND
                    was_clamped = True
            else:
                mode = MotionMode.STAND
                was_clamped = True

        heading_target = cmd_dict.get("heading_target")
        if heading_target is not None:
            try:
                heading_target = float(heading_target)
            except (ValueError, TypeError):
                heading_target = None

        parsed_cmd = MotionCommand(
            vx=clamped_vx,
            vy=clamped_vy,
            yaw_rate=clamped_yaw,
            duration=clamped_dur,
            mode=mode,
            heading_target=heading_target,
            waypoint=waypoint_obj,
        )
        parsed_commands.append(parsed_cmd)

    if not parsed_commands:
        parsed_commands.append(MotionCommand(mode=MotionMode.STAND, duration=1.0))

    # Safety convention: if sequence has movement, ensure it ends with a safe stand
    last_cmd = parsed_commands[-1]
    if last_cmd.mode != MotionMode.STAND and (
        abs(last_cmd.vx) > 0.01 or abs(last_cmd.vy) > 0.01 or abs(last_cmd.yaw_rate) > 0.02
    ):
        parsed_commands.append(MotionCommand(mode=MotionMode.STAND, duration=1.0))

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


def np_clip(val: float, low: float, high: float) -> float:
    """Clips a value to [low, high] without requiring heavy numpy imports."""
    if val < low:
        return low
    if val > high:
        return high
    return val


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
        audit_file: Optional[Union[str, Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.audit_file = Path(audit_file) if audit_file else None
        self.bounds = bounds or SafetyBounds()

        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def check_availability(self) -> Tuple[bool, str]:
        """Check if local Ollama is reachable and deepseek-r1:8b is loaded."""
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Go2-CHAMP/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status != 200:
                    return False, f"Ollama returned HTTP status {resp.status}"
                data = json.loads(resp.read().decode("utf-8"))
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
                resp_json = json.loads(resp.read().decode("utf-8"))
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

        self._write_audit(result)
        return result

    def _write_audit(self, result: LLMCommandResult):
        """Append the query and result to the JSONL audit trail."""
        if not self.audit_file:
            return
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_audit_dict()) + "\n")
        except Exception as err:
            print(f"[Warning] Failed to write LLM audit log: {err}")


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
        audit_file: Optional[Union[str, Path]] = DEFAULT_AUDIT_PATH,
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
        self.audit_file = Path(audit_file) if audit_file else None
        self.bounds = bounds or SafetyBounds()

        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)

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
            self._write_audit(res)
            return res

        sys_prompt = system_prompt or SYSTEM_PROMPT
        url = f"{DEFAULT_GEMINI_API_URL}/models/{self.model}:generateContent?key={self.api_key}"

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
                resp_json = json.loads(resp.read().decode("utf-8"))

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

        self._write_audit(result)
        return result

    def _write_audit(self, result: LLMCommandResult):
        """Append the query and result to the JSONL audit trail."""
        if not self.audit_file:
            return
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_audit_dict()) + "\n")
        except Exception as err:
            print(f"[Warning] Failed to write LLM audit log: {err}")


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
        audit_file: Optional[Union[str, Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
    ):
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/v1"):
            clean_url = f"{clean_url}/v1"
        self.base_url = clean_url
        self.model = model
        self.api_key = (api_key or DEFAULT_OMNIROUTE_API_KEY).strip()
        self.timeout_s = timeout_s
        self.audit_file = Path(audit_file) if audit_file else None
        self.bounds = bounds or SafetyBounds()

        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def check_availability(self) -> Tuple[bool, str]:
        """Check if the OmniRoute / OpenAI-compatible endpoint is reachable."""
        models_url = f"{self.base_url}/models"
        try:
            headers = {"User-Agent": "Go2-CHAMP/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(models_url, headers=headers)
            with urllib.request.urlopen(req, timeout=5.0) as resp:
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
                resp_json = json.loads(resp.read().decode("utf-8"))
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

        self._write_audit(result)
        return result

    def _write_audit(self, result: LLMCommandResult):
        """Append the query and result to the JSONL audit trail."""
        if not self.audit_file:
            return
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_audit_dict()) + "\n")
        except Exception as err:
            print(f"[Warning] Failed to write LLM audit log: {err}")


# -----------------------------------------------------------------------------
# Live Model Service Client (Connects to start_model_server.py)
# -----------------------------------------------------------------------------

class LiveModelServiceClient:
    """Client that communicates with the dedicated local start_model_server.py process."""

    def __init__(
        self,
        server_url: str = DEFAULT_MODEL_SERVER_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        audit_file: Optional[Union[str, Path]] = DEFAULT_AUDIT_PATH,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.audit_file = Path(audit_file) if audit_file else None
        self.bounds = bounds or SafetyBounds()

        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def check_availability(self) -> Tuple[bool, str]:
        """Check if the live model server process is active and ready."""
        url = f"{self.server_url}/health"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Go2-CHAMP/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
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

        t_start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=req_body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                t_elapsed = time.perf_counter() - t_start
                if resp.status != 200:
                    raise urllib.error.HTTPError(
                        url, resp.status, f"HTTP Error {resp.status}", resp.headers, None
                    )
                resp_json = json.loads(resp.read().decode("utf-8"))

                # Parse motion commands
                parsed_commands: List[MotionCommand] = []
                for c in resp_json.get("commands", []):
                    wp = None
                    if c.get("waypoint"):
                        wp_dict = c["waypoint"]
                        wp = Waypoint(
                            x=float(wp_dict.get("x", 0.0)),
                            y=float(wp_dict.get("y", 0.0)),
                            heading=wp_dict.get("heading"),
                            tolerance=float(wp_dict.get("tolerance", 0.10)),
                            speed=float(wp_dict.get("speed", 0.30)),
                        )
                    mode_val = c.get("mode", "stand")
                    try:
                        mode_enum = MotionMode(mode_val)
                    except Exception:
                        mode_enum = MotionMode.STAND

                    parsed_commands.append(
                        MotionCommand(
                            vx=float(c.get("vx", 0.0)),
                            vy=float(c.get("vy", 0.0)),
                            yaw_rate=float(c.get("yaw_rate", 0.0)),
                            duration=float(c.get("duration", 1.0)),
                            mode=mode_enum,
                            heading_target=c.get("heading_target"),
                            waypoint=wp,
                        )
                    )

                result = LLMCommandResult(
                    user_prompt=user_prompt,
                    raw_response=resp_json.get("raw_response", ""),
                    thinking=resp_json.get("thinking"),
                    parsed_json=resp_json.get("parsed_json"),
                    commands=parsed_commands or [MotionCommand(mode=MotionMode.STAND, duration=1.0)],
                    status=resp_json.get("status", "success"),
                    explanation=resp_json.get("explanation", "Executed by Live Model Server."),
                    error_message=resp_json.get("error_message"),
                    wall_time_s=resp_json.get("wall_time_s", t_elapsed),
                    model_name=resp_json.get("model_name", "live-model-server"),
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

        self._write_audit(result)
        return result

    def _write_audit(self, result: LLMCommandResult):
        """Append the query and result to the JSONL audit trail."""
        if not self.audit_file:
            return
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_audit_dict()) + "\n")
        except Exception as err:
            print(f"[Warning] Failed to write LLM audit log: {err}")


# -----------------------------------------------------------------------------
# Mock Client for Offline Testing and CI
# -----------------------------------------------------------------------------

class MockDeepSeekClient:
    """Mock DeepSeek client that produces deterministic responses without Ollama."""

    def __init__(
        self,
        audit_file: Optional[Union[str, Path]] = None,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.audit_file = Path(audit_file) if audit_file else None
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
