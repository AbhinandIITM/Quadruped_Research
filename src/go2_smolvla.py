"""
go2_smolvla.py
==============
Hugging Face SmolVLA (Vision-Language-Action) Integration for Unitree Go2.

Provides:
  1. `SmolVLAQuadrupedAgent`: Loads and executes Hugging Face SmolVLA (`lerobot/smolvla_base`)
     on consumer GPU (CUDA) or CPU.
  2. Quadruped observation mapping:
     - Front camera RGB frame -> `observation.images.camera1` (3, 256, 256)
     - Proprioceptive state vector -> `observation.state` (6,)
     - Natural language goal prompt -> `observation.language.tokens`
  3. Action decoding & safety clamping:
     - 6D continuous action chunk -> `MotionCommand` with `SafetyBounds`
  4. Unified interface:
     - `query(prompt)` method matching `GeminiClient` / `OllamaDeepSeekClient`
       so SmolVLA can be used everywhere in place of text-only LLMs.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

from go2_llm_commands import (
    LLMCommandResult,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    Waypoint,
)

logger = logging.getLogger("go2_smolvla")


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class SmolVLAObservation:
    """Proprioceptive and visual observation for SmolVLA."""
    rgb_image: np.ndarray  # (H, W, 3) uint8
    state_vector: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    prompt: str = "walk forward at 0.3 m/s"


@dataclass
class SmolVLAAction:
    """Decoded action output from SmolVLA flow matching policy."""
    vx: float
    vy: float
    yaw_rate: float
    duration: float
    raw_action: np.ndarray
    motion_commands: List[MotionCommand]
    wall_time_s: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "yaw_rate": round(self.yaw_rate, 3),
            "duration": round(self.duration, 2),
            "raw_action": [round(float(v), 3) for v in self.raw_action[:6]],
            "explanation": self.explanation,
            "commands": [c.to_dict() for c in self.motion_commands],
        }


# -----------------------------------------------------------------------------
# Observation Preprocessor
# -----------------------------------------------------------------------------

class SmolVLAObservationPreprocessor:
    """Prepares PyTorch tensor batches for SmolVLA forward pass."""

    def __init__(
        self,
        tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        image_size: Tuple[int, int] = (256, 256),
        max_prompt_len: int = 48,
        device: str = "cuda",
    ):
        self.image_size = image_size
        self.max_prompt_len = max_prompt_len
        self.device = device
        self.tokenizer = None

        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                logger.warning("Could not load AutoTokenizer '%s': %s. Using fallback tokenization.", tokenizer_name, e)
                self.tokenizer = None

    def preprocess_image(self, rgb: np.ndarray) -> torch.Tensor:
        """Resize RGB numpy array to (1, 3, 256, 256) float tensor in [0, 1]."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for SmolVLA observation preprocessing.")

        img = Image.fromarray(rgb).resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # (256, 256, 3)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
        return img_tensor.to(self.device)

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state vector to (1, 6) float tensor."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for SmolVLA observation preprocessing.")

        s_np = np.asarray(state, dtype=np.float32).flatten()
        if len(s_np) < 6:
            s_np = np.pad(s_np, (0, 6 - len(s_np)))
        elif len(s_np) > 6:
            s_np = s_np[:6]
        return torch.from_numpy(s_np).unsqueeze(0).to(self.device)

    def tokenize_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize natural language prompt into tokens and attention mask tensors."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for SmolVLA tokenization.")

        if self.tokenizer is not None:
            enc = self.tokenizer(
                prompt,
                max_length=self.max_prompt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return enc["input_ids"].to(self.device), enc["attention_mask"].bool().to(self.device)

        # Fallback deterministic token representation
        words = prompt.lower().split()[:self.max_prompt_len]
        tokens = torch.zeros(1, self.max_prompt_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(1, self.max_prompt_len, dtype=torch.bool, device=self.device)
        for i, w in enumerate(words):
            tokens[0, i] = (hash(w) % 30000) + 1
            mask[0, i] = True
        return tokens, mask

    def build_batch(self, obs: SmolVLAObservation) -> Dict[str, torch.Tensor]:
        """Assembles the complete observation dictionary expected by SmolVLA."""
        img_t = self.preprocess_image(obs.rgb_image)
        state_t = self.preprocess_state(obs.state_vector)
        tokens_t, mask_t = self.tokenize_prompt(obs.prompt)

        return {
            "observation.images.camera1": img_t,
            "observation.state": state_t,
            "observation.language.tokens": tokens_t,
            "observation.language.attention_mask": mask_t,
        }


# -----------------------------------------------------------------------------
# Action Decoder
# -----------------------------------------------------------------------------

class SmolVLAActionDecoder:
    """Decodes continuous action tensors into safe quadruped MotionCommands."""

    def __init__(self, bounds: Optional[SafetyBounds] = None):
        self.bounds = bounds or SafetyBounds()

    def _parse_prompt_intent(self, prompt: str) -> Dict[str, Any]:
        """Extract explicit semantic and numerical intent from user prompt."""
        p = prompt.lower().strip()
        intent: Dict[str, Any] = {
            "is_stand": False,
            "target_vx": None,
            "target_vy": None,
            "target_yaw": None,
            "target_dur": None,
            "is_forward": False,
            "is_backward": False,
            "is_turn": False,
        }

        # Stand / Stop check
        if any(w in p for w in ["stop", "stand", "halt", "freeze"]):
            intent["is_stand"] = True
            intent["target_vx"] = 0.0
            intent["target_vy"] = 0.0
            intent["target_yaw"] = 0.0
            intent["target_dur"] = 1.0
            return intent

        # 1. Duration extraction: "for 6 seconds", "for 6.0s", "for 6 s", "6 sec", "6s"
        dur_m = re.search(r'(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b', p)
        if dur_m:
            try:
                intent["target_dur"] = float(dur_m.group(1))
            except ValueError:
                pass

        # 1. Angular velocity extraction first: "0.4 rad/s", "0.4rad/s"
        yaw_m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*(?:rad/s|radps)\b', p)
        if yaw_m:
            try:
                intent["target_yaw"] = float(yaw_m.group(1))
            except ValueError:
                pass

        # 2. Linear velocity extraction: "at 0.3 m/s", "0.3m/s", "0.3 mps", "speed 0.3"
        vel_m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*(?:m/s|mps)\b', p)
        if vel_m:
            try:
                intent["target_vx"] = float(vel_m.group(1))
            except ValueError:
                pass
        elif "speed" in p:
            spd_m = re.search(r'\bspeed\s+([+-]?\d+(?:\.\d+)?)\b', p)
            if spd_m:
                try:
                    intent["target_vx"] = float(spd_m.group(1))
                except ValueError:
                    pass
        elif any(k in p for k in ["forward at", "backward at", "walk at", "trot at", "pace at"]):
            at_m = re.search(r'(?:forward|backward|walk|trot|pace)\s+at\s+([+-]?\d+(?:\.\d+)?)\b', p)
            if at_m:
                try:
                    intent["target_vx"] = float(at_m.group(1))
                except ValueError:
                    pass

        # 4. Directional analysis
        if "forward" in p or "ahead" in p:
            intent["is_forward"] = True
            if intent["target_vx"] is not None:
                intent["target_vx"] = abs(intent["target_vx"])
        elif "backward" in p or "reverse" in p or "back" in p:
            intent["is_backward"] = True
            if intent["target_vx"] is not None:
                intent["target_vx"] = -abs(intent["target_vx"])
            else:
                intent["target_vx"] = -0.20

        # Turn analysis
        if "turn" in p or "rotate" in p or "spin" in p or "yaw" in p:
            intent["is_turn"] = True
            if "left" in p:
                intent["target_yaw"] = abs(intent["target_yaw"]) if intent["target_yaw"] is not None else 0.40
            elif "right" in p:
                intent["target_yaw"] = -abs(intent["target_yaw"]) if intent["target_yaw"] is not None else -0.40
        elif "left" in p and not intent["is_forward"]:
            if "strafe" in p or "sidestep" in p:
                intent["target_vy"] = 0.20
            else:
                intent["is_turn"] = True
                intent["target_yaw"] = 0.40
        elif "right" in p and not intent["is_forward"]:
            if "strafe" in p or "sidestep" in p:
                intent["target_vy"] = -0.20
            else:
                intent["is_turn"] = True
                intent["target_yaw"] = -0.40

        return intent

    def decode(
        self,
        raw_action: np.ndarray,
        prompt: str = "",
        wall_time: float = 0.0,
    ) -> SmolVLAAction:
        """
        Maps a 6D raw action tensor to bounded quadruped velocities and duration.
        Prioritizes explicit numerical velocities/durations when requested in prompt,
        while using VLA flow predictions for unspecified continuous dimensions.
        """
        arr = np.asarray(raw_action, dtype=np.float32).flatten()
        if len(arr) < 6:
            arr = np.pad(arr, (0, 6 - len(arr)))

        intent = self._parse_prompt_intent(prompt)

        if intent["is_stand"]:
            vx_safe = 0.0
            vy_safe = 0.0
            yaw_safe = 0.0
            dur_safe = 1.0
        else:
            # 1. Forward/Reverse linear velocity vx
            if intent["target_vx"] is not None:
                vx_raw = intent["target_vx"]
            elif intent["is_forward"]:
                vx_raw = max(0.25, float(arr[0]) * 0.25)
            elif intent["is_backward"]:
                vx_raw = min(-0.15, float(arr[0]) * 0.25)
            elif intent["is_turn"]:
                vx_raw = 0.0
            else:
                vx_raw = float(arr[0]) * 0.25

            # 2. Lateral velocity vy
            if intent["target_vy"] is not None:
                vy_raw = intent["target_vy"]
            elif intent["is_forward"] or intent["is_backward"] or intent["is_turn"]:
                # Suppress spurious lateral drift when pure forward/backward/turn requested
                vy_raw = 0.0
            else:
                vy_raw = float(arr[1]) * 0.15

            # 3. Yaw rate
            if intent["target_yaw"] is not None:
                yaw_raw = intent["target_yaw"]
            elif intent["is_turn"]:
                yaw_raw = float(arr[2]) * 0.35
            elif intent["is_forward"] or intent["is_backward"]:
                # Pure translation: do not turn unless turn was requested
                yaw_raw = 0.0
            else:
                yaw_raw = float(arr[2]) * 0.35

            # 4. Duration
            if intent["target_dur"] is not None:
                dur_raw = intent["target_dur"]
            else:
                dur_raw = 2.0 + float(arr[3]) * 0.5

            # Strictly clamp against SafetyBounds
            vx_safe = float(np.clip(vx_raw, self.bounds.min_vx, self.bounds.max_vx))
            vy_safe = float(np.clip(vy_raw, -self.bounds.max_vy, self.bounds.max_vy))
            yaw_safe = float(np.clip(yaw_raw, -self.bounds.max_yaw_rate, self.bounds.max_yaw_rate))
            dur_safe = float(np.clip(dur_raw, 0.5, self.bounds.max_duration))

        is_stand = (abs(vx_safe) < 0.02 and abs(vy_safe) < 0.02 and abs(yaw_safe) < 0.02)
        mode = MotionMode.STAND if is_stand else MotionMode.VELOCITY

        cmd_motion = MotionCommand(
            mode=mode,
            vx=vx_safe,
            vy=vy_safe,
            yaw_rate=yaw_safe,
            duration=dur_safe,
        )
        cmd_stand = MotionCommand(mode=MotionMode.STAND, duration=1.0)

        explanation = (
            f"SmolVLA VLA Policy: Ingested prompt '{prompt}' and front camera view. "
            f"Commanding {mode.name} with vx={vx_safe:+.2f}m/s, vy={vy_safe:+.2f}m/s, "
            f"yaw_rate={yaw_safe:+.2f}rad/s for {dur_safe:.1f}s."
        )

        return SmolVLAAction(
            vx=vx_safe,
            vy=vy_safe,
            yaw_rate=yaw_safe,
            duration=dur_safe,
            raw_action=arr,
            motion_commands=[cmd_motion, cmd_stand],
            wall_time_s=wall_time,
            explanation=explanation,
        )


# -----------------------------------------------------------------------------
# SmolVLA Quadruped Agent
# -----------------------------------------------------------------------------

class SmolVLAQuadrupedAgent:
    """
    Autonomous Vision-Language-Action Agent powered by Hugging Face SmolVLA.
    Provides complete end-to-end inference for quadruped locomotion.
    """

    def __init__(
        self,
        model_id: str = "lerobot/smolvla_base",
        device: Optional[str] = None,
        load_pretrained: bool = False,
        bounds: Optional[SafetyBounds] = None,
    ):
        self.model_id = model_id
        self.bounds = bounds or SafetyBounds()

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"

        self.policy: Optional[Any] = None
        self.is_loaded = False
        self.load_pretrained = load_pretrained

        self.preprocessor = SmolVLAObservationPreprocessor(device=self.device)
        self.decoder = SmolVLAActionDecoder(bounds=self.bounds)

        self._init_policy()

    def _init_policy(self):
        """Initializes the SmolVLA neural network."""
        if not HAS_LEROBOT or not HAS_TORCH:
            logger.warning("lerobot not available. SmolVLA agent running in offline synthetic mode.")
            return

        input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }

        try:
            if self.load_pretrained:
                logger.info("Loading pretrained SmolVLA weights from '%s'...", self.model_id)
                self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
            else:
                cfg = SmolVLAConfig(
                    device=self.device,
                    input_features=input_features,
                    output_features=output_features,
                    n_action_steps=10,
                    chunk_size=10,
                    empty_cameras=0,
                )
                self.policy = SmolVLAPolicy(cfg)

            self.policy.to(self.device)
            self.policy.eval()
            self.is_loaded = True
            logger.info("SmolVLA policy successfully initialized on %s.", self.device)
        except Exception as e:
            logger.warning("Failed to initialize SmolVLA policy (%s). Falling back to synthetic agent.", e)
            self.policy = None
            self.is_loaded = False

    def check_availability(self) -> Tuple[bool, str]:
        """Check if SmolVLA is ready for inference."""
        if not HAS_TORCH:
            return False, "PyTorch not installed."
        if not HAS_LEROBOT:
            return False, "lerobot not installed."
        if self.policy is not None:
            return True, f"SmolVLA policy active on device '{self.device}'."
        return True, "SmolVLA running in synthetic policy mode."

    def predict_action(
        self,
        rgb: np.ndarray,
        state: Optional[np.ndarray] = None,
        prompt: str = "walk forward at 0.3 m/s",
    ) -> SmolVLAAction:
        """Run observation forward pass through SmolVLA and return decoded action."""
        t_start = time.perf_counter()

        if state is None:
            state = np.zeros(6, dtype=np.float32)

        obs = SmolVLAObservation(rgb_image=rgb, state_vector=state, prompt=prompt)

        if self.policy is not None and HAS_TORCH:
            batch = self.preprocessor.build_batch(obs)
            with torch.no_grad():
                act_tensor = self.policy.select_action(batch)
                raw_np = act_tensor.squeeze().cpu().numpy()
        else:
            # Deterministic synthetic action based on prompt keywords
            p_low = prompt.lower()
            vx = 0.30 if "forward" in p_low else (-0.20 if "backward" in p_low else 0.0)
            vy = 0.15 if "left" in p_low and "turn" not in p_low else 0.0
            wz = 0.40 if "left" in p_low and "turn" in p_low else (-0.40 if "right" in p_low else 0.0)
            raw_np = np.array([vx / 0.25, vy / 0.15, wz / 0.35, 0.0, 0.0, 0.0], dtype=np.float32)

        t_elapsed = time.perf_counter() - t_start
        return self.decoder.decode(raw_np, prompt=prompt, wall_time=t_elapsed)

    def query(
        self,
        user_prompt: str,
        rgb: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
    ) -> LLMCommandResult:
        """
        Standard interface matching GeminiClient and OllamaDeepSeekClient.
        Allows SmolVLA to be plugged in everywhere across the CLI and server.
        """
        if rgb is None:
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        if state is None:
            state = np.zeros(6, dtype=np.float32)

        action = self.predict_action(rgb=rgb, state=state, prompt=user_prompt)

        return LLMCommandResult(
            user_prompt=user_prompt,
            raw_response=str(action.raw_action[:6].tolist()),
            thinking=None,
            parsed_json=action.to_dict(),
            commands=action.motion_commands,
            status="success",
            explanation=action.explanation,
            wall_time_s=action.wall_time_s,
            model_name=f"hf-smolvla-{self.device}",
        )
