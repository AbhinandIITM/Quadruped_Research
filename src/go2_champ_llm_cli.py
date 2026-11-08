"""Natural Language Simulation CLI for Go2 CHAMP Locomotion.

Connects human natural language prompts to the Go2 CHAMP locomotion controller
in MuJoCo simulation via local Ollama (DeepSeek-R1), Google AI Studio / Gemini API (Gemini/Gemma),
live model server, or OmniRoute with real-time live telemetry tracking plots.

SAFETY GUARANTEES (SIMULATION-ONLY):
- Operates strictly within local MuJoCo simulation.
- Never interacts with real robot hardware, networks, or shell commands.
- All LLM responses are parsed into structured MotionCommands and validated
  against SafetyBounds.
- Out-of-bounds, malformed, or ambiguous commands fail closed to a safe stand.
- Every run logs full telemetry, metrics, plots, and an audit trail to logs/.

Usage Examples:
    # Single prompt with local Ollama DeepSeek-R1 (auto-detected default):
    python go2_champ_llm_cli.py --prompt "Walk forward at 0.35 m/s for 3 seconds"

    # Single prompt with Google Gemini / Gemma API:
    python go2_champ_llm_cli.py --gemini --model gemini-2.5-flash --prompt "Turn left for 2 seconds"
    python go2_champ_llm_cli.py --gemini --gemini-api-key YOUR_KEY --prompt "Walk forward at 0.3 m/s"

    # Headless run without GUI (for automated CI/CD):
    python go2_champ_llm_cli.py --no-gui --prompt "Walk forward at 0.35 m/s for 3 seconds"

    # Interactive REPL:
    python go2_champ_llm_cli.py --interactive
    python go2_champ_llm_cli.py --gemini --interactive

    # Waypoint navigation:
    python go2_champ_llm_cli.py --prompt "Navigate to waypoint at 1.0, 1.0"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime, timezone
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if "--no-gui" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from go2_champ_control import (
    FOOT_BODY_NAMES,
    HIP_POSITIONS,
    IS_LEFT,
    JOINT_NAMES,
    LEG_NAMES,
    POSITION_KD,
    POSITION_KP,
    ChampConfig,
    Go2ChampController,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    Waypoint,
    configure_model,
    initialize_simulation_standing,
    quaternion_to_euler_xyz,
)
from go2_llm_commands import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_SERVER_URL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OMNIROUTE_MODEL,
    DEFAULT_OMNIROUTE_URL,
    GeminiClient,
    LLMCommandResult,
    LiveModelServiceClient,
    OllamaDeepSeekClient,
    OmniRouteClient,
)
from go2_vlm_perception import (
    BoundingBox2D,
    VLMGoalGrounder,
    VLMGroundingResult,
    create_vlm_scene_xml,
)


_src_dir = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = (
    os.path.dirname(_src_dir)
    if os.path.exists(os.path.join(os.path.dirname(_src_dir), "unitree_go2"))
    else _src_dir
)
ROOT_DIR = _REPO_ROOT

# Model and run-log locations default to the in-repo paths, but an operator can
# redirect either from the environment without editing source (same convention
# as the endpoint URLs in go2_llm_commands.py). Defaults are unchanged.
# NOTE: the model path literal is also duplicated in evaluate_champ_trajectories.py,
# go2_champ_trot.py, go2_mujoco_trot.py and go2_mujoco_trot_traj.py -- those files
# are owned elsewhere and are deliberately left alone here.
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "unitree_go2", "go2_description.xml")
DEFAULT_LOG_DIR = os.path.join(ROOT_DIR, "logs", "llm_runs")
MODEL_PATH = os.environ.get("GO2_MODEL_PATH", DEFAULT_MODEL_PATH)
LOG_DIR = os.environ.get("GO2_LOG_DIR", DEFAULT_LOG_DIR)

logger = logging.getLogger("go2_champ_llm_cli")


# -----------------------------------------------------------------------------
# Real-time loop hygiene helpers
#
# The MuJoCo model timestep is 2 ms (500 Hz), so everything that runs from
# inside a control tick must fit a 2 ms budget. The helpers below keep the
# expensive, periodic and blocking work -- viewer drawing, plotting, terminal
# I/O, vision -- decimated, bounded, or off the control thread entirely.
# -----------------------------------------------------------------------------

# Periodic control-loop work, expressed in wall/sim seconds and converted to
# step counts at run time (a step is one `model.opt.timestep`).
LIVE_PLOT_PERIOD_S = 0.2        # Live plot redraw rate (5 Hz; was every tick).
VIEWER_SYNC_PERIOD_S = 0.02     # GUI viewer draw rate (50 Hz; was every tick).
TELEMETRY_PRINT_PERIOD_S = 0.5  # Terminal telemetry line cadence (unchanged).
VISION_PERIOD_S = 0.05          # Nominal vision pipeline cadence (unchanged, 20 Hz).
VISION_MAX_PERIOD_S = 1.0       # Slowest vision cadence after budget back-off.

# A vision run that costs more than this many control steps' worth of time is
# treated as over budget and widens the vision period.
VISION_BUDGET_STEPS = 2.0


def steps_for_period(period_s: float, dt: float) -> int:
    """Number of control steps spanning roughly ``period_s`` seconds.

    Used to rate-limit periodic work in step counts. Always at least 1, so a
    decimated task can never accidentally run on every tick or never at all.
    """
    if dt <= 0.0:
        return 1
    return max(1, int(round(period_s / dt)))


def pacing_sleep_seconds(next_step_wall_time: float, now: float) -> float:
    """Seconds to sleep in order to hold an absolute wall-clock deadline.

    Never negative: when the loop is already late the caller simply does not
    sleep, which is what keeps pacing self-correcting instead of compounding a
    fixed per-tick delay into unbounded drift (see ``go2_champ_trot.py``).
    """
    return max(0.0, next_step_wall_time - now)


class BackgroundPrinter:
    """Formats nothing, blocks never: writes telemetry lines from a daemon thread.

    ``submit`` only enqueues an already-formatted string, so the console write
    itself (which can stall for milliseconds on a busy Windows terminal) never
    happens on the control thread. If the writer falls behind, the stalest queued
    line is discarded rather than the control loop being made to wait -- a lost
    telemetry line is always cheaper than a missed deadline.
    """

    def __init__(self, maxsize: int = 16):
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._drain, name="go2-telemetry-print", daemon=True
        )
        self._thread.start()

    def submit(self, line: str) -> None:
        """Enqueue a line for the writer thread. Never blocks."""
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop the stalest line, keep the newest
                self._queue.put_nowait(line)
            except (queue.Empty, queue.Full):
                pass

    def _drain(self) -> None:
        while True:
            line = self._queue.get()
            if line is None:
                return
            try:
                print(line, flush=True)
            except Exception:  # never let a console error kill the writer thread
                pass

    def close(self, timeout: float = 2.0) -> None:
        """Flush pending lines and stop the writer thread."""
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


class VisionScheduler:
    """Gate the vision pipeline on a measured time budget, not a raw modulo.

    A vision run is due only when at least ``period_steps`` control ticks have
    passed *and* the loop still has slack before the tick's absolute wall-clock
    deadline. Measured runs that cost more than ``budget_s`` back the period off
    (doubling, capped at ``max_period_steps``); runs that fit comfortably let it
    recover towards ``period_steps``. A pipeline that turns out to be expensive
    therefore throttles its own update rate instead of stealing the control tick.
    """

    def __init__(
        self,
        dt: float,
        period_s: float = VISION_PERIOD_S,
        budget_s: Optional[float] = None,
        max_period_s: float = VISION_MAX_PERIOD_S,
        backoff: float = 2.0,
    ):
        self.min_period_steps = steps_for_period(period_s, dt)
        self.max_period_steps = max(
            self.min_period_steps, steps_for_period(max_period_s, dt)
        )
        self.period_steps = self.min_period_steps
        self.budget_s = float(dt) * VISION_BUDGET_STEPS if budget_s is None else float(budget_s)
        self.backoff = float(backoff)

        self.last_run_step: Optional[int] = None
        self.last_duration_s = 0.0
        self.overruns = 0
        self.skips = 0
        self.failures = 0

    def due(
        self,
        step_idx: int,
        deadline: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """True when a vision run may start on this tick without blowing the deadline."""
        if self.last_run_step is not None and (step_idx - self.last_run_step) < self.period_steps:
            return False
        if deadline is not None and now is not None and now > deadline:
            # No slack left in this tick. Skipping costs one stale clearance
            # estimate; overrunning costs the whole control deadline.
            self.skips += 1
            return False
        self.last_run_step = step_idx
        return True

    def record(self, duration_s: float) -> None:
        """Record the measured cost of a completed run and adapt the period."""
        self.last_duration_s = float(duration_s)
        if duration_s > self.budget_s:
            self.overruns += 1
            self.period_steps = min(
                self.max_period_steps,
                max(self.period_steps + 1, int(self.period_steps * self.backoff)),
            )
            return
        if self.period_steps > self.min_period_steps:
            self.period_steps = max(self.min_period_steps, self.period_steps // 2)

    def record_failure(self) -> int:
        """Count a failed vision run; returns the running failure count."""
        self.failures += 1
        return self.failures


class SimulationLivePlotter:
    """Manages real-time 2x2 tracking matplotlib window during simulation."""

    def __init__(self, prompt: str = ""):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(11, 7.5))
        title = "Go2 CHAMP Real-Time Telemetry"
        if prompt:
            short_prompt = prompt if len(prompt) <= 45 else prompt[:42] + "..."
            title += f": \"{short_prompt}\""
        self.fig.suptitle(title, fontsize=11, fontweight="bold")

        # 1. Top-Left: 2D Ground Trajectory
        self.ax_traj = self.axes[0, 0]
        (self.line_traj,) = self.ax_traj.plot([], [], color="#2563eb", lw=2, label="Actual Path")
        (self.point_start,) = self.ax_traj.plot([], [], "go", markersize=7, label="Start")
        (self.point_curr,) = self.ax_traj.plot([], [], "ro", markersize=7, label="Current")
        self.ax_traj.set_xlabel("World X (m)")
        self.ax_traj.set_ylabel("World Y (m)")
        self.ax_traj.set_title("2D Ground Trajectory (X vs Y)")
        self.ax_traj.grid(True, alpha=0.3)
        self.ax_traj.legend(loc="upper left", fontsize=8)
        self.ax_traj.set_xlim(-0.3, 1.5)
        self.ax_traj.set_ylim(-0.6, 0.6)

        # 2. Top-Right: Forward Velocity vx Tracking
        self.ax_vx = self.axes[0, 1]
        (self.line_cmd_vx,) = self.ax_vx.plot([], [], color="#dc2626", linestyle="--", lw=1.5, label="Command vx")
        (self.line_act_vx,) = self.ax_vx.plot([], [], color="#2563eb", lw=1.8, label="Observed vx")
        self.ax_vx.set_xlabel("Time (s)")
        self.ax_vx.set_ylabel("Velocity (m/s)")
        self.ax_vx.set_title("Forward Velocity vx Tracking")
        self.ax_vx.grid(True, alpha=0.3)
        self.ax_vx.legend(loc="upper left", fontsize=8)
        self.ax_vx.set_ylim(-0.1, 0.6)

        # 3. Bottom-Left: Lateral Velocity vy Tracking
        self.ax_vy = self.axes[1, 0]
        (self.line_cmd_vy,) = self.ax_vy.plot([], [], color="#dc2626", linestyle="--", lw=1.5, label="Command vy")
        (self.line_act_vy,) = self.ax_vy.plot([], [], color="#059669", lw=1.8, label="Observed vy")
        self.ax_vy.set_xlabel("Time (s)")
        self.ax_vy.set_ylabel("Velocity (m/s)")
        self.ax_vy.set_title("Lateral Velocity vy Tracking")
        self.ax_vy.grid(True, alpha=0.3)
        self.ax_vy.legend(loc="upper left", fontsize=8)
        self.ax_vy.set_ylim(-0.25, 0.25)

        # 4. Bottom-Right: Yaw Rate wz Tracking
        self.ax_wz = self.axes[1, 1]
        (self.line_cmd_wz,) = self.ax_wz.plot([], [], color="#dc2626", linestyle="--", lw=1.5, label="Command Yaw Rate")
        (self.line_act_wz,) = self.ax_wz.plot([], [], color="#7c3aed", lw=1.8, label="Observed Yaw Rate")
        self.ax_wz.set_xlabel("Time (s)")
        self.ax_wz.set_ylabel("Yaw Rate (rad/s)")
        self.ax_wz.set_title("Yaw Rate wz Tracking")
        self.ax_wz.grid(True, alpha=0.3)
        self.ax_wz.legend(loc="upper left", fontsize=8)
        self.ax_wz.set_ylim(-1.0, 1.0)

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(
        self,
        times: List[float],
        xs: List[float],
        ys: List[float],
        cmd_vxs: List[float],
        act_vxs: List[float],
        cmd_vys: List[float],
        act_vys: List[float],
        cmd_wzs: List[float],
        act_wzs: List[float],
    ):
        if not times:
            return

        # 1. 2D Path
        self.line_traj.set_data(xs, ys)
        self.point_start.set_data([xs[0]], [ys[0]])
        self.point_curr.set_data([xs[-1]], [ys[-1]])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        margin = 0.3
        self.ax_traj.set_xlim(min(-0.2, x_min - margin), max(1.0, x_max + margin))
        self.ax_traj.set_ylim(min(-0.4, y_min - margin), max(0.4, y_max + margin))

        # 2. vx
        t_max = max(1.0, times[-1] * 1.05)
        self.line_cmd_vx.set_data(times, cmd_vxs)
        self.line_act_vx.set_data(times, act_vxs)
        self.ax_vx.set_xlim(0, t_max)
        all_vx = cmd_vxs + act_vxs
        if all_vx:
            self.ax_vx.set_ylim(min(-0.1, min(all_vx) - 0.05), max(0.5, max(all_vx) + 0.05))

        # 3. vy
        self.line_cmd_vy.set_data(times, cmd_vys)
        self.line_act_vy.set_data(times, act_vys)
        self.ax_vy.set_xlim(0, t_max)
        all_vy = cmd_vys + act_vys
        if all_vy:
            self.ax_vy.set_ylim(min(-0.15, min(all_vy) - 0.05), max(0.15, max(all_vy) + 0.05))

        # 4. wz
        self.line_cmd_wz.set_data(times, cmd_wzs)
        self.line_act_wz.set_data(times, act_wzs)
        self.ax_wz.set_xlim(0, t_max)
        all_wz = cmd_wzs + act_wzs
        if all_wz:
            self.ax_wz.set_ylim(min(-0.5, min(all_wz) - 0.1), max(0.5, max(all_wz) + 0.1))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


def select_live_llm_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    server_url: str = DEFAULT_MODEL_SERVER_URL,
    omniroute_url: str = DEFAULT_OMNIROUTE_URL,
    omniroute_api_key: str = "",
    gemini_api_key: Optional[str] = None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    auth_token: Optional[str] = None,
) -> Any:
    """Connect strictly to a live LLM backend. Offline mock fallbacks are completely forbidden.

    ``auth_token`` is the shared secret for the Live Model Server. When it is
    ``None`` the client keeps its own MODEL_SERVER_TOKEN environment fallback, so
    omitting the flag preserves the previous behaviour exactly.
    """
    if backend == "mock":
        raise ValueError(
            "Mock client is disabled. A live LLM backend is required.\n"
            "Please use local Ollama, Google Gemini, Model Server, or OmniRoute."
        )

    # 1. Explicit backend specified
    if backend == "server":
        client = LiveModelServiceClient(server_url=server_url, auth_token=auth_token)
        available, msg = client.check_availability()
        if not available:
            raise ConnectionError(f"Live Model Server unavailable at {server_url}: {msg}")
        print(f"✓ Connected to Live Model Server: {msg}")
        return client

    if backend == "ollama":
        model_name = model or DEFAULT_MODEL_NAME
        client = OllamaDeepSeekClient(host=ollama_host, model=model_name)
        available, msg = client.check_availability()
        if not available:
            raise ConnectionError(f"Local Ollama unavailable at {ollama_host}: {msg}")
        print(f"✓ Connected to local Ollama ({model_name}) at {ollama_host}.")
        return client

    if backend == "gemini":
        model_name = model or DEFAULT_GEMINI_MODEL
        client = GeminiClient(api_key=gemini_api_key, model=model_name)
        available, msg = client.check_availability()
        if not available:
            raise ConnectionError(f"Google Gemini API unavailable: {msg}")
        print(f"✓ Connected to Google Gemini API ({model_name}).")
        return client

    if backend == "omniroute":
        model_name = model or DEFAULT_OMNIROUTE_MODEL
        client = OmniRouteClient(base_url=omniroute_url, model=model_name, api_key=omniroute_api_key)
        available, msg = client.check_availability()
        if not available:
            raise ConnectionError(f"OmniRoute endpoint unavailable at {omniroute_url}: {msg}")
        print(f"✓ Connected to OmniRoute ({model_name}) at {omniroute_url}.")
        return client

    if backend == "smolvla":
        from go2_smolvla import SmolVLAQuadrupedAgent
        model_name = model or "lerobot/smolvla_base"
        client = SmolVLAQuadrupedAgent(model_id=model_name, load_pretrained=False)
        available, msg = client.check_availability()
        if not available:
            raise ConnectionError(f"SmolVLA unavailable: {msg}")
        print(f"✓ Connected to Hugging Face SmolVLA: {msg}")
        return client

    # 2. Auto-detection order across live backends
    # Priority 1: Live Model Server
    srv_client = LiveModelServiceClient(server_url=server_url, auth_token=auth_token)
    available, msg = srv_client.check_availability()
    if available:
        print(f"✓ Auto-detected Live Model Server: {msg}")
        return srv_client

    # Priority 2: Local Ollama
    model_name = model or DEFAULT_MODEL_NAME
    ollama_client = OllamaDeepSeekClient(host=ollama_host, model=model_name)
    available, msg = ollama_client.check_availability()
    if available:
        print(f"✓ Auto-detected local Ollama ({model_name}) at {ollama_host}.")
        return ollama_client

    # Priority 3: Google Gemini
    model_name = model or DEFAULT_GEMINI_MODEL
    gemini_client = GeminiClient(api_key=gemini_api_key, model=model_name)
    available, msg = gemini_client.check_availability()
    if available:
        print(f"✓ Auto-detected Google Gemini API ({model_name}).")
        return gemini_client

    # Priority 4: OmniRoute
    model_name = model or DEFAULT_OMNIROUTE_MODEL
    omni_client = OmniRouteClient(base_url=omniroute_url, model=model_name, api_key=omniroute_api_key)
    available, msg = omni_client.check_availability()
    if available:
        print(f"✓ Auto-detected OmniRoute ({model_name}) at {omniroute_url}.")
        return omni_client

    # 3. Fail fast if no live LLM is accessible
    raise RuntimeError(
        "CRITICAL ERROR: No live LLM backend is available!\n"
        "Offline mock client fallback has been strictly disabled.\n\n"
        "Please do one of the following to connect to a live LLM:\n"
        "  1. Start local Ollama: run 'ollama serve' and ensure deepseek-r1:8b is installed.\n"
        "  2. Configure GEMINI_API_KEY in your .env file or pass --gemini-api-key.\n"
        "  3. Launch local model server: run 'python start_model_server.py'.\n"
        "  4. Launch OmniRoute: run 'python start_omniroute.py'."
    )


class Go2ChampSimulationRunner:
    """Executes a validated list of MotionCommands in MuJoCo simulation."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        xml_string: Optional[str] = None,
        kp: float = POSITION_KP,
        kd: float = POSITION_KD,
        config: Optional[ChampConfig] = None,
        bounds: Optional[SafetyBounds] = None,
        decimation: int = 4,
    ):
        if xml_string:
            self.model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MuJoCo model not found at {model_path}")
            self.model = mujoco.MjModel.from_xml_path(str(model_path))

        self.data = mujoco.MjData(self.model)
        configure_model(self.model, kp=kp, kd=kd)

        self.config = config or ChampConfig()
        self.bounds = bounds or SafetyBounds()
        self.controller = Go2ChampController(config=self.config, bounds=self.bounds)
        self.decimation = decimation

        # Cache IDs
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FOOT_BODY_NAMES
        ]
        self.qpos_addrs = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
        self.dof_addrs = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]

        # Vision perception pipeline
        self.vision_pipeline = None
        try:
            from go2_vision_perception import DepthPerceptionPipeline
            self.vision_pipeline = DepthPerceptionPipeline(self.model)
        except Exception as e:
            # Fail-safe (run without reactive clearance), but never silently.
            logger.warning("Vision perception unavailable; running without reactive clearance: %s", e)
            self.vision_pipeline = None

        self.reset_to_standing()

    def reset_to_standing(self):
        """Reset the MuJoCo simulation to a stable standing configuration."""
        initialize_simulation_standing(self.model, self.data, body_height=self.config.body_height)
        self.controller.reset()

    def foot_contacts(self) -> List[int]:
        """Detect ground contact for each foot."""
        contacts = [0, 0, 0, 0]
        foot_lookup = {body_id: idx for idx, body_id in enumerate(self.foot_body_ids)}
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            for b in (b1, b2):
                if b in foot_lookup:
                    contacts[foot_lookup[b]] = 1
        return contacts

    def hold_current_joints(self) -> None:
        """Command a neutral hold at the measured joint positions.

        Called on every exit path from ``execute_commands`` so the actuators are
        never left latched to an arbitrary stale ``data.ctrl``. This is a minimal,
        deliberately dumb safe state only -- the controller owns the real e-stop
        semantics, which are not redefined here.
        """
        try:
            measured = np.array([self.data.qpos[a] for a in self.qpos_addrs])
            self.data.ctrl[self.actuator_ids] = measured
            mujoco.mj_step(self.model, self.data)
            logger.info("Safe hold applied on exit: actuators commanded to measured joint positions.")
        except Exception as e:
            logger.warning("Failed to command safe hold on exit: %s", e)

    @staticmethod
    def _extend_plot_buffers(
        buffers: Dict[str, List[float]],
        history: List[Dict[str, Any]],
        cursor: int,
    ) -> int:
        """Append only the telemetry samples recorded since ``cursor``.

        Keeps the live plot fed from fixed buffers instead of rebuilding Python
        lists out of the whole growing history on every redraw.
        """
        for rec in history[cursor:]:
            buffers["t"].append(rec["time_s"])
            buffers["x"].append(rec["base_x_m"])
            buffers["y"].append(rec["base_y_m"])
            buffers["cmd_vx"].append(rec["cmd_vx"])
            buffers["obs_vx"].append(rec["base_vx_body_mps"])
            buffers["cmd_vy"].append(rec["cmd_vy"])
            buffers["obs_vy"].append(rec["base_vy_body_mps"])
            buffers["cmd_wz"].append(rec["cmd_yaw"])
            buffers["obs_wz"].append(rec["base_wz_radps"])
        return len(history)

    def execute_commands(
        self,
        commands: List[MotionCommand],
        gui: bool = False,
        max_duration: float = 30.0,
        live_plot: bool = True,
        prompt: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run the simulation stepping through the provided MotionCommands."""
        self.reset_to_standing()
        dt = self.model.opt.timestep
        history: List[Dict[str, Any]] = []

        # Build schedule: [(start_time, end_time, command), ...]
        schedule: List[Tuple[float, float, MotionCommand]] = []
        t_accum = 0.0
        for cmd in commands:
            cmd_dur = min(cmd.duration, max_duration - t_accum)
            if cmd_dur <= 0:
                break
            schedule.append((t_accum, t_accum + cmd_dur, cmd))
            t_accum += cmd_dur

        total_sim_time = min(t_accum, max_duration)
        if total_sim_time <= 0:
            total_sim_time = 2.0
            schedule = [(0.0, 2.0, MotionCommand(mode=MotionMode.STAND, duration=2.0))]

        sim_time = 0.0
        step_idx = 0

        viewer_ctx = None
        if gui:
            try:
                viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                print(f"[Warning] Failed to launch GUI viewer: {e}. Falling back to headless.")
                viewer_ctx = None

        live_plotter = None
        if gui and live_plot:
            try:
                live_plotter = SimulationLivePlotter(prompt=prompt)
            except Exception as e:
                print(f"[Warning] Failed to launch live plotter: {e}.")
                live_plotter = None

        # Decimation periods for periodic work, in control steps.
        plot_period_steps = steps_for_period(LIVE_PLOT_PERIOD_S, dt)
        viewer_period_steps = steps_for_period(VIEWER_SYNC_PERIOD_S, dt)
        print_period_steps = steps_for_period(TELEMETRY_PRINT_PERIOD_S, dt)

        # Off-control-path helpers: blocking console I/O runs on a writer thread,
        # and vision is admitted by a measured budget rather than a raw modulo.
        telemetry_printer = BackgroundPrinter()
        vision_scheduler = (
            VisionScheduler(dt=dt)
            if self.vision_pipeline is not None and self.vision_pipeline.is_available
            else None
        )
        last_clearance = None

        # Only GUI runs are paced against wall time; headless runs stay as fast as
        # the CPU allows, exactly as before.
        pace_viewer = viewer_ctx is not None
        wall_clock_reference = time.perf_counter()

        # Live-plot buffers fed incrementally from newly recorded samples.
        plot_buffers: Dict[str, List[float]] = {
            k: [] for k in ("t", "x", "y", "cmd_vx", "obs_vx", "cmd_vy", "obs_vy", "cmd_wz", "obs_wz")
        }
        plot_cursor = 0

        try:
            while sim_time < total_sim_time:
                # Absolute wall-clock deadline for the tick in flight (GUI-paced only).
                tick_deadline = (
                    wall_clock_reference + (step_idx + 1) * dt if pace_viewer else None
                )

                # 1. Determine active command
                active_cmd = MotionCommand(mode=MotionMode.STAND, duration=1.0)
                for t_start, t_end, cmd in schedule:
                    if t_start <= sim_time < t_end:
                        active_cmd = cmd
                        break

                # 2. Extract robot state
                base_pos = self.data.qpos[:3].copy()
                base_quat = self.data.qpos[3:7].copy()
                base_euler = quaternion_to_euler_xyz(base_quat)
                base_rot = self.data.xmat[self.base_body_id].reshape(3, 3).copy()

                linvel_world = self.data.qvel[:3].copy()
                linvel_body = base_rot.T @ linvel_world
                angvel_body = self.data.qvel[3:6].copy()

                joint_pos = np.array([self.data.qpos[a] for a in self.qpos_addrs])
                joint_vel = np.array([self.data.qvel[a] for a in self.dof_addrs])

                # 3. Vision perception clearance update, admitted by a measured
                #    time budget rather than a raw modulo. A slow or failing run
                #    backs its own period off and can never gate the tick.
                sector_clearance = None
                if vision_scheduler is not None:
                    if vision_scheduler.due(step_idx, deadline=tick_deadline, now=time.perf_counter()):
                        t_vision = time.perf_counter()
                        try:
                            depth_map = self.vision_pipeline.capture_depth(self.data)
                            last_clearance = self.vision_pipeline.analyze_clearance(depth_map)
                        except Exception as exc:
                            # Fail-safe: keep walking with no clearance information, but
                            # never silently -- count it and warn at a bounded rate. A
                            # vision failure is never allowed to command motion itself.
                            last_clearance = None
                            failures = vision_scheduler.record_failure()
                            if failures <= 3 or failures % 100 == 0:
                                logger.warning(
                                    "Vision perception failed (%d failure(s)); continuing without "
                                    "clearance information: %s",
                                    failures,
                                    exc,
                                )
                        vision_scheduler.record(time.perf_counter() - t_vision)
                    sector_clearance = last_clearance

                # 4. Controller step
                target_q, desired_feet, st_phases, sw_phases, info = self.controller.step(
                    command=active_cmd,
                    step_time=sim_time,
                    dt=dt,
                    base_pos=base_pos,
                    base_euler=base_euler,
                    base_linvel_body=linvel_body,
                    base_angvel_body=angvel_body,
                    joint_positions=joint_pos,
                    sector_clearance=sector_clearance,
                )

                # 5. Actuator command
                self.data.ctrl[self.actuator_ids] = target_q

                # 6. Physics step
                mujoco.mj_step(self.model, self.data)

                # 7. Record telemetry
                if step_idx % self.decimation == 0:
                    contacts = self.foot_contacts()
                    torques = self.data.actuator_force[self.actuator_ids].copy()
                    power = float(np.sum(np.abs(torques * joint_vel)))

                    rec = {
                        "time_s": sim_time,
                        "cmd_vx": info.get("cmd_vx", 0.0),
                        "cmd_vy": info.get("cmd_vy", 0.0),
                        "cmd_yaw": info.get("cmd_yaw", 0.0),
                        "mode": info.get("mode", "stand"),
                        "base_x_m": float(base_pos[0]),
                        "base_y_m": float(base_pos[1]),
                        "base_z_m": float(base_pos[2]),
                        "base_vx_body_mps": float(linvel_body[0]),
                        "base_vy_body_mps": float(linvel_body[1]),
                        "base_vz_body_mps": float(linvel_body[2]),
                        "base_roll_rad": float(base_euler[0]),
                        "base_pitch_rad": float(base_euler[1]),
                        "base_yaw_rad": float(base_euler[2]),
                        "base_wx_radps": float(angvel_body[0]),
                        "base_wy_radps": float(angvel_body[1]),
                        "base_wz_radps": float(angvel_body[2]),
                        "power_w": power,
                        "fault": info.get("fault"),
                    }
                    for leg_idx, leg_name in enumerate(LEG_NAMES):
                        rec[f"{leg_name}_contact"] = int(contacts[leg_idx])
                        rec[f"{leg_name}_st_phase"] = float(st_phases[leg_idx])
                        rec[f"{leg_name}_sw_phase"] = float(sw_phases[leg_idx])
                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        rec[f"{j_name}_target_rad"] = float(target_q[j_idx])
                        rec[f"{j_name}_actual_rad"] = float(joint_pos[j_idx])
                        rec[f"{j_name}_torque_nm"] = float(torques[j_idx])

                    history.append(rec)

                # 8. Periodic terminal telemetry log: formatted here, written by the
                #    background writer thread, so console I/O never blocks a tick.
                if step_idx % print_period_steps == 0:
                    telemetry_printer.submit(
                        f"  [sim {sim_time:4.1f}s] "
                        f"cmd_vx: {info.get('cmd_vx', 0.0):+.2f} | obs_vx: {linvel_body[0]:+.2f} m/s | "
                        f"cmd_vy: {info.get('cmd_vy', 0.0):+.2f} | obs_vy: {linvel_body[1]:+.2f} m/s | "
                        f"obs_wz: {angvel_body[2]:+.2f} rad/s"
                    )

                # 9. Live tracking plot update, hard-decimated (5 Hz) and fed from
                #    incremental buffers, so the plot cannot gate the simulation. A
                #    plotting failure disables the window rather than repeatedly
                #    costing the control loop.
                if live_plotter is not None and step_idx % plot_period_steps == 0:
                    plot_cursor = self._extend_plot_buffers(plot_buffers, history, plot_cursor)
                    if plot_buffers["t"]:
                        try:
                            live_plotter.update(
                                plot_buffers["t"], plot_buffers["x"], plot_buffers["y"],
                                plot_buffers["cmd_vx"], plot_buffers["obs_vx"],
                                plot_buffers["cmd_vy"], plot_buffers["obs_vy"],
                                plot_buffers["cmd_wz"], plot_buffers["obs_wz"],
                            )
                        except Exception as e:
                            logger.warning("Live plot update failed; disabling live plotting: %s", e)
                            live_plotter.close()
                            live_plotter = None

                step_idx += 1
                sim_time += dt

                # 10. GUI pacing, outside the control work: the viewer draw is
                #     decimated and the sleep targets the tick's absolute wall-clock
                #     deadline, so a slow tick is never compounded by a fixed sleep
                #     and the sim clock cannot drift away from wall time.
                if pace_viewer and viewer_ctx.is_running():
                    if step_idx % viewer_period_steps == 0:
                        viewer_ctx.sync()
                    time.sleep(
                        pacing_sleep_seconds(
                            wall_clock_reference + step_idx * dt, time.perf_counter()
                        )
                    )

            # Final live plot flush from the same incremental buffers
            if live_plotter is not None:
                plot_cursor = self._extend_plot_buffers(plot_buffers, history, plot_cursor)
                if plot_buffers["t"]:
                    try:
                        live_plotter.update(
                            plot_buffers["t"], plot_buffers["x"], plot_buffers["y"],
                            plot_buffers["cmd_vx"], plot_buffers["obs_vx"],
                            plot_buffers["cmd_vy"], plot_buffers["obs_vy"],
                            plot_buffers["cmd_wz"], plot_buffers["obs_wz"],
                        )
                    except Exception as e:
                        logger.warning("Final live plot update failed: %s", e)

        except KeyboardInterrupt:
            print(f"\n[Interrupted] Simulation stopped by user (Ctrl+C) at t={sim_time:.2f}s.")
        finally:
            # Defined safe state on every exit path (normal, Ctrl+C, or an escaping
            # exception): the actuators are commanded to a measured neutral hold
            # instead of being left latched to an arbitrary stale data.ctrl. This is
            # a minimal hold only -- the controller owns the real e-stop semantics.
            self.hold_current_joints()
            telemetry_printer.close()
            if viewer_ctx is not None:
                viewer_ctx.close()
            if live_plotter is not None:
                live_plotter.close()

        if vision_scheduler is not None and vision_scheduler.failures:
            logger.warning(
                "Vision pipeline failed %d time(s) during this run; %d run(s) skipped for "
                "lack of tick slack, %d overran the %.3fs budget.",
                vision_scheduler.failures,
                vision_scheduler.skips,
                vision_scheduler.overruns,
                vision_scheduler.budget_s,
            )

        metrics = self._calculate_metrics(history, total_sim_time)
        return history, metrics

    def _calculate_metrics(self, history: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
        if not history:
            return {"status": "FAILED", "reason": "No telemetry recorded"}

        times = np.array([r["time_s"] for r in history])
        x_pos = np.array([r["base_x_m"] for r in history])
        y_pos = np.array([r["base_y_m"] for r in history])
        z_pos = np.array([r["base_z_m"] for r in history])
        vx_body = np.array([r["base_vx_body_mps"] for r in history])
        cmd_vx = np.array([r["cmd_vx"] for r in history])
        vy_body = np.array([r["base_vy_body_mps"] for r in history])
        cmd_vy = np.array([r["cmd_vy"] for r in history])
        yaw_rate = np.array([r["base_wz_radps"] for r in history])
        cmd_yaw = np.array([r["cmd_yaw"] for r in history])
        roll = np.array([r["base_roll_rad"] for r in history])
        pitch = np.array([r["base_pitch_rad"] for r in history])
        powers = np.array([r["power_w"] for r in history])

        vx_error = vx_body - cmd_vx
        vx_rmse = float(np.sqrt(np.mean(vx_error * vx_error)))
        vy_error = vy_body - cmd_vy
        vy_rmse = float(np.sqrt(np.mean(vy_error * vy_error)))
        yaw_error = yaw_rate - cmd_yaw
        yaw_rmse = float(np.sqrt(np.mean(yaw_error * yaw_error)))

        fwd_displacement = float(x_pos[-1] - x_pos[0])
        lat_displacement = float(y_pos[-1] - y_pos[0])
        total_distance = float(np.sum(np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2)))

        min_height = float(np.min(z_pos))
        max_roll = float(np.max(np.abs(roll)))
        max_pitch = float(np.max(np.abs(pitch)))
        mean_power = float(np.mean(powers))

        passed = bool(
            min_height >= 0.22
            and max_roll <= 0.35
            and max_pitch <= 0.35
            and np.all(np.isfinite(x_pos))
        )

        return {
            "status": "PASS" if passed else "FAIL",
            "duration_s": round(float(times[-1] - times[0]), 2),
            "forward_displacement_m": round(fwd_displacement, 3),
            "lateral_displacement_m": round(lat_displacement, 3),
            "total_distance_m": round(total_distance, 3),
            "mean_vx_mps": round(float(np.mean(vx_body)), 3),
            "peak_vx_mps": round(float(np.max(np.abs(vx_body))), 3),
            "vx_tracking_rmse_mps": round(vx_rmse, 4),
            "mean_vy_mps": round(float(np.mean(vy_body)), 3),
            "peak_vy_mps": round(float(np.max(np.abs(vy_body))), 3),
            "vy_tracking_rmse_mps": round(vy_rmse, 4),
            "yaw_rate_tracking_rmse_radps": round(yaw_rmse, 4),
            "min_base_height_m": round(min_height, 4),
            "max_abs_roll_rad": round(max_roll, 4),
            "max_abs_pitch_rad": round(max_pitch, 4),
            "mean_power_w": round(mean_power, 2),
        }


def save_artifacts(
    history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    llm_result: LLMCommandResult,
    out_dir,
):
    """Save data.csv, metrics.json, plots.png, and run_meta.json."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save data.csv
    csv_path = os.path.join(out_dir, "data.csv")
    if history:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # 2. Save metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 3. Save run_meta.json
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_prompt": llm_result.user_prompt,
        "model_name": llm_result.model_name,
        "status": llm_result.status,
        "explanation": llm_result.explanation,
        "wall_time_s": llm_result.wall_time_s,
        "commands": [c.to_dict() for c in llm_result.commands],
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 4. Generate diagnostic plot
    if history:
        generate_plots(history, metrics, llm_result.user_prompt, os.path.join(out_dir, "plots.png"))


def generate_plots(history: List[Dict[str, Any]], metrics: Dict[str, Any], prompt: str, save_path):
    """Generate diagnostic tracking and state plots."""
    times = [r["time_s"] for r in history]
    x_pos = [r["base_x_m"] for r in history]
    y_pos = [r["base_y_m"] for r in history]
    z_pos = [r["base_z_m"] for r in history]
    vx_act = [r["base_vx_body_mps"] for r in history]
    vx_cmd = [r["cmd_vx"] for r in history]
    vy_act = [r["base_vy_body_mps"] for r in history]
    vy_cmd = [r["cmd_vy"] for r in history]
    wz_act = [r["base_wz_radps"] for r in history]
    wz_cmd = [r["cmd_yaw"] for r in history]
    roll = [r["base_roll_rad"] for r in history]
    pitch = [r["base_pitch_rad"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Go2 CHAMP Trajectory: '{prompt}'\nStatus: {metrics.get('status')} | Duration: {metrics.get('duration_s')}s", fontsize=11, fontweight="bold")

    # 1. 2D Trajectory
    ax0 = axes[0, 0]
    ax0.plot(x_pos, y_pos, color="#2563eb", lw=2, label="Actual Path")
    ax0.scatter([x_pos[0]], [y_pos[0]], color="green", s=50, label="Start", zorder=5)
    ax0.scatter([x_pos[-1]], [y_pos[-1]], color="red", s=50, label="End", zorder=5)
    ax0.set_xlabel("World X (m)")
    ax0.set_ylabel("World Y (m)")
    ax0.set_title("2D Ground Trajectory")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.axis("equal")

    # 2. Forward & Lateral Velocity Tracking (vx, vy)
    ax1 = axes[0, 1]
    ax1.plot(times, vx_cmd, color="#dc2626", linestyle="--", label="Cmd vx")
    ax1.plot(times, vx_act, color="#2563eb", lw=1.5, label="Obs vx")
    ax1.plot(times, vy_cmd, color="#ea580c", linestyle=":", label="Cmd vy")
    ax1.plot(times, vy_act, color="#059669", lw=1.5, label="Obs vy")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_title(f"Velocity Tracking (vx RMSE: {metrics.get('vx_tracking_rmse_mps', 0.0):.3f}, vy RMSE: {metrics.get('vy_tracking_rmse_mps', 0.0):.3f} m/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3. Yaw Rate Tracking
    ax2 = axes[1, 0]
    ax2.plot(times, wz_cmd, color="#dc2626", linestyle="--", label="Command Yaw Rate")
    ax2.plot(times, wz_act, color="#7c3aed", lw=1.5, label="Measured Yaw Rate")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Yaw Rate (rad/s)")
    ax2.set_title(f"Heading Rate Tracking (RMSE: {metrics.get('yaw_rate_tracking_rmse_radps', 0.0):.3f} rad/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 4. Base Attitude & Height
    ax3 = axes[1, 1]
    ax3.plot(times, z_pos, color="#2563eb", label="Height (z)")
    ax3.plot(times, roll, color="#d97706", alpha=0.7, label="Roll (rad)")
    ax3.plot(times, pitch, color="#0891b2", alpha=0.7, label="Pitch (rad)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("State")
    ax3.set_title(f"Base Height & Attitude (Min Height: {metrics.get('min_base_height_m', 0.0):.3f} m)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main CLI Loop
# -----------------------------------------------------------------------------

import csv


def query_llm_with_cancellation(client: Any, prompt: str) -> Optional[LLMCommandResult]:
    """Query LLM in a daemon worker thread with 150ms main-thread yielding.
    
    This guarantees that Ctrl+C (SIGINT) is caught and handled immediately on Windows
    instead of freezing in a blocking socket read.
    """
    box: List[LLMCommandResult] = []
    err: List[Exception] = []

    def worker():
        try:
            res = client.query(prompt)
            box.append(res)
        except Exception as e:
            err.append(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    t_start = time.perf_counter()
    last_print = t_start
    model_disp = getattr(client, "model", "LLM")
    print(f"Waiting for [{model_disp}]... (Press Ctrl+C to abort)", end="", flush=True)

    try:
        while thread.is_alive():
            thread.join(timeout=0.15)
            now = time.perf_counter()
            if now - last_print >= 1.0:
                elapsed = int(now - t_start)
                print(f"\rWaiting for [{model_disp}]... [{elapsed}s elapsed] (Press Ctrl+C to abort)   ", end="", flush=True)
                last_print = now
        print()  # newline
    except KeyboardInterrupt:
        print(f"\n\n[Aborted] LLM query interrupted by user (Ctrl+C) after {time.perf_counter() - t_start:.1f}s.")
        return None

    if err:
        raise err[0]
    return box[0] if box else None


def process_prompt(
    prompt: str,
    client: Any,
    runner: Go2ChampSimulationRunner,
    gui: bool = True,
    live_plot: bool = True,
    max_duration: float = 30.0,
) -> Tuple[Optional[LLMCommandResult], Optional[Dict[str, Any]]]:
    """Translate prompt via LLM, execute in simulation, and save artifacts."""
    print(f"\n=======================================================")
    print(f"Prompt: \"{prompt}\"")
    print(f"=======================================================")

    t0 = time.perf_counter()
    llm_result = query_llm_with_cancellation(client, prompt)
    t_llm = time.perf_counter() - t0

    if llm_result is None:
        return None, None

    print(f"LLM [{llm_result.model_name}] Response ({t_llm:.2f}s):")
    print(f"  Status:      {llm_result.status.upper()}")
    print(f"  Explanation: {llm_result.explanation}")
    print(f"  Commands:    {len(llm_result.commands)}")
    for idx, cmd in enumerate(llm_result.commands):
        if cmd.mode == MotionMode.WAYPOINT and cmd.waypoint:
            print(f"    [{idx}] {cmd.mode.name:<8} target=({cmd.waypoint.x:+.2f}, {cmd.waypoint.y:+.2f})m speed={cmd.waypoint.speed:.2f}m/s dur={cmd.duration:.1f}s (Closed-Loop Waypoint Navigation)")
        else:
            print(f"    [{idx}] {cmd.mode.name:<8} vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} yaw={cmd.yaw_rate:+.2f} dur={cmd.duration:.1f}s")

    print("\nExecuting motion in MuJoCo simulation...")
    t_sim0 = time.perf_counter()
    history, metrics = runner.execute_commands(
        commands=llm_result.commands,
        gui=gui,
        max_duration=max_duration,
        live_plot=live_plot,
        prompt=prompt,
    )
    t_sim = time.perf_counter() - t_sim0

    print(f"\nSimulation Finished ({t_sim:.2f}s):")
    print(f"  Status:               {metrics.get('status')}")
    print(f"  Duration:             {metrics.get('duration_s')} s")
    print(f"  Forward Displacement: {metrics.get('forward_displacement_m')} m")
    print(f"  Lateral Displacement: {metrics.get('lateral_displacement_m')} m")
    print(f"  Total 2D Distance:    {metrics.get('total_distance_m')} m")
    print(f"  Observed vx:          mean={metrics.get('mean_vx_mps', 0.0):+.3f} m/s, peak={metrics.get('peak_vx_mps', 0.0):.3f} m/s (RMSE: {metrics.get('vx_tracking_rmse_mps', 0.0):.4f} m/s)")
    print(f"  Observed vy:          mean={metrics.get('mean_vy_mps', 0.0):+.3f} m/s, peak={metrics.get('peak_vy_mps', 0.0):.3f} m/s (RMSE: {metrics.get('vy_tracking_rmse_mps', 0.0):.4f} m/s)")
    print(f"  Yaw Tracking RMSE:    {metrics.get('yaw_rate_tracking_rmse_radps')} rad/s")
    print(f"  Min Base Height:      {metrics.get('min_base_height_m')} m")

    # Create run output directory
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip()[:30]).strip("_").lower()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOG_DIR, f"{timestamp_str}_{slug}")
    save_artifacts(history, metrics, llm_result, out_dir)
    print(f"Artifacts saved to: {out_dir}")

    return llm_result, metrics


def process_vlm_prompt(
    prompt: str,
    vlm_grounder: VLMGoalGrounder,
    runner: Go2ChampSimulationRunner,
    gui: bool = True,
    live_plot: bool = True,
    clearance_m: float = 0.5,
    max_duration: float = 30.0,
) -> Tuple[Optional[VLMGroundingResult], Optional[Dict[str, Any]]]:
    """Execute natural language visual grounding via Gemini 2.5 Flash VLM and waypoint navigation."""
    print(f"\n=======================================================")
    print(f"VLM Visual Goal Prompt: \"{prompt}\"")
    print(f"=======================================================")

    renderer = mujoco.Renderer(runner.model, height=240, width=320)
    runner.reset_to_standing()
    mujoco.mj_step(runner.model, runner.data)

    print("Capturing front_cam RGB-D and querying Gemini 2.5 Flash Multimodal...")
    t0 = time.perf_counter()
    vlm_res = vlm_grounder.ground_instruction(
        renderer=renderer,
        model=runner.model,
        data=runner.data,
        instruction=prompt,
        clearance_m=clearance_m,
    )
    t_vlm = time.perf_counter() - t0

    print(f"\nVLM [{vlm_res.model_name}] Response ({t_vlm:.2f}s):")
    if vlm_res.target_found and vlm_res.target:
        tgt = vlm_res.target
        print(f"  Target Detected:      YES ('{tgt.label}', confidence: {tgt.confidence:.2f})")
        print(f"  Visual Description:   {tgt.description}")
        print(f"  Bounding Box [0..1k]: [{int(tgt.box_2d.ymin)}, {int(tgt.box_2d.xmin)}, {int(tgt.box_2d.ymax)}, {int(tgt.box_2d.xmax)}]")
        print(f"  Observed Depth:       {tgt.depth_m:.3f} m")
        print(f"  3D Body Position:     X={tgt.body_coords[0]:+.2f}m, Y={tgt.body_coords[1]:+.2f}m, Z={tgt.body_coords[2]:+.2f}m")
        print(f"  Waypoint Target:      X={tgt.target_waypoint[0]:+.2f}m, Y={tgt.target_waypoint[1]:+.2f}m (clearance buffer: {clearance_m:.2f}m)")
    else:
        print(f"  Target Detected:      NO")
        print(f"  Explanation:          {vlm_res.explanation}")

    print(f"  Motion Commands:      {len(vlm_res.motion_commands)}")
    for idx, cmd in enumerate(vlm_res.motion_commands):
        if cmd.mode == MotionMode.WAYPOINT and cmd.waypoint:
            print(f"    [{idx}] {cmd.mode.name:<8} target=({cmd.waypoint.x:+.2f}, {cmd.waypoint.y:+.2f})m speed={cmd.waypoint.speed:.2f}m/s dur={cmd.duration:.1f}s (Closed-Loop Waypoint Navigation)")
        else:
            print(f"    [{idx}] {cmd.mode.name:<8} vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} yaw={cmd.yaw_rate:+.2f} dur={cmd.duration:.1f}s")

    print("\nExecuting motion in MuJoCo simulation...")
    t_sim0 = time.perf_counter()
    history, metrics = runner.execute_commands(
        commands=vlm_res.motion_commands,
        gui=gui,
        max_duration=max_duration,
        live_plot=live_plot,
        prompt=prompt,
    )
    t_sim = time.perf_counter() - t_sim0

    print(f"\nSimulation Finished ({t_sim:.2f}s):")
    print(f"  Status:               {metrics.get('status')}")
    print(f"  Duration:             {metrics.get('duration_s')} s")
    print(f"  Forward Displacement: {metrics.get('forward_displacement_m')} m")
    print(f"  Lateral Displacement: {metrics.get('lateral_displacement_m')} m")
    print(f"  Total 2D Distance:    {metrics.get('total_distance_m')} m")
    print(f"  Observed vx:          mean={metrics.get('mean_vx_mps', 0.0):+.3f} m/s, peak={metrics.get('peak_vx_mps', 0.0):.3f} m/s (RMSE: {metrics.get('vx_tracking_rmse_mps', 0.0):.4f} m/s)")
    print(f"  Observed vy:          mean={metrics.get('mean_vy_mps', 0.0):+.3f} m/s, peak={metrics.get('peak_vy_mps', 0.0):.3f} m/s (RMSE: {metrics.get('vy_tracking_rmse_mps', 0.0):.4f} m/s)")
    print(f"  Yaw Tracking RMSE:    {metrics.get('yaw_rate_tracking_rmse_radps')} rad/s")
    print(f"  Min Base Height:      {metrics.get('min_base_height_m')} m")

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip()[:30]).strip("_").lower()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LOG_DIR, f"{timestamp_str}_vlm_{slug}")

    llm_equivalent = LLMCommandResult(
        user_prompt=prompt,
        raw_response=vlm_res.raw_response,
        thinking=None,
        parsed_json=vlm_res.to_dict(),
        commands=vlm_res.motion_commands,
        status="success" if vlm_res.target_found else "fail",
        explanation=vlm_res.explanation,
        error_message=vlm_res.error_message,
        wall_time_s=t_vlm,
        model_name=vlm_res.model_name,
    )
    save_artifacts(history, metrics, llm_equivalent, out_dir)
    print(f"Artifacts saved to: {out_dir}")

    return vlm_res, metrics


def main():
    parser = argparse.ArgumentParser(description="Go2 CHAMP Natural Language Interface in MuJoCo")
    parser.add_argument("--prompt", type=str, default=None, help="Single natural language command to execute")
    parser.add_argument("--prompt-file", type=pathlib.Path, default=None, help="File containing prompts (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL command prompt")

    # VLM Multimodal Spatial Grounding parameters
    parser.add_argument("--vlm", action="store_true", help="Enable VLM multimodal visual grounding via Gemini 2.5 Flash")
    parser.add_argument("--target-x", type=float, default=2.0, help="X coordinate of benchmark landmark object in meters (default: 2.0)")
    parser.add_argument("--target-y", type=float, default=0.3, help="Y coordinate of benchmark landmark object in meters (default: 0.3)")
    parser.add_argument("--target-color", type=str, default="red", help="Color or name of landmark object (default: red)")
    parser.add_argument("--clearance", type=float, default=0.5, help="Stopping distance clearance from grounded object in meters (default: 0.5)")

    # Provider selection (Mock client completely removed - live LLM connection is required)
    parser.add_argument("--backend", "--provider", choices=["server", "omniroute", "ollama", "gemini", "smolvla"], default=None, help="LLM/VLA provider backend")
    parser.add_argument("--server", "--live-server", action="store_true", help="Connect to the live running Model Server (start_model_server.py)")
    parser.add_argument("--omniroute", action="store_true", help="Use OmniRoute / OpenAI-compatible endpoint directly")
    parser.add_argument("--gemini", action="store_true", help="Use Google AI Studio Gemini/Gemma API backend")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama backend")
    parser.add_argument("--smolvla", action="store_true", help="Use Hugging Face SmolVLA (lerobot/smolvla_base) as local VLA policy")

    # API & Host parameters
    parser.add_argument("--server-url", type=str, default=DEFAULT_MODEL_SERVER_URL, help=f"Live Model Server URL (default: {DEFAULT_MODEL_SERVER_URL})")
    parser.add_argument("--token", type=str, default=None, help="Shared secret for the Live Model Server (or set via MODEL_SERVER_TOKEN environment variable)")
    parser.add_argument("--omniroute-url", type=str, default=DEFAULT_OMNIROUTE_URL, help=f"OmniRoute base URL (default: {DEFAULT_OMNIROUTE_URL})")
    parser.add_argument("--omniroute-api-key", type=str, default="", help="OmniRoute API key if required")
    parser.add_argument("--gemini-api-key", type=str, default=None, help="Google Gemini API key (or set via GEMINI_API_KEY environment variable)")
    parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST, help="Ollama server host URL")
    parser.add_argument("--model", type=str, default=None, help="Model name override")

    # Simulation, GUI & Live Plotting parameters
    parser.add_argument("--gui", dest="gui", action="store_true", default=True, help="Launch interactive MuJoCo GUI passive viewer (default: True)")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Run headlessly without GUI viewer")
    parser.add_argument("--live-plot", dest="live_plot", action="store_true", default=True, help="Show real-time live matplotlib tracking plots while sim is running (default: True)")
    parser.add_argument("--no-live-plot", dest="live_plot", action="store_false", help="Disable real-time live matplotlib tracking plots")
    parser.add_argument("--duration-limit", type=float, default=25.0, help="Maximum duration cap in seconds")
    args = parser.parse_args()

    # Mode: VLM Multimodal Spatial Grounding
    if args.vlm:
        vlm_grounder = VLMGoalGrounder(api_key=args.gemini_api_key, model=args.model or DEFAULT_GEMINI_MODEL)
        available, msg = vlm_grounder.check_availability()
        if not available:
            raise ConnectionError(f"VLM Grounder unavailable: {msg}")
        print(f"✓ Connected to Gemini VLM Grounder: {msg}")

        # Create scene with benchmark landmark target
        xml_scene = create_vlm_scene_xml(target_pos=(args.target_x, args.target_y, 0.4))
        runner = Go2ChampSimulationRunner(xml_string=xml_scene)

        default_prompt = args.prompt or f"Walk to the {args.target_color} cylinder"
        process_vlm_prompt(
            default_prompt,
            vlm_grounder,
            runner,
            gui=args.gui,
            live_plot=args.live_plot,
            clearance_m=args.clearance,
            max_duration=args.duration_limit,
        )
        return

    # Determine backend
    backend = None
    if args.server:
        backend = "server"
    elif args.omniroute:
        backend = "omniroute"
    elif args.gemini:
        backend = "gemini"
    elif args.ollama:
        backend = "ollama"
    elif args.smolvla:
        backend = "smolvla"
    elif args.backend:
        backend = args.backend

    # Connect strictly to live LLM backend (raises error if unavailable; no mock fallback)
    client = select_live_llm_client(
        backend=backend,
        model=args.model,
        server_url=args.server_url,
        omniroute_url=args.omniroute_url,
        omniroute_api_key=args.omniroute_api_key,
        gemini_api_key=args.gemini_api_key,
        ollama_host=args.ollama_host,
        auth_token=args.token,
    )

    runner = Go2ChampSimulationRunner()

    try:
        # Mode 1: Single Prompt
        if args.prompt:
            process_prompt(args.prompt, client, runner, gui=args.gui, live_plot=args.live_plot, max_duration=args.duration_limit)
            return

        # Mode 2: Prompt File
        if args.prompt_file:
            if not args.prompt_file.exists():
                print(f"Error: Prompt file {args.prompt_file} not found.")
                sys.exit(1)
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
            for p in prompts:
                res, met = process_prompt(p, client, runner, gui=args.gui, live_plot=args.live_plot, max_duration=args.duration_limit)
                if res is None:
                    break
            return

        # Mode 3: Interactive REPL (default if neither prompt nor prompt-file given)
        print("\n" + "=" * 60)
        print("Go2 CHAMP Natural Language Simulation REPL")
        print("Commands:")
        print("  Type any natural language instruction (e.g. 'walk forward 2 meters', 'turn left')")
        print("  Type 'quit' or 'exit' to exit.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Go2> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Exiting Go2 CHAMP REPL. Goodbye.")
                    break
                process_prompt(user_input, client, runner, gui=args.gui, live_plot=args.live_plot, max_duration=args.duration_limit)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Go2 CHAMP REPL. Goodbye.")
                break
    except KeyboardInterrupt:
        print("\n[Exiting] Program aborted by user (Ctrl+C). Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
