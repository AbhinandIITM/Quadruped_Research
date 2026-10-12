# Unitree Go2 CHAMP Locomotion & Local DeepSeek-R1 Natural Language Interface

A modular, reproducible, simulation-only quadruped locomotion stack porting the open-source **CHAMP** (MIT Cheetah-inspired) hierarchical gait kinematics to a **Unitree Go2** in MuJoCo, controlled via high-level commands and a local **DeepSeek-R1** (`deepseek-r1:8b` via Ollama) natural language interface.

---

## Key Features

1. **CHAMP Gait Kinematics & Analytic 3-DOF IK**:
   - Exact analytic inverse kinematics for Unitree Go2 geometry ($l_1 = 0.213\,\text{m}, l_2 = 0.213\,\text{m}, y_{\text{offset}} = 0.0955\,\text{m}$).
   - Diagonal trot phase generator ($T_{\text{stride}} = 0.44\,\text{s}$, $T_{\text{stance}} = 0.22\,\text{s}$, $T_{\text{swing}} = 0.22\,\text{s}$) pairing Front-Right/Rear-Left at phase 0.0 and Front-Left/Rear-Right at phase 0.5.
   - 12-point Bézier swing curves providing $C^2$ boundary continuity (zero velocity and acceleration at liftoff/touchdown for minimal ground impact).
   - Cosine stance retraction profiles with vertical compliance.
   - Foothold planning incorporating the Raibert velocity heuristic, stance stabilization, and hip tangential velocity coupling for omnidirectional translation and yaw steering.

2. **Trajectory & 2D Waypoint Following**:
   - Closed-loop body-frame forward velocity ($v_x$), lateral velocity ($v_y$), and yaw rate ($\omega_z$) tracking.
   - Acceleration slew-rate limiters ($a_{\text{lin, max}} = 1.0\,\text{m/s}^2, \alpha_{\text{yaw, max}} = 2.0\,\text{rad/s}^2$) preventing dynamic tipping.
   - 2D world-frame waypoint navigation with heading regulation and automatic in-place rotation for heading errors exceeding $0.45\,\text{rad}$ ($25^\circ$).

3. **Local DeepSeek-R1 (Ollama) Natural Language Interface**:
   - Communicates with local `deepseek-r1:8b` at `http://localhost:11434/api/chat` using Python's standard-library `urllib.request` (zero third-party API dependencies).
   - Enforces strict structured output via Ollama's `"format": MOTION_COMMAND_SCHEMA`.
   - Isolates model `<think>` reasoning traces from the clean JSON command payload.
   - Automatically falls back to an offline mock client when Ollama is unavailable.
   - Immutable JSONL audit logging (`logs/llm_audit.jsonl`) recording prompt, model reasoning, raw payload, parsed commands, and validation status.

4. **Strict Simulation-Only Safety Guardrails**:
   - **Simulation Isolation**: Operates exclusively inside local MuJoCo physics. Never connects to physical robot hardware, Unitree SDKs, shell commands, or external networks.
   - **No Joint/Torque Exposure**: The LLM is restricted to high-level `MotionCommand` structures (`vx`, `vy`, `yaw_rate`, `duration`, `waypoint`); it cannot synthesize joint angles or motor torques.
   - **Parameter Clamping & Fail-Closed Behavior**: Velocities, durations, and coordinates are strictly clamped against `SafetyBounds`. Non-finite values (`NaN`, `Inf`) and malformed responses immediately fail closed to a safe 1.0 s stationary stand.
   - **Runtime Safety Supervisor**: Continuously monitors pitch, roll, base height, and joint limits in MuJoCo, initiating an emergency stand upon fault detection.

---

## Architecture Overview

```
                                  +------------------------------------+
                                  |    Natural Language Prompt         |
                                  | ("Walk forward at 0.35 m/s for 3s")|
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Local DeepSeek-R1 (Ollama :11434)  |
                                  | Structured Output: JSON Schema     |
                                  +------------------------------------+
                                                    |
                                  +-----------------+------------------+
                                  |                                    |
                                  v                                    v
                     +--------------------------+         +--------------------------+
                     |  llm_audit.jsonl Logging |         | Strict Schema Parser &   |
                     |  (Prompt, Think, Status) |         | SafetyBounds Clamp       |
                     +--------------------------+         +--------------------------+
                                                                       |
                                                                       v Validated MotionCommand(s)
+---------------------------------------------------------------------------------------------------+
| MuJoCo Simulation Loop (Go2ChampSimulationRunner / Go2ChampController)                           |
|                                                                                                   |
|  1. Waypoint & Velocity Regulation  --> Slew Limiter (a_max=1.0 m/s^2, alpha_max=2.0 rad/s^2)    |
|  2. Hip Kinematic Coupling           --> v_hip = v_body + omega x r_hip                           |
|  3. CHAMP Phase Generator            --> Diagonal Trot (T_st=0.22s, T_sw=0.22s)                   |
|  4. Foothold & Trajectory Generator  --> 12-point Bezier Swing + Cosine Stance Retraction         |
|  5. 3-DOF Analytic IK                --> theta_hip, theta_thigh, theta_calf                       |
|  6. Safety Supervisor                --> Pitch/Roll/Height Fault Detection (Fail-Closed Stand)    |
|  7. MuJoCo Physics Step              --> mj_step(model, data) at 500 Hz (Decimated Telemetry)     |
+---------------------------------------------------------------------------------------------------+
                                                    |
                                                    v
                               +------------------------------------------+
                               | Artifacts: data.csv, metrics.json, plots |
                               +------------------------------------------+
```

---

## Directory Structure

```
Quadruped/
├── PROGRESS_REPORT.md            # Comprehensive progress report, subsystem matrix, and roadmap
├── WALKTHROUGH.md                # Full capability walkthrough (B -> D -> C -> A) and live telemetry
├── CAPABILITIES_PLAN.md          # Technical specifications and phased execution plan
├── GO2_CHAMP_LLM_REPORT.md       # Full engineering report and mathematical derivations
├── METHODOLOGY.md                # Core algorithmic and locomotion methodology
├── go2_champ_control.py          # Core kinematics, IK, phase generator, Bézier curves, controller
├── go2_balance_qp.py             # Convex MPC / SRBD active-set QP ground reaction force solver
├── go2_multi_gait.py             # Unified multi-gait engine (Trot, Pace, Bound, Pronk, Walk)
├── go2_vision_perception.py      # Depth buffer extraction, sector clearance, reactive steering
├── go2_vlm_perception.py         # Gemini 2.5 Flash RGB-D multimodal bounding box & 3D back-projection
├── go2_smolvla.py                # Hugging Face SmolVLA (lerobot/smolvla_base) continuous policy agent
├── go2_terrain_generator.py      # Procedural MuJoCo stair and slope XML world generator
├── go2_champ_trot.py             # Standalone open-loop/closed-loop trot simulation entry point
├── go2_llm_commands.py           # Multi-provider LLM client (Ollama, Gemini, OmniRoute, Live Server)
├── go2_champ_llm_cli.py          # Interactive natural language CLI & simulation runner (Live Plotting)
├── evaluate_champ_trajectories.py# 7-scenario benchmark suite & metrics generator
├── scenarios/                    # Scenario definitions (JSON command schedules)
│   ├── forward_speed_step.json   # Multi-step velocity profile (0.2 -> 0.45 m/s)
│   ├── obstacle_avoidance.json   # Forward locomotion with frontal obstacle avoidance
│   ├── s_curve.json              # Forward velocity + alternating sinusoidal yaw rates
│   ├── stand_hold.json           # Stationary standing stability verification
│   ├── stop_and_hold.json        # Fast trot deceleration to a stable stand
│   ├── turn_in_place.json        # Pure yaw rotation without linear translation
│   └── waypoint_square.json      # 4-corner 2D world-frame waypoint navigation
├── tests/                        # Comprehensive unit test suite (86 tests)
│   ├── test_champ_control.py     # Tests for IK, phase generator, Bezier curves, slew, supervisor
│   ├── test_llm_commands.py      # Tests for JSON parsing, clamping, adversarial inputs, mock network
│   ├── test_vision_perception.py # Tests for RGB-D depth perception and reactive avoidance
│   ├── test_balance_qp.py        # Tests for SRBD active-set QP solver and leg Jacobian
│   ├── test_multi_gait.py        # Tests for trot, pace, bound, pronk, walk transitions
│   ├── test_terrain_adaptation.py# Tests for procedural stair and slope terrain generation
│   ├── test_vlm_perception.py    # Tests for Gemini 2.5 Flash multimodal spatial grounding
│   ├── test_smolvla.py           # Tests for Hugging Face SmolVLA preprocessing, policy, action decoder
│   └── fixtures/                 # Static JSON test fixtures for LLM responses
└── unitree_go2/                  # Go2 URDF and MuJoCo XML description files
```

---

## Quickstart Guide

### Prerequisites

- Python 3.10+
- MuJoCo (`pip install mujoco matplotlib numpy`)
- (Optional for live LLM inference) [Ollama](https://ollama.com/) with `deepseek-r1:8b`:
  ```bash
  ollama pull deepseek-r1:8b
  ollama serve
  ```

### 1. Run Unit Tests

Execute the 86 unit tests covering kinematics, phase transitions, slew limits, vision perception, QP balance force allocation, multi-gait scheduling, procedural terrains, VLM spatial grounding, and Hugging Face SmolVLA policy (runs headlessly in ~0.88s):

```bash
python -m unittest discover -s tests -v
```

All 86 tests run headlessly in $<0.90\,\text{s}$ using offline fixtures and mocked HTTP transports without requiring active model servers.

### 2. Run the Benchmark Evaluation Suite

Run all benchmark scenarios in MuJoCo (GUI by default, `--no-gui` for headless CI):

```bash
python evaluate_champ_trajectories.py --scenario all
```

Or run a specific scenario:

```bash
python evaluate_champ_trajectories.py --scenario s_curve
python evaluate_champ_trajectories.py --scenario waypoint_square
python evaluate_champ_trajectories.py --scenario obstacle_avoidance
```

This generates `data.csv` (250 Hz telemetry), `metrics.json`, and `plots.png` under `logs/benchmark_suite_<timestamp>/`.

### 3. VLM Multimodal Spatial Grounding (Gemini 2.5 Flash)

The robot ingests RGB-D camera frames from `<camera name="front_cam">`, queries Google Gemini 2.5 Flash Multimodal with natural language instructions, pinhole back-projects the detected 2D bounding box to 3D body coordinates, and executes safe closed-loop waypoint navigation with clearance buffering:

```bash
# Interactive MuJoCo 3D Viewer + 20 Hz Live Tracking Plots:
python go2_champ_llm_cli.py --vlm --prompt "Walk to the red cylinder" --gui

# Custom landmark coordinates and clearance:
python go2_champ_llm_cli.py --vlm --prompt "Approach the red cylinder" --target-x 2.5 --target-y -0.4 --clearance 0.6 --gui
```

### 4. Hugging Face SmolVLA Policy (`lerobot/smolvla_base`)

Load and execute Hugging Face's 450M parameter flow-matching Vision-Language-Action policy on consumer GPUs (NVIDIA RTX 4060) or CPU. SmolVLA ingests front camera RGB images ($256 \times 256$), 6D proprioceptive states $[v_x, v_y, \omega_z, \text{roll}, \text{pitch}, z]$, and text prompt tokens, predicting safe continuous velocity actions:

```bash
# Direct CLI execution (GUI viewer by default):
python go2_champ_llm_cli.py --smolvla --prompt "Walk forward at 0.3 m/s for 3 seconds" --gui

# Or specify a device / custom repo:
python go2_champ_llm_cli.py --smolvla --device cuda --model-repo lerobot/smolvla_base --gui
```

### 5. Natural Language Commands (CLI)

The CLI connects strictly to live LLM/VLA providers:
1. **Hugging Face SmolVLA Policy** (`--smolvla`, `lerobot/smolvla_base`)
2. **VLM Multimodal Grounding** (`--vlm`, Google Gemini 2.5 Flash)
3. **Auto-Connected Live Model Server** (`start_model_server.py` on port 5005)
4. **OmniRoute / OpenAI-Compatible Model Gateways** (`--omniroute`)
5. **Local Ollama Models** (`--ollama`, DeepSeek-R1 / Qwen)
6. **Google AI Studio** (`--gemini`, Gemini 2.5 / Gemma 2)

---

#### Option 1: Dedicated Model Server (Recommended Workflow)

With this decoupled microservice architecture, model selection and credentials live **strictly** in `start_model_server.py`. The CLI automatically detects and connects to the running server:

**Step 1: Start the Model Server:**
```bash
# Start with OmniRoute / OpenAI Gateway (default):
python start_model_server.py

# Or start with a specific model:
python start_model_server.py --backend omniroute --model deepseek-r1:8b --url http://localhost:8000/v1

# Or start with local Ollama:
python start_model_server.py --backend ollama --model deepseek-r1:8b

# Or start with Google Gemini:
python start_model_server.py --backend gemini --model gemini-2.5-flash

# Or serve Hugging Face SmolVLA locally:
python start_model_server.py --backend smolvla --device cuda
```

**Step 2: Run Prompts in MuJoCo (CLI connects automatically):**
```bash
python go2_champ_llm_cli.py --prompt "Walk forward at 0.35 m/s for 3 seconds"
python go2_champ_llm_cli.py --interactive
```

---

#### Option 2: Direct CLI Provider Modes

#### A. OmniRoute / OpenAI-Compatible Gateway:
Configure your `.env` file (or copy `.env.example`):
```bash
OMNIROUTE_BASE_URL=http://localhost:8000/v1
OMNIROUTE_MODEL=deepseek-r1:8b
```
Then run:
```bash
python go2_champ_llm_cli.py --omniroute --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

#### B. Local Ollama (DeepSeek-R1 / Qwen):
```bash
# Default DeepSeek-R1:8b:
python go2_champ_llm_cli.py --ollama --prompt "Walk forward at 0.35 m/s for 3 seconds"

# Custom local model:
python go2_champ_llm_cli.py --ollama --model qwen3-vl:latest --prompt "Turn left for 2 seconds"
```

#### C. Google AI Studio (Gemini / Gemma):
Add your API key to `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
Then run:
```bash
# Using default Gemini (gemini-2.5-flash):
python go2_champ_llm_cli.py --gemini --prompt "Walk forward at 0.35 m/s for 3 seconds"

# Using Gemma models (e.g. gemma-2-27b-it):
python go2_champ_llm_cli.py --gemini --model gemma-2-27b-it --prompt "Turn left for 2 seconds"
```

#### D. Headless Run without GUI:
```bash
python go2_champ_llm_cli.py --no-gui --prompt "Turn left for 3 seconds"
```

#### E. Interactive REPL:
```bash
python go2_champ_llm_cli.py --interactive
```

#### F. Offline Mock Client (Fast / Testing):
```bash
python go2_champ_llm_cli.py --mock --prompt "Walk forward slowly for 2 seconds"
```

---

## Benchmark Evaluation Results

The evaluation harness evaluates 6 reproducible scenarios under fixed initial state ($z_0 = 0.28\,\text{m}$, nominal standing configuration):

| Scenario | Status | Forward Dist (m) | Lateral Dist (m) | $v_x$ Tracking RMSE (m/s) | Yaw Tracking RMSE (rad/s) | Mean Base Height (m) | Peak Roll | Peak Pitch | Cost of Transport (CoT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `forward_speed_step` | ✅ **PASS** | 1.92 m | 0.08 m | 0.065 m/s | 0.515 rad/s | 0.269 m | 3.2° | 5.1° | 1.48 |
| `s_curve` | ✅ **PASS** | 1.32 m | 1.06 m | 0.068 m/s | 0.521 rad/s | 0.270 m | 3.9° | 5.8° | 1.51 |
| `stand_hold` | ✅ **PASS** | 0.00 m | 0.00 m | 0.011 m/s | 0.000 rad/s | 0.281 m | 0.0° | 0.8° | 0.38 |
| `stop_and_hold` | ✅ **PASS** | 0.83 m | 0.04 m | 0.090 m/s | 0.467 rad/s | 0.273 m | 3.4° | 5.5° | 1.51 |
| `turn_in_place` | ✅ **PASS** | -0.26 m | -0.21 m | 0.068 m/s | 0.443 rad/s | 0.268 m | 2.3° | 2.8° | 3.07 |
| `waypoint_square` | ✅ **PASS** | 0.04 m | 0.13 m | 0.067 m/s | 0.537 rad/s | 0.269 m | 3.3° | 5.4° | 1.73 |

### Acceptance Criteria
- **Stability**: Zero falls across all runs; base height strictly maintained above $0.22\,\text{m}$ (nominal $0.28\,\text{m}$).
- **Attitude Limits**: Peak body roll $<4.0^\circ$ and peak pitch $<6.0^\circ$ across all dynamic maneuvers.
- **Finite State**: $100\%$ finite numbers across joint angles, velocities, and actuator forces (zero `NaN` / `Inf`).
- **Telemetry Fidelity**: Full 250 Hz state logging with command tracking errors and energy expenditure.

---

## Safety Guardrail Specifications

```python
@dataclass(frozen=True)
class SafetyBounds:
    max_vx: float = 0.60            # Maximum forward velocity (m/s)
    min_vx: float = -0.25           # Maximum reverse velocity (m/s)
    max_vy: float = 0.35            # Maximum lateral velocity (m/s)
    max_yaw_rate: float = 1.20      # Maximum heading rate (rad/s)
    max_duration: float = 60.0      # Maximum execution duration per command (s)
    max_linear_accel: float = 1.0   # Slew rate limit on linear velocity (m/s^2)
    max_yaw_accel: float = 2.0      # Slew rate limit on yaw rate (rad/s^2)
    min_height_m: float = 0.18      # Minimum torso ground clearance (m)
    max_roll_rad: float = 0.45      # Maximum allowable body roll (~25.8 deg)
    max_pitch_rad: float = 0.45     # Maximum allowable body pitch (~25.8 deg)
```

1. **Strict Fail-Closed Architecture**: Non-JSON strings, missing keys, or unparseable text automatically default to:
   ```json
   {"mode": "stand", "duration": 1.0, "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
   ```
2. **Deterministic Sequence Conclusion**: Every dynamic movement sequence generated by the LLM concludes with an automatic 1.0 s stationary stand command to guarantee that the robot halts safely.
3. **Structured JSON Validation**: DeepSeek-R1 receives Ollama's `format` parameter enforcing schema compliance at the sampling level, eliminating free-form conversational chatter.

---

## References

1. **CHAMP Quadruped Controller**: [chvmp/champ](https://github.com/chvmp/champ) - Open-source ROS-based locomotion framework for quadrupedal robots.
2. **MIT Cheetah Locomotion**: Bledt, G., et al., *"Hierarchical controller for highly dynamic locomotion utilizing pattern modulation and impedance control: implementation on the MIT Cheetah robot"*, 2018.
3. **Unitree Go2 MuJoCo Description**: Unitree Robotics official Go2 model descriptions and URDF assets.
