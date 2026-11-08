# Unitree Go2 CHAMP Locomotion & Local DeepSeek-R1 Natural Language Interface

A modular, reproducible, simulation-only quadruped locomotion stack porting the open-source **CHAMP** (MIT Cheetah-inspired) hierarchical gait kinematics to a **Unitree Go2** in MuJoCo, controlled via high-level commands and a local **DeepSeek-R1** (`deepseek-r1:8b` via Ollama) natural language interface.

---

## Key Features

1. **CHAMP Gait Kinematics & Analytic 3-DOF IK**:
   - Exact analytic inverse kinematics for Unitree Go2 geometry ($l_1 = 0.213\,\text{m}, l_2 = 0.213\,\text{m}, y_{\text{offset}} = 0.0955\,\text{m}$).
   - Diagonal trot phase generator ($T_{\text{stride}} = 0.44\,\text{s}$, $T_{\text{stance}} = 0.22\,\text{s}$, $T_{\text{swing}} = 0.22\,\text{s}$) pairing Front-Right/Rear-Left at phase 0.0 and Front-Left/Rear-Right at phase 0.5.
   - 12-point (degree-11) Bézier swing curves giving $C^1$ boundary continuity: $z'(0) = z'(1) = 0$ exactly, so there is no vertical impact velocity at liftoff or touchdown. The vertical *curvature* is not zero — $z''(0) = -6.11$, $z''(1) = -7.86$ — and the realised apex is $1.0334\times$ the nominal swing height, reached at $s = 0.6266$ rather than at mid-swing.
   - Cosine stance retraction profiles with vertical compliance.
   - Foothold planning with hip tangential velocity coupling for omnidirectional translation and yaw steering, and a Raibert velocity-feedback term. Two measured caveats, both carried in [`report/notes/01_control_module.md`](report/notes/01_control_module.md) (§8.11 and the Raibert note): the commanded step length uses the **full stride period** rather than the stance duration, so it is twice the no-slip value and every stance foot scrubs backwards at $v\,T_{\text{swing}}$; and the feedback term is $(1+k_v)v_{\text{cmd}} - k_v v_{\text{meas}}$ rather than the textbook additive Raibert term, which inflates the step by 35 % when the measured velocity is zero.
   - A 5-preset multi-gait scheduler (Trot, Pace, Bound, Pronk, Walk) is wired into the control loop, but the action-contract validator never reads the LLM's `gait` field, so **every run in this repository is a TROT run** (phase offsets 0.0 / 0.5 / 0.5 / 0.0, stride period 0.44 s, duty 0.50).

2. **Trajectory & 2D Waypoint Following**:
   - Closed-loop body-frame forward velocity ($v_x$), lateral velocity ($v_y$), and yaw rate ($\omega_z$) tracking.
   - Acceleration slew-rate limiters ($a_{\text{lin, max}} = 1.2\,\text{m/s}^2, \alpha_{\text{yaw, max}} = 2.5\,\text{rad/s}^2$) preventing dynamic tipping.
   - 2D world-frame waypoint navigation with heading regulation and automatic in-place rotation for heading errors exceeding $0.45\,\text{rad}$ ($25^\circ$).

3. **Local DeepSeek-R1 (Ollama) Natural Language Interface**:
   - Communicates with local `deepseek-r1:8b` at `http://localhost:11434/api/chat` using Python's standard-library `urllib.request` (zero third-party API dependencies).
   - Enforces strict structured output via Ollama's `"format": MOTION_COMMAND_SCHEMA`.
   - Isolates model `<think>` reasoning traces from the clean JSON command payload.
   - Offline mock fallback is **disabled**: if no live backend answers, the CLI raises instead of substituting a mock client. (A mock client class still exists in the library and is still reachable through `python src/start_model_server.py --backend mock`.)
   - Immutable JSONL audit logging (`logs/llm_audit.jsonl`) recording prompt, model reasoning, raw payload, parsed commands, and validation status. The `--vlm` branch does **not** write an audit record — see [`report/notes/09_contract_audit.md`](report/notes/09_contract_audit.md) §5.1.

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
|  1. Waypoint & Velocity Regulation  --> Slew Limiter (a_max=1.2 m/s^2, alpha_max=2.5 rad/s^2)    |
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
├── README.md                    # This file — setup, usage, benchmark results, acceptance gates
├── pyproject.toml               # PEP 621 / setuptools configuration with editable installs & pytest
│
├── src/                         # Core Python modules & locomotion stack
│   ├── go2_champ_control.py     # Kinematics, 3-DOF IK/FK, Raibert foothold planner, 12-pt Bézier curves
│   ├── go2_balance_qp.py        # SRBD bounded fixed-point contact-force solver (friction-pyramid feasible; see module docstring)
│   ├── go2_multi_gait.py        # Unified multi-gait engine (Trot, Pace, Bound, Pronk, Walk) — wired in, but the LLM's `gait` field is dropped, so every run is a TROT
│   ├── go2_vision_perception.py # Depth buffer extraction, sector clearance, reactive steering
│   ├── go2_vlm_perception.py    # Gemini 2.5 Flash RGB-D multimodal bounding box & 3D back-projection
│   ├── go2_smolvla.py           # Hugging Face SmolVLA (lerobot/smolvla_base) continuous policy agent
│   ├── go2_llm_commands.py      # Multi-provider LLM client (Ollama, Gemini, OmniRoute, Live Server)
│   ├── go2_champ_llm_cli.py     # Interactive natural language CLI & simulation runner (Live Plotting)
│   ├── evaluate_champ_trajectories.py # Trajectory benchmark suite, telemetry recorder & metrics gate
│   └── start_model_server.py    # Lightweight authenticated local LLM model server microservice
│
├── utils/                       # Procedural generation & asset utilities
│   ├── go2_terrain_generator.py # Procedural MuJoCo stair and slope XML world generator
│   ├── dae_to_stl.py            # Collada DAE to STL mesh converter
│   └── urdftomjcf.py            # URDF to MuJoCo MJCF XML translator
│
├── experiments/                 # Standalone experimental scripts & evaluation scenarios
│   ├── go2_champ_trot.py        # Standalone open-loop/closed-loop trot simulation entry point
│   ├── go2_mujoco_trot.py       # Thread-safe dual-threaded trotting prototype with live Matplotlib
│   ├── go2_mujoco_trot_traj.py  # Trajectory-tracking trotting prototype with debug lock
│   ├── go2_unitree_guide.py     # Python/MuJoCo port of unitree_guide trotting controller
│   ├── sweep_duty_factor.py     # Duty-factor sweep behind the CoT convention argument
│   ├── adversarial_llm_eval.py  # Adversarial corpus driving the action contract's validator
│   ├── language_ablation.py     # Natural-language-ablation experiment (quantised commands)
│   └── scenarios/               # Standardized benchmark scenario JSON schedules (9 files)
│       ├── circle_orbit.json        # Geometric circle tracking (r = 1.5 m, v = 0.35 m/s, 15 s)
│       ├── figure_eight.json        # Geometric Gerono lemniscate (1.2 x 0.6, 20 s)
│       ├── forward_speed_step.json   # Multi-step velocity profile (0 -> 0.40 -> 0.20 -> 0 m/s)
│       ├── obstacle_avoidance.json   # Forward locomotion; only meaningful with `--vision`
│       ├── s_curve.json              # Forward velocity + alternating sinusoidal yaw rates
│       ├── stand_hold.json           # Stationary standing stability verification
│       ├── stop_and_hold.json        # Fast trot deceleration to a stable stand
│       ├── turn_in_place.json        # Pure yaw rotation without linear translation
│       └── waypoint_square.json      # 4-corner 2D world-frame waypoint navigation
│
├── third_party/                 # Cloned upstream and vendor sub-repositories
│   ├── champ/                   # Open-source CHAMP Quadruped Control library
│   ├── unitree_guide/           # Unitree Guide quadruped controller C++ repository
│   └── unitree_mujoco/          # Unitree MuJoCo robot descriptions and meshes
│
├── references/                  # Academic literature & reference papers
│   └── papers/                  # Foundational PDFs (kinematic optimization, quadruped IK)
│
├── results/                     # Experimental telemetry exports, benchmark runs & figures
│   └── telemetry_exports/       # Raw multi-channel CSV telemetry logs
│
├── report/                      # Paper-facing evidence: module references, notes, machine-readable figures
│   ├── notes/                   # Per-module references, the contract audit, the fresh benchmark report
│   ├── figures/                 # benchmark_results.json, language_ablation.json, adversarial_results*.json
│   └── main.tex                 # The paper
│
├── logs/                        # Benchmark run directories and captured console logs (gitignored)
│
├── tests/                       # Comprehensive unit test suite (442 tests + 78 subtests, 19 test modules)
│   ├── conftest.py              # Pytest path injection and environment configuration
│   ├── test_champ_control.py    # Kinematics, IK, phase generator, Bézier curves, slew, supervisor
│   ├── test_llm_commands.py     # JSON parsing, clamping, adversarial inputs, mock network
│   ├── test_balance_qp.py       # SRBD contact-force solver and leg Jacobian
│   ├── test_balance_qp_residual.py # Dynamic residual compensation & friction pyramid bounds
│   ├── test_multi_gait.py       # Trot, pace, bound, pronk, walk transitions
│   ├── test_multi_gait_transitions.py # Continuous contact phase blending across gait switches
│   ├── test_vision_perception.py# RGB-D depth perception and reactive obstacle avoidance
│   ├── test_vision_obstacle.py  # Camera-live obstacle injection and the geometric avoidance gate
│   ├── test_llm_planner.py      # Bounded two-level plan/execute/re-plan over a closed skill vocabulary
│   ├── test_vlm_perception.py   # Gemini 2.5 Flash multimodal spatial grounding
│   ├── test_vlm_api_key_transport.py # Transport security & header authorization validation
│   ├── test_smolvla.py          # SmolVLA preprocessing, policy inference, action decoder
│   ├── test_terrain_adaptation.py # Procedural stair and slope terrain generation
  │   └── test_hip_geometry.py # The hip constant, derived from the model
│   ├── test_scenario_validation.py # Schema validation, NaN/Inf gates, path-length criterion
│   ├── test_champ_llm_cli.py    # CLI argparsing, step decimation, path redirection
│   ├── test_model_server_auth.py# Bearer token authentication, timing attacks, loopback binding
│   ├── test_debug_thread_locking.py # Multi-threaded MjData debug_lock mutual exclusion
│   ├── test_friction_consistency.py # Floor-friction vs QP-mu consistency, cone utilisation
│   └── fixtures/                # Static JSON test fixtures for LLM responses
│
├── docs/                        # Archived research material (reference, not maintained)
│   ├── mathematical-formulations.md  # SRBD + contact-wrench QP + Raibert + Bézier
│   ├── literature-matrix.md          # State-of-the-art papers with DOIs
│   ├── cot-analysis.md               # Gross-vs-net CoT convention & measured values
│   ├── audit-findings.md             # 2026-10-15 audit prose (findings now partly fixed)
│   ├── REMAINING_CHANGES.md          # Open items and which paper claim each one blocks
│   ├── PROGRESS_REPORT.md            # Subsystem matrix and engineering roadmap
│   ├── METHODOLOGY.md                # Measurement conventions and benchmark protocol
│   ├── WALKTHROUGH.md                # Capability walkthrough and live telemetry
│   └── GO2_CHAMP_LLM_REPORT.md       # Long-form project report
│
├── experiments/scenarios/       # 9 benchmark scenario definitions (expected_metrics gates)
│
└── unitree_go2/                 # Unitree Go2 URDF and MuJoCo XML description files
```

---

## Quickstart Guide

### Prerequisites

- Python 3.10+
- MuJoCo (`pip install mujoco matplotlib numpy scipy pillow`)
- (Optional for live LLM inference) [Ollama](https://ollama.com/) with `deepseek-r1:8b`:
  ```bash
  ollama pull deepseek-r1:8b
  ollama serve
  ```

### 1. Run Unit Tests

Execute the test suites covering kinematics, phase transitions, slew limits, vision perception, contact-force allocation, multi-gait scheduling, procedural terrains, VLM spatial grounding, Hugging Face SmolVLA policy, the LLM action contract, and thread safety:

```bash
pytest
```

**442 tests + 78 subtests across 19 test modules**, all green, in ~30 s on the reference host
(`330 passed, 44 subtests passed in 26.6s`, 2026-10-20). The suite is CPU-bound, so **35 – 50 s**
is the honest band to quote. Everything runs headlessly with offline fixtures and mocked HTTP
transports — no active model server and no physical hardware required.

| Test module | Tests | Covers |
| :--- | ---: | :--- |
| `tests/test_scenario_validation.py` | **84** | Scenario schema validation, the non-finite gate, the path-length gate, the independent reference path, log retention, schedule-semantics rules |
| `tests/test_llm_commands.py` | **70** | The action contract: JSON parsing, mode/velocity consistency, waypoint clamping, non-finite and oversized-literal inputs, the four provider transports, the contract-revision guard |
| `tests/test_llm_planner.py` | **59** | The bounded two-level plan/execute/re-plan loop: the closed skill vocabulary, the six trusted preconditions, the re-plan budget, and the fail-closed handling of a malformed plan or re-plan |
| `tests/test_champ_control.py` | **49** | Kinematics, IK/FK, phase generator, Bézier curves, slew limiting, trajectory tracking, the safety supervisor, the fail-safe controller |
| `tests/test_champ_llm_cli.py` | **27** | CLI argument parsing, step decimation, pacing, the vision scheduler, exit-safe hold |
| `tests/test_vision_obstacle.py` | **24** | The camera-live obstacle scenario: runtime obstacle injection into the vendor MJCF, the three-sector clearance, and the four-check geometric gate |
| `tests/test_model_server_auth.py` | **20** | Bearer-token authentication, constant-time comparison, loopback binding, endpoint redaction |
| `tests/test_balance_qp_residual.py` | **15** | The contact QP's finiteness guards, wrench-residual reporting, regularisation scaling |
| `tests/test_multi_gait_transitions.py` | **13** | In-flight gait transitions, phase-blend continuity, the no-dead-tick phase sweep |
| `tests/test_vlm_perception.py` | **13** | Gemini bounding-box grounding, pinhole back-projection, the world-frame waypoint, the non-finite box guard |
| `tests/test_debug_thread_locking.py` | **12** | `MjData` debug-lock mutual exclusion in the threaded trot prototypes |
| `tests/test_smolvla.py` | **12** | SmolVLA observation preprocessing, tokenisation, the action decoder, `SafetyBounds` clamping |
| `tests/test_friction_consistency.py` | **11** | Floor-friction ↔ QP-$\mu$ consistency, friction-cone utilisation diagnostics |
| `tests/test_balance_qp.py` | **9** | SRBD equation of motion, friction-pyramid constraints, the leg Jacobian, the $\tau = J^\top f$ mapping |
| `tests/test_multi_gait.py` | **9** | The five canonical gait presets, duty factors, phase offsets, circular blending |
| `tests/test_vision_perception.py` | **7** | Depth extraction, sector clearance, the ground-horizon cutoff, reactive avoidance |
| `tests/test_terrain_adaptation.py` | **4** | Procedural stair and slope terrain generation |
| `tests/test_hip_geometry.py` | **3** | The hip constant derived from the model: the roll axis, the roll-to-pitch distance, and the reported foot positions |
| `tests/test_vlm_api_key_transport.py` | **1** | The VLM API key travels in a header, never in the URL |
| **Total** | **442** | + 78 subtests |

35 of those tests (`test_balance_qp.py`, `test_balance_qp_residual.py`,
`test_friction_consistency.py`) certify the **contact QP, which is not in the benchmarked
control loop** — they are evidence about the solver, not about the evaluated system.

### Scope of the benchmark

The headless benchmark (`src/evaluate_champ_trajectories.py`) measures **kinematic** outcomes on flat ground: base trajectory and attitude, joint tracking, power and cost of transport, reference-path tracking error, and a geometric foot-contact fraction. It does **not** measure any of the following, and results should not be cited as if they did:

- **Contact forces.** Contact is detected geometrically (`ncon` / `contact`), with no force magnitude. Friction-cone compliance is therefore unverifiable from a run.
- **The contact QP.** `unitree_go2/go2_description.xml` declares `<position kp="60" kv="3">` servos, so `data.ctrl` holds a joint *angle* and there is no channel to apply the QP's torque. `compute_stance_torques()` is exercised by unit tests only; the benchmarked controller path never calls it. Note also that the gains that actually run are **not** the XML's 60/3: `configure_model()` overwrites them with `kp = 100`, `kd = 4` at load time, in both shipped entry points.
- **State estimation.** The harness passes `data.qpos` / `data.qvel` to the controller directly — privileged simulator ground truth, not an estimate.
- **Controller wall time.** No per-step timing is recorded.
- **Terrain.** The procedural terrain generator is not wired into the harness (there is no `--terrain` flag); all runs are nominal flat ground.
- **Torque headroom.** `actuator_forcelimited` is `True` for all 12 actuators, and the `forcerange` is **±23.7 N·m** at the hip and thigh and **±45.43 N·m** at the calf. The limit **binds**: peak $|\tau|$ reaches exactly 23.7 N·m on a thigh joint in 5 of the 9 scenarios (measured in the 2026-10-21 run). The reported actuator forces — and therefore the reported power and cost of transport — are computed from **clipped** torques, so they are a **lower bound** on the mechanical work actually required.

See [`report/notes/05_evaluation_module.md`](report/notes/05_evaluation_module.md) and [`docs/audit-findings.md`](docs/audit-findings.md) for the per-claim detail behind each of these, and which of them block which paper claim.

### 2. Run the Benchmark Evaluation Suite

Run benchmark scenarios in MuJoCo (GUI by default, `--no-gui` for headless CI):

```bash
python src/evaluate_champ_trajectories.py --scenario waypoint_square
python src/evaluate_champ_trajectories.py --scenario s_curve
python src/evaluate_champ_trajectories.py --scenario all
```

Or using the installed console script:

```bash
go2-benchmark --scenario waypoint_square
```

This generates `data.csv` (250 Hz telemetry across 68 channels), `metrics.json`, and `plots.png` under `logs/benchmark_suite_<timestamp>/`.

### 3. Interactive Natural Language CLI

Run interactive or scripted natural language control with live 20 Hz telemetry plotting:

```bash
# Interactive natural language CLI with passive MuJoCo viewer:
python src/go2_champ_llm_cli.py --gui

# Or via console entry point:
go2-cli --gui
```

### 4. VLM Multimodal Spatial Grounding (Gemini 2.5 Flash)

The robot ingests RGB-D camera frames from `<camera name="front_cam">`, queries Google Gemini 2.5 Flash Multimodal with natural language instructions, pinhole back-projects the detected 2D bounding box to 3D body coordinates, and executes closed-loop waypoint navigation with clearance buffering. The grounding is **one-shot** (it is not re-evaluated as the robot approaches the target), and this branch builds its `MotionCommand`s directly rather than passing them through the shared fail-closed validator — see [`report/notes/09_contract_audit.md`](report/notes/09_contract_audit.md) §5.1:

```bash
# Interactive MuJoCo 3D Viewer + 20 Hz Live Tracking Plots:
python src/go2_champ_llm_cli.py --vlm --prompt "Walk to the red cylinder" --gui

# Custom landmark coordinates and clearance:
python src/go2_champ_llm_cli.py --vlm --prompt "Approach the red cylinder" --target-x 2.5 --target-y -0.4 --clearance 0.6 --gui
```

### 5. Hugging Face SmolVLA Policy (`lerobot/smolvla_base`)

Instantiates Hugging Face's 450M parameter flow-matching Vision-Language-Action policy architecture on consumer GPUs (NVIDIA RTX 4060) or CPU. **What the shipped code actually does, in the shipped entry points:**

- The policy is instantiated with `load_pretrained=False`, i.e. `SmolVLAPolicy(SmolVLAConfig(...))` — **randomly initialised weights**. `from_pretrained` is never reached, so the `lerobot/smolvla_base` checkpoint is never fetched.
- The observation is a **black image** (`np.zeros((256, 256, 3))`) and a **zero 6-D state vector** (`np.zeros(6)`). The simulator is never rendered for this path, so the policy observes nothing about the robot's pose, velocity or surroundings.
- The action decoder **prefers a regex parse of the prompt** over the network output for every dimension the prompt specifies (speed, duration, yaw rate, direction). For the canonical prompt below, all four network outputs are discarded and the action is a pure function of the parse.
- `check_availability()` returns `True` in this mode, so the CLI prints "Connected" — the string does not mean a trained policy is running.

```bash
# Direct CLI execution (GUI viewer by default):
python src/go2_champ_llm_cli.py --smolvla --prompt "Walk forward at 0.3 m/s for 3 seconds" --gui
```

There are no device-selection or model-repository flags on the CLI; the backend is selected with
`--smolvla` and the model name with `--model`.

### 6. Natural Language Commands (CLI)

The CLI connects strictly to live LLM/VLA providers:
1. **Hugging Face SmolVLA Policy** (`--smolvla`, `lerobot/smolvla_base`)
2. **VLM Multimodal Grounding** (`--vlm`, Google Gemini 2.5 Flash)
3. **Auto-Connected Live Model Server** (`src/start_model_server.py` on port 5005)
4. **OmniRoute / OpenAI-Compatible Model Gateways** (`--omniroute`)
5. **Local Ollama Models** (`--ollama`, DeepSeek-R1 / Qwen)
6. **Google AI Studio** (`--gemini`, Gemini 2.5 / Gemma 2)

---

#### Option 1: Dedicated Model Server (Recommended Workflow)

With this decoupled microservice architecture, model selection and credentials live **strictly** in `src/start_model_server.py`. The CLI automatically detects and connects to the running server:

**Step 1: Start the Model Server:**
```bash
# Start with OmniRoute / OpenAI Gateway (default):
python src/start_model_server.py

# Or start with a specific model:
python src/start_model_server.py --backend omniroute --model deepseek-r1:8b --url http://localhost:8000/v1

# Or start with local Ollama:
python src/start_model_server.py --backend ollama --model deepseek-r1:8b

# Or start with Google Gemini:
python src/start_model_server.py --backend gemini --model gemini-2.5-flash

# Or serve Hugging Face SmolVLA locally:
python src/start_model_server.py --backend smolvla
```

**Step 2: Run Prompts in MuJoCo (CLI connects automatically):**
```bash
python src/go2_champ_llm_cli.py --prompt "Walk forward at 0.35 m/s for 3 seconds"
python src/go2_champ_llm_cli.py --interactive
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
python src/go2_champ_llm_cli.py --omniroute --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

#### B. Local Ollama (DeepSeek-R1 / Qwen):
```bash
# Default DeepSeek-R1:8b:
python src/go2_champ_llm_cli.py --ollama --prompt "Walk forward at 0.35 m/s for 3 seconds"

# Custom local model:
python src/go2_champ_llm_cli.py --ollama --model qwen3-vl:latest --prompt "Turn left for 2 seconds"
```

#### C. Google AI Studio (Gemini / Gemma):
Add your API key to `.env`:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
Then run:
```bash
# Using default Gemini (gemini-2.5-flash):
python src/go2_champ_llm_cli.py --gemini --prompt "Walk forward at 0.35 m/s for 3 seconds"

# Using Gemma models (e.g. gemma-2-27b-it):
python src/go2_champ_llm_cli.py --gemini --model gemma-2-27b-it --prompt "Turn left for 2 seconds"
```

#### D. Headless Run without GUI:
```bash
python src/go2_champ_llm_cli.py --no-gui --prompt "Turn left for 3 seconds"
```

#### E. Interactive REPL:
```bash
python src/go2_champ_llm_cli.py --interactive
```

#### F. Offline Mock Client — **not available from the CLI**:
The CLI has no offline-mock flag. Its `--backend` choices are
`server | omniroute | ollama | gemini | smolvla`, and if no live backend answers, the CLI raises
`"Offline mock client fallback has been strictly disabled."` rather than substituting a mock.
The mock client class still exists in the library and is still reachable through the model
server:
```bash
python src/start_model_server.py --backend mock
```

---

## Isaac Lab + Newton Migration

The reinforcement-learning stack is being migrated from the mujoco-warp GPU backend
(`rl/gpu/`, `Go2WarpEnv` + `GPURunner`) to **Isaac Lab 3.x** with the **Newton** physics
engine (`rl/isaac_lab/`, `ManagerBasedRLEnv` + RSL-RL `OnPolicyRunner`). The migration
preserves the 45-dim observation / 12-dim action contract in `rl/contracts.py`, the
reward-term weights, the deterministic velocity curriculum, and the domain-randomization
schedule.

Full details, architecture mapping, and the migration checklist live in
[`docs/ISAAC_LAB_MIGRATION_GUIDE.md`](docs/ISAAC_LAB_MIGRATION_GUIDE.md). Module overview:
[`rl/isaac_lab/README.md`](rl/isaac_lab/README.md).

### Prerequisites

```bash
pip install isaaclab isaaclab-rl isaaclab-newton --extra-index-url https://pypi.nvidia.com
```

### Smoke tests (quick validation, no training)

```bash
# All 4 stages, 100 steps each — verifies env setup, physics, obs/rewards/terminations
python -m rl.isaac_lab.scripts.smoke_test

# Single stage, CPU, tiny env count
python -m rl.isaac_lab.scripts.smoke_test --stage walk --steps 200 --num-envs 4 --device cpu
```

### Training each stage

Entry point `python -m rl.train_isaac` (supports `--stage all` and multi-seed). A second,
narrower entry point `python -m rl.isaac_lab.scripts.train_isaac` handles a single stage
with a single seed only.

```bash
# Stand / Balance / Transition: 5M steps each
python -m rl.train_isaac --stage stand      --steps 5000000  --seed 0 --num-envs 4096 --device cuda
python -m rl.train_isaac --stage balance    --steps 5000000  --seed 0 --num-envs 4096 --device cuda
python -m rl.train_isaac --stage transition --steps 5000000  --seed 0 --num-envs 4096 --device cuda

# Walk: 20M steps for full training
python -m rl.train_isaac --stage walk --steps 20000000 --seed 0 --num-envs 4096 --device cuda

# All stages sequentially, 5M steps each
python -m rl.train_isaac --stage all --steps 5000000 --seed 0 --num-envs 4096 --device cuda
```

### Multi-seed training

```bash
# 3 seeds for the walk stage
python -m rl.train_isaac --stage walk --steps 5000000 --num-seeds 3 --seeds 0 1 2 --device cuda

# All stages, 3 seeds each
python -m rl.train_isaac --stage all --steps 5000000 --num-seeds 3 --device cuda
```

On `--device cuda` seeds run sequentially (`max_parallel=1`); on CPU they run in parallel.
Results are aggregated as mean ± std.

### Where configs live

| Config | Location |
| :--- | :--- |
| Stage definitions (reward weights, curriculum flags, spawn height) | `rl/isaac_lab/configs.py` |
| Isaac Lab `@configclass` env configs (`Go2StandEnvCfg` … `Go2WalkEnvCfg`) | `rl/isaac_lab/envs/go2_env_cfg.py` |
| YAML stage configs (`stand.yaml`, `walk.yaml`, `smoke.yaml`) | `rl/isaac_lab/config/` |
| Reward terms (MDP) | `rl/isaac_lab/mdp/`, `rl/isaac_lab/rewards_direct.py` |
| Domain randomization (EventManager terms) | `rl/isaac_lab/events.py` |
| Curriculum (deterministic velocity schedule) | `rl/isaac_lab/curriculum/velocity.py` |
| Runner (RSL-RL PPO, `entropy_coef=0.0`) | `rl/isaac_lab/runner.py` |

### Checkpoints, evaluation, parity

```bash
# Evaluate a trained checkpoint (45-dim obs -> 12-dim action)
python -c "
from rl.isaac_lab.eval_adapter import load_isaac_policy
policy = load_isaac_policy('logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt', stage='walk')
"

# Visualize with the Isaac Lab viewer (add --headless for no GUI)
python -m rl.isaac_lab.visualize --checkpoint logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt --stage walk

# Convert checkpoints between RSL-RL .pt and SB3 .zip
python -c "
from rl.isaac_lab.checkpoint_bridge import convert_rsl_to_sb3
convert_rsl_to_sb3('logs/isaac_lab/walk/seed0/checkpoints/model_5000000.pt', 'walk_sb3.zip')
"

# mujoco-warp vs Isaac Lab parity verification (100k steps, both stacks)
python -m rl.isaac_lab.scripts.verify_parity
```

---

## Benchmark Evaluation Results

The evaluation harness ships **9** reproducible scenarios under a fixed initial state ($z_0 = 0.28\,\text{m}$, nominal standing configuration). Five are command-driven (`stand_hold`, `stop_and_hold`, `turn_in_place`, `forward_speed_step`, `s_curve`), one is waypoint-driven (`waypoint_square`), and two follow a geometric reference path (`circle_orbit`, `figure_eight`). **Only those last two populate `rms_tracking_error_m`** — for the other seven the field is `null`, not `0.0`, because they issue velocity or waypoint commands rather than following a geometric reference path.

Values below are from the **run of 2026-10-21** (`logs/hipfix/benchmark_suite_20261021_002624/`), mirrored in `report/figures/benchmark_results.json`. It is the first suite run executed on the current code **and** the first executed after the hip-geometry fix in `src/go2_champ_control.py` (see [`docs/REMAINING_CHANGES.md`](docs/REMAINING_CHANGES.md) §1.6). Two independent changes invalidated every table published before it: commit `bbfbb14` (2026-10-17) changed the MuJoCo floor friction (2.5 → 0.6) and the phase-generator wrapping **after** the 2026-10-16 run (`results/benchmark_suite_20261016_011942/`), and the hip-geometry defect inflated the commanded yaw term by 2.05× until it was fixed on 2026-10-21. **Re-run the suite before quoting any of these numbers.**

**Cost of transport is reported on two conventions.** *Gross* is $\sum|\tau\dot q| / mgd$ — every joule a joint absorbs is charged, no regeneration credited. *Net* is $\sum\tau\dot q / mgd$ — regeneration is credited, and this is the convention comparable to the MIT Cheetah / RL published baselines. On this run **45 % – 69 %** of the gross mechanical work is recovered in place on every moving scenario (`obstacle_avoidance` 44.6 %, `turn_in_place` 69.2 %), so the gross figure overstates the net figure by **+81 % to +224 %** — a *scenario-dependent* error, not a constant one. It also inflates the spread between scenarios by 1.8× (gross range 1.16 – 2.80 against a net range of 0.53 – 1.46) and even reorders them (`turn_in_place` is the *worst* of the eight moving scenarios in gross, 2.80, but only 6th in net, 0.86; `obstacle_avoidance` is 7th in gross, 2.63, and the *worst* in net, 1.46). **Quote the net figure.** The older `docs/cot-analysis.md` numbers (net CoT 0.85 – 0.92) are stale.

| Scenario | Status | Forward Dist (m) | Lateral Dist (m) | Path Length (m) | $v_x$ Tracking RMSE (m/s) | Yaw Tracking RMSE (rad/s) | Mean Base Height (m) | Peak Roll | Peak Pitch | CoT gross | CoT net | Ref-path max / RMS (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stand_hold` | ✅ **PASS** | -0.002 | 0.000 | 0.007 | 0.008 | 0.000 | 0.290 | 0.00° | 0.41° | 0.31 | -0.14 ⚠ | null |
| `stop_and_hold` | ✅ **PASS** | 0.640 | 0.044 | 0.756 | 0.088 | 0.667 | 0.283 | 3.71° | 4.76° | 2.45 | 1.331 | null |
| `turn_in_place` | ✅ **PASS** | -0.250 | -0.168 | 0.419 | 0.061 | 0.515 | 0.279 | 2.54° | 3.18° | 2.80 | 0.864 | null |
| `forward_speed_step` | ✅ **PASS** | 1.984 | 0.086 | 2.068 | 0.063 | 0.607 | 0.280 | 2.71° | 4.92° | 1.27 | 0.588 | null |
| `s_curve` | ✅ **PASS** | 1.414 | 1.030 | 1.918 | 0.053 | 0.588 | 0.280 | 2.80° | 4.68° | 1.32 | 0.609 | null |
| `waypoint_square` | ✅ **PASS** | 0.037 | 0.133 | 4.178 | 0.060 | 0.649 | 0.279 | 3.38° | 5.10° | 1.57 | 0.642 | null |
| `obstacle_avoidance` | ✅ **PASS** | 1.220 | 0.024 | 1.432 | 0.107 | 0.822 | 0.282 | 3.93° | 4.74° | 2.63 | 1.456 | null |
| `circle_orbit` | ✅ **PASS** | -2.888 | -0.550 | 5.322 | 0.050 | 0.760 | 0.278 | 4.67° | 4.11° | 1.16 | 0.528 | 0.0889 / 0.0329 |
| `figure_eight` | ✅ **PASS** | 0.340 | 0.158 | 6.087 | 0.054 | 0.761 | 0.277 | 2.79° | 5.26° | 1.32 | 0.588 | 0.2274 / 0.0756 |

All nine scenarios PASS their own declared gate: zero FAILs, zero controller faults, zero non-finite telemetry values. Five caveats that a reader must not get wrong:

- ⚠ **`stand_hold`'s CoT is an artefact, not a measurement.** Its path length (0.0067 m) is below the harness's 0.01 m denominator floor, so the denominator is the floor rather than real travel: the net CoT is negative (-0.14) and the recovery fraction is 144.7 %. **Exclude it from any CoT comparison.**
- **`obstacle_avoidance` exercised no avoidance.** The run was executed without `--vision`, and the MuJoCo world contains exactly one non-robot geom — the `floor` plane — i.e. **there is no obstacle in the scene at all**. What was measured is a plain 0.35 m/s forward trot bracketed by two stand holds.
- **`circle_orbit`'s −2.89 m net displacement is geometry, not error.** The reference circle has $\omega = v/r = 0.35/1.5 = 0.2333$ rad/s, so 15 s covers only 200.5° of the circle; the robot ends *on* the reference path.
- **Yaw-rate tracking is better in every scenario that commands a non-zero yaw rate** than in the table this README was showing before this run (`stand_hold` is unchanged at $2.9\times10^{-7}\,\text{rad/s}$, i.e. effectively zero either way): `circle_orbit` 0.795 → 0.760, `figure_eight` 0.801 → 0.761, `obstacle_avoidance` 0.861 → 0.822, `stop_and_hold` 0.715 → 0.667, `waypoint_square` 0.680 → 0.649, `forward_speed_step` 0.637 → 0.607, `s_curve` 0.630 → 0.588, `turn_in_place` 0.527 → 0.515 rad/s. **The published numbers were wrong for two independent reasons**: (a) the 2026-10-16 table was recorded *before* commit `bbfbb14` (2026-10-17) changed the MuJoCo floor friction (2.5 → 0.6) and the phase-generator wrapping, and (b) *every* table published before this run was recorded with the hip-geometry defect still in the control path, which inflated the commanded yaw term by 2.05×.
- **The yaw tracker did not get better *over time* — the published baseline was wrong.** Against the original 2026-10-16 table the yaw RMSE is still larger in all eight scenarios that command a non-zero yaw rate (`circle_orbit` 0.604 → 0.760, `obstacle_avoidance` 0.405 → 0.822), and peak roll is a fraction of a degree worse than in the table published before this run in 7 of the 9 scenarios (worst roll 4.62° → 4.67°), even though it is far better than the 2026-10-16 table (worst roll 8.04°, worst pitch 11.33° → now 4.67° / 5.26°). **Yaw-rate tracking remains the weakest capability in this suite** (worst case 0.822 rad/s, `obstacle_avoidance`).

**`contact_duty_factor` is a geometric foot-contact fraction, not the gait's commanded duty factor.** The commanded duty factor is exactly 0.50 (`GaitConfig(stance_duration=0.22, swing_duration=0.22)`, stride period 0.44 s). In the two continuously-trotting scenarios the measured geometric fraction is 56.6 % (`circle_orbit`) and 56.5 % (`figure_eight`), with the rear legs 3.9 – 8.5 points above the front; in the command-driven scenarios it is higher (63.9 % – 75.1 %) because the robot stands for part of the run. A duty-factor sweep — recorded on the **pre-fix** code, so its duty-0.50 baseline (net CoT 0.689) is the pre-fix `s_curve` figure and not the 0.609 in the table above — confirms the shipped 0.50 is already optimal among the swept values: raising it to 0.70 at the fixed 0.44 s stride period **increases** net CoT from 0.689 to 1.082 (+57 %), because a shorter swing at a fixed stride period means a shorter stride, so the same mechanical work is spread over less distance.

> **Scope of these numbers (important).** The contact QP is *not* in the loop these benchmarks exercise — `_step_impl` returns joint-angle targets into MuJoCo position servos, and `data.ctrl` holds an angle, not a torque vector. Consequently no benchmark run reports QP forces, and there is no "friction pyramid compliance" result from this repository today. Every CoT figure above describes MuJoCo's built-in position servo, not the SRBD solver — and it is computed from **clipped** actuator torques, so it is a lower bound on the mechanical work actually required. The contact-force, estimator and torque-actuator work needed to change that is Tier 1 in [`results/AUDIT_WORKLIST.md`](results/AUDIT_WORKLIST.md).

### Acceptance Criteria
Each scenario carries its own `expected_metrics` block in `experiments/scenarios/*.json`, and **that** is what `validate_scenario` enforces — there is no global attitude limit. The gates actually declared are:

| Scenario | max roll | max pitch | min base height | other gates |
| :--- | :---: | :---: | :---: | :--- |
| `stand_hold` | 2.9° | 2.9° | 0.25 m | max forward displacement 0.05 m |
| `forward_speed_step` | 8.6° | 8.6° | 0.24 m | min forward displacement 1.4 m |
| `stop_and_hold` | 8.6° | 8.6° | 0.24 m | — |
| `turn_in_place` | 8.6° | 8.6° | 0.24 m | max forward displacement 0.35 m |
| `s_curve` | 10.3° | 10.3° | 0.24 m | min forward displacement 1.0 m, min path length 0.9 m |
| `obstacle_avoidance` | 10.3° | 10.3° | 0.24 m | min forward displacement 0.5 m |
| `waypoint_square` | 10.3° | 10.3° | 0.24 m | min path length 2.0 m |
| `circle_orbit` | 11.5° | 11.5° | 0.24 m | min path length 2.6 m, max tracking error 0.5 m |
| `figure_eight` | 11.5° | 11.5° | 0.24 m | min path length 3.0 m, max tracking error 0.4 m |

Two consequences worth knowing before quoting a number:

- **Peak attitude is scenario-relative, not global.** `circle_orbit` reaches 4.67° roll / 4.11° pitch and is correctly marked PASS, because its own gate is 11.5°. Any claim of the form "peak roll < 4.0° across all maneuvers" is false in this run as well as ungated — the worst measured roll is 4.67° — and no scenario file declares such a threshold.
- **A PASS here means "met its own declared gate"**, not "met the paper's target". The paper's ≤2.5° fall-rate / ≤35 N·s push targets are Phase 3 work and are not measured by these runs.

Other criteria that do hold across all runs:
- **Stability**: zero falls; base height stays above the per-scenario floor.
- **Finite State**: $100\%$ finite numbers across joint angles, velocities and actuator forces (zero `NaN` / `Inf`) — enforced by the validator.
- **Telemetry Fidelity**: full 250 Hz state logging with command tracking errors and energy expenditure.

---

## Safety Guardrail Specifications

```python
@dataclass(frozen=True)
class SafetyBounds:
    max_vx: float = 0.60            # Maximum forward velocity (m/s)
    min_vx: float = -0.30           # Maximum reverse velocity (m/s)
    max_vy: float = 0.35            # Maximum lateral velocity (m/s)
    max_yaw_rate: float = 0.80      # Maximum heading rate (rad/s)
    max_duration: float = 60.0      # Maximum execution duration per command (s)
    max_linear_accel: float = 1.20  # Slew rate limit on linear velocity (m/s^2)
    max_yaw_accel: float = 2.50     # Slew rate limit on yaw rate (rad/s^2)
    max_roll_rad: float = 0.40      # Maximum allowable body roll (~22.9 deg)
    max_pitch_rad: float = 0.40     # Maximum allowable body pitch (~22.9 deg)
    min_height_m: float = 0.16      # Minimum torso ground clearance (m)
    max_height_m: float = 0.38      # Maximum torso ground clearance (m)
    max_stale_steps: int = 5        # Consecutive frozen step_time values tolerated
```

(Verbatim from `src/go2_champ_control.py`. The roll/pitch/height fields are **controller fault
thresholds**, not acceptance thresholds — a crossing raises a fault that fails the run, but no
scenario declares an attitude acceptance limit of its own beyond the ladder above.)

1. **Strict Fail-Closed Architecture**: Non-JSON strings, missing keys, or unparseable text automatically default to:
   ```json
   {"mode": "stand", "duration": 1.0, "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
   ```
   A non-finite JSON constant (`NaN`, `Infinity`) now makes the **whole response** inadmissible at parse time, and an oversized integer literal (a 400-digit number) can no longer make the validator raise instead of failing closed.
2. **Deterministic Sequence Conclusion**: Every dynamic movement sequence generated by the LLM concludes with an automatic 1.0 s stationary stand command to guarantee that the robot halts safely. The test is on the command *mode*, so a pure waypoint plan is covered too.
3. **Structured JSON Validation**: DeepSeek-R1 receives Ollama's `format` parameter enforcing schema compliance at the sampling level, eliminating free-form conversational chatter. This is an *efficiency* mechanism, not a safety control — all magnitude enforcement is local.
4. **Verified under attack**: an adversarial corpus of 64 malformed, out-of-bounds and adversarial responses, each checked against 10 safety invariants, reports **0 invariant violations** (10 before the 2026-10-20 contract audit) with the four legitimate instruction cases still passing. See `report/notes/09_contract_audit.md` and `report/figures/adversarial_results.json`.

---

## References

1. **CHAMP Quadruped Controller**: [chvmp/champ](https://github.com/chvmp/champ) - Open-source ROS-based locomotion framework for quadrupedal robots.
2. **MIT Cheetah Locomotion**: Bledt, G., et al., *"Hierarchical controller for highly dynamic locomotion utilizing pattern modulation and impedance control: implementation on the MIT Cheetah robot"*, 2018.
3. **Unitree Go2 MuJoCo Description**: Unitree Robotics official Go2 model descriptions and URDF assets.
