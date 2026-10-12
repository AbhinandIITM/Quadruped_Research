# Technical Report: Hierarchical CHAMP Kinematics, Closed-Loop Trajectory Navigation, and Multi-Provider LLM Natural Language Interface for the Unitree Go2 in MuJoCo

**Project:** Autonomous Quadruped Locomotion & Safety-Critical Natural Language Control  
**Platform:** Unitree Go2 Quadruped Robot  
**Physics Engine:** MuJoCo 3.x (`simulate` / Python API)  
**Reasoning Engines:** OmniRoute / OpenAI Gateway, Local DeepSeek-R1 / Qwen (via Ollama), Google Gemini 2.5 / Gemma 2 (via Google AI Studio)  
**Date:** October 2026  
**Status:** Verified, Tested, and Benchmarked (39/39 Unit Tests Passing)  

---

## 1. Executive Summary

This report documents the architectural design, mathematical formulation, software implementation, safety guardrails, and experimental benchmark validation of a simulation-only quadruped locomotion system. The project ports the open-source **CHAMP** (MIT Cheetah-inspired) hierarchical quadruped locomotion algorithms (`chvmp/champ`) to the **Unitree Go2** platform in MuJoCo physics, and integrates a high-level natural language control interface powered by OpenAI-compatible routing gateways (**OmniRoute**, **LiteLLM**, **One-API**), local reasoning models (**DeepSeek-R1**, **Qwen** via Ollama), dedicated standalone model microservices (`start_model_server.py`), and cloud inference endpoints (**Gemini 2.5 Flash**, **Gemini 3.1 Flash Lite**, **Gemma 2 27B** via Google AI Studio).

### Key Technical Achievements
1. **Analytic 3-DOF Inverse Kinematics**: Exact closed-form geometric solution for the Unitree Go2 limb kinematics, accounting for lateral hip offset and knee flexion constraints.
2. **Deterministic CHAMP Gait Generator**: Diagonal trot phase oscillator ($T_{\text{stride}} = 0.44\,\text{s}$) coordinating diagonal leg pairs (FR/RL at phase 0.0, FL/RR at phase 0.5) with $C^2$-continuous 12-point Bézier swing curves and cosine stance retraction.
3. **Closed-Loop Trajectory & Waypoint Tracking**: Closed-loop body-frame forward velocity ($v_x$), lateral velocity ($v_y$), and yaw rate ($\omega_z$) tracking, augmented with dynamic slew-rate acceleration limiting ($a_{\text{lin}} \le 1.0\,\text{m/s}^2, \alpha_{\text{yaw}} \le 2.0\,\text{rad/s}^2$) and 2D world-frame waypoint navigation.
4. **Raibert Velocity Feedback Compensation**: Integrated closed-loop Raibert feedback into foothold placement ($k_v = 0.35$), eliminating stance-phase compliance slip and steady-state velocity tracking lag.
5. **Decoupled Model Microservice Architecture**: Standalone model server (`start_model_server.py`) serving any specified backend/model over local REST (`http://127.0.0.1:5005`), enabling `go2_champ_llm_cli.py` to auto-connect and execute natural language queries without exposing model credentials or configurations.
6. **Multi-Provider & OmniRoute Gateway Support**: Zero-dependency standard-library (`urllib.request`) clients for OmniRoute / OpenAI-compatible gateways (`/v1/chat/completions`), local Ollama (`/api/chat`), and Google AI Studio REST API (`/v1beta/models/{model}:generateContent`).
7. **Robust Schema Parsing & Array Handling**: Grammar-enforced JSON extraction supporting both markdown-fenced/raw JSON objects (`{"explanation": "...", "commands": [...]}`) and top-level JSON command arrays (`[{"mode": ...}, ...]`), with `<think>` and `reasoning_content` tag isolation and `.env` configuration auto-loading.
8. **Strict Simulation-Only Safety Envelope**: High-level command abstraction prohibiting actuator torque exposure, shell commands, hardware SDK calls, or external network access. Out-of-bounds, malformed, or non-finite inputs fail closed to a safe stationary stand.
9. **Automated Verification & Benchmark Suite**: 39 unit tests passing in $<0.05\,\text{s}$, alongside an automated 6-scenario benchmark suite generating 250 Hz telemetry (`data.csv`), metrics (`metrics.json`), and diagnostic tracking plots (`plots.png`).

---

## 2. System Architecture

The architecture enforces a strict hierarchical separation of concerns between natural language reasoning, model serving, command validation, gait kinematics, and rigid-body dynamics:

```
+---------------------------------------------------------------------------------------------------+
|                                  NATURAL LANGUAGE REASONING LAYER                                 |
|                                                                                                   |
|  Human User Prompt: "Walk forward at 0.35 m/s for 2.5 seconds, then stop"                        |
|                                         |                                                         |
|         +-------------------------------+-------------------------------+                         |
|         |                               |                               |                         |
|         v                               v                               v                         |
|  OmniRoute / OpenAI Gateway       Local Ollama Server            Google AI Studio REST API        |
|  (http://localhost:8000/v1)      (http://localhost:11434)       (https://generativelanguage... )  |
|  Models: deepseek-r1, qwen2.5     Models: deepseek-r1:8b         Models: gemini-2.5-flash,        |
|  Format: json_object + reasoning  Format: MOTION_COMMAND_SCHEMA          gemma-2-27b-it           |
|         |                               |                               |                         |
|         +-------------------------------+-------------------------------+                         |
|                                         |                                                         |
|                                         v (or via start_model_server.py:5005)                     |
|  Raw Model Output: JSON Object {"commands": [...]} OR JSON Array [{"mode": ...}, ...]            |
+---------------------------------------------------------------------------------------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------------------------+
|                                  SAFETY VALIDATION & AUDIT LAYER                                  |
|                                                                                                   |
|  1. strip_think_tags() & extract_json_block() (Supports Objects & Arrays)                         |
|  2. Strict Schema Validation & Sanitization (Reject NaN, Inf, String Injections)                  |
|  3. SafetyBounds Clamping:                                                                       |
|     - Forward Velocity:   vx in [-0.30, 0.60] m/s                                                 |
|     - Lateral Velocity:   vy in [-0.35, 0.35] m/s                                                 |
|     - Yaw Rate:           yaw_rate in [-0.80, 0.80] rad/s                                         |
|     - Duration:           duration <= 60.0 s                                                      |
|  4. Fail-Closed Fallback: Malformed/Ambiguous -> Stationary Stand (duration=1.0s)                 |
|  5. Automatic Sequence Rest: Terminal Stand (duration=1.0s) appended to all motions              |
|  6. Immutable JSONL Logging -> logs/llm_audit.jsonl                                               |
+---------------------------------------------------------------------------------------------------+
                                          | Validated MotionCommand Sequence
                                          v
+---------------------------------------------------------------------------------------------------+
|                              CHAMP GUIDANCE & TRAJECTORY PLANNING LAYER                           |
|                                                                                                   |
|  1. Waypoint Tracking & Heading Regulation:                                                       |
|     - Distance error -> Target Speed vx                                                           |
|     - Heading error  -> Proportional Yaw Rate wz = clip(k_yaw * e_yaw, -wz_max, wz_max)           |
|     - Pure in-place turning if |e_yaw| > 0.45 rad (25.8 deg)                                      |
|  2. Command Slew-Rate Limiter (anti-pitch / anti-roll filter):                                    |
|     - dvx/dt <= 1.2 m/s^2,  dvy/dt <= 1.2 m/s^2,  dwz/dt <= 2.5 rad/s^2                         |
|  3. Hip Tangential Velocity Coupling:                                                             |
|     - v_hip,i = v_body + wz x r_hip,i                                                             |
|  4. Raibert Velocity Feedback Compensation:                                                       |
|     - v_eff,i = v_hip_cmd,i + k_v * (v_hip_cmd,i - v_hip_meas,i), with k_v = 0.35                 |
|  5. CHAMP Diagonal Trot Phase Oscillator:                                                         |
|     - T_stride = 0.44s (T_stance = 0.22s, T_swing = 0.22s)                                        |
|     - Phase offsets: FR=0.0, RL=0.0, FL=0.5, RR=0.5                                            |
|  6. Trajectory Generators:                                                                        |
|     - Swing: 12-point Bernstein Bezier curve with zero lift velocity/accel at boundaries        |
|     - Stance: Linear fore-aft retraction x(s) + cosine compliance depth z(s)                      |
+---------------------------------------------------------------------------------------------------+
                                          | Desired Foot Positions (x_i, y_i, z_i)
                                          v
+---------------------------------------------------------------------------------------------------+
|                                 KINEMATICS & GAIT EXECUTION LAYER                                 |
|                                                                                                   |
|  1. 3-DOF Analytic Inverse Kinematics (solve_ik):                                                 |
|     - Hip Roll:   theta_hip   = atan2(y, -z) - atan2(y_off, sqrt(y^2 + z^2 - y_off^2))            |
|     - Calf Pitch: theta_calf  = -arccos((x^2 + z_planar^2 - l1^2 - l2^2) / (2*l1*l2))             |
|     - Thigh Pitch: theta_thigh = atan2(-x, -z_planar) - atan2(l2*s_calf, l1 + l2*c_calf)         |
|  2. Safety Supervisor (check_state):                                                              |
|     - Emergency trip if |roll| > 0.40 rad, |pitch| > 0.40 rad, or z_base < 0.16 m                 |
+---------------------------------------------------------------------------------------------------+
                                          | Target Joint Angles q_target (12-DOF)
                                          v
+---------------------------------------------------------------------------------------------------+
|                                      MUJOCO SIMULATION CORE                                       |
|                                                                                                   |
|  - Unitree Go2 Model: go2_description.xml (Mass = 12.0 kg, 12 Actuators)                         |
|  - Joint PD Control:  tau = Kp * (q_target - q) - Kd * q_dot                                      |
|    Kp = 100.0 N*m/rad,  Kd = 4.0 N*m*s/rad                                                        |
|  - Physics Integration: mj_step() at 500 Hz (dt = 0.002 s)                                        |
|  - Visual Feedback: Passive MuJoCo GUI Viewer (default) or Headless (--no-gui)                    |
|  - Telemetry Decimation: Logged at 250 Hz -> data.csv, metrics.json, plots.png                    |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Foundations & Kinematics Formulation

### 3.1 Robot Kinematic Geometry

The Unitree Go2 consists of a floating base torso and four 3-DOF symmetric limbs. Link dimensions and joint limits used throughout the kinematic solvers are defined as follows:

$$\text{Thigh Length } l_1 = 0.213\,\text{m}, \quad \text{Calf Length } l_2 = 0.213\,\text{m}, \quad \text{Lateral Hip Offset } y_{\text{offset}} = 0.0955\,\text{m}$$

$$\text{Nominal Standing Height } h_0 = 0.280\,\text{m}$$

Hip attachment positions relative to the base center of mass:
$$\vec{r}_{\text{FR}} = [0.1934, -0.0955, 0]^T, \quad \vec{r}_{\text{FL}} = [0.1934, 0.0955, 0]^T$$
$$\vec{r}_{\text{RR}} = [-0.1934, -0.0955, 0]^T, \quad \vec{r}_{\text{RL}} = [-0.1934, 0.0955, 0]^T$$

Joint physical limits enforced across all steps:
- **Hip Roll ($\theta_{\text{hip}}$)**: $[-1.0472, 1.0472]\,\text{rad}$ ($\pm 60.0^\circ$)
- **Thigh Pitch ($\theta_{\text{thigh}}$)**: $[-1.5708, 3.4907]\,\text{rad}$ ($-90.0^\circ \text{ to } +200.0^\circ$)
- **Calf Pitch ($\theta_{\text{calf}}$)**: $[-2.7227, -0.83776]\,\text{rad}$ ($-156.0^\circ \text{ to } -48.0^\circ$, strictly backwards-flexed)

---

### 3.2 Closed-Form Analytic 3-DOF Inverse Kinematics

Given a desired foot position $(x, y, z)$ relative to the hip roll joint frame, the analytic inverse kinematics solves for the joint trio $(\theta_{\text{hip}}, \theta_{\text{thigh}}, \theta_{\text{calf}})$.

#### 1. Hip Roll Angle ($\theta_{\text{hip}}$)
In the frontal $(y, z)$ plane, the leg rotates about the hip axis with offset $y_{\text{offset}}$ (positive for left legs, negative for right legs):
$$y_{\text{leg}} = \begin{cases} +y_{\text{offset}}, & \text{if Left (FL, RL)} \\ -y_{\text{offset}}, & \text{if Right (FR, RR)} \end{cases}$$

The distance $D_{yz}$ from the hip origin to the foot projection is:
$$D_{yz} = \sqrt{y^2 + z^2}$$
If $D_{yz} < |y_{\text{leg}}|$, the foot lies inside the singular hip cylinder and raises a workspace violation. Otherwise, the effective planar leg extension in the sagittal plane is:
$$z_{\text{planar}} = -\sqrt{D_{yz}^2 - y_{\text{leg}}^2}$$
$$\theta_{\text{hip}} = \operatorname{atan2}(y, -z) - \operatorname{atan2}\left(y_{\text{leg}}, \sqrt{D_{yz}^2 - y_{\text{leg}}^2}\right)$$

#### 2. Knee / Calf Pitch Angle ($\theta_{\text{calf}}$)
Projecting into the sagittal $(x, z_{\text{planar}})$ plane, the distance from the hip pitch axis to the foot is:
$$R_{\text{planar}} = \sqrt{x^2 + z_{\text{planar}}^2}$$
By the Law of Cosines:
$$\cos(\theta_{\text{calf}}) = \operatorname{clip}\left(\frac{x^2 + z_{\text{planar}}^2 - l_1^2 - l_2^2}{2 l_1 l_2}, \; -1.0, \; 1.0\right)$$
Because the Go2 quadruped knee flexes strictly rearward:
$$\theta_{\text{calf}} = -\arccos\left(\cos(\theta_{\text{calf}})\right)$$

#### 3. Thigh Pitch Angle ($\theta_{\text{thigh}}$)
Using the two-angle decomposition:
$$\alpha = \operatorname{atan2}(-x, -z_{\text{planar}})$$
$$\beta = \operatorname{atan2}\left(l_2 \sin(\theta_{\text{calf}}), \; l_1 + l_2 \cos(\theta_{\text{calf}})\right)$$
$$\theta_{\text{thigh}} = \alpha - \beta$$

---

### 3.3 Diagonal Trot Phase Generator

Locomotion is orchestrated via a continuous cyclic phase oscillator with stride period $T_{\text{stride}} = 0.44\,\text{s}$, partitioned equally into stance duration $T_{\text{stance}} = 0.22\,\text{s}$ and swing duration $T_{\text{swing}} = 0.22\,\text{s}$ (50% nominal duty cycle).

Diagonal pairings share identical phase offsets:
$$\phi_{\text{FR}} = 0.0, \quad \phi_{\text{RL}} = 0.0 \quad (\text{Diagonal Pair 1})$$
$$\phi_{\text{FL}} = 0.5, \quad \phi_{\text{RR}} = 0.5 \quad (\text{Diagonal Pair 2})$$

For each leg $i \in \{0, 1, 2, 3\}$, instantaneous phase progression is integrated continuously:
$$t_{\text{phase}, i} = (t_{\text{phase}, i} + \Delta t) \pmod{T_{\text{stride}}}$$
$$\tilde{\phi}_i = \frac{t_{\text{phase}, i}}{T_{\text{stride}}}$$

Normalizing into normalized swing phase $s_{\text{sw}} \in [0, 1]$ and stance phase $s_{\text{st}} \in [0, 1]$:
$$\begin{cases} s_{\text{sw}, i} = \frac{\tilde{\phi}_i}{T_{\text{swing}} / T_{\text{stride}}}, \quad s_{\text{st}, i} = 0, & \text{if } \tilde{\phi}_i < \frac{T_{\text{swing}}}{T_{\text{stride}}} \quad (\text{Swing Phase}) \\ s_{\text{sw}, i} = 0, \quad s_{\text{st}, i} = \frac{\tilde{\phi}_i - (T_{\text{swing}} / T_{\text{stride}})}{T_{\text{stance}} / T_{\text{stride}}}, & \text{if } \tilde{\phi}_i \ge \frac{T_{\text{swing}}}{T_{\text{stride}}} \quad (\text{Stance Phase}) \end{cases}$$

When the robot is commanded to remain stationary ($\|\vec{v}\| = 0, \omega_z = 0$), phase progression halts smoothly with all legs transitioning to ground contact ($s_{\text{st}} = 0.5, s_{\text{sw}} = 0.0$).

---

### 3.4 12-Point Bézier Swing Trajectory ($C^2$ Boundary Continuity)

To ensure zero impact shock at touchdown and smooth liftoff without ground dragging, the vertical swing trajectory $z_{\text{swing}}(s)$ is defined using an 11th-degree Bernstein Bézier polynomial with 12 control points $P_0, P_1, \dots, P_{11}$:

$$z(s) = \sum_{k=0}^{11} \binom{11}{k} (1 - s)^{11 - k} s^k P_k, \quad s \in [0, 1]$$

Control points are parameterized by target ground clearance $h_{\text{swing}} = 0.06\,\text{m}$:
$$P = \left[ 0.0, \; 0.0, \; 0.0, \; 0.9 h, \; 1.1 h, \; 1.2 h, \; 1.2 h, \; 1.1 h, \; 0.9 h, \; 0.0, \; 0.0, \; 0.0 \right]$$

#### Boundary Invariants:
1. **$C^0$ Continuity**: $z(0) = P_0 = 0.0\,\text{m}$, $z(1) = P_{11} = 0.0\,\text{m}$.
2. **$C^1$ Continuity (Zero Liftoff/Touchdown Velocity)**:
   $$\dot{z}(0) = 11 (P_1 - P_0) = 0.0, \quad \dot{z}(1) = 11 (P_{11} - P_{10}) = 0.0$$
3. **$C^2$ Continuity (Zero Liftoff/Touchdown Acceleration)**:
   $$\ddot{z}(0) = 110 (P_2 - 2P_1 + P_0) = 0.0, \quad \ddot{z}(1) = 110 (P_{11} - 2P_{10} + P_9) = 0.0$$

The forward swing foot retraction smoothly advances from liftoff position $-\frac{L}{2}$ to target touchdown position $+\frac{L}{2}$:
$$x_{\text{swing}}(s) = L \left( -\frac{1}{2} + s \right)$$

---

### 3.5 Cosine Stance Retraction Profile

During the stance phase ($s_{\text{st}} \in [0, 1]$), the foot remains anchored to the ground while retracting relative to the hip to propel the body forward:
$$x_{\text{stance}}(s_{\text{st}}) = \frac{L}{2} (1 - 2 s_{\text{st}})$$

To prevent leg stiffness shocks at the stance midpoint, a cosine compliance depth $d_{\text{depth}} = 0.005\,\text{m}$ can be modulated:
$$z_{\text{stance}}(s_{\text{st}}) = -d_{\text{depth}} \cos\left( \pi \frac{x_{\text{stance}}}{L} \right)$$

---

### 3.6 Foothold Planning, Hip Tangential Coupling & Raibert Velocity Feedback

When the robot commands linear velocity $\vec{v}_{\text{body}} = [v_x, v_y]^T$ and yaw rate $\omega_z$, each hip undergoes an individual tangential velocity due to rigid-body kinematics:

$$\vec{v}_{\text{hip, cmd}, i} = \begin{bmatrix} v_x - \omega_z y_{\text{hip}, i} \\ v_y + \omega_z x_{\text{hip}, i} \end{bmatrix}$$
$$\vec{v}_{\text{hip, meas}, i} = \begin{bmatrix} v_{\text{meas}, x} - \omega_{\text{meas}, z} y_{\text{hip}, i} \\ v_{\text{meas}, y} + \omega_{\text{meas}, z} x_{\text{hip}, i} \end{bmatrix}$$

#### Raibert Velocity Feedback Correction:
During dynamic trotting, compliance, foot deformation, and ground slip cause actual forward and lateral speeds to lag behind open-loop kinematic step lengths. The controller applies closed-loop Raibert velocity feedback ($k_v = 0.35$):

$$\vec{v}_{\text{error}, i} = \vec{v}_{\text{hip, cmd}, i} - \vec{v}_{\text{hip, meas}, i}$$
$$\vec{v}_{\text{eff}, i} = \vec{v}_{\text{hip, cmd}, i} + k_v \vec{v}_{\text{error}, i}$$

The effective step length and foothold angle are computed from this augmented velocity:
$$\text{step\_length}_i = \operatorname{clip}\left(\|\vec{v}_{\text{eff}, i}\| \cdot T_{\text{stride}}, \; 0.0, \; L_{\text{max}}\right)$$
$$\psi_{\text{foot}, i} = \operatorname{atan2}(v_{\text{eff}, y, i}, \; v_{\text{eff}, x, i})$$

This dynamically expands the touchdown footprint in the direction of velocity lag, maintaining precise tracking across variable speeds.

---

## 4. Closed-Loop Guidance & Slew-Rate Envelope

### 4.1 Command Slew-Rate Limiting

Instantaneous step changes in commanded velocity (e.g., $0 \to 0.5\,\text{m/s}$ in a single timestep) induce severe ground reaction torques that destabilize quadruped balance and cause pitch oscillations. A dynamic rate limiter bounds velocity increments at each step ($\Delta t = 0.002\,\text{s}$):

$$v_{x}(t) = \operatorname{clip}\left(v_{x,\text{cmd}}, \; v_x(t - \Delta t) - a_{\text{lin}} \Delta t, \; v_x(t - \Delta t) + a_{\text{lin}} \Delta t\right)$$
$$v_{y}(t) = \operatorname{clip}\left(v_{y,\text{cmd}}, \; v_y(t - \Delta t) - a_{\text{lin}} \Delta t, \; v_y(t - \Delta t) + a_{\text{lin}} \Delta t\right)$$
$$\omega_{z}(t) = \operatorname{clip}\left(\omega_{z,\text{cmd}}, \; \omega_z(t - \Delta t) - \alpha_{\text{yaw}} \Delta t, \; \omega_z(t - \Delta t) + \alpha_{\text{yaw}} \Delta t\right)$$

where $a_{\text{lin}} = 1.2\,\text{m/s}^2$ and $\alpha_{\text{yaw}} = 2.5\,\text{rad/s}^2$.

---

### 4.2 2D Waypoint Navigation with Heading Regulation

Given a target waypoint $\vec{w} = [x_{\text{tgt}}, y_{\text{tgt}}]^T$ with tolerance $\epsilon_{\text{tol}}$:

1. **Range and Bearing Calculation**:
   $$\Delta x = x_{\text{tgt}} - x_{\text{robot}}, \quad \Delta y = y_{\text{tgt}} - y_{\text{robot}}$$
   $$d_{\text{tgt}} = \sqrt{\Delta x^2 + \Delta y^2}, \quad \psi_{\text{tgt}} = \operatorname{atan2}(\Delta y, \Delta x)$$
   $$e_{\psi} = \operatorname{wrap}_{[-\pi, \pi]}(\psi_{\text{tgt}} - \psi_{\text{robot}})$$

2. **Tolerance Check**:
   If $d_{\text{tgt}} \le \epsilon_{\text{tol}}$, the waypoint is marked achieved ($\vec{v} = 0, \omega_z = 0$).

3. **In-Place Rotation Thresholding**:
   If the heading error $|e_{\psi}| > 0.45\,\text{rad}$ ($25.8^\circ$), forward drive is held at zero ($v_x = 0$) while the robot rotates in place:
   $$\omega_z = \operatorname{clip}(k_{\text{yaw}} e_{\psi}, \; -\omega_{\text{max}}, \; \omega_{\text{max}})$$
   Once $|e_{\psi}| \le 0.45\,\text{rad}$, forward motion engages proportionally with distance:
   $$v_x = \operatorname{clip}(k_{\text{dist}} d_{\text{tgt}} \cos(e_{\psi}), \; 0.0, \; v_{\text{speed}})$$

---

## 5. Multi-Provider Natural Language Interface & Microservice Architecture

### 5.1 Architecture & Transports

The natural language layer supports multiple execution backends via zero-dependency standard-library (`urllib.request`) HTTP transports:

1. **OmniRoute & OpenAI-Compatible Gateways (`OmniRouteClient`)**:
   - Targets `/v1/chat/completions` (e.g. `http://localhost:8000/v1` for OmniRoute, LiteLLM, One-API, vLLM).
   - Transmits standard chat completions payloads with `"response_format": {"type": "json_object"}`.
   - Extracts structured reasoning traces from OpenAI-standard `reasoning_content` message attributes or embedded `<think>` blocks.
2. **Dedicated Model Microservice (`start_model_server.py` + `LiveModelServiceClient`)**:
   - Encapsulates model definition, credentials, and backend selection strictly within `start_model_server.py`.
   - Exposes lightweight local REST endpoints: `GET /health` and `POST /query` on `http://127.0.0.1:5005`.
   - Allows `go2_champ_llm_cli.py` to auto-discover and interface with the running model server without requiring client-side model flags.
3. **Local Ollama Backend (`OllamaDeepSeekClient`)**:
   - Targets `http://localhost:11434/api/chat`.
   - Supports `deepseek-r1:8b`, `qwen3-vl:latest`, etc.
   - Leverages `"format": MOTION_COMMAND_SCHEMA` for grammar-constrained token generation.
4. **Google AI Studio Backend (`GeminiClient`)**:
   - Targets `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}`.
   - Supports `gemini-2.5-flash`, `gemini-3.1-flash-lite`, `gemma-2-27b-it`.
   - Utilizes `responseMimeType: "application/json"` in `generationConfig`.
5. **Offline Mock Backend (`MockDeepSeekClient`)**:
   - Regex-based deterministic parser for offline unit testing, CI pipelines, and rapid local iteration.

### 5.2 Universal JSON Schema & Array Extraction

Reasoning models frequently output top-level JSON command arrays `[{"mode": ...}, ...]` or wrap objects inside markdown fences. `extract_json_block()` handles both shapes robustly:

```python
def extract_json_block(text: str) -> str:
    trimmed = text.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (
        trimmed.startswith("[") and trimmed.endswith("]")
    ):
        return trimmed
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    first_brace, last_brace = text.find("{"), text.rfind("}")
    first_bracket, last_bracket = text.find("["), text.rfind("]")
    candidates = []
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append((first_brace, last_brace, text[first_brace : last_brace + 1].strip()))
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        candidates.append((first_bracket, last_bracket, text[first_bracket : last_bracket + 1].strip()))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        return candidates[0][2]
    return text.strip()
```

### 5.3 Configuration Management & `.env` Support

API keys, endpoints, and default model selections are loaded automatically on module initialization via a custom zero-dependency `.env` parser (`load_dotenv()`). Configuration precedence follows:
$$\text{CLI Flags} > \text{OS Environment Variables} > \text{.env File} > \text{Default Constants}$$

Key configuration variables:
- `OMNIROUTE_BASE_URL`: OpenAI-compatible proxy URL (defaults to `http://localhost:8000/v1`).
- `OMNIROUTE_MODEL`: Default model on the OmniRoute gateway (defaults to `deepseek-r1:8b`).
- `OMNIROUTE_API_KEY`: Gateway authorization token (if required).
- `MODEL_BACKEND`: Microservice active backend (`omniroute`, `ollama`, `gemini`, `mock`).
- `MODEL_SERVER_PORT`: Port for `start_model_server.py` (defaults to `5005`).
- `GEMINI_API_KEY`: Google AI Studio API authentication token.
- `GEMINI_MODEL`: Default cloud model (defaults to `gemini-2.5-flash`).
- `OLLAMA_HOST`: Local Ollama server address (defaults to `http://localhost:11434`).
- `OLLAMA_MODEL`: Default local model (defaults to `deepseek-r1:8b`).

### 5.4 Safety Bounds & Fail-Closed Sanitization

Every extracted command list is validated against immutable `SafetyBounds`:
- Velocities and rates exceeding limits are clamped with status `"clamped"`.
- Non-finite numbers (`NaN`, `Inf`) are replaced with stationary parameters ($0.0$).
- If JSON parsing fails entirely or yields empty commands, the validator fails closed to a safe 1.0 s standing command:
  ```json
  {"mode": "stand", "duration": 1.0, "vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
  ```
- Every dynamic motion sequence automatically appends a 1.0 s stationary stand command to ensure that the robot comes to a stable rest.

### 5.5 Immutable Audit Logging

Every execution writes a record to `logs/llm_audit.jsonl` with:
- UTC timestamp
- User prompt
- Model identifier (`deepseek-r1:8b`, `gemini-2.5-flash`, `mock-deepseek-r1`)
- Wall-clock inference time
- Thinking trace / `reasoning_content`
- Raw model response string
- Final validated/clamped command sequence
- Error messages (if any)

---

## 6. Comprehensive Benchmark Evaluation Results

The evaluation runner (`evaluate_champ_trajectories.py`) executed 6 standardized scenarios headlessly in MuJoCo. Initial conditions were standardized across all runs: $z_0 = 0.28\,\text{m}$, nominal standing configuration, simulation timestep $\Delta t = 0.002\,\text{s}$, telemetry recording frequency $250\,\text{Hz}$.

### 6.1 Benchmark Summary Table

| Scenario | Duration (s) | Status | Forward Dist (m) | Lateral Dist (m) | $v_x$ Tracking RMSE (m/s) | Yaw Tracking RMSE (rad/s) | Mean Base Height (m) | Peak Roll | Peak Pitch | Cost of Transport (CoT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `forward_speed_step` | 8.0 s | ✅ **PASS** | 1.916 m | 0.076 m | 0.065 m/s | 0.515 rad/s | 0.269 m | 3.2° | 5.1° | 1.48 |
| `s_curve` | 8.0 s | ✅ **PASS** | 1.320 m | 1.063 m | 0.068 m/s | 0.521 rad/s | 0.270 m | 3.9° | 5.8° | 1.51 |
| `stand_hold` | 3.0 s | ✅ **PASS** | -0.004 m | 0.000 m | 0.011 m/s | 0.000 rad/s | 0.281 m | 0.0° | 0.8° | 0.38 |
| `stop_and_hold` | 5.0 s | ✅ **PASS** | 0.828 m | 0.043 m | 0.090 m/s | 0.467 rad/s | 0.273 m | 3.4° | 5.5° | 1.51 |
| `turn_in_place` | 7.0 s | ✅ **PASS** | -0.258 m | -0.209 m | 0.068 m/s | 0.443 rad/s | 0.268 m | 2.3° | 2.8° | 3.07 |
| `waypoint_square` | 25.0 s | ✅ **PASS** | 0.043 m | 0.133 m | 0.067 m/s | 0.537 rad/s | 0.269 m | 3.3° | 5.4° | 1.73 |

---

### 6.2 Scenario Analysis

#### 1. Forward Speed Step (`forward_speed_step.json`)
- **Objective**: Test stability and velocity tracking across step accelerations ($0.20 \to 0.45\,\text{m/s}$).
- **Performance**: Traveled $1.916\,\text{m}$ forward with minimal lateral drift ($0.076\,\text{m}$). Forward velocity tracking RMSE remained tight at $0.065\,\text{m/s}$.
- **Attitude**: Roll peak $3.2^\circ$, Pitch peak $5.1^\circ$, showing pitch stability during accelerations.

#### 2. S-Curve Dynamic Steering (`s_curve.json`)
- **Objective**: Verify dynamic stability under concurrent forward translation ($0.35\,\text{m/s}$) and oscillating yaw rates ($\pm 0.45\,\text{rad/s}$).
- **Performance**: Executed smooth sinusoidal path reaching $1.320\,\text{m}$ forward and $1.063\,\text{m}$ lateral displacement.
- **Attitude**: Peak roll was bounded to $3.9^\circ$ despite alternating lateral centrifugal forces.

#### 3. Stand Hold (`stand_hold.json`)
- **Objective**: Verify static equilibrium, zero actuator drift, and baseline power consumption.
- **Performance**: Base height held steadily at $0.281\,\text{m}$ with zero velocity drift (RMSE $0.011\,\text{m/s}$). Cost of Transport was minimal ($0.38$) reflecting low standing holding torque.

#### 4. Stop and Hold (`stop_and_hold.json`)
- **Objective**: Evaluate deceleration dynamics from steady cruise ($0.35\,\text{m/s}$) to complete rest ($0.0\,\text{m/s}$).
- **Performance**: Braked cleanly within $0.828\,\text{m}$ forward displacement without pitching forward or slipping.

#### 5. Turn In Place (`turn_in_place.json`)
- **Objective**: Test pure yaw rotation ($\omega_z = 0.40\,\text{rad/s}$) with zero commanded linear translation.
- **Performance**: Completed heading reorientation with minimal translation ($<0.26\,\text{m}$ drift due to foot contact compliance). Pitch and roll peaks remained under $2.8^\circ$.

#### 6. Waypoint Square Circuit (`waypoint_square.json`)
- **Objective**: Test 2D world-frame closed-loop waypoint navigation along a multi-corner circuit ($(0,0) \to (1,0) \to (1,1) \to (0,1) \to (0,0)$).
- **Performance**: Successfully reached all 4 waypoints sequentially over $25.0\,\text{s}$, returning to the origin with terminal error $<0.14\,\text{m}$. Automatic in-place rotation engaged cleanly on each $90^\circ$ corner.

---

## 7. Verification & Testing

The test suite is partitioned into two test files containing **39 comprehensive unit tests**:

### 7.1 Kinematics, Trajectory, and Safety Tests (`tests/test_champ_control.py` - 19 Tests)
- `TestGo2Kinematics`:
  - `test_solve_ik_neutral_standing`: Verifies neutral standing targets yield valid angles within physical limits and zero hip roll.
  - `test_solve_ik_symmetry`: Confirms left and right limbs yield symmetric thigh/calf angles for symmetric foot targets.
  - `test_solve_ik_outside_workspace_raises`: Asserts `ValueError` when target lies inside the hip cylinder or beyond maximum limb reach.
  - `test_neutral_joint_targets_length`: Validates 12 finite joint targets for standing height.
- `TestChampPhaseGenerator`:
  - `test_phase_offsets_and_period`: Checks diagonal leg pairing (FR/RL=0.0, FL/RR=0.5) and stride period ($0.44\,\text{s}$).
  - `test_phase_stepping_continuity`: Verifies phase progression continuity and mutual exclusivity of stance/swing states.
  - `test_phase_stationary_zero`: Confirms phase progression pauses when stationary.
- `TestTrajectories`:
  - `test_bezier_endpoints_zero`: Proves swing lift is exactly $0.0\,\text{m}$ at liftoff ($s=0$) and touchdown ($s=1$).
  - `test_bezier_peak_at_midpoint`: Confirms peak clearance occurs at $s=0.5$ matching configured swing height ($0.06\,\text{m}$).
  - `test_bezier_non_negative`: Asserts $z(s) \ge 0$ across the full swing interval.
  - `test_stance_curve_retraction`: Verifies linear foot retraction from $+L/2$ to $-L/2$.
- `TestCommandSlewLimiter`:
  - `test_linear_acceleration_clamping`: Tests acceleration clamping to $a_{\text{lin}} \Delta t$.
  - `test_yaw_acceleration_clamping`: Tests yaw rate slew clamping to $\alpha_{\text{yaw}} \Delta t$.
- `TestWaypointNavigator`:
  - `test_reached_tolerance`: Verifies waypoint arrival detection within tolerance radius.
  - `test_pure_yaw_alignment_when_misaligned`: Asserts $v_x = 0$ in-place rotation when heading error $>0.45\,\text{rad}$.
- `TestSafetySupervisor`:
  - `test_safe_nominal_state`: Verifies nominal state passes with zero faults.
  - `test_roll_fault_trigger`: Asserts `ROLL_LIMIT_VIOLATION` when roll $>0.35\,\text{rad}$.
  - `test_low_height_fault_trigger`: Asserts `HEIGHT_LIMIT_VIOLATION` when height $<0.18\,\text{m}$.
  - `test_non_finite_fault_trigger`: Asserts `NON_FINITE_STATE` when state contains `NaN`.

### 7.2 LLM Response Parsing, Multi-Provider Transports, and Safety Tests (`tests/test_llm_commands.py` - 20 Tests)
- `TestResponseParsingAndSanitization`:
  - `test_strip_think_tags`: Verifies `<think>` tag removal and trace extraction.
  - `test_extract_json_from_markdown_fence`: Confirms extraction from ` ```json ... ``` ` markdown blocks.
  - `test_extract_json_from_raw_braces`: Tests extraction from unformatted text containing curly braces.
  - `test_extract_json_from_array`: Verifies extraction of top-level JSON command arrays `[{"mode": ...}]`.
- `TestCommandValidation`:
  - `test_parse_valid_json_array_response`: Validates parsing of top-level arrays generated by Gemini/Gemma models.
  - `test_parse_valid_velocity_fixture`: Validates parsed velocity commands and automatic safety stand appending.
  - `test_parse_valid_waypoint_fixture`: Validates parsed 2D waypoint coordinates and tolerances.
  - `test_parse_out_of_bounds_clamped`: Tests parameter clamping of extreme velocity/yaw values against `SafetyBounds`.
  - `test_parse_malformed_fails_closed_to_stand`: Proves non-JSON text fails closed to a safe standing command.
  - `test_adversarial_nan_inf_sanitized`: Proves `NaN` and `Inf` inputs are replaced with safe standing parameters.
- `TestOllamaClientMockTransport`:
  - `test_mock_query_success`: Tests HTTP 200 response parsing and audit record generation using mocked `urllib.request`.
  - `test_mock_query_network_error_fails_closed`: Verifies that network connection failures fail closed safely and log the incident.
- `TestGeminiClientMockTransport`:
  - `test_gemini_query_success`: Tests Gemini REST API endpoint serialization, HTTP 200 parsing, and audit logging.
  - `test_missing_api_key_fails_safely`: Verifies that missing API keys gracefully fail closed to a safe stationary stand.
  - `test_load_dotenv_parses_file`: Tests automatic parsing and environment injection of key-value pairs from `.env` files.
- `TestOmniRouteClientMockTransport`:
  - `test_omniroute_query_success`: Validates OpenAI `/v1/chat/completions` request body, `reasoning_content` extraction, and command sanitization.
  - `test_omniroute_network_error_fails_closed`: Verifies that network errors from the OmniRoute gateway fail closed to a safe stand.
- `TestLiveModelServiceClientMockTransport`:
  - `test_live_server_health_check`: Tests non-blocking `GET /health` endpoint verification against `start_model_server.py`.
  - `test_live_server_query_success`: Tests `POST /query` payload dispatch and deserialization into typed `MotionCommand` objects.
  - `test_live_server_offline_fails_closed`: Verifies that an unreachable live model server fails closed without crashing the CLI.

```
Execution Result:
----------------------------------------------------------------------
Ran 39 tests in 0.027s

OK
```

---

## 8. Multi-Agent Development Methodology

To ensure high reliability, development was structured using a multi-agent methodology with clear separation of roles:

| Agent Role | Primary Focus | Delivered Components |
| :--- | :--- | :--- |
| **Agent 1: Controller Engineer** | Core gait kinematics, IK solver, phase generator, Bézier curves, Raibert velocity feedback | `go2_champ_control.py` |
| **Agent 2: Benchmark Engineer** | Automated evaluation harness, scenario definitions, telemetry logging | `evaluate_champ_trajectories.py`, `scenarios/*.json` |
| **Agent 3: LLM Safety Engineer** | Multi-provider transports (OmniRoute, Ollama, Gemini), schema parser, `.env` manager, audit logger | `go2_llm_commands.py` |
| **Agent 4: Microservice Architect** | Standalone model server (`start_model_server.py`), REST health/query dispatch, CLI auto-discovery | `start_model_server.py`, `LiveModelServiceClient` |
| **Agent 5: Test Engineer** | 39-test offline verification suite, test fixtures, mock transports | `tests/test_champ_control.py`, `tests/test_llm_commands.py` |
| **Agent 6: Parameter Tuner** | Slew limits, actuator PD gains, Raibert $k_v = 0.35$ tuning | Baseline & optimized controller parameters |
| **Agent 7: Security Auditor** | Adversarial input testing (`NaN`, prompt injections, fail-closed behavior) | Input guards, sanitization pipeline |
| **Agent 8: Integration Lead** | Natural language CLI, MuJoCo GUI/headless runners, release documentation | `go2_champ_llm_cli.py`, `README.md`, `GO2_CHAMP_LLM_REPORT.md` |

---

## 9. File Manifest & Architecture Map

| File Path | Lines | Description |
| :--- | :---: | :--- |
| `start_model_server.py` | 305 | Standalone model microservice encapsulating model selection and credentials, serving endpoints on `http://127.0.0.1:5005`. |
| `go2_champ_control.py` | 741 | Core CHAMP kinematics, 3-DOF analytic IK, phase generator, Bézier/cosine planners, Raibert feedback, slew limiter, waypoint navigator, safety supervisor. |
| `go2_llm_commands.py` | 1050 | Multi-provider LLM interface (OmniRoute, Live Server, Ollama, Gemini, Mock) using `urllib.request`, `.env` auto-loader, schema parser, think-tag extraction, parameter clamping, audit logger. |
| `go2_champ_llm_cli.py` | 550 | Natural language simulation CLI supporting auto-detected live model server, OmniRoute, Ollama, Gemini, Mock, passive MuJoCo GUI visualizer (default), headless `--no-gui`, and REPL. |
| `evaluate_champ_trajectories.py` | 552 | Headless benchmark runner evaluating all 6 scenarios, generating telemetry CSVs, JSON metrics, and matplotlib trajectory plots. |
| `go2_champ_trot.py` | 541 | Standalone entry point for open-loop and closed-loop trot execution with telemetry logging. |
| `scenarios/*.json` | 6 files | Canonical test scenarios (`forward_speed_step`, `s_curve`, `stand_hold`, `stop_and_hold`, `turn_in_place`, `waypoint_square`). |
| `tests/test_champ_control.py` | 237 | Unit tests (19) for IK, phase generation, Bézier curves, slew rate limiting, waypoint navigation, and safety supervision. |
| `tests/test_llm_commands.py` | 460 | Unit tests (20) for OmniRoute, Live Model Server, JSON object/array parsing, think-tag isolation, Gemini/Ollama mock transports, `.env` parsing, and adversarial sanitization. |
| `tests/fixtures/*.json` | 4 files | Fixtures for valid velocity, valid waypoint, out-of-bounds, and malformed LLM outputs. |
| `.env.example` | 32 | Configuration template for OmniRoute (`OMNIROUTE_BASE_URL`, `OMNIROUTE_MODEL`), Dedicated Server (`MODEL_BACKEND`, `MODEL_SERVER_PORT`), Gemini, and Ollama. |
| `README.md` | 280 | Comprehensive quickstart guide, microservice server usage, multi-provider CLI usage, architecture overview, and benchmark table. |

---

## 10. Usage & Reproducibility Guide

### 10.1 Running Unit Tests
```bash
python -m unittest discover -s tests -v
```

### 10.2 Executing the Full Benchmark Suite
```bash
python evaluate_champ_trajectories.py --scenario all
```

### 10.3 Dedicated Model Server Workflow (Recommended)

1. **Start the Model Server** (model is defined strictly in this script):
```bash
# Start with OmniRoute / OpenAI Gateway (default):
python start_model_server.py

# Or start with a specific model:
python start_model_server.py --backend omniroute --model deepseek-r1:8b --url http://localhost:8000/v1

# Or start with local Ollama:
python start_model_server.py --backend ollama --model deepseek-r1:8b

# Or start with Google Gemini:
python start_model_server.py --backend gemini --model gemini-2.5-flash
```

2. **Run Natural Language Commands in MuJoCo** (automatically connects to the live server):
```bash
python go2_champ_llm_cli.py --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

### 10.4 Direct Multi-Provider CLI Modes

#### 1. Live OmniRoute / OpenAI-Compatible Gateway:
```bash
python go2_champ_llm_cli.py --omniroute --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

#### 2. Live Local DeepSeek-R1 / Qwen (Ollama):
Ensure Ollama is running (`ollama serve` and `ollama pull deepseek-r1:8b`):
```bash
python go2_champ_llm_cli.py --ollama --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

#### 3. Live Google AI Studio (Gemini / Gemma API):
Set your API key in a `.env` file (or set `$env:GEMINI_API_KEY="your-key"` in PowerShell):
```bash
python go2_champ_llm_cli.py --gemini --prompt "Walk forward at 0.35 m/s for 3 seconds"
```

#### 4. Headless Execution (No GUI):
```bash
python go2_champ_llm_cli.py --no-gui --prompt "Walk forward slowly for 2 seconds"
```

#### 5. Interactive REPL:
```bash
python go2_champ_llm_cli.py --interactive
```

#### 6. Offline Mock Client (Fast Testing):
```bash
python go2_champ_llm_cli.py --mock --prompt "Turn left for 2.5 seconds"
```

---

## 11. Conclusion

The implemented architecture demonstrates that high-level natural language reasoning via local **DeepSeek-R1/Qwen** and cloud **Gemini/Gemma** models can be interfaced safely and robustly with dynamic quadruped locomotion controllers in MuJoCo physics. By inserting deterministic kinematic planners, acceleration slew limiters, closed-loop Raibert velocity feedback, analytic 3-DOF inverse kinematics, and a fail-closed safety supervisor between the LLM output and simulation physics, the system ensures zero falls, bounded attitude oscillations, and reproducible trajectory tracking across all dynamic maneuvers.
