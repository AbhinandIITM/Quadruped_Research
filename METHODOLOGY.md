# Perception, Estimation, Guidance, and Control Methodologies for J100 Mobile Manipulator in MuJoCo

---

## 1. System Architecture & Coordinate Conventions

The system integrates a **Clearpath Jackal (J100) 4-wheel skid-steer mobile base** and a **Universal Robots UR5e 6-DOF articulated manipulator** inside a unified multi-rate physics environment in MuJoCo.

```
+-----------------------------------------------------------------------------------------+
|                                    PERCEPTION LAYER                                     |
|  RGB-D Camera (30 Hz) --------> Color Segmentation / Centroid ------> LOS Bearing λ,    |
|                                 Depth Map Median Extractor            Range R           |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                STATE ESTIMATION LAYER                                   |
|  IMU Gyroscope (500 Hz) -------> Kinematic LOS Propagation:  dλ/dt = -ω_z               |
|  Visual Range-Rate Filter -----> Target Velocity Estimate:   v_tgt = v_robot*cos(λ) + dR|
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                GUIDANCE & QP CONTROL LAYER                              |
|  Control Barrier Function (CBF)  --> Safety Invariance:   h(R) = R - d_target >= 0      |
|  Control Lyapunov Function (CLF) --> Asymptotic Descent:  V_dot <= -gamma_v * V + delta |
|  Dynamic Acceleration Slew Limit --> Anti-Pitch Filter:   dv/dt <= 1.0 m/s^2            |
|  Active-Set QP Solver (500 Hz)   --> Optimal Control:     [v*, ω*]                      |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                   ACTUATION LAYER                                       |
|  Skid-Steer Kinematics ---------> Slip-Compensated Wheel Speeds [FL, FR, RR, RL]        |
|  UR5e 6-DOF DLS IK -------------> Joint Position & Torque Reference                     |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Perception & Sensor Modeling

### 2.1 Pinhole RGB-D Visual Detector (`30 Hz`)
The forward-facing RGB-D camera (`front_cam`) mounted on the robot base is calibrated using pinhole geometry with field of view $\text{FOVy} = 60^\circ$:
$$f_y = \frac{H / 2}{\tan(\text{FOVy} / 2)}, \quad f_x = f_y, \quad c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$

1. **Target Segmentation**: The target beacon is segmented from the RGB frame using a vectorized chromatic color mask:
   $$\mathcal{M}(u, v) = \left( R > 100 \right) \land \left( R - G > 30 \right) \land \left( R - B > 30 \right)$$
2. **Centroid Extraction**:
   $$\bar{u} = \frac{1}{|\mathcal{M}|} \sum_{(u, v) \in \mathcal{M}} u, \quad \bar{v} = \frac{1}{|\mathcal{M}|} \sum_{(u, v) \in \mathcal{M}} v$$
3. **LOS Bearing Angle ($\lambda$)**:
   $$\lambda_{\text{vis}} = -\arctan\left( \frac{\bar{u} - c_x}{f_x} \right)$$
   *(Positive $\lambda$ indicates target is to the robot's left; negative to the right)*.
4. **Range Estimation ($R$)**: Extracted as the median valid depth value across the segmented mask:
   $$R_{\text{vis}} = \operatorname{median}\left\{ D(u, v) \mid (u, v) \in \mathcal{M}, \; 0.1\text{m} < D(u, v) < 100\text{m} \right\}$$

### 2.2 High-Rate IMU Gyro State Propagation (`500 Hz`)
Between 30 Hz camera frames, the Line-of-Sight bearing angle $\lambda$ is kinematically propagated at 500 Hz using the onboard rate gyroscope:
$$\frac{d\lambda}{dt} = -\omega_z \implies \lambda(t + \Delta t) = \operatorname{wrap}_{[-\pi, \pi]}\left(\lambda(t) - \omega_z \Delta t\right)$$

---

## 3. Skid-Steer Mobile Base Kinematics & Slew Rate Envelope

### 3.1 Skid-Steer Mapping with Slip Factor
For a 4-wheel skid-steer platform with track width $L = 0.368\text{ m}$ and wheel radius $r = 0.098\text{ m}$, wheel slip during turning reduces the effective yaw rate. An empirical slip compensation factor $\gamma_{\text{slip}} \in [0.30, 0.34]$ is applied:
$$\omega_{\text{eff}} = \frac{\omega_{\text{cmd}}}{\gamma_{\text{slip}}}$$
$$v_L = v_{\text{cmd}} - \omega_{\text{eff}} \frac{L}{2}, \quad v_R = v_{\text{cmd}} + \omega_{\text{eff}} \frac{L}{2}$$
$$\omega_{\text{wheel}, L} = \frac{v_L}{r}, \quad \omega_{\text{wheel}, R} = \frac{v_R}{r}$$

### 3.2 Dynamic Acceleration & Anti-Pitch Slew Limiting
Sudden velocity commands $0 \rightarrow v_{\text{max}}$ in a single timestep induce aggressive reaction torques on the chassis, causing the front caster/wheels to pitch up ($\theta_{\text{pitch}} > 8^\circ$).

To eliminate pitch bounce and maintain flat chassis attitude, dynamic rate-of-change bounds are strictly enforced on each 500 Hz tick:
$$v_{\text{cmd}}(t) \in \left[ \max(0.0, \; v(t-\Delta t) - a_{\text{dec}} \Delta t), \; \min(v_{\text{max}}, \; v(t-\Delta t) + a_{\text{acc}} \Delta t) \right]$$
$$\omega_{\text{cmd}}(t) \in \left[ \omega(t-\Delta t) - \alpha_{\text{acc}} \Delta t, \; \omega(t-\Delta t) + \alpha_{\text{acc}} \Delta t \right]$$
where $a_{\text{acc}} = 1.0\text{ m/s}^2$, $a_{\text{dec}} = 1.8\text{ m/s}^2$, and $\alpha_{\text{acc}} = 3.5\text{ rad/s}^2$.

---

## 4. Control Lyapunov & Control Barrier Functions (CLF-CBF)

The guidance stack is formulated as an active-set Quadratic Program (QP) combining safety invariance (CBF), asymptotic convergence (CLF), and pitch slew limiting.

### 4.1 Control Barrier Function (CBF) — Zero-Overshoot Invariance
We define the candidate barrier function representing safe standoff distance:
$$h(R) = R - d_{\text{target}} \ge 0$$

For a dynamic target moving with projected forward velocity $v_{\text{target}}$, the relative range dynamics are:
$$\dot{h}(R) = \dot{R} = v_{\text{target}} - v \cos(\lambda)$$

Enforcing the Nagumo-type forward invariance condition $\dot{h} + \gamma_h h \ge 0$:
$$v_{\text{target}} - v \cos(\lambda) + \gamma_h (R - d_{\text{target}}) \ge 0 \implies v \cos(\lambda) \le v_{\text{target}} + \gamma_h (R - d_{\text{target}})$$

To account for physical braking deceleration $a_{\text{dec}}$, the second-order kinematic braking limit is integrated into the barrier:
$$v_{\text{cbf}} = \frac{v_{\text{target}} + \min\left(\gamma_h (R - d_{\text{target}}), \; \sqrt{2 a_{\text{dec}} \max(0, R - d_{\text{target}})}\right)}{\max(0.1, \; \cos\lambda)}$$

* **Static Target ($v_{\text{target}} = 0$)**: As $R \to d_{\text{target}}$, $v_{\text{cbf}} \to 0.0\text{ m/s}$. The robot comes to rest with **zero distance overshoot** without ever reversing.
* **Moving Target ($v_{\text{target}} > 0$)**: As $R \to d_{\text{target}}$, $v_{\text{cbf}} \to v_{\text{target}}$. The robot smoothly matches the target's orbital speed while maintaining distance $d_{\text{target}}$.

### 4.2 Target Forward Velocity Estimation (No Integrator Windup)
Rather than relying on integral windup, the target's velocity $v_{\text{target}}$ is filtered directly from the visual range-rate $\dot{R}$:
$$\dot{R}_{\text{raw}} = \frac{R(t) - R(t - \Delta t_{\text{vis}})}{\Delta t_{\text{vis}}}$$
$$\dot{R}_{\text{filtered}} \leftarrow (1 - \alpha_f) \dot{R}_{\text{filtered}} + \alpha_f \dot{R}_{\text{raw}}$$
$$v_{\text{target\_est}} = \operatorname{clip}\left(v_{\text{robot}} \cos\lambda + \dot{R}_{\text{filtered}}, \; 0.0, \; v_{\text{max}}\right)$$

### 4.3 Control Lyapunov Function (CLF) — Exponential Stability
To drive the distance error $e_R = R - d_{\text{target}}$ asymptotically to zero:
$$V(R) = \frac{1}{2} k_R (R - d_{\text{target}})^2$$
$$\dot{V}(R) = k_R (R - d_{\text{target}}) \dot{R} = -k_R (R - d_{\text{target}}) \cos(\lambda) v$$

The exponential stability descent condition is:
$$\dot{V}(R) \le -\gamma_V V(R) + \delta \quad (\delta \ge 0 \text{ slack relaxation})$$

### 4.4 Analytical Active-Set QP Solver (`< 1 µs` at 500 Hz)
At each 500 Hz control cycle, the optimal forward velocity $v^*$ is computed by solving:
$$\min_{v, \delta} \frac{1}{2} (v - v_{\text{ref}})^2 + \frac{1}{2} p_{\delta} \delta^2$$
$$\text{subject to} \quad \begin{cases} v \le v_{\text{cbf}} & \text{(CBF Safety Invariance)} \\ L_g V \cdot v - \delta \le -\gamma_V V & \text{(CLF Stability)} \\ v \in [v_{\text{lb}}, \; v_{\text{ub}}] & \text{(Dynamic Slew / Rate Limit)} \end{cases}$$

---

## 5. UR5e Manipulator Cartesian Trajectory Tracking

### 5.1 Damped Least Squares (DLS) Inverse Kinematics
Given target end-effector Cartesian velocity $\dot{x}_{\text{cmd}} = \dot{x}_{\text{ref}} + K_p (x_{\text{ref}} - x_{\text{EEF}})$, joint velocities are computed via Levenberg-Marquardt Damped Pseudoinverse:
$$J^\dagger = J^T \left( J J^T + \lambda_{\text{dls}}^2 I \right)^{-1}$$

### 5.2 Nullspace Posture Optimization
Arm redundancy ($n=6$) is optimized in the nullspace projector $N = I - J^\dagger J$ to maintain an optimal resting posture $q_{\text{rest}}$:
$$\dot{q}_{\text{cmd}} = J^\dagger \dot{x}_{\text{cmd}} + \left( I - J^\dagger J \right) K_{\text{null}} \left( q_{\text{rest}} - q \right)$$

### 5.3 Integrated Reference State Integration
To prevent gravity droop against UR5e link weights ($\pm 150\text{ Nm}$), the commanded joint position reference $q_{\text{ref}}$ is integrated:
$$q_{\text{ref}}(t + \Delta t) = q_{\text{ref}}(t) + \dot{q}_{\text{cmd}} \Delta t$$
$$\tau = K_{\text{pos}} (q_{\text{ref}} - q) + K_{\text{vel}} (\dot{q}_{\text{cmd}} - \dot{q}) + \tau_{\text{grav\_comp}}$$

---

## 6. Experimental Performance Validation

### Summary Metrics

| Metric | Baseline P/PI Controller | CLF-CBF Active-Set QP Controller |
| :--- | :--- | :--- |
| **Static Target Docking** | Penetrated to $0.66\text{ m}$, then reversed back to $0.80\text{ m}$ | **Glides monotonically to $0.80\text{ m}$** (Zero Overshoot, No Reversal) |
| **Chassis Pitch Oscillation** | $> 8.0^\circ$ (Front wheels lifting up on startup) | **$0.23^\circ$** (Chassis remains completely flat) |
| **Moving Target Range Error** | $9.81^\circ$ bearing lag, range divergence | **$0.80\text{ m}$ exact distance hold**, velocity matching $0.80\text{ m/s}$ |
| **Mean LOS Bearing Alignment** | $9.81^\circ$ | **$0.22^\circ$** steady-state |
| **UR5e Trajectory Error** | — | **$2.48\text{ mm}$** steady-state tracking error |

---

## 7. Command Reference

### Running CLF-CBF Guidance Stack
```powershell
# 1. Static Target Precision Docking (Zero Overshoot & Anti-Pitch Slew)
python simulate_j100_clf_cbf.py --scenario static --d_stop 0.80

# 2. Moving Target Pursuit (Dynamic CBF Speed Matching)
python simulate_j100_clf_cbf.py --scenario moving --d_stop 0.80

# 3. Multi-Waypoint Circuit Navigation
python simulate_j100_clf_cbf.py --scenario waypoints
```

### Running UR5e Manipulator Cartesian Trajectory Tracking
```powershell
# 1. 3D Figure-8 Trajectory (Spatial Cartesian Ribbon)
python ur5e/simulate_ur5e.py --trajectory figure8

# 2. Spatial Circular Orbit
python ur5e/simulate_ur5e.py --trajectory circle

# 3. 3D Helix / Spiral Trajectory
python ur5e/simulate_ur5e.py --trajectory spiral
```
