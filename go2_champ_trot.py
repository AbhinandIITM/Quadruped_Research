"""Run a CHAMP-style open-loop trot for Unitree Go2 in MuJoCo.

The controller ports CHAMP's phase oscillator, 12-point Bezier swing path,
cosine stance path, and analytic 3-DOF leg inverse kinematics.  Each run
writes its raw telemetry, derived metrics, and diagnostic plots to logs/.

Examples:
    python go2_champ_trot.py --duration 10
    python go2_champ_trot.py --headless --duration 5 --target-vel 0.32
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "unitree_go2" / "go2_description.xml"
LOG_ROOT = ROOT_DIR / "logs"

# Go2 dimensions [m], taken from go2_description.xml.
HIP_OFFSET = 0.0955
THIGH = 0.213
CALF = 0.213
BODY_HEIGHT = 0.28

# CHAMP gait configuration.  The nominal stride period is 0.44 s.
STANCE_DURATION = 0.22
SWING_DURATION = 0.22
SWING_HEIGHT = 0.06
STANCE_DEPTH = 0.0
MAX_STEP_LENGTH = 0.18

# The model's stock position controllers are compliant under two-leg support.
POSITION_KP = 100.0
POSITION_KD = 4.0
SETTLE_DURATION = 1.0
GRAVITY = 9.81

LEG_NAMES = ("FR", "FL", "RR", "RL")
IS_LEFT = (False, True, False, True)
HIP_POSITIONS = np.array(
    [
        [0.1934, -0.0955, 0.0],
        [0.1934, 0.0955, 0.0],
        [-0.1934, -0.0955, 0.0],
        [-0.1934, 0.0955, 0.0],
    ],
    dtype=float,
)
JOINT_NAMES = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


# -----------------------------------------------------------------------------
# CHAMP algorithm port
# -----------------------------------------------------------------------------

def solve_ik(x, y, z, is_left=False):
    """Solve Go2 hip-roll, thigh-pitch, and calf-pitch for a hip-frame foot.

    The signs follow the +Y pitch axes in the Go2 MuJoCo model: moving a foot
    forward (+X) decreases the thigh-pitch command.
    """
    hip_lateral_offset = HIP_OFFSET if is_left else -HIP_OFFSET
    yz_squared = y * y + z * z
    if yz_squared <= hip_lateral_offset * hip_lateral_offset:
        raise ValueError("Foot target is inside the hip-roll workspace")

    theta_hip = math.atan2(y, -z) - math.atan2(
        hip_lateral_offset,
        math.sqrt(yz_squared - hip_lateral_offset * hip_lateral_offset),
    )
    z_planar = -math.sqrt(yz_squared - hip_lateral_offset * hip_lateral_offset)

    distance_squared = x * x + z_planar * z_planar
    cosine_knee = (distance_squared - THIGH * THIGH - CALF * CALF) / (2.0 * THIGH * CALF)
    cosine_knee = float(np.clip(cosine_knee, -1.0, 1.0))
    theta_calf = -math.acos(cosine_knee)

    alpha = math.atan2(-x, -z_planar)
    beta = math.atan2(
        CALF * math.sin(theta_calf),
        THIGH + CALF * math.cos(theta_calf),
    )
    theta_thigh = alpha - beta
    return theta_hip, theta_thigh, theta_calf


class ChampTrajectoryPlanner:
    """CHAMP's 12-control-point Bezier swing and cosine stance planner."""

    def __init__(self, step_length, swing_height, stance_depth):
        self.step_length = step_length
        self.stance_depth = stance_depth
        self.total_control_points = 12
        self.factorial = (
            1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0,
            362880.0, 3628800.0, 39916800.0, 479001600.0,
        )
        ref_control_points_x = (
            -0.15, -0.2805, -0.3, -0.3, -0.3, 0.0,
            0.0, 0.0, 0.3032, 0.3032, 0.2826, 0.15,
        )
        ref_control_points_y = (
            -0.5, -0.5, -0.3611, -0.3611, -0.3611, -0.3611,
            -0.3611, -0.3214, -0.3214, -0.3214, -0.5, -0.5,
        )

        length_ratio = step_length / 0.4
        self.control_points_x = [point * length_ratio for point in ref_control_points_x]
        self.control_points_x[0] = -step_length / 2.0
        self.control_points_x[-1] = step_length / 2.0

        height_ratio = swing_height / 0.15
        self.control_points_y = [
            -((point * height_ratio) + (0.5 * height_ratio))
            for point in ref_control_points_y
        ]

    def generate(self, swing_phase, stance_phase):
        """Return local forward/up trajectory offsets for a single leg."""
        if stance_phase > swing_phase:
            x = (self.step_length / 2.0) * (1.0 - 2.0 * stance_phase)
            z = -self.stance_depth * math.cos(math.pi * x / (self.step_length + 1e-9))
            return x, z

        if swing_phase > stance_phase:
            x = 0.0
            z = 0.0
            polynomial_degree = self.total_control_points - 1
            for index in range(self.total_control_points):
                coefficient = self.factorial[polynomial_degree] / (
                    self.factorial[index] * self.factorial[polynomial_degree - index]
                )
                bernstein = coefficient * (swing_phase ** index) * (
                    (1.0 - swing_phase) ** (polynomial_degree - index)
                )
                x += bernstein * self.control_points_x[index]
                z -= bernstein * self.control_points_y[index]
            return x, z

        # At the exact phase boundary retain the neutral target rather than
        # selecting a discontinuous trajectory branch.
        return 0.0, 0.0


class ChampPhaseGenerator:
    """Open-loop trot oscillator with CHAMP's diagonal phase offsets."""

    def __init__(self, stance_duration, swing_duration):
        self.stance_duration = stance_duration
        self.swing_duration = swing_duration
        self.stride_period = stance_duration + swing_duration
        # FR/RL together, then FL/RR together.
        self.offsets = (0.0, 0.5, 0.5, 0.0)

    def run(self, elapsed_time):
        stance_phases = [0.0] * 4
        swing_phases = [0.0] * 4
        stride_time = elapsed_time % self.stride_period

        for index, offset in enumerate(self.offsets):
            leg_clock = stride_time - offset * self.stride_period
            while leg_clock < -self.swing_duration:
                leg_clock += self.stride_period
            while leg_clock > self.stance_duration:
                leg_clock -= self.stride_period

            if 0.0 < leg_clock < self.stance_duration:
                stance_phases[index] = leg_clock / self.stance_duration
            elif -self.swing_duration < leg_clock <= 0.0:
                swing_phases[index] = (leg_clock + self.swing_duration) / self.swing_duration
            elif self.stance_duration < leg_clock <= self.stride_period:
                swing_phases[index] = (leg_clock - self.stance_duration) / self.swing_duration

        return stance_phases, swing_phases


# -----------------------------------------------------------------------------
# Measurements, files, and diagnostics
# -----------------------------------------------------------------------------

def quaternion_to_euler_xyz(quaternion):
    """Convert a MuJoCo wxyz unit quaternion to roll, pitch, yaw [rad]."""
    w, x, y, z = quaternion
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_argument)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array((roll, pitch, yaw), dtype=float)


def create_run_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = LOG_ROOT / f"champ_trot_{timestamp}"
    suffix = 1
    while run_directory.exists():
        run_directory = LOG_ROOT / f"champ_trot_{timestamp}_{suffix:02d}"
        suffix += 1
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def build_csv_fieldnames():
    fields = [
        "time_s", "base_x_m", "base_y_m", "base_z_m",
        "base_vx_mps", "base_vy_mps", "base_vz_mps",
        "base_roll_rad", "base_pitch_rad", "base_yaw_rad",
        "base_wx_radps", "base_wy_radps", "base_wz_radps",
        "estimated_power_w",
    ]
    for joint_name in JOINT_NAMES:
        fields.extend((
            f"{joint_name}_target_rad", f"{joint_name}_actual_rad",
            f"{joint_name}_velocity_radps", f"{joint_name}_torque_nm",
        ))
    for leg_name in LEG_NAMES:
        fields.extend((
            f"{leg_name}_stance_phase", f"{leg_name}_swing_phase", f"{leg_name}_contact",
            f"{leg_name}_foot_x_body_m", f"{leg_name}_foot_y_body_m", f"{leg_name}_foot_z_body_m",
            f"{leg_name}_desired_x_body_m", f"{leg_name}_desired_y_body_m", f"{leg_name}_desired_z_body_m",
        ))
    return fields


def foot_contact_states(model, data, foot_body_ids):
    """Return whether each foot body has a contact in the current simulation step."""
    contacts = np.zeros(4, dtype=int)
    foot_lookup = {body_id: index for index, body_id in enumerate(foot_body_ids)}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        bodies = (
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        )
        for body_id in bodies:
            leg_index = foot_lookup.get(body_id)
            if leg_index is not None:
                contacts[leg_index] = 1
    return contacts


def record_sample(
    model,
    data,
    sample_time,
    target_q,
    stance_phases,
    swing_phases,
    desired_foot_positions,
    joint_qpos_addresses,
    joint_dof_addresses,
    foot_body_ids,
    base_body_id,
):
    base_position = data.qpos[:3].copy()
    base_rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    base_euler = quaternion_to_euler_xyz(data.qpos[3:7])
    contacts = foot_contact_states(model, data, foot_body_ids)

    joint_positions = np.array([data.qpos[address] for address in joint_qpos_addresses])
    joint_velocities = np.array([data.qvel[address] for address in joint_dof_addresses])
    joint_torques = data.actuator_force.copy()
    estimated_power = float(np.sum(np.abs(joint_torques * joint_velocities)))

    sample = {
        "time_s": sample_time,
        "base_x_m": base_position[0],
        "base_y_m": base_position[1],
        "base_z_m": base_position[2],
        "base_vx_mps": data.qvel[0],
        "base_vy_mps": data.qvel[1],
        "base_vz_mps": data.qvel[2],
        "base_roll_rad": base_euler[0],
        "base_pitch_rad": base_euler[1],
        "base_yaw_rad": base_euler[2],
        "base_wx_radps": data.qvel[3],
        "base_wy_radps": data.qvel[4],
        "base_wz_radps": data.qvel[5],
        "estimated_power_w": estimated_power,
    }
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        sample[f"{joint_name}_target_rad"] = target_q[joint_index]
        sample[f"{joint_name}_actual_rad"] = joint_positions[joint_index]
        sample[f"{joint_name}_velocity_radps"] = joint_velocities[joint_index]
        sample[f"{joint_name}_torque_nm"] = joint_torques[joint_index]

    for leg_index, leg_name in enumerate(LEG_NAMES):
        foot_position_world = data.xpos[foot_body_ids[leg_index]]
        foot_position_body = base_rotation.T @ (foot_position_world - base_position)
        desired_foot_position = desired_foot_positions[leg_index]
        sample[f"{leg_name}_stance_phase"] = stance_phases[leg_index]
        sample[f"{leg_name}_swing_phase"] = swing_phases[leg_index]
        sample[f"{leg_name}_contact"] = contacts[leg_index]
        sample[f"{leg_name}_foot_x_body_m"] = foot_position_body[0]
        sample[f"{leg_name}_foot_y_body_m"] = foot_position_body[1]
        sample[f"{leg_name}_foot_z_body_m"] = foot_position_body[2]
        sample[f"{leg_name}_desired_x_body_m"] = desired_foot_position[0]
        sample[f"{leg_name}_desired_y_body_m"] = desired_foot_position[1]
        sample[f"{leg_name}_desired_z_body_m"] = desired_foot_position[2]
    return sample


def write_csv(run_directory, samples):
    csv_path = run_directory / "data.csv"
    fieldnames = build_csv_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    return csv_path


def samples_to_array(samples, field):
    return np.asarray([sample[field] for sample in samples], dtype=float)


def calculate_metrics(samples, target_velocity, model, step_length, duration):
    time_values = samples_to_array(samples, "time_s")
    forward_position = samples_to_array(samples, "base_x_m")
    forward_velocity = samples_to_array(samples, "base_vx_mps")
    base_height = samples_to_array(samples, "base_z_m")
    roll = samples_to_array(samples, "base_roll_rad")
    pitch = samples_to_array(samples, "base_pitch_rad")
    power = samples_to_array(samples, "estimated_power_w")

    forward_displacement = float(forward_position[-1] - forward_position[0])
    measured_duration = float(time_values[-1] - time_values[0]) if len(samples) > 1 else 0.0
    mean_forward_velocity = forward_displacement / measured_duration if measured_duration > 0.0 else 0.0
    tracking_error = forward_velocity - target_velocity
    tracking_rmse = float(np.sqrt(np.mean(tracking_error * tracking_error)))

    joint_errors = []
    for joint_name in JOINT_NAMES:
        joint_errors.extend(
            samples_to_array(samples, f"{joint_name}_actual_rad")
            - samples_to_array(samples, f"{joint_name}_target_rad")
        )
    joint_tracking_rmse = float(np.sqrt(np.mean(np.square(joint_errors))))

    estimated_energy = float(np.trapezoid(power, time_values)) if len(samples) > 1 else 0.0
    average_power = estimated_energy / measured_duration if measured_duration > 0.0 else 0.0
    robot_mass = float(model.body_mass.sum())
    cost_of_transport = (
        estimated_energy / (robot_mass * GRAVITY * forward_displacement)
        if forward_displacement > 1e-4
        else None
    )

    contact_duty_factors = {
        leg_name: float(np.mean(samples_to_array(samples, f"{leg_name}_contact")))
        for leg_name in LEG_NAMES
    }
    return {
        "run_duration_s": measured_duration,
        "requested_duration_s": duration,
        "target_velocity_mps": target_velocity,
        "commanded_step_length_m": step_length,
        "stride_period_s": STANCE_DURATION + SWING_DURATION,
        "forward_displacement_m": forward_displacement,
        "mean_forward_velocity_mps": mean_forward_velocity,
        "velocity_tracking_rmse_mps": tracking_rmse,
        "joint_tracking_rmse_rad": joint_tracking_rmse,
        "base_height_mean_m": float(np.mean(base_height)),
        "base_height_std_m": float(np.std(base_height)),
        "base_roll_std_rad": float(np.std(roll)),
        "base_pitch_std_rad": float(np.std(pitch)),
        "base_roll_peak_abs_rad": float(np.max(np.abs(roll))),
        "base_pitch_peak_abs_rad": float(np.max(np.abs(pitch))),
        "estimated_energy_j": estimated_energy,
        "estimated_average_power_w": average_power,
        "estimated_cost_of_transport": cost_of_transport,
        "contact_duty_factor": contact_duty_factors,
    }


def save_metrics(run_directory, metrics):
    metrics_path = run_directory / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics_path


def apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#1E293B",
        "text.color": "#0F172A",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#CBD5E1",
        "grid.linewidth": 0.7,
        "font.size": 10,
    })


def finalize_axis(axis, title, x_label, y_label):
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def generate_plots(run_directory, samples, target_velocity):
    """Create diagnostic engineering plots from exactly the logged samples."""
    apply_plot_style()
    time_values = samples_to_array(samples, "time_s")

    # Velocity tracking
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    axis.plot(time_values, samples_to_array(samples, "base_vx_mps"), color="#0072B2", linewidth=2, label="Measured forward velocity")
    axis.axhline(target_velocity, color="#D55E00", linewidth=2, linestyle="--", label="Commanded velocity")
    finalize_axis(axis, "Forward Velocity Tracking", "Simulation time [s]", "Velocity [m/s]")
    axis.legend(frameon=False)
    figure.savefig(run_directory / "velocity_tracking.png", dpi=180)
    plt.close(figure)

    # Body height and attitude
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    axes[0].plot(time_values, samples_to_array(samples, "base_z_m"), color="#0072B2", linewidth=2)
    finalize_axis(axes[0], "Base Height", "", "Height [m]")
    axes[1].plot(time_values, np.rad2deg(samples_to_array(samples, "base_roll_rad")), color="#009E73", linewidth=2, label="Roll")
    axes[1].plot(time_values, np.rad2deg(samples_to_array(samples, "base_pitch_rad")), color="#CC79A7", linewidth=2, label="Pitch")
    finalize_axis(axes[1], "Base Attitude", "Simulation time [s]", "Angle [deg]")
    axes[1].legend(frameon=False, ncol=2)
    figure.savefig(run_directory / "body_attitude_height.png", dpi=180)
    plt.close(figure)

    # Actual and commanded feet in the body frame.
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False, sharey=True, layout="constrained")
    for leg_index, (leg_name, axis) in enumerate(zip(LEG_NAMES, axes.flat)):
        actual_x = samples_to_array(samples, f"{leg_name}_foot_x_body_m")
        actual_z = samples_to_array(samples, f"{leg_name}_foot_z_body_m")
        desired_x = samples_to_array(samples, f"{leg_name}_desired_x_body_m")
        desired_z = samples_to_array(samples, f"{leg_name}_desired_z_body_m")
        axis.plot(actual_x, actual_z, color="#0072B2", linewidth=1.8, label="Actual")
        axis.plot(desired_x, desired_z, color="#D55E00", linewidth=1.8, linestyle="--", label="Commanded")
        finalize_axis(axis, f"{leg_name} Foot Path", "Body X [m]", "Body Z [m]")
        if leg_index == 0:
            axis.legend(frameon=False)
    figure.savefig(run_directory / "foot_trajectories.png", dpi=180)
    plt.close(figure)

    # Joint command versus realized position, split by the three joint types.
    joint_types = ("hip", "thigh", "calf")
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for joint_type_index, (joint_type, axis) in enumerate(zip(joint_types, axes)):
        for leg_index, (leg_name, color) in enumerate(zip(LEG_NAMES, colors)):
            joint_name = f"{leg_name}_{joint_type}_joint"
            actual = samples_to_array(samples, f"{joint_name}_actual_rad")
            commanded = samples_to_array(samples, f"{joint_name}_target_rad")
            axis.plot(time_values, actual, color=color, linewidth=1.6, label=f"{leg_name} actual")
            axis.plot(time_values, commanded, color=color, linewidth=1.0, linestyle="--", alpha=0.9, label=f"{leg_name} command")
        finalize_axis(axis, f"{joint_type.capitalize()} Joint Tracking", "" if joint_type != "calf" else "Simulation time [s]", "Angle [rad]")
        if joint_type_index == 0:
            axis.legend(frameon=False, ncol=4, fontsize=8)
    figure.savefig(run_directory / "joint_tracking.png", dpi=180)
    plt.close(figure)


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def configure_model(model):
    model.opt.gravity[:] = (0.0, 0.0, -GRAVITY)
    model.actuator_gainprm[:, 0] = POSITION_KP
    model.actuator_biasprm[:, 1] = -POSITION_KP
    model.actuator_biasprm[:, 2] = -POSITION_KD


def neutral_joint_targets():
    targets = []
    for is_left in IS_LEFT:
        lateral_position = HIP_OFFSET if is_left else -HIP_OFFSET
        targets.extend(solve_ik(0.0, lateral_position, -BODY_HEIGHT, is_left))
    return np.asarray(targets, dtype=float)


def desired_leg_targets(phase_generator, trajectory_planner, elapsed_time):
    stance_phases, swing_phases = phase_generator.run(elapsed_time)
    target_q = []
    desired_foot_positions = []
    for leg_index, is_left in enumerate(IS_LEFT):
        forward_offset, vertical_offset = trajectory_planner.generate(
            swing_phases[leg_index], stance_phases[leg_index]
        )
        lateral_position = HIP_OFFSET if is_left else -HIP_OFFSET
        foot_z = -BODY_HEIGHT + vertical_offset
        target_q.extend(solve_ik(forward_offset, lateral_position, foot_z, is_left))
        desired_foot_positions.append(
            HIP_POSITIONS[leg_index] + np.array((forward_offset, 0.0, foot_z))
        )
    return np.asarray(target_q, dtype=float), stance_phases, swing_phases, desired_foot_positions


def run_simulation(duration, target_velocity, headless):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    configure_model(model)

    actuator_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in JOINT_NAMES
    ])
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in JOINT_NAMES
    ])
    joint_qpos_addresses = model.jnt_qposadr[joint_ids]
    joint_dof_addresses = model.jnt_dofadr[joint_ids]
    foot_body_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in FOOT_BODY_NAMES
    ])
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    phase_generator = ChampPhaseGenerator(STANCE_DURATION, SWING_DURATION)
    step_length = float(np.clip(target_velocity * phase_generator.stride_period, 0.02, MAX_STEP_LENGTH))
    trajectory_planner = ChampTrajectoryPlanner(step_length, SWING_HEIGHT, STANCE_DEPTH)
    neutral_targets = neutral_joint_targets()

    run_directory = create_run_directory()
    print(f"Logging run data to: {run_directory}")
    print(f"Settling for {SETTLE_DURATION:.1f} s before starting the trot.")

    data.ctrl[actuator_ids] = neutral_targets
    settle_steps = int(round(SETTLE_DURATION / model.opt.timestep))
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    samples = []
    max_steps = int(math.ceil(duration / model.opt.timestep))

    def step_controller(step_index):
        elapsed_time = step_index * model.opt.timestep
        target_q, stance_phases, swing_phases, desired_feet = desired_leg_targets(
            phase_generator, trajectory_planner, elapsed_time
        )
        data.ctrl[actuator_ids] = target_q
        mujoco.mj_step(model, data)
        samples.append(record_sample(
            model, data, elapsed_time, target_q, stance_phases, swing_phases,
            desired_feet, joint_qpos_addresses, joint_dof_addresses,
            foot_body_ids, base_body_id,
        ))

    if headless:
        for step_index in range(max_steps):
            step_controller(step_index)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_clock_reference = time.perf_counter()
            for step_index in range(max_steps):
                if not viewer.is_running():
                    print("Viewer closed; finalizing the partial run.")
                    break
                step_controller(step_index)
                viewer.sync()
                next_step_wall_time = wall_clock_reference + (step_index + 1) * model.opt.timestep
                time.sleep(max(0.0, next_step_wall_time - time.perf_counter()))

    if not samples:
        raise RuntimeError("No simulation samples were recorded.")

    write_csv(run_directory, samples)
    metrics = calculate_metrics(samples, target_velocity, model, step_length, duration)
    save_metrics(run_directory, metrics)
    generate_plots(run_directory, samples, target_velocity)
    return run_directory, metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Trot duration after settling, in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--target-vel", type=float, default=0.32,
        help="Forward velocity command in m/s (default: 0.32).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without the MuJoCo viewer; suitable for repeatable batch runs.",
    )
    arguments = parser.parse_args()
    if arguments.duration <= 0.0:
        parser.error("--duration must be positive")
    if arguments.target_vel <= 0.0:
        parser.error("--target-vel must be positive")
    return arguments


def main():
    arguments = parse_arguments()
    run_directory, metrics = run_simulation(
        duration=arguments.duration,
        target_velocity=arguments.target_vel,
        headless=arguments.headless,
    )
    print(
        "Completed CHAMP trot: "
        f"{metrics['mean_forward_velocity_mps']:.3f} m/s mean velocity, "
        f"{metrics['forward_displacement_m']:.3f} m forward."
    )
    print(f"CSV, metrics, and plots saved in: {run_directory}")


if __name__ == "__main__":
    main()
