"""Headless evaluation harness and benchmark runner for Go2 CHAMP locomotion scenarios.

Executes deterministic locomotion scenarios (stand hold, speed step, stop & hold,
turn in place, S-curve, waypoint navigation), logs full telemetry, computes
rigorous biomechanical/tracking metrics, generates diagnostic figures, and
produces benchmark acceptance reports.

Usage:
    # Run all benchmark scenarios headlessly
    python evaluate_champ_trajectories.py --scenario all

    # Run a specific scenario
    python evaluate_champ_trajectories.py --scenario stand_hold
    python evaluate_champ_trajectories.py --scenario turn_in_place
    python evaluate_champ_trajectories.py --scenario s_curve
    python evaluate_champ_trajectories.py --scenario waypoint_square
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from go2_champ_control import (
    DEFAULT_NEUTRAL_Q,
    GRAVITY,
    HIP_OFFSET,
    HIP_POSITIONS,
    IS_LEFT,
    JOINT_NAMES,
    LEG_NAMES,
    NOMINAL_BODY_HEIGHT,
    ChampTrajectoryPlanner,
    CommandSlewLimiter,
    FootholdPlanner,
    GaitConfig,
    Go2ChampController,
    MotionCommand,
    MotionMode,
    SafetyBounds,
    Waypoint,
    WaypointNavigator,
    configure_model,
    initialize_simulation_standing,
    neutral_joint_targets,
    quaternion_to_euler_xyz,
    solve_ik,
)


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "unitree_go2" / "go2_description.xml"
SCENARIOS_DIR = ROOT_DIR / "scenarios"
LOGS_DIR = ROOT_DIR / "logs"

FOOT_BODY_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


class ScenarioRunner:
    """Runs a single scenario in MuJoCo, records telemetry, and generates metrics."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Configure actuator gains for stable tracking (Kp=60, Kd=5)
        configure_model(self.model)

        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
        self.foot_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self.qpos_addrs = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addrs = self.model.jnt_dofadr[self.joint_ids]

        # Total robot mass
        self.total_mass = float(np.sum(self.model.body_mass))

    def reset_to_standing(self):
        """Deterministically reset the simulation to the neutral standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        initialize_simulation_standing(self.model, self.data, body_height=NOMINAL_BODY_HEIGHT, settle_time=0.20)

    def foot_contacts(self) -> np.ndarray:
        """Return 4-element binary array indicating contact for each foot."""
        contacts = np.zeros(4, dtype=int)
        foot_lookup = {body_id: idx for idx, body_id in enumerate(self.foot_body_ids)}
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            for b in (b1, b2):
                leg_idx = foot_lookup.get(b)
                if leg_idx is not None:
                    contacts[leg_idx] = 1
        return contacts

    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        output_dir: Optional[Path] = None,
        save_plots: bool = True,
        gui: bool = False,
        vision: bool = False,
    ) -> Dict[str, Any]:
        """Execute a scenario schedule and record full performance artifacts."""
        name = scenario_config.get("name", "unnamed_scenario")
        total_duration = float(scenario_config.get("duration_s", 5.0))
        commands_list = scenario_config.get("commands", [])
        scenario_mode = scenario_config.get("mode", "velocity")
        waypoints_data = scenario_config.get("waypoints", [])

        # Reset simulation state
        self.reset_to_standing()

        controller = Go2ChampController()
        if scenario_mode == "waypoint" and waypoints_data:
            wps = [
                Waypoint(
                    x=float(w["x"]),
                    y=float(w["y"]),
                    heading=w.get("heading"),
                    tolerance=float(w.get("tolerance", 0.15)),
                    speed=float(w.get("speed", 0.30)),
                )
                for w in waypoints_data
            ]
            controller.waypoint_navigator.reset(wps)

        # Vision perception pipeline if enabled
        vision_pipeline = None
        last_clearance = None
        if vision:
            try:
                from go2_vision_perception import DepthPerceptionPipeline
                vision_pipeline = DepthPerceptionPipeline(self.model)
            except Exception as e:
                print(f"[Warning] Failed to initialize vision pipeline: {e}")
                vision_pipeline = None

        dt = float(self.model.opt.timestep)
        telemetry_rate_hz = 250
        decimation = max(1, int(round(1.0 / (telemetry_rate_hz * dt))))

        history: List[Dict[str, Any]] = []
        step_idx = 0
        current_time = 0.0

        viewer_ctx = None
        if gui:
            try:
                viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
                print(f"[GUI] Passive viewer opened for scenario '{name}' ({total_duration:.1f}s)")
            except Exception as e:
                print(f"[Warning] Failed to launch GUI viewer: {e}. Falling back to headless.")
                viewer_ctx = None

        wall_clock_ref = time.perf_counter()

        try:
            # Run main simulation loop
            while current_time < total_duration:
                # 1. Determine active command
                active_cmd: MotionCommand
                if scenario_mode == "waypoint":
                    active_cmd = MotionCommand(
                        mode=MotionMode.WAYPOINT,
                        waypoint=controller.waypoint_navigator.current_waypoint(),
                    )
                else:
                    # Find matching command from schedule
                    matched = None
                    for cmd_entry in commands_list:
                        t_start = float(cmd_entry.get("start_time_s", 0.0))
                        t_dur = float(cmd_entry.get("duration_s", total_duration))
                        if t_start <= current_time < t_start + t_dur:
                            matched = cmd_entry
                            break
                    if matched is None and len(commands_list) > 0:
                        matched = commands_list[-1]

                    if matched is not None:
                        mode_str = matched.get("mode", "trot").lower()
                        m_mode = MotionMode.STAND if mode_str == "stand" else MotionMode.VELOCITY
                        active_cmd = MotionCommand(
                            vx=float(matched.get("vx", 0.0)),
                            vy=float(matched.get("vy", 0.0)),
                            yaw_rate=float(matched.get("yaw_rate", 0.0)),
                            mode=m_mode,
                        )
                    else:
                        active_cmd = MotionCommand(mode=MotionMode.STAND)

                # 2. Extract current state
                base_pos = self.data.qpos[:3].copy()
                base_quat = self.data.qpos[3:7].copy()
                base_euler = quaternion_to_euler_xyz(base_quat)
                base_rot_mat = self.data.xmat[self.base_body_id].reshape(3, 3).copy()

                # World linear velocity -> Body linear velocity
                linvel_world = self.data.qvel[:3].copy()
                linvel_body = base_rot_mat.T @ linvel_world

                # Angular velocity in body frame (qvel[3:6] in MuJoCo freejoint is body-frame)
                angvel_body = self.data.qvel[3:6].copy()

                joint_positions = np.array([self.data.qpos[addr] for addr in self.qpos_addrs])
                joint_velocities = np.array([self.data.qvel[addr] for addr in self.dof_addrs])

                # 3. Vision perception clearance update (20 Hz)
                sector_clearance = None
                if vision_pipeline is not None and vision_pipeline.is_available:
                    if step_idx % 25 == 0:
                        try:
                            depth_map = vision_pipeline.capture_depth(self.data)
                            last_clearance = vision_pipeline.analyze_clearance(depth_map)
                        except Exception:
                            last_clearance = None
                    sector_clearance = last_clearance

                # 4. Controller step
                target_q, desired_feet, st_phases, sw_phases, info = controller.step(
                    command=active_cmd,
                    step_time=current_time,
                    dt=dt,
                    base_pos=base_pos,
                    base_euler=base_euler,
                    base_linvel_body=linvel_body,
                    base_angvel_body=angvel_body,
                    joint_positions=joint_positions,
                    sector_clearance=sector_clearance,
                )

                # 4. Apply actuator control targets
                self.data.ctrl[self.actuator_ids] = target_q

                # 5. Advance physics
                mujoco.mj_step(self.model, self.data)

                # 6. Synchronize passive GUI viewer if active
                if viewer_ctx is not None:
                    if not viewer_ctx.is_running():
                        print("\n[Viewer closed by user - finalizing scenario run early]")
                        break
                    viewer_ctx.sync()
                    next_step_wall_time = wall_clock_ref + (step_idx + 1) * dt
                    sleep_dur = next_step_wall_time - time.perf_counter()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)

                # 7. Record telemetry periodically
                if step_idx % decimation == 0:
                    contacts = self.foot_contacts()
                    joint_torques = self.data.actuator_force[self.actuator_ids].copy()
                    power = float(np.sum(np.abs(joint_torques * joint_velocities)))

                    record = {
                        "time_s": current_time,
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
                        record[f"{leg_name}_contact"] = int(contacts[leg_idx])
                        record[f"{leg_name}_st_phase"] = float(st_phases[leg_idx])
                        record[f"{leg_name}_sw_phase"] = float(sw_phases[leg_idx])
                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        record[f"{j_name}_target_rad"] = float(target_q[j_idx])
                        record[f"{j_name}_actual_rad"] = float(joint_positions[j_idx])
                        record[f"{j_name}_torque_nm"] = float(joint_torques[j_idx])

                    history.append(record)

                step_idx += 1
                current_time += dt
        finally:
            if viewer_ctx is not None:
                viewer_ctx.close()

        # Compute summary metrics and evaluate pass criteria
        metrics = self._compute_metrics(scenario_config, history, total_duration)

        # Output logging directory
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_csv(history, output_dir / "data.csv")
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            if save_plots:
                self._generate_plots(history, metrics, scenario_config, output_dir / "plots.png")

        return metrics

    def _compute_metrics(
        self,
        scenario_config: Dict[str, Any],
        history: List[Dict[str, Any]],
        requested_duration: float,
    ) -> Dict[str, Any]:
        """Compute aggregate biomechanical, tracking, and energy metrics."""
        if not history:
            return {"error": "No telemetry recorded"}

        times = np.array([r["time_s"] for r in history])
        run_duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        # Kinematic states
        x = np.array([r["base_x_m"] for r in history])
        y = np.array([r["base_y_m"] for r in history])
        z = np.array([r["base_z_m"] for r in history])
        vx_meas = np.array([r["base_vx_body_mps"] for r in history])
        vx_cmd = np.array([r["cmd_vx"] for r in history])
        wz_meas = np.array([r["base_wz_radps"] for r in history])
        wz_cmd = np.array([r["cmd_yaw"] for r in history])
        roll = np.array([r["base_roll_rad"] for r in history])
        pitch = np.array([r["base_pitch_rad"] for r in history])
        yaw = np.array([r["base_yaw_rad"] for r in history])

        # Tracking performance
        vx_rmse = float(np.sqrt(np.mean((vx_meas - vx_cmd) ** 2)))
        wz_rmse = float(np.sqrt(np.mean((wz_meas - wz_cmd) ** 2)))

        # Joint tracking RMSE
        joint_errors = []
        for j_name in JOINT_NAMES:
            actual = np.array([r[f"{j_name}_actual_rad"] for r in history])
            target = np.array([r[f"{j_name}_target_rad"] for r in history])
            joint_errors.append(actual - target)
        joint_errors_mat = np.array(joint_errors)
        joint_tracking_rmse = float(np.sqrt(np.mean(joint_errors_mat ** 2)))

        # Attitude and height stability
        base_height_mean = float(np.mean(z))
        base_height_std = float(np.std(z))
        base_height_min = float(np.min(z))
        base_roll_std = float(np.std(roll))
        base_pitch_std = float(np.std(pitch))
        base_roll_peak = float(np.max(np.abs(roll)))
        base_pitch_peak = float(np.max(np.abs(pitch)))

        # Displacement
        delta_x = float(x[-1] - x[0])
        delta_y = float(y[-1] - y[0])
        total_dist = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

        # Energy & Power
        powers = np.array([r["power_w"] for r in history])
        dt_sample = float(np.mean(np.diff(times))) if len(times) > 1 else 0.004
        energy_j = float(np.sum(powers) * dt_sample)
        avg_power = float(np.mean(powers))
        cot = float(energy_j / (self.total_mass * GRAVITY * max(0.01, total_dist)))

        # Contact duty factors
        duty_factors = {}
        for leg_name in LEG_NAMES:
            c = np.array([r[f"{leg_name}_contact"] for r in history])
            duty_factors[leg_name] = float(np.mean(c))

        # Check faults
        faults_recorded = [r["fault"] for r in history if r["fault"] is not None]

        # Evaluate against scenario expected_metrics
        expected = scenario_config.get("expected_metrics", {})
        passed = True
        failure_reasons = []

        if len(faults_recorded) > 0:
            passed = False
            failure_reasons.append(f"Controller faults triggered: {set(faults_recorded)}")

        if "min_base_height_m" in expected and base_height_min < float(expected["min_base_height_m"]):
            passed = False
            failure_reasons.append(f"Min base height {base_height_min:.3f}m < expected {expected['min_base_height_m']}m")

        if "max_roll_rad" in expected and base_roll_peak > float(expected["max_roll_rad"]):
            passed = False
            failure_reasons.append(f"Peak roll {base_roll_peak:.3f}rad > expected {expected['max_roll_rad']}rad")

        if "max_pitch_rad" in expected and base_pitch_peak > float(expected["max_pitch_rad"]):
            passed = False
            failure_reasons.append(f"Peak pitch {base_pitch_peak:.3f}rad > expected {expected['max_pitch_rad']}rad")

        if "min_forward_displacement_m" in expected and delta_x < float(expected["min_forward_displacement_m"]):
            passed = False
            failure_reasons.append(f"Forward displacement {delta_x:.3f}m < expected {expected['min_forward_displacement_m']}m")

        if "max_forward_displacement_m" in expected and delta_x > float(expected["max_forward_displacement_m"]):
            passed = False
            failure_reasons.append(f"Forward displacement {delta_x:.3f}m > expected {expected['max_forward_displacement_m']}m")

        metrics = {
            "scenario": scenario_config.get("name"),
            "passed": passed,
            "failure_reasons": failure_reasons,
            "run_duration_s": run_duration,
            "requested_duration_s": requested_duration,
            "forward_displacement_m": delta_x,
            "lateral_displacement_m": delta_y,
            "total_path_length_m": total_dist,
            "final_position": [float(x[-1]), float(y[-1]), float(z[-1])],
            "final_heading_rad": float(yaw[-1]),
            "velocity_tracking_rmse_mps": vx_rmse,
            "yaw_rate_tracking_rmse_radps": wz_rmse,
            "joint_tracking_rmse_rad": joint_tracking_rmse,
            "base_height_mean_m": base_height_mean,
            "base_height_std_m": base_height_std,
            "base_height_min_m": base_height_min,
            "base_roll_std_rad": base_roll_std,
            "base_pitch_std_rad": base_pitch_std,
            "base_roll_peak_abs_rad": base_roll_peak,
            "base_pitch_peak_abs_rad": base_pitch_peak,
            "estimated_energy_j": energy_j,
            "estimated_average_power_w": avg_power,
            "estimated_cost_of_transport": cot,
            "contact_duty_factor": duty_factors,
        }
        return metrics

    def _save_csv(self, history: List[Dict[str, Any]], filepath: Path):
        """Save telemetry records to CSV file."""
        if not history:
            return
        fieldnames = list(history[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    def _generate_plots(
        self,
        history: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        scenario_config: Dict[str, Any],
        filepath: Path,
    ):
        """Generate high-density diagnostic multi-panel figure."""
        times = [r["time_s"] for r in history]
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
        fig.suptitle(
            f"Go2 CHAMP Benchmark: {scenario_config.get('name')} (Status: {'PASS' if metrics.get('passed') else 'FAIL'})",
            fontsize=15, fontweight="bold",
        )

        # 1. Forward and Lateral Velocities
        ax = axes[0, 0]
        ax.plot(times, [r["cmd_vx"] for r in history], "k--", label="Cmd vx (mps)", alpha=0.8)
        ax.plot(times, [r["base_vx_body_mps"] for r in history], "b-", label="Meas vx (mps)", alpha=0.9)
        ax.plot(times, [r["base_vy_body_mps"] for r in history], "g-", label="Meas vy (mps)", alpha=0.7)
        ax.set_title("Body Linear Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 2. Heading / Yaw Rate
        ax = axes[0, 1]
        ax.plot(times, [r["cmd_yaw"] for r in history], "k--", label="Cmd Yaw Rate (rad/s)", alpha=0.8)
        ax.plot(times, [r["base_wz_radps"] for r in history], "m-", label="Meas Yaw Rate (rad/s)", alpha=0.9)
        ax.plot(times, [r["base_yaw_rad"] for r in history], "c-.", label="Heading Yaw (rad)", alpha=0.7)
        ax.set_title("Heading and Rotational Tracking")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rate / Angle (rad/s, rad)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 3. Base Height and Attitude
        ax = axes[1, 0]
        ax.plot(times, [r["base_z_m"] for r in history], "r-", label="Base Height (m)")
        ax.plot(times, [r["base_roll_rad"] for r in history], "b:", label="Roll (rad)")
        ax.plot(times, [r["base_pitch_rad"] for r in history], "g:", label="Pitch (rad)")
        ax.axhline(NOMINAL_BODY_HEIGHT, color="gray", linestyle="--", alpha=0.5, label="Nominal Height")
        ax.set_title("Base Height and Attitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (m) / Angle (rad)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        # 4. 2D Horizontal Trajectory
        ax = axes[1, 1]
        x = [r["base_x_m"] for r in history]
        y = [r["base_y_m"] for r in history]
        ax.plot(x, y, "b-", label="Robot Path", linewidth=2.0)
        ax.plot(x[0], y[0], "go", markersize=8, label="Start")
        ax.plot(x[-1], y[-1], "rs", markersize=8, label="End")

        # Plot scenario waypoints if available
        waypoints = scenario_config.get("waypoints", [])
        if waypoints:
            wpx = [w["x"] for w in waypoints]
            wpy = [w["y"] for w in waypoints]
            ax.plot(wpx, wpy, "k--o", label="Target Waypoints", alpha=0.7)
            for idx, w in enumerate(waypoints):
                ax.annotate(f"WP{idx+1}", (w["x"], w["y"]), textcoords="offset points", xytext=(5,5))

        ax.set_title("2D Ground Trajectory (X-Y)")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.axis("equal")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")

        # 5. Foot Contact States
        ax = axes[2, 0]
        colors = ["blue", "orange", "green", "red"]
        for idx, leg_name in enumerate(LEG_NAMES):
            contacts = [r[f"{leg_name}_contact"] * 0.8 + idx for r in history]
            ax.plot(times, contacts, label=f"{leg_name} (Duty: {metrics['contact_duty_factor'][leg_name]:.2f})", color=colors[idx])
        ax.set_yticks(range(4))
        ax.set_yticklabels(LEG_NAMES)
        ax.set_title("Foot Contact Duty Cycle")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Leg")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

        # 6. Electrical / Mechanical Power
        ax = axes[2, 1]
        powers = [r["power_w"] for r in history]
        ax.plot(times, powers, "purple", alpha=0.7, label=f"Power (Avg: {metrics['estimated_average_power_w']:.1f} W)")
        ax.set_title(f"Actuator Power (CoT: {metrics['estimated_cost_of_transport']:.2f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        plt.savefig(filepath, dpi=150)
        plt.close(fig)


def run_benchmark_suite(
    scenario_names: Optional[List[str]] = None,
    scenarios_dir: Path = SCENARIOS_DIR,
    logs_dir: Path = LOGS_DIR,
    gui: bool = False,
    vision: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the full suite of scenarios and print comparison summary table."""
    if not scenario_names or "all" in scenario_names:
        json_files = sorted(scenarios_dir.glob("*.json"))
    else:
        json_files = [scenarios_dir / f"{name}.json" if not name.endswith(".json") else scenarios_dir / name for name in scenario_names]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = logs_dir / f"benchmark_suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    runner = ScenarioRunner()
    results = []

    print("\n" + "=" * 80)
    print(f"RUNNING GO2 CHAMP BENCHMARK SUITE ({len(json_files)} SCENARIOS)")
    print("=" * 80)

    for sf in json_files:
        if not sf.exists():
            print(f"[!] Warning: Scenario file {sf} not found, skipping.")
            continue

        with open(sf, "r") as f:
            cfg = json.load(f)

        s_name = cfg.get("name", sf.stem)
        scenario_out = suite_dir / f"scenario_{s_name}"
        print(f"\n--> Running scenario: '{s_name}' ({cfg.get('duration_s')}s)...")

        t0 = time.time()
        metrics = runner.run_scenario(cfg, output_dir=scenario_out, save_plots=True, gui=gui, vision=vision)
        elapsed = time.time() - t0

        status_str = "[PASS]" if metrics.get("passed") else "[FAIL]"
        print(f"    Result: {status_str} in {elapsed:.2f}s wall-clock")
        print(f"    Displacement: fwd={metrics.get('forward_displacement_m', 0.0):.3f}m, lat={metrics.get('lateral_displacement_m', 0.0):.3f}m")
        print(f"    Tracking: vx RMSE={metrics.get('velocity_tracking_rmse_mps', 0.0):.3f} m/s, yaw RMSE={metrics.get('yaw_rate_tracking_rmse_radps', 0.0):.3f} rad/s")
        print(f"    Attitude: roll peak={math.degrees(metrics.get('base_roll_peak_abs_rad', 0.0)):.1f} deg, pitch peak={math.degrees(metrics.get('base_pitch_peak_abs_rad', 0.0)):.1f} deg")
        if not metrics.get("passed"):
            print(f"    Failures: {metrics.get('failure_reasons')}")

        results.append(metrics)

    # Save summary report
    summary_path = suite_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown table
    md_lines = [
        f"# Go2 CHAMP Trajectory Benchmark Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Scenario | Status | Fwd Dist (m) | Vx RMSE (m/s) | Yaw RMSE (rad/s) | Height (m) | Roll Peak (deg) | Pitch Peak (deg) | CoT |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m in results:
        status_badge = "✅ PASS" if m.get("passed") else "❌ FAIL"
        roll_deg = math.degrees(m.get("base_roll_peak_abs_rad", 0.0))
        pitch_deg = math.degrees(m.get("base_pitch_peak_abs_rad", 0.0))
        md_lines.append(
            f"| `{m.get('scenario')}` | {status_badge} | {m.get('forward_displacement_m', 0.0):.2f} | "
            f"{m.get('velocity_tracking_rmse_mps', 0.0):.3f} | {m.get('yaw_rate_tracking_rmse_radps', 0.0):.3f} | "
            f"{m.get('base_height_mean_m', 0.0):.3f} | {roll_deg:.1f}° | {pitch_deg:.1f}° | {m.get('estimated_cost_of_transport', 0.0):.2f} |"
        )
    md_lines.append("")
    report_md = "\n".join(md_lines)
    with open(suite_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(report_md)
    print(f"\nArtifacts saved in: {suite_dir}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CHAMP trajectory tracking on Unitree Go2.")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        help="Scenario name(s) to run (e.g. stand_hold, turn_in_place, s_curve) or 'all'",
    )
    parser.add_argument("--gui", dest="gui", action="store_true", default=True, help="Launch MuJoCo passive GUI viewer during scenario execution (default: True)")
    parser.add_argument("--no-gui", "--headless", dest="gui", action="store_false", help="Run headlessly without GUI viewer (for automated tests)")
    parser.add_argument("--vision", action="store_true", default=False, help="Enable closed-loop depth vision perception and reactive obstacle avoidance")
    parser.add_argument("--scenarios-dir", type=Path, default=SCENARIOS_DIR, help="Directory containing scenario JSON files")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Directory to save run logs")
    args = parser.parse_args()

    run_benchmark_suite(
        scenario_names=args.scenario,
        scenarios_dir=args.scenarios_dir,
        logs_dir=args.logs_dir,
        gui=args.gui,
        vision=args.vision,
    )


if __name__ == "__main__":
    main()
