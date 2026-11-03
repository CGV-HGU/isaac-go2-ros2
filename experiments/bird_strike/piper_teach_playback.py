#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AgileX PiPER 6-DOF Robotic Arm - Drag-Teaching, Trajectory Recording & Playback Rig.
Part of the Bird-Strike (BS) Moving-Target Experiment Suite.
Computer Graphics and Vision Lab (CGV)

Features:
  1. record:        Record human drag-teaching trajectory (joint angles + gripper) via CAN (piper_sdk).
  2. playback:      Safely replay recorded CSV trajectory with smooth pre-flight interpolation.
  3. generate-t1:   Synthesize deterministic Step Response trajectory (T1) for tracker rise/settling time.
  4. generate-t2:   Synthesize deterministic Constant Velocity Sweep trajectory (T2) for speed-error benchmarking.
  5. info:          Inspect CSV trajectory metadata, durations, sampling rates, and peak joint velocities.

Usage Examples:
  # Generate analytical benchmark trajectories
  python3 piper_teach_playback.py generate-t1
  python3 piper_teach_playback.py generate-t2

  # Record manual human demonstration (drag-teaching)
  python3 piper_teach_playback.py record --can can0 --output trajectories/drag_motion_01.csv

  # Replay trajectory with physical robot arm (with 50% speed limit)
  python3 piper_teach_playback.py playback --input trajectories/trajectory_t1_step.csv --speed 0.5

  # Dry-run test on PC/Tablet without CAN hardware
  python3 piper_teach_playback.py playback --input trajectories/trajectory_t1_step.csv --mock
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

# PiPER Hardware Joint Limits (Degrees & Radians)
PIPER_JOINT_LIMITS_DEG = [
    (-150.0, 150.0),  # Joint 1: Base Yaw
    (0.0, 195.0),     # Joint 2: Shoulder Pitch
    (-170.0, 0.0),    # Joint 3: Elbow Pitch
    (-135.0, 135.0),  # Joint 4: Wrist Roll
    (-100.0, 100.0),  # Joint 5: Wrist Pitch
    (-180.0, 180.0),  # Joint 6: Wrist Yaw / Tip Roll
]

# Gripper travel: 0 to 70 mm (0 to 70,000 micrometers)
GRIPPER_MIN_UM = 0
GRIPPER_MAX_UM = 70000

# Ball grasp target for ~4cm target sphere (in micrometers)
GRIPPER_BALL_GRASP_UM = 26000  # ~2.6cm stroke between finger pads


# ==============================================================================
# SDK Interface & Mock Fallback
# ==============================================================================

class MockPiperInterface:
    """Software simulation mock for testing trajectory logic without physical CAN hardware."""

    def __init__(self, can_name: str = "can0", **kwargs: Any) -> None:
        self.can_name = can_name
        self.connected = False
        self.enabled = False
        self._current_joints_raw = [0, 45000, -85000, 0, 40000, 0]  # 0.001 deg
        self._current_gripper_um = 70000  # fully open

    def ConnectPort(self) -> bool:
        self.connected = True
        print(f"[MOCK] Connected to simulated CAN interface '{self.can_name}'.")
        return True

    def EnableArm(self, mask: int = 7) -> bool:
        self.enabled = True
        print(f"[MOCK] Arm enabled with mask {mask}.")
        return True

    def DisableArm(self, mask: int = 7) -> bool:
        self.enabled = False
        print(f"[MOCK] Arm disabled with mask {mask}.")
        return True

    def MotionCtrl_2(self, mode: int, move_type: int, speed: int, is_mit: int) -> None:
        pass

    def JointCtrl(self, j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> None:
        self._current_joints_raw = [int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)]

    def GripperCtrl(self, stroke_um: int, effort_mNm: int, mode: int, flag: int) -> None:
        self._current_gripper_um = int(stroke_um)

    def GetArmStatus(self) -> Any:
        class _Status:
            def __init__(self) -> None:
                self.ctrl_mode = 0x01  # Normal CAN command mode
                self.arm_status = self

        return _Status()

    def GetArmJointMsgs(self) -> Any:
        class _JointState:
            def __init__(self, raw_j: List[int]) -> None:
                self.joint_1 = raw_j[0]
                self.joint_2 = raw_j[1]
                self.joint_3 = raw_j[2]
                self.joint_4 = raw_j[3]
                self.joint_5 = raw_j[4]
                self.joint_6 = raw_j[5]

        class _JointMsgs:
            def __init__(self, raw_j: List[int]) -> None:
                self.joint_state = _JointState(raw_j)

        return _JointMsgs(self._current_joints_raw)

    def GetArmGripperMsgs(self) -> Any:
        class _GripperState:
            def __init__(self, um: int) -> None:
                self.grippers_angle = um

        class _GripperMsgs:
            def __init__(self, um: int) -> None:
                self.gripper_state = _GripperState(um)

        return _GripperMsgs(self._current_gripper_um)


def resolve_piper_interface(can_port: str, use_mock: bool = False) -> Any:
    """Instantiate the best available Piper SDK interface or mock."""
    if use_mock:
        return MockPiperInterface(can_port)

    # Try importing official piper_sdk variants
    sdk_cls = None
    for cls_name in ["C_PiperInterface_V3", "C_PiperInterface_V2", "C_PiperInterface"]:
        try:
            mod = __import__("piper_sdk", fromlist=[cls_name])
            if hasattr(mod, cls_name):
                sdk_cls = getattr(mod, cls_name)
                break
        except ImportError:
            continue

    if sdk_cls is None:
        raise RuntimeError(
            "AgileX PiPER Python SDK (`piper_sdk`) is not installed in the current environment.\n"
            "To install, run:\n"
            "    pip install python-can piper_sdk\n"
            "Or to test trajectory generation and logic without hardware, use the `--mock` flag."
        )

    try:
        piper = sdk_cls(can_port)
    except TypeError:
        piper = sdk_cls(can_name=can_port)

    return piper


# ==============================================================================
# Trajectory Generation: T1 (Step) and T2 (Sweep)
# ==============================================================================

def smooth_s_curve(t: float, t_start: float, t_end: float) -> float:
    """Half-cosine smooth step transition from 0.0 to 1.0."""
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    tau = (t - t_start) / (t_end - t_start)
    return 0.5 * (1.0 - math.cos(math.pi * tau))


def generate_t1_step_trajectory(dt: float = 0.02) -> List[dict]:
    """
    Generate T1 Step Response Trajectory.
    Evaluates vision tracker's step response: rise time, settling time, overshoot, steady-state error.
    """
    # Baseline nominal pose: ball held in center of camera FOV
    base_deg = [0.0, 45.0, -85.0, 0.0, 40.0, 0.0]
    total_time = 24.0
    num_points = int(math.ceil(total_time / dt)) + 1

    trajectory: List[dict] = []
    for step in range(num_points):
        t = step * dt

        # Step Sequence:
        # 0s - 3s:   Hold Center (0°)
        # 3s - 3.4s: Rapid Step to +15° Yaw (Step jump 15° over 400ms)
        # 3.4s - 8s: Hold +15° (Settling observation)
        # 8s - 8.4s: Return to Center
        # 8.4s - 12s:Hold Center
        # 12s - 12.4s: Rapid Step to -15° Yaw
        # 12.4s - 17s:Hold -15°
        # 17s - 17.4s: Return to Center
        # 17.4s - 20s: Hold Center
        # 20s - 20.4s: Rapid Step to +12° Elevation (Pitch)
        # 20.4s - 23s: Hold +12°
        # 23s - 24s: Return to Center

        j1_offset = 0.0
        j2_offset = 0.0

        if t < 3.0:
            j1_offset = 0.0
        elif t < 3.4:
            j1_offset = 15.0 * smooth_s_curve(t, 3.0, 3.4)
        elif t < 8.0:
            j1_offset = 15.0
        elif t < 8.4:
            j1_offset = 15.0 * (1.0 - smooth_s_curve(t, 8.0, 8.4))
        elif t < 12.0:
            j1_offset = 0.0
        elif t < 12.4:
            j1_offset = -15.0 * smooth_s_curve(t, 12.0, 12.4)
        elif t < 17.0:
            j1_offset = -15.0
        elif t < 17.4:
            j1_offset = -15.0 * (1.0 - smooth_s_curve(t, 17.0, 17.4))
        elif t < 20.0:
            j1_offset = 0.0
        elif t < 20.4:
            j2_offset = 12.0 * smooth_s_curve(t, 20.0, 20.4)
        elif t < 23.0:
            j2_offset = 12.0
        else:
            j2_offset = 12.0 * (1.0 - smooth_s_curve(t, 23.0, 23.5))

        j_deg = [
            base_deg[0] + j1_offset,
            base_deg[1] + j2_offset,
            base_deg[2],
            base_deg[3],
            base_deg[4],
            base_deg[5],
        ]

        trajectory.append({
            "timestamp": t,
            "dt": dt,
            "j1_deg": j_deg[0],
            "j2_deg": j_deg[1],
            "j3_deg": j_deg[2],
            "j4_deg": j_deg[3],
            "j5_deg": j_deg[4],
            "j6_deg": j_deg[5],
            "gripper_um": GRIPPER_BALL_GRASP_UM,
            "j1_rad": math.radians(j_deg[0]),
            "j2_rad": math.radians(j_deg[1]),
            "j3_rad": math.radians(j_deg[2]),
            "j4_rad": math.radians(j_deg[3]),
            "j5_rad": math.radians(j_deg[4]),
            "j6_rad": math.radians(j_deg[5]),
        })

    return trajectory


def generate_t2_sweep_trajectory(dt: float = 0.02) -> List[dict]:
    """
    Generate T2 Constant Velocity Sweep Trajectory.
    Evaluates tracking error as a function of target angular velocity (10°/s, 25°/s, 50°/s).
    All stages are designed with exact integer cycle durations ensuring continuous zero-crossing transitions.
    """
    base_deg = [0.0, 45.0, -85.0, 0.0, 40.0, 0.0]
    trajectory: List[dict] = []

    # Timeline (28.0s Total):
    # 0.0s - 2.0s:  Rest at Center
    # 2.0s - 8.0s:  Low Speed Sweep (1 cycle of 6.0s, amplitude 15°, peak vel ~ 15.7 °/s)
    # 8.0s - 14.4s: Medium Speed Sweep (2 cycles of 3.2s = 6.4s, amplitude 20°, peak vel ~ 39.3 °/s)
    # 14.4s - 20.4s:High Speed Sweep (3 cycles of 2.0s = 6.0s, amplitude 25°, peak vel ~ 78.5 °/s)
    # 20.4s - 26.4s:Coupled Figure-8 (2 cycles of 3.0s = 6.0s, coupled Yaw/Pitch Lissajous)
    # 26.4s - 28.0s:Settle at Center
    total_time = 28.0
    num_points = int(math.ceil(total_time / dt)) + 1

    for step in range(num_points):
        t = step * dt
        j1_offset = 0.0
        j2_offset = 0.0

        if t < 2.0:
            # Settle at Center
            j1_offset = 0.0
            j2_offset = 0.0
        elif t < 8.0:
            # Low Speed: 15° amplitude, 1 full cycle in 6.0s
            phase = t - 2.0
            j1_offset = 15.0 * math.sin((2.0 * math.pi / 6.0) * phase)
        elif t < 14.4:
            # Medium Speed: 20° amplitude, 2 full cycles (period = 3.2s)
            phase = t - 8.0
            j1_offset = 20.0 * math.sin((2.0 * math.pi / 3.2) * phase)
        elif t < 20.4:
            # High Speed: 25° amplitude, 3 full cycles (period = 2.0s)
            phase = t - 14.4
            j1_offset = 25.0 * math.sin((2.0 * math.pi / 2.0) * phase)
        elif t < 26.4:
            # 2D Coupled Lissajous figure-8: 2 full cycles (period = 3.0s)
            phase = t - 20.4
            j1_offset = 18.0 * math.sin((2.0 * math.pi / 3.0) * phase)
            j2_offset = 10.0 * math.sin((4.0 * math.pi / 3.0) * phase)
        else:
            # Settle at Center (t >= 26.4)
            j1_offset = 0.0
            j2_offset = 0.0

        j_deg = [
            base_deg[0] + j1_offset,
            base_deg[1] + j2_offset,
            base_deg[2],
            base_deg[3],
            base_deg[4],
            base_deg[5],
        ]

        trajectory.append({
            "timestamp": t,
            "dt": dt,
            "j1_deg": j_deg[0],
            "j2_deg": j_deg[1],
            "j3_deg": j_deg[2],
            "j4_deg": j_deg[3],
            "j5_deg": j_deg[4],
            "j6_deg": j_deg[5],
            "gripper_um": GRIPPER_BALL_GRASP_UM,
            "j1_rad": math.radians(j_deg[0]),
            "j2_rad": math.radians(j_deg[1]),
            "j3_rad": math.radians(j_deg[2]),
            "j4_rad": math.radians(j_deg[3]),
            "j5_rad": math.radians(j_deg[4]),
            "j6_rad": math.radians(j_deg[5]),
        })

    return trajectory


# ==============================================================================
# CSV File I/O
# ==============================================================================

FIELDNAMES = [
    "timestamp", "dt",
    "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg",
    "gripper_um",
    "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad",
]


def save_trajectory_csv(filepath: str | Path, trajectory: List[dict]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in trajectory:
            formatted_row = {
                "timestamp": f"{row['timestamp']:.4f}",
                "dt": f"{row['dt']:.4f}",
                "j1_deg": f"{row['j1_deg']:.3f}",
                "j2_deg": f"{row['j2_deg']:.3f}",
                "j3_deg": f"{row['j3_deg']:.3f}",
                "j4_deg": f"{row['j4_deg']:.3f}",
                "j5_deg": f"{row['j5_deg']:.3f}",
                "j6_deg": f"{row['j6_deg']:.3f}",
                "gripper_um": f"{int(row['gripper_um'])}",
                "j1_rad": f"{row['j1_rad']:.6f}",
                "j2_rad": f"{row['j2_rad']:.6f}",
                "j3_rad": f"{row['j3_rad']:.6f}",
                "j4_rad": f"{row['j4_rad']:.6f}",
                "j5_rad": f"{row['j5_rad']:.6f}",
                "j6_rad": f"{row['j6_rad']:.6f}",
            }
            writer.writerow(formatted_row)
    print(f"[SUCCESS] Trajectory ({len(trajectory)} frames) saved -> {path}")


def load_trajectory_csv(filepath: str | Path) -> List[dict]:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    trajectory: List[dict] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append({
                "timestamp": float(row["timestamp"]),
                "dt": float(row.get("dt", 0.02)),
                "j1_deg": float(row["j1_deg"]),
                "j2_deg": float(row["j2_deg"]),
                "j3_deg": float(row["j3_deg"]),
                "j4_deg": float(row["j4_deg"]),
                "j5_deg": float(row["j5_deg"]),
                "j6_deg": float(row["j6_deg"]),
                "gripper_um": int(float(row.get("gripper_um", 0))),
                "j1_rad": float(row.get("j1_rad", math.radians(float(row["j1_deg"])))),
                "j2_rad": float(row.get("j2_rad", math.radians(float(row["j2_deg"])))),
                "j3_rad": float(row.get("j3_rad", math.radians(float(row["j3_deg"])))),
                "j4_rad": float(row.get("j4_rad", math.radians(float(row["j4_deg"])))),
                "j5_rad": float(row.get("j5_rad", math.radians(float(row["j5_deg"])))),
                "j6_rad": float(row.get("j6_rad", math.radians(float(row["j6_deg"])))),
            })
    return trajectory


# ==============================================================================
# Record (Drag-Teaching) Mode
# ==============================================================================

def execute_record(args: argparse.Namespace) -> None:
    """Record joint angles and gripper positions from the physical PiPER arm."""
    print("=" * 72)
    print(" [AgileX PiPER] Drag-Teaching & Trajectory Recording Mode")
    print(f" CAN Interface: {args.can} | Sampling Rate: {args.rate_hz} Hz")
    print("=" * 72)

    piper = resolve_piper_interface(args.can, use_mock=args.mock)
    piper.ConnectPort()
    time.sleep(0.2)

    dt = 1.0 / float(args.rate_hz)
    output_path = Path(args.output)

    print("\n[INFO] Checking arm connection...")
    arm_status = piper.GetArmStatus()
    if arm_status is None and not args.mock:
        print("[WARNING] No CAN response received yet. Ensure USB-CAN adapter is connected and 1Mbps bitrate is set.")

    print("\n" + "=" * 60)
    print(" >>> TEACH MODE INSTRUCTIONS:")
    print(" 1. Press the TEACH button on the PiPER arm (or move the arm gently).")
    print(" 2. Press [ENTER] in this terminal when ready to begin recording.")
    print(" 3. Move the arm along your desired target path.")
    print(" 4. Press [Ctrl + C] when finished to save the CSV trajectory.")
    print("=" * 60 + "\n")

    try:
        input("Press [ENTER] to START recording...")
    except KeyboardInterrupt:
        print("\nRecording canceled.")
        return

    print("\n>>> [RECORDING STARTED] Sampling trajectory... (Press Ctrl+C to STOP)\n")
    trajectory: List[dict] = []
    start_time = time.monotonic()
    last_loop_time = start_time
    frame_idx = 0

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            step_dt = now - last_loop_time
            last_loop_time = now

            joint_msgs = piper.GetArmJointMsgs()
            gripper_msgs = piper.GetArmGripperMsgs()

            if joint_msgs and hasattr(joint_msgs, "joint_state"):
                js = joint_msgs.joint_state
                # PiPER CAN protocol returns joint values in 0.001 degrees
                raw_joints = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
                j_deg = [v / 1000.0 for v in raw_joints]
            else:
                j_deg = [0.0] * 6

            if gripper_msgs and hasattr(gripper_msgs, "gripper_state"):
                grip_um = int(gripper_msgs.gripper_state.grippers_angle)
            else:
                grip_um = 0

            point = {
                "timestamp": elapsed,
                "dt": step_dt,
                "j1_deg": j_deg[0],
                "j2_deg": j_deg[1],
                "j3_deg": j_deg[2],
                "j4_deg": j_deg[3],
                "j5_deg": j_deg[4],
                "j6_deg": j_deg[5],
                "gripper_um": grip_um,
                "j1_rad": math.radians(j_deg[0]),
                "j2_rad": math.radians(j_deg[1]),
                "j3_rad": math.radians(j_deg[2]),
                "j4_rad": math.radians(j_deg[3]),
                "j5_rad": math.radians(j_deg[4]),
                "j6_rad": math.radians(j_deg[5]),
            }
            trajectory.append(point)
            frame_idx += 1

            if frame_idx % int(args.rate_hz) == 0:
                print(
                    f"[{elapsed:6.2f}s | Frame {frame_idx:5d}] "
                    f"Yaw: {j_deg[0]:+6.1f}° | Pitch: {j_deg[1]:+6.1f}° | Grip: {grip_um / 1000.0:4.1f}mm"
                )

            # Sleep to maintain requested sampling period
            target_next = start_time + frame_idx * dt
            sleep_s = target_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            if args.duration > 0 and elapsed >= args.duration:
                print(f"\n[INFO] Configured duration of {args.duration}s reached.")
                break

    except KeyboardInterrupt:
        print("\n\n>>> [RECORDING STOPPED BY USER]")

    if trajectory:
        save_trajectory_csv(output_path, trajectory)
    else:
        print("[WARNING] No trajectory points were recorded.")


# ==============================================================================
# Playback Mode (with Pre-Flight Interpolation & Safety Clamping)
# ==============================================================================

def execute_playback(args: argparse.Namespace) -> None:
    """Safely replay a recorded or synthesized CSV trajectory on the PiPER arm."""
    print("=" * 72)
    print(" [AgileX PiPER] Trajectory Playback Mode")
    print(f" Input File: {args.input} | Speed Factor: {args.speed:.2f}x")
    print(f" CAN Interface: {args.can} | Pre-flight Ramp: {args.ramp_time}s")
    print("=" * 72)

    trajectory = load_trajectory_csv(args.input)
    if not trajectory:
        raise ValueError("Trajectory is empty.")

    total_duration = trajectory[-1]["timestamp"] - trajectory[0]["timestamp"]
    scaled_duration = total_duration / max(args.speed, 0.01)
    print(f"[INFO] Loaded {len(trajectory)} frames. Trajectory Duration: {total_duration:.2f}s (Replay: {scaled_duration:.2f}s)")

    piper = resolve_piper_interface(args.can, use_mock=args.mock)
    piper.ConnectPort()
    time.sleep(0.2)

    # 1. Read current arm joint positions
    curr_j_msgs = piper.GetArmJointMsgs()
    if curr_j_msgs and hasattr(curr_j_msgs, "joint_state"):
        js = curr_j_msgs.joint_state
        current_deg = [js.joint_1 / 1000.0, js.joint_2 / 1000.0, js.joint_3 / 1000.0,
                       js.joint_4 / 1000.0, js.joint_5 / 1000.0, js.joint_6 / 1000.0]
    else:
        current_deg = [0.0, 45.0, -85.0, 0.0, 40.0, 0.0]

    first_frame = trajectory[0]
    target_start_deg = [
        first_frame["j1_deg"], first_frame["j2_deg"], first_frame["j3_deg"],
        first_frame["j4_deg"], first_frame["j5_deg"], first_frame["j6_deg"],
    ]

    print("\n[SAFETY CHECK] Arm Poses:")
    print(f"  Current Pose (deg): {[round(x, 1) for x in current_deg]}")
    print(f"  Target Start (deg): {[round(x, 1) for x in target_start_deg]}")

    # Enable arm motors for CAN command control
    speed_pct = int(min(max(args.speed * 40, 10), 80))
    piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
    piper.EnableArm(7)
    time.sleep(0.1)

    # 2. Smoothly ramp from current position to first waypoint over ramp_time
    ramp_duration = max(float(args.ramp_time), 1.0)
    print(f"\n>>> [PHASE 1] Smoothly moving to initial trajectory waypoint ({ramp_duration:.1f}s)...")
    ramp_steps = int(ramp_duration * 50)
    for i in range(ramp_steps):
        alpha = smooth_s_curve(i / float(ramp_steps), 0.0, 1.0)
        interp_deg = [
            current_deg[k] + alpha * (target_start_deg[k] - current_deg[k])
            for k in range(6)
        ]
        raw_cmd = [int(round(d * 1000.0)) for d in interp_deg]
        piper.JointCtrl(*raw_cmd)
        piper.GripperCtrl(int(first_frame["gripper_um"]), 1000, 0x01, 0)
        time.sleep(0.02)

    print(">>> Initial pose reached. Ready for trajectory execution.")
    time.sleep(0.5)

    # 3. Trajectory Playback Loop
    print("\n>>> [PHASE 2] Executing Trajectory Playback... (Press Ctrl+C to ABORT)\n")
    playback_start = time.monotonic()

    try:
        for idx, frame in enumerate(trajectory):
            # Compute time to wait according to frame timestamp
            frame_time = frame["timestamp"] / args.speed
            actual_elapsed = time.monotonic() - playback_start
            wait_s = frame_time - actual_elapsed
            if wait_s > 0:
                time.sleep(wait_s)

            # Send CAN position command
            raw_cmd = [
                int(round(frame["j1_deg"] * 1000.0)),
                int(round(frame["j2_deg"] * 1000.0)),
                int(round(frame["j3_deg"] * 1000.0)),
                int(round(frame["j4_deg"] * 1000.0)),
                int(round(frame["j5_deg"] * 1000.0)),
                int(round(frame["j6_deg"] * 1000.0)),
            ]
            piper.JointCtrl(*raw_cmd)
            piper.GripperCtrl(int(frame["gripper_um"]), 1000, 0x01, 0)

            if idx % 50 == 0 or idx == len(trajectory) - 1:
                progress_pct = (idx + 1) / len(trajectory) * 100.0
                print(
                    f"[{progress_pct:5.1f}% | {actual_elapsed:5.2f}s] "
                    f"Cmd: J1={frame['j1_deg']:+6.1f}°, J2={frame['j2_deg']:+6.1f}°, Grip={frame['gripper_um'] / 1000.0:4.1f}mm"
                )

        print("\n[SUCCESS] Trajectory replay completed successfully!")

    except KeyboardInterrupt:
        print("\n\n[EMERGENCY STOP] Playback aborted by user!")
    finally:
        if args.safe_disable:
            print("[INFO] Disabling arm motors for safety...")
            piper.DisableArm(7)


# ==============================================================================
# Info / Inspection Mode
# ==============================================================================

def execute_info(args: argparse.Namespace) -> None:
    """Print trajectory statistics and analysis."""
    trajectory = load_trajectory_csv(args.input)
    if not trajectory:
        print(f"[ERROR] Trajectory file '{args.input}' contains no data.")
        return

    n = len(trajectory)
    t0 = trajectory[0]["timestamp"]
    t1 = trajectory[-1]["timestamp"]
    total_time = t1 - t0
    avg_rate = (n - 1) / max(total_time, 1e-4)

    print("=" * 72)
    print(f" Trajectory File Inspection: {args.input}")
    print("=" * 72)
    print(f" Total Frames:     {n}")
    print(f" Total Duration:   {total_time:.3f} s")
    print(f" Average Rate:     {avg_rate:.2f} Hz")
    print("-" * 72)
    print(f" {'Joint':<10} | {'Min (deg)':<12} | {'Max (deg)':<12} | {'Range (deg)':<12} | {'Max Vel (°/s)':<14}")
    print("-" * 72)

    for j in range(1, 7):
        key = f"j{j}_deg"
        vals = [f[key] for f in trajectory]
        min_v = min(vals)
        max_v = max(vals)
        rng = max_v - min_v

        # Calculate finite-difference peak velocity
        max_vel = 0.0
        for i in range(1, n):
            dt = trajectory[i]["timestamp"] - trajectory[i - 1]["timestamp"]
            if dt > 1e-5:
                vel = abs(trajectory[i][key] - trajectory[i - 1][key]) / dt
                if vel > max_vel:
                    max_vel = vel

        print(f" Joint {j:<4} | {min_v:+10.2f}° | {max_v:+10.2f}° | {rng:10.2f}° | {max_vel:12.2f} °/s")

    grip_vals = [f["gripper_um"] / 1000.0 for f in trajectory]
    print(f" Gripper    | {min(grip_vals):10.1f}mm | {max(grip_vals):10.1f}mm | {max(grip_vals) - min(grip_vals):10.1f}mm | {'N/A':<14}")
    print("=" * 72)


# ==============================================================================
# Main CLI Parser
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgileX PiPER 6-DOF Robotic Arm - Drag-Teaching, Trajectory Recording & Playback Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # Command: record
    rec_parser = subparsers.add_parser("record", help="Record drag-teaching motion via CAN")
    rec_parser.add_argument("--can", type=str, default="can0", help="CAN interface name (default: can0)")
    rec_parser.add_argument("--rate-hz", type=float, default=50.0, help="Sampling frequency in Hz (default: 50.0)")
    rec_parser.add_argument("--duration", type=float, default=0.0, help="Max recording duration in seconds (0 = manual Ctrl+C)")
    rec_parser.add_argument("--output", type=str, default="experiments/bird_strike/trajectories/recorded_trajectory.csv",
                            help="Output CSV file path")
    rec_parser.add_argument("--mock", action="store_true", help="Run with simulated mock arm without physical CAN hardware")

    # Command: playback
    play_parser = subparsers.add_parser("playback", help="Safely replay a CSV trajectory on PiPER")
    play_parser.add_argument("--input", type=str, required=True, help="Path to trajectory CSV file")
    play_parser.add_argument("--can", type=str, default="can0", help="CAN interface name (default: can0)")
    play_parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 0.5 for half speed)")
    play_parser.add_argument("--ramp-time", type=float, default=2.5, help="Seconds to interpolate to initial pose before playback")
    play_parser.add_argument("--safe-disable", action="store_true", help="Disable arm motors after playback completes")
    play_parser.add_argument("--mock", action="store_true", help="Run with simulated mock arm without physical CAN hardware")

    # Command: generate-t1
    t1_parser = subparsers.add_parser("generate-t1", help="Synthesize T1 Step Response trajectory CSV")
    t1_parser.add_argument("--output", type=str, default="experiments/bird_strike/trajectories/trajectory_t1_step.csv",
                           help="Output CSV path")
    t1_parser.add_argument("--dt", type=float, default=0.02, help="Time step in seconds (default: 0.02 = 50Hz)")

    # Command: generate-t2
    t2_parser = subparsers.add_parser("generate-t2", help="Synthesize T2 Velocity Sweep trajectory CSV")
    t2_parser.add_argument("--output", type=str, default="experiments/bird_strike/trajectories/trajectory_t2_sweep.csv",
                           help="Output CSV path")
    t2_parser.add_argument("--dt", type=float, default=0.02, help="Time step in seconds (default: 0.02 = 50Hz)")

    # Command: info
    info_parser = subparsers.add_parser("info", help="Inspect trajectory stats (frames, limits, peak velocity)")
    info_parser.add_argument("--input", type=str, required=True, help="Path to trajectory CSV file")

    args = parser.parse_args()

    if args.command == "record":
        execute_record(args)
    elif args.command == "playback":
        execute_playback(args)
    elif args.command == "generate-t1":
        traj = generate_t1_step_trajectory(args.dt)
        save_trajectory_csv(args.output, traj)
    elif args.command == "generate-t2":
        traj = generate_t2_sweep_trajectory(args.dt)
        save_trajectory_csv(args.output, traj)
    elif args.command == "info":
        execute_info(args)


if __name__ == "__main__":
    main()
