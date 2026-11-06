# Copyright (c) 2022-2026, Computer Graphics and Vision Lab (CGV).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unit and regression test suite for AgileX PiPER Bird Strike (BS) Moving Target Suite.
Verifies:
1. Smooth half-cosine S-curve interpolation (boundary values, monotonicity, smoothness).
2. Deterministic benchmark trajectory synthesis (T1 Step Response and T2 Velocity Sweep).
3. CSV persistence, roundtrip fidelity, and flexible column handling (degrees, radians, missing fields).
4. Kinematic statistics, range validation, limit violation detection, and finite-difference velocity calculation.
5. Mock hardware interface, automated recording, and safe playback execution (pre-flight ramp, limit clamping, safe-disable).
6. Simulation math and trajectory sampling in sim_piper_ball_pick_and_move.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from experiments.bird_strike import (
    GRIPPER_BALL_GRASP_UM,
    GRIPPER_CLOSED_BALL,
    GRIPPER_MAX_UM,
    GRIPPER_MIN_UM,
    GRIPPER_OPEN,
    PIPER_JOINT_LIMITS_DEG,
    POSE_GRASP,
    POSE_HOME,
    POSE_HOVER,
    POSE_LIFT,
    MockPiperInterface,
    clamp_gripper_um,
    clamp_joint_angles,
    compute_dynamic_shake_pose,
    compute_trajectory_stats,
    generate_t1_step_trajectory,
    generate_t2_sweep_trajectory,
    gripper_um_to_finger_joint_m,
    interpolate_pose,
    load_trajectory_csv,
    prepare_csv_trajectory_frames,
    resolve_piper_interface,
    sample_trajectory_frame,
    save_trajectory_csv,
    smooth_s_curve,
)
from experiments.bird_strike.piper_teach_playback import (
    execute_info,
    execute_playback,
    execute_record,
)
from experiments.bird_strike.sim_piper_ball_pick_and_move import build_arg_parser


class TestInterpolationAndCurves:
    """Test mathematical properties of smooth trajectory interpolation curves."""

    def test_smooth_s_curve_boundaries(self):
        t0, t1 = 2.0, 4.0
        assert smooth_s_curve(1.5, t0, t1) == 0.0
        assert smooth_s_curve(2.0, t0, t1) == 0.0
        assert smooth_s_curve(3.0, t0, t1) == pytest.approx(0.5, abs=1e-6)
        assert smooth_s_curve(4.0, t0, t1) == 1.0
        assert smooth_s_curve(4.5, t0, t1) == 1.0

    def test_smooth_s_curve_monotonicity(self):
        t0, t1 = 0.0, 1.0
        steps = 50
        values = [smooth_s_curve(i / steps, t0, t1) for i in range(steps + 1)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestTrajectorySynthesis:
    """Test synthesis of deterministic benchmark trajectories T1 and T2."""

    def test_t1_step_trajectory_structure_and_limits(self):
        traj = generate_t1_step_trajectory(dt=0.02)
        assert len(traj) == 1201
        assert traj[0]["timestamp"] == pytest.approx(0.0)
        assert traj[-1]["timestamp"] == pytest.approx(24.0)

        # Base nominal pose: J1=0, J2=45, J3=-85, J4=0, J5=40, J6=0
        assert traj[0]["j1_deg"] == pytest.approx(0.0)
        assert traj[0]["j2_deg"] == pytest.approx(45.0)
        assert traj[0]["j3_deg"] == pytest.approx(-85.0)

        j1_vals = [f["j1_deg"] for f in traj]
        j2_vals = [f["j2_deg"] for f in traj]
        # J1 reaches +15 deg and -15 deg steps
        assert max(j1_vals) == pytest.approx(15.0, abs=1e-3)
        assert min(j1_vals) == pytest.approx(-15.0, abs=1e-3)
        # J2 reaches +57 deg (+12 deg elevation step)
        assert max(j2_vals) == pytest.approx(57.0, abs=1e-3)

        # Gripper is firmly set to ball grasp width
        for f in traj:
            assert f["gripper_um"] == GRIPPER_BALL_GRASP_UM

    def test_t2_sweep_trajectory_structure_and_speed_regimes(self):
        traj = generate_t2_sweep_trajectory(dt=0.02)
        assert len(traj) == 1401
        assert traj[0]["timestamp"] == pytest.approx(0.0)
        assert traj[-1]["timestamp"] == pytest.approx(28.0)

        # J1 reaches up to 25 deg in the high speed regime
        j1_vals = [f["j1_deg"] for f in traj]
        assert max(j1_vals) == pytest.approx(25.0, abs=1e-2)
        assert min(j1_vals) == pytest.approx(-25.0, abs=1e-2)

        # At the end of the trajectory, arm settles back to neutral center
        last = traj[-1]
        assert last["j1_deg"] == pytest.approx(0.0, abs=1e-3)
        assert last["j2_deg"] == pytest.approx(45.0, abs=1e-3)


class TestCsvSerializationAndRoundtrip:
    """Test saving, loading, flexible column resolution, and error handling for CSV trajectories."""

    def test_csv_save_and_load_roundtrip(self, tmp_path):
        traj_orig = generate_t1_step_trajectory(dt=0.1)
        csv_path = tmp_path / "test_t1.csv"
        save_trajectory_csv(csv_path, traj_orig)
        assert csv_path.is_file()

        traj_loaded = load_trajectory_csv(csv_path)
        assert len(traj_loaded) == len(traj_orig)
        for orig, loaded in zip(traj_orig, traj_loaded):
            assert loaded["timestamp"] == pytest.approx(orig["timestamp"], abs=1e-4)
            assert loaded["j1_deg"] == pytest.approx(orig["j1_deg"], abs=1e-3)
            assert loaded["j2_deg"] == pytest.approx(orig["j2_deg"], abs=1e-3)
            assert loaded["j1_rad"] == pytest.approx(orig["j1_rad"], abs=1e-5)
            assert loaded["gripper_um"] == orig["gripper_um"]

    def test_load_csv_with_only_deg_columns(self, tmp_path):
        csv_path = tmp_path / "only_deg.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "j1_deg", "j2_deg", "j3_deg", "j4_deg", "j5_deg", "j6_deg", "gripper_um"])
            writer.writerow([0.0, 10.0, 20.0, -30.0, 0.0, 15.0, 0.0, 26000])

        loaded = load_trajectory_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0]["j1_deg"] == pytest.approx(10.0)
        assert loaded[0]["j1_rad"] == pytest.approx(math.radians(10.0))

    def test_load_csv_with_only_rad_columns(self, tmp_path):
        csv_path = tmp_path / "only_rad.csv"
        r1, r2 = math.radians(15.0), math.radians(45.0)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "j1_rad", "j2_rad", "j3_rad", "j4_rad", "j5_rad", "j6_rad", "gripper_um"])
            writer.writerow([0.0, r1, r2, 0.0, 0.0, 0.0, 0.0, 26000])

        loaded = load_trajectory_csv(csv_path)
        assert len(loaded) == 1
        assert loaded[0]["j1_rad"] == pytest.approx(r1)
        assert loaded[0]["j1_deg"] == pytest.approx(15.0)

    def test_load_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_trajectory_csv("non_existent_trajectory_file_xyz.csv")

    def test_load_csv_missing_joint_raises_key_error(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        with open(bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "j1_deg"])
            writer.writerow([0.0, 10.0])

        with pytest.raises(KeyError, match="missing joint"):
            load_trajectory_csv(bad_csv)


class TestHardwareSafetyAndClamping:
    """Test angle and gripper clamping against physical limits."""

    def test_clamp_joint_angles_within_and_exceeding_limits(self):
        # Within limits
        safe_deg = [0.0, 45.0, -85.0, 0.0, 40.0, 0.0]
        assert clamp_joint_angles(safe_deg) == safe_deg

        # Out-of-bounds: J1 > 150 -> 150, J2 < 0 -> 0, J3 > 0 -> 0, J4 < -135 -> -135
        out_deg = [180.0, -10.0, 10.0, -200.0, 150.0, -250.0]
        clamped = clamp_joint_angles(out_deg)
        assert clamped[0] == 150.0
        assert clamped[1] == 0.0
        assert clamped[2] == 0.0
        assert clamped[3] == -135.0
        assert clamped[4] == 100.0
        assert clamped[5] == -180.0

    def test_clamp_joint_angles_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected 6 joint angles"):
            clamp_joint_angles([0.0, 10.0])

    def test_clamp_gripper_um(self):
        assert clamp_gripper_um(26000) == 26000
        assert clamp_gripper_um(-1000) == GRIPPER_MIN_UM
        assert clamp_gripper_um(100000) == GRIPPER_MAX_UM


class TestTrajectoryStatsAndMockExecution:
    """Test stats computation, mock hardware recording, playback execution, and info inspection."""

    def test_compute_trajectory_stats_normal_and_violation(self):
        traj = generate_t1_step_trajectory(dt=0.02)
        stats = compute_trajectory_stats(traj)
        assert stats["num_frames"] == 1201
        assert stats["total_duration_s"] == pytest.approx(24.0)
        assert stats["average_rate_hz"] == pytest.approx(50.0, abs=0.1)
        assert stats["has_joint_limit_violation"] is False
        assert stats["has_gripper_limit_violation"] is False

        j1_stats = stats["joints"]["joint_1"]
        assert j1_stats["min_deg"] == pytest.approx(-15.0, abs=1e-2)
        assert j1_stats["max_deg"] == pytest.approx(15.0, abs=1e-2)
        assert j1_stats["range_deg"] == pytest.approx(30.0, abs=1e-2)
        assert j1_stats["max_vel_degps"] > 40.0  # ~58 deg/s

        assert stats["gripper"]["min_mm"] == pytest.approx(26.0)
        assert stats["gripper"]["max_mm"] == pytest.approx(26.0)

        # Test violation detection with out-of-bound point
        traj_violation = list(traj[:5])
        traj_violation.append({
            "timestamp": 0.12,
            "dt": 0.02,
            "j1_deg": 190.0,  # exceeds +150
            "j2_deg": 45.0,
            "j3_deg": -85.0,
            "j4_deg": 0.0,
            "j5_deg": 40.0,
            "j6_deg": 0.0,
            "gripper_um": 90000,  # exceeds 70000
            "j1_rad": 0.0,
            "j2_rad": 0.0,
            "j3_rad": 0.0,
            "j4_rad": 0.0,
            "j5_rad": 0.0,
            "j6_rad": 0.0,
        })
        v_stats = compute_trajectory_stats(traj_violation)
        assert v_stats["has_joint_limit_violation"] is True
        assert v_stats["has_gripper_limit_violation"] is True
        assert len(v_stats["violations"]) == 2

    def test_mock_piper_interface_lifecycle(self):
        piper = resolve_piper_interface("can0", use_mock=True)
        assert isinstance(piper, MockPiperInterface)

        assert piper.ConnectPort() is True
        assert piper.connected is True
        assert piper.EnableArm(7) is True
        assert piper.enabled is True

        # Test command sending
        piper.JointCtrl(10000, 20000, -30000, 0, 40000, 0)
        piper.GripperCtrl(26000, 1000, 0x01, 0)

        # Test joint & gripper state queries
        j_msgs = piper.GetArmJointMsgs()
        assert j_msgs.joint_state.joint_1 == 10000
        g_msgs = piper.GetArmGripperMsgs()
        assert g_msgs.gripper_state.grippers_angle == 26000

        # Safe disable
        assert piper.DisableArm(7) is True
        assert piper.enabled is False

    def test_mock_playback_execution_with_clamping_and_zero_speed_guard(self, tmp_path):
        traj = generate_t1_step_trajectory(dt=0.1)[:5]
        csv_path = tmp_path / "short_t1.csv"
        save_trajectory_csv(csv_path, traj)

        # Normal mock playback with speed=50.0
        args = argparse.Namespace(
            input=str(csv_path),
            can="can0",
            speed=50.0,
            ramp_time=0.01,
            safe_disable=True,
            mock=True,
        )
        execute_playback(args)

        # Speed <= 0 should raise ValueError
        args_zero = argparse.Namespace(
            input=str(csv_path),
            can="can0",
            speed=0.0,
            ramp_time=0.01,
            safe_disable=True,
            mock=True,
        )
        with pytest.raises(ValueError, match="Playback speed must be strictly positive"):
            execute_playback(args_zero)

    def test_mock_record_execution(self, tmp_path):
        out_csv = tmp_path / "mock_record.csv"
        args = argparse.Namespace(
            can="can0",
            rate_hz=50.0,
            duration=0.05,  # 50ms record -> 2-3 frames
            output=str(out_csv),
            mock=True,
            no_confirm=True,
        )
        execute_record(args)
        assert out_csv.is_file()
        rec_data = load_trajectory_csv(out_csv)
        assert len(rec_data) >= 1
        assert "j1_deg" in rec_data[0]
        assert "gripper_um" in rec_data[0]

    def test_execute_info(self, tmp_path, capsys):
        traj = generate_t1_step_trajectory(dt=0.1)[:5]
        csv_path = tmp_path / "info_traj.csv"
        save_trajectory_csv(csv_path, traj)

        args = argparse.Namespace(input=str(csv_path))
        execute_info(args)
        captured = capsys.readouterr().out
        assert "Trajectory File Inspection" in captured
        assert "Total Frames:     5" in captured


class TestSimulationMathAndTrajectoryHelpers:
    """Test pure-python simulation math and helper functions from sim_piper_ball_pick_and_move."""

    def test_interpolate_pose_boundaries_and_halfway(self):
        p0 = [0.0, 10.0, -20.0]
        p1 = [10.0, 30.0, 0.0]
        assert interpolate_pose(p0, p1, 0.0) == pytest.approx(p0)
        assert interpolate_pose(p0, p1, 1.0) == pytest.approx(p1)
        half = interpolate_pose(p0, p1, 0.5)
        assert half == pytest.approx([5.0, 20.0, -10.0])

    def test_gripper_um_to_finger_joint_m(self):
        # 0 um -> (0.0, 0.0)
        assert gripper_um_to_finger_joint_m(0) == pytest.approx((0.0, 0.0))
        # 70000 um (70mm) -> finger stroke 35mm = 0.035m (symmetric joint7 & joint8)
        assert gripper_um_to_finger_joint_m(70000) == pytest.approx((0.035, -0.035))
        # Upper clamp
        assert gripper_um_to_finger_joint_m(100000) == pytest.approx((0.035, -0.035))
        # Lower clamp
        assert gripper_um_to_finger_joint_m(-5000) == pytest.approx((0.0, 0.0))

    def test_prepare_csv_trajectory_frames(self):
        raw = [
            {
                "timestamp": 0.0,
                "j1_rad": 0.1,
                "j2_rad": 0.2,
                "j3_rad": -0.3,
                "j4_rad": 0.0,
                "j5_rad": 0.5,
                "j6_rad": 0.0,
                "gripper_um": 70000,
            }
        ]
        frames = prepare_csv_trajectory_frames(raw)
        assert len(frames) == 1
        assert frames[0]["timestamp"] == 0.0
        assert frames[0]["arm_rad"] == [0.1, 0.2, -0.3, 0.0, 0.5, 0.0]
        assert frames[0]["gripper_rad"] == pytest.approx([0.035, -0.035])

    def test_sample_trajectory_frame(self):
        frames = [
            {"timestamp": 0.0, "arm_rad": [0.0] * 6, "gripper_rad": [0.035, -0.035]},
            {"timestamp": 0.5, "arm_rad": [0.5] * 6, "gripper_rad": [0.020, -0.020]},
            {"timestamp": 1.0, "arm_rad": [1.0] * 6, "gripper_rad": [0.013, -0.013]},
        ]
        arm_t0, grip_t0 = sample_trajectory_frame(frames, sim_time=0.0, speed=1.0)
        assert arm_t0 == [0.0] * 6
        assert grip_t0 == pytest.approx([0.035, -0.035])

        arm_t_mid, grip_t_mid = sample_trajectory_frame(frames, sim_time=0.5, speed=1.0)
        assert arm_t_mid == [0.5] * 6
        assert grip_t_mid == pytest.approx([0.020, -0.020])

        arm_t1_hold, grip_t1_hold = sample_trajectory_frame(frames, sim_time=1.0, speed=1.0, loop=False)
        assert arm_t1_hold == [1.0] * 6
        assert grip_t1_hold == pytest.approx([0.013, -0.013])

        # Empty fallback
        arm_def, grip_def = sample_trajectory_frame([], sim_time=0.0)
        assert arm_def == POSE_HOME
        assert grip_def == GRIPPER_OPEN

    def test_compute_dynamic_shake_pose(self):
        shake_0 = compute_dynamic_shake_pose(phase_timer=0.0)
        assert len(shake_0) == 6
        # At t=0: sin(0)=0, cos(0)=1
        assert shake_0[0] == pytest.approx(POSE_LIFT[0])
        assert shake_0[1] == pytest.approx(POSE_LIFT[1])
        assert shake_0[2] == pytest.approx(POSE_LIFT[2] + 0.10)
        assert shake_0[4] == pytest.approx(POSE_LIFT[4] + 0.25)

    def test_build_arg_parser(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--ball_radius", "0.030", "--speed", "2.0"])
        assert args.ball_radius == pytest.approx(0.030)
        assert args.speed == pytest.approx(2.0)
        assert args.as2_mounted is False
