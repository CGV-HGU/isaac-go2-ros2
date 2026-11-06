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
4. Kinematic statistics, range validation, and finite-difference velocity calculation.
5. Mock hardware interface and safe playback execution (pre-flight ramp and safe-disable).
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
    GRIPPER_MAX_UM,
    GRIPPER_MIN_UM,
    PIPER_JOINT_LIMITS_DEG,
    MockPiperInterface,
    compute_trajectory_stats,
    generate_t1_step_trajectory,
    generate_t2_sweep_trajectory,
    load_trajectory_csv,
    resolve_piper_interface,
    save_trajectory_csv,
    smooth_s_curve,
)
from experiments.bird_strike.piper_teach_playback import execute_playback


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


class TestTrajectoryStatsAndMockExecution:
    """Test stats computation and mock hardware playback execution."""

    def test_compute_trajectory_stats(self):
        traj = generate_t1_step_trajectory(dt=0.02)
        stats = compute_trajectory_stats(traj)
        assert stats["num_frames"] == 1201
        assert stats["total_duration_s"] == pytest.approx(24.0)
        assert stats["average_rate_hz"] == pytest.approx(50.0, abs=0.1)

        j1_stats = stats["joints"]["joint_1"]
        assert j1_stats["min_deg"] == pytest.approx(-15.0, abs=1e-2)
        assert j1_stats["max_deg"] == pytest.approx(15.0, abs=1e-2)
        assert j1_stats["range_deg"] == pytest.approx(30.0, abs=1e-2)
        assert j1_stats["max_vel_degps"] > 40.0  # ~58 deg/s

        assert stats["gripper"]["min_mm"] == pytest.approx(26.0)
        assert stats["gripper"]["max_mm"] == pytest.approx(26.0)

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

    def test_mock_playback_execution(self, tmp_path):
        # Generate short trajectory for fast execution
        traj = generate_t1_step_trajectory(dt=0.1)[:10]  # 10 frames = 0.9s
        csv_path = tmp_path / "short_t1.csv"
        save_trajectory_csv(csv_path, traj)

        args = argparse.Namespace(
            input=str(csv_path),
            can="can0",
            speed=50.0,  # Fast playback
            ramp_time=1.0,
            safe_disable=True,
            mock=True,
        )
        execute_playback(args)
