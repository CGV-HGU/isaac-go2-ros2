# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for AS2 + AgileX PiPER Sim-to-Real whole-body bridge logic."""

from __future__ import annotations

import math
import numpy as np
import pytest

from as2_piper_bringup.sim2real_bridge_node import (
    AS2_DEFAULT_LEG_POS_12,
    AS2_LEG_JOINT_NAMES,
    PIPER_ARM_JOINT_NAMES,
    PIPER_DEFAULT_ARM_POS_8,
    actions_to_leg_joint_targets,
    build_locomotion_observation,
    compute_projected_gravity_wxyz,
    load_door_manipulation_module,
    propagate_latched_handle_in_base,
    quat_rotate_inverse_wxyz,
)


class TestQuaternionAndCoordinateMath:
    """Test unit quaternion transformations and projected gravity computations."""

    def test_identity_quaternion_preserves_vector(self):
        vec = (1.5, -2.0, 3.25)
        rot = quat_rotate_inverse_wxyz((1.0, 0.0, 0.0, 0.0), vec)
        assert rot.tolist() == pytest.approx(list(vec), abs=1e-6)

    def test_90_deg_yaw_rotation(self):
        # 90 deg counter-clockwise yaw around +Z: q = [cos(45°), 0, 0, sin(45°)]
        s2 = math.sqrt(2.0) / 2.0
        q_yaw_90 = (s2, 0.0, 0.0, s2)
        # Vector pointing world +X
        v_world = (1.0, 0.0, 0.0)
        # In body frame (which rotated +90 deg), the world vector points towards body -Y
        v_body = quat_rotate_inverse_wxyz(q_yaw_90, v_world)
        assert v_body.tolist() == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)

    def test_zero_quaternion_raises_value_error(self):
        with pytest.raises(ValueError, match="Quaternion norm is zero"):
            quat_rotate_inverse_wxyz((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_projected_gravity_upright_and_inverted(self):
        # Upright: gravity points down along body -Z
        g_up = compute_projected_gravity_wxyz((1.0, 0.0, 0.0, 0.0))
        assert g_up.tolist() == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

        # Upside down (180 deg roll around +X): q = [0, 1, 0, 0]
        g_inverted = compute_projected_gravity_wxyz((0.0, 1.0, 0.0, 0.0))
        assert g_inverted.tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

        # Pitch 90 deg nose-down (around body +Y): q = [cos(45°), 0, sin(45°), 0]
        s2 = math.sqrt(2.0) / 2.0
        g_pitch = compute_projected_gravity_wxyz((s2, 0.0, s2, 0.0))
        assert g_pitch.tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


class TestHandleTargetDeadReckoning:
    """Test planar propagation of latched door handle target during base motion."""

    def test_static_base_preserves_handle_position(self):
        h0 = (0.65, 0.10, 0.25)
        h1 = propagate_latched_handle_in_base(h0, vx_b=0.0, vy_b=0.0, wz_b=0.0, dt=0.02)
        assert h1 == pytest.approx(h0)

    def test_forward_translation_shifts_handle_closer(self):
        h0 = (0.70, 0.0, 0.20)
        # Base moves forward at 0.5 m/s for 0.1 s -> 0.05m forward
        h1 = propagate_latched_handle_in_base(h0, vx_b=0.5, vy_b=0.0, wz_b=0.0, dt=0.10)
        assert h1[0] == pytest.approx(0.65, abs=1e-5)
        assert h1[1] == pytest.approx(0.0, abs=1e-5)
        assert h1[2] == pytest.approx(0.20, abs=1e-5)

    def test_yaw_rotation_shifts_handle_in_base(self):
        h0 = (0.50, 0.0, 0.30)
        # Base turns left (positive wz) at 0.5 rad/s for 0.2 s -> yaw = 0.1 rad
        dt = 0.2
        wz = 0.5
        dtheta = wz * dt
        h1 = propagate_latched_handle_in_base(h0, vx_b=0.0, vy_b=0.0, wz_b=wz, dt=dt)
        expected_x = 0.50 * math.cos(-dtheta)
        expected_y = 0.50 * math.sin(-dtheta)
        assert h1[0] == pytest.approx(expected_x, abs=1e-5)
        assert h1[1] == pytest.approx(expected_y, abs=1e-5)
        assert h1[2] == pytest.approx(0.30, abs=1e-5)


class TestObservationBuilderAndPolicyInterface:
    """Test deterministic observation tensor construction for 48-D (AS2) and 64-D (AS2+PiPER)."""

    def test_observation_dimensions_and_default_neutral_slice(self):
        # 48-D AS2 Solo
        obs48 = build_locomotion_observation(
            base_lin_vel_b=(0.2, 0.0, 0.0),
            base_ang_vel_b=(0.0, 0.0, 0.05),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            cmd_vel_xy_yaw=(0.5, 0.0, 0.0),
            leg_pos_12=AS2_DEFAULT_LEG_POS_12,
            leg_vel_12=[0.0] * 12,
            last_action_12=[0.0] * 12,
            expected_obs_dim=48,
        )
        assert obs48.shape == (48,)
        # Relative leg position slice [12:24] must be exactly zero at default standing angles
        assert obs48[12:24].tolist() == pytest.approx([0.0] * 12, abs=1e-6)

        # 64-D AS2 + PiPER Mobile Manipulator
        obs64 = build_locomotion_observation(
            base_lin_vel_b=(0.2, 0.0, 0.0),
            base_ang_vel_b=(0.0, 0.0, 0.05),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            cmd_vel_xy_yaw=(0.5, 0.0, 0.0),
            leg_pos_12=AS2_DEFAULT_LEG_POS_12,
            leg_vel_12=[0.0] * 12,
            last_action_12=[0.0] * 12,
            arm_pos_8=PIPER_DEFAULT_ARM_POS_8,
            arm_vel_8=[0.0] * 8,
            expected_obs_dim=64,
        )
        assert obs64.shape == (64,)
        # Relative joints slice [12:32] (12 leg + 8 arm) must be exactly zero at neutral pose
        assert obs64[12:32].tolist() == pytest.approx([0.0] * 20, abs=1e-6)

    def test_observation_builder_validates_dimensions(self):
        with pytest.raises(ValueError, match="Unsupported policy obs dimension"):
            build_locomotion_observation(
                base_lin_vel_b=(0.0, 0.0, 0.0),
                base_ang_vel_b=(0.0, 0.0, 0.0),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                cmd_vel_xy_yaw=(0.0, 0.0, 0.0),
                leg_pos_12=[0.0] * 12,
                leg_vel_12=[0.0] * 12,
                last_action_12=[0.0] * 12,
                expected_obs_dim=50,
            )

        with pytest.raises(ValueError, match="leg_pos_12 must have length 12"):
            build_locomotion_observation(
                base_lin_vel_b=(0.0, 0.0, 0.0),
                base_ang_vel_b=(0.0, 0.0, 0.0),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                cmd_vel_xy_yaw=(0.0, 0.0, 0.0),
                leg_pos_12=[0.0] * 10,  # Invalid count
                leg_vel_12=[0.0] * 12,
                last_action_12=[0.0] * 12,
                expected_obs_dim=48,
            )

    def test_actions_to_targets_scaling_and_clipping(self):
        zeros_12 = [0.0] * 12
        targets, clipped = actions_to_leg_joint_targets(zeros_12)
        assert targets.tolist() == pytest.approx(list(AS2_DEFAULT_LEG_POS_12))
        assert clipped.tolist() == pytest.approx(zeros_12)

        raw = [200.0] * 12
        targets_clip, clipped_act = actions_to_leg_joint_targets(raw, action_scale=0.25, action_clip=50.0)
        assert clipped_act.tolist() == pytest.approx([50.0] * 12)
        expected = np.array(AS2_DEFAULT_LEG_POS_12) + 0.25 * 50.0
        assert targets_clip.tolist() == pytest.approx(expected.tolist())

    def test_joint_naming_constants(self):
        assert len(AS2_LEG_JOINT_NAMES) == 12
        assert len(AS2_DEFAULT_LEG_POS_12) == 12
        assert len(PIPER_ARM_JOINT_NAMES) == 8
        assert len(PIPER_DEFAULT_ARM_POS_8) == 8

    def test_door_manipulation_module_resolution(self):
        mod = load_door_manipulation_module()
        assert hasattr(mod, "DoorAutoOpener")
        assert hasattr(mod, "DOOR_TYPES")
        assert "lever_push" in mod.DOOR_TYPES

    def test_package_init_exports(self):
        import as2_piper_bringup as pkg
        assert hasattr(pkg, "build_locomotion_observation")
        assert hasattr(pkg, "detect_door_and_handle_rgbd_lidar")
        assert hasattr(pkg, "decode_pointcloud2_xyz")
        assert hasattr(pkg, "AS2_DEFAULT_LEG_POS_12")
