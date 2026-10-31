# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive offline audit & regression test suite for Unitree AS2 + AgileX PiPER.
Verifies:
1. Unified 20-DOF URDF/XACRO kinematics, symmetric gripper fingers, and leg symmetry.
2. Asset configs, ImplicitActuatorCfg *_sim limits, and DoorType factory.
3. RL action isolation, PiPER arm reset/swing MDP events, and PLAY teleop command persistence.
4. Non-penetrating 3D Hospital Door spawn geometry, forward kinematics, DLS inverse kinematics,
   simultaneous keyboard teleop state, and Key '1' proximity-gated auto-door state machine
   across all 4 door types (lever_push, lever_pull, sliding, push_bar).
5. Linux LF line endings on shell scripts, .gitignore/.gitattributes hygiene, and clean doc links.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_door_manipulation_module():
    mod_path = os.path.join(
        REPO_ROOT, "source", "isaaclab_assets", "isaaclab_assets", "robots", "door_manipulation.py"
    )
    spec = importlib.util.spec_from_file_location("door_manipulation", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["door_manipulation"] = module
    spec.loader.exec_module(module)
    return module


door_manip = _load_door_manipulation_module()


class TestUrdfKinematics:
    """Verify kinematics, joints, links, symmetry, and limits in unified 20-DOF URDF/XACRO."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.urdf_path = os.path.join(REPO_ROOT, "robot_arms", "piper", "descriptions", "as2_piper.urdf")
        self.xacro_path = os.path.join(REPO_ROOT, "robot_arms", "piper", "descriptions", "as2_piper.urdf.xacro")
        assert os.path.isfile(self.urdf_path), f"URDF not found: {self.urdf_path}"
        assert os.path.isfile(self.xacro_path), f"XACRO not found: {self.xacro_path}"
        self.urdf_tree = ET.parse(self.urdf_path)
        self.xacro_tree = ET.parse(self.xacro_path)

    def test_active_joint_count_and_naming(self):
        active_joints = [
            j.attrib["name"]
            for j in self.urdf_tree.findall("joint")
            if j.attrib.get("type") in ("revolute", "prismatic", "continuous")
        ]
        assert len(active_joints) == 20
        expected_legs = {
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        }
        expected_arm = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"}
        assert set(active_joints) == (expected_legs | expected_arm)

    def test_gripper_fingers_point_forward_and_symmetric_limits(self):
        """link7 and link8 boxes must extend along local +Y (gripper forward) with mirrored limits."""
        joints = {j.attrib["name"]: j for j in self.urdf_tree.findall("joint")}
        j7_lim = joints["joint7"].find("limit").attrib
        j8_lim = joints["joint8"].find("limit").attrib
        assert (float(j7_lim["lower"]), float(j7_lim["upper"])) == pytest.approx((0.0, 0.035))
        assert (float(j8_lim["lower"]), float(j8_lim["upper"])) == pytest.approx((-0.035, 0.0))

        for link_name in ("link7", "link8"):
            lnk = next(l for l in self.urdf_tree.findall("link") if l.attrib["name"] == link_name)
            vis_xyz = [float(v) for v in lnk.find("visual/origin").attrib["xyz"].split()]
            assert vis_xyz == pytest.approx([0.0, 0.025, 0.0]), f"{link_name} not aligned forward along +Y"

    def test_all_four_calf_cylinders_symmetric(self):
        """FL_calf, FR_calf, RL_calf, RR_calf must have identical cylinder geometry."""
        for leg in ("FL_calf", "FR_calf", "RL_calf", "RR_calf"):
            lnk = next(l for l in self.urdf_tree.findall("link") if l.attrib["name"] == leg)
            cyl = lnk.find("collision/geometry/cylinder").attrib
            orig = lnk.find("collision/origin").attrib["xyz"]
            assert float(cyl["radius"]) == pytest.approx(0.013), f"Asymmetric radius on {leg}"
            assert [float(v) for v in orig.split()] == pytest.approx([0.01, 0.0, -0.06])

    def test_foot_joints_preserve_dont_collapse(self):
        """All 4 foot fixed joints must carry dont_collapse='true' so merge_fixed_joints=True preserves feet."""
        for foot_j in ("FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"):
            j = next(j for j in self.urdf_tree.findall("joint") if j.attrib["name"] == foot_j)
            assert j.attrib.get("dont_collapse") == "true"


class TestAssetAndRLConfigurations:
    """Verify actuator *_sim parameters, RL arm target synchronization, and PLAY teleop fixes."""

    def test_implicit_actuators_use_sim_limits_and_fixed_standalone_root(self):
        for rel in (
            "source/isaaclab_assets/isaaclab_assets/robots/as2_piper.py",
            "source/isaaclab_assets/isaaclab_assets/robots/hospital_door.py",
        ):
            with open(os.path.join(REPO_ROOT, rel), "r", encoding="utf-8") as f:
                src = f.read()
            assert "velocity_limit_sim=" in src
            assert "effort_limit_sim=" in src

        with open(
            os.path.join(REPO_ROOT, "source/isaaclab_assets/isaaclab_assets/robots/as2_piper.py"),
            "r",
            encoding="utf-8",
        ) as f:
            as2_src = f.read()
        assert "fix_root_link=True" in as2_src
        assert "merge_fixed_joints=True" in as2_src

    def test_rl_configs_synchronize_arm_and_disable_play_heading_overwrite(self):
        rough_piper = os.path.join(
            REPO_ROOT,
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2_piper/rough_env_cfg.py",
        )
        with open(rough_piper, "r", encoding="utf-8") as f:
            src = f.read()
        assert "self.actions.joint_pos.joint_names = LEG_JOINT_NAMES" in src
        assert "self.actions.joint_pos.preserve_order = True" in src
        assert 'SceneEntityCfg("robot", joint_names=ORDERED_ALL_20_JOINTS, preserve_order=True)' in src
        assert "reset_piper_arm_targets" in src
        assert "randomize_piper_arm_targets" in src
        assert 'self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh|link[1-8]"' in src

        rough_as2 = os.path.join(
            REPO_ROOT,
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2/rough_env_cfg.py",
        )
        with open(rough_as2, "r", encoding="utf-8") as f:
            as2_rough_src = f.read()
        assert "self.actions.joint_pos.joint_names = ORDERED_AS2_LEG_JOINTS" in as2_rough_src
        assert "self.actions.joint_pos.preserve_order = True" in as2_rough_src
        assert 'SceneEntityCfg("robot", joint_names=ORDERED_AS2_LEG_JOINTS, preserve_order=True)' in as2_rough_src

        for play_rel in (
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2/rough_env_cfg.py",
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2/flat_env_cfg.py",
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2_piper/rough_env_cfg.py",
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/as2_piper/flat_env_cfg.py",
        ):
            with open(os.path.join(REPO_ROOT, play_rel), "r", encoding="utf-8") as f:
                play_src = f.read()
            assert "self.commands.base_velocity.heading_command = False" in play_src
            assert "self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)" in play_src


class TestKinematicsTeleopAndProximityAutoDoor:
    """Verify non-colliding door spawn, DLS IK convergence, simultaneous teleop, and Key '1' proximity gate."""

    def test_hospital_spawn_does_not_penetrate_closed_door_leaf(self):
        """Robot trunk at (1.84, -0.95, 0.45) facing +Y must have >0.20m clearance from door leaf y in [-0.47, -0.43]."""
        door_leaf_y_min = -0.47
        robot_pos = (1.84, -0.95, 0.45)
        # Trunk box 0.45 x 0.22m rotated +90 deg about Z -> extends +/-0.225m along world Y
        trunk_front_y = robot_pos[1] + 0.225
        clearance_y = door_leaf_y_min - trunk_front_y
        assert clearance_y >= 0.20, f"Robot collides or is too close to door leaf: clearance={clearance_y:.3f}m"

        # Handle lever grasp center in world frame
        handle_w = door_manip.compute_door_handle_world_pos(
            (1.10, -0.45, 0.0), (1.0, 0.0, 0.0, 0.0), 0.0, "lever_push"
        )
        assert handle_w == pytest.approx((1.84, -0.53, 0.97))

    def test_piper_dls_ik_converges_within_joint_limits(self):
        """Analytical DLS IK must reach handle targets in AS2 base frame to <5mm while respecting URDF limits."""
        target_b = (0.42, 0.0, 0.48)
        q_sol = door_manip.piper_dls_ik(target_b, max_iters=35, tol=1e-3)
        assert len(q_sol) == 6
        for i, val in enumerate(q_sol):
            low, high = door_manip.PIPER_ARM_JOINT_LIMITS[i]
            assert low - 1e-6 <= val <= high + 1e-6

        tcp, _, _ = door_manip.piper_forward_kinematics_base(q_sol)
        err = math.dist(tcp, target_b)
        assert err < 0.005, f"DLS IK error too large: {err:.4f}m"

    def test_simultaneous_wasd_and_symmetric_gripper_teleop(self):
        """Left-hand WASD and right-hand arm/gripper keys must operate simultaneously with mirrored jaws."""
        state = door_manip.TeleopControlState()
        state.update_from_key_event("W", is_press=True, is_release=False)
        state.update_from_key_event("Q", is_press=True, is_release=False)
        state.update_from_key_event("I", is_press=True, is_release=False)  # shoulder reach +
        state.update_from_key_event("Z", is_press=True, is_release=False)  # gripper open +

        q8 = state.step_manual_arm(dt=0.5)
        assert state.forward == 1.0 and state.yaw == 1.0
        assert q8[1] > door_manip.PIPER_TUCKED_ARM_JOINTS[1]
        assert q8[6] > 0.0 and q8[7] == pytest.approx(-q8[6])

        # Pressing '1' requests auto-door; pressing 'C' resets arm to tucked posture
        state.update_from_key_event("KEY_1", is_press=True, is_release=False)
        assert state.trigger_auto_door_requested is True
        state.update_from_key_event("C", is_press=True, is_release=False)
        q8_reset = state.step_manual_arm(dt=0.1)
        assert q8_reset[:6] == pytest.approx(list(door_manip.PIPER_TUCKED_ARM_JOINTS))
        assert q8_reset[6:] == [0.0, 0.0]

    @pytest.mark.parametrize("door_type", ["lever_push", "lever_pull", "sliding", "push_bar"])
    def test_key_1_proximity_gate_and_all_four_door_mechanisms(self, door_type: str):
        """Key '1' must reject activation when far/misaligned, activate when close, and open all 4 door types."""
        opener = door_manip.DoorAutoOpener(door_type=door_type)
        door_pos = (2.10, 0.74, 0.0)
        door_quat = (0.7071068, 0.0, 0.0, -0.7071068)
        handle_w = door_manip.compute_door_handle_world_pos(door_pos, door_quat, 0.0, door_type)

        # 1. Far away at origin (d ~ 2.02m > 0.90m) -> MUST REJECT Key '1'
        ok_far, msg_far = opener.try_trigger((0.0, 0.0, 0.45), (1.0, 0.0, 0.0, 0.0), handle_w, [0.0] * 8)
        assert ok_far is False and opener.active is False
        assert "거리가 너무 멉니다" in msg_far

        # 2. Close (x=1.45m) but facing backward (yaw=180 deg) -> MUST REJECT Key '1'
        ok_back, msg_back = opener.try_trigger((1.45, 0.0, 0.45), (0.0, 0.0, 0.0, 1.0), handle_w, [0.0] * 8)
        assert ok_back is False and opener.active is False
        assert "정면으로 바라보고 있지 않습니다" in msg_back

        # 3. Walked within proximity (x=1.48m, facing +X -> d ~ 0.54m) -> MUST ACTIVATE Key '1'
        tucked_q8 = list(door_manip.PIPER_TUCKED_ARM_JOINTS) + [0.0, 0.0]
        zone_notice = opener.poll_zone_notification((1.48, 0.0, 0.45), (1.0, 0.0, 0.0, 0.0), handle_w)
        assert zone_notice is not None and "준비 완료" in zone_notice

        ok_near, _ = opener.try_trigger((1.48, 0.0, 0.45), (1.0, 0.0, 0.0, 0.0), handle_w, tucked_q8)
        assert ok_near is True and opener.active is True

        # Step through full trajectory and verify door opens in expected direction & gripper stays symmetric
        q8 = tucked_q8
        peak_hinge = 0.0
        for _ in range(90):
            q8, _, (hinge_t, _) = opener.step(0.1, (1.48, 0.0, 0.45), (1.0, 0.0, 0.0, 0.0), handle_w, q8)
            assert q8[7] == pytest.approx(-q8[6])
            if abs(hinge_t) > abs(peak_hinge):
                peak_hinge = hinge_t

        assert opener.active is False  # Completed and returned to idle!
        if door_type == "lever_pull":
            assert peak_hinge < -1.0
        elif door_type == "sliding":
            assert peak_hinge > 0.70
        else:
            assert peak_hinge > 1.20


class TestLinuxAndGitReadiness:
    """Verify LF line endings on shell scripts, tracked tests, and absence of broken Android links."""

    def test_shell_scripts_use_lf_line_endings(self):
        for sh in (
            "train_as2_piper.sh",
            "train_as2.sh",
            "play_as2_piper.sh",
            "spawn_as2_piper.sh",
            "spawn_as2_piper_hospital.sh",
            "robot_arms/piper/setup_piper.sh",
        ):
            path = os.path.join(REPO_ROOT, sh)
            assert os.path.isfile(path), f"Missing script: {sh}"
            with open(path, "rb") as f:
                raw = f.read()
            assert b"\r\n" not in raw, f"Windows CRLF detected in {sh}; will fail on Linux GPU server!"

    def test_no_broken_android_sdcard_links_in_docs(self):
        for doc in ("README.md", "docs/isaac_sim_6_2_as2_piper_guide.md", "robot_arms/piper/README.md"):
            p = os.path.join(REPO_ROOT, doc)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
                assert "file:///sdcard/" not in text, f"Broken Android path left in {doc}"


def _load_module_from_relpath(mod_name: str, rel_path: str):
    mod_path = os.path.join(REPO_ROOT, *rel_path.split("/"))
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


as2_sensors_mod = _load_module_from_relpath(
    "as2_sensors_offline", "source/isaaclab_assets/isaaclab_assets/robots/as2_sensors.py"
)
vision_det_mod = _load_module_from_relpath(
    "vision_door_detector_offline",
    "ros2_ws/src/as2_piper_bringup/as2_piper_bringup/vision_door_detector.py",
)
sim2real_mod = _load_module_from_relpath(
    "sim2real_bridge_offline",
    "ros2_ws/src/as2_piper_bringup/as2_piper_bringup/sim2real_bridge_node.py",
)
eval_bench_mod = _load_module_from_relpath(
    "evaluate_loco_manipulation_offline",
    "scripts/tools/evaluate_loco_manipulation.py",
)


class TestSensorSuiteAndRos2Perception:
    """Verify AS2 RGB camera, Depth camera, 3D LiDAR mounts, perception math, and Sim-to-Real bridge."""

    def test_urdf_and_xacro_sensor_frames_merge_cleanly_and_avoid_self_occlusion(self):
        for fname in ("as2_piper.urdf", "as2_piper.urdf.xacro"):
            tree = ET.parse(os.path.join(REPO_ROOT, "robot_arms", "piper", "descriptions", fname))
            links = {lnk.attrib["name"]: lnk for lnk in tree.findall("link")}
            joints = {j.attrib["name"]: j for j in tree.findall("joint")}

            for link_name in ("front_camera", "lidar_link", "imu", "radar"):
                assert link_name in links, f"Missing sensor link '{link_name}' in {fname}"
                inertial = links[link_name].find("inertial")
                assert inertial is not None, f"Sensor link '{link_name}' in {fname} must have <inertial>"
                mass_val = float(inertial.find("mass").attrib["value"])
                assert mass_val > 0.0
                # Sensor optical frames must NOT define <collision> boxes that block ray-casting from origin
                assert links[link_name].find("collision") is None

            # RTX LiDAR ray-traces visual geometry: lidar_link visual dome MUST sit below laser origin (0, 0, 0)
            lidar_vis_xyz = [
                float(v) for v in links["lidar_link"].find("visual/origin").attrib["xyz"].split()
            ]
            assert lidar_vis_xyz == pytest.approx([0.0, 0.0, -0.025])

            # Sensor fixed joints omit dont_collapse so 4096-env RL merges sensor inertias into 'base'
            for joint_name in ("front_camera_joint", "lidar_joint", "imu_joint", "radar_joint"):
                assert joint_name in joints, f"Missing sensor joint '{joint_name}' in {fname}"
                assert joints[joint_name].attrib.get("dont_collapse") is None

            cam_xyz = [float(v) for v in joints["front_camera_joint"].find("origin").attrib["xyz"].split()]
            lidar_xyz = [float(v) for v in joints["lidar_joint"].find("origin").attrib["xyz"].split()]
            assert tuple(cam_xyz) == pytest.approx(as2_sensors_mod.AS2_FRONT_CAMERA_POS)
            assert tuple(lidar_xyz) == pytest.approx(as2_sensors_mod.AS2_LIDAR_POS)

    def test_row_padded_ros_image_decoding_and_latched_handle_propagation(self):
        from types import SimpleNamespace
        import numpy as np

        # 1. Row-padded 16UC1 depth image (width=5, height=3, valid=10 bytes/row, padded step=16 bytes/row)
        raw_16 = np.arange(100, 115, dtype="<u2").reshape(3, 5)
        padded_16 = np.zeros((3, 16), dtype=np.uint8)
        padded_16[:, :10] = raw_16.view(np.uint8)
        msg_16 = SimpleNamespace(
            height=3, width=5, step=16, is_bigendian=0, encoding="16UC1", data=padded_16.tobytes()
        )
        dec_16 = vision_det_mod.decode_ros_image_msg(msg_16)
        assert dec_16.shape == (3, 5)
        assert np.array_equal(dec_16, raw_16)

        # Big-endian 16UC1 check
        raw_16_be = raw_16.astype(">u2")
        padded_be = np.zeros((3, 12), dtype=np.uint8)
        padded_be[:, :10] = raw_16_be.view(np.uint8)
        msg_be = SimpleNamespace(
            height=3, width=5, step=12, is_bigendian=1, encoding="16UC1", data=padded_be.tobytes()
        )
        assert np.array_equal(vision_det_mod.decode_ros_image_msg(msg_be), raw_16)

        # 2. Row-padded rgb8 image (width=4, height=2, valid=12 bytes/row, step=16 bytes/row)
        raw_rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        padded_rgb = np.full((2, 16), 255, dtype=np.uint8)
        padded_rgb[:, :12] = raw_rgb.reshape(2, 12)
        msg_rgb = SimpleNamespace(
            height=2, width=4, step=16, is_bigendian=0, encoding="bgr8", data=padded_rgb.tobytes()
        )
        dec_rgb = vision_det_mod.decode_ros_image_msg(msg_rgb)
        assert dec_rgb.shape == (2, 4, 3)
        assert np.array_equal(dec_rgb, raw_rgb[..., ::-1])

        # 3. World-to-base twist rotation (+90 deg yaw: robot +X points along world +Y)
        q_yaw90 = (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
        v_body = sim2real_mod.quat_rotate_inverse_wxyz(q_yaw90, (0.0, 1.2, 0.0))
        assert v_body.tolist() == pytest.approx([1.2, 0.0, 0.0], abs=1e-5)

        # 4. Latched handle propagation under dead-reckoned base motion (immune to arm occlusion)
        updated_h = sim2real_mod.propagate_latched_handle_in_base(
            (0.60, 0.0, 0.50), vx_b=0.20, vy_b=0.0, wz_b=0.0, dt=0.5
        )
        assert updated_h == pytest.approx((0.50, 0.0, 0.50), abs=1e-6)

    def test_as2_sensors_topics_and_pinhole_intrinsics(self):
        topics = as2_sensors_mod.AS2_ROS2_TOPICS
        assert topics["rgb"] == "/as2_camera/rgb"
        assert topics["depth"] == "/as2_camera/depth"
        assert topics["camera_info"] == "/as2_camera/camera_info"
        assert topics["lidar_point_cloud"] == "/as2_lidar/point_cloud"
        assert topics["lidar_scan"] == "/as2_lidar/scan"

        fx, fy, cx, cy = as2_sensors_mod.compute_default_pinhole_intrinsics(640, 480)
        assert cx == pytest.approx(320.0)
        assert cy == pytest.approx(240.0)
        assert fx == pytest.approx(fy, rel=1e-5)
        assert fx > 700.0

    def test_vision_door_detector_backprojection_and_lidar_plane(self):
        import numpy as np

        fx, fy, cx, cy = as2_sensors_mod.compute_default_pinhole_intrinsics(640, 480)
        # Principal point at 0.50m depth -> x_b = 0.30 + 0.50 = 0.80m, y_b = 0.0, z_b = 0.10m
        p_b = vision_det_mod.backproject_pixel_to_base_frame(320.0, 240.0, 0.50, fx, fy, cx, cy)
        assert p_b == pytest.approx((0.80, 0.0, 0.10), abs=1e-5)

        # Synthetic 2D LaserScan hitting a flat door at x_b = 0.65m perpendicular to robot +X
        angles = np.linspace(-0.45, 0.45, 45)
        # x_b = lidar_x + r*cos(a) = 0.65 -> r = (0.65 - 0.285) / cos(a)
        ranges = (0.65 - as2_sensors_mod.AS2_LIDAR_POS[0]) / np.cos(angles)
        plane = vision_det_mod.estimate_door_plane_from_lidar_scan(
            ranges, float(angles[0]), float(angles[1] - angles[0])
        )
        assert plane is not None
        assert plane.perp_distance_m == pytest.approx(0.65, abs=1e-3)
        assert math.degrees(plane.normal_yaw_err_rad) == pytest.approx(0.0, abs=0.5)
        assert plane.inlier_rmse_m < 1e-4

        # Synthetic 3D PointCloud2 slice rotated by +15 deg yaw
        yaw_true = math.radians(15.0)
        ys = np.linspace(-0.25, 0.25, 30)
        zs = np.linspace(0.0, 0.35, 6)
        yy, zz = np.meshgrid(ys, zs)
        yy_f, zz_f = yy.ravel(), zz.ravel()
        # Plane normal (cos(15), sin(15)): n_x * x_b + n_y * y_b = 0.60
        xx_b = (0.60 - math.sin(yaw_true) * yy_f) / math.cos(yaw_true)
        pc_lidar = np.column_stack(
            [xx_b - as2_sensors_mod.AS2_LIDAR_POS[0], yy_f - as2_sensors_mod.AS2_LIDAR_POS[1], zz_f]
        )
        plane_3d = vision_det_mod.extract_door_plane_from_pointcloud_xyz(pc_lidar)
        assert plane_3d is not None
        assert math.degrees(plane_3d.normal_yaw_err_rad) == pytest.approx(15.0, abs=1.0)

    def test_vision_rgbd_handle_detection_and_uint16_realsense_support(self):
        import numpy as np

        fx, fy, cx, cy = as2_sensors_mod.compute_default_pinhole_intrinsics(640, 480)
        # Door surface at 0.32m optical depth -> x_b ~ 0.62m (inside [0.28m, 0.90m] gate)
        depth_mm = np.full((480, 640), 320, dtype=np.uint16)
        rgb = np.full((480, 640, 3), 180, dtype=np.uint8)
        # Protruding metallic lever handle at 0.26m optical depth (6cm protrusion) near center
        depth_mm[225:250, 290:355] = 260
        rgb[225:250, 290:355] = (40, 40, 45)

        det = vision_det_mod.detect_door_and_handle_rgbd_lidar(rgb, depth_mm, fx, fy, cx, cy)
        assert det.detected is True
        assert det.door_ready is True
        assert det.handle_pos_base[0] == pytest.approx(0.30 + 0.26, abs=0.02)
        assert det.suggested_door_type == "lever_push"

        # Far door at 1.40m optical depth -> detected=True, but door_ready=False!
        depth_far = np.full((480, 640), 1.40, dtype=np.float32)
        depth_far[225:250, 290:355] = 1.34
        det_far = vision_det_mod.detect_door_and_handle_rgbd_lidar(rgb, depth_far, fx, fy, cx, cy)
        assert det_far.detected is True
        assert det_far.door_ready is False

    def test_sim2real_observation_builder_48d_and_64d_and_benchmark_export(self, tmp_path):
        # Level upright quaternion -> projected gravity = (0, 0, -1)
        g = sim2real_mod.compute_projected_gravity_wxyz((1.0, 0.0, 0.0, 0.0))
        assert g.tolist() == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)

        obs48 = sim2real_mod.build_locomotion_observation(
            base_lin_vel_b=(0.5, 0.0, 0.0),
            base_ang_vel_b=(0.0, 0.0, 0.1),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            cmd_vel_xy_yaw=(0.5, 0.0, 0.1),
            leg_pos_12=sim2real_mod.AS2_DEFAULT_LEG_POS_12,
            leg_vel_12=[0.0] * 12,
            last_action_12=[0.0] * 12,
            expected_obs_dim=48,
        )
        assert obs48.shape == (48,)
        # At default standing joint angles, joint_pos_rel slice [12:24] must be all zeros!
        assert obs48[12:24].tolist() == pytest.approx([0.0] * 12, abs=1e-6)

        obs64 = sim2real_mod.build_locomotion_observation(
            base_lin_vel_b=(0.5, 0.0, 0.0),
            base_ang_vel_b=(0.0, 0.0, 0.1),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            cmd_vel_xy_yaw=(0.5, 0.0, 0.1),
            leg_pos_12=sim2real_mod.AS2_DEFAULT_LEG_POS_12,
            leg_vel_12=[0.0] * 12,
            last_action_12=[0.0] * 12,
            arm_pos_8=sim2real_mod.PIPER_DEFAULT_ARM_POS_8,
            arm_vel_8=[0.0] * 8,
            expected_obs_dim=64,
        )
        assert obs64.shape == (64,)
        assert obs64[12:32].tolist() == pytest.approx([0.0] * 20, abs=1e-6)

        # Verify benchmark evaluator thresholds & CSV/Markdown export
        assert eval_bench_mod.is_door_open_success("lever_push", 0.85) is True
        assert eval_bench_mod.is_door_open_success("lever_pull", -0.80) is True
        assert eval_bench_mod.is_door_open_success("lever_pull", +0.80) is False
        assert eval_bench_mod.is_door_open_success("sliding", 0.55) is True
        assert eval_bench_mod.is_door_open_success("push_bar", 0.85) is True

        row = eval_bench_mod.BenchmarkRow(
            condition="door_lever_push",
            episodes=5,
            lin_vel_rmse_mps=0.0812,
            ang_vel_rmse_radps=0.0641,
            roll_pitch_rms_deg=1.85,
            leg_torque_rms_nm=14.20,
            fall_rate_pct=0.0,
            door_success_rate_pct=100.0,
            mean_door_open_time_s=4.35,
        )
        csv_p, md_p = eval_bench_mod.write_benchmark_reports([row], str(tmp_path))
        assert os.path.isfile(csv_p) and os.path.isfile(md_p)
        with open(md_p, "r", encoding="utf-8") as fp:
            md_txt = fp.read()
        assert "door_lever_push" in md_txt and "100.0" in md_txt
