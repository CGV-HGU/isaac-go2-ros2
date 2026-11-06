# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for AS2 RGB-D + 3D LiDAR door and handle perception algorithms."""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace
import numpy as np
import pytest

from as2_piper_bringup.vision_door_detector import (
    DEFAULT_CAM_POS_BASE,
    DEFAULT_LIDAR_POS_BASE,
    backproject_pixel_to_base_frame,
    decode_pointcloud2_xyz,
    decode_ros_image_msg,
    detect_door_and_handle_rgbd_lidar,
    estimate_door_plane_from_lidar_scan,
    extract_door_plane_from_pointcloud_xyz,
    fit_door_plane_points_xy,
    normalize_depth_to_meters,
)


class TestBackprojectionAndDepthNormalization:
    """Test camera pinhole geometry, pitch rotation, and metric depth conversion."""

    def test_backproject_center_pixel_level_camera(self):
        # Principal point (cx, cy) = (320, 240) at 1.0m optical depth
        pt = backproject_pixel_to_base_frame(
            u=320.0,
            v=240.0,
            depth_m=1.0,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            cam_pos_b=(0.30, 0.0, 0.10),
            cam_pitch_rad=0.0,
        )
        # z_opt = 1.0 forward -> base x = 0.30 + 1.0 = 1.30, y = 0.0, z = 0.10
        assert pt[0] == pytest.approx(1.30, abs=1e-5)
        assert pt[1] == pytest.approx(0.0, abs=1e-5)
        assert pt[2] == pytest.approx(0.10, abs=1e-5)

    def test_backproject_invalid_depth_and_focal_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="depth_m must be finite and positive"):
            backproject_pixel_to_base_frame(100, 100, depth_m=-0.5, fx=500, fy=500, cx=320, cy=240)
        with pytest.raises(ValueError, match="depth_m must be finite and positive"):
            backproject_pixel_to_base_frame(100, 100, depth_m=float("nan"), fx=500, fy=500, cx=320, cy=240)
        with pytest.raises(ValueError, match="Camera focal lengths must be positive"):
            backproject_pixel_to_base_frame(100, 100, depth_m=1.0, fx=0.0, fy=500, cx=320, cy=240)

    def test_normalize_depth_to_meters(self):
        # 32FC1 meters pass-through
        depth_f32 = np.array([[1.25, 2.50]], dtype=np.float32)
        norm_f32 = normalize_depth_to_meters(depth_f32)
        assert norm_f32.dtype == np.float32
        np.testing.assert_allclose(norm_f32, [[1.25, 2.50]], rtol=1e-5)

        # 16UC1 millimeters scaled by 0.001
        depth_u16 = np.array([[1250, 2500]], dtype=np.uint16)
        norm_u16 = normalize_depth_to_meters(depth_u16)
        assert norm_u16.dtype == np.float32
        np.testing.assert_allclose(norm_u16, [[1.25, 2.50]], rtol=1e-5)

        # Invalid 3D depth image with >1 channels
        with pytest.raises(ValueError, match="Expected 2D depth image"):
            normalize_depth_to_meters(np.zeros((10, 10, 3)))


class TestDoorPlaneFitting:
    """Test SVD 2D/3D door plane estimation from 2D LaserScan and 3D PointCloud2."""

    def test_plane_fitting_insufficient_points_or_span_returns_none(self):
        # Fewer than 5 points
        assert fit_door_plane_points_xy(np.zeros((3, 2))) is None

        # Lateral span < 8cm
        narrow_pts = np.array([[0.5, 0.0], [0.5, 0.01], [0.5, 0.02], [0.5, 0.03], [0.5, 0.04]])
        assert fit_door_plane_points_xy(narrow_pts) is None

    def test_straight_wall_scan_fits_perpendicular_plane(self):
        # Straight wall at x_b = 0.80m, lateral span from -0.30 to +0.30m
        angles = np.linspace(-0.35, 0.35, 41)
        # In lidar frame (lidar_x = 0.285): x_w = 0.80 -> lidar x = 0.80 - 0.285 = 0.515
        dx_lidar = 0.80 - DEFAULT_LIDAR_POS_BASE[0]
        ranges = dx_lidar / np.cos(angles)

        plane = estimate_door_plane_from_lidar_scan(
            ranges=ranges,
            angle_min=angles[0],
            angle_increment=angles[1] - angles[0],
            lidar_pos_b=DEFAULT_LIDAR_POS_BASE,
        )
        assert plane is not None
        assert plane.perp_distance_m == pytest.approx(0.80, abs=1e-3)
        assert plane.normal_yaw_err_rad == pytest.approx(0.0, abs=1e-3)
        assert plane.inlier_rmse_m < 1e-4

    def test_pointcloud_plane_extraction_with_yaw_tilt(self):
        # 3D points rotated by +10 deg yaw
        yaw_rad = math.radians(10.0)
        ys = np.linspace(-0.25, 0.25, 25)
        zs = np.linspace(0.10, 0.40, 5)
        yy, zz = np.meshgrid(ys, zs)
        yy_f, zz_f = yy.ravel(), zz.ravel()
        # Wall at x_b ~ 0.70m with 10 deg normal: n_x * x + n_y * y = 0.70
        xx_b = (0.70 - math.sin(yaw_rad) * yy_f) / math.cos(yaw_rad)
        pts_lidar = np.column_stack(
            [xx_b - DEFAULT_LIDAR_POS_BASE[0], yy_f - DEFAULT_LIDAR_POS_BASE[1], zz_f]
        )
        plane = extract_door_plane_from_pointcloud_xyz(pts_lidar, lidar_pos_b=DEFAULT_LIDAR_POS_BASE)
        assert plane is not None
        assert math.degrees(plane.normal_yaw_err_rad) == pytest.approx(10.0, abs=1.0)


class TestImageAndPointCloudDecoders:
    """Test ROS Image and PointCloud2 decoders with strides, padding, and byte-orders."""

    def test_decode_image_32fc1(self):
        data = np.array([[0.5, 1.2], [2.1, 3.4]], dtype=np.float32)
        msg = SimpleNamespace(
            encoding="32FC1",
            height=2,
            width=2,
            step=8,
            is_bigendian=False,
            data=data.tobytes(),
        )
        decoded = decode_ros_image_msg(msg)
        assert decoded.shape == (2, 2)
        assert decoded.dtype == np.float32
        np.testing.assert_allclose(decoded, data, rtol=1e-5)

    def test_decode_image_16uc1_and_row_padding(self):
        raw_row = np.array([500, 1200], dtype=np.uint16).tobytes() + b"\x00\x00\x00\x00"  # 4 bytes padding
        msg = SimpleNamespace(
            encoding="16UC1",
            height=1,
            width=2,
            step=len(raw_row),  # 8 bytes step instead of 4
            is_bigendian=False,
            data=raw_row,
        )
        decoded = decode_ros_image_msg(msg)
        assert decoded.shape == (1, 2)
        assert decoded.dtype == np.uint16
        assert decoded.tolist() == [[500, 1200]]

    def test_decode_image_mono8(self):
        data = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        msg = SimpleNamespace(
            encoding="mono8",
            height=2,
            width=2,
            step=2,
            is_bigendian=False,
            data=data.tobytes(),
        )
        decoded = decode_ros_image_msg(msg)
        assert decoded.shape == (2, 2)
        assert decoded.dtype == np.uint8
        assert decoded.tolist() == data.tolist()

    def test_decode_image_rgb8_and_bgr8(self):
        # Color pixel [R=255, G=128, B=64]
        rgb = np.array([[[255, 128, 64]]], dtype=np.uint8)
        msg_rgb = SimpleNamespace(
            encoding="rgb8",
            height=1,
            width=1,
            step=3,
            is_bigendian=False,
            data=rgb.tobytes(),
        )
        assert decode_ros_image_msg(msg_rgb).tolist() == [[[255, 128, 64]]]

        bgr = np.array([[[64, 128, 255]]], dtype=np.uint8)
        msg_bgr = SimpleNamespace(
            encoding="bgr8",
            height=1,
            width=1,
            step=3,
            is_bigendian=False,
            data=bgr.tobytes(),
        )
        # BGR must be flipped to RGB
        assert decode_ros_image_msg(msg_bgr).tolist() == [[[255, 128, 64]]]

    def test_decode_pointcloud2_xyz(self):
        # Create mock PointCloud2 with 3 points
        points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        buf = bytearray()
        for x, y, z in points:
            buf.extend(struct.pack("<fff", x, y, z))

        fields = [
            SimpleNamespace(name="x", offset=0),
            SimpleNamespace(name="y", offset=4),
            SimpleNamespace(name="z", offset=8),
        ]
        msg = SimpleNamespace(
            fields=fields,
            width=3,
            height=1,
            point_step=12,
            is_bigendian=False,
            data=bytes(buf),
        )
        pts = decode_pointcloud2_xyz(msg)
        assert pts.shape == (3, 3)
        np.testing.assert_allclose(pts, points, rtol=1e-5)

        # Edge cases: max_points <= 0 must safely return at most 1 point without dividing by zero
        pts_zero = decode_pointcloud2_xyz(msg, max_points=0)
        assert pts_zero.shape == (1, 3)
        pts_neg = decode_pointcloud2_xyz(msg, max_points=-5)
        assert pts_neg.shape == (1, 3)

        # Missing fields / empty data
        msg_no_fields = SimpleNamespace(fields=[], width=3, height=1, point_step=12, data=b"")
        assert decode_pointcloud2_xyz(msg_no_fields).shape == (0, 3)


class TestFullPerceptionGateAndTypeClassification:
    """Test RGB-D handle localization, Key '1' proximity gating, and door type classification."""

    def test_door_ready_gate_and_lever_handle(self):
        h, w = 480, 640
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0

        # Background door surface at 0.35m depth -> base X = 0.30 + 0.35 = 0.65m (inside [0.28m, 0.90m])
        depth = np.full((h, w), 0.35, dtype=np.float32)
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        # Metallic lever handle protruding by 5cm (depth = 0.30m) near image center
        depth[230:250, 300:340] = 0.30
        rgb[230:250, 300:340] = 30

        res = detect_door_and_handle_rgbd_lidar(rgb, depth, fx=fx, fy=fy, cx=cx, cy=cy)
        assert res.detected is True
        assert res.door_ready is True
        assert res.suggested_door_type == "lever_push"
        assert res.handle_pos_base[0] == pytest.approx(0.60, abs=0.03)

    def test_far_door_is_detected_but_not_ready(self):
        h, w = 480, 640
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0

        # Door at 1.50m (far away)
        depth = np.full((h, w), 1.50, dtype=np.float32)
        rgb = np.full((h, w, 3), 180, dtype=np.uint8)
        depth[230:250, 300:340] = 1.45

        res = detect_door_and_handle_rgbd_lidar(rgb, depth, fx=fx, fy=fy, cx=cx, cy=cy)
        assert res.detected is True
        assert res.door_ready is False
        assert "더 접근 필요" in res.reason
