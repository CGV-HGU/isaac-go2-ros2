# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RGB + Depth + 3D LiDAR Door & Handle Perception Node for Unitree AS2 + AgileX PiPER.

Features:
- Stride-aware & endian-aware ``sensor_msgs/Image`` decoding (supports row-padded RealSense ``16UC1``
  millimeter streams and Isaac Sim ``32FC1`` meter streams without ``cv_bridge``).
- Parameterized camera & LiDAR extrinsics (including pitch tilt) + SVD 2D/3D door plane fitting.
- Timestamp freshness expiry on cached RGB and LiDAR plane measurements (no stale detections).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time
from typing import Sequence

import numpy as np

DEFAULT_CAM_POS_BASE: tuple[float, float, float] = (0.30, 0.0, 0.10)
DEFAULT_LIDAR_POS_BASE: tuple[float, float, float] = (0.285, 0.0, 0.045)


@dataclass(frozen=True)
class LidarDoorPlaneEstimate:
    """Planar wall/door fit extracted from 2D LaserScan or 3D PointCloud2 slice in AS2 `base` frame."""

    perp_distance_m: float
    normal_yaw_err_rad: float
    inlier_rmse_m: float
    num_points: int


@dataclass(frozen=True)
class DoorDetectionResult:
    """Fused RGB-D + LiDAR door handle localization and Key '1' proximity readiness."""

    detected: bool
    door_ready: bool
    handle_pos_base: tuple[float, float, float]
    planar_dist_m: float
    yaw_err_deg: float
    door_depth_m: float
    suggested_door_type: str
    confidence: float
    reason: str


def backproject_pixel_to_base_frame(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_pos_b: tuple[float, float, float] = DEFAULT_CAM_POS_BASE,
    cam_pitch_rad: float = 0.0,
) -> tuple[float, float, float]:
    """Back-project pixel ``(u, v)`` with metric depth ``depth_m`` into AS2 ``base`` frame ``(x, y, z)``.

    Follows standard ROS optical convention (``+Z`` forward, ``+X`` right, ``+Y`` down) with optional
    camera pitch ``cam_pitch_rad`` (positive = nose down around ``+Y_base``).
    """
    if not math.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError(f"depth_m must be finite and positive, got {depth_m}.")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Camera focal lengths must be positive, got fx={fx}, fy={fy}.")

    x_opt = (float(u) - float(cx)) * float(depth_m) / float(fx)
    y_opt = (float(v) - float(cy)) * float(depth_m) / float(fy)
    z_opt = float(depth_m)

    # Camera body frame before pitch: x_cam = z_opt, y_cam = -x_opt, z_cam = -y_opt
    xc, yc, zc = z_opt, -x_opt, -y_opt
    cp, sp = math.cos(cam_pitch_rad), math.sin(cam_pitch_rad)
    x_b = cam_pos_b[0] + cp * xc + sp * zc
    y_b = cam_pos_b[1] + yc
    z_b = cam_pos_b[2] - sp * xc + cp * zc
    return (x_b, y_b, z_b)


def fit_door_plane_points_xy(points_xy: np.ndarray) -> LidarDoorPlaneEstimate | None:
    """Fit a planar door line to ``(N, 2)`` forward points ``(x_b, y_b)`` in AS2 ``base`` frame via SVD."""
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] < 5:
        return None

    lateral_span = float(np.max(points_xy[:, 1]) - np.min(points_xy[:, 1]))
    if lateral_span < 0.08:
        return None

    centroid = np.mean(points_xy, axis=0)
    centered = points_xy - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[0] < 0.0:
        normal = -normal

    perp_dist = float(abs(np.dot(centroid, normal)))
    residuals = centered @ normal
    rmse = float(np.sqrt(np.mean(residuals**2)))
    yaw_err = float(math.atan2(float(normal[1]), float(normal[0])))
    return LidarDoorPlaneEstimate(
        perp_distance_m=perp_dist,
        normal_yaw_err_rad=yaw_err,
        inlier_rmse_m=rmse,
        num_points=int(points_xy.shape[0]),
    )


def estimate_door_plane_from_lidar_scan(
    ranges: Sequence[float] | np.ndarray,
    angle_min: float,
    angle_increment: float,
    lidar_pos_b: tuple[float, float, float] = DEFAULT_LIDAR_POS_BASE,
    fov_half_rad: float = 0.60,
    min_range_m: float = 0.12,
    max_range_m: float = 3.00,
) -> LidarDoorPlaneEstimate | None:
    """Extract forward-sector 2D ``LaserScan`` hits, transform to AS2 ``base`` frame, and fit door plane."""
    r = np.asarray(ranges, dtype=np.float64)
    if r.size == 0 or angle_increment == 0.0:
        return None

    angles = angle_min + np.arange(r.size, dtype=np.float64) * angle_increment
    valid = (
        np.isfinite(r)
        & (r >= min_range_m)
        & (r <= max_range_m)
        & (np.abs(angles) <= fov_half_rad)
    )
    if int(np.count_nonzero(valid)) < 5:
        return None

    rv = r[valid]
    av = angles[valid]
    x_b = lidar_pos_b[0] + rv * np.cos(av)
    y_b = lidar_pos_b[1] + rv * np.sin(av)
    pts = np.stack([x_b, y_b], axis=-1)

    x_ref = float(np.percentile(pts[:, 0], 20.0))
    inliers = pts[np.abs(pts[:, 0] - x_ref) <= 0.20]
    return fit_door_plane_points_xy(inliers)


def extract_door_plane_from_pointcloud_xyz(
    points_xyz_lidar: np.ndarray,
    lidar_pos_b: tuple[float, float, float] = DEFAULT_LIDAR_POS_BASE,
    lidar_pitch_rad: float = 0.0,
    z_band_base_m: tuple[float, float] = (-0.05, 0.60),
    fov_half_rad: float = 0.60,
    min_range_m: float = 0.12,
    max_range_m: float = 3.00,
) -> LidarDoorPlaneEstimate | None:
    """Transform 3D LiDAR ``(N, 3)`` points to ``base`` frame, slice door-panel height, and fit plane."""
    pts = np.asarray(points_xyz_lidar, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] < 5:
        return None

    finite_mask = np.all(np.isfinite(pts[:, :3]), axis=1)
    pts = pts[finite_mask, :3]
    if pts.shape[0] < 5:
        return None

    cp, sp = math.cos(lidar_pitch_rad), math.sin(lidar_pitch_rad)
    x_b = lidar_pos_b[0] + cp * pts[:, 0] + sp * pts[:, 2]
    y_b = lidar_pos_b[1] + pts[:, 1]
    z_b = lidar_pos_b[2] - sp * pts[:, 0] + cp * pts[:, 2]

    xy_norm = np.hypot(x_b - lidar_pos_b[0], y_b - lidar_pos_b[1])
    yaw = np.arctan2(y_b - lidar_pos_b[1], x_b - lidar_pos_b[0])
    band = (
        (z_b >= z_band_base_m[0])
        & (z_b <= z_band_base_m[1])
        & (xy_norm >= min_range_m)
        & (xy_norm <= max_range_m)
        & (np.abs(yaw) <= fov_half_rad)
    )
    if int(np.count_nonzero(band)) < 5:
        return None

    xy_base = np.column_stack([x_b[band], y_b[band]])
    x_ref = float(np.percentile(xy_base[:, 0], 20.0))
    inliers = xy_base[np.abs(xy_base[:, 0] - x_ref) <= 0.20]
    return fit_door_plane_points_xy(inliers)


def normalize_depth_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    """Normalize ``32FC1`` (meters) or ``16UC1`` (millimeters, RealSense default) depth array to float32 meters."""
    arr = np.asarray(depth_raw)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D depth image, got shape {arr.shape}.")
    if arr.dtype.kind == "u" and arr.dtype.itemsize == 2:
        return arr.astype(np.float32) * 0.001
    return arr.astype(np.float32)


def detect_door_and_handle_rgbd_lidar(
    rgb_image: np.ndarray | None,
    depth_image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    lidar_plane: LidarDoorPlaneEstimate | None = None,
    cam_pos_b: tuple[float, float, float] = DEFAULT_CAM_POS_BASE,
    cam_pitch_rad: float = 0.0,
    min_dist_m: float = 0.28,
    max_dist_m: float = 0.90,
    max_yaw_deg: float = 60.0,
) -> DoorDetectionResult:
    """Localize door handle 3D pose in AS2 ``base`` frame and evaluate Key ``1`` proximity gate."""
    depth_m = normalize_depth_to_meters(depth_image)
    h, w = depth_m.shape

    v0, v1 = int(0.15 * h), int(0.85 * h)
    u0, u1 = int(0.15 * w), int(0.85 * w)
    roi_depth = depth_m[v0:v1, u0:u1]
    valid_roi = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0.12) & (roi_depth < 3.00)]

    if valid_roi.size < 50:
        return DoorDetectionResult(
            detected=False,
            door_ready=False,
            handle_pos_base=(0.0, 0.0, 0.0),
            planar_dist_m=float("inf"),
            yaw_err_deg=0.0,
            door_depth_m=float("inf"),
            suggested_door_type="lever_push",
            confidence=0.0,
            reason="전방 유효 Depth 포인트 부족 (문이 시야에 없거나 너무 멂)",
        )

    door_depth_m = float(np.percentile(valid_roi, 65.0))
    protrusion = door_depth_m - roi_depth
    protruding_mask = (
        np.isfinite(roi_depth)
        & (roi_depth > 0.12)
        & (protrusion >= 0.015)
        & (protrusion <= 0.140)
    )

    weights = np.where(protruding_mask, np.clip(protrusion, 0.0, 0.14), 0.0).astype(np.float32)
    if rgb_image is not None:
        rgb_arr = np.asarray(rgb_image)
        if rgb_arr.ndim == 3 and rgb_arr.shape[:2] == (h, w) and rgb_arr.shape[2] >= 3:
            gray = rgb_arr[v0:v1, u0:u1, :3].astype(np.float32).mean(axis=-1)
            gy, gx = np.gradient(gray)
            edge_mag = np.hypot(gx, gy)
            edge_norm = edge_mag / max(float(np.percentile(edge_mag, 95.0)), 1.0)
            weights = weights * (0.5 + np.clip(edge_norm, 0.0, 2.0))

    active_pixels = int(np.count_nonzero(weights > 0.005))
    if active_pixels < 12:
        return DoorDetectionResult(
            detected=False,
            door_ready=False,
            handle_pos_base=(0.0, 0.0, 0.0),
            planar_dist_m=cam_pos_b[0] + door_depth_m,
            yaw_err_deg=math.degrees(lidar_plane.normal_yaw_err_rad) if lidar_plane is not None else 0.0,
            door_depth_m=door_depth_m,
            suggested_door_type="lever_push",
            confidence=0.2,
            reason="문 평면은 감지되었으나 돌출된 손잡이(Handle) 클러스터 미검출",
        )

    ys_roi, xs_roi = np.nonzero(weights > 0.005)
    w_vals = weights[ys_roi, xs_roi]
    w_sum = float(np.sum(w_vals))
    u_handle = float(u0 + np.sum(xs_roi * w_vals) / w_sum)
    v_handle = float(v0 + np.sum(ys_roi * w_vals) / w_sum)

    handle_depths = roi_depth[ys_roi, xs_roi]
    z_handle_m = float(np.median(handle_depths))
    handle_pos_b = backproject_pixel_to_base_frame(
        u_handle,
        v_handle,
        z_handle_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        cam_pos_b=cam_pos_b,
        cam_pitch_rad=cam_pitch_rad,
    )

    u_span_px = float(np.percentile(xs_roi, 90.0) - np.percentile(xs_roi, 10.0))
    v_span_px = float(np.percentile(ys_roi, 90.0) - np.percentile(ys_roi, 10.0))
    width_m = u_span_px * z_handle_m / max(fx, 1e-6)
    height_m = v_span_px * z_handle_m / max(fy, 1e-6)
    if width_m >= 0.34 and width_m > 2.2 * max(height_m, 0.02):
        suggested_type = "push_bar"
    elif height_m >= 0.14 and height_m > 1.7 * max(width_m, 0.02):
        suggested_type = "sliding"
    else:
        suggested_type = "lever_push"

    planar_dist_m = math.hypot(handle_pos_b[0], handle_pos_b[1])
    if lidar_plane is not None and lidar_plane.inlier_rmse_m < 0.06:
        yaw_err_deg = math.degrees(lidar_plane.normal_yaw_err_rad)
        conf = min(1.0, 0.65 + min(active_pixels, 200) / 600.0)
    else:
        yaw_err_deg = math.degrees(math.atan2(handle_pos_b[1], handle_pos_b[0]))
        conf = min(0.85, 0.45 + min(active_pixels, 200) / 600.0)

    in_dist = min_dist_m <= planar_dist_m <= max_dist_m
    in_yaw = abs(yaw_err_deg) <= max_yaw_deg
    ready = in_dist and in_yaw

    if ready:
        reason = (
            f"READY ({suggested_type}): dist={planar_dist_m:.2f}m, yaw_err={yaw_err_deg:+.1f}deg "
            f"-> [1]번 키 자동 개방 활성화 가능"
        )
    elif planar_dist_m > max_dist_m:
        reason = f"거리가 멂 ({planar_dist_m:.2f}m > {max_dist_m:.2f}m) — 더 접근 필요"
    elif planar_dist_m < min_dist_m:
        reason = f"너무 가까움 ({planar_dist_m:.2f}m < {min_dist_m:.2f}m) — 소폭 후진 필요"
    else:
        reason = f"정렬 각도 초과 ({yaw_err_deg:+.1f}deg, 허용 ±{max_yaw_deg:.0f}deg)"

    return DoorDetectionResult(
        detected=True,
        door_ready=ready,
        handle_pos_base=handle_pos_b,
        planar_dist_m=planar_dist_m,
        yaw_err_deg=yaw_err_deg,
        door_depth_m=door_depth_m,
        suggested_door_type=suggested_type,
        confidence=conf,
        reason=reason,
    )


def decode_ros_image_msg(msg) -> np.ndarray:
    """Decode ``sensor_msgs/Image`` respecting both ``msg.step`` row padding and ``msg.is_bigendian``."""
    enc = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions {w}x{h}.")
    buf = bytes(msg.data)
    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"

    if enc == "32fc1":
        row_bytes = w * 4
        step = int(getattr(msg, "step", 0)) or row_bytes
        if step < row_bytes or len(buf) < h * step:
            raise ValueError(f"Buffer length {len(buf)} smaller than {h}*{step} for 32FC1.")
        rows = np.frombuffer(buf, dtype=np.uint8, count=h * step).reshape(h, step)[:, :row_bytes]
        return np.ascontiguousarray(rows).view(np.dtype(f"{endian}f4")).reshape(h, w).astype(np.float32)

    if enc in ("16uc1", "mono16"):
        row_bytes = w * 2
        step = int(getattr(msg, "step", 0)) or row_bytes
        if step < row_bytes or len(buf) < h * step:
            raise ValueError(f"Buffer length {len(buf)} smaller than {h}*{step} for {msg.encoding}.")
        rows = np.frombuffer(buf, dtype=np.uint8, count=h * step).reshape(h, step)[:, :row_bytes]
        return np.ascontiguousarray(rows).view(np.dtype(f"{endian}u2")).reshape(h, w).astype(np.uint16)

    if enc in ("rgb8", "bgr8"):
        row_bytes = w * 3
        step = int(getattr(msg, "step", 0)) or row_bytes
        if step < row_bytes or len(buf) < h * step:
            raise ValueError(f"Buffer length {len(buf)} smaller than {h}*{step} for {msg.encoding}.")
        arr = np.frombuffer(buf, dtype=np.uint8, count=h * step).reshape(h, step)[:, :row_bytes].reshape(h, w, 3)
        return arr[..., ::-1].copy() if enc == "bgr8" else arr.copy()

    if enc in ("rgba8", "bgra8"):
        row_bytes = w * 4
        step = int(getattr(msg, "step", 0)) or row_bytes
        if step < row_bytes or len(buf) < h * step:
            raise ValueError(f"Buffer length {len(buf)} smaller than {h}*{step} for {msg.encoding}.")
        arr = np.frombuffer(buf, dtype=np.uint8, count=h * step).reshape(h, step)[:, :row_bytes].reshape(h, w, 4)
        return arr[..., [2, 1, 0]].copy() if enc == "bgra8" else arr[..., :3].copy()

    raise ValueError(f"Unsupported sensor_msgs/Image encoding: '{msg.encoding}'.")


def _decode_pointcloud2_xyz(msg, max_points: int = 4096) -> np.ndarray:
    """Extract ``(N, 3)`` float32 ``(x, y, z)`` points from ``sensor_msgs/PointCloud2``."""
    offsets = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(offsets):
        return np.empty((0, 3), dtype=np.float32)

    total_pts = int(msg.width) * int(msg.height)
    if total_pts <= 0 or msg.point_step <= 0:
        return np.empty((0, 3), dtype=np.float32)

    stride = max(1, total_pts // max_points)
    endian = ">" if msg.is_bigendian else "<"
    ox, oy, oz = offsets["x"], offsets["y"], offsets["z"]
    raw = memoryview(msg.data)
    pts = []
    for idx in range(0, total_pts, stride):
        base = idx * msg.point_step
        if base + max(ox, oy, oz) + 4 > len(raw):
            break
        x = struct.unpack_from(f"{endian}f", raw, base + ox)[0]
        y = struct.unpack_from(f"{endian}f", raw, base + oy)[0]
        z = struct.unpack_from(f"{endian}f", raw, base + oz)[0]
        pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)


def main(args: list[str] | None = None) -> None:
    """Entry point for ROS 2 Humble ``vision_door_detector`` node."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
        from std_msgs.msg import Bool, String
    except ImportError as exc:
        raise SystemExit(
            f"[ERROR] ROS 2 Python packages (rclpy / sensor_msgs) are required to run this node: {exc}"
        ) from exc

    class VisionDoorDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("vision_door_detector")
            self.declare_parameter("rgb_topic", "/as2_camera/rgb")
            self.declare_parameter("depth_topic", "/as2_camera/depth")
            self.declare_parameter("camera_info_topic", "/as2_camera/camera_info")
            self.declare_parameter("lidar_scan_topic", "/as2_lidar/scan")
            self.declare_parameter("lidar_pc_topic", "/as2_lidar/point_cloud")
            self.declare_parameter("cam_pos_base", list(DEFAULT_CAM_POS_BASE))
            self.declare_parameter("cam_pitch_deg", 0.0)
            self.declare_parameter("lidar_pos_base", list(DEFAULT_LIDAR_POS_BASE))
            self.declare_parameter("lidar_pitch_deg", 0.0)
            self.declare_parameter("sensor_timeout_s", 0.5)
            self.declare_parameter("min_dist_m", 0.28)
            self.declare_parameter("max_dist_m", 0.90)
            self.declare_parameter("max_yaw_deg", 60.0)

            sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

            self._fx = 640.0 * (24.0 / 20.955)
            self._fy = self._fx
            self._cx = 320.0
            self._cy = 240.0
            self._latest_rgb: np.ndarray | None = None
            self._rgb_stamp_s: float = 0.0
            self._latest_lidar_plane: LidarDoorPlaneEstimate | None = None
            self._lidar_stamp_s: float = 0.0
            self._last_pc_stamp_s: float = 0.0

            self.create_subscription(
                CameraInfo, self.get_parameter("camera_info_topic").value, self._on_camera_info, sensor_qos
            )
            self.create_subscription(Image, self.get_parameter("rgb_topic").value, self._on_rgb, sensor_qos)
            self.create_subscription(Image, self.get_parameter("depth_topic").value, self._on_depth, sensor_qos)
            self.create_subscription(
                LaserScan, self.get_parameter("lidar_scan_topic").value, self._on_scan, sensor_qos
            )
            self.create_subscription(
                PointCloud2, self.get_parameter("lidar_pc_topic").value, self._on_pointcloud, sensor_qos
            )

            self._handle_pub = self.create_publisher(PoseStamped, "/as2_piper/detected_handle_base", 10)
            self._ready_pub = self.create_publisher(Bool, "/as2_piper/door_ready", 10)
            self._type_pub = self.create_publisher(String, "/as2_piper/detected_door_type", 10)
            self.get_logger().info("AS2 RGB-D + 3D LiDAR Door Detector initialized.")

        def _on_camera_info(self, msg: CameraInfo) -> None:
            if len(msg.k) == 9 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
                self._fx = float(msg.k[0])
                self._fy = float(msg.k[4])
                self._cx = float(msg.k[2])
                self._cy = float(msg.k[5])

        def _on_rgb(self, msg: Image) -> None:
            try:
                self._latest_rgb = decode_ros_image_msg(msg)
                self._rgb_stamp_s = time.monotonic()
            except ValueError as exc:
                self._latest_rgb = None
                self.get_logger().warning(f"RGB decode failed: {exc}")

        def _on_scan(self, msg: LaserScan) -> None:
            now_s = time.monotonic()
            # Prefer 3D PointCloud2 when actively streaming so 2D LaserScan does not race/overwrite it
            if now_s - self._last_pc_stamp_s < float(self.get_parameter("sensor_timeout_s").value):
                return
            lidar_pos = tuple(float(v) for v in self.get_parameter("lidar_pos_base").value)
            self._latest_lidar_plane = estimate_door_plane_from_lidar_scan(
                msg.ranges, msg.angle_min, msg.angle_increment, lidar_pos_b=lidar_pos  # type: ignore[arg-type]
            )
            self._lidar_stamp_s = now_s

        def _on_pointcloud(self, msg: PointCloud2) -> None:
            now_s = time.monotonic()
            self._last_pc_stamp_s = now_s
            pts = _decode_pointcloud2_xyz(msg)
            lidar_pos = tuple(float(v) for v in self.get_parameter("lidar_pos_base").value)
            lidar_pitch = math.radians(float(self.get_parameter("lidar_pitch_deg").value))
            self._latest_lidar_plane = extract_door_plane_from_pointcloud_xyz(
                pts, lidar_pos_b=lidar_pos, lidar_pitch_rad=lidar_pitch  # type: ignore[arg-type]
            )
            self._lidar_stamp_s = now_s

        def _publish_not_ready(self) -> None:
            msg = Bool()
            msg.data = False
            self._ready_pub.publish(msg)

        def _on_depth(self, msg: Image) -> None:
            try:
                depth_arr = decode_ros_image_msg(msg)
            except ValueError as exc:
                self._publish_not_ready()
                self.get_logger().warning(f"Depth decode failed: {exc}")
                return

            now_s = time.monotonic()
            timeout_s = float(self.get_parameter("sensor_timeout_s").value)
            rgb_fresh = self._latest_rgb if (now_s - self._rgb_stamp_s <= timeout_s) else None
            lidar_fresh = self._latest_lidar_plane if (now_s - self._lidar_stamp_s <= timeout_s) else None
            cam_pos = tuple(float(v) for v in self.get_parameter("cam_pos_base").value)
            cam_pitch = math.radians(float(self.get_parameter("cam_pitch_deg").value))

            res = detect_door_and_handle_rgbd_lidar(
                rgb_image=rgb_fresh,
                depth_image=depth_arr,
                fx=self._fx,
                fy=self._fy,
                cx=self._cx,
                cy=self._cy,
                lidar_plane=lidar_fresh,
                cam_pos_b=cam_pos,  # type: ignore[arg-type]
                cam_pitch_rad=cam_pitch,
                min_dist_m=float(self.get_parameter("min_dist_m").value),
                max_dist_m=float(self.get_parameter("max_dist_m").value),
                max_yaw_deg=float(self.get_parameter("max_yaw_deg").value),
            )

            ready_msg = Bool()
            ready_msg.data = bool(res.door_ready)
            self._ready_pub.publish(ready_msg)

            if res.detected:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = "base"
                pose_msg.pose.position.x = float(res.handle_pos_base[0])
                pose_msg.pose.position.y = float(res.handle_pos_base[1])
                pose_msg.pose.position.z = float(res.handle_pos_base[2])
                pose_msg.pose.orientation.w = 1.0
                self._handle_pub.publish(pose_msg)

                type_msg = String()
                type_msg.data = res.suggested_door_type
                self._type_pub.publish(type_msg)

    rclpy.init(args=args)
    node = VisionDoorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
