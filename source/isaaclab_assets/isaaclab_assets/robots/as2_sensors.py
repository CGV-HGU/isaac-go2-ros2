# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unified RGB Camera, Depth Camera, and 3D LiDAR Sensor Suite for Unitree AS2 & AS2 + AgileX PiPER.

Provides:
1. Shared physical sensor mount poses matching ``as2_piper.urdf`` (``front_camera`` & ``lidar_link``)
   and canonical ROS 2 Humble topic/frame names (importable offline without Omniverse for ROS 2 & tests).
2. Isaac Lab ``CameraCfg`` (simultaneous ``rgb`` + ``distance_to_image_plane`` metric depth) and
   ``RayCasterCfg`` (360-deg multi-channel 3D ``LidarPatternCfg`` for static terrain/ground ray-casting).
   Note: ``RayCasterCfg`` ray-casts static ground/terrain meshes in vectorized RL environments; dynamic
   articulated doors and walls are sensed via ``rgbd_camera`` depth and RTX 3D LiDAR (`spawn_as2_rtx_lidar_prim`).
3. Isaac Sim 5.x / 6.x (and 4.5 fallback) OmniGraph setup publishing RGB, Depth, CameraInfo (in ROS optical
   frame ``front_camera_optical_frame``), RTX 3D LiDAR PointCloud2 & LaserScan (in ``lidar_link``), TF, and Clock.
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Physical Mounts (relative to Unitree AS2 `base` link, matching as2_piper.urdf)
# =============================================================================
AS2_FRONT_CAMERA_POS: tuple[float, float, float] = (0.30, 0.0, 0.10)
AS2_FRONT_CAMERA_QUAT_WORLD: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Front-nose 3D LiDAR laser origin (visual cylinder in URDF is offset z=-0.025 below origin to avoid self-hits)
AS2_LIDAR_POS: tuple[float, float, float] = (0.285, 0.0, 0.045)
AS2_LIDAR_QUAT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

AS2_CHIN_RADAR_POS: tuple[float, float, float] = (0.28945, 0.0, -0.046825)

AS2_CAMERA_WIDTH: int = 640
AS2_CAMERA_HEIGHT: int = 480
AS2_CAMERA_FOCAL_LENGTH_MM: float = 24.0
AS2_CAMERA_HORIZONTAL_APERTURE_MM: float = 20.955
AS2_CAMERA_CLIPPING_RANGE_M: tuple[float, float] = (0.10, 15.0)

AS2_LIDAR_CHANNELS: int = 16
AS2_LIDAR_VERTICAL_FOV_DEG: tuple[float, float] = (-15.0, 30.0)
AS2_LIDAR_HORIZONTAL_FOV_DEG: tuple[float, float] = (-180.0, 180.0)
AS2_LIDAR_HORIZONTAL_RES_DEG: float = 1.0
AS2_LIDAR_MAX_DISTANCE_M: float = 25.0

AS2_ROS2_TOPICS: dict[str, str] = {
    "rgb": "/as2_camera/rgb",
    "depth": "/as2_camera/depth",
    "camera_info": "/as2_camera/camera_info",
    "lidar_point_cloud": "/as2_lidar/point_cloud",
    "lidar_scan": "/as2_lidar/scan",
    "tf": "/tf",
    "clock": "/clock",
}

AS2_ROS2_FRAMES: dict[str, str] = {
    "base": "base",
    "camera": "front_cam",
    "camera_optical": "front_camera_optical_frame",
    "lidar": "lidar_link",
}


def compute_default_pinhole_intrinsics(
    width: int = AS2_CAMERA_WIDTH,
    height: int = AS2_CAMERA_HEIGHT,
    focal_length_mm: float = AS2_CAMERA_FOCAL_LENGTH_MM,
    horizontal_aperture_mm: float = AS2_CAMERA_HORIZONTAL_APERTURE_MM,
) -> tuple[float, float, float, float]:
    """Compute pixel-space ``(fx, fy, cx, cy)`` matching Isaac Sim's ``PinholeCameraCfg``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Camera dimensions must be positive, got {width}x{height}.")
    if focal_length_mm <= 0.0 or horizontal_aperture_mm <= 0.0:
        raise ValueError("Focal length and horizontal aperture must be positive.")
    fx = width * (focal_length_mm / horizontal_aperture_mm)
    vertical_aperture_mm = horizontal_aperture_mm * (float(height) / float(width))
    fy = height * (focal_length_mm / vertical_aperture_mm)
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy


# =============================================================================
# Isaac Lab Sensor Configuration Builders (requires Isaac Lab / Omniverse)
# =============================================================================
def create_as2_rgbd_camera_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base/front_cam",
    width: int = AS2_CAMERA_WIDTH,
    height: int = AS2_CAMERA_HEIGHT,
    update_period: float = 0.05,
    data_types: list[str] | None = None,
) -> Any:
    """Create an Isaac Lab ``CameraCfg`` providing simultaneous RGB + metric Depth tensors."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    enabled_types = list(data_types) if data_types is not None else ["rgb", "distance_to_image_plane"]
    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=enabled_types,
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=AS2_CAMERA_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=AS2_CAMERA_HORIZONTAL_APERTURE_MM,
            clipping_range=AS2_CAMERA_CLIPPING_RANGE_M,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=AS2_FRONT_CAMERA_POS,
            rot=AS2_FRONT_CAMERA_QUAT_WORLD,
            convention="world",
        ),
    )


def create_as2_lidar_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base",
    mesh_prim_paths: list[str] | None = None,
    channels: int = AS2_LIDAR_CHANNELS,
    vertical_fov_range: tuple[float, float] = AS2_LIDAR_VERTICAL_FOV_DEG,
    horizontal_fov_range: tuple[float, float] = AS2_LIDAR_HORIZONTAL_FOV_DEG,
    horizontal_res: float = AS2_LIDAR_HORIZONTAL_RES_DEG,
    max_distance: float = AS2_LIDAR_MAX_DISTANCE_M,
    update_period: float = 0.05,
    debug_vis: bool = False,
) -> Any:
    """Create an Isaac Lab GPU ``RayCasterCfg`` (3D ``LidarPatternCfg``) for static ground/terrain."""
    from isaaclab.sensors import RayCasterCfg, patterns

    target_meshes = list(mesh_prim_paths) if mesh_prim_paths is not None else ["/World/ground"]
    return RayCasterCfg(
        prim_path=prim_path,
        update_period=update_period,
        offset=RayCasterCfg.OffsetCfg(pos=AS2_LIDAR_POS, rot=AS2_LIDAR_QUAT),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels,
            vertical_fov_range=vertical_fov_range,
            horizontal_fov_range=horizontal_fov_range,
            horizontal_res=horizontal_res,
        ),
        max_distance=max_distance,
        debug_vis=debug_vis,
        mesh_prim_paths=target_meshes,
    )


def resolve_scene_ground_prim_path(scene_cfg: Any) -> str:
    """Infer the primary ground/terrain prim path on an ``InteractiveSceneCfg`` for ``RayCasterCfg``."""
    terrain_cfg = getattr(scene_cfg, "terrain", None)
    if terrain_cfg is not None and getattr(terrain_cfg, "prim_path", None):
        return str(terrain_cfg.prim_path)
    ground_cfg = getattr(scene_cfg, "ground", None)
    if ground_cfg is not None and getattr(ground_cfg, "prim_path", None):
        return str(ground_cfg.prim_path)
    return "/World/ground"


def attach_as2_sensors_to_scene_cfg(
    scene_cfg: Any,
    enable_rgbd: bool = True,
    enable_lidar: bool = True,
    ground_prim_path: str | None = None,
    width: int = AS2_CAMERA_WIDTH,
    height: int = AS2_CAMERA_HEIGHT,
    lidar_debug_vis: bool = False,
) -> Any:
    """Attach ``rgbd_camera`` (RGB + Depth) and ``lidar`` (3D terrain RayCaster) onto a scene cfg."""
    if enable_rgbd:
        scene_cfg.rgbd_camera = create_as2_rgbd_camera_cfg(width=width, height=height)
    if enable_lidar:
        mesh_path = ground_prim_path if ground_prim_path is not None else resolve_scene_ground_prim_path(scene_cfg)
        scene_cfg.lidar = create_as2_lidar_cfg(mesh_prim_paths=[mesh_path], debug_vis=lidar_debug_vis)
    return scene_cfg


def summarize_as2_sensor_readings(scene: Any) -> dict[str, float]:
    """Compute lightweight live diagnostics from attached ``rgbd_camera`` and ``lidar`` sensors (env 0)."""
    import torch

    summary: dict[str, float] = {}
    scene_keys = scene.keys() if hasattr(scene, "keys") else ()

    if "rgbd_camera" in scene_keys:
        cam_out = scene["rgbd_camera"].data.output
        if "rgb" in cam_out:
            rgb0 = cam_out["rgb"][0]
            summary["rgb_mean_intensity"] = float(rgb0[..., :3].float().mean().item())
        if "distance_to_image_plane" in cam_out:
            depth0 = cam_out["distance_to_image_plane"][0]
            if depth0.ndim == 3:
                depth0 = depth0[..., 0]
            h, w = depth0.shape[:2]
            roi = depth0[int(0.4 * h) : int(0.6 * h), int(0.4 * w) : int(0.6 * w)]
            finite_roi = roi[torch.isfinite(roi) & (roi > 0.05)]
            if finite_roi.numel() > 0:
                summary["center_depth_m"] = float(finite_roi.median().item())

    if "lidar" in scene_keys:
        lidar_data = scene["lidar"].data
        hits_w = lidar_data.ray_hits_w[0]
        origin_w = lidar_data.pos_w[0].unsqueeze(0)
        dists = torch.linalg.norm(hits_w - origin_w, dim=-1)
        valid = dists[torch.isfinite(dists) & (dists > 0.05) & (dists < AS2_LIDAR_MAX_DISTANCE_M - 1e-3)]
        if valid.numel() > 0:
            summary["lidar_terrain_min_range_m"] = float(valid.min().item())

    return summary


def spawn_as2_rtx_lidar_prim(
    stage: Any,
    simulation_app: Any,
    robot_base_path: str = "/World/envs/env_0/Robot/base",
) -> str:
    """Enable RTX sensor extension and spawn a 3D RTX LiDAR prim at ``{robot_base_path}/lidar_link``."""
    try:
        from isaacsim.core.utils.extensions import enable_extension

        lidar_ext = "isaacsim.sensors.rtx"
    except ImportError:
        from omni.isaac.core.utils.extensions import enable_extension

        lidar_ext = "omni.isaac.sensor"

    if not enable_extension(lidar_ext):
        raise RuntimeError(f"RTX LiDAR extension '{lidar_ext}' could not be enabled.")
    simulation_app.update()

    import omni.kit.commands
    from pxr import Gf

    lidar_prim_path = f"{robot_base_path}/{AS2_ROS2_FRAMES['lidar']}"
    if not stage.GetPrimAtPath(lidar_prim_path).IsValid():
        ok, lidar_sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=AS2_ROS2_FRAMES["lidar"],
            parent=robot_base_path,
            config="Example_Rotary",
            translation=Gf.Vec3d(*AS2_LIDAR_POS),
            orientation=Gf.Quatd(*AS2_LIDAR_QUAT),
        )
        if not ok or lidar_sensor is None:
            raise RuntimeError(f"IsaacSensorCreateRtxLidar failed to create prim at '{lidar_prim_path}'.")
        lidar_prim_path = str(lidar_sensor.GetPath())

    if not stage.GetPrimAtPath(lidar_prim_path).IsValid():
        raise RuntimeError(f"RTX LiDAR prim at '{lidar_prim_path}' is invalid after creation.")
    return lidar_prim_path


# =============================================================================
# Isaac Sim 5.x / 6.x OmniGraph ROS 2 Bridge (RGB + Depth + RTX 3D LiDAR + TF)
# =============================================================================
def setup_as2_ros2_sensors_graph(
    stage: Any,
    simulation_app: Any,
    robot_base_path: str = "/World/envs/env_0/Robot/base",
    enable_rgbd: bool = True,
    enable_rtx_lidar: bool = True,
    width: int = AS2_CAMERA_WIDTH,
    height: int = AS2_CAMERA_HEIGHT,
    existing_camera_rp_path: str | None = None,
    graph_path: str = "/World/ROS2_AS2_Sensors_Graph",
) -> dict[str, Any]:
    """Build Isaac Sim 5.x/6.x (with 4.5 fallback) ROS 2 OmniGraph for AS2 RGB, Depth, and 3D LiDAR."""
    try:
        from isaacsim.core.utils.extensions import enable_extension

        ros2_ext = "isaacsim.ros2.bridge"
        core_nodes_pkg = "isaacsim.core.nodes"
        ros2_nodes_pkg = "isaacsim.ros2.bridge"
    except ImportError:
        from omni.isaac.core.utils.extensions import enable_extension

        ros2_ext = "omni.isaac.ros2_bridge"
        core_nodes_pkg = "omni.isaac.core_nodes"
        ros2_nodes_pkg = "omni.isaac.ros2_bridge"

    if not enable_extension(ros2_ext):
        raise RuntimeError(f"Extension '{ros2_ext}' could not be enabled.")
    simulation_app.update()

    import omni.graph.core as og
    from pxr import Gf, UsdGeom

    if not stage.GetPrimAtPath(robot_base_path).IsValid():
        raise RuntimeError(f"Robot base prim not found at '{robot_base_path}'.")

    tf_targets_under_base: list[str] = []
    active_topics: list[str] = [AS2_ROS2_TOPICS["tf"], AS2_ROS2_TOPICS["clock"]]

    create_nodes: list[tuple[str, str]] = [
        ("OnTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", f"{core_nodes_pkg}.IsaacReadSimulationTime"),
        ("ROS2TF_World", f"{ros2_nodes_pkg}.ROS2PublishTransformTree"),
        ("ROS2Clock", f"{ros2_nodes_pkg}.ROS2PublishClock"),
    ]
    set_values: list[tuple[str, Any]] = [
        ("ROS2TF_World.inputs:parentPrim", "/World"),
        ("ROS2TF_World.inputs:targetPrims", [robot_base_path]),
        ("ROS2TF_World.inputs:topicName", AS2_ROS2_TOPICS["tf"]),
        ("ROS2Clock.inputs:topicName", AS2_ROS2_TOPICS["clock"]),
    ]
    connections: list[tuple[str, str]] = [
        ("OnTick.outputs:tick", "ROS2TF_World.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "ROS2TF_World.inputs:timeStamp"),
        ("OnTick.outputs:tick", "ROS2Clock.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "ROS2Clock.inputs:timeStamp"),
    ]

    # 1. Front RGB + Depth Camera prim & ROS optical TF frame (+X right, +Y down, +Z forward)
    camera_path = f"{robot_base_path}/{AS2_ROS2_FRAMES['camera']}"
    optical_frame_path = f"{robot_base_path}/{AS2_ROS2_FRAMES['camera_optical']}"
    if enable_rgbd:
        if not stage.GetPrimAtPath(camera_path).IsValid():
            cam = UsdGeom.Camera.Define(stage, camera_path)
            cam.AddTranslateOp().Set(Gf.Vec3d(*AS2_FRONT_CAMERA_POS))
            cam.AddRotateXYZOp().Set(Gf.Vec3d(90.0, 0.0, -90.0))
            cam.GetFocalLengthAttr().Set(AS2_CAMERA_FOCAL_LENGTH_MM)
            cam.GetFocusDistanceAttr().Set(400.0)
            cam.GetHorizontalApertureAttr().Set(AS2_CAMERA_HORIZONTAL_APERTURE_MM)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(*AS2_CAMERA_CLIPPING_RANGE_M))

        if not stage.GetPrimAtPath(optical_frame_path).IsValid():
            opt_xform = UsdGeom.Xform.Define(stage, optical_frame_path)
            opt_xform.AddTranslateOp().Set(Gf.Vec3d(*AS2_FRONT_CAMERA_POS))
            # ROS optical convention relative to AS2 base (+X fwd, +Y left, +Z up):
            # X_opt = -Y_base (right), Y_opt = -Z_base (down), Z_opt = +X_base (forward) -> q=(0.5, -0.5, 0.5, -0.5)
            opt_xform.AddOrientOp().Set(Gf.Quatf(0.5, Gf.Vec3f(-0.5, 0.5, -0.5)))

        cam_info_node_type = f"{ros2_nodes_pkg}.ROS2CameraInfoHelper"
        if og.get_node_type_interface().get_node_type(cam_info_node_type) is None:
            cam_info_node_type = f"{ros2_nodes_pkg}.ROS2CameraHelper"

        create_nodes.extend(
            [
                ("ROS2CameraRGB", f"{ros2_nodes_pkg}.ROS2CameraHelper"),
                ("ROS2CameraDepth", f"{ros2_nodes_pkg}.ROS2CameraHelper"),
                ("ROS2CameraInfo", cam_info_node_type),
            ]
        )
        set_values.extend(
            [
                ("ROS2CameraRGB.inputs:type", "rgb"),
                ("ROS2CameraRGB.inputs:topicName", AS2_ROS2_TOPICS["rgb"]),
                ("ROS2CameraRGB.inputs:frameId", AS2_ROS2_FRAMES["camera_optical"]),
                ("ROS2CameraDepth.inputs:type", "depth"),
                ("ROS2CameraDepth.inputs:topicName", AS2_ROS2_TOPICS["depth"]),
                ("ROS2CameraDepth.inputs:frameId", AS2_ROS2_FRAMES["camera_optical"]),
                ("ROS2CameraInfo.inputs:topicName", AS2_ROS2_TOPICS["camera_info"]),
                ("ROS2CameraInfo.inputs:frameId", AS2_ROS2_FRAMES["camera_optical"]),
            ]
        )
        if cam_info_node_type.endswith("ROS2CameraHelper"):
            set_values.append(("ROS2CameraInfo.inputs:type", "camera_info"))

        if existing_camera_rp_path:
            # Reuse existing Isaac Lab Camera render product to avoid duplicate GPU ray-tracing
            set_values.extend(
                [
                    ("ROS2CameraRGB.inputs:renderProductPath", existing_camera_rp_path),
                    ("ROS2CameraDepth.inputs:renderProductPath", existing_camera_rp_path),
                    ("ROS2CameraInfo.inputs:renderProductPath", existing_camera_rp_path),
                ]
            )
            connections.extend(
                [
                    ("OnTick.outputs:tick", "ROS2CameraRGB.inputs:execIn"),
                    ("OnTick.outputs:tick", "ROS2CameraDepth.inputs:execIn"),
                    ("OnTick.outputs:tick", "ROS2CameraInfo.inputs:execIn"),
                ]
            )
        else:
            create_nodes.append(("CamRenderProduct", f"{core_nodes_pkg}.IsaacCreateRenderProduct"))
            set_values.extend(
                [
                    ("CamRenderProduct.inputs:cameraPrim", camera_path),
                    ("CamRenderProduct.inputs:width", int(width)),
                    ("CamRenderProduct.inputs:height", int(height)),
                ]
            )
            connections.extend(
                [
                    ("OnTick.outputs:tick", "CamRenderProduct.inputs:execIn"),
                    ("CamRenderProduct.outputs:execOut", "ROS2CameraRGB.inputs:execIn"),
                    ("CamRenderProduct.outputs:renderProductPath", "ROS2CameraRGB.inputs:renderProductPath"),
                    ("CamRenderProduct.outputs:execOut", "ROS2CameraDepth.inputs:execIn"),
                    ("CamRenderProduct.outputs:renderProductPath", "ROS2CameraDepth.inputs:renderProductPath"),
                    ("CamRenderProduct.outputs:execOut", "ROS2CameraInfo.inputs:execIn"),
                    ("CamRenderProduct.outputs:renderProductPath", "ROS2CameraInfo.inputs:renderProductPath"),
                ]
            )

        tf_targets_under_base.append(optical_frame_path)
        active_topics.extend(
            [AS2_ROS2_TOPICS["rgb"], AS2_ROS2_TOPICS["depth"], AS2_ROS2_TOPICS["camera_info"]]
        )

    # 2. Front-Nose RTX 3D LiDAR prim (`{robot_base_path}/lidar_link` so TF frameId == "lidar_link")
    rtx_lidar_attached = False
    if enable_rtx_lidar:
        rtx_lidar_helper_type = f"{ros2_nodes_pkg}.ROS2RtxLidarHelper"
        if og.get_node_type_interface().get_node_type(rtx_lidar_helper_type) is None:
            raise RuntimeError(
                f"OmniGraph node type '{rtx_lidar_helper_type}' is not registered; cannot attach RTX 3D LiDAR."
            )
        lidar_prim_path = spawn_as2_rtx_lidar_prim(stage, simulation_app, robot_base_path=robot_base_path)
        create_nodes.extend(
            [
                ("LidarRenderProduct", f"{core_nodes_pkg}.IsaacCreateRenderProduct"),
                ("ROS2LidarPointCloud", rtx_lidar_helper_type),
                ("ROS2LidarScan", rtx_lidar_helper_type),
            ]
        )
        set_values.extend(
            [
                ("LidarRenderProduct.inputs:cameraPrim", lidar_prim_path),
                ("LidarRenderProduct.inputs:width", 1),
                ("LidarRenderProduct.inputs:height", 1),
                ("ROS2LidarPointCloud.inputs:type", "point_cloud"),
                ("ROS2LidarPointCloud.inputs:topicName", AS2_ROS2_TOPICS["lidar_point_cloud"]),
                ("ROS2LidarPointCloud.inputs:frameId", AS2_ROS2_FRAMES["lidar"]),
                ("ROS2LidarScan.inputs:type", "laser_scan"),
                ("ROS2LidarScan.inputs:topicName", AS2_ROS2_TOPICS["lidar_scan"]),
                ("ROS2LidarScan.inputs:frameId", AS2_ROS2_FRAMES["lidar"]),
            ]
        )
        connections.extend(
            [
                ("OnTick.outputs:tick", "LidarRenderProduct.inputs:execIn"),
                ("LidarRenderProduct.outputs:execOut", "ROS2LidarPointCloud.inputs:execIn"),
                ("LidarRenderProduct.outputs:renderProductPath", "ROS2LidarPointCloud.inputs:renderProductPath"),
                ("LidarRenderProduct.outputs:execOut", "ROS2LidarScan.inputs:execIn"),
                ("LidarRenderProduct.outputs:renderProductPath", "ROS2LidarScan.inputs:renderProductPath"),
            ]
        )
        tf_targets_under_base.append(lidar_prim_path)
        active_topics.extend([AS2_ROS2_TOPICS["lidar_point_cloud"], AS2_ROS2_TOPICS["lidar_scan"]])
        rtx_lidar_attached = True

    if tf_targets_under_base:
        create_nodes.append(("ROS2TF_Sensors", f"{ros2_nodes_pkg}.ROS2PublishTransformTree"))
        set_values.extend(
            [
                ("ROS2TF_Sensors.inputs:parentPrim", robot_base_path),
                ("ROS2TF_Sensors.inputs:targetPrims", tf_targets_under_base),
                ("ROS2TF_Sensors.inputs:topicName", AS2_ROS2_TOPICS["tf"]),
            ]
        )
        connections.extend(
            [
                ("OnTick.outputs:tick", "ROS2TF_Sensors.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "ROS2TF_Sensors.inputs:timeStamp"),
            ]
        )

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.SET_VALUES: set_values,
            og.Controller.Keys.CONNECT: connections,
        },
    )
    return {
        "graph_path": graph_path,
        "topics": active_topics,
        "rtx_lidar_attached": rtx_lidar_attached,
        "rgbd_attached": enable_rgbd,
    }
