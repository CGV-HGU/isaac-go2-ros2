import omni
import omni.graph.core as og
from pxr import UsdGeom, Gf

def setup_ros2_sensors(stage, robot_base_path="/World/envs/env_0/Robot/base", camera_path="/World/envs/env_0/Robot/base/front_camera"):
    """
    Sets up all necessary OmniGraph nodes for ROS 2 communication (Cameras, Odom, TF, Clock).
    This function isolates the ROS 2 setup logic from the main reinforcement learning loop.
    """
    print("[INFO] Setting up ROS 2 Bridge and Sensor OmniGraphs...")

    # Enable required extensions for Isaac Sim 5.1.0
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    # 1. Create Camera Prim attached to the robot's base
    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        cam = UsdGeom.Camera.Define(stage, camera_path)
    else:
        cam = UsdGeom.Camera(cam_prim)

    # Position: 0.25m forward, 0.1m up (near head)
    if not cam.GetTranslateOp():
        cam.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, 0.1))
    else:
        cam.GetTranslateOp().Set(Gf.Vec3d(0.25, 0.0, 0.1))

    # Rotation: 정면(+X)을 바라보도록 수정 (현재 90도 오른쪽 편향 보고에 따라 -180 -> -90으로 조정)
    if not cam.GetRotateXYZOp():
        cam.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))
    else:
        cam.GetRotateXYZOp().Set(Gf.Vec3d(90, 0, -90))
    
    cam.GetFocalLengthAttr().Set(24.0)
    # [수정] Depth 데이터 왜곡 방지를 위한 클리핑 평면 설정 (아주 가까운 것도 보이게)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    # 2. Create OmniGraph for ROS 2 Bridge (Full SLAM setup)
    og.Controller.edit(
        {"graph_path": "/World/ROS2_Camera_Graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("ROS2CameraRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2CameraDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2CameraInfoRGB", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("ROS2CameraInfoDepth", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("ROS2Odometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("ROS2TF_World", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ROS2TF_Base", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("ROS2CmdVel", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ],
            og.Controller.Keys.SET_VALUES: [
                # Render Product
                ("RenderProduct.inputs:cameraPrim", camera_path),
                ("RenderProduct.inputs:width", 640),
                ("RenderProduct.inputs:height", 480),
                
                # RGB & Depth Image
                ("ROS2CameraRGB.inputs:type", "rgb"),
                ("ROS2CameraRGB.inputs:topicName", "/go2_camera/rgb/image_raw"),
                ("ROS2CameraRGB.inputs:frameId", "front_camera"),
                # [수정] Depth 이미지 포맷 명시 (RViz 깨짐 방지용 32비트 부동소수점 명시)
                ("ROS2CameraDepth.inputs:type", "depth"),
                ("ROS2CameraDepth.inputs:topicName", "/go2_camera/depth/image_raw"),
                ("ROS2CameraDepth.inputs:frameId", "front_camera"),

                # Camera Info
                ("ROS2CameraInfoRGB.inputs:topicName", "/go2_camera/rgb/camera_info"),
                ("ROS2CameraInfoRGB.inputs:frameId", "front_camera"),
                ("ROS2CameraInfoDepth.inputs:topicName", "/go2_camera/depth/camera_info"),
                ("ROS2CameraInfoDepth.inputs:frameId", "front_camera"),

                # Odometry
                ("ComputeOdometry.inputs:chassisPrim", robot_base_path),
                ("ROS2Odometry.inputs:odomFrameId", "World"),
                ("ROS2Odometry.inputs:chassisFrameId", "base"),
                ("ROS2Odometry.inputs:topicName", "/odom"),

                # TF Tree 1 (world -> base)
                ("ROS2TF_World.inputs:parentPrim", "/World"),
                ("ROS2TF_World.inputs:targetPrims", [robot_base_path]),
                ("ROS2TF_World.inputs:topicName", "/tf"),

                # TF Tree 2 (base -> front_camera)
                ("ROS2TF_Base.inputs:parentPrim", robot_base_path),
                ("ROS2TF_Base.inputs:targetPrims", [camera_path]),
                ("ROS2TF_Base.inputs:topicName", "/tf"),

                # Cmd_vel Subscriber
                ("ROS2CmdVel.inputs:topicName", "/cmd_vel"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "ROS2Clock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "ROS2Clock.inputs:timeStamp"),
                
                ("OnTick.outputs:tick", "RenderProduct.inputs:execIn"),
                
                # [수정] RenderProduct의 execOut 대신 OnTick에서 병렬로 신호를 줘서 멈춤 현상(Freezing) 방지
                ("OnTick.outputs:tick", "ROS2CameraRGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraRGB.inputs:renderProductPath"),
                
                ("OnTick.outputs:tick", "ROS2CameraDepth.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraDepth.inputs:renderProductPath"),
                
                ("OnTick.outputs:tick", "ROS2CameraInfoRGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraInfoRGB.inputs:renderProductPath"),
                
                ("OnTick.outputs:tick", "ROS2CameraInfoDepth.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraInfoDepth.inputs:renderProductPath"),

                ("OnTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "ROS2Odometry.inputs:execIn"),
                ("ComputeOdometry.outputs:position", "ROS2Odometry.inputs:position"),
                ("ComputeOdometry.outputs:orientation", "ROS2Odometry.inputs:orientation"),
                ("ComputeOdometry.outputs:linearVelocity", "ROS2Odometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:angularVelocity", "ROS2Odometry.inputs:angularVelocity"),
                ("ReadSimTime.outputs:simulationTime", "ROS2Odometry.inputs:timeStamp"),

                ("OnTick.outputs:tick", "ROS2TF_World.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "ROS2TF_World.inputs:timeStamp"),

                ("OnTick.outputs:tick", "ROS2TF_Base.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "ROS2TF_Base.inputs:timeStamp"),

                ("OnTick.outputs:tick", "ROS2CmdVel.inputs:execIn"),
            ],
        },
    )
    print("[INFO] ROS 2 Sensor OmniGraphs successfully created.")
