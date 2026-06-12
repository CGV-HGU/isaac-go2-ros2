#!/bin/bash
# Go2 로봇 [로컬라이제이션 + Nav2 자율주행] 통합 실행 스크립트

echo "📍 1. Map Server를 켜서 2D 지도를 불러옵니다..."
MAP_FILE_PATH="/home/hayoung/.ros/rtabmap.yaml"

ros2 run nav2_map_server map_server \
  --ros-args -p yaml_filename:=${MAP_FILE_PATH} -p use_sim_time:=true &
sleep 2

ros2 run nav2_lifecycle_manager lifecycle_manager \
  --ros-args -p autostart:=true -p node_names:='["map_server"]' \
  -p bond_timeout:=0.0 -p attempt_respawn_reconnection:=true -p use_sim_time:=true &
sleep 2

echo "🤖 1.5 Robot State Publisher (URDF)를 실행합니다..."
# 경로를 절대경로(PWD 기반)로 확실하게 잡아줌
URDF_FILE_PATH="$(pwd)/../go2-ros2/robots/go2_description/urdf/go2_description.urdf"

# URDF 파일이 존재하는지 확인 후 실행
if [ -f "$URDF_FILE_PATH" ]; then
    ros2 run robot_state_publisher robot_state_publisher $URDF_FILE_PATH --ros-args -p use_sim_time:=true &
    sleep 2
else
    echo "⚠️ URDF 파일을 찾을 수 없습니다. 경로를 확인해주세요: $URDF_FILE_PATH"
fi

echo "🔗 Odom -> World Static TF 연결 중..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom World &

echo "🔦 2. Depth 이미지를 2D Laser Scan으로 변환합니다..."
ros2 run depthimage_to_laserscan depthimage_to_laserscan_node \
    --ros-args \
    -r depth:=/go2_camera/depth/image_raw \
    -r depth_camera_info:=/go2_camera/depth/camera_info \
    -r scan:=/scan \
    -p output_frame:=base \
    -p use_sim_time:=true \
    -p scan_height:=10 \
    -p range_min:=0.2 \
    -p range_max:=5.0 &

echo "📍 3. RTAB-Map 로컬라이제이션 모드를 시작합니다..."
ros2 launch rtabmap_launch rtabmap.launch.py \
    localization:=true \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=odom \
    visual_odometry:=false \
    frame_id:=base \
    grid_map_topic:=/rtabmap/map \
    approx_sync:=true \
    use_sim_time:=true &

sleep 5

echo "🚗 3. Nav2 자율주행 스택을 시작합니다..."
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    autostart:=True \
    params_file:=$(pwd)/nav2/nav2_params.yaml &

sleep 3

echo "🖥️ 4. RViz2 (자율주행 조종 화면)를 시작합니다..."
ros2 launch nav2_bringup rviz_launch.py \
    use_sim_time:=True

wait
