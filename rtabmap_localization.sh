#!/bin/bash
# Go2 로봇 [로컬라이제이션 + Nav2 자율주행] 통합 실행 스크립트
# 터미널에서 ./rtabmap_localization.sh 를 입력하면 내 위치 추적과 자율주행 기사님이 동시에 켜집니다.

echo "📍 1. RTAB-Map 로컬라이제이션 모드를 백그라운드에서 시작합니다..."
echo "주의: 매핑 모드에서 저장된 지도(~/.ros/rtabmap.db)를 불러옵니다."

ros2 launch rtabmap_launch rtabmap.launch.py \
    localization:=true \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=World \
    visual_odometry:=false \
    frame_id:=base \
    approx_sync:=true \
    use_sim_time:=true &

sleep 5

echo "🚗 2. Nav2 자율주행 스택을 시작합니다..."
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    params_file:=$(pwd)/nav2/nav2_params.yaml

wait

