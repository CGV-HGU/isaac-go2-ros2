#!/bin/bash
# Go2 로봇 RTAB-Map [로컬라이제이션 모드] 실행 스크립트
# 터미널에서 ./rtabmap_localization.sh 를 입력하면 기존 지도를 불러와 내 위치만 추적합니다.

echo "📍 RTAB-Map 로컬라이제이션 모드(Localization)를 시작합니다..."
echo "주의: 매핑 모드에서 저장된 지도(~/.ros/rtabmap.db)를 불러옵니다."

# Isaac Sim 토픽 이름에 맞춰 RTAB-Map 실행 (로컬라이제이션)
ros2 launch rtabmap_launch rtabmap.launch.py \
    localization:=true \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    frame_id:=base_link \
    approx_sync:=true \
    use_sim_time:=true
