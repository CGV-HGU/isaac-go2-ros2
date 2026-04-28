#!/bin/bash
# Go2 로봇 RTAB-Map [매핑 모드] 실행 스크립트
# 터미널에서 ./rtabmap_mapping.sh 를 입력하면 백지 상태에서 3D 지도를 새로 그리기 시작합니다.

echo "🗺️ RTAB-Map 매핑 모드(Mapping)를 시작합니다..."
echo "주의: 이전 지도는 삭제되고 새로 그립니다. (DB 초기화)"

echo "🔗 Odom -> World Static TF 연결 중..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom World &

# Isaac Sim 토픽 이름에 맞춰 RTAB-Map 실행 (매핑)
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start" \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=odom \
    visual_odometry:=false \
    frame_id:=base \
    approx_sync:=true \
    use_sim_time:=true
