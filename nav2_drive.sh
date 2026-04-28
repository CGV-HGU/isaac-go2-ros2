#!/bin/bash
# Go2 로봇 Nav2 자율주행 실행 스크립트
# 터미널에서 ./nav2_drive.sh 를 입력하면 Nav2 스택이 켜집니다.
# 주의: 이 스크립트를 실행하기 전에 반드시 RTAB-Map(매핑 또는 로컬라이제이션)이 켜져 있어야 합니다!

echo "🚗 Nav2 자율주행 스택을 시작합니다..."
echo "사용한 파라미터: nav2/nav2_params.yaml"

# Nav2 bringup 패키지를 사용하여 실행
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    params_file:=$(pwd)/nav2/nav2_params.yaml
