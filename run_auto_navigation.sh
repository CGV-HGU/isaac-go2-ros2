#!/bin/bash

# 1. 자동 목적지 전송 스크립트를 백그라운드에서 실행합니다.
# 이 스크립트는 내부적으로 30초를 대기한 후 Nav2에 명령을 보냅니다.
echo "🚀 30초 후 자동으로 목적지(3.5, 0.0)를 전송하도록 예약되었습니다..."
python3 auto_goal_sender.py &

# 2. 기존 로컬라이제이션 및 Nav2 실행 스크립트를 시작합니다.
echo "📍 RTAB-Map 및 Nav2 자율주행 스택을 실행합니다..."
./rtabmap_localization_urdf.sh
