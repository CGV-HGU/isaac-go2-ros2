#!/bin/bash

# 모든 백그라운드 프로세스를 한꺼번에 종료하기 위한 설정
trap "kill 0" EXIT

echo "======================================================"
echo "🚀 [All-in-One] Go2 자율주행 자동 테스트 시작"
echo "======================================================"

# 1. 시뮬레이션 실행 (Isaac Lab)
echo "🎮 1단계: 시뮬레이션(play_ms.sh) 실행 중..."
./play_ms.sh &
sleep 40 # 아이작 심이 완전히 켜질 때까지 대기

# 2. 로컬라이제이션 및 Nav2 실행
echo "📍 2단계: Nav2 및 RTAB-Map 실행 중..."
./rtabmap_localization_urdf.sh &
sleep 15 # Nav2 노드들이 올라올 때까지 대기

# 3. 목적지 자동 전송
echo "🎯 3단계: 목적지 자동 전송 시작 (auto_goal_sender.py)"
# 시뮬레이션 시간 동기화를 위해 use_sim_time 파라미터를 추가합니다.
python3 auto_goal_sender.py --ros-args -p use_sim_time:=true

echo "✅ 모든 명령이 전달되었습니다. 시뮬레이션을 관찰하세요."

# 모든 프로세스가 유지되도록 대기
wait
