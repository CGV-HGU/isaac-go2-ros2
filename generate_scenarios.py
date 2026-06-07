import json
import random

# 1. 교수님 피드백: 수동으로 제한할 좌표 범위 지정
X_MIN, X_MAX = -3.0, 3.0
Y_BOUND = 0.76
FIXED_Z = -0.17  # 바닥에 딱 붙는 Z 고도 고정

# 고정할 출발지와 목적지 좌표 (필요에 따라 수정)
START_POS = [-4.0, 0.0, 0.0]
GOAL_POS = [4.0, 0.0, 0.0]

scenarios = {}
TOTAL_TRIALS = 150  # 100~200번 실험

for i in range(1, TOTAL_TRIALS + 1):
    # 장애물의 위치와 회전각 랜덤 생성
    obs_x = round(random.uniform(X_MIN, X_MAX), 2)
    obs_y = round(random.uniform(-Y_BOUND, Y_BOUND), 2)
    obs_yaw = round(random.uniform(0, 360), 1)
    
    scenarios[f"trial_{i:03d}"] = {
        "robot_start": START_POS,
        "robot_goal": GOAL_POS,
        "obstacle_pos": [obs_x, obs_y, FIXED_Z],
        "obstacle_yaw": obs_yaw
    }

# JSON 파일로 저장
with open("scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=4)

print(f"[SUCCESS] {TOTAL_TRIALS}개의 고정 실험 시나리오가 생성되었습니다!")
