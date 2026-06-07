import json
import random

# 하영님 설정: 100회 실험
TOTAL_TRIALS = 100

# 출발지와 목적지 고정 (맵 상황에 맞게 좌표 수정 가능)
START_POS = [-3.5, 0.0, 0.0]
GOAL_POS = [3.5, 0.0, 0.0]

# 장애물 배치 구역 (출발지와 목적지 사이 통로)
X_MIN, X_MAX = 0.5, 1.5 
Y_BOUND = 0.6
FIXED_Z = 0.0  # 바닥 높이

scenarios = {}

for i in range(1, TOTAL_TRIALS + 1):
    obs_x = round(random.uniform(X_MIN, X_MAX), 2)
    obs_y = round(random.uniform(-Y_BOUND, Y_BOUND), 2)
    obs_yaw = round(random.uniform(0, 360), 1)
    
    scenarios[f"trial_{i:03d}"] = {
        "robot_start": START_POS,
        "robot_goal": GOAL_POS,
        "obstacle_pos": [obs_x, obs_y, FIXED_Z],
        "obstacle_yaw": obs_yaw
    }

with open("scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=4)

print(f"[SUCCESS] {TOTAL_TRIALS}개의 고정 실험 시나리오가 생성되었습니다!")
