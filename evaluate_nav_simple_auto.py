import os
import random
import math
import time
import csv
from omni.isaac.kit import SimulationApp

# 1. 시뮬레이션 앱 설정
simulation_app = SimulationApp({"headless": False})

import omni
import omni.graph.core as og
from pxr import UsdGeom, Gf
import omni.physx

# Isaac Sim 내부의 시뮬레이션 컨텍스트 가져오기
from omni.isaac.core import SimulationContext

def main():
    sim_context = SimulationContext(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    
    # 하영님의 Go2.usd 환경에 맞춰 로봇과 장애물 경로 설정
    robot_path = "/World/envs/env_0/Robot" 
    drawer_path = "/World/Moveable_Objects/Drawer"
    
    TOTAL_TRIALS = 100
    RESULT_FILE = "eval_results_simple.csv"
    
    # 결과 파일(CSV) 초기화 및 헤더 작성
    with open(RESULT_FILE, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(["Trial", "Success", "Goal_X", "Goal_Y", "Obstacle_X", "Obstacle_Y", "Duration"])

    print("\n🚀 [INFO] 단순 기록 방식 100회 평가 자동화 시작")

    for trial in range(1, TOTAL_TRIALS + 1):
        print(f"\n--- [시도 {trial} / {TOTAL_TRIALS}] ---")
        
        # 1. 로봇 위치 초기화 (스폰 높이를 안정적인 0.4m로 고정하여 추락 방지)
        START_X, START_Y, START_Z = -1.0, 0.0, 0.4
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim.IsValid():
            UsdGeom.Xformable(robot_prim).ClearXformOpOrder()
            UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(Gf.Vec3d(START_X, START_Y, START_Z))
            print(f"[로봇 초기화] 위치: ({START_X}, {START_Y}, {START_Z})")
        
        # 2. 목적지(Goal) 랜덤 생성 (X는 5.0 고정, Y는 -1.5 ~ 1.5 사이 랜덤)
        GOAL_X = 5.0
        GOAL_Y = random.uniform(-1.5, 1.5)
        print(f"[목적지 설정] X: {GOAL_X}, Y: {GOAL_Y:.2f}")
        
        # 3. 출발지와 목적지 사이 영역에 장애물(Drawer) 랜덤 배치
        obs_x = random.uniform(1.0, 3.5)
        obs_y = random.uniform(-1.0, 1.0)
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            UsdGeom.Xformable(drawer_prim).ClearXformOpOrder()
            UsdGeom.Xformable(drawer_prim).AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, 0.0))
            omni.physx.get_physx_interface().update_transform(drawer_path)
            print(f"[장애물 배치] X: {obs_x:.2f}, Y: {obs_y:.2f}")
            
        print("▶ Nav2 주행 감지 중... (도착 여부를 모니터링합니다)")
        
        success = False
        start_time = time.time()
        timeout = 60.0  # 한 판당 최대 제한 시간 (60초)
        
        # 실시간 위치 모니터링 루프
        while time.time() - start_time < timeout:
            sim_context.step(render=True)
            
            if robot_prim.IsValid():
                # 현재 로봇의 실시간 월드 좌표 추출
                pose = omni.usd.utils.get_world_transform_matrix(robot_prim)
                curr_x = pose.ExtractTranslation()[0]
                curr_y = pose.ExtractTranslation()[1]
                
                # 목적지까지의 오차 거리 계산
                dist_to_goal = math.dist([curr_x, curr_y], [GOAL_X, GOAL_Y])
                
                # 0.5m 이내로 접근하면 성공 판정 후 루프 탈출
                if dist_to_goal < 0.5:
                    success = True
                    print("🏆 [성공] 목적지 도달 완료!")
                    break
                    
        duration = time.time() - start_time
        if not success:
            print("❌ [실패] 제한 시간 초과")
            
        # 4. CSV 파일에 데이터 누적 기록
        with open(RESULT_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([trial, int(success), GOAL_X, f"{GOAL_Y:.2f}", f"{obs_x:.2f}", f"{obs_y:.2f}", f"{duration:.2f}"])
            
    simulation_app.close()

if __name__ == "__main__":
    main()
