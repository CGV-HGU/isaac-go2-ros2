import argparse
import json
import csv
import time
import os
import numpy as np

# 1. Isaac Lab 표준 앱 실행기
from isaaclab.app import AppLauncher

# 인자 설정
parser = argparse.ArgumentParser(description="Go2 Navigation Evaluation")
parser.add_argument("--scenario", type=str, default="scenarios.json", help="Scenario file")
parser.add_argument("--result", type=str, default="experiment_results.csv", help="Result file")
parser.add_argument("--usd_path", type=str, default="/home/hayoung/workspaces/Go2.usd", help="USD Map path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 이제부터 아이작 심 API 사용 가능 ---
import omni
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

# [중요] 최신 버전(isaacsim)과 이전 버전(omni.isaac) 모두 지원하는 임포트 로직
try:
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.nucleus import get_assets_root_path
except ImportError:
    try:
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.nucleus import get_assets_root_path
    except ImportError:
        print("[ERROR] Isaac Sim 핵심 라이브러리를 찾을 수 없습니다.")
        simulation_app.close()
        exit()

# ROS2 Bridge 활성화 (최신/이전 명칭 모두 시도)
for ext in ["isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"]:
    try:
        enable_extension(ext)
        print(f"[INFO] {ext} 활성화 성공")
        break
    except:
        continue

class EvaluationManager:
    def __init__(self, scenario_file, result_file):
        self.stage = omni.usd.get_context().get_stage()
        self.scenario_file = scenario_file
        self.result_file = result_file

        # 하영님 USD 내부 경로 (필요시 Stage 창 보고 수정)
        self.ROBOT_PATH = "/World/Robot" 
        self.PARENT_OBJ_PATH = "/World/Moveable_Objects/Drawer"
        
        # 로봇 생성 확인
        self._ensure_robot()

        # 결과 파일 초기화
        if not os.path.exists(self.result_file):
            with open(self.result_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Trial", "Obs_X", "Obs_Y", "Obs_Yaw", "Result", "Time_Taken"])

    def _ensure_robot(self):
        if not prim_utils.is_prim_path_valid(self.ROBOT_PATH):
            print(f"[INFO] 로봇 생성 중...")
            assets_root = get_assets_root_path()
            robot_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
            prim_utils.create_prim(self.ROBOT_PATH, usd_path=robot_usd)
            for _ in range(100): simulation_app.update()

    def teleport_robot(self, pos):
        if prim_utils.is_prim_path_valid(self.ROBOT_PATH):
            prim_utils.set_prim_translation(self.ROBOT_PATH, Gf.Vec3d(pos[0], pos[1], pos[2] + 0.3))

    def teleport_obstacle(self, pos, yaw):
        if prim_utils.is_prim_path_valid(self.PARENT_OBJ_PATH):
            prim_utils.set_prim_translation(self.PARENT_OBJ_PATH, Gf.Vec3d(pos[0], pos[1], pos[2]))
            prim_utils.set_prim_rotation(self.PARENT_OBJ_PATH, Gf.Vec3d(0, 0, yaw))

    def get_robot_position(self):
        if prim_utils.is_prim_path_valid(self.ROBOT_PATH):
            pos, _ = prim_utils.get_prim_world_pose(self.ROBOT_PATH)
            return [float(pos[0]), float(pos[1])]
        return [0.0, 0.0]

    def run(self):
        with open(self.scenario_file, "r") as f:
            scenarios = json.load(f)

        print(f"[INFO] 총 {len(scenarios)}개 실험 시작")

        for trial_name, data in scenarios.items():
            print(f"\n[🚀 START] {trial_name}")
            self.teleport_obstacle(data["obstacle_pos"], data["obstacle_yaw"])
            self.teleport_robot(data["robot_start"])
            
            for _ in range(100): simulation_app.update()

            print(f"[WAIT] 목적지 {data['robot_goal']} 대기 중 (다른 터미널에서 send_goal.py 실행)")
            start_time = time.time()
            result = "TIMEOUT"
            
            while time.time() - start_time < 60.0:
                simulation_app.update()
                pos = self.get_robot_position()
                dist = np.linalg.norm(np.array(pos) - np.array(data["robot_goal"][:2]))
                if dist < 0.6:
                    result = "SUCCESS"
                    break

            print(f"[🏁 END] {result} ({round(time.time()-start_time, 2)}s)")
            with open(self.result_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial_name, data["obstacle_pos"][0], data["obstacle_pos"][1], data["obstacle_yaw"], result, round(time.time()-start_time, 2)])

if __name__ == "__main__":
    # 맵 열기
    print(f"[INFO] 맵 로딩 중: {args_cli.usd_path}")
    omni.usd.get_context().open_stage(args_cli.usd_path)
    for _ in range(200): simulation_app.update()
    
    manager = EvaluationManager(args_cli.scenario, args_cli.result)
    manager.run()
    simulation_app.close()
