import argparse
import json
import csv
import time
import os
import numpy as np

# 1. AppLauncher를 사용하여 아이작 심 실행
from isaaclab.app import AppLauncher

# 실행 시 인자 처리
parser = argparse.ArgumentParser(description="Go2 Navigation Evaluation")
parser.add_argument("--scenario", type=str, default="scenarios.json", help="Scenario file")
parser.add_argument("--result", type=str, default="experiment_results.csv", help="Result file")
parser.add_argument("--map_path", type=str, default="/home/hayoung/workspaces/Go2.usd", help="USD Map path")

# 오타 수정된 부분: add_app_launcher_args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 아이작 심 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 이제부터 아이작 심 API 사용 가능 ---
# 최신 Isaac Sim 대응 임포트
import omni
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema

# 라이브러리 경로 변경 (omni.isaac -> isaacsim 혹은 다른 경로)
try:
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.nucleus import get_assets_root_path
except ImportError:
    # 하위 호환성 유지
    from omni.isaac.core.utils.extensions import enable_extension
    from omni.isaac.core.utils.nucleus import get_assets_root_path

# ROS2 Bridge 활성화 (최신 명칭 적용)
try:
    enable_extension("isaacsim.ros2.bridge")
except:
    enable_extension("omni.isaac.ros2_bridge")


class EvaluationManager:
    def __init__(self, map_path, scenario_file, result_file):
        # 맵 열기
        print(f"[INFO] 맵 로딩 중: {map_path}")
        omni.usd.get_context().open_stage(map_path)
        self.stage = omni.usd.get_context().get_stage()
        
        # 맵 로딩 기다리기
        for _ in range(100):
            simulation_app.update()

        self.scenario_file = scenario_file
        self.result_file = result_file

        # 경로 설정
        self.ROBOT_PATH = "/World/Robot" 
        self.PARENT_OBJ_PATH = "/World/Moveable_Objects/Drawer"
        self.MESH_OBJ_PATH = "/World/Moveable_Objects/Drawer/Drawer_Mesh"
        
        # 로봇 생성 확인
        self._ensure_robot()
        
        # 결과 파일 초기화
        if not os.path.exists(self.result_file):
            with open(self.result_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Trial", "Obs_X", "Obs_Y", "Obs_Yaw", "Result", "Time_Taken"])
        
        # 서랍장 충돌 설정
        self._setup_collision(self.MESH_OBJ_PATH)

    def _ensure_robot(self):
        prim = self.stage.GetPrimAtPath(self.ROBOT_PATH)
        if not prim.IsValid():
            print(f"[INFO] 로봇 생성 중...")
            assets_root = get_assets_root_path()
            robot_usd = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
            robot_prim = self.stage.DefinePrim(self.ROBOT_PATH, "Xform")
            robot_prim.GetReferences().AddReference(robot_usd)

    def _setup_collision(self, path):
        prim = self.stage.GetPrimAtPath(path)
        if prim.IsValid():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_api.CreateApproximationAttr().Set(PhysxSchema.Tokens.convexHull)

    def teleport_robot(self, pos):
        prim = self.stage.GetPrimAtPath(self.ROBOT_PATH)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2] + 0.1))

    def teleport_obstacle(self, pos, yaw):
        prim = self.stage.GetPrimAtPath(self.PARENT_OBJ_PATH)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
            xform.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, yaw))

    def get_robot_position(self):
        prim = self.stage.GetPrimAtPath(self.ROBOT_PATH)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            mat = xform.ComputeLocalToWorldTransform(0)
            pos = mat.ExtractTranslation()
            return [float(pos[0]), float(pos[1])]
        return [0.0, 0.0]

    def run(self):
        with open(self.scenario_file, "r") as f:
            scenarios = json.load(f)

        for trial_name, data in scenarios.items():
            print(f"\n[🚀 START] {trial_name}")
            self.teleport_obstacle(data["obstacle_pos"], data["obstacle_yaw"])
            self.teleport_robot(data["robot_start"])
            
            for _ in range(60):
                simulation_app.update()

            print(f"[WAIT] 목적지 {data['robot_goal']} 전송 대기 중...")
            start_time = time.time()
            result = "TIMEOUT"

            while time.time() - start_time < 60.0:
                simulation_app.update()
                pos = self.get_robot_position()
                dist = np.linalg.norm(np.array(pos) - np.array(data["robot_goal"][:2]))
                if dist < 0.5:
                    result = "SUCCESS"
                    break

            time_taken = round(time.time() - start_time, 2)
            print(f"[🏁 END] {result} ({time_taken}s)")

            with open(self.result_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([trial_name, data["obstacle_pos"][0], data["obstacle_pos"][1], data["obstacle_yaw"], result, time_taken])

if __name__ == "__main__":
    # map_path 인자를 명시적으로 전달
    manager = EvaluationManager(args_cli.map_path, args_cli.scenario, args_cli.result)
    manager.run()
    simulation_app.close()
