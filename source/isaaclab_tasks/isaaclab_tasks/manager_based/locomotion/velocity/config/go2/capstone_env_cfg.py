from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from .rough_env_cfg import UnitreeGo2RoughEnvCfg

@configclass
class UnitreeGo2CapstoneEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 지형 로드: 하드코딩된 외부 USD 대신 기본 평지(Ground Plane) 사용
        from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        
        # 🎯 로봇 초기 위치 변경 (0.9m -> 0.35m로 낮춰 충격 완화)
        self.scene.robot.init_state.pos = (0.0, -0.5, 0.35) 
        self.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)

        # 환경 방해 요소 제거
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2CapstoneEnvCfg_PLAY(UnitreeGo2CapstoneEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        
        # 모든 외부 이벤트 제거 (로봇 떨림의 원인)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        
        # 로봇 초기 관절 고정
        self.scene.robot.init_state.joint_pos = {".*_hip_joint": 0.0, ".*_thigh_joint": 0.8, ".*_calf_joint": -1.5}
        
        # 액추에이터 강성 상향 (떠는 현상 방지)
        for name in self.scene.robot.actuators:
            self.scene.robot.actuators[name].stiffness = 250.0
            self.scene.robot.actuators[name].damping = 8.0
