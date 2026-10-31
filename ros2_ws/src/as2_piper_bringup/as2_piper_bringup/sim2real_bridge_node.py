# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Sim-to-Real Whole-Body ROS 2 Bridge for Unitree AS2 Quadruped + AgileX PiPER 6-DoF Arm.

Safety & Sim-to-Real Invariants:
1. Deterministic 48-D (AS2) and 64-D (AS2 + PiPER) observation ordering matching
   ``UnitreeAS2RoughEnvCfg`` / ``UnitreeAS2PiperRoughEnvCfg`` (``preserve_order=True``).
2. State liveness watchdog: low-level ONNX leg actuation & CAN arm commands only stream when
   ``/imu``, ``/odom``, and all 12 leg joints in ``/joint_states`` are fresh within ``state_timeout_s``.
3. Handle target latching: once ``DoorAutoOpener`` triggers on a fresh ``door_ready`` detection,
   the handle pose is latched and propagated via dead-reckoned base twist so the reaching PiPER arm
   entering the front camera ROI never corrupts the IK target mid-grasp.
4. Standalone deployment: resolves ``door_manipulation.py`` from ``share/as2_piper_bringup`` when
   installed on the onboard robot PC without the full Isaac Lab repository.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from typing import Any, Sequence

import numpy as np

AS2_LEG_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

AS2_DEFAULT_LEG_POS_12: tuple[float, ...] = (
    0.1,
    -0.1,
    0.1,
    -0.1,
    0.8,
    0.8,
    1.0,
    1.0,
    -1.5,
    -1.5,
    -1.5,
    -1.5,
)

PIPER_ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "joint8",
)

PIPER_DEFAULT_ARM_POS_8: tuple[float, ...] = (0.0, 0.6, -1.2, 0.0, 0.6, 0.0, 0.0, 0.0)


def load_door_manipulation_module(repo_root: str | None = None) -> Any:
    """Load ``door_manipulation.py`` from installed share directory, env var, or repo tree."""
    if "door_manipulation" in sys.modules:
        return sys.modules["door_manipulation"]

    candidate_files: list[str] = []

    # 1. Installed ROS 2 package share directory (standalone onboard robot PC)
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_share = get_package_share_directory("as2_piper_bringup")
        candidate_files.append(os.path.join(pkg_share, "door_manipulation.py"))
    except Exception:
        pass

    # 2. Explicit repo_root / AS2_PIPER_REPO_ROOT / parent directory search
    roots: list[str] = []
    if repo_root:
        roots.append(repo_root)
    if os.getenv("AS2_PIPER_REPO_ROOT"):
        roots.append(os.environ["AS2_PIPER_REPO_ROOT"])
    here = os.path.abspath(os.path.dirname(__file__))
    for depth in range(1, 7):
        roots.append(os.path.abspath(os.path.join(here, *([".."] * depth))))

    for root in roots:
        candidate_files.append(
            os.path.join(root, "source", "isaaclab_assets", "isaaclab_assets", "robots", "door_manipulation.py")
        )

    for mod_path in candidate_files:
        if os.path.isfile(mod_path):
            spec = importlib.util.spec_from_file_location("door_manipulation", mod_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules["door_manipulation"] = module
                spec.loader.exec_module(module)
                return module

    raise FileNotFoundError(
        "Could not locate 'door_manipulation.py' in share/as2_piper_bringup or repository tree. "
        "Re-run 'colcon build' or set AS2_PIPER_REPO_ROOT."
    )


def quat_rotate_inverse_wxyz(quat_wxyz: Sequence[float], vec_w: Sequence[float]) -> np.ndarray:
    """Rotate world-frame 3-vector ``vec_w`` into body frame using unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("Quaternion norm is zero.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = float(vec_w[0]), float(vec_w[1]), float(vec_w[2])

    # v' = v - 2*w*(q_vec x v) + 2*(q_vec x (q_vec x v))
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx

    ccx = y * cz - z * cy
    ccy = z * cx - x * cz
    ccz = x * cy - y * cx

    return np.array(
        [
            vx - 2.0 * w * cx + 2.0 * ccx,
            vy - 2.0 * w * cy + 2.0 * ccy,
            vz - 2.0 * w * cz + 2.0 * ccz,
        ],
        dtype=np.float32,
    )


def compute_projected_gravity_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Rotate world gravity ``(0, 0, -1)`` into robot ``base`` frame given quaternion ``(w, x, y, z)``."""
    return quat_rotate_inverse_wxyz(quat_wxyz, (0.0, 0.0, -1.0))


def propagate_latched_handle_in_base(
    handle_b: tuple[float, float, float],
    vx_b: float,
    vy_b: float,
    wz_b: float,
    dt: float,
) -> tuple[float, float, float]:
    """Update a latched target point in robot ``base`` frame under planar base motion ``(vx, vy, wz)``."""
    hx, hy, hz = handle_b
    dx = hx - vx_b * dt
    dy = hy - vy_b * dt
    c, s = math.cos(-wz_b * dt), math.sin(-wz_b * dt)
    return (c * dx - s * dy, s * dx + c * dy, hz)


def build_locomotion_observation(
    base_lin_vel_b: Sequence[float],
    base_ang_vel_b: Sequence[float],
    quat_wxyz: Sequence[float],
    cmd_vel_xy_yaw: Sequence[float],
    leg_pos_12: Sequence[float],
    leg_vel_12: Sequence[float],
    last_action_12: Sequence[float],
    arm_pos_8: Sequence[float] | None = None,
    arm_vel_8: Sequence[float] | None = None,
    expected_obs_dim: int = 64,
) -> np.ndarray:
    """Construct exact unscaled SI policy observation matching ``rough_env_cfg.py`` (``preserve_order=True``)."""
    lin_v = np.asarray(base_lin_vel_b, dtype=np.float32).reshape(3)
    ang_v = np.asarray(base_ang_vel_b, dtype=np.float32).reshape(3)
    proj_g = compute_projected_gravity_wxyz(quat_wxyz)
    cmd = np.asarray(cmd_vel_xy_yaw, dtype=np.float32).reshape(3)

    q_leg_rel = np.asarray(leg_pos_12, dtype=np.float32).reshape(12) - np.asarray(
        AS2_DEFAULT_LEG_POS_12, dtype=np.float32
    )
    dq_leg = np.asarray(leg_vel_12, dtype=np.float32).reshape(12)
    act = np.asarray(last_action_12, dtype=np.float32).reshape(12)

    if expected_obs_dim == 48:
        obs = np.concatenate([lin_v, ang_v, proj_g, cmd, q_leg_rel, dq_leg, act], axis=0)
    elif expected_obs_dim == 64:
        q_arm = (
            np.asarray(arm_pos_8, dtype=np.float32).reshape(8)
            if arm_pos_8 is not None
            else np.asarray(PIPER_DEFAULT_ARM_POS_8, dtype=np.float32)
        )
        q_arm_rel = q_arm - np.asarray(PIPER_DEFAULT_ARM_POS_8, dtype=np.float32)
        dq_arm = (
            np.asarray(arm_vel_8, dtype=np.float32).reshape(8)
            if arm_vel_8 is not None
            else np.zeros(8, dtype=np.float32)
        )
        q_all = np.concatenate([q_leg_rel, q_arm_rel], axis=0)
        dq_all = np.concatenate([dq_leg, dq_arm], axis=0)
        obs = np.concatenate([lin_v, ang_v, proj_g, cmd, q_all, dq_all, act], axis=0)
    else:
        raise ValueError(
            f"Unsupported policy obs dimension {expected_obs_dim}. Expected 48 (AS2 Flat) or 64 (AS2+PiPER Flat)."
        )
    return obs.astype(np.float32)


def actions_to_leg_joint_targets(
    raw_actions_12: Sequence[float],
    action_scale: float = 0.25,
    action_clip: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip raw policy output and convert to 12 leg joint position targets (radians)."""
    clipped = np.clip(np.asarray(raw_actions_12, dtype=np.float32).reshape(12), -action_clip, action_clip)
    targets = np.asarray(AS2_DEFAULT_LEG_POS_12, dtype=np.float32) + action_scale * clipped
    return targets, clipped


def _load_piper_sdk_class() -> Any:
    """Resolve ``C_PiperInterface_V2`` or ``C_PiperInterface`` across ``piper_sdk`` releases."""
    try:
        from piper_sdk import C_PiperInterface_V2

        return C_PiperInterface_V2
    except ImportError:
        from piper_sdk import C_PiperInterface

        return C_PiperInterface


def main(args: list[str] | None = None) -> None:
    """Entry point for ROS 2 Humble ``sim2real_bridge_node``."""
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, JointState
        from std_msgs.msg import Bool, String
    except ImportError as exc:
        raise SystemExit(
            f"[ERROR] ROS 2 Python packages (rclpy / sensor_msgs) are required to run this node: {exc}"
        ) from exc

    door_mod = load_door_manipulation_module()
    DoorAutoOpener = door_mod.DoorAutoOpener

    class AS2PiperSim2RealBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("as2_piper_sim2real_bridge")
            self.declare_parameter("onnx_policy_path", "")
            self.declare_parameter("door_type", "lever_push")
            self.declare_parameter("control_rate_hz", 50.0)
            self.declare_parameter("state_timeout_s", 0.25)
            self.declare_parameter("detection_timeout_s", 0.50)
            self.declare_parameter("enable_piper_can", False)
            self.declare_parameter("piper_can_port", "can0")

            self._dt = 1.0 / max(float(self.get_parameter("control_rate_hz").value), 1.0)
            self._door_type = str(self.get_parameter("door_type").value)
            self._door_opener = DoorAutoOpener(door_type=self._door_type)

            self._ort_session = None
            self._ort_input_name = ""
            self._obs_dim = 64
            onnx_path = str(self.get_parameter("onnx_policy_path").value).strip()
            if onnx_path:
                if not os.path.isfile(onnx_path):
                    raise FileNotFoundError(f"Configured onnx_policy_path does not exist: '{onnx_path}'")
                import onnxruntime as ort

                self._ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                inp = self._ort_session.get_inputs()[0]
                self._ort_input_name = inp.name
                self._obs_dim = int(inp.shape[-1])
                self.get_logger().info(f"Loaded ONNX locomotion policy ({self._obs_dim}-D) from {onnx_path}")
            else:
                self.get_logger().info(
                    "No onnx_policy_path set — running high-level /as2/cmd_vel_out + PiPER arm controller."
                )

            self._piper_hw = None
            if bool(self.get_parameter("enable_piper_can").value):
                can_port = str(self.get_parameter("piper_can_port").value)
                PiperCls = _load_piper_sdk_class()
                self._piper_hw = PiperCls(can_port)
                self._piper_hw.ConnectPort()
                enable_ok = self._piper_hw.EnableArm(7)
                if enable_ok is False:
                    raise RuntimeError(f"AgileX PiPER EnableArm(7) returned False on CAN port '{can_port}'.")
                self.get_logger().info(f"Connected to physical AgileX PiPER arm over CAN ({can_port}).")

            self._cmd_vel = [0.0, 0.0, 0.0]
            self._base_lin_vel = [0.0, 0.0, 0.0]
            self._base_ang_vel = [0.0, 0.0, 0.0]
            self._quat_wxyz = [1.0, 0.0, 0.0, 0.0]
            self._leg_pos = list(AS2_DEFAULT_LEG_POS_12)
            self._leg_vel = [0.0] * 12
            self._arm_pos = list(PIPER_DEFAULT_ARM_POS_8)
            self._arm_vel = [0.0] * 8
            self._arm_targets = list(PIPER_DEFAULT_ARM_POS_8)
            self._last_action = np.zeros(12, dtype=np.float32)

            # Liveness & detection freshness timestamps
            self._imu_stamp_s: float = 0.0
            self._odom_stamp_s: float = 0.0
            self._legs_stamp_s: float = 0.0
            self._detected_handle_b: tuple[float, float, float] | None = None
            self._handle_stamp_s: float = 0.0
            self._door_ready_flag: bool = False
            self._latched_handle_b: tuple[float, float, float] | None = None
            self._warned_stale_state: bool = False

            self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
            self.create_subscription(Imu, "/imu", self._on_imu, 10)
            self.create_subscription(Odometry, "/odom", self._on_odom, 10)
            self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
            self.create_subscription(PoseStamped, "/as2_piper/detected_handle_base", self._on_handle_pose, 10)
            self.create_subscription(Bool, "/as2_piper/door_ready", self._on_door_ready, 10)
            self.create_subscription(String, "/as2_piper/detected_door_type", self._on_door_type, 10)
            self.create_subscription(Bool, "/as2_piper/auto_door_trigger", self._on_auto_door_trigger, 10)
            self.create_subscription(JointState, "/piper/teleop_joint_targets", self._on_manual_arm, 10)

            self._cmd_vel_pub = self.create_publisher(Twist, "/as2/cmd_vel_out", 10)
            self._leg_cmd_pub = self.create_publisher(JointState, "/as2/leg_joint_commands", 10)
            self._arm_cmd_pub = self.create_publisher(JointState, "/piper/joint_commands", 10)

            self.create_timer(self._dt, self._control_step)
            self.get_logger().info("AS2 + PiPER Sim-to-Real Whole-Body Bridge active.")

        def _on_cmd_vel(self, msg: Twist) -> None:
            self._cmd_vel = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]

        def _on_imu(self, msg: Imu) -> None:
            self._quat_wxyz = [
                float(msg.orientation.w),
                float(msg.orientation.x),
                float(msg.orientation.y),
                float(msg.orientation.z),
            ]
            self._base_ang_vel = [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ]
            self._imu_stamp_s = time.monotonic()

        def _on_odom(self, msg: Odometry) -> None:
            v_raw = (
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.linear.z),
            )
            child = msg.child_frame_id.strip().lstrip("/")
            if child in ("base", "base_link", "trunk"):
                self._base_lin_vel = list(v_raw)
            else:
                # Rotate world-frame odometry linear velocity into robot base frame
                self._base_lin_vel = quat_rotate_inverse_wxyz(self._quat_wxyz, v_raw).tolist()
            self._odom_stamp_s = time.monotonic()

        def _on_joint_states(self, msg: JointState) -> None:
            pos_map = {n: p for n, p in zip(msg.name, msg.position)}
            vel_map = {n: v for n, v in zip(msg.name, msg.velocity)} if len(msg.velocity) == len(msg.name) else {}
            matched_legs = 0
            for i, name in enumerate(AS2_LEG_JOINT_NAMES):
                if name in pos_map:
                    self._leg_pos[i] = float(pos_map[name])
                    matched_legs += 1
                if name in vel_map:
                    self._leg_vel[i] = float(vel_map[name])
            if matched_legs == 12:
                self._legs_stamp_s = time.monotonic()
            for i, name in enumerate(PIPER_ARM_JOINT_NAMES):
                if name in pos_map:
                    self._arm_pos[i] = float(pos_map[name])
                if name in vel_map:
                    self._arm_vel[i] = float(vel_map[name])

        def _on_handle_pose(self, msg: PoseStamped) -> None:
            # Ignore live camera updates while PiPER arm is actively reaching across the front camera FOV
            if self._door_opener.active:
                return
            self._detected_handle_b = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            )
            self._handle_stamp_s = time.monotonic()

        def _on_door_ready(self, msg: Bool) -> None:
            self._door_ready_flag = bool(msg.data)
            if not self._door_ready_flag and not self._door_opener.active:
                # Clear stale handle if vision explicitly reports not ready
                pass

        def _on_door_type(self, msg: String) -> None:
            if not self._door_opener.active and msg.data in door_mod.DOOR_TYPES:
                if msg.data != self._door_type:
                    self._door_type = msg.data
                    self._door_opener = DoorAutoOpener(door_type=self._door_type)

        def _on_manual_arm(self, msg: JointState) -> None:
            if not self._door_opener.active and len(msg.position) == 8:
                self._arm_targets = [float(v) for v in msg.position]

        def _on_auto_door_trigger(self, msg: Bool) -> None:
            if not msg.data:
                if self._door_opener.active:
                    self._latched_handle_b = None
                    self.get_logger().info(self._door_opener.abort())
                return

            det_timeout = float(self.get_parameter("detection_timeout_s").value)
            age_s = time.monotonic() - self._handle_stamp_s
            if self._detected_handle_b is None or age_s > det_timeout or not self._door_ready_flag:
                self.get_logger().warning(
                    f"[AUTO-DOOR 거부] 최신 문 손잡이 근접 인식 정보가 없습니다 (ready={self._door_ready_flag}, age={age_s:.2f}s)."
                )
                return

            ok, status_msg = self._door_opener.try_trigger(
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), self._detected_handle_b, self._arm_targets
            )
            if ok:
                self._latched_handle_b = self._detected_handle_b
            self.get_logger().info(status_msg)

        def _states_are_fresh(self, now_s: float) -> bool:
            timeout = float(self.get_parameter("state_timeout_s").value)
            return (
                (now_s - self._imu_stamp_s <= timeout)
                and (now_s - self._odom_stamp_s <= timeout)
                and (now_s - self._legs_stamp_s <= timeout)
            )

        def publish_zero_twist(self) -> None:
            self._cmd_vel_pub.publish(Twist())

        def _control_step(self) -> None:
            now_s = time.monotonic()
            vx, vy, wz = self._cmd_vel

            # 1. Step proximity-gated automatic door manipulation using latched handle pose
            if self._door_opener.active and self._latched_handle_b is not None:
                self._latched_handle_b = propagate_latched_handle_in_base(
                    self._latched_handle_b,
                    vx_b=self._base_lin_vel[0] if (now_s - self._odom_stamp_s <= 0.5) else vx,
                    vy_b=self._base_lin_vel[1] if (now_s - self._odom_stamp_s <= 0.5) else vy,
                    wz_b=self._base_ang_vel[2] if (now_s - self._imu_stamp_s <= 0.5) else wz,
                    dt=self._dt,
                )
                arm_q8, assist_vel, _ = self._door_opener.step(
                    self._dt, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), self._latched_handle_b, self._arm_targets
                )
                self._arm_targets = list(arm_q8)
                if not self._door_opener.active:
                    self._latched_handle_b = None
                if abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(wz) < 1e-4:
                    vx, vy, wz = assist_vel

            now_stamp = self.get_clock().now().to_msg()

            # 2. Publish high-level velocity command
            twist_out = Twist()
            twist_out.linear.x = float(vx)
            twist_out.linear.y = float(vy)
            twist_out.angular.z = float(wz)
            self._cmd_vel_pub.publish(twist_out)

            # 3. Low-level ONNX locomotion policy (strictly gated on fresh IMU + Odom + 12-leg JointState)
            if self._ort_session is not None:
                if not self._states_are_fresh(now_s):
                    if not self._warned_stale_state:
                        self.get_logger().warning(
                            "Waiting for fresh /imu, /odom, and 12-DoF /joint_states before publishing ONNX leg targets."
                        )
                        self._warned_stale_state = True
                else:
                    self._warned_stale_state = False
                    obs = build_locomotion_observation(
                        base_lin_vel_b=self._base_lin_vel,
                        base_ang_vel_b=self._base_ang_vel,
                        quat_wxyz=self._quat_wxyz,
                        cmd_vel_xy_yaw=(vx, vy, wz),
                        leg_pos_12=self._leg_pos,
                        leg_vel_12=self._leg_vel,
                        last_action_12=self._last_action,
                        arm_pos_8=self._arm_pos,
                        arm_vel_8=self._arm_vel,
                        expected_obs_dim=self._obs_dim,
                    )
                    raw_act = self._ort_session.run(None, {self._ort_input_name: obs.reshape(1, -1)})[0][0]
                    leg_targets, self._last_action = actions_to_leg_joint_targets(raw_act)

                    leg_msg = JointState()
                    leg_msg.header.stamp = now_stamp
                    leg_msg.name = list(AS2_LEG_JOINT_NAMES)
                    leg_msg.position = [float(x) for x in leg_targets]
                    self._leg_cmd_pub.publish(leg_msg)

            # 4. Publish AgileX PiPER 8-DoF arm + symmetric gripper command
            arm_msg = JointState()
            arm_msg.header.stamp = now_stamp
            arm_msg.name = list(PIPER_ARM_JOINT_NAMES)
            arm_msg.position = [float(x) for x in self._arm_targets]
            self._arm_cmd_pub.publish(arm_msg)

            if self._piper_hw is not None:
                deg_001 = [int(round(math.degrees(rad) * 1000.0)) for rad in self._arm_targets[:6]]
                grip_um = int(round(max(0.0, self._arm_targets[6]) * 2.0 * 1_000_000.0))
                self._piper_hw.JointCtrl(*deg_001)
                self._piper_hw.GripperCtrl(grip_um, 1000, 0x01, 0)

    rclpy.init(args=args)
    node = AS2PiperSim2RealBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero_twist()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
