# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematics, keyboard teleoperation state, and proximity-gated multi-type door auto-opening controller
for Unitree AS2 + AgileX PiPER.

This module intentionally avoids top-level Omniverse Kit imports so that forward/inverse kinematics,
proximity gating, and door-type state machines can be unit-tested offline as well as run inside Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

DOOR_TYPES = ("lever_push", "lever_pull", "sliding", "push_bar")

# Exact joint limits from robot_arms/piper/descriptions/as2_piper.urdf
PIPER_ARM_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.618, 2.168),   # joint1: base yaw
    (0.0, 3.14),       # joint2: shoulder pitch
    (-2.967, 0.0),     # joint3: elbow pitch
    (-1.745, 1.745),   # joint4: wrist roll
    (-1.22, 1.22),     # joint5: wrist pitch
    (-2.0944, 2.0944), # joint6: wrist yaw
)
PIPER_GRIPPER_MAX_OPEN: float = 0.035  # joint7 in [0, +0.035], joint8 in [-0.035, 0]

# Default tucked posture matching AS2_PIPER_CFG.init_state.joint_pos
PIPER_TUCKED_ARM_JOINTS: tuple[float, ...] = (0.0, 0.6, -1.2, 0.0, 0.6, 0.0)
PIPER_NOMINAL_REACH_JOINTS: tuple[float, ...] = (0.0, 0.95, -1.25, 0.0, 0.30, 0.0)

# Exact URDF kinematic chain from AS2 'base' to 'joint1..6' ((xyz), (rpy))
_MOUNT_XYZ = (0.10, 0.0, 0.13)
_PIPER_URDF_CHAIN: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),                    # joint1
    ((0.0, 0.0, 0.0), (1.5708, -0.1359, -3.1416)),           # joint2
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7939)),              # joint3
    ((-0.021984, -0.25075, 0.0), (1.5708, 0.0, 0.0)),        # joint4
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),                  # joint5
    ((0.000088, -0.091, 0.0), (1.5708, 0.0, 0.0)),           # joint6
)
_TCP_OFFSET_Z: float = 0.065  # gripper_base (0.04m) + finger grasp center (0.025m)


# ---------------------------------------------------------------------------
# Pure-math 3D SE(3) & Quaternion Utilities (w, x, y, z convention)
# ---------------------------------------------------------------------------


def _rot_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rot_z(theta: float) -> list[list[float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _mat3_vec(r: list[list[float]], v: Sequence[float]) -> tuple[float, float, float]:
    return (
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    )


def quat_to_rot_matrix(quat_wxyz: Sequence[float]) -> list[list[float]]:
    """Convert quaternion ``(w, x, y, z)`` into a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-9:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def world_to_body_point(
    point_w: Sequence[float],
    base_pos_w: Sequence[float],
    base_quat_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    """Transform a 3D world point into the robot's ``base`` coordinate frame."""
    r_wb = quat_to_rot_matrix(base_quat_wxyz)
    dx = point_w[0] - base_pos_w[0]
    dy = point_w[1] - base_pos_w[1]
    dz = point_w[2] - base_pos_w[2]
    # R_wb^T * d
    return (
        r_wb[0][0] * dx + r_wb[1][0] * dy + r_wb[2][0] * dz,
        r_wb[0][1] * dx + r_wb[1][1] * dy + r_wb[2][1] * dz,
        r_wb[0][2] * dx + r_wb[1][2] * dy + r_wb[2][2] * dz,
    )


def compute_door_handle_world_pos(
    door_pos_w: Sequence[float],
    door_quat_wxyz: Sequence[float],
    hinge_val: float = 0.0,
    door_type: str = "lever_push",
) -> tuple[float, float, float]:
    """Compute the live 3D world grasp center of the door handle given door state and mechanism."""
    if door_type == "sliding":
        # Prismatic rail along -X: handle starts at (0.80, -0.08, 0.97) in door frame
        local_pos = (0.80 - hinge_val, -0.08, 0.97)
    elif door_type == "push_bar":
        # Push pad contact zone centered on strike side of panel
        radius = 0.72
        local_pos = (
            radius * math.cos(hinge_val) + 0.04 * math.sin(hinge_val),
            radius * math.sin(hinge_val) - 0.04 * math.cos(hinge_val),
            0.92,
        )
    else:
        # 'lever_push' (hinge >= 0) or 'lever_pull' (hinge <= 0):
        # Lever grasp center in leaf frame is x=0.74 (0.80 - 0.06), y=-0.08 (-0.04 - 0.04), z=0.97
        lx, ly = 0.74, -0.08
        c, s = math.cos(hinge_val), math.sin(hinge_val)
        local_pos = (c * lx - s * ly, s * lx + c * ly, 0.97)

    r_wd = quat_to_rot_matrix(door_quat_wxyz)
    rot_p = _mat3_vec(r_wd, local_pos)
    return (
        door_pos_w[0] + rot_p[0],
        door_pos_w[1] + rot_p[1],
        door_pos_w[2] + rot_p[2],
    )


# ---------------------------------------------------------------------------
# AgileX PiPER 6-DoF Forward Kinematics & Analytical DLS Inverse Kinematics
# ---------------------------------------------------------------------------


def piper_forward_kinematics_base(
    q6: Sequence[float],
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Compute TCP position in AS2 ``base`` frame plus joint origins and Z-axes for Jacobian assembly."""
    pos = list(_MOUNT_XYZ)
    rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    joint_origins: list[tuple[float, float, float]] = []
    joint_axes: list[tuple[float, float, float]] = []

    for idx in range(6):
        xyz, rpy = _PIPER_URDF_CHAIN[idx]
        dp = _mat3_vec(rot, xyz)
        pos[0] += dp[0]
        pos[1] += dp[1]
        pos[2] += dp[2]
        rot = _mat3_mul(rot, _rot_rpy(rpy[0], rpy[1], rpy[2]))

        # Joint axis is local +Z before applying Rz(q_i)
        joint_origins.append((pos[0], pos[1], pos[2]))
        joint_axes.append((rot[0][2], rot[1][2], rot[2][2]))

        rot = _mat3_mul(rot, _rot_z(float(q6[idx])))

    tcp_dp = _mat3_vec(rot, (0.0, 0.0, _TCP_OFFSET_Z))
    tcp_pos = (pos[0] + tcp_dp[0], pos[1] + tcp_dp[1], pos[2] + tcp_dp[2])
    return tcp_pos, joint_origins, joint_axes


def piper_dls_ik(
    target_pos_b: Sequence[float],
    q_init: Sequence[float] | None = None,
    damping: float = 0.06,
    max_iters: int = 25,
    tol: float = 2e-3,
    nullspace_gain: float = 0.08,
) -> list[float]:
    """Solve 6-DOF PiPER joint angles for ``target_pos_b`` (in AS2 ``base`` frame) via Damped Least Squares."""
    q = list(q_init[:6]) if q_init is not None else list(PIPER_NOMINAL_REACH_JOINTS)
    lam2 = damping * damping

    for _ in range(max_iters):
        tcp, origins, axes = piper_forward_kinematics_base(q)
        err = [
            target_pos_b[0] - tcp[0],
            target_pos_b[1] - tcp[1],
            target_pos_b[2] - tcp[2],
        ]
        err_norm = math.sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2])
        if err_norm <= tol:
            break

        # Clamp per-iteration Cartesian step for smooth convergence on distant targets
        if err_norm > 0.08:
            scale = 0.08 / err_norm
            err = [e * scale for e in err]

        # Build 3x6 linear geometric Jacobian J_i = z_i x (p_tcp - p_i)
        j_mat = [[0.0] * 6 for _ in range(3)]
        for i in range(6):
            rx = tcp[0] - origins[i][0]
            ry = tcp[1] - origins[i][1]
            rz = tcp[2] - origins[i][2]
            zx, zy, zz = axes[i]
            j_mat[0][i] = zy * rz - zz * ry
            j_mat[1][i] = zz * rx - zx * rz
            j_mat[2][i] = zx * ry - zy * rx

        # Compute A = J * J^T + lambda^2 * I (3x3 symmetric positive definite)
        a = [[0.0] * 3 for _ in range(3)]
        for r in range(3):
            for c in range(3):
                a[r][c] = sum(j_mat[r][k] * j_mat[c][k] for k in range(6))
            a[r][r] += lam2

        # Solve 3x3 linear system A * y = err via Cramer's / Gaussian elimination
        y = _solve_3x3(a, err)

        # Primary task step dq_task = J^T * y + subtle posture regularization toward nominal elbow-up
        for i in range(6):
            dq_i = j_mat[0][i] * y[0] + j_mat[1][i] * y[1] + j_mat[2][i] * y[2]
            dq_null = nullspace_gain * (PIPER_NOMINAL_REACH_JOINTS[i] - q[i])
            q[i] = max(
                PIPER_ARM_JOINT_LIMITS[i][0],
                min(PIPER_ARM_JOINT_LIMITS[i][1], q[i] + dq_i + dq_null * 0.2),
            )

    return q


def _solve_3x3(a: list[list[float]], b: Sequence[float]) -> list[float]:
    """Solve a 3x3 SPD linear system ``A x = b`` with partial pivoting."""
    m = [[a[r][c] for c in range(3)] + [float(b[r])] for r in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        denom = m[col][col]
        if abs(denom) < 1e-12:
            return [0.0, 0.0, 0.0]
        for r in range(col + 1, 3):
            factor = m[r][col] / denom
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        s = m[r][3] - sum(m[r][c] * x[c] for c in range(r + 1, 3))
        x[r] = s / m[r][r]
    return x


# ---------------------------------------------------------------------------
# Simultaneous Keyboard Teleoperation State (WASD + Right-Hand Arm + Key '1')
# ---------------------------------------------------------------------------


@dataclass
class TeleopControlState:
    """Holds continuous keyboard rates for AS2 locomotion, PiPER 6-DOF arm + symmetric gripper, and triggers."""

    # Quadruped base velocity commands (W/S, A/D, Q/E)
    forward: float = 0.0
    side: float = 0.0
    yaw: float = 0.0

    # Arm joint velocity directions (-1.0, 0.0, +1.0) for joint1..joint6
    arm_delta_dir: list[float] = field(default_factory=lambda: [0.0] * 6)
    # Gripper opening direction (-1.0 = close 'X', +1.0 = open 'Z')
    gripper_dir: float = 0.0

    # Current desired 8-DOF arm + gripper target [joint1..joint6, joint7, joint8]
    arm_targets: list[float] = field(
        default_factory=lambda: list(PIPER_TUCKED_ARM_JOINTS) + [0.0, 0.0]
    )
    gripper_opening: float = 0.0

    # Active held key names to prevent opposite-key release glitches
    pressed_keys: set[str] = field(default_factory=set)

    # Edge-triggered events
    trigger_auto_door_requested: bool = False  # Key '1' / 'KEY_1' / 'NUMPAD_1'
    reset_arm_requested: bool = False          # Key 'C' / 'BACKSPACE'

    def update_from_key_event(self, key_name: str, is_press: bool, is_release: bool) -> None:
        """Update locomotion, arm, gripper, and macro trigger state from a normalized key event."""
        if not (is_press or is_release):
            return

        key = key_name.upper()
        if is_press:
            if key in ("KEY_1", "NUMPAD_1", "1"):
                self.trigger_auto_door_requested = True
                return
            if key in ("C", "BACKSPACE"):
                self.reset_arm_requested = True
                return
            self.pressed_keys.add(key)
        elif is_release:
            self.pressed_keys.discard(key)

        pk = self.pressed_keys
        # Left hand: AS2 quadruped locomotion
        self.forward = float("W" in pk) - float("S" in pk)
        self.side = float("A" in pk) - float("D" in pk)
        self.yaw = float("Q" in pk) - float("E" in pk)

        # Right hand: AgileX PiPER 6-DOF arm joints
        self.arm_delta_dir[0] = float("J" in pk) - float("L" in pk)  # joint1: base yaw (+left / -right)
        self.arm_delta_dir[1] = float("I" in pk) - float("K" in pk)  # joint2: shoulder (+reach / -retract)
        self.arm_delta_dir[2] = float("U" in pk) - float("O" in pk)  # joint3: elbow (+extend / -fold)
        self.arm_delta_dir[3] = float("F" in pk) - float("H" in pk)  # joint4: wrist roll
        self.arm_delta_dir[4] = float("T" in pk) - float("G" in pk)  # joint5: wrist pitch
        self.arm_delta_dir[5] = float("R" in pk) - float("Y" in pk)  # joint6: wrist yaw

        # Symmetric 2-finger gripper: Z (open), X (close)
        self.gripper_dir = float("Z" in pk) - float("X" in pk)

    def reset_arm_to_tucked(self) -> None:
        self.arm_targets = list(PIPER_TUCKED_ARM_JOINTS) + [0.0, 0.0]
        self.gripper_opening = 0.0

    def step_manual_arm(self, dt: float, joint_speed: float = 1.2, gripper_speed: float = 0.04) -> list[float]:
        """Integrate held manual arm/gripper keys over ``dt`` and enforce symmetric gripper & URDF limits."""
        if self.reset_arm_requested:
            self.reset_arm_to_tucked()
            self.reset_arm_requested = False
            return list(self.arm_targets)

        for idx in range(6):
            if self.arm_delta_dir[idx] != 0.0:
                low, high = PIPER_ARM_JOINT_LIMITS[idx]
                updated = self.arm_targets[idx] + self.arm_delta_dir[idx] * joint_speed * dt
                self.arm_targets[idx] = max(low, min(high, updated))

        if self.gripper_dir != 0.0:
            self.gripper_opening = max(
                0.0,
                min(PIPER_GRIPPER_MAX_OPEN, self.gripper_opening + self.gripper_dir * gripper_speed * dt),
            )
        # Enforce symmetric jaw travel: joint7 in [0, +0.035], joint8 in [-0.035, 0]
        self.arm_targets[6] = self.gripper_opening
        self.arm_targets[7] = -self.gripper_opening
        return list(self.arm_targets)


# ---------------------------------------------------------------------------
# Proximity-Gated Multi-Type Door Auto-Opener (Triggered by Key '1')
# ---------------------------------------------------------------------------


@dataclass
class ProximityCheckResult:
    """Result of checking whether the robot is close enough and aligned to activate auto-door."""

    can_activate: bool
    distance_xy: float
    heading_error_deg: float
    handle_pos_b: tuple[float, float, float]
    message: str


class DoorAutoOpener:
    """Proximity-gated, door-type-aware automatic door opening state machine triggered by key ``1``."""

    def __init__(
        self,
        door_type: str = "lever_push",
        min_distance_xy: float = 0.28,
        max_distance_xy: float = 0.90,
        max_heading_deg: float = 60.0,
    ) -> None:
        if door_type not in DOOR_TYPES:
            raise ValueError(f"Unsupported door_type '{door_type}'. Expected one of {DOOR_TYPES}.")
        self.door_type = door_type
        self.min_distance_xy = min_distance_xy
        self.max_distance_xy = max_distance_xy
        self.max_heading_deg = max_heading_deg

        self.active: bool = False
        self.elapsed: float = 0.0
        self._was_in_zone: bool = False
        self._start_arm_q: list[float] = list(PIPER_TUCKED_ARM_JOINTS)
        self._locked_handle_b: tuple[float, float, float] = (0.50, 0.0, 0.45)

    @property
    def total_duration(self) -> float:
        return 8.5 if self.door_type != "push_bar" else 7.0

    def check_proximity(
        self,
        base_pos_w: Sequence[float],
        base_quat_wxyz: Sequence[float],
        handle_pos_w: Sequence[float],
    ) -> ProximityCheckResult:
        """Evaluate distance & heading alignment between AS2 base and the door handle."""
        handle_b = world_to_body_point(handle_pos_w, base_pos_w, base_quat_wxyz)
        dist_xy = math.hypot(handle_b[0], handle_b[1])
        heading_err_deg = abs(math.degrees(math.atan2(handle_b[1], handle_b[0])))

        if dist_xy > self.max_distance_xy:
            return ProximityCheckResult(
                can_activate=False,
                distance_xy=dist_xy,
                heading_error_deg=heading_err_deg,
                handle_pos_b=handle_b,
                message=(
                    f"[AUTO-DOOR 비활성] 문 손잡이와 거리가 너무 멉니다 "
                    f"(현재: {dist_xy:.2f}m > 활성화 반경: {self.max_distance_xy:.2f}m). "
                    "WASD로 문 앞까지 접근해 주세요."
                ),
            )
        if dist_xy < self.min_distance_xy:
            return ProximityCheckResult(
                can_activate=False,
                distance_xy=dist_xy,
                heading_error_deg=heading_err_deg,
                handle_pos_b=handle_b,
                message=(
                    f"[AUTO-DOOR 비활성] 문에 너무 밀착되어 팔 작업 공간이 부족합니다 "
                    f"(현재: {dist_xy:.2f}m < 최소: {self.min_distance_xy:.2f}m). "
                    "S 키로 살짝 뒤로 물러나 주세요."
                ),
            )
        if heading_err_deg > self.max_heading_deg:
            return ProximityCheckResult(
                can_activate=False,
                distance_xy=dist_xy,
                heading_error_deg=heading_err_deg,
                handle_pos_b=handle_b,
                message=(
                    f"[AUTO-DOOR 비활성] 로봇이 문을 정면으로 바라보고 있지 않습니다 "
                    f"(방향 오차: {heading_err_deg:.1f}° > 허용: {self.max_heading_deg:.0f}°). "
                    "Q/E 키로 문 손잡이 방향을 맞춰 주세요."
                ),
            )

        return ProximityCheckResult(
            can_activate=True,
            distance_xy=dist_xy,
            heading_error_deg=heading_err_deg,
            handle_pos_b=handle_b,
            message=(
                f"[AUTO-DOOR 준비 완료] 문 감지됨 (타입: {self.door_type} | 거리: {dist_xy:.2f}m | "
                f"방향오차: {heading_err_deg:.1f}°) -> [1] 키를 누르면 자동 문 열기를 수행합니다."
            ),
        )

    def poll_zone_notification(
        self,
        base_pos_w: Sequence[float],
        base_quat_wxyz: Sequence[float],
        handle_pos_w: Sequence[float],
    ) -> str | None:
        """Return a one-shot HUD/console message when the robot newly enters the valid activation zone."""
        if self.active:
            return None
        res = self.check_proximity(base_pos_w, base_quat_wxyz, handle_pos_w)
        if res.can_activate and not self._was_in_zone:
            self._was_in_zone = True
            return res.message
        if not res.can_activate:
            self._was_in_zone = False
        return None

    def try_trigger(
        self,
        base_pos_w: Sequence[float],
        base_quat_wxyz: Sequence[float],
        handle_pos_w: Sequence[float],
        current_arm_q8: Sequence[float],
    ) -> tuple[bool, str]:
        """Attempt to start the door-opening sequence when key ``1`` is pressed."""
        if self.active:
            return False, f"[AUTO-DOOR] 이미 '{self.door_type}' 자동 문 열기가 진행 중입니다 ('C' 키로 중단 가능)."

        prox = self.check_proximity(base_pos_w, base_quat_wxyz, handle_pos_w)
        if not prox.can_activate:
            return False, prox.message

        self.active = True
        self.elapsed = 0.0
        self._start_arm_q = list(current_arm_q8[:6])
        self._locked_handle_b = prox.handle_pos_b
        return (
            True,
            f"[AUTO-DOOR 시작] '{self.door_type}' 자동 문 열기 시퀀스를 가동합니다 "
            f"(거리: {prox.distance_xy:.2f}m, 중단: 'C' 키).",
        )

    def abort(self) -> str:
        """Cancel active sequence immediately and return manual control."""
        self.active = False
        self.elapsed = 0.0
        return "[AUTO-DOOR 중단] 사용자 요청('C' 키)으로 자동 문 열기를 중단하고 기본 접힘 자세로 복귀합니다."

    def step(
        self,
        dt: float,
        base_pos_w: Sequence[float],
        base_quat_wxyz: Sequence[float],
        handle_pos_w: Sequence[float],
        current_arm_q8: Sequence[float],
    ) -> tuple[list[float], tuple[float, float, float], tuple[float, float]]:
        """Advance active door-opening sequence by ``dt``.

        Returns:
            - ``arm_targets_8``: ``[joint1..joint6, joint7, joint8]``
            - ``base_vel_cmd``: ``(lin_vel_x, lin_vel_y, ang_vel_z)`` coordinated locomotion assist
            - ``door_joint_targets``: ``(door_hinge_target, door_handle_target)``
        """
        if not self.active:
            return list(current_arm_q8), (0.0, 0.0, 0.0), (0.0, 0.0)

        self.elapsed += dt
        t = self.elapsed
        live_handle_b = world_to_body_point(handle_pos_w, base_pos_w, base_quat_wxyz)
        hx, hy, hz = self._locked_handle_b
        # Blend live handle tracking while clamped to safe PiPER reach [0.28m, 0.58m]
        tx = max(0.28, min(0.56, live_handle_b[0]))
        ty = max(-0.28, min(0.28, live_handle_b[1]))
        tz = max(0.20, min(0.58, live_handle_b[2]))

        if self.door_type == "lever_push":
            return self._step_lever_push(t, (hx, hy, hz), (tx, ty, tz), current_arm_q8)
        if self.door_type == "lever_pull":
            return self._step_lever_pull(t, (hx, hy, hz), (tx, ty, tz), current_arm_q8)
        if self.door_type == "sliding":
            return self._step_sliding(t, (hx, hy, hz), (tx, ty, tz), current_arm_q8)
        return self._step_push_bar(t, (hx, hy, hz), current_arm_q8)

    def _finish_if_done(self, t: float) -> None:
        if t >= self.total_duration:
            self.active = False
            self.elapsed = 0.0

    def _step_lever_push(
        self,
        t: float,
        init_h: tuple[float, float, float],
        live_h: tuple[float, float, float],
        cur_q8: Sequence[float],
    ) -> tuple[list[float], tuple[float, float, float], tuple[float, float]]:
        hx, hy, hz = init_h
        hx = max(0.32, min(0.54, hx))
        hy = max(-0.22, min(0.22, hy))
        hz = max(0.30, min(0.54, hz))

        if t < 1.5:
            # Phase 1: Open gripper & pre-grasp 8cm in front of lever
            alpha = t / 1.5
            target_b = (hx - 0.08 * (1.0 - alpha * 0.5), hy, hz)
            q6 = piper_dls_ik(target_b, cur_q8[:6])
            q6 = _blend_q(self._start_arm_q, q6, min(1.0, alpha * 1.2))
            grip = 0.032 * alpha
            return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.0, 0.0)

        if t < 2.8:
            # Phase 2: Reach onto lever, close gripper, depress lever downward to unlatch
            p = (t - 1.5) / 1.3
            depress = 0.05 * math.sin(min(1.0, p) * math.pi * 0.5)
            q6 = piper_dls_ik((hx, hy, hz - depress), cur_q8[:6])
            grip = max(0.008, 0.032 * (1.0 - p))
            handle_angle = 0.48 * p
            return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.0, handle_angle)

        if t < 6.5:
            # Phase 3: Push door open along smooth swing + gentle forward step
            p = (t - 2.8) / 3.7
            swing = 1.35 * (1.0 - math.cos(p * math.pi * 0.5))
            handle_angle = 0.48 * max(0.0, 1.0 - 1.5 * p)
            push_target = (
                min(0.56, live_h[0] + 0.03),
                live_h[1],
                max(0.34, hz - 0.03),
            )
            q6 = piper_dls_ik(push_target, cur_q8[:6])
            grip = 0.008 if p < 0.75 else 0.025
            vx = 0.18 * math.sin(p * math.pi)
            return q6 + [grip, -grip], (vx, 0.0, 0.0), (swing, handle_angle)

        # Phase 4 (6.5 ~ 8.5s): Release & retract smoothly to tucked pose (keep door open!)
        p = min(1.0, (t - 6.5) / 2.0)
        q6 = _blend_q(cur_q8[:6], PIPER_TUCKED_ARM_JOINTS, p)
        grip = 0.025 * (1.0 - p)
        self._finish_if_done(t)
        return q6 + [grip, -grip], (0.0, 0.0, 0.0), (1.35, 0.0)

    def _step_lever_pull(
        self,
        t: float,
        init_h: tuple[float, float, float],
        live_h: tuple[float, float, float],
        cur_q8: Sequence[float],
    ) -> tuple[list[float], tuple[float, float, float], tuple[float, float]]:
        hx = max(0.32, min(0.54, init_h[0]))
        hy = max(-0.22, min(0.22, init_h[1]))
        hz = max(0.30, min(0.54, init_h[2]))

        if t < 1.5:
            alpha = t / 1.5
            q6 = _blend_q(self._start_arm_q, piper_dls_ik((hx - 0.06, hy, hz), cur_q8[:6]), alpha)
            grip = 0.032 * alpha
            return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.0, 0.0)

        if t < 2.8:
            p = (t - 1.5) / 1.3
            q6 = piper_dls_ik((hx, hy, hz - 0.05 * p), cur_q8[:6])
            grip = max(0.006, 0.032 * (1.0 - p))
            return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.0, 0.50 * p)

        if t < 6.5:
            # Pull door toward -Y while stepping backward so door leaf clears quadruped body
            p = (t - 2.8) / 3.7
            swing = -1.25 * (1.0 - math.cos(p * math.pi * 0.5))
            handle_angle = 0.50 * max(0.0, 1.0 - 1.5 * p)
            pull_target = (
                max(0.30, live_h[0]),
                live_h[1],
                max(0.34, hz - 0.03),
            )
            q6 = piper_dls_ik(pull_target, cur_q8[:6])
            vx = -0.22 * math.sin(p * math.pi)
            vy = -0.06 * math.sin(p * math.pi)
            return q6 + [0.006, -0.006], (vx, vy, 0.0), (swing, handle_angle)

        p = min(1.0, (t - 6.5) / 2.0)
        q6 = _blend_q(cur_q8[:6], PIPER_TUCKED_ARM_JOINTS, p)
        grip = 0.025 * (1.0 - p)
        self._finish_if_done(t)
        return q6 + [grip, -grip], (0.0, 0.0, 0.0), (-1.25, 0.0)

    def _step_sliding(
        self,
        t: float,
        init_h: tuple[float, float, float],
        live_h: tuple[float, float, float],
        cur_q8: Sequence[float],
    ) -> tuple[list[float], tuple[float, float, float], tuple[float, float]]:
        hx = max(0.32, min(0.54, init_h[0]))
        hy = max(-0.22, min(0.22, init_h[1]))
        hz = max(0.30, min(0.54, init_h[2]))

        if t < 1.8:
            alpha = t / 1.8
            q6 = _blend_q(self._start_arm_q, piper_dls_ik((hx, hy, hz), cur_q8[:6]), alpha)
            # Rotate wrist roll (joint4) 90 deg for vertical bar handle
            q6[3] = 1.50 * alpha
            grip = 0.032 * (1.0 if alpha < 0.7 else (1.0 - (alpha - 0.7) / 0.3 * 0.75))
            return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.0, 0.0)

        if t < 6.5:
            p = (t - 1.8) / 4.7
            slide = 0.78 * (1.0 - math.cos(p * math.pi * 0.5))
            q6 = piper_dls_ik((live_h[0], live_h[1], hz), cur_q8[:6])
            q6[3] = 1.50
            # Coordinated lateral strafe along +Y (robot left = world -X slide direction)
            vy = 0.16 * math.sin(p * math.pi)
            return q6 + [0.008, -0.008], (0.0, vy, 0.0), (slide, 0.0)

        p = min(1.0, (t - 6.5) / 2.0)
        q6 = _blend_q(cur_q8[:6], PIPER_TUCKED_ARM_JOINTS, p)
        grip = 0.025 * (1.0 - p)
        self._finish_if_done(t)
        return q6 + [grip, -grip], (0.0, 0.0, 0.0), (0.78, 0.0)

    def _step_push_bar(
        self,
        t: float,
        init_h: tuple[float, float, float],
        cur_q8: Sequence[float],
    ) -> tuple[list[float], tuple[float, float, float], tuple[float, float]]:
        hx = max(0.32, min(0.52, init_h[0]))
        hy = max(-0.20, min(0.20, init_h[1]))
        hz = max(0.30, min(0.50, init_h[2]))

        if t < 1.5:
            # Closed gripper push-pad posture
            alpha = t / 1.5
            q6 = _blend_q(self._start_arm_q, piper_dls_ik((hx - 0.04, hy, hz), cur_q8[:6]), alpha)
            return q6 + [0.0, 0.0], (0.0, 0.0, 0.0), (0.0, 0.0)

        if t < 5.2:
            p = (t - 1.5) / 3.7
            swing = 1.35 * (1.0 - math.cos(p * math.pi * 0.5))
            q6 = piper_dls_ik((min(0.56, hx + 0.12 * p), hy, hz), cur_q8[:6])
            vx = 0.24 * math.sin(p * math.pi)
            return q6 + [0.0, 0.0], (vx, 0.0, 0.0), (swing, 0.0)

        p = min(1.0, (t - 5.2) / 1.8)
        q6 = _blend_q(cur_q8[:6], PIPER_TUCKED_ARM_JOINTS, p)
        self._finish_if_done(t)
        return q6 + [0.0, 0.0], (0.0, 0.0, 0.0), (1.35, 0.0)


def _blend_q(q_a: Sequence[float], q_b: Sequence[float], alpha: float) -> list[float]:
    a = max(0.0, min(1.0, alpha))
    return [(1.0 - a) * float(q_a[i]) + a * float(q_b[i]) for i in range(6)]
