# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantitative Benchmark & Ablation Evaluator for Unitree AS2 + AgileX PiPER (Capstone / Paper Ready).

Evaluates a trained RSL-RL checkpoint across two quantitative protocols and exports publication-ready
CSV + GitHub-Markdown tables (``logs/eval/ablation_results.csv`` & ``logs/eval/ablation_summary.md``):

1. **Whole-Body Locomotion Stability Ablation**:
   - ``locomotion_static_arm`` vs ``locomotion_dynamic_arm_swing``
2. **Proximity-Gated Multi-Door Manipulation Benchmark**:
   - Evaluates autonomous approach + Key ``1`` proximity auto-door across all 4 hospital door types
     (``lever_push``, ``lever_pull``, ``sliding``, ``push_bar``).
   - By default (``--no-assist_door``), the door hinge stiffness is set to ``0.0`` (passive damped door)
     so ``door_success_rate_pct`` measures genuine physical PiPER/body contact force opening.
   - Cleanly tears down PhysX timelines, Warp ray-caster caches, and USD stages between runs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import gc
import math
import os
from typing import Sequence

DOOR_OPEN_THRESHOLDS: dict[str, float] = {
    "lever_push": 0.70,   # rad (~40.1 deg)
    "lever_pull": 0.65,   # rad (~37.2 deg magnitude)
    "push_bar": 0.70,     # rad (~40.1 deg)
    "sliding": 0.45,      # meters along prismatic rail
}


@dataclass(frozen=True)
class BenchmarkRow:
    """Aggregated quantitative metrics for a single ablation condition or door type."""

    condition: str
    episodes: int
    lin_vel_rmse_mps: float
    ang_vel_rmse_radps: float
    roll_pitch_rms_deg: float
    leg_torque_rms_nm: float
    fall_rate_pct: float
    door_success_rate_pct: float | None = None
    mean_door_open_time_s: float | None = None


def quat_wxyz_to_roll_pitch_deg(quat_wxyz: Sequence[float]) -> tuple[float, float]:
    """Convert quaternion ``(w, x, y, z)`` into ``(roll_deg, pitch_deg)``."""
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return 0.0, 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return math.degrees(roll), math.degrees(pitch)


def is_door_open_success(door_type: str, hinge_position: float) -> bool:
    """Check whether articulated door joint displacement meets quantitative opening success."""
    if door_type not in DOOR_OPEN_THRESHOLDS:
        raise ValueError(f"Unknown door_type '{door_type}'. Expected one of {tuple(DOOR_OPEN_THRESHOLDS)}.")
    threshold = DOOR_OPEN_THRESHOLDS[door_type]
    if door_type == "lever_pull":
        return hinge_position <= -threshold
    return hinge_position >= threshold


def format_ablation_markdown_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render benchmark rows as a publication-ready Markdown summary table."""
    lines = [
        "# Unitree AS2 + AgileX PiPER Quantitative Evaluation Summary",
        "",
        "| Condition / Door Type | Episodes | Lin Vel RMSE (m/s) | Ang Vel RMSE (rad/s) | Roll/Pitch RMS (deg) | Leg Torque RMS (N·m) | Fall Rate (%) | Door Success (%) | Mean Open Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        succ_str = f"{r.door_success_rate_pct:.1f}" if r.door_success_rate_pct is not None else "N/A"
        time_str = f"{r.mean_door_open_time_s:.2f}" if r.mean_door_open_time_s is not None else "N/A"
        lines.append(
            f"| `{r.condition}` | {r.episodes} | {r.lin_vel_rmse_mps:.4f} | {r.ang_vel_rmse_radps:.4f} | "
            f"{r.roll_pitch_rms_deg:.2f} | {r.leg_torque_rms_nm:.2f} | {r.fall_rate_pct:.1f} | "
            f"{succ_str} | {time_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_benchmark_reports(rows: Sequence[BenchmarkRow], output_dir: str) -> tuple[str, str]:
    """Write ``ablation_results.csv`` and ``ablation_summary.md`` to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ablation_results.csv")
    md_path = os.path.join(output_dir, "ablation_summary.md")

    fieldnames = [
        "condition",
        "episodes",
        "lin_vel_rmse_mps",
        "ang_vel_rmse_radps",
        "roll_pitch_rms_deg",
        "leg_torque_rms_nm",
        "fall_rate_pct",
        "door_success_rate_pct",
        "mean_door_open_time_s",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="\n") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with open(md_path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(format_ablation_markdown_table(rows))

    return csv_path, md_path


def _run_isaac_evaluation() -> None:
    """Launch Isaac Sim and execute automated locomotion + 4-door quantitative evaluation."""
    import argparse
    import sys

    from isaaclab.app import AppLauncher

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rsl_rl_dir = os.path.join(repo_root, "scripts", "reinforcement_learning", "rsl_rl")
    if rsl_rl_dir not in sys.path:
        sys.path.insert(0, rsl_rl_dir)
    import cli_args  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(description="Quantitative benchmark for AS2 + PiPER loco-manipulation.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Velocity-Flat-Unitree-AS2-Piper-Play-v0",
        help="Registered AS2+PiPER Gym task to evaluate.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="RL agent configuration entry point.",
    )
    parser.add_argument(
        "--mode",
        choices=("locomotion_ablation", "door_benchmark", "all"),
        default="all",
        help="Which benchmark suite to execute.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per condition.")
    parser.add_argument(
        "--episode_steps",
        type=int,
        default=600,
        help="Steps per episode (at 50 Hz -> 12.0s, covering approach + full 8.5s door macro + retract).",
    )
    parser.add_argument(
        "--assist_door",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, drive door hinge via actuator targets (kinematic assist); default False tests pure physical contact.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(repo_root, "logs", "eval"),
        help="Output folder for CSV and Markdown benchmark reports.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata
    import gymnasium as gym
    import omni.timeline
    import omni.usd
    from packaging import version
    import torch
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab.assets import Articulation
    from isaaclab.sensors.ray_caster import RayCaster
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_assets.robots.door_manipulation import (
        DOOR_TYPES,
        PIPER_TUCKED_ARM_JOINTS,
        DoorAutoOpener,
        compute_door_handle_world_pos,
    )
    from isaaclab_assets.robots.hospital_door import get_hospital_door_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import load_cfg_from_registry

    installed_version = metadata.version("rsl-rl-lib")
    rsl_v4 = version.parse(installed_version) >= version.parse("4.0.0")

    def _reset_policy_memory(policy, runner, dones_tensor) -> None:
        if rsl_v4:
            policy.reset(dones_tensor)
        else:
            policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic
            policy_nn.reset(dones_tensor)

    def _teardown_env(vec_env, runner) -> None:
        """Stop PhysX timeline, clear Warp raycaster cache, and create a fresh USD stage between runs."""
        omni.timeline.get_timeline_interface().stop()
        vec_env.close()
        del vec_env, runner
        gc.collect()
        RayCaster.meshes.clear()
        omni.usd.get_context().new_stage()
        simulation_app.update()

    def _load_env_and_policy(door_type: str | None):
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.heading_command = False
            env_cfg.commands.base_velocity.rel_standing_envs = 0.0
            env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        if door_type is not None:
            door_cfg = get_hospital_door_cfg(door_type).replace(prim_path="{ENV_REGEX_NS}/Door")
            if not args_cli.assist_door and "door_panel" in door_cfg.actuators:
                # Passive unmotorized hinge (damped only) so physical PiPER contact forces must swing/slide the door
                door_cfg.actuators["door_panel"].stiffness = 0.0
            door_cfg.init_state.pos = (1.55, 0.74, 0.0)
            door_cfg.init_state.rot = (0.7071068, 0.0, 0.0, -0.7071068)
            env_cfg.scene.door = door_cfg
            if hasattr(env_cfg, "events") and getattr(env_cfg.events, "reset_base", None) is not None:
                env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = (
            retrieve_file_path(args_cli.checkpoint)
            if args_cli.checkpoint
            else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        )

        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
        runner = runner_cls(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        return vec_env, runner, policy

    results: list[BenchmarkRow] = []

    # 1. Locomotion Stability Ablation (Static Arm vs Dynamic Arm Swings)
    if args_cli.mode in ("locomotion_ablation", "all"):
        vec_env, runner, policy = _load_env_and_policy(door_type=None)
        robot: Articulation = vec_env.unwrapped.scene["robot"]
        arm_ids, _ = robot.find_joints(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"], preserve_order=True)
        if len(arm_ids) != 8:
            raise ValueError(
                f"Task '{args_cli.task}' has {len(arm_ids)} arm joints (expected 8 for AS2+PiPER ablation)."
            )
        leg_ids, _ = robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
        dt = vec_env.unwrapped.step_dt
        reset_mask = torch.ones(1, device=vec_env.unwrapped.device, dtype=torch.bool)

        for cond_name, swing_arm in (("locomotion_static_arm", False), ("locomotion_dynamic_arm_swing", True)):
            lin_sq_errs: list[float] = []
            ang_sq_errs: list[float] = []
            rp_sq_degs: list[float] = []
            torque_sq: list[float] = []
            falls = 0

            for _ in range(args_cli.episodes):
                vec_env.reset()
                _reset_policy_memory(policy, runner, reset_mask)
                fell_this_ep = False
                for step_i in range(args_cli.episode_steps):
                    t = step_i * dt
                    cmd_vx = 0.6
                    cmd_vy = 0.15 * math.sin(0.8 * t)
                    cmd_wz = 0.30 * math.cos(0.6 * t)

                    if swing_arm:
                        arm_q = [
                            0.75 * math.sin(1.4 * t),
                            0.85 + 0.45 * math.sin(1.1 * t),
                            -1.35 + 0.50 * math.cos(1.1 * t),
                            0.40 * math.sin(1.8 * t),
                            0.50 * math.cos(1.3 * t),
                            0.40 * math.sin(1.5 * t),
                            0.02,
                            -0.02,
                        ]
                    else:
                        arm_q = list(PIPER_TUCKED_ARM_JOINTS) + [0.0, 0.0]
                    arm_t = torch.tensor(arm_q, device=vec_env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
                    robot.set_joint_position_target(arm_t, joint_ids=arm_ids)

                    cmd_t = torch.tensor([[cmd_vx, cmd_vy, cmd_wz]], device=vec_env.unwrapped.device, dtype=torch.float32)
                    vec_env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd_t
                    with torch.inference_mode():
                        obs = vec_env.get_observations()
                        actions = policy(obs)
                        obs, _, dones, extras = vec_env.step(actions)
                        _reset_policy_memory(policy, runner, dones)

                    v_b = robot.data.root_lin_vel_b[0]
                    w_b = robot.data.root_ang_vel_b[0]
                    lin_sq_errs.append(float((v_b[0].item() - cmd_vx) ** 2 + (v_b[1].item() - cmd_vy) ** 2))
                    ang_sq_errs.append(float((w_b[2].item() - cmd_wz) ** 2))
                    roll_d, pitch_d = quat_wxyz_to_roll_pitch_deg(robot.data.root_quat_w[0].tolist())
                    rp_sq_degs.append(0.5 * (roll_d * roll_d + pitch_d * pitch_d))
                    if leg_ids:
                        tau = robot.data.applied_torque[0, leg_ids]
                        torque_sq.append(float(torch.mean(tau * tau).item()))

                    if bool(dones[0].item()):
                        time_outs = extras.get("time_outs", None) if isinstance(extras, dict) else None
                        if time_outs is None or not bool(time_outs[0].item()):
                            fell_this_ep = True
                            break
                if fell_this_ep:
                    falls += 1

            results.append(
                BenchmarkRow(
                    condition=cond_name,
                    episodes=args_cli.episodes,
                    lin_vel_rmse_mps=math.sqrt(sum(lin_sq_errs) / max(len(lin_sq_errs), 1)),
                    ang_vel_rmse_radps=math.sqrt(sum(ang_sq_errs) / max(len(ang_sq_errs), 1)),
                    roll_pitch_rms_deg=math.sqrt(sum(rp_sq_degs) / max(len(rp_sq_degs), 1)),
                    leg_torque_rms_nm=math.sqrt(sum(torque_sq) / max(len(torque_sq), 1)),
                    fall_rate_pct=100.0 * falls / max(args_cli.episodes, 1),
                )
            )
        _teardown_env(vec_env, runner)

    # 2. Multi-Door Autonomous Approach + Key '1' Proximity Opening Benchmark
    if args_cli.mode in ("door_benchmark", "all"):
        mode_suffix = "_assisted" if args_cli.assist_door else "_physical"
        for d_type in DOOR_TYPES:
            vec_env, runner, policy = _load_env_and_policy(door_type=d_type)
            robot = vec_env.unwrapped.scene["robot"]
            door: Articulation = vec_env.unwrapped.scene["door"]
            arm_ids, _ = robot.find_joints(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"], preserve_order=True)
            if len(arm_ids) != 8:
                raise ValueError(f"Task '{args_cli.task}' lacks 8-DoF PiPER arm joints required for door benchmark.")
            leg_ids, _ = robot.find_joints([".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
            hinge_ids, _ = door.find_joints(["door_hinge"])
            handle_ids, _ = door.find_joints(["door_handle_joint"])
            dt = vec_env.unwrapped.step_dt
            reset_mask = torch.ones(1, device=vec_env.unwrapped.device, dtype=torch.bool)

            lin_sq_errs = []
            ang_sq_errs = []
            rp_sq_degs = []
            torque_sq = []
            falls = 0
            successes = 0
            open_times: list[float] = []

            for _ in range(args_cli.episodes):
                vec_env.reset()
                _reset_policy_memory(policy, runner, reset_mask)
                opener = DoorAutoOpener(door_type=d_type)
                arm_targets = list(PIPER_TUCKED_ARM_JOINTS) + [0.0, 0.0]
                triggered = False
                opened_at_s: float | None = None
                trigger_time_s = 0.0
                fell_this_ep = False

                for step_i in range(args_cli.episode_steps):
                    t = step_i * dt
                    base_pos_w = robot.data.root_pos_w[0].tolist()
                    base_quat_w = robot.data.root_quat_w[0].tolist()
                    door_pos_w = door.data.root_pos_w[0].tolist()
                    door_quat_w = door.data.root_quat_w[0].tolist()
                    hinge_val = door.data.joint_pos[0, hinge_ids[0]].item() if hinge_ids else 0.0
                    handle_pos_w = compute_door_handle_world_pos(
                        door_pos_w, door_quat_w, hinge_val=hinge_val, door_type=d_type
                    )

                    if not triggered:
                        prox = opener.check_proximity(base_pos_w, base_quat_w, handle_pos_w)
                        if prox.can_activate:
                            ok, _ = opener.try_trigger(base_pos_w, base_quat_w, handle_pos_w, arm_targets)
                            if ok:
                                triggered = True
                                trigger_time_s = t
                        vx, vy, wz = (0.0, 0.0, 0.0) if triggered else (0.38, 0.0, 0.0)
                    else:
                        if opener.active:
                            arm_q8, (vx, vy, wz), (hinge_t, handle_t) = opener.step(
                                dt, base_pos_w, base_quat_w, handle_pos_w, arm_targets
                            )
                            arm_targets = list(arm_q8)
                            if args_cli.assist_door:
                                d_targets = door.data.default_joint_pos.clone()
                                if hinge_ids:
                                    d_targets[:, hinge_ids[0]] = hinge_t
                                if handle_ids:
                                    d_targets[:, handle_ids[0]] = handle_t
                                door.set_joint_position_target(d_targets)
                        else:
                            vx, vy, wz = (0.0, 0.0, 0.0)

                    arm_t = torch.tensor(arm_targets, device=vec_env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
                    robot.set_joint_position_target(arm_t, joint_ids=arm_ids)

                    cmd_t = torch.tensor([[vx, vy, wz]], device=vec_env.unwrapped.device, dtype=torch.float32)
                    vec_env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd_t
                    with torch.inference_mode():
                        obs = vec_env.get_observations()
                        actions = policy(obs)
                        obs, _, dones, extras = vec_env.step(actions)
                        _reset_policy_memory(policy, runner, dones)

                    v_b = robot.data.root_lin_vel_b[0]
                    w_b = robot.data.root_ang_vel_b[0]
                    lin_sq_errs.append(float((v_b[0].item() - vx) ** 2 + (v_b[1].item() - vy) ** 2))
                    ang_sq_errs.append(float((w_b[2].item() - wz) ** 2))
                    roll_d, pitch_d = quat_wxyz_to_roll_pitch_deg(robot.data.root_quat_w[0].tolist())
                    rp_sq_degs.append(0.5 * (roll_d * roll_d + pitch_d * pitch_d))
                    if leg_ids:
                        tau = robot.data.applied_torque[0, leg_ids]
                        torque_sq.append(float(torch.mean(tau * tau).item()))

                    cur_hinge = door.data.joint_pos[0, hinge_ids[0]].item() if hinge_ids else 0.0
                    if opened_at_s is None and is_door_open_success(d_type, cur_hinge):
                        opened_at_s = max(0.05, t - trigger_time_s)

                    if bool(dones[0].item()):
                        time_outs = extras.get("time_outs", None) if isinstance(extras, dict) else None
                        if time_outs is None or not bool(time_outs[0].item()):
                            fell_this_ep = True
                            break

                if fell_this_ep:
                    falls += 1
                if opened_at_s is not None and not fell_this_ep:
                    successes += 1
                    open_times.append(opened_at_s)

            results.append(
                BenchmarkRow(
                    condition=f"door_{d_type}{mode_suffix}",
                    episodes=args_cli.episodes,
                    lin_vel_rmse_mps=math.sqrt(sum(lin_sq_errs) / max(len(lin_sq_errs), 1)),
                    ang_vel_rmse_radps=math.sqrt(sum(ang_sq_errs) / max(len(ang_sq_errs), 1)),
                    roll_pitch_rms_deg=math.sqrt(sum(rp_sq_degs) / max(len(rp_sq_degs), 1)),
                    leg_torque_rms_nm=math.sqrt(sum(torque_sq) / max(len(torque_sq), 1)),
                    fall_rate_pct=100.0 * falls / max(args_cli.episodes, 1),
                    door_success_rate_pct=100.0 * successes / max(args_cli.episodes, 1),
                    mean_door_open_time_s=(sum(open_times) / len(open_times)) if open_times else None,
                )
            )
            _teardown_env(vec_env, runner)

    csv_path, md_path = write_benchmark_reports(results, args_cli.output_dir)
    print("\n" + format_ablation_markdown_table(results))
    print(f"[INFO] Saved CSV report     : {csv_path}")
    print(f"[INFO] Saved Markdown table : {md_path}")
    simulation_app.close()


if __name__ == "__main__":
    _run_isaac_evaluation()
