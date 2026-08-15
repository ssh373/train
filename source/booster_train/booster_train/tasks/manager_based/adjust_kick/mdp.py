"""MDP terms for autonomous approach, alignment, kick, and recovery."""

from __future__ import annotations

import math
import os

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

from booster_train.tasks.manager_based.adjust_kick import standalone_mdp as kick_mdp


def _curriculum_stage(
    env: ManagerBasedEnv,
    stage_steps: tuple[int, int, int],
) -> int:
    step = int(getattr(env, "common_step_counter", 0))
    if step < stage_steps[0]:
        return 0
    if step < stage_steps[1]:
        return 1
    if step < stage_steps[2]:
        return 2
    return 3


def reset_adjust_kick_scenario(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    stage_steps: tuple[int, int, int] = (100_000, 250_000, 500_000),
    visualize_target: bool = False,
    target_radius: float = 0.15,
):
    """Reset ball and target using a four-stage 360-degree curriculum."""
    stage = _curriculum_stage(env, stage_steps)
    distributions = (
        # Learn the current near-ball kick and a small adjustment first.
        ((0.25, 0.60), (-45.0, 45.0), (3.5, 4.5), (-30.0, 30.0)),
        # Add fast approach and side-on adjustment.
        ((0.40, 1.20), (-120.0, 120.0), (3.5, 5.5), (-90.0, 90.0)),
        # Full bearing with medium-distance approach and arbitrary targets.
        ((0.35, 2.00), (-180.0, 180.0), (3.5, 6.0), (-180.0, 180.0)),
        # Final deployment distribution: near/far ball, full 360-degree geometry.
        ((0.25, 3.00), (-180.0, 180.0), (3.0, 7.0), (-180.0, 180.0)),
    )
    ball_distance, ball_bearing, target_distance, target_angle = distributions[stage]
    kick_mdp.reset_ball_in_front(
        env,
        env_ids,
        x_range=ball_distance,
        y_range=(-1.0, 1.0),
        angle_range_deg=ball_bearing,
        height=0.105,
    )
    kick_mdp.reset_kick_target(
        env,
        env_ids,
        distance_range=target_distance,
        angle_range_deg=target_angle,
        visualize_target=visualize_target,
        target_radius=target_radius,
        origin_at_ball=True,
    )

    count = env.num_envs
    device = env.device
    if not hasattr(env, "_adjust_ready_latched"):
        env._adjust_ready_latched = torch.zeros(count, dtype=torch.bool, device=device)
        env._adjust_kick_elapsed = torch.zeros(count, device=device)
        env._adjust_prev_distance = torch.full((count,), torch.nan, device=device)
        env._adjust_prev_heading_error = torch.full((count,), torch.nan, device=device)
        env._adjust_curriculum_stage = torch.zeros(count, dtype=torch.long, device=device)
    env._adjust_ready_latched[env_ids] = False
    env._adjust_kick_elapsed[env_ids] = 0.0
    env._adjust_prev_distance[env_ids] = torch.nan
    env._adjust_prev_heading_error[env_ids] = torch.nan
    env._adjust_curriculum_stage[env_ids] = stage


def _adjust_geometry(
    env: ManagerBasedEnv,
    standoff: float = 0.32,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return desired-base error in body frame, distance, and heading error."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    target_w = kick_mdp._kick_target_w(env, (4.0, 0.0))
    direction_w = target_w - ball.data.root_pos_w[:, :2]
    direction_w = direction_w / torch.norm(direction_w, dim=1, keepdim=True).clamp_min(1.0e-6)
    desired_base_w = ball.data.root_pos_w[:, :2] - standoff * direction_w
    error_w = desired_base_w - robot.data.root_pos_w[:, :2]
    error_3d = torch.cat((error_w, torch.zeros(env.num_envs, 1, device=env.device)), dim=1)
    error_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), error_3d)[:, :2]
    distance = torch.norm(error_w, dim=1)
    desired_heading = torch.atan2(direction_w[:, 1], direction_w[:, 0])
    heading_error = wrap_to_pi(desired_heading - robot.data.heading_w)
    return error_b, distance, heading_error


def approach_heading_error(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Always keep the robot's forward axis pointed toward the ball."""
    ball_b = kick_mdp.ball_pos_b(env)
    return torch.atan2(ball_b[:, 1], ball_b[:, 0])


def _ready_latch(
    env: ManagerBasedEnv,
    position_tolerance: float = 0.16,
    heading_tolerance_deg: float = 12.0,
    ball_speed_tolerance: float = 0.08,
) -> torch.Tensor:
    if not hasattr(env, "_adjust_ready_latched"):
        env._adjust_ready_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    _, distance, heading_error = _adjust_geometry(env)
    ball: RigidObject = env.scene["ball"]
    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    ball_start = getattr(env, "_kick_ball_start_xy", ball.data.root_pos_w[:, :2])
    ball_displacement = torch.norm(ball.data.root_pos_w[:, :2] - ball_start, dim=1)
    ready_now = (
        (distance <= position_tolerance)
        & (heading_error.abs() <= math.radians(heading_tolerance_deg))
        & (ball_speed <= ball_speed_tolerance)
        & (ball_displacement <= 0.05)
    )
    env._adjust_ready_latched |= ready_now
    return env._adjust_ready_latched


def adjust_position_progress(env: ManagerBasedRLEnv, max_progress_per_step: float = 0.06) -> torch.Tensor:
    """Reward fast reduction of distance to the behind-ball pre-kick pose."""
    _, distance, _ = _adjust_geometry(env)
    if not hasattr(env, "_adjust_prev_distance"):
        env._adjust_prev_distance = distance.detach().clone()
        return torch.zeros_like(distance)
    previous = env._adjust_prev_distance
    progress = torch.where(torch.isfinite(previous), previous - distance, torch.zeros_like(distance))
    env._adjust_prev_distance.copy_(distance)
    active = ~_ready_latch(env)
    return (progress / max_progress_per_step).clamp(-1.0, 1.0) * active.float()


def adjust_heading_progress(env: ManagerBasedRLEnv, max_progress_deg_per_step: float = 8.0) -> torch.Tensor:
    """Reward fast reduction of heading error toward the ball."""
    heading_error = approach_heading_error(env)
    absolute_error = heading_error.abs()
    if not hasattr(env, "_adjust_prev_heading_error"):
        env._adjust_prev_heading_error = absolute_error.detach().clone()
        return torch.zeros_like(absolute_error)
    previous = env._adjust_prev_heading_error
    progress = torch.where(torch.isfinite(previous), previous - absolute_error, torch.zeros_like(absolute_error))
    env._adjust_prev_heading_error.copy_(absolute_error)
    active = ~_ready_latch(env)
    scale = math.radians(max_progress_deg_per_step)
    return (progress / scale).clamp(-1.0, 1.0) * active.float()


def face_ball_alignment(
    env: ManagerBasedRLEnv,
    heading_std_deg: float = 20.0,
) -> torch.Tensor:
    """Continuously reward facing the ball throughout approach and adjustment."""
    heading_error = approach_heading_error(env)
    quality = torch.exp(
        -(heading_error / math.radians(heading_std_deg)).square()
    )
    kick_happened = getattr(
        env,
        "_kick_happened",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    return quality * (~_ready_latch(env)).float() * (~kick_happened).float()


def fast_approach_velocity(env: ManagerBasedRLEnv, target_speed: float = 1.2) -> torch.Tensor:
    """Reward body velocity toward the desired pre-kick pose, capped at target speed."""
    robot: Articulation = env.scene["robot"]
    error_b, distance, _ = _adjust_geometry(env)
    direction_b = error_b / distance.unsqueeze(1).clamp_min(1.0e-6)
    toward = torch.sum(robot.data.root_lin_vel_b[:, :2] * direction_b, dim=1)
    active = ~_ready_latch(env)
    return (toward / target_speed).clamp(0.0, 1.0) * active.float()


def adjust_pose_accuracy(
    env: ManagerBasedRLEnv,
    position_std: float = 0.20,
    heading_std_deg: float = 15.0,
) -> torch.Tensor:
    """Reward precise placement behind the ball without exposing a phase observation."""
    _, distance, heading_error = _adjust_geometry(env)
    quality = torch.exp(
        -(distance / position_std).square()
        - (heading_error / math.radians(heading_std_deg)).square()
    )
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return quality * (~kick_happened).float()


def early_ball_motion(
    env: ManagerBasedRLEnv,
    speed_tolerance: float = 0.05,
    displacement_tolerance: float = 0.03,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize touching/moving the ball before the pre-kick pose is reached."""
    ball: RigidObject = env.scene[ball_cfg.name]
    speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    ball_start = getattr(env, "_kick_ball_start_xy", ball.data.root_pos_w[:, :2])
    displacement = torch.norm(ball.data.root_pos_w[:, :2] - ball_start, dim=1)
    not_ready = ~_ready_latch(env)
    speed_cost = (speed - speed_tolerance).clamp_min(0.0).square()
    displacement_cost = (displacement - displacement_tolerance).clamp_min(0.0).square()
    return (speed_cost + 4.0 * displacement_cost) * not_ready.float()


def early_foot_ball_proximity(
    env: ManagerBasedRLEnv,
    safe_distance: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Keep both feet away from the ball until position and heading are aligned."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    minimum_distance = torch.norm(feet - ball.data.root_pos_w[:, None, :], dim=2).min(dim=1).values
    not_ready = ~_ready_latch(env)
    return ((safe_distance - minimum_distance).clamp_min(0.0) / safe_distance).square() * not_ready.float()


def approach_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step cost until the robot reaches the valid pre-kick pose."""
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return ((~_ready_latch(env)) & (~kick_happened)).float()


def gated_kick_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return kick_mdp.ball_velocity_to_target(env, decay_distance=4.0, max_reward=10.0) * _ready_latch(env).float()


def gated_kick_accuracy(env: ManagerBasedRLEnv, std: float = 0.25) -> torch.Tensor:
    return kick_mdp.ball_target_accuracy(env, std=std) * _ready_latch(env).float()


def kick_direction_accuracy(env: ManagerBasedRLEnv, minimum_speed: float = 0.10) -> torch.Tensor:
    """Dense accuracy reward for the immediate ball velocity direction."""
    ball: RigidObject = env.scene["ball"]
    target_w = kick_mdp._kick_target_w(env, (4.0, 0.0))
    target_direction = target_w - ball.data.root_pos_w[:, :2]
    target_direction = target_direction / torch.norm(target_direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    velocity = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(velocity, dim=1)
    direction_score = torch.sum(velocity * target_direction, dim=1) / speed.clamp_min(1.0e-6)
    valid_kick = getattr(
        env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    active = _ready_latch(env) & valid_kick & (speed >= minimum_speed)
    return direction_score.clamp(-1.0, 1.0).sub(0.8).div(0.2).clamp(0.0, 1.0) * active.float()


def gated_kick_lateral_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return kick_mdp.ball_lateral_velocity(env) * _ready_latch(env).float()


def gated_kicking_foot_approach(
    env: ManagerBasedRLEnv,
    proximity_std: float = 0.1,
    stationary_speed: float = 0.1,
    velocity_weight: float = 0.3,
    center_deadband: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"]
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    reward = kick_mdp.kicking_foot_approach_ball(
        env,
        proximity_std=proximity_std,
        stationary_speed=stationary_speed,
        velocity_weight=velocity_weight,
        center_deadband=center_deadband,
        asset_cfg=asset_cfg,
        ball_cfg=ball_cfg,
    )
    return reward * _ready_latch(env).float()


def gated_kicking_foot_progress(
    env: ManagerBasedRLEnv,
    center_deadband: float = 0.03,
    max_progress_per_step: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"]
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    reward = kick_mdp.kicking_foot_approach_progress(
        env,
        center_deadband=center_deadband,
        max_progress_per_step=max_progress_per_step,
        asset_cfg=asset_cfg,
        ball_cfg=ball_cfg,
    )
    return reward * _ready_latch(env).float()


def gated_foot_kick_event(
    env: ManagerBasedRLEnv,
    speed_increase_threshold: float = 0.08,
    max_contact_distance: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Latch a valid kick only after alignment, without prescribing inside-foot contact."""
    event = kick_mdp.preferred_foot_kick_event(
        env,
        speed_increase_threshold=speed_increase_threshold,
        asset_cfg=asset_cfg,
        ball_cfg=ball_cfg,
    ).abs()
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    foot_distance = torch.norm(
        robot.data.body_pos_w[:, asset_cfg.body_ids] - ball.data.root_pos_w[:, None, :], dim=2
    ).min(dim=1).values
    ready_event = (
        (event > 0.0) & (foot_distance < max_contact_distance) & _ready_latch(env)
    )
    if not hasattr(env, "_kick_valid_foot_kick"):
        env._kick_valid_foot_kick = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._kick_valid_foot_kick |= ready_event
    return ready_event.float()


def adjusted_ball_success(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    target_radius: float = 0.15,
    min_direction_score: float = 0.98,
    max_speed: float = 2.5,
    recovery_time: float = 0.8,
    max_base_speed: float = 0.35,
    max_tilt: float = 0.2,
    max_mean_joint_deviation: float = 0.35,
) -> torch.Tensor:
    success = kick_mdp.ball_success(
        env,
        target_xy=target_xy,
        target_radius=target_radius,
        min_direction_score=min_direction_score,
        max_speed=max_speed,
        recovery_time=recovery_time,
        max_base_speed=max_base_speed,
        max_tilt=max_tilt,
        max_mean_joint_deviation=max_mean_joint_deviation,
    )
    return success & _ready_latch(env)


def adjusted_ball_not_kicked(
    env: ManagerBasedRLEnv,
    time_limit: float = 2.5,
    movement_speed: float = 0.12,
    min_direction_cos: float = 0.9396926208,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Start the no-kick timeout only after the geometric kick-ready latch."""
    ready = _ready_latch(env)
    if not hasattr(env, "_adjust_kick_elapsed"):
        env._adjust_kick_elapsed = torch.zeros(env.num_envs, device=env.device)
    env._adjust_kick_elapsed += ready.float() * env.step_dt

    ball: RigidObject = env.scene[ball_cfg.name]
    if not hasattr(env, "_kick_directional_happened"):
        env._kick_directional_happened = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    velocity = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(velocity, dim=1)
    target_dir = getattr(env, "_kick_direction_w", torch.zeros_like(velocity))
    direction_cos = torch.sum(velocity * target_dir, dim=1) / speed.clamp_min(1.0e-6)
    valid_foot_kick = getattr(
        env,
        "_kick_valid_foot_kick",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    env._kick_directional_happened |= (
        ready
        & valid_foot_kick
        & (speed > movement_speed)
        & (direction_cos >= min_direction_cos)
    )
    return (
        ready
        & (env._adjust_kick_elapsed > time_limit)
        & ~env._kick_directional_happened
    )


def walk_teacher_tracking(
    env: ManagerBasedRLEnv,
    teacher_env_var: str = "ADJUST_KICK_WALK_TEACHER_JIT",
    std: float = 0.20,
) -> torch.Tensor:
    """Imitate the validated walk teacher during approach."""
    action_term = env.action_manager.get_term("joint_pos")
    if hasattr(action_term, "student_processed_actions"):
        error = torch.mean(
            (action_term.student_processed_actions - action_term.teacher_target).square(), dim=1
        )
        return torch.exp(-error / (std * std)) * action_term.frozen_walk_active.float()

    bundled_teacher = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "locomotion_kick",
            "robots",
            "k1",
            "velocity_kick",
            "models",
            "velocity_teacher.pt",
        )
    )
    path = os.path.expanduser(os.environ.get(teacher_env_var, bundled_teacher))
    if not os.path.isfile(path):
        if not hasattr(env, "_adjust_teacher_warning_printed"):
            print(
                f"[adjust-kick] walk teacher not found at {path!r}; preservation reward is disabled.",
                flush=True,
            )
            env._adjust_teacher_warning_printed = True
        return torch.zeros(env.num_envs, device=env.device)

    robot: Articulation = env.scene["robot"]
    step = env.episode_length_buf.clone()
    if not hasattr(env, "_adjust_walk_teacher"):
        env._adjust_walk_teacher = torch.jit.load(path, map_location=env.device).eval()
        probe = env._adjust_walk_teacher(torch.zeros(1, 54, device=env.device))
        if tuple(probe.shape) != (1, 12):
            raise ValueError(f"{teacher_env_var} must accept 54 observations and return 12 actions.")
        env._adjust_teacher_last_action = torch.zeros(env.num_envs, 12, device=env.device)
        env._adjust_teacher_step = torch.full_like(step, -1)
        env._adjust_gait_phase = torch.zeros(env.num_envs, device=env.device)
        env._adjust_teacher_target = torch.zeros(env.num_envs, 12, device=env.device)

    if not torch.equal(step, env._adjust_teacher_step):
        error_b, distance, heading_error = _adjust_geometry(env)
        command = torch.zeros(env.num_envs, 3, device=env.device)
        command[:, 0] = (1.4 * error_b[:, 0]).clamp(-1.5, 1.5)
        command[:, 1] = (1.8 * error_b[:, 1]).clamp(-1.2, 1.2)
        command_heading = approach_heading_error(env)
        command[:, 2] = (2.5 * command_heading).clamp(-1.6, 1.6)
        close = distance < 0.45
        command[close, :2] *= (distance[close] / 0.45).unsqueeze(1)
        active = ~_ready_latch(env)
        env._adjust_gait_phase = torch.fmod(
            env._adjust_gait_phase + active.float() * env.step_dt * 2.0, 1.0
        )
        phase = torch.stack(
            (
                torch.cos(2.0 * math.pi * env._adjust_gait_phase),
                torch.sin(2.0 * math.pi * env._adjust_gait_phase),
            ),
            dim=1,
        )
        internal = torch.zeros(env.num_envs, 7, device=env.device)
        internal[:, 0] = 2.0
        q_rel = robot.data.joint_pos - robot.data.default_joint_pos
        obs = torch.cat(
            (
                command,
                internal,
                phase,
                robot.data.projected_gravity_b,
                robot.data.root_ang_vel_b,
                q_rel,
                robot.data.joint_vel,
                env._adjust_teacher_last_action,
            ),
            dim=1,
        )
        with torch.inference_mode():
            action = env._adjust_walk_teacher(obs)
        walk_default = robot.data.default_joint_pos.clone()
        env._adjust_teacher_target.copy_(walk_default + action)
        env._adjust_teacher_last_action.copy_(action)
        env._adjust_teacher_step.copy_(step)

    student_target = action_term.processed_actions
    error = torch.mean((student_target - env._adjust_teacher_target).square(), dim=1)
    active = (~_ready_latch(env)).float()
    return torch.exp(-error / (std * std)) * active


def kick_teacher_tracking(env: ManagerBasedRLEnv, std: float = 0.20) -> torch.Tensor:
    """Preserve the validated deploy kick after geometric alignment."""
    action_term = env.action_manager.get_term("joint_pos")
    if not hasattr(action_term, "kick_teacher_target"):
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.mean(
        (action_term.student_processed_actions - action_term.kick_teacher_target).square(),
        dim=1,
    )
    return torch.exp(-error / (std * std)) * action_term.kick_teacher_active.float()
