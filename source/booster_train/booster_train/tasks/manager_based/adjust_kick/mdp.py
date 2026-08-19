"""MDP terms for autonomous approach, alignment, kick, and recovery."""

from __future__ import annotations

import math

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
        # The ball always starts inside the kickable distance band. Difficulty
        # comes from progressively wider initial bearings, i.e. fast orbit and
        # heading alignment around the nearby ball. Long-range walking is not
        # silently assumed here because the supplied adjust teacher was not
        # trained as a 1.2 m approach controller.
        ((0.22, 0.38), (-15.0, 15.0), (3.5, 4.5), (-30.0, 30.0)),
        ((0.22, 0.38), (-60.0, 60.0), (3.5, 5.5), (-30.0, 30.0)),
        ((0.22, 0.38), (-120.0, 120.0), (3.5, 6.0), (-30.0, 30.0)),
        ((0.22, 0.38), (-180.0, 180.0), (3.0, 7.0), (-30.0, 30.0)),
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


def facing_translation_scale(
    heading_error: torch.Tensor,
    full_speed_deg: float = 15.0,
    stop_deg: float = 45.0,
) -> torch.Tensor:
    """Allow translation only while the ball remains near the forward axis."""
    error_deg = torch.rad2deg(heading_error.abs())
    return (
        (stop_deg - error_deg) / max(stop_deg - full_speed_deg, 1.0e-6)
    ).clamp(0.0, 1.0)


def _ready_latch(
    env: ManagerBasedEnv,
    standoff: float = 0.30,
    position_tolerance: float = 0.22,
    heading_tolerance_deg: float = 25.0,
    ball_speed_tolerance: float = 0.15,
    min_robot_ball_distance: float = 0.10,
    max_robot_ball_distance: float = 0.35,
    max_ball_displacement: float = 0.10,
) -> torch.Tensor:
    if not hasattr(env, "_adjust_ready_latched"):
        env._adjust_ready_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    _, distance, heading_error = _adjust_geometry(env, standoff=standoff)
    ball: RigidObject = env.scene["ball"]
    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    robot: Articulation = env.scene["robot"]
    robot_ball_distance = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=1
    )
    ball_start = getattr(env, "_kick_ball_start_xy", ball.data.root_pos_w[:, :2])
    ball_displacement = torch.norm(ball.data.root_pos_w[:, :2] - ball_start, dim=1)
    ready_now = (
        (distance <= position_tolerance)
        & (heading_error.abs() <= math.radians(heading_tolerance_deg))
        & (ball_speed <= ball_speed_tolerance)
        & (robot_ball_distance >= min_robot_ball_distance)
        & (robot_ball_distance <= max_robot_ball_distance)
        & (ball_displacement <= max_ball_displacement)
    )
    transition_active = getattr(
        env,
        "_adjust_transition_active",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    ready_now &= ~transition_active
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


def face_ball_violation(
    env: ManagerBasedRLEnv,
    tolerance_deg: float = 20.0,
    full_penalty_deg: float = 45.0,
) -> torch.Tensor:
    """Penalize turning the ball away from the robot's forward view after handoff."""
    action_term = env.action_manager.get_term("joint_pos")
    adjust_active = (
        ~action_term.frozen_walk_active
        if hasattr(action_term, "frozen_walk_active")
        else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    error_deg = torch.rad2deg(approach_heading_error(env).abs())
    violation = (
        (error_deg - tolerance_deg)
        / max(full_penalty_deg - tolerance_deg, 1.0e-6)
    ).clamp(0.0, 1.0)
    return violation * adjust_active.float() * (~_ready_latch(env)).float()


def lost_ball_heading(
    env: ManagerBasedRLEnv,
    max_heading_error_deg: float = 45.0,
) -> torch.Tensor:
    """Terminate adjustment if the robot turns the ball outside its forward view."""
    action_term = env.action_manager.get_term("joint_pos")
    if not hasattr(action_term, "frozen_walk_active"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    adjust_active = ~action_term.frozen_walk_active
    heading_lost = approach_heading_error(env).abs() > math.radians(max_heading_error_deg)
    return adjust_active & (~_ready_latch(env)) & heading_lost


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


def phase_base_height_l2(
    env: ManagerBasedRLEnv,
    pre_kick_target: float = 0.52,
    post_kick_target: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use a shallow crouch for adjustment/kick, then recover upright."""
    robot: Articulation = env.scene[asset_cfg.name]
    kick_happened = getattr(
        env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    target = torch.where(
        kick_happened,
        torch.full_like(robot.data.root_pos_w[:, 2], post_kick_target),
        torch.full_like(robot.data.root_pos_w[:, 2], pre_kick_target),
    )
    return (robot.data.root_pos_w[:, 2] - target).square()


def gated_kick_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return kick_mdp.ball_velocity_to_target(env, decay_distance=4.0, max_reward=10.0) * _ready_latch(env).float()


def direction_gated_kick_speed(
    env: ManagerBasedRLEnv,
    target_speed: float = 3.0,
    min_direction_score: float = 0.98,
) -> torch.Tensor:
    """Reward a strong kick only after its immediate direction is accurate."""
    ball: RigidObject = env.scene["ball"]
    target_w = kick_mdp._kick_target_w(env, (4.0, 0.0))
    target_direction = target_w - ball.data.root_pos_w[:, :2]
    target_direction = target_direction / torch.norm(target_direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    velocity = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(velocity, dim=1)
    direction_score = torch.sum(velocity * target_direction, dim=1) / speed.clamp_min(1.0e-6)
    direction_gate = ((direction_score - min_direction_score) / (1.0 - min_direction_score)).clamp(0.0, 1.0)
    valid_kick = getattr(
        env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    active = _ready_latch(env) & valid_kick
    return (speed / target_speed).clamp(0.0, 1.0) * direction_gate * active.float()


def gated_kick_accuracy(env: ManagerBasedRLEnv, std: float = 0.25) -> torch.Tensor:
    return kick_mdp.ball_target_accuracy(env, std=std) * _ready_latch(env).float()


def kick_direction_accuracy(env: ManagerBasedRLEnv, minimum_speed: float = 0.10) -> torch.Tensor:
    """Signed reward for the *actual* post-contact ball travel direction.

    Keep wrong-direction kicks negative instead of turning them into zero
    reward.  This makes the objective care about the launched ball velocity,
    independently of the robot's pre-kick yaw.
    """
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
    # score=+1 is a perfectly aligned launch and score=-1 is fully reversed.
    # The signed value is intentional: a fast miss must be worse than no kick.
    return direction_score.clamp(-1.0, 1.0) * active.float()


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
    min_direction_score: float = 0.995,
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


def composite_teacher_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    transition_multiplier: float = 3.0,
) -> torch.Tensor:
    """Preserve both experts and emphasize the only newly learned boundary.

    The comparison is in the teachers' original normalized 12-D action space,
    so joints with different physical action scales contribute consistently.
    """
    action_term = env.action_manager.get_term("joint_pos")
    if not hasattr(action_term, "teacher_action"):
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.mean(
        (action_term.student_action - action_term.teacher_action).square(), dim=1
    )
    reward = torch.exp(-error / (std * std))
    transition = getattr(
        action_term,
        "transition_active",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    return reward * torch.where(
        transition,
        torch.full_like(reward, transition_multiplier),
        torch.ones_like(reward),
    )


def ball_distance_band_penalty(
    env: ManagerBasedRLEnv,
    minimum_distance: float = 0.20,
    maximum_distance: float = 0.40,
    scale: float = 0.10,
    near_gate_distance: float = 0.60,
) -> torch.Tensor:
    """Enforce the 0.30 +/- 0.10 m band only near the ball.

    Far-away starts are intentional in the all-direction approach curriculum,
    so being 1 m away must not overwhelm the progress/approach objective. Once
    the robot enters the near-ball gate, the upper and lower band edges become
    active.
    """
    robot: Articulation = env.scene["robot"]
    ball: RigidObject = env.scene["ball"]
    distance = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=1
    )
    near_ball = distance <= near_gate_distance
    violation = torch.relu(minimum_distance - distance) + torch.relu(
        distance - maximum_distance
    ) * near_ball.float()
    action_term = env.action_manager.get_term("joint_pos")
    kick_phase = getattr(action_term, "phase", torch.zeros_like(distance)) == 2
    return (violation / scale).square() * (~kick_phase).float()
