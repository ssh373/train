"""MDP terms for contact-free K1 target alignment.

The task ends at a stable pre-kick alignment.  It never rewards or requires a
ball launch; the next behavior can therefore hand the state to ``kick_001``.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

# Keep the locomotion/kick observation and regularization terms available under
# the same names used by kick_001.  Task-specific terms below are independent.
from booster_train.tasks.manager_based.kick.mdp import *  # noqa: F401,F403,E402
from booster_train.tasks.manager_based.kick.mdp import kick_mdp as _kick_mdp


def _robot(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def _ball(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg) -> RigidObject:
    return env.scene[ball_cfg.name]


def _target_direction_w(env: ManagerBasedEnv) -> torch.Tensor:
    if hasattr(env, "_adjust_target_w"):
        target_w = env._adjust_target_w
    else:
        target_w = env.scene.env_origins[:, :2] + torch.tensor(
            (4.0, 0.0), device=env.device
        )
    ball = env.scene["ball"]
    direction = target_w - ball.data.root_pos_w[:, :2]
    return direction / torch.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)


def ball_position_camera_b(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    base_noise_std: float = 0.03,
    distance_noise_ratio: float = 0.02,
    dropout_rate_per_s: float = 0.50,
    dropout_duration_range: tuple[float, float] = (0.08, 0.30),
) -> torch.Tensor:
    """Use the same camera-like ball observation contract as kick_001."""

    return _kick_mdp.camera_ball_pos_b(
        env,
        ball_cfg=ball_cfg,
        base_noise_std=base_noise_std,
        distance_noise_ratio=distance_noise_ratio,
        dropout_rate_per_s=dropout_rate_per_s,
        dropout_duration_range=dropout_duration_range,
    )


def target_direction_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Unit vector from the ball to the kick target in the robot yaw frame."""

    robot = env.scene["robot"]
    direction_w = torch.cat(
        (_target_direction_w(env), torch.zeros(env.num_envs, 1, device=env.device)), dim=1
    )
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), direction_w)[:, :2]


def heading_target_error(env: ManagerBasedEnv) -> torch.Tensor:
    """Absolute robot-yaw error to the ball-to-target direction."""

    robot = env.scene["robot"]
    forward_local = torch.zeros(env.num_envs, 3, device=env.device)
    forward_local[:, 0] = 1.0
    forward_w = quat_apply(yaw_quat(robot.data.root_quat_w), forward_local)[:, :2]
    target_w = _target_direction_w(env)
    dot = torch.sum(forward_w * target_w, dim=1).clamp(-1.0, 1.0)
    cross = forward_w[:, 0] * target_w[:, 1] - forward_w[:, 1] * target_w[:, 0]
    return torch.atan2(cross.abs(), dot)


def heading_target_error_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Column-vector form for the privileged critic observation."""

    return heading_target_error(env).unsqueeze(1)


def ball_velocity_b(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exact ball-relative velocity for the privileged critic."""

    robot = _robot(env, robot_cfg)
    ball = _ball(env, ball_cfg)
    velocity_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), velocity_w)[:, :2]


def ball_speed(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball = _ball(env, ball_cfg)
    return torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)


def ball_displacement(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball = _ball(env, ball_cfg)
    start = getattr(env, "_adjust_ball_start_xy", ball.data.root_pos_w[:, :2])
    return torch.norm(ball.data.root_pos_w[:, :2] - start, dim=1)


def ball_displacement_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Column-vector form for concatenated observation groups."""

    return ball_displacement(env).unsqueeze(1)


def task_phase(env: ManagerBasedEnv, value: float = 0.0) -> torch.Tensor:
    """Reserved phase slot for the future unified adjust->kick policy."""

    return torch.full((env.num_envs, 1), value, device=env.device)


def _stage_index(
    env: ManagerBasedEnv,
    stage_steps: tuple[int, ...],
    number_of_stages: int,
    curriculum_stage: int,
) -> int:
    if curriculum_stage >= 0:
        return min(curriculum_stage, number_of_stages - 1)
    step = int(getattr(env, "common_step_counter", 0))
    return min(sum(step >= threshold for threshold in stage_steps), number_of_stages - 1)


def reset_adjust_scenario(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ball_x_range: tuple[float, float],
    ball_y_range: tuple[float, float],
    target_distance_range: tuple[float, float],
    target_angle_ranges_deg: tuple[tuple[float, float], ...],
    visualize_target: bool = False,
    # common_step_counter values corresponding to PPO iterations
    # 2,000, 5,000, and 10,000 when num_steps_per_env is 24.
    stage_steps: tuple[int, ...] = (48_000, 120_000, 240_000),
    curriculum_stage: int = -1,
    ball_height: float = 0.105,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset a front ball and a curriculum-sampled 360-degree target.

    The desired robot state is a ball-relative line segment rather than a
    sampled point: behind the ball along the target direction, 0.15--0.35 m
    from the ball. The ball itself is always sampled at positive robot-frame x,
    so the first observation is always a front-ball scenario.
    """

    robot = _robot(env, robot_cfg)
    ball = _ball(env, ball_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = len(env_ids)
    stage = _stage_index(env, stage_steps, len(target_angle_ranges_deg), curriculum_stage)

    local = torch.zeros(count, 3, device=env.device)
    local[:, 0] = torch.empty(count, device=env.device).uniform_(*ball_x_range)
    local[:, 1] = torch.empty(count, device=env.device).uniform_(*ball_y_range)
    local[:, 2] = ball_height

    robot_yaw = yaw_quat(robot.data.root_quat_w[env_ids])
    w, z = robot_yaw[:, 0], robot_yaw[:, 3]
    cos_yaw = 1.0 - 2.0 * z.square()
    sin_yaw = 2.0 * w * z
    offset_w = local.clone()
    offset_w[:, 0] = cos_yaw * local[:, 0] - sin_yaw * local[:, 1]
    offset_w[:, 1] = sin_yaw * local[:, 0] + cos_yaw * local[:, 1]

    pose = ball.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = robot.data.root_pos_w[env_ids] + offset_w
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + ball_height
    velocity = torch.zeros(count, 6, device=env.device)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    if not hasattr(env, "_adjust_ball_start_xy"):
        env._adjust_ball_start_xy = torch.zeros(env.num_envs, 2, device=env.device)
    env._adjust_ball_start_xy[env_ids] = pose[:, :2]

    relative_angle = torch.empty(count, device=env.device).uniform_(*target_angle_ranges_deg[stage])
    relative_angle = relative_angle * math.pi / 180.0
    initial_yaw = torch.atan2(2.0 * robot_yaw[:, 0] * robot_yaw[:, 3], 1.0 - 2.0 * robot_yaw[:, 3].square())
    target_angle = initial_yaw + relative_angle
    target_distance = torch.empty(count, device=env.device).uniform_(*target_distance_range)
    direction_w = torch.stack((torch.cos(target_angle), torch.sin(target_angle)), dim=1)
    target_w = pose[:, :2] + target_distance.unsqueeze(1) * direction_w

    if not hasattr(env, "_adjust_target_w"):
        env._adjust_target_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._adjust_prev_alignment_error = torch.full((env.num_envs,), torch.nan, device=env.device)
        env._adjust_stable_time = torch.zeros(env.num_envs, device=env.device)
        env._adjust_alignment_achieved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._adjust_target_w[env_ids] = target_w
    env._adjust_prev_alignment_error[env_ids] = torch.nan
    env._adjust_stable_time[env_ids] = 0.0
    env._adjust_alignment_achieved[env_ids] = False
    env._adjust_curriculum_stage = stage

    if visualize_target:
        if not hasattr(env, "_adjust_target_visualizer"):
            env._adjust_target_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Adjust/target",
                    markers={
                        "target": sim_utils.CylinderCfg(
                            radius=0.18,
                            height=0.02,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.95, 0.08, 0.03),
                                emissive_color=(0.55, 0.02, 0.01),
                            ),
                        )
                    },
                )
            )
            env._adjust_direction_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Adjust/direction",
                    markers={
                        "direction": sim_utils.SphereCfg(
                            radius=0.08,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.75, 0.02),
                                emissive_color=(0.65, 0.35, 0.01),
                            ),
                        )
                    },
                )
            )
            env._adjust_alignment_visualizer = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Adjust/alignment",
                    markers={
                        "alignment": sim_utils.CylinderCfg(
                            radius=0.12,
                            height=0.025,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.05, 0.25, 1.0),
                                emissive_color=(0.01, 0.08, 0.65),
                            ),
                        )
                    },
                )
            )

        target_marker = torch.zeros(env.num_envs, 3, device=env.device)
        target_marker[:, :2] = env._adjust_target_w
        target_marker[:, 2] = env.scene.env_origins[:, 2] + 0.012

        direction_marker = torch.zeros(env.num_envs, 3, device=env.device)
        direction_marker[:, 2] = env.scene.env_origins[:, 2] + 0.035
        alignment_marker = torch.zeros(env.num_envs, 3, device=env.device)
        alignment_marker[:, 2] = env.scene.env_origins[:, 2] + 0.012
        if not hasattr(env, "_adjust_direction_marker_w"):
            env._adjust_direction_marker_w = torch.zeros(env.num_envs, 2, device=env.device)
            env._adjust_alignment_marker_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._adjust_direction_marker_w[env_ids] = pose[:, :2] + 0.8 * direction_w
        env._adjust_alignment_marker_w[env_ids] = pose[:, :2] - 0.28 * direction_w
        direction_marker[:, :2] = env._adjust_direction_marker_w
        alignment_marker[:, :2] = env._adjust_alignment_marker_w

        env._adjust_target_visualizer.visualize(target_marker)
        env._adjust_direction_visualizer.visualize(direction_marker)
        env._adjust_alignment_visualizer.visualize(alignment_marker)


def alignment_line_band_error(
    env: ManagerBasedRLEnv,
    minimum_distance: float = 0.15,
    maximum_distance: float = 0.35,
) -> torch.Tensor:
    """Distance to the target->ball line segment behind the ball."""

    robot = env.scene["robot"]
    ball = env.scene["ball"]
    direction_w = _target_direction_w(env)
    robot_from_ball = robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2]
    behind_distance = torch.sum(robot_from_ball * (-direction_w), dim=1)
    lateral_distance = torch.abs(
        robot_from_ball[:, 0] * direction_w[:, 1]
        - robot_from_ball[:, 1] * direction_w[:, 0]
    )
    distance_band_error = torch.where(
        behind_distance < minimum_distance,
        minimum_distance - behind_distance,
        (behind_distance - maximum_distance).clamp_min(0.0),
    )
    return torch.sqrt(lateral_distance.square() + distance_band_error.square())


def alignment_line_band_error_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """Column-vector form for concatenated observation groups."""

    return alignment_line_band_error(env).unsqueeze(1)


def alignment_position_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return alignment_line_band_error(env)


def alignment_position_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
) -> torch.Tensor:
    error = alignment_position_error(env)
    return torch.exp(-error.square() / (std * std))


def heading_target_reward(
    env: ManagerBasedRLEnv,
    std_deg: float = 15.0,
) -> torch.Tensor:
    """Reward the robot body facing the ball-to-target direction."""

    std = math.radians(std_deg)
    error = heading_target_error(env)
    return torch.exp(-error.square() / (std * std))


def alignment_position_progress(
    env: ManagerBasedRLEnv,
    max_progress_per_step: float = 0.04,
) -> torch.Tensor:
    current = alignment_position_error(env)
    if not hasattr(env, "_adjust_prev_alignment_error"):
        env._adjust_prev_alignment_error = current.detach().clone()
        return torch.zeros_like(current)
    previous = env._adjust_prev_alignment_error
    valid = torch.isfinite(previous)
    progress = torch.where(valid, previous - current, torch.zeros_like(current))
    env._adjust_prev_alignment_error.copy_(current)
    return (progress / max_progress_per_step).clamp(-1.0, 1.0)


def orbit_tangent_velocity_reward(
    env: ManagerBasedRLEnv,
    speed_scale: float = 0.25,
    angle_deadband: float = 0.12,
) -> torch.Tensor:
    """Reward moving around the ball in the direction of the final alignment.

    The shortest collision-free route to the target line is an arc around the
    ball.  This term rewards the robot's base velocity along that arc while
    the robot is still angularly misaligned.  The final line-segment reward
    remains responsible for stopping at the kick-ready position.
    """

    robot = env.scene["robot"]
    ball = env.scene["ball"]
    robot_from_ball = robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2]
    radius = torch.norm(robot_from_ball, dim=1, keepdim=True).clamp_min(1.0e-4)
    radial_unit = robot_from_ball / radius
    target_robot_direction = -_target_direction_w(env)

    # Signed shortest angular error from the current ball-centered radius to
    # the desired radius.  At the exact 180-degree ambiguity choose one side
    # deterministically so the policy still receives an orbit direction.
    cross = (
        radial_unit[:, 0] * target_robot_direction[:, 1]
        - radial_unit[:, 1] * target_robot_direction[:, 0]
    )
    dot = torch.sum(radial_unit * target_robot_direction, dim=1).clamp(-1.0, 1.0)
    angular_error = torch.atan2(cross, dot)
    turn_sign = torch.where(angular_error >= 0.0, 1.0, -1.0)
    tangent_unit = torch.stack((-radial_unit[:, 1], radial_unit[:, 0]), dim=1)
    tangent_unit = tangent_unit * turn_sign.unsqueeze(1)

    tangent_velocity = torch.sum(robot.data.root_lin_vel_w[:, :2] * tangent_unit, dim=1)
    active = (angular_error.abs() > angle_deadband).float()
    return (tangent_velocity / speed_scale).clamp(-1.0, 1.0) * active


def orbit_radius_reward(
    env: ManagerBasedRLEnv,
    target_radius: float = 0.28,
    radius_std: float = 0.08,
) -> torch.Tensor:
    """Prefer a safe ball-centered travel radius during the orbit."""

    robot = env.scene["robot"]
    ball = env.scene["ball"]
    radius = torch.norm(
        robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=1
    )
    return torch.exp(-((radius - target_radius) / radius_std).square())


def alignment_time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def ball_motion_penalty(
    env: ManagerBasedRLEnv,
    speed_scale: float = 0.08,
    displacement_scale: float = 0.02,
) -> torch.Tensor:
    speed = ball_speed(env)
    displacement = ball_displacement(env)
    return (speed / speed_scale + displacement / displacement_scale).clamp(0.0, 10.0)


def feet_ball_proximity_penalty(
    env: ManagerBasedRLEnv,
    safe_distance: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    ball = env.scene["ball"]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    distance = torch.norm(feet - ball.data.root_pos_w[:, None, :], dim=2).amin(dim=1)
    return ((safe_distance - distance).clamp_min(0.0) / safe_distance).square()


def alignment_success(
    env: ManagerBasedRLEnv,
    position_tolerance: float = 0.06,
    heading_tolerance_deg: float = 15.0,
    ball_speed_tolerance: float = 0.05,
    ball_displacement_tolerance: float = 0.02,
    stable_time: float = 1.50,
) -> torch.Tensor:
    position_ready = alignment_position_error(env) <= position_tolerance
    heading_ready = heading_target_error(env) <= math.radians(heading_tolerance_deg)
    ball_ready = (ball_speed(env) <= ball_speed_tolerance) & (
        ball_displacement(env) <= ball_displacement_tolerance
    )
    ready = position_ready & heading_ready & ball_ready
    if not hasattr(env, "_adjust_stable_time"):
        env._adjust_stable_time = torch.zeros(env.num_envs, device=env.device)
        env._adjust_alignment_achieved = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._adjust_stable_time += ready.float() * env.step_dt
    env._adjust_stable_time[~ready] = 0.0
    env._adjust_alignment_achieved |= env._adjust_stable_time >= stable_time
    return env._adjust_alignment_achieved


def ball_motion_termination(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.08,
    displacement_threshold: float = 0.02,
) -> torch.Tensor:
    """Fail the alignment episode when contact causes meaningful ball motion."""

    return (ball_speed(env) > speed_threshold) | (
        ball_displacement(env) > displacement_threshold
    )
