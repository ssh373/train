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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat

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
    """Absolute robot-yaw error to the ball-facing direction.

    During adjustment the robot should keep looking at the ball while it
    walks around it.  The kick direction is still used to compute the final
    alignment line, but it is not the heading target for the approach phase.
    """

    robot = env.scene["robot"]
    forward_local = torch.zeros(env.num_envs, 3, device=env.device)
    forward_local[:, 0] = 1.0
    forward_w = quat_apply(yaw_quat(robot.data.root_quat_w), forward_local)[:, :2]
    ball = env.scene["ball"]
    ball_direction_w = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    ball_direction_w = ball_direction_w / torch.norm(ball_direction_w, dim=1, keepdim=True).clamp_min(1.0e-6)
    dot = torch.sum(forward_w * ball_direction_w, dim=1).clamp(-1.0, 1.0)
    cross = forward_w[:, 0] * ball_direction_w[:, 1] - forward_w[:, 1] * ball_direction_w[:, 0]
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
    target_angle_magnitude_ranges_deg: tuple[tuple[float, float], ...],
    visualize_target: bool = False,
    # common_step_counter values corresponding to PPO iterations
    # 2,000, 5,000, 8,000, and 10,000 PPO iterations when
    # num_steps_per_env is 24.
    stage_steps: tuple[int, ...] = (48_000, 120_000, 192_000, 240_000),
    curriculum_stage: int = -1,
    ball_height: float = 0.105,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset a front ball and a curriculum-sampled 360-degree target.

    The desired robot state is the exact point 0.28 m behind the ball along
    the target direction.  Early curriculum stages sample a non-zero absolute
    target angle so standing still is not a useful shortcut.
    """

    robot = _robot(env, robot_cfg)
    ball = _ball(env, ball_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    count = len(env_ids)
    stage = _stage_index(env, stage_steps, len(target_angle_magnitude_ranges_deg), curriculum_stage)

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

    min_magnitude, max_magnitude = target_angle_magnitude_ranges_deg[stage]
    angle_magnitude = torch.empty(count, device=env.device).uniform_(min_magnitude, max_magnitude)
    angle_sign = torch.where(
        torch.rand(count, device=env.device) < 0.5,
        -torch.ones(count, device=env.device),
        torch.ones(count, device=env.device),
    )
    relative_angle = angle_sign * angle_magnitude * math.pi / 180.0
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
    target_radius: float = 0.28,
) -> torch.Tensor:
    """Distance to the exact kick-entry point behind the ball."""

    robot = env.scene["robot"]
    ball = env.scene["ball"]
    direction_w = _target_direction_w(env)
    alignment_point_w = ball.data.root_pos_w[:, :2] - target_radius * direction_w
    return torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - alignment_point_w, dim=1)


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
    """Reward the robot body facing the ball during adjustment."""

    std = math.radians(std_deg)
    error = heading_target_error(env)
    return torch.exp(-error.square() / (std * std))


def alignment_pose_reward(
    env: ManagerBasedRLEnv,
    heading_radius: float = 0.28,
    gain: float = 8.0,
) -> torch.Tensor:
    """GoTo-style unified SE(2) reward for the pre-kick pose.

    Position and ball-facing heading are optimized as one geometric objective
    instead of competing dense terms.  ``heading_radius`` converts heading
    error to the equivalent displacement on the 0.28 m ball orbit.
    """

    position_error = alignment_position_error(env)
    heading_error = heading_target_error(env)
    geometry = position_error.square() + 2.0 * heading_radius**2 * (
        1.0 - torch.cos(heading_error)
    )
    return torch.exp(-gain * geometry)


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


def step_progress_efficiency_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_progress_per_step: float = 0.18,
    maximum_progress_per_step: float = 0.30,
) -> torch.Tensor:
    """Reward large useful target progress per completed foot placement.

    Squaring normalized progress makes two productive large steps worth more
    than many small steps covering the same distance.  There is no direct step
    count penalty, so the term does not reward standing still or dragging both
    feet; the existing slide penalty handles the latter.
    """

    current_error = alignment_position_error(env)
    if not hasattr(env, "_adjust_last_touchdown_error"):
        env._adjust_last_touchdown_error = current_error.detach().clone()
        return torch.zeros_like(current_error)

    last_error = env._adjust_last_touchdown_error
    reset = env.episode_length_buf <= 1
    last_error[reset] = current_error[reset]

    sensor = env.scene.sensors[sensor_cfg.name]
    touchdown = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].any(dim=1)
    progress = (last_error - current_error).clamp(0.0, maximum_progress_per_step)
    # Do not saturate at the nominal target.  Productive progress keeps
    # earning a super-linear return up to the configured maximum, so one 0.30 m
    # placement is worth more than several small placements covering the same
    # total distance.
    efficiency = (progress / target_progress_per_step).square()
    reward = efficiency * touchdown.float()
    last_error[touchdown] = current_error[touchdown]
    return reward


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


def _adjust_gait_gate(
    env: ManagerBasedRLEnv,
    error_threshold: float = 0.08,
    speed_threshold: float = 0.03,
) -> torch.Tensor:
    """Enable walking-specific rewards only while the robot is travelling."""

    robot = env.scene["robot"]
    moving = torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1) > speed_threshold
    not_aligned = alignment_position_error(env) > error_threshold
    return (moving | not_aligned).float()


def adjust_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_air_time: float = 0.18,
    air_time_std: float = 0.10,
    max_air_time: float = 0.45,
    minimum_swing_height: float = 0.025,
) -> torch.Tensor:
    """Reward alternating single-support steps instead of base-only leaning.

    A reward is provided when exactly one foot supports the robot and the
    other foot has been in swing for a useful amount of time.  Touchdown gets
    an additional reward, which makes the policy complete a step rather than
    simply holding one foot in the air.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    body_ids = sensor_cfg.body_ids
    foot_ids = asset_cfg.body_ids
    contact_time = sensor.data.current_contact_time[:, body_ids]
    air_time = sensor.data.current_air_time[:, body_ids]
    in_contact = contact_time > 0.0
    foot_height = robot.data.body_pos_w[:, foot_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    in_swing = (~in_contact) & (foot_height > minimum_swing_height)
    single_support = (in_contact.sum(dim=1) == 1) & (in_swing.sum(dim=1) == 1)
    swing_time = torch.where(in_swing, air_time, torch.zeros_like(air_time)).max(dim=1).values
    swing_progress = (swing_time / target_air_time).clamp(0.0, 1.0)
    swing_reward = swing_progress * (swing_time <= max_air_time).float() * single_support.float()

    touchdown = sensor.compute_first_contact(env.step_dt)[:, body_ids]
    last_air_time = sensor.data.last_air_time[:, body_ids]
    touchdown_reward = torch.exp(-((last_air_time - target_air_time) / air_time_std).square())
    touchdown_reward = (touchdown_reward * touchdown.float()).sum(dim=1).clamp(max=1.0)

    return _adjust_gait_gate(env) * (0.7 * swing_reward + 0.3 * touchdown_reward)


def adjust_walking_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_air_time: float = 0.18,
    air_time_std: float = 0.10,
    target_clearance: float = 0.055,
    clearance_std: float = 0.025,
    swing_speed_scale: float = 0.15,
    minimum_swing_height: float = 0.035,
    alignment_gate_error: float = 0.08,
) -> torch.Tensor:
    """Evaluate the complete moving gait state with one dense term.

    The term covers support alternation, swing duration, swing-foot clearance,
    and horizontal foot motion together.  It does not depend on a special
    first-lift event, and it is disabled once the robot is already at the
    alignment point so the policy can settle there.
    """

    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    sensor_ids = sensor_cfg.body_ids
    foot_ids = asset_cfg.body_ids

    contact_time = sensor.data.current_contact_time[:, sensor_ids]
    air_time = sensor.data.current_air_time[:, sensor_ids]
    in_contact = contact_time > 0.0
    foot_height = robot.data.body_pos_w[:, foot_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    in_swing = (~in_contact) & (foot_height > minimum_swing_height)

    single_support = (in_contact.sum(dim=1) == 1) & (in_swing.sum(dim=1) == 1)
    swing_time = torch.where(in_swing, air_time, torch.zeros_like(air_time)).max(dim=1).values
    air_quality = torch.exp(-((swing_time - target_air_time) / air_time_std).square())
    clearance_quality = torch.exp(
        -((foot_height - target_clearance) / clearance_std).square()
    )
    clearance_quality = torch.where(in_swing, clearance_quality, torch.zeros_like(clearance_quality))
    clearance_quality = clearance_quality.max(dim=1).values

    foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, foot_ids, :2], dim=2)
    swing_speed_quality = torch.tanh(
        torch.where(in_swing, foot_speed, torch.zeros_like(foot_speed)).max(dim=1).values
        / swing_speed_scale
    )
    support_quality = single_support.float() * (
        0.35 * air_quality + 0.40 * clearance_quality + 0.25 * swing_speed_quality
    )
    active = (alignment_position_error(env) > alignment_gate_error).float()
    return active * support_quality


def low_foot_drag_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    drag_height: float = 0.045,
    speed_deadband: float = 0.03,
    speed_scale: float = 0.15,
    alignment_gate_error: float = 0.08,
    maximum_penalty: float = 4.0,
) -> torch.Tensor:
    """Penalize horizontal foot motion before the foot has cleared the floor.

    The regular contact-based slide term can miss toe skimming and weak
    grazing contacts.  This term is deliberately contact-independent: while
    travelling, any foot moving horizontally below ``drag_height`` pays a
    quadratic cost.  The policy can avoid it by lifting first, moving the foot
    through swing, and only then placing it down.
    """

    robot = env.scene[asset_cfg.name]
    foot_ids = asset_cfg.body_ids
    foot_height = (
        robot.data.body_pos_w[:, foot_ids, 2]
        - env.scene.env_origins[:, 2].unsqueeze(1)
    )
    foot_speed = torch.linalg.vector_norm(
        robot.data.body_lin_vel_w[:, foot_ids, :2], dim=2
    )
    low_height_gate = ((drag_height - foot_height) / drag_height).clamp(0.0, 1.0)
    moving_cost = (
        torch.relu(foot_speed - speed_deadband) / speed_scale
    ).square()
    drag_cost = torch.mean(low_height_gate * moving_cost, dim=1)
    active = (alignment_position_error(env) > alignment_gate_error).float()
    return active * drag_cost.clamp(max=maximum_penalty)


def adjust_foot_clearance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.045,
    height_std: float = 0.025,
) -> torch.Tensor:
    """Reward a moving foot clearing the ground by roughly 4--5 cm."""

    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene[asset_cfg.name]
    body_ids = sensor_cfg.body_ids
    foot_ids = asset_cfg.body_ids
    in_swing = sensor.data.current_air_time[:, body_ids] > 0.0
    foot_height = robot.data.body_pos_w[:, foot_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    clearance = torch.exp(-((foot_height - target_height) / height_std).square())
    return _adjust_gait_gate(env).unsqueeze(1).mul(clearance * in_swing.float()).sum(dim=1).clamp(max=1.0)


def adjust_feet_lateral_spacing_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    minimum_spacing: float = 0.20,
    maximum_spacing: float = 0.32,
    lower_violation_scale: float = 0.08,
    upper_violation_scale: float = 0.05,
) -> torch.Tensor:
    """Softly keep the moving stance inside a wide, usable interval.

    This is not a hard action limit.  The policy may leave the interval when
    that is useful, but very narrow or excessively splayed stances receive a
    growing penalty.
    """

    robot = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
    yaw = robot.data.heading_w
    lateral_spacing = -torch.sin(yaw) * delta_w[:, 0] + torch.cos(yaw) * delta_w[:, 1]
    width = torch.abs(lateral_spacing)
    lower_violation = torch.relu(minimum_spacing - width) / lower_violation_scale
    upper_violation = torch.relu(width - maximum_spacing) / upper_violation_scale
    return (lower_violation.square() + upper_violation.square()) * _adjust_gait_gate(env)


def adjust_feet_lateral_spacing_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_spacing: float = 0.22,
) -> torch.Tensor:
    """GoTo-style nominal lateral foot spacing in the trunk yaw frame."""

    robot = env.scene[asset_cfg.name]
    feet_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta_w = feet_w[:, 0] - feet_w[:, 1]
    yaw = robot.data.heading_w
    lateral_spacing = -torch.sin(yaw) * delta_w[:, 0] + torch.cos(yaw) * delta_w[:, 1]
    return (lateral_spacing - target_spacing).square()


def lower_leg_forward_alignment_penalty(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    ankle_roll_cfg: SceneEntityCfg,
    foot_yaw_free_deg: float = 15.0,
    foot_yaw_scale_deg: float = 10.0,
    ankle_roll_free_deg: float = 12.0,
    ankle_roll_scale_deg: float = 10.0,
    ankle_roll_weight: float = 0.5,
) -> torch.Tensor:
    """Softly keep both feet pointing forward without blocking useful steps.

    K1 has no ankle-yaw joint, so visible toe-in/toe-out is generated mainly
    by the hip-yaw chain.  Measuring each foot link relative to the trunk yaw
    catches the resulting whole-leg twist directly.  A separate ankle-roll
    guard prevents short corrections from being solved by folding an ankle.
    Both guards have a free region and are penalties rather than action clips.
    """

    robot = env.scene[feet_cfg.name]
    foot_quat = robot.data.body_quat_w[:, feet_cfg.body_ids]
    w, x, y, z = foot_quat.unbind(-1)
    foot_yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )
    yaw_error = torch.abs(wrap_to_pi(foot_yaw - robot.data.heading_w[:, None]))
    yaw_violation = torch.relu(yaw_error - math.radians(foot_yaw_free_deg))
    yaw_penalty = torch.mean(
        (yaw_violation / math.radians(foot_yaw_scale_deg)).square(), dim=1
    )

    ankle_roll = torch.abs(robot.data.joint_pos[:, ankle_roll_cfg.joint_ids])
    ankle_violation = torch.relu(
        ankle_roll - math.radians(ankle_roll_free_deg)
    )
    ankle_penalty = torch.mean(
        (ankle_violation / math.radians(ankle_roll_scale_deg)).square(), dim=1
    )
    return yaw_penalty + ankle_roll_weight * ankle_penalty


def alignment_stillness_penalty(
    env: ManagerBasedRLEnv,
    alignment_scale: float = 0.10,
    yaw_weight: float = 0.25,
) -> torch.Tensor:
    """Penalize residual base motion only inside the final alignment region."""

    robot = env.scene["robot"]
    error = alignment_position_error(env)
    near_goal = torch.exp(-((error / alignment_scale).square()))
    planar_speed = torch.sum(robot.data.root_lin_vel_b[:, :2].square(), dim=1)
    yaw_speed = robot.data.root_ang_vel_b[:, 2].square()
    return near_goal * (planar_speed + yaw_weight * yaw_speed)


def adjust_dynamic_base_height_l2(
    env: ManagerBasedRLEnv,
    arrival_height: float = 0.55,
    travel_height_drop: float = 0.03,
    upright_error: float = 0.06,
    full_drop_error: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use a shallow travelling posture and recover the kick-entry height.

    Far from the alignment point the target base height is lowered by roughly
    3 cm to make a long push-off and swing placement easier.  It blends back
    through the recovery band and is fully upright throughout the success
    position tolerance instead of teaching a permanently crouched gait.
    """

    robot = env.scene[asset_cfg.name]
    recovery_width = max(full_drop_error - upright_error, 1.0e-6)
    travel_ratio = (
        (alignment_position_error(env) - upright_error) / recovery_width
    ).clamp(0.0, 1.0)
    target_height = arrival_height - travel_height_drop * travel_ratio
    return (robot.data.root_pos_w[:, 2] - target_height).square()


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


def _alignment_ready(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    position_tolerance: float = 0.06,
    heading_tolerance_deg: float = 15.0,
    linear_speed_tolerance: float = 0.10,
    yaw_speed_tolerance: float = 0.10,
    contact_threshold: float = 1.0,
    ball_speed_tolerance: float = 0.05,
    ball_displacement_tolerance: float = 0.02,
) -> torch.Tensor:
    robot = env.scene["robot"]
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    grounded = torch.all(
        torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1).values
        > contact_threshold,
        dim=1,
    )
    position_ready = alignment_position_error(env) <= position_tolerance
    heading_ready = heading_target_error(env) <= math.radians(heading_tolerance_deg)
    motion_ready = (
        torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        <= linear_speed_tolerance
    ) & (torch.abs(robot.data.root_ang_vel_b[:, 2]) <= yaw_speed_tolerance)
    ball_ready = (ball_speed(env) <= ball_speed_tolerance) & (
        ball_displacement(env) <= ball_displacement_tolerance
    )
    return position_ready & heading_ready & motion_ready & grounded & ball_ready


def alignment_ready_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    position_tolerance: float = 0.06,
    heading_tolerance_deg: float = 15.0,
    linear_speed_tolerance: float = 0.10,
    yaw_speed_tolerance: float = 0.10,
    contact_threshold: float = 1.0,
    ball_speed_tolerance: float = 0.05,
    ball_displacement_tolerance: float = 0.02,
) -> torch.Tensor:
    """Reward a genuinely kick-ready, grounded and stationary arrival."""

    return _alignment_ready(
        env,
        sensor_cfg=sensor_cfg,
        position_tolerance=position_tolerance,
        heading_tolerance_deg=heading_tolerance_deg,
        linear_speed_tolerance=linear_speed_tolerance,
        yaw_speed_tolerance=yaw_speed_tolerance,
        contact_threshold=contact_threshold,
        ball_speed_tolerance=ball_speed_tolerance,
        ball_displacement_tolerance=ball_displacement_tolerance,
    ).float()


def alignment_success(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    position_tolerance: float = 0.06,
    heading_tolerance_deg: float = 15.0,
    linear_speed_tolerance: float = 0.10,
    yaw_speed_tolerance: float = 0.10,
    contact_threshold: float = 1.0,
    ball_speed_tolerance: float = 0.05,
    ball_displacement_tolerance: float = 0.02,
    stable_time: float = 1.50,
) -> torch.Tensor:
    ready = _alignment_ready(
        env,
        sensor_cfg=sensor_cfg,
        position_tolerance=position_tolerance,
        heading_tolerance_deg=heading_tolerance_deg,
        linear_speed_tolerance=linear_speed_tolerance,
        yaw_speed_tolerance=yaw_speed_tolerance,
        contact_threshold=contact_threshold,
        ball_speed_tolerance=ball_speed_tolerance,
        ball_displacement_tolerance=ball_displacement_tolerance,
    )
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
