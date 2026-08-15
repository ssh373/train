"""Observations, rewards, events and terminations for K1 ball kicking."""

from __future__ import annotations

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


def external_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    push_interval_s: float,
    push_duration_s: float,
    force_randomization: dict,
    torque_randomization: dict,
):
    """Apply walk-compatible randomized force/torque disturbances."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device=asset.device)
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.device)
    if not hasattr(env, "pushing_forces"):
        env.pushing_forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=asset.device)
        env.pushing_torques = torch.zeros_like(env.pushing_forces)

    interval_steps = max(1, int(math.ceil(push_interval_s / env.step_dt)))
    duration_steps = max(1, int(math.ceil(push_duration_s / env.step_dt)))
    phase = int(env.common_step_counter) % interval_steps

    def sample(params: dict, shape: tuple[int, ...]) -> torch.Tensor:
        mean, std = params["range"]
        return mean + std * torch.randn(shape, device=asset.device)

    if phase == 0:
        env.pushing_forces[env_ids[:, None], body_ids, :] = sample(
            force_randomization, (len(env_ids), len(body_ids), 3)
        )
        env.pushing_torques[env_ids[:, None], body_ids, :] = sample(
            torque_randomization, (len(env_ids), len(body_ids), 3)
        )
    elif phase == duration_steps:
        env.pushing_forces[env_ids[:, None], body_ids, :].zero_()
        env.pushing_torques[env_ids[:, None], body_ids, :].zero_()

    asset.set_external_force_and_torque(
        env.pushing_forces[env_ids[:, None], body_ids, :],
        env.pushing_torques[env_ids[:, None], body_ids, :],
        env_ids=env_ids,
        body_ids=body_ids,
    )


def _kick_target_w(env: ManagerBasedEnv, fallback_xy: tuple[float, float]) -> torch.Tensor:
    """Return the per-environment target, or a fixed local fallback before reset."""
    if hasattr(env, "_kick_target_w"):
        return env._kick_target_w
    return env.scene.env_origins[:, :2] + torch.tensor(fallback_xy, device=env.device)


def ball_pos_b(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position relative to the robot, expressed in the robot yaw frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    delta_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)[:, :2]


def ball_vel_b(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball velocity relative to the robot, expressed in the robot yaw frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    delta_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)[:, :2]


def kick_target_pos_b(
    env: ManagerBasedEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Kick target relative to the robot, expressed in the robot yaw frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target_w = torch.zeros(env.num_envs, 3, device=env.device)
    target_w[:, :2] = _kick_target_w(env, target_xy)
    target_w[:, 2] = robot.data.root_pos_w[:, 2]
    delta_w = target_w - robot.data.root_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), delta_w)[:, :2]


def _camera_ball_observation(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    base_noise_std: float = 0.01,
    distance_noise_ratio: float = 0.01,
    dropout_rate_per_s: float = 0.35,
    dropout_duration_range: tuple[float, float] = (0.05, 0.25),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a cached noisy, intermittently missing camera observation.

    During a dropout the actor receives the last detected XY coordinate plus
    visibility=0, increasing age, and confidence=0.  Rewards and critic terms
    continue to use exact simulator state.
    """
    true_pos = ball_pos_b(env, ball_cfg=ball_cfg)[:, :2]
    episode_time = env.episode_length_buf.float() * env.step_dt

    if not hasattr(env, "_kick_camera_last_pos_b"):
        env._kick_camera_last_pos_b = true_pos.detach().clone()
        env._kick_camera_dropout_until = torch.zeros(env.num_envs, device=env.device)
        env._kick_camera_last_seen_time = episode_time.detach().clone()
        env._kick_camera_visible = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._kick_camera_confidence = torch.ones(env.num_envs, device=env.device)
        env._kick_camera_last_episode_length = torch.full(
            (env.num_envs,), -1, dtype=env.episode_length_buf.dtype, device=env.device
        )

    # Observation terms are evaluated separately; update the shared camera
    # state only once per environment step.
    if not torch.equal(env._kick_camera_last_episode_length, env.episode_length_buf):
        reset = env.episode_length_buf <= 1
        if reset.any():
            env._kick_camera_last_pos_b[reset] = true_pos[reset]
            env._kick_camera_dropout_until[reset] = 0.0
            env._kick_camera_last_seen_time[reset] = episode_time[reset]

        currently_visible = episode_time >= env._kick_camera_dropout_until
        start_dropout = currently_visible & ~reset & (
            torch.rand(env.num_envs, device=env.device) < dropout_rate_per_s * env.step_dt
        )
        if start_dropout.any():
            duration = torch.empty(env.num_envs, device=env.device).uniform_(*dropout_duration_range)
            env._kick_camera_dropout_until[start_dropout] = episode_time[start_dropout] + duration[start_dropout]

        visible = episode_time >= env._kick_camera_dropout_until
        distance = torch.norm(true_pos, dim=1, keepdim=True)
        noise_std = base_noise_std + distance_noise_ratio * distance
        noisy_pos = true_pos + torch.randn_like(true_pos) * noise_std
        env._kick_camera_last_pos_b[visible] = noisy_pos[visible]
        env._kick_camera_last_seen_time[visible] = episode_time[visible]
        env._kick_camera_visible.copy_(visible)
        confidence = 0.75 + 0.25 * torch.rand(env.num_envs, device=env.device)
        env._kick_camera_confidence.copy_(torch.where(visible, confidence, torch.zeros_like(confidence)))
        env._kick_camera_last_episode_length.copy_(env.episode_length_buf)

    age = (episode_time - env._kick_camera_last_seen_time).clamp(0.0, 0.3)
    return (
        env._kick_camera_last_pos_b,
        env._kick_camera_visible.float().unsqueeze(1),
        age.unsqueeze(1),
        env._kick_camera_confidence.unsqueeze(1),
    )


def camera_ball_pos_b(
    env: ManagerBasedEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    base_noise_std: float = 0.01,
    distance_noise_ratio: float = 0.01,
    dropout_rate_per_s: float = 0.35,
    dropout_duration_range: tuple[float, float] = (0.05, 0.25),
) -> torch.Tensor:
    return _camera_ball_observation(
        env,
        ball_cfg=ball_cfg,
        base_noise_std=base_noise_std,
        distance_noise_ratio=distance_noise_ratio,
        dropout_rate_per_s=dropout_rate_per_s,
        dropout_duration_range=dropout_duration_range,
    )[0]


def ball_visible(env: ManagerBasedEnv) -> torch.Tensor:
    return _camera_ball_observation(env)[1]


def ball_time_since_seen(env: ManagerBasedEnv) -> torch.Tensor:
    return _camera_ball_observation(env)[2]


def ball_confidence(env: ManagerBasedEnv) -> torch.Tensor:
    return _camera_ball_observation(env)[3]


def ball_pos_w(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_pos_w - env.scene.env_origins


def ball_vel_w(env: ManagerBasedEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_lin_vel_w


def feet_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    delta_w = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w[:, None, :]
    quat = yaw_quat(robot.data.root_quat_w)[:, None, :].expand(-1, len(asset_cfg.body_ids), -1)
    return quat_apply_inverse(quat.reshape(-1, 4), delta_w.reshape(-1, 3)).reshape(env.num_envs, -1)


def reset_ball_in_front(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    angle_range_deg: tuple[float, float] | None = None,
    height: float = 0.055,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Reset the ball at a randomized position in front of each robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    count = len(env_ids)
    local = torch.zeros(count, 3, device=env.device)
    if angle_range_deg is None:
        local[:, 0].uniform_(*x_range)
        local[:, 1].uniform_(*y_range)
    else:
        distance = torch.empty(count, device=env.device).uniform_(*x_range)
        angle = torch.empty(count, device=env.device).uniform_(*angle_range_deg) * torch.pi / 180.0
        local[:, 0] = distance * torch.cos(angle)
        local[:, 1] = distance * torch.sin(angle)
    local[:, 2] = height

    robot_quat = yaw_quat(robot.data.root_quat_w[env_ids])
    # Rotate only x/y without depending on a quaternion helper's broadcasting rules.
    w, z = robot_quat[:, 0], robot_quat[:, 3]
    cos_yaw = 1.0 - 2.0 * z.square()
    sin_yaw = 2.0 * w * z
    offset_w = local.clone()
    offset_w[:, 0] = cos_yaw * local[:, 0] - sin_yaw * local[:, 1]
    offset_w[:, 1] = sin_yaw * local[:, 0] + cos_yaw * local[:, 1]

    pose = ball.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = robot.data.root_pos_w[env_ids] + offset_w
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + height
    velocity = torch.zeros(count, 6, device=env.device)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    if not hasattr(env, "_kick_preferred_foot"):
        env._kick_preferred_foot = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # Body-frame +y is left (index 0), -y is right (index 1).
    side_preference = torch.where(local[:, 1] >= 0.0, 0, 1)
    env._kick_preferred_foot[env_ids] = torch.where(local[:, 1].abs() <= 0.03, -1, side_preference)

    if not hasattr(env, "_kick_ball_start_xy"):
        env._kick_ball_start_xy = torch.zeros(env.num_envs, 2, device=env.device)
    env._kick_ball_start_xy[env_ids] = pose[:, :2]

    if hasattr(env, "_kick_prev_ball_vel"):
        env._kick_prev_ball_vel[env_ids] = 0.0
    if hasattr(env, "_kick_happened"):
        env._kick_happened[env_ids] = False
        env._kick_recovery_time[env_ids] = 0.0
    if hasattr(env, "_kick_target_achieved"):
        env._kick_target_achieved[env_ids] = False
    if hasattr(env, "_kick_recovery_stable_time"):
        env._kick_recovery_stable_time[env_ids] = 0.0
    if hasattr(env, "_kick_prev_selected_foot_distance"):
        env._kick_prev_selected_foot_distance[env_ids] = torch.nan
    if hasattr(env, "_kick_prev_ball_speed_for_foot"):
        env._kick_prev_ball_speed_for_foot[env_ids] = 0.0
    if hasattr(env, "_kick_prev_ball_speed_inside"):
        env._kick_prev_ball_speed_inside[env_ids] = 0.0
    if hasattr(env, "_kick_valid_foot_kick"):
        env._kick_valid_foot_kick[env_ids] = False
    if hasattr(env, "_kick_directional_happened"):
        env._kick_directional_happened[env_ids] = False
    if hasattr(env, "_kick_camera_last_episode_length"):
        env._kick_camera_last_episode_length[env_ids] = -1


def reset_kick_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float] = (4.0, 4.0),
    angle_range_deg: tuple[float, float] = (-60.0, 60.0),
    visualize_target: bool = False,
    target_radius: float = 0.15,
    origin_at_ball: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Sample a target angle, measuring distance from the robot or ball."""
    robot: Articulation = env.scene[robot_cfg.name]
    count = len(env_ids)
    if not hasattr(env, "_kick_target_w"):
        env._kick_target_w = torch.zeros(env.num_envs, 2, device=env.device)

    distance = torch.empty(count, device=env.device).uniform_(*distance_range)
    relative_angle = torch.empty(count, device=env.device).uniform_(*angle_range_deg) * torch.pi / 180.0
    q = yaw_quat(robot.data.root_quat_w[env_ids])
    robot_yaw = torch.atan2(2.0 * q[:, 0] * q[:, 3], 1.0 - 2.0 * q[:, 3].square())
    world_angle = robot_yaw + relative_angle
    ball: RigidObject = env.scene[ball_cfg.name]
    target_origin = ball.data.root_pos_w[env_ids, :2] if origin_at_ball else robot.data.root_pos_w[env_ids, :2]
    env._kick_target_w[env_ids, 0] = target_origin[:, 0] + distance * torch.cos(world_angle)
    env._kick_target_w[env_ids, 1] = target_origin[:, 1] + distance * torch.sin(world_angle)

    # Choose the inside-foot geometry from ball -> target, not merely ball side.
    # A target left of the ball is struck with the right foot and vice versa.
    direction_w = torch.zeros(count, 3, device=env.device)
    direction_w[:, :2] = env._kick_target_w[env_ids] - ball.data.root_pos_w[env_ids, :2]
    if not hasattr(env, "_kick_direction_w"):
        env._kick_direction_w = torch.zeros(env.num_envs, 2, device=env.device)
    env._kick_direction_w[env_ids] = direction_w[:, :2] / torch.norm(
        direction_w[:, :2], dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    direction_b = quat_apply_inverse(yaw_quat(robot.data.root_quat_w[env_ids]), direction_w)
    lateral = direction_b[:, 1]
    preferred = torch.where(lateral > 0.03, 1, 0)
    preferred = torch.where(lateral < -0.03, 0, preferred)
    env._kick_preferred_foot[env_ids] = torch.where(lateral.abs() <= 0.03, -1, preferred)

    # SceneEntityCfg.body_ids can remain as ``slice(None)`` for this reset
    # callback, even when body_names contains only the two feet.  Resolve the
    # names explicitly so the captured stance always has shape (N, 2, 3).
    if feet_cfg.body_names is not None:
        feet_body_ids, _ = robot.find_bodies(
            feet_cfg.body_names, preserve_order=feet_cfg.preserve_order
        )
    else:
        feet_body_ids = feet_cfg.body_ids
    feet_pos_w = robot.data.body_pos_w[env_ids][:, feet_body_ids]
    feet_delta_w = feet_pos_w - robot.data.root_pos_w[env_ids, None, :]
    root_yaw = yaw_quat(robot.data.root_quat_w[env_ids])
    # Keep the environment/foot dimensions aligned.  Flattening here can
    # mismatch the resolved body-id dimension during reset (e.g. 8192 vs
    # 53248), causing quat_apply_inverse() to fail in TorchScript.
    foot_count = feet_delta_w.shape[1]
    feet_pos_b = quat_apply_inverse(
        root_yaw[:, None, :].expand(-1, foot_count, -1).reshape(-1, 4),
        feet_delta_w.reshape(-1, 3),
    ).reshape(count, foot_count, 3)
    if not hasattr(env, "_kick_start_feet_b"):
        env._kick_start_feet_b = torch.zeros(env.num_envs, 2, 3, device=env.device)
        env._kick_start_feet_w = torch.zeros(env.num_envs, 2, 3, device=env.device)
        env._kick_start_stance_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._kick_start_feet_b[env_ids] = feet_pos_b
    env._kick_start_feet_w[env_ids] = feet_pos_w
    # Body transforms may still contain the previous episode during reset.
    # Capture them again on the first post-reset reward step.
    env._kick_start_stance_valid[env_ids] = False
    nearest_foot = torch.argmin(
        torch.norm(feet_pos_w - ball.data.root_pos_w[env_ids, None, :], dim=2), dim=1
    )
    if not hasattr(env, "_kick_recovery_foot"):
        env._kick_recovery_foot = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._kick_recovery_foot[env_ids] = torch.where(
        env._kick_preferred_foot[env_ids] >= 0, env._kick_preferred_foot[env_ids], nearest_foot
    )

    if visualize_target:
        if not hasattr(env, "_kick_target_visualizer"):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Kick/target",
                markers={
                    "target": sim_utils.CylinderCfg(
                        radius=target_radius,
                        height=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.95, 0.12, 0.05),
                            emissive_color=(0.55, 0.03, 0.01),
                        ),
                    ),
                },
            )
            env._kick_target_visualizer = VisualizationMarkers(marker_cfg)

        # Targets are static during an episode.  Redraw the full batch whenever
        # any environments reset so asynchronously reset markers stay aligned.
        marker_positions = torch.zeros(env.num_envs, 3, device=env.device)
        marker_positions[:, :2] = env._kick_target_w
        marker_positions[:, 2] = env.scene.env_origins[:, 2] + 0.008
        env._kick_target_visualizer.visualize(marker_positions)


def survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def stationary_base_xy(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.sum(robot.data.root_lin_vel_b[:, :2].square(), dim=1) / (std * std))


def stationary_base_yaw(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-robot.data.root_ang_vel_b[:, 2].square() / (std * std))


def base_height_below_minimum_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Allow a controlled crouch and penalize only excessive body lowering."""
    robot: Articulation = env.scene[asset_cfg.name]
    return (minimum_height - robot.data.root_pos_w[:, 2]).clamp_min(0.0).square()


def base_horizontal_sway(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Penalize large horizontal trunk translation without opposing a vertical crouch."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(robot.data.root_lin_vel_b[:, :2].square(), dim=1)


def ball_velocity_to_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    decay_distance: float = 4.0,
    max_reward: float = 10.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    target_w = _kick_target_w(env, target_xy)
    direction = target_w - ball.data.root_pos_w[:, :2]
    direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    forward_speed = torch.sum(ball.data.root_lin_vel_w[:, :2] * direction, dim=1).clamp_min(0.0)
    # Spatial decay preserves the legacy intent without requiring a stateful moving-time buffer.
    distance_from_origin = torch.norm(ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    valid_kick = getattr(env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return (forward_speed * torch.exp(-distance_from_origin / decay_distance)).clamp_max(max_reward) * valid_kick.float()


def ball_target_accuracy(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward the ball for approaching the actual target point."""
    ball: RigidObject = env.scene[ball_cfg.name]
    target_w = _kick_target_w(env, target_xy)
    distance = torch.norm(ball.data.root_pos_w[:, :2] - target_w, dim=1)
    valid_kick = getattr(env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return torch.exp(-distance.square() / (std * std)) * valid_kick.float()


def ball_lateral_velocity(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize velocity perpendicular to the ball-to-target direction."""
    ball: RigidObject = env.scene[ball_cfg.name]
    target_w = _kick_target_w(env, target_xy)
    direction = target_w - ball.data.root_pos_w[:, :2]
    direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    velocity = ball.data.root_lin_vel_w[:, :2]
    forward = torch.sum(velocity * direction, dim=1, keepdim=True) * direction
    return torch.sum((velocity - forward).square(), dim=1)


def ball_overspeed(
    env: ManagerBasedRLEnv,
    max_speed: float = 2.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize only the portion of ball speed above the useful range."""
    ball: RigidObject = env.scene[ball_cfg.name]
    excess = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) - max_speed).clamp_min(0.0)
    return excess.square()


def kicking_foot_approach_ball(
    env: ManagerBasedRLEnv,
    proximity_std: float = 0.1,
    stationary_speed: float = 0.1,
    velocity_weight: float = 0.3,
    center_deadband: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward the foot selected from the ball's lateral position.

    Positive robot-frame y selects the left foot and negative y selects the right
    foot. Inside a small center deadband, the physically nearer foot is selected.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    foot_pos_all = robot.data.body_pos_w[:, asset_cfg.body_ids]
    foot_vel_all = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    distances = torch.norm(foot_pos_all - ball.data.root_pos_w[:, None, :], dim=2)

    ball_local_y = ball_pos_b(env, ball_cfg=ball_cfg)[:, 1]
    preferred_idx = getattr(env, "_kick_preferred_foot", torch.where(ball_local_y >= 0.0, 0, 1))
    nearest_idx = torch.argmin(distances, dim=1)
    selected_idx = torch.where(ball_local_y.abs() <= center_deadband, nearest_idx, preferred_idx)
    env_ids = torch.arange(env.num_envs, device=env.device)
    foot_pos = foot_pos_all[env_ids, selected_idx]
    foot_vel = foot_vel_all[env_ids, selected_idx]
    to_ball = ball.data.root_pos_w - foot_pos
    distance = torch.norm(to_ball, dim=1)
    direction = to_ball / distance.unsqueeze(1).clamp_min(1.0e-6)
    proximity_gate = torch.exp(-distance / proximity_std)
    closing_speed = torch.sum(foot_vel * direction, dim=1).clamp_min(0.0)
    stationary = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < stationary_speed).float()
    # No reward is given for merely remaining near the ball. Proximity only gates
    # positive motion toward it, so a motionless policy receives exactly zero.
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return closing_speed * ((1.0 - velocity_weight) * proximity_gate + velocity_weight) * stationary * (~kick_happened).float()


def kicking_foot_approach_progress(
    env: ManagerBasedRLEnv,
    center_deadband: float = 0.03,
    max_progress_per_step: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Dense reward for reducing the selected foot-to-ball distance each step."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    distances = torch.norm(feet - ball.data.root_pos_w[:, None, :], dim=2)
    ball_y = ball_pos_b(env, ball_cfg=ball_cfg)[:, 1]
    preferred = getattr(env, "_kick_preferred_foot", torch.where(ball_y >= 0.0, 0, 1))
    nearest = torch.argmin(distances, dim=1)
    selected = torch.where(ball_y.abs() <= center_deadband, nearest, preferred)
    current = distances[torch.arange(env.num_envs, device=env.device), selected]

    if not hasattr(env, "_kick_prev_selected_foot_distance"):
        env._kick_prev_selected_foot_distance = current.detach().clone()
        return torch.zeros_like(current)
    previous = env._kick_prev_selected_foot_distance
    initialized = torch.isfinite(previous)
    progress = torch.where(initialized, previous - current, torch.zeros_like(current))
    progress = progress.clamp(min=-max_progress_per_step, max=max_progress_per_step)
    env._kick_prev_selected_foot_distance.copy_(current)
    ball_stationary = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.1).float()
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return progress / max_progress_per_step * ball_stationary * (~kick_happened).float()


def preferred_foot_kick_event(
    env: ManagerBasedRLEnv,
    speed_increase_threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Return +1 for a preferred-foot kick event and -1 for a wrong-foot event."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    if not hasattr(env, "_kick_prev_ball_speed_for_foot"):
        env._kick_prev_ball_speed_for_foot = speed.detach().clone()
        return torch.zeros_like(speed)
    speed_increase = speed - env._kick_prev_ball_speed_for_foot
    env._kick_prev_ball_speed_for_foot.copy_(speed)
    event = speed_increase > speed_increase_threshold

    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    actual_foot = torch.argmin(torch.norm(feet - ball.data.root_pos_w[:, None, :], dim=2), dim=1)
    ball_y = ball_pos_b(env, ball_cfg=ball_cfg)[:, 1]
    preferred = getattr(env, "_kick_preferred_foot", torch.where(ball_y >= 0.0, 0, 1))
    correct = (preferred < 0) | (actual_foot == preferred)
    signed = torch.where(correct, torch.ones_like(speed), -torch.ones_like(speed))
    return signed * event.float()


def wrong_foot_proximity(
    env: ManagerBasedRLEnv,
    proximity_std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize bringing the non-selected foot close to a stationary ball."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet = robot.data.body_pos_w[:, asset_cfg.body_ids]
    ball_y = ball_pos_b(env, ball_cfg=ball_cfg)[:, 1]
    preferred = getattr(env, "_kick_preferred_foot", torch.where(ball_y >= 0.0, 0, 1))
    has_preference = preferred >= 0
    wrong = (1 - preferred).clamp(min=0, max=1)
    distance = torch.norm(feet[torch.arange(env.num_envs, device=env.device), wrong] - ball.data.root_pos_w, dim=1)
    stationary = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.1).float()
    return torch.exp(-distance / proximity_std) * stationary * has_preference.float()


def inside_foot_kick_quality(
    env: ManagerBasedRLEnv,
    speed_increase_threshold: float = 0.08,
    max_contact_distance: float = 0.25,
    ideal_contact_distance: float = 0.15,
    contact_distance_std: float = 0.035,
    full_reward_angle_deg: float = 10.0,
    zero_reward_angle_deg: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Score the inferred contact as an inside-foot kick at the kick event.

    Left-foot medial normal is local -y and right-foot medial normal is local +y.
    A high-quality event has the ball on that medial side, the medial normal aimed
    toward the target, and the foot velocity moving toward the target.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
    feet_quat = robot.data.body_quat_w[:, asset_cfg.body_ids]
    feet_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]

    ball_y = ball_pos_b(env, ball_cfg=ball_cfg)[:, 1]
    preferred = getattr(env, "_kick_preferred_foot", torch.where(ball_y >= 0.0, 0, 1))
    distances = torch.norm(feet_pos - ball.data.root_pos_w[:, None, :], dim=2)
    nearest = torch.argmin(distances, dim=1)
    selected = torch.where(preferred >= 0, preferred, nearest)
    ids = torch.arange(env.num_envs, device=env.device)
    foot_pos = feet_pos[ids, selected]
    foot_quat = feet_quat[ids, selected]
    foot_vel = feet_vel[ids, selected]
    distance = distances[ids, selected]

    medial_local = torch.zeros(env.num_envs, 3, device=env.device)
    medial_local[:, 1] = torch.where(selected == 0, -1.0, 1.0)
    medial_w = quat_apply(foot_quat, medial_local)
    medial_xy = medial_w[:, :2]
    medial_xy = medial_xy / torch.norm(medial_xy, dim=1, keepdim=True).clamp_min(1.0e-6)

    to_ball = ball.data.root_pos_w[:, :2] - foot_pos[:, :2]
    to_ball = to_ball / torch.norm(to_ball, dim=1, keepdim=True).clamp_min(1.0e-6)
    # Use the direction latched at reset instead of recomputing from a moving
    # ball every frame.  This keeps the swing direction stable under ball,
    # target, and base motion.
    if hasattr(env, "_kick_direction_w"):
        target_dir = env._kick_direction_w
    else:
        target_dir = _kick_target_w(env, (4.0, 0.0)) - ball.data.root_pos_w[:, :2]
        target_dir = target_dir / torch.norm(target_dir, dim=1, keepdim=True).clamp_min(1.0e-6)
    foot_speed = torch.norm(foot_vel[:, :2], dim=1)
    foot_dir = foot_vel[:, :2] / foot_speed.unsqueeze(1).clamp_min(1.0e-6)

    ball_on_inside = torch.sum(medial_xy * to_ball, dim=1).clamp(0.0, 1.0)
    medial_alignment = torch.sum(medial_xy * target_dir, dim=1).clamp(-1.0, 1.0)
    cos_full = torch.cos(torch.tensor(full_reward_angle_deg * torch.pi / 180.0, device=env.device))
    cos_zero = torch.cos(torch.tensor(zero_reward_angle_deg * torch.pi / 180.0, device=env.device))
    # Full reward inside 10 degrees, smooth rapid decay to zero at 20 degrees.
    face_to_target = ((medial_alignment - cos_zero) / (cos_full - cos_zero)).clamp(0.0, 1.0)
    face_to_target = face_to_target.square()
    swing_to_target = torch.sum(foot_dir * target_dir, dim=1).clamp(0.0, 1.0)
    # The medial axis should remain close to horizontal.  This discourages a
    # rolled/pitched toe contact that happens to have a similar XY projection.
    medial_level = (1.0 - medial_w[:, 2].abs() / 0.35).clamp(0.0, 1.0)
    contact_distance_quality = torch.exp(
        -(distance - ideal_contact_distance).square() / (contact_distance_std * contact_distance_std)
    )
    quality = ball_on_inside * face_to_target * swing_to_target * medial_level * contact_distance_quality
    quality *= (distance < max_contact_distance).float()

    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1)
    if not hasattr(env, "_kick_prev_ball_speed_inside"):
        env._kick_prev_ball_speed_inside = ball_speed.detach().clone()
        return torch.zeros_like(ball_speed)
    kick_event = (ball_speed - env._kick_prev_ball_speed_inside) > speed_increase_threshold
    env._kick_prev_ball_speed_inside.copy_(ball_speed)
    if not hasattr(env, "_kick_valid_foot_kick"):
        env._kick_valid_foot_kick = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._kick_valid_foot_kick |= kick_event & (distance < max_contact_distance)

    # Range [-1, 1]: a poor/non-inside kick is penalized at the event.
    return (2.0 * quality - 1.0) * kick_event.float()


def inside_foot_preparation(
    env: ManagerBasedRLEnv,
    full_reward_angle_deg: float = 10.0,
    zero_reward_angle_deg: float = 35.0,
    max_distance: float = 0.40,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward presenting the medial foot toward the target before contact."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
    feet_quat = robot.data.body_quat_w[:, asset_cfg.body_ids]
    distances = torch.norm(feet_pos - ball.data.root_pos_w[:, None, :], dim=2)
    preferred = getattr(
        env, "_kick_preferred_foot",
        torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
    )
    nearest = torch.argmin(distances, dim=1)
    selected = torch.where(preferred >= 0, preferred, nearest)
    ids = torch.arange(env.num_envs, device=env.device)
    foot_pos = feet_pos[ids, selected]
    foot_quat = feet_quat[ids, selected]
    distance = distances[ids, selected]

    medial_local = torch.zeros(env.num_envs, 3, device=env.device)
    medial_local[:, 1] = torch.where(selected == 0, -1.0, 1.0)
    medial_w = quat_apply(foot_quat, medial_local)
    medial_xy = medial_w[:, :2]
    medial_xy = medial_xy / torch.norm(medial_xy, dim=1, keepdim=True).clamp_min(1.0e-6)
    target_dir = getattr(env, "_kick_direction_w", torch.zeros_like(medial_xy))
    to_ball = ball.data.root_pos_w[:, :2] - foot_pos[:, :2]
    to_ball = to_ball / torch.norm(to_ball, dim=1, keepdim=True).clamp_min(1.0e-6)

    ball_on_inside = torch.sum(medial_xy * to_ball, dim=1).clamp(0.0, 1.0)
    alignment = torch.sum(medial_xy * target_dir, dim=1).clamp(-1.0, 1.0)
    cos_full = torch.cos(torch.tensor(full_reward_angle_deg * torch.pi / 180.0, device=env.device))
    cos_zero = torch.cos(torch.tensor(zero_reward_angle_deg * torch.pi / 180.0, device=env.device))
    face_to_target = ((alignment - cos_zero) / (cos_full - cos_zero)).clamp(0.0, 1.0).square()
    medial_level = (1.0 - medial_w[:, 2].abs() / 0.35).clamp(0.0, 1.0)
    proximity = (1.0 - distance / max_distance).clamp(0.0, 1.0)
    before_kick = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.15
    return ball_on_inside * face_to_target * medial_level * proximity * before_kick.float()


def pre_kick_crouch(
    env: ManagerBasedRLEnv,
    target_height: float = 0.53,
    height_std: float = 0.025,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward a modest and stable lowering of the base before contact."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    height_quality = torch.exp(-((height - target_height) / height_std).square())
    tilt = torch.sum(robot.data.projected_gravity_b[:, :2].square(), dim=1)
    ang_vel = torch.sum(robot.data.root_ang_vel_b[:, :2].square(), dim=1)
    stable = torch.exp(-8.0 * tilt - 0.3 * ang_vel)
    before_kick = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.15
    return height_quality * stable * before_kick.float()


def controlled_kicking_foot_speed(
    env: ManagerBasedRLEnv,
    far_horizontal_limit: float = 0.8,
    contact_horizontal_limit: float = 1.6,
    vertical_limit: float = 0.45,
    landing_vertical_limit: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize violent lifting/lowering while allowing useful contact speed."""
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    feet_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]
    feet_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    distances = torch.norm(feet_pos - ball.data.root_pos_w[:, None, :], dim=2)
    preferred = getattr(
        env, "_kick_preferred_foot",
        torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
    )
    nearest = torch.argmin(distances, dim=1)
    selected = torch.where(preferred >= 0, preferred, nearest)
    ids = torch.arange(env.num_envs, device=env.device)
    foot_vel = feet_vel[ids, selected]
    distance = distances[ids, selected]

    horizontal_speed = torch.norm(foot_vel[:, :2], dim=1)
    horizontal_limit = torch.where(
        distance < 0.20,
        torch.full_like(distance, contact_horizontal_limit),
        torch.full_like(distance, far_horizontal_limit),
    )
    pre_penalty = (
        (horizontal_speed - horizontal_limit).clamp_min(0.0).square()
        + (foot_vel[:, 2].abs() - vertical_limit).clamp_min(0.0).square()
    )
    before_kick = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.15
    kick_happened = getattr(
        env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    landing_penalty = (
        feet_vel[:, :, 2].abs() - landing_vertical_limit
    ).clamp_min(0.0).square().mean(dim=1)
    return pre_penalty * before_kick.float() + landing_penalty * kick_happened.float()


def ball_acceleration_to_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    scale: float = 80.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    velocity = ball.data.root_lin_vel_w[:, :2]
    if not hasattr(env, "_kick_prev_ball_vel"):
        env._kick_prev_ball_vel = velocity.detach().clone()
    acceleration = (velocity - env._kick_prev_ball_vel) / env.step_dt
    env._kick_prev_ball_vel.copy_(velocity)
    target_w = _kick_target_w(env, target_xy)
    direction = target_w - ball.data.root_pos_w[:, :2]
    direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)
    forward = torch.sum(acceleration * direction, dim=1)
    lateral = torch.norm(acceleration - forward.unsqueeze(1) * direction, dim=1)
    valid_kick = getattr(env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return torch.tanh((forward - lateral).clamp_min(0.0) / scale) * valid_kick.float()


def waiting_penalty(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    progress = env.episode_length_buf.float() / env.max_episode_length
    stationary = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < 0.1).float()
    return progress.square() * stationary


def pre_kick_stability(
    env: ManagerBasedRLEnv,
    ball_speed_threshold: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward a quiet, upright base while the ball has not yet been kicked."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    before_kick = (torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) < ball_speed_threshold).float()
    lin_error = torch.sum(robot.data.root_lin_vel_b.square(), dim=1)
    ang_error = torch.sum(robot.data.root_ang_vel_b.square(), dim=1)
    tilt_error = torch.sum(robot.data.projected_gravity_b[:, :2].square(), dim=1)
    return torch.exp(-2.0 * lin_error - 0.5 * ang_error - 8.0 * tilt_error) * before_kick


def post_kick_recovery(
    env: ManagerBasedRLEnv,
    kick_speed_threshold: float = 0.2,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Track the post-kick phase and reward holding any dynamically stable stance."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    if not hasattr(env, "_kick_happened"):
        env._kick_happened = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._kick_recovery_time = torch.zeros(env.num_envs, device=env.device)

    valid_foot_kick = getattr(
        env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env._kick_happened |= valid_foot_kick & (
        torch.norm(ball.data.root_lin_vel_w[:, :2], dim=1) > kick_speed_threshold
    )
    env._kick_recovery_time += env._kick_happened.float() * env.step_dt

    joint_speed = torch.mean(robot.data.joint_vel.square(), dim=1)
    base_lin = torch.sum(robot.data.root_lin_vel_b.square(), dim=1)
    base_ang = torch.sum(robot.data.root_ang_vel_b.square(), dim=1)
    tilt = torch.sum(robot.data.projected_gravity_b[:, :2].square(), dim=1)
    stability = torch.exp(
        -0.08 * joint_speed - 2.0 * base_lin - 0.5 * base_ang - 10.0 * tilt
    )
    return stability * env._kick_happened.float()


def walk_ready_after_kick(
    env: ManagerBasedRLEnv,
    return_delay: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """After initial bracing, softly return joints to a locomotion-ready state."""
    robot: Articulation = env.scene[robot_cfg.name]
    recovery_time = getattr(env, "_kick_recovery_time", torch.zeros(env.num_envs, device=env.device))
    active = recovery_time >= return_delay
    joint_error = torch.mean((robot.data.joint_pos - robot.data.default_joint_pos).square(), dim=1)
    joint_speed = torch.mean(robot.data.joint_vel.square(), dim=1)
    readiness = torch.exp(-4.0 * joint_error - 0.05 * joint_speed)
    return readiness * active.float()


def post_kick_feet_grounded(
    env: ManagerBasedRLEnv,
    return_delay: float = 0.25,
    height_std: float = 0.035,
    velocity_scale: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
) -> torch.Tensor:
    """After the kick, reward both feet returning to the pre-kick ground stance."""
    robot: Articulation = env.scene[asset_cfg.name]
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    recovery_time = getattr(env, "_kick_recovery_time", torch.zeros(env.num_envs, device=env.device))
    if not hasattr(env, "_kick_start_feet_b"):
        return torch.zeros(env.num_envs, device=env.device)

    if asset_cfg.body_names is not None:
        feet_body_ids, _ = robot.find_bodies(
            asset_cfg.body_names, preserve_order=asset_cfg.preserve_order
        )
    else:
        feet_body_ids = asset_cfg.body_ids
    feet_w = robot.data.body_pos_w[:, feet_body_ids]
    if feet_w.shape[1] != 2:
        raise ValueError(
            f"post_kick_feet_grounded requires exactly two feet, got {feet_w.shape[1]} bodies."
        )
    feet_delta_w = feet_w - robot.data.root_pos_w[:, None, :]
    root_yaw = yaw_quat(robot.data.root_quat_w)
    foot_count = feet_delta_w.shape[1]
    feet_b = quat_apply_inverse(
        root_yaw[:, None, :].expand(-1, foot_count, -1).reshape(-1, 4),
        feet_delta_w.reshape(-1, 3),
    ).reshape(env.num_envs, foot_count, 3)
    height_error = torch.mean((feet_b[:, :, 2] - env._kick_start_feet_b[:, :, 2]).square(), dim=1)
    foot_speed = torch.mean(robot.data.body_lin_vel_w[:, feet_body_ids].square(), dim=(1, 2))
    grounded = torch.exp(-height_error / (height_std * height_std) - velocity_scale * foot_speed)
    active = kick_happened & (recovery_time >= return_delay)
    return grounded * active.float()


def recovery_step_to_start_stance(
    env: ManagerBasedRLEnv,
    return_delay: float = 0.5,
    min_support_advance: float = 0.08,
    stance_std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
) -> torch.Tensor:
    """Step the trailing support foot forward and restore the episode-start stance."""
    robot: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_kick_start_feet_b"):
        return torch.zeros(env.num_envs, device=env.device)
    if asset_cfg.body_names is not None:
        feet_body_ids, _ = robot.find_bodies(
            asset_cfg.body_names, preserve_order=asset_cfg.preserve_order
        )
    else:
        feet_body_ids = asset_cfg.body_ids
    feet_w = robot.data.body_pos_w[:, feet_body_ids]
    if feet_w.shape[1] != 2:
        raise ValueError(
            f"recovery_step_to_start_stance requires exactly two feet, got {feet_w.shape[1]} bodies."
        )
    feet_delta_w = feet_w - robot.data.root_pos_w[:, None, :]
    root_yaw = yaw_quat(robot.data.root_quat_w)
    foot_count = feet_delta_w.shape[1]
    feet_b = quat_apply_inverse(
        root_yaw[:, None, :].expand(-1, foot_count, -1).reshape(-1, 4),
        feet_delta_w.reshape(-1, 3),
    ).reshape(env.num_envs, foot_count, 3)
    invalid = ~env._kick_start_stance_valid
    if torch.any(invalid):
        env._kick_start_feet_b[invalid] = feet_b[invalid].detach()
        env._kick_start_feet_w[invalid] = feet_w[invalid].detach()
        env._kick_start_stance_valid[invalid] = True
    stance_error = torch.mean((feet_b[:, :, :2] - env._kick_start_feet_b[:, :, :2]).square(), dim=(1, 2))
    stance_quality = torch.exp(-stance_error / (stance_std * stance_std))

    support_foot = 1 - env._kick_recovery_foot
    ids = torch.arange(env.num_envs, device=env.device)
    support_delta = feet_w[ids, support_foot, :2] - env._kick_start_feet_w[ids, support_foot, :2]
    support_advance = torch.sum(support_delta * env._kick_direction_w, dim=1)
    step_quality = (support_advance / min_support_advance).clamp(0.0, 1.0)
    recovery_time = getattr(env, "_kick_recovery_time", torch.zeros(env.num_envs, device=env.device))
    return stance_quality * step_quality * (recovery_time >= return_delay).float()


def hip_roll_spread(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_pos[:, asset_cfg.joint_ids].square(), dim=1)


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal foot speed while that foot is in ground contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    robot: Articulation = env.scene[asset_cfg.name]
    foot_velocity_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(foot_velocity_xy.norm(dim=-1) * contacts, dim=1)


def feet_too_close(
    env: ManagerBasedRLEnv,
    min_lateral_separation: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
    ),
) -> torch.Tensor:
    """Penalize a stance narrow enough for the legs or feet to interfere.

    Separation is measured between the foot-link centers along the robot yaw
    frame's lateral axis.  This catches both a narrow stance and crossed legs,
    while leaving wider kick poses unpenalized.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    feet_delta_w = robot.data.body_pos_w[:, asset_cfg.body_ids] - robot.data.root_pos_w[:, None, :]
    root_yaw = yaw_quat(robot.data.root_quat_w)
    feet_b = quat_apply_inverse(
        root_yaw[:, None, :].expand(-1, feet_delta_w.shape[1], -1).reshape(-1, 4),
        feet_delta_w.reshape(-1, 3),
    ).reshape(env.num_envs, feet_delta_w.shape[1], 3)
    lateral_separation = feet_b[:, 0, 1] - feet_b[:, 1, 1]
    shortage = (min_lateral_separation - lateral_separation).clamp_min(0.0)
    return (shortage / min_lateral_separation).square()


def body_twist(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b[:, 2].square()


def ball_success(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (4.0, 0.0),
    target_radius: float = 0.15,
    min_direction_score: float = 0.98,
    max_speed: float = 2.5,
    recovery_time: float = 0.8,
    max_base_speed: float = 0.35,
    max_tilt: float = 0.2,
    max_mean_joint_deviation: float = 0.35,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    if hasattr(env, "_kick_ball_start_xy"):
        relative_pos = ball.data.root_pos_w[:, :2] - env._kick_ball_start_xy
    else:
        relative_pos = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_w = _kick_target_w(env, target_xy)
    start_to_target = target_w - getattr(env, "_kick_ball_start_xy", env.scene.env_origins[:, :2])
    target_dir = start_to_target / torch.norm(start_to_target, dim=1, keepdim=True).clamp_min(1.0e-6)
    target_distance = torch.norm(ball.data.root_pos_w[:, :2] - target_w, dim=1)
    velocity = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(velocity, dim=1)
    travel = torch.norm(relative_pos, dim=1)
    direction_score = torch.sum(relative_pos * target_dir, dim=1) / travel.clamp_min(1.0e-6)
    kick_happened = getattr(env, "_kick_happened", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    if not hasattr(env, "_kick_target_achieved"):
        env._kick_target_achieved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._kick_target_achieved |= (
        (target_distance < target_radius)
        & (speed < max_speed)
        & (direction_score > min_direction_score)
        & kick_happened
    )

    # Match the checkpoint-era recovery criterion: wait after the kick and
    # require general balance, but do not prescribe foot placement, stance
    # width, a support-foot step, or an exact joint pose.
    recovered_long_enough = getattr(
        env, "_kick_recovery_time", torch.zeros(env.num_envs, device=env.device)
    ) >= recovery_time
    base_speed = torch.norm(robot.data.root_lin_vel_b, dim=1)
    tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
    mean_joint_deviation = torch.mean(
        torch.abs(robot.data.joint_pos - robot.data.default_joint_pos), dim=1
    )
    stable = (
        (base_speed < max_base_speed)
        & (tilt < max_tilt)
        & (mean_joint_deviation < max_mean_joint_deviation)
    )
    return env._kick_target_achieved & recovered_long_enough & stable


def ball_too_far(env: ManagerBasedEnv, max_distance: float, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")):
    ball: RigidObject = env.scene[ball_cfg.name]
    return torch.norm(ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1) > max_distance


def ball_not_kicked_in_time(
    env: ManagerBasedRLEnv,
    time_limit: float = 3.0,
    movement_speed: float = 0.12,
    min_direction_cos: float = 0.9396926208,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Fail unless a valid foot kick sends the ball toward the target in time."""
    ball: RigidObject = env.scene[ball_cfg.name]
    elapsed = env.episode_length_buf.float() * env.step_dt
    if not hasattr(env, "_kick_directional_happened"):
        env._kick_directional_happened = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    velocity = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(velocity, dim=1)
    target_dir = getattr(env, "_kick_direction_w", torch.zeros_like(velocity))
    direction_cos = torch.sum(velocity * target_dir, dim=1) / speed.clamp_min(1.0e-6)
    valid_foot_kick = getattr(
        env, "_kick_valid_foot_kick", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env._kick_directional_happened |= (
        valid_foot_kick & (speed > movement_speed) & (direction_cos >= min_direction_cos)
    )
    return (elapsed > time_limit) & ~env._kick_directional_happened


def no_kick_failure_penalty(
    env: ManagerBasedRLEnv,
    time_limit: float = 3.0,
    movement_speed: float = 0.12,
    min_direction_cos: float = 0.9396926208,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """One-step penalty aligned with the no-kick termination condition."""
    return ball_not_kicked_in_time(
        env, time_limit, movement_speed, min_direction_cos, ball_cfg
    ).float()
