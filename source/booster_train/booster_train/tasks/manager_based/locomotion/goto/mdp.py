"""Isaac Lab manager terms for end-to-end goal-conditioned K1 locomotion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi


class UniformSE2GoalCommand(CommandTerm):
    """World-frame pose goal whose policy command is relative to the current body yaw.

    A command resample changes only the target buffers.  It does not reset the
    environment, so RSL-RL recurrent state is intentionally preserved.
    """

    cfg: "UniformSE2GoalCommandCfg"

    def __init__(self, cfg: "UniformSE2GoalCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.goal_pose_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.category = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.just_resampled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        for name in ("position_error", "orientation_error", "target_distance", "target_angle"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_b

    def _relative_pose(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = self.goal_pose_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        yaw = self.robot.data.heading_w
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx = c * delta[:, 0] + s * delta[:, 1]
        dy = -s * delta[:, 0] + c * delta[:, 1]
        dtheta = wrap_to_pi(self.goal_pose_w[:, 2] - yaw)
        return dx, dy, dtheta

    def _distance(self, dx, dy, dtheta):
        return dx.square() + dy.square() + 2.0 * self.cfg.constellation_radius**2 * (1.0 - torch.cos(dtheta))

    def _update_metrics(self):
        dx, dy, dtheta = self._relative_pose()
        self.metrics["position_error"] += torch.sqrt(dx.square() + dy.square())
        self.metrics["orientation_error"] += torch.abs(dtheta)
        self.metrics["target_distance"] += torch.sqrt(dx.square() + dy.square())
        self.metrics["target_angle"] += torch.abs(dtheta)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        probability = torch.tensor(self.cfg.category_probabilities, device=self.device)
        if probability.numel() != 5 or torch.any(probability < 0) or not torch.isclose(probability.sum(), torch.tensor(1.0, device=self.device), atol=1e-6):
            raise ValueError("category_probabilities must be five non-negative values summing to one")
        category = torch.multinomial(probability, n, replacement=True)
        rel = torch.empty(n, 3, device=self.device)
        rel[:, 0].uniform_(*self.cfg.ranges.delta_x)
        rel[:, 1].uniform_(*self.cfg.ranges.delta_y)
        rel[:, 2].uniform_(*self.cfg.ranges.delta_yaw)
        rel[category == 0] = 0.0
        rel[category == 1, 1:] = 0.0
        rel[category == 2, 0] = 0.0
        rel[category == 2, 2] = 0.0
        rel[category == 3, :2] = 0.0
        yaw = self.robot.data.heading_w[ids]
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.goal_pose_w[ids, 0] = self.robot.data.root_pos_w[ids, 0] + c * rel[:, 0] - s * rel[:, 1]
        self.goal_pose_w[ids, 1] = self.robot.data.root_pos_w[ids, 1] + s * rel[:, 0] + c * rel[:, 1]
        self.goal_pose_w[ids, 2] = wrap_to_pi(yaw + rel[:, 2])
        self.category[ids] = category
        self.just_resampled[ids] = True

    def _update_command(self):
        dx, dy, dtheta = self._relative_pose()
        current = self._distance(dx, dy, dtheta)
        self.progress[:] = self.previous_distance - current
        self.progress[self.just_resampled] = 0.0
        self.previous_distance[:] = current
        self.just_resampled[:] = False
        self.goal_b[:, 0] = dx
        self.goal_b[:, 1] = dy
        self.goal_b[:, 2] = torch.sin(dtheta)
        self.goal_b[:, 3] = torch.cos(dtheta)


class MixedDynamicSE2GoalCommand(UniformSE2GoalCommand):
    """Static, moving, and A*-style goals with the original policy interface."""

    STATIC = 0
    MOVING = 1
    WAYPOINT = 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.goal_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_velocity_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self.route_heading_w = self.robot.data.heading_w.clone()
        self.goal_jitter_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_motion_speed = torch.zeros(self.num_envs, device=self.device)
        self.command_settle_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_update_counter = 0
        self.metrics["curriculum_stage"] = torch.zeros(self.num_envs, device=self.device)

    def _curriculum_stage(self) -> int:
        """Return the 30K-run curriculum stage (24 environment steps/iteration)."""
        step = int(self._env.common_step_counter)
        if step < 5_000 * 24:
            return 0
        if step < 12_000 * 24:
            return 1
        if step < 22_000 * 24:
            return 2
        return 3

    @staticmethod
    def _stage_values(stage: int):
        # static/moving/A* probabilities, moving speed, A* hold, jitter sigma,
        # sharp-replan probability, and maximum sharp turn.
        return (
            ((0.70, 0.20, 0.10), (0.10, 0.30), (2.0, 4.0), (0.00, 0.00), 0.00, 0.0),
            ((0.40, 0.35, 0.25), (0.10, 0.60), (1.0, 3.0), (0.01, math.radians(1.0)), 0.02, math.radians(25.0)),
            ((0.20, 0.45, 0.35), (0.15, 1.00), (0.8, 2.5), (0.02, math.radians(2.0)), 0.05, math.radians(40.0)),
            ((0.10, 0.55, 0.35), (0.20, 1.50), (0.6, 2.0), (0.02, math.radians(2.0)), 0.08, math.radians(50.0)),
        )[stage]

    def _sample_time(self, ids, low: float, high: float):
        if ids.numel() > 0:
            self.time_left[ids] = torch.empty(ids.numel(), device=self.device).uniform_(low, high)

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        super()._resample_command(ids)
        n = ids.numel()
        stage = self._curriculum_stage()
        probabilities, speed_range, waypoint_hold, _, sharp_probability, sharp_max = self._stage_values(stage)
        self.metrics["curriculum_stage"].fill_(float(stage))
        mode = torch.multinomial(torch.tensor(probabilities, device=self.device), n, replacement=True)
        self.goal_mode[ids] = mode
        self.goal_velocity_w[ids] = 0.0
        self.goal_yaw_rate[ids] = 0.0
        self.command_settle_steps[ids] = 15

        moving = ids[mode == self.MOVING]
        if moving.numel() > 0:
            k = moving.numel()
            yaw = self.robot.data.heading_w[moving]
            direction = yaw + torch.empty(k, device=self.device).uniform_(-0.7, 0.7)
            distance = torch.empty(k, device=self.device).uniform_(0.5, 1.5)
            speed = torch.empty(k, device=self.device).uniform_(*speed_range)
            self.goal_pose_w[moving, 0] = self.robot.data.root_pos_w[moving, 0] + distance * torch.cos(direction)
            self.goal_pose_w[moving, 1] = self.robot.data.root_pos_w[moving, 1] + distance * torch.sin(direction)
            self.goal_pose_w[moving, 2] = direction
            self.goal_velocity_w[moving, 0] = speed * torch.cos(direction)
            self.goal_velocity_w[moving, 1] = speed * torch.sin(direction)
            self.goal_yaw_rate[moving] = torch.empty(k, device=self.device).uniform_(-0.45, 0.45)

        waypoint = ids[mode == self.WAYPOINT]
        if waypoint.numel() > 0:
            k = waypoint.numel()
            gentle = torch.empty(k, device=self.device).uniform_(-math.radians(18.0), math.radians(18.0))
            sharp = torch.rand(k, device=self.device) < sharp_probability
            replan = torch.empty(k, device=self.device).uniform_(-sharp_max, sharp_max)
            self.route_heading_w[waypoint] = wrap_to_pi(
                self.route_heading_w[waypoint] + torch.where(sharp, replan, gentle)
            )
            direction = self.route_heading_w[waypoint]
            distance = torch.empty(k, device=self.device).uniform_(0.35, 0.90)
            self.goal_pose_w[waypoint, 0] = self.robot.data.root_pos_w[waypoint, 0] + distance * torch.cos(direction)
            self.goal_pose_w[waypoint, 1] = self.robot.data.root_pos_w[waypoint, 1] + distance * torch.sin(direction)
            self.goal_pose_w[waypoint, 2] = direction
            self.category[waypoint] = 4

        self._sample_time(ids[mode == self.STATIC], 2.0, 6.0)
        self._sample_time(moving, 2.0, 5.0)
        self._sample_time(waypoint, *waypoint_hold)

    def _update_command(self):
        moving = self.goal_mode == self.MOVING
        self.goal_pose_w[moving, :2] += self.goal_velocity_w[moving] * self._env.step_dt
        self.goal_pose_w[moving, 2] = wrap_to_pi(
            self.goal_pose_w[moving, 2] + self.goal_yaw_rate[moving] * self._env.step_dt
        )
        delta = self.goal_pose_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        too_far = moving & (distance > 2.0)
        self.goal_pose_w[too_far, :2] = (
            self.robot.data.root_pos_w[too_far, :2]
            + 2.0 * delta[too_far] / distance[too_far, None].clamp_min(1.0e-6)
        )

        super()._update_command()
        waypoint = self.goal_mode == self.WAYPOINT
        reached = waypoint & (torch.linalg.vector_norm(self.goal_b[:, :2], dim=1) < 0.20)
        self.time_left[reached] = 0.0

        self.command_settle_steps.sub_(1).clamp_(min=0)
        moving_speed = torch.linalg.vector_norm(self.goal_velocity_w, dim=1)
        self.target_motion_speed = torch.where(moving, moving_speed, torch.zeros_like(moving_speed))
        self.target_motion_speed = torch.where(
            self.command_settle_steps > 0, torch.ones_like(self.target_motion_speed), self.target_motion_speed
        )

        stage = self._curriculum_stage()
        _, _, _, jitter, _, _ = self._stage_values(stage)
        if self._dynamic_update_counter % 5 == 0:
            self.goal_jitter_b[:, :2] = (
                torch.randn_like(self.goal_jitter_b[:, :2]).clamp_(-2.5, 2.5) * jitter[0]
            )
            self.goal_jitter_b[:, 2] = (
                torch.randn_like(self.goal_jitter_b[:, 2]).clamp_(-3.0, 3.0) * jitter[1]
            )
        self._dynamic_update_counter += 1
        self.goal_b[:, :2] += self.goal_jitter_b[:, :2]
        noisy_heading = torch.atan2(self.goal_b[:, 2], self.goal_b[:, 3]) + self.goal_jitter_b[:, 2]
        flicker = (torch.rand(self.num_envs, device=self.device) < 0.001) if stage >= 3 else torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if flicker.any():
            count = int(flicker.sum().item())
            self.goal_b[flicker, :2] += torch.empty((count, 2), device=self.device).uniform_(-0.30, 0.30)
            noisy_heading[flicker] += torch.empty(count, device=self.device).uniform_(
                -math.radians(15.0), math.radians(15.0)
            )
        self.goal_b[:, 2] = torch.sin(noisy_heading)
        self.goal_b[:, 3] = torch.cos(noisy_heading)


@configclass
class UniformSE2GoalCommandCfg(CommandTermCfg):
    class_type: type = UniformSE2GoalCommand
    asset_name: str = MISSING
    category_probabilities: tuple[float, float, float, float, float] = (0.10, 0.20, 0.20, 0.20, 0.30)
    constellation_radius: float = 1.0

    @configclass
    class Ranges:
        delta_x: tuple[float, float] = (-2.0, 2.0)
        delta_y: tuple[float, float] = (-1.5, 1.5)
        delta_yaw: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = Ranges()


def goal_command(env, command_name: str = "pose_goal"):
    return env.command_manager.get_command(command_name)


def _term(env, command_name: str) -> UniformSE2GoalCommand:
    return env.command_manager.get_term(command_name)


def constellation_reward(env, command_name="pose_goal", radius=1.0):
    term = _term(env, command_name)
    dx, dy, dtheta = term._relative_pose()
    distance = dx.square() + dy.square() + 2.0 * radius**2 * (1.0 - torch.cos(dtheta))
    return torch.exp(-0.2 * distance)


def standing_mask(env, command_name="pose_goal", position_threshold=0.05, orientation_threshold=0.10):
    """Shared pose-only standing gate used by paper reward terms."""
    dx, dy, dtheta = _term(env, command_name)._relative_pose()
    return (torch.sqrt(dx.square() + dy.square()) < position_threshold) & (torch.abs(dtheta) < orientation_threshold)


def base_height_reward(env, nominal_base_height: float, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return torch.exp(-20.0 * torch.abs(robot.data.root_pos_w[:, 2] - nominal_base_height))


def feet_airtime_reward(env, command_name: str, sensor_cfg: SceneEntityCfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    touchdown = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    walking_reward = torch.sum((last_air_time - 0.4) * touchdown.float(), dim=1)
    return torch.where(standing_mask(env, command_name), torch.ones_like(walking_reward), walking_reward)


def _quat_to_rpy(quat):
    """Convert Isaac Lab wxyz quaternions to wrapped XYZ Euler angles."""
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return roll, pitch, yaw


def feet_orientation_reward(env, command_name: str, asset_cfg: SceneEntityCfg):
    robot = env.scene[asset_cfg.name]
    roll, pitch, yaw = _quat_to_rpy(robot.data.body_quat_w[:, asset_cfg.body_ids])
    yaw_error = torch.abs(wrap_to_pi(yaw - robot.data.heading_w[:, None]))
    error = torch.sum(torch.abs(roll) + torch.abs(pitch), dim=1)
    turning = torch.abs(_term(env, command_name)._relative_pose()[2]) > 0.10
    return torch.exp(-error - torch.where(turning, torch.zeros_like(error), torch.sum(yaw_error, dim=1)))


def _feet_in_yaw_frame(robot, body_ids):
    delta = robot.data.body_pos_w[:, body_ids] - robot.data.root_pos_w[:, None, :]
    yaw = robot.data.heading_w
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), dim=-1)


class feet_position_reward(ManagerTermBase):
    """Standing foot-position reward with nominal FK cached before reset randomization."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        robot = env.scene[asset_cfg.name]
        # Cache a body-relative value from the spawned nominal K1 pose. This is
        # morphology-derived and is unaffected by later world/reset poses.
        self.nominal_foot_pos_b = _feet_in_yaw_frame(robot, asset_cfg.body_ids).detach().clone()

    def __call__(self, env, command_name: str, asset_cfg: SceneEntityCfg):
        robot = env.scene[asset_cfg.name]
        current = _feet_in_yaw_frame(robot, asset_cfg.body_ids)
        error = torch.sum(torch.abs(current - self.nominal_foot_pos_b), dim=(1, 2))
        standing_reward = torch.exp(-3.0 * error)
        return torch.where(standing_mask(env, command_name), standing_reward, torch.ones_like(error))


def base_acceleration_reward(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    velocity = robot.data.root_lin_vel_w
    if not hasattr(env, "_goto_previous_root_lin_vel_w"):
        env._goto_previous_root_lin_vel_w = velocity.detach().clone()
    reset = env.episode_length_buf <= 1
    acceleration = (velocity - env._goto_previous_root_lin_vel_w) / env.step_dt
    acceleration[reset] = 0.0
    env._goto_previous_root_lin_vel_w.copy_(velocity)
    return torch.exp(-0.01 * torch.sum(torch.abs(acceleration), dim=1))


def action_difference_reward(env):
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    return torch.exp(-0.02 * torch.sum(torch.abs(action - previous), dim=1))


def normalized_torque_reward(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    torque = torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids])
    if hasattr(robot.data, "soft_joint_effort_limits"):
        limits = robot.data.soft_joint_effort_limits[:, asset_cfg.joint_ids]
    elif hasattr(robot.data, "joint_effort_limits"):
        limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    else:
        raise AttributeError("K1 articulation data exposes no per-joint effort limits")
    return torch.exp(-0.02 * torch.mean(torque / torch.clamp(limits, min=1.0e-6), dim=1))


def tilt_safety(env, safe_tilt=math.radians(5.0), termination_tilt=math.radians(60.0), asset_cfg=SceneEntityCfg("robot")):
    quat = env.scene[asset_cfg.name].data.root_quat_w
    body_up_dot = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    tilt = torch.acos(torch.clamp(body_up_dot, -1.0, 1.0))
    violation = torch.clamp((tilt - safe_tilt) / (termination_tilt - safe_tilt), 0.0, 1.0)
    return violation.square()


def excessive_tilt(env, termination_tilt=math.radians(60.0), asset_cfg=SceneEntityCfg("robot")):
    quat = env.scene[asset_cfg.name].data.root_quat_w
    body_up_dot = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    return torch.acos(torch.clamp(body_up_dot, -1.0, 1.0)) >= termination_tilt


def goal_success(env, command_name="pose_goal", position_threshold=0.05, orientation_threshold=0.1,
                 linear_speed_threshold=0.1, angular_speed_threshold=0.1,
                 sensor_cfg=SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
                 contact_threshold=1.0):
    term = _term(env, command_name)
    dx, dy, dtheta = term._relative_pose()
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    grounded = torch.all(torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > contact_threshold, dim=1)
    return ((dx.square() + dy.square() < position_threshold**2) & (torch.abs(dtheta) < orientation_threshold)
            & (torch.linalg.norm(term.robot.data.root_lin_vel_b[:, :2], dim=1) < linear_speed_threshold)
            & (torch.abs(term.robot.data.root_ang_vel_b[:, 2]) < angular_speed_threshold) & grounded).float()


def stable_at_goal(env, command_name="pose_goal", **kwargs):
    return goal_success(env, command_name=command_name, **kwargs)


def base_height_below_ratio(env, nominal_height: float, fall_height_ratio: float,
                            asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < nominal_height * fall_height_ratio


def mechanical_power_l1(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids] * robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def feet_grounded(env, sensor_cfg=SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]), threshold=1.0):
    """Return a column observation indicating whether both configured feet contact the ground."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # Observation terms must retain an explicit feature dimension. Returning
    # ``(num_envs,)`` makes Isaac Lab infer an empty per-environment shape ``()``
    # and prevents concatenation with vector-valued critic terms.
    grounded = torch.all(torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold, dim=1)
    return grounded.float().unsqueeze(-1)


def feet_lateral_spacing_l2(
    env,
    target_spacing: float,
    asset_cfg=SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
):
    """Penalize deviation from a nominal left/right foot spacing in the body frame."""
    robot = env.scene[asset_cfg.name]
    feet_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta_w = feet_w[:, 0] - feet_w[:, 1]
    yaw = robot.data.heading_w
    lateral_delta = -torch.sin(yaw) * delta_w[:, 0] + torch.cos(yaw) * delta_w[:, 1]
    return torch.square(lateral_delta - target_spacing)


def feet_crossing_penalty(
    env,
    minimum_spacing: float,
    asset_cfg=SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
):
    """Penalize feet that cross or approach closer than the safe lateral gap."""
    robot = env.scene[asset_cfg.name]
    feet_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta_w = feet_w[:, 0] - feet_w[:, 1]
    yaw = robot.data.heading_w
    lateral_delta = -torch.sin(yaw) * delta_w[:, 0] + torch.cos(yaw) * delta_w[:, 1]
    return torch.square(torch.clamp(minimum_spacing - lateral_delta, min=0.0))


def sustained_random_push(
    env,
    env_ids,
    push_interval_s: float,
    push_duration_s: float,
    force_magnitude_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg=SceneEntityCfg("robot", body_names="Trunk"),
):
    """Apply a random horizontal trunk force continuously for part of each interval."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=robot.device)
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=robot.device)

    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else robot.num_bodies
    buffer_shape = (env.num_envs, num_bodies, 3)
    if not hasattr(env, "goto_push_forces"):
        env.goto_push_forces = torch.zeros(buffer_shape, device=robot.device)
        env.goto_push_torques = torch.zeros_like(env.goto_push_forces)

    interval_steps = max(2, int(round(push_interval_s / env.step_dt)))
    duration_steps = max(1, min(interval_steps - 1, int(round(push_duration_s / env.step_dt))))
    start_step = interval_steps - duration_steps
    phase = int(env.common_step_counter) % interval_steps

    if phase == 0:
        env.goto_push_forces[env_ids].zero_()
        env.goto_push_torques[env_ids].zero_()
    elif phase == start_step:
        angle = 2.0 * math.pi * torch.rand(len(env_ids), num_bodies, device=robot.device)
        magnitude = torch.empty(len(env_ids), num_bodies, device=robot.device).uniform_(
            *force_magnitude_range
        )
        env.goto_push_forces[env_ids, :, 0] = magnitude * torch.cos(angle)
        env.goto_push_forces[env_ids, :, 1] = magnitude * torch.sin(angle)
        env.goto_push_forces[env_ids, :, 2] = 0.0
        env.goto_push_torques[env_ids].uniform_(*torque_range)

    robot.set_external_force_and_torque(
        env.goto_push_forces[env_ids],
        env.goto_push_torques[env_ids],
        env_ids=env_ids,
        body_ids=body_ids,
    )


def body_linear_velocity_w(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return robot.data.body_lin_vel_w[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
