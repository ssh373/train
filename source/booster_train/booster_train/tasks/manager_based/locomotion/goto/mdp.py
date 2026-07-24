"""Isaac Lab manager terms for end-to-end goal-conditioned K1 locomotion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
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
        return dx.square() + dy.square() + self.cfg.constellation_inertia * 2.0 * (1.0 - torch.cos(dtheta))

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


@configclass
class UniformSE2GoalCommandCfg(CommandTermCfg):
    class_type: type = UniformSE2GoalCommand
    asset_name: str = MISSING
    category_probabilities: tuple[float, float, float, float, float] = (0.10, 0.20, 0.20, 0.20, 0.30)
    constellation_inertia: float = 1.0

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


def constellation_reward(env, command_name="pose_goal", weight=0.2, reward_scale=1.0):
    term = _term(env, command_name)
    dx, dy, dtheta = term._relative_pose()
    return reward_scale * torch.exp(-weight * term._distance(dx, dy, dtheta))


def goal_progress(env, command_name="pose_goal"):
    return _term(env, command_name).progress


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
