"""Commands, observations and teacher rewards for conditional walk-kick training."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.kick.mdp import kick_mdp as kick_mdp


class WalkKickCommand(CommandTerm):
    """High-level walk/kick/recovery state and velocity command.

    Command layout is [vx, vy, wz, cos_phase, sin_phase, walk, kick, recovery].
    """

    cfg: "WalkKickCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 8, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, device=self.device)
        self.walk_speed = torch.zeros(self.num_envs, device=self.device)
        self.kick_distance = torch.zeros(self.num_envs, device=self.device)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mode_time = torch.zeros(self.num_envs, device=self.device)
        self.robot = env.scene[cfg.asset_name]
        self.ball = env.scene["ball"]

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self.walk_speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.walk_speed_range
        )
        self.kick_distance[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.kick_distance_range
        )
        self.mode[env_ids] = 0
        self.mode_time[env_ids] = 0.0
        self.gait_process[env_ids] = 0.0

    def _update_command(self):
        self.mode_time += self._env.step_dt
        ball_b = kick_mdp.ball_pos_b(self._env)[:, :2]
        enter_kick = (self.mode == 0) & (
            ((ball_b[:, 0] < self.kick_distance) & (ball_b[:, 0] > 0.12))
            | (self.mode_time > self.cfg.max_walk_time)
        )
        self.mode[enter_kick] = 1
        self.mode_time[enter_kick] = 0.0
        kick_happened = getattr(
            self._env, "_kick_happened", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        valid_foot_kick = getattr(
            self._env, "_kick_valid_foot_kick", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        ball_speed = torch.norm(self.ball.data.root_lin_vel_w[:, :2], dim=1)
        enter_recovery = (self.mode == 1) & (
            kick_happened | (valid_foot_kick & (ball_speed > 0.2)) | (self.mode_time > self.cfg.max_kick_time)
        )
        self.mode[enter_recovery] = 2
        self.mode_time[enter_recovery] = 0.0
        return_walk = (self.mode == 2) & (self.mode_time > self.cfg.recovery_time)
        self.mode[return_walk] = 0
        self.mode_time[return_walk] = 0.0
        self.walk_speed[return_walk] = 0.0

        walking = self.mode == 0
        self.gait_process = torch.fmod(
            self.gait_process + walking.float() * self._env.step_dt * self.cfg.gait_frequency, 1.0
        )
        self._command.zero_()
        self._command[:, 0] = self.walk_speed * walking.float()
        self._command[:, 3] = torch.cos(2.0 * math.pi * self.gait_process)
        self._command[:, 4] = torch.sin(2.0 * math.pi * self.gait_process)
        self._command[:, 5] = (self.mode == 0).float()
        self._command[:, 6] = (self.mode == 1).float()
        self._command[:, 7] = (self.mode == 2).float()

    def _update_metrics(self):
        pass


@configclass
class WalkKickCommandCfg(CommandTermCfg):
    class_type: type = WalkKickCommand
    asset_name: str = "robot"
    walk_speed_range: tuple[float, float] = (0.0, 1.0)
    gait_frequency: float = 2.0
    kick_distance_range: tuple[float, float] = (0.35, 0.50)
    max_walk_time: float = 3.0
    max_kick_time: float = 3.0
    recovery_time: float = 1.0


def locomotion_command(env, command_name="walk_kick"):
    return env.command_manager.get_command(command_name)[:, :3]


def gait_phase(env, command_name="walk_kick"):
    return env.command_manager.get_command(command_name)[:, 3:5]


def skill_mode(env, command_name="walk_kick"):
    return env.command_manager.get_command(command_name)[:, 5:8]


def _teacher_actions(env, walk_model_path: str, kick_model_path: str):
    step = env.episode_length_buf.clone()
    if not hasattr(env, "_walk_kick_teachers"):
        walk_teacher = torch.jit.load(walk_model_path, map_location=env.device).eval()
        kick_teacher = torch.jit.load(kick_model_path, map_location=env.device).eval()
        with torch.inference_mode():
            walk_probe = walk_teacher(torch.zeros(1, 54, device=env.device))
            kick_probe = kick_teacher(torch.zeros(1, 49, device=env.device))
        if tuple(walk_probe.shape) != (1, 12):
            raise ValueError(
                f"VELOCITY_TEACHER_JIT must accept 54 observations and return 12 actions; "
                f"got output {tuple(walk_probe.shape)}."
            )
        if tuple(kick_probe.shape) != (1, 12):
            raise ValueError(
                f"KICK_TEACHER_JIT must accept 49 observations and return 12 actions; "
                f"got output {tuple(kick_probe.shape)}."
            )
        env._walk_kick_teachers = (walk_teacher, kick_teacher)
        env._walk_teacher_last_action = torch.zeros(env.num_envs, 12, device=env.device)
        env._kick_teacher_last_action = torch.zeros(env.num_envs, 12, device=env.device)
        env._walk_kick_teacher_step = torch.full_like(step, -1)
    if not torch.equal(step, env._walk_kick_teacher_step):
        robot: Articulation = env.scene["robot"]
        cmd = env.command_manager.get_command("walk_kick")
        internal = torch.zeros(env.num_envs, 7, device=env.device)
        internal[:, 0] = 2.0
        walk_q_rel = robot.data.joint_pos - robot.data.default_joint_pos
        # The locomotion teacher used -0.2 ankle pitch versus kick's -0.25.
        for i, name in enumerate(robot.joint_names):
            if "Ankle_Pitch" in name:
                walk_q_rel[:, i] -= 0.05
        walk_obs = torch.cat((cmd[:, :3], internal, cmd[:, 3:5], robot.data.projected_gravity_b,
                              robot.data.root_ang_vel_b, walk_q_rel, robot.data.joint_vel,
                              env._walk_teacher_last_action), dim=1)
        kick_obs = torch.cat((robot.data.projected_gravity_b, robot.data.root_ang_vel_b,
                              kick_mdp.ball_pos_b(env)[:, :2], kick_mdp.kick_target_pos_b(env),
                              kick_mdp.ball_visible(env), kick_mdp.ball_time_since_seen(env),
                              kick_mdp.ball_confidence(env),
                              robot.data.joint_pos - robot.data.default_joint_pos,
                              0.1 * robot.data.joint_vel, env._kick_teacher_last_action), dim=1)
        with torch.inference_mode():
            walk_action = env._walk_kick_teachers[0](walk_obs)
            kick_action = env._walk_kick_teachers[1](kick_obs)
        env._walk_teacher_last_action.copy_(walk_action)
        env._kick_teacher_last_action.copy_(kick_action)
        env._walk_kick_teacher_step.copy_(step)
    return env._walk_teacher_last_action, env._kick_teacher_last_action


def teacher_joint_target_tracking(env, walk_model_path: str, kick_model_path: str, std: float = 0.20):
    walk_action, kick_action = _teacher_actions(env, walk_model_path, kick_model_path)
    robot: Articulation = env.scene["robot"]
    mode = env.command_manager.get_command("walk_kick")[:, 5:8]
    walk_default = robot.data.default_joint_pos.clone()
    for i, name in enumerate(robot.joint_names):
        if "Ankle_Pitch" in name:
            walk_default[:, i] += 0.05
    walk_target = walk_default + walk_action
    scale = torch.ones_like(kick_action)
    from booster_train.assets.robots.booster import K1_ACTION_SCALE
    for i, name in enumerate(robot.joint_names):
        for pattern, value in K1_ACTION_SCALE.items():
            import re
            if re.fullmatch(pattern, name):
                scale[:, i] = value
                break
    kick_target = robot.data.default_joint_pos + kick_action * scale
    teacher_target = torch.where(mode[:, 0:1] > 0.5, walk_target, kick_target)
    action_term = env.action_manager.get_term("joint_pos")
    student_target = action_term.processed_actions
    error = torch.mean((student_target - teacher_target).square(), dim=1)
    return torch.exp(-error / (std * std))


def walk_velocity_tracking(env, command_name="walk_kick", std=0.25):
    robot: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    error = torch.sum((robot.data.root_lin_vel_b[:, :2] - cmd[:, :2]).square(), dim=1)
    return torch.exp(-error / (std * std)) * cmd[:, 5]


def kick_ball_velocity(env):
    mode = env.command_manager.get_command("walk_kick")[:, 6]
    return kick_mdp.ball_velocity_to_target(env, target_xy=(4.0, 0.0), decay_distance=4.0, max_reward=10.0) * mode


def kick_ball_accuracy(env):
    active = env.command_manager.get_command("walk_kick")[:, 6:].sum(dim=1)
    return kick_mdp.ball_target_accuracy(env, target_xy=(4.0, 0.0), std=0.4) * active


def kick_inside_quality(env):
    mode = env.command_manager.get_command("walk_kick")[:, 6]
    return kick_mdp.inside_foot_kick_quality(env, speed_increase_threshold=0.08, max_contact_distance=0.25) * mode


def recovery_stability(env):
    recovery = env.command_manager.get_command("walk_kick")[:, 7]
    return kick_mdp.post_kick_recovery(env, kick_speed_threshold=0.2) * recovery
