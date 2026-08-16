"""Frozen-walk/learned-adjust action composition for the standalone task."""

from __future__ import annotations

import math
import os
import re
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from booster_train.assets.robots.booster import K1_ACTION_SCALE

from .mdp import (
    _adjust_geometry,
    _ready_latch,
    approach_heading_error,
    facing_translation_scale,
)
from .standalone_mdp import ball_pos_b, kick_target_pos_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FrozenWalkAdjustAction(JointPositionAction):
    """Execute the frozen walk teacher until the close-adjust boundary is reached."""

    cfg: "FrozenWalkAdjustActionCfg"

    def __init__(self, cfg: "FrozenWalkAdjustActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        teacher_path = os.path.expanduser(os.environ.get(cfg.teacher_env_var, cfg.teacher_path))
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError(
                f"Frozen walk teacher not found at {teacher_path!r}. "
                f"Set {cfg.teacher_env_var} to a compatible 54-to-12 TorchScript policy."
            )
        self._teacher = torch.jit.load(teacher_path, map_location=self.device).eval()
        with torch.inference_mode():
            probe = self._teacher(torch.zeros(1, 54, device=self.device))
        if tuple(probe.shape) != (1, self.action_dim):
            raise ValueError(
                f"Frozen walk teacher must map (N, 54) to (N, {self.action_dim}); "
                f"got {tuple(probe.shape)}."
            )

        kick_teacher_path = os.path.expanduser(
            os.environ.get(cfg.kick_teacher_env_var, cfg.kick_teacher_path)
        )
        if not os.path.isfile(kick_teacher_path):
            raise FileNotFoundError(
                f"Frozen kick teacher not found at {kick_teacher_path!r}. "
                f"Set {cfg.kick_teacher_env_var} to the deploy-compatible 49-to-12 TorchScript policy."
            )
        self._kick_teacher = torch.jit.load(kick_teacher_path, map_location=self.device).eval()
        with torch.inference_mode():
            kick_probe = self._kick_teacher(torch.zeros(1, 49, device=self.device))
        if tuple(kick_probe.shape) != (1, self.action_dim):
            raise ValueError(
                f"Frozen kick teacher must map (N, 49) to (N, {self.action_dim}); "
                f"got {tuple(kick_probe.shape)}."
            )

        self._teacher_last_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._teacher_target = torch.zeros_like(self._teacher_last_action)
        self._kick_teacher_last_action = torch.zeros_like(self._teacher_last_action)
        self._kick_teacher_target = torch.zeros_like(self._teacher_last_action)
        self._kick_teacher_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._student_processed_actions = torch.zeros_like(self._teacher_last_action)
        self._walk_teacher_reference_active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._adjust_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not cfg.execute_walk_teacher:
            self._adjust_latched.fill_(True)
        self._transition_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._transition_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Reward/termination terms use the same transition mask so kick-ready
        # cannot start while control is still being blended.
        self._env._adjust_transition_active = self._transition_active
        self._handoff_distance = torch.zeros(self.num_envs, device=self.device)
        self._sample_handoff_distance(slice(None))

        # Walk, adjust, kick, and recovery all share the task's fixed
        # -0.20 rad ankle-pitch nominal pose. There is no kick-only offset.
        self._kick_default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._kick_action_scale = torch.ones(self.action_dim, device=self.device)
        if isinstance(self._joint_ids, slice):
            policy_joint_names = self._asset.joint_names[self._joint_ids]
        else:
            policy_joint_names = [self._asset.joint_names[int(index)] for index in self._joint_ids]
        for joint_index, joint_name in enumerate(policy_joint_names):
            for pattern, value in K1_ACTION_SCALE.items():
                if re.fullmatch(pattern, joint_name):
                    self._kick_action_scale[joint_index] = value
                    break

    @property
    def student_processed_actions(self) -> torch.Tensor:
        """Student target before the frozen walk target replaces it."""
        return self._student_processed_actions

    @property
    def teacher_target(self) -> torch.Tensor:
        return self._teacher_target

    @property
    def kick_teacher_target(self) -> torch.Tensor:
        return self._kick_teacher_target

    @property
    def kick_teacher_active(self) -> torch.Tensor:
        return self._kick_teacher_active

    @property
    def frozen_walk_active(self) -> torch.Tensor:
        return (~self._adjust_latched) | self._transition_active

    @property
    def walk_teacher_reference_active(self) -> torch.Tensor:
        return self._walk_teacher_reference_active

    @property
    def handoff_distance(self) -> torch.Tensor:
        return self._handoff_distance

    def _sample_handoff_distance(self, env_ids) -> None:
        step = int(getattr(self._env, "common_step_counter", 0))
        stage = 0
        for threshold in self.cfg.handoff_stage_steps:
            stage += int(step >= threshold)
        low, high = self.cfg.handoff_distance_ranges[min(stage, 3)]
        selected = self._handoff_distance[env_ids]
        samples = torch.empty_like(selected).uniform_(low, high)
        self._handoff_distance[env_ids] = samples

    def _kick_teacher_blend(self) -> float:
        step = int(getattr(self._env, "common_step_counter", 0))
        stage = sum(step >= threshold for threshold in self.cfg.kick_teacher_blend_steps)
        return self.cfg.kick_teacher_blend[min(stage, 3)]

    def _process_kick_teacher(self) -> None:
        kick_happened = getattr(
            self._env,
            "_kick_happened",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        self._kick_teacher_active.copy_(
            self._adjust_latched
            & ~self._transition_active
            & _ready_latch(self._env)
            & ~kick_happened
        )
        if not self._kick_teacher_active.any():
            return

        perception = torch.zeros(self.num_envs, 3, device=self.device)
        # The validated deploy checkpoint was trained with fixed valid vision
        # metadata. Keep this exact actor-side contract for teacher roll-in.
        perception[:, 0] = 1.0
        perception[:, 2] = 1.0
        kick_obs = torch.cat(
            (
                self._asset.data.projected_gravity_b,
                self._asset.data.root_ang_vel_b,
                ball_pos_b(self._env).clamp(-3.0, 3.0),
                kick_target_pos_b(self._env).clamp(-2.0, 2.0) * 0.25,
                perception,
                self._asset.data.joint_pos[:, self._joint_ids] - self._kick_default,
                0.1 * self._asset.data.joint_vel[:, self._joint_ids],
                self._kick_teacher_last_action,
            ),
            dim=1,
        )
        with torch.inference_mode():
            kick_action = self._kick_teacher(kick_obs)
        self._kick_teacher_last_action.copy_(kick_action)
        self._kick_teacher_target.copy_(
            self._kick_default + kick_action * self._kick_action_scale
        )

        blend = self._kick_teacher_blend()
        if blend > 0.0:
            active = self._kick_teacher_active
            self._processed_actions[active] = torch.lerp(
                self._student_processed_actions[active],
                self._kick_teacher_target[active],
                blend,
            )

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._student_processed_actions.copy_(self._processed_actions)

        error_b, pre_kick_pose_distance, heading_error = _adjust_geometry(self._env)
        robot_ball_distance = torch.norm(ball_pos_b(self._env), dim=1)
        face_ball_error = approach_heading_error(self._env).abs()
        # Keep the validated walk teacher in control until the robot is both
        # close enough and actually looking at the ball. This still supports
        # 360-degree starts: the walk teacher rotates first, then hands off.
        if self.cfg.execute_walk_teacher:
            enter_adjust = (
                (robot_ball_distance <= self._handoff_distance)
                & (face_ball_error <= math.radians(self.cfg.handoff_heading_tolerance_deg))
            )
        else:
            # Adjust-only task: the student owns control from the first step.
            enter_adjust = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        newly_handed_off = enter_adjust & ~self._adjust_latched
        self._adjust_latched |= enter_adjust
        self._transition_elapsed[newly_handed_off] = 0.0
        if self.cfg.execute_walk_teacher and self.cfg.transition_duration_s > 0.0:
            self._transition_active.copy_(
                self._adjust_latched
                & (self._transition_elapsed < self.cfg.transition_duration_s)
            )
        else:
            self._transition_active.zero_()
        walk_active = ~self._adjust_latched
        kick_happened = getattr(
            self._env,
            "_kick_happened",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        self._walk_teacher_reference_active.copy_(
            (~_ready_latch(self._env)) & (~kick_happened)
        )
        walk_needed = (
            walk_active
            | self._transition_active
            | self._walk_teacher_reference_active
        )
        if walk_needed.any():
            command = torch.zeros(self.num_envs, 3, device=self.device)
            command[:, 0] = (self.cfg.command_gain_x * error_b[:, 0]).clamp(-1.5, 1.5)
            command[:, 1] = (self.cfg.command_gain_y * error_b[:, 1]).clamp(-1.2, 1.2)
            command_heading = approach_heading_error(self._env)
            command[:, 2] = (self.cfg.command_gain_yaw * command_heading).clamp(-1.6, 1.6)
            command[:, :2] *= facing_translation_scale(
                command_heading,
                full_speed_deg=self.cfg.walk_full_speed_heading_deg,
                stop_deg=self.cfg.walk_stop_translation_heading_deg,
            ).unsqueeze(1)
            close = pre_kick_pose_distance < self.cfg.slowdown_distance
            command[close, :2] *= (
                pre_kick_pose_distance[close] / self.cfg.slowdown_distance
            ).unsqueeze(1)

            self._gait_phase = torch.fmod(
                self._gait_phase + walk_needed.float() * self._env.step_dt * self.cfg.gait_frequency,
                1.0,
            )
            phase = torch.stack(
                (
                    torch.cos(2.0 * math.pi * self._gait_phase),
                    torch.sin(2.0 * math.pi * self._gait_phase),
                ),
                dim=1,
            )
            internal = torch.zeros(self.num_envs, 7, device=self.device)
            internal[:, 0] = self.cfg.gait_frequency
            q_rel = (
                self._asset.data.joint_pos[:, self._joint_ids]
                - self._asset.data.default_joint_pos[:, self._joint_ids]
            )
            obs = torch.cat(
                (
                    command,
                    internal,
                    phase,
                    self._asset.data.projected_gravity_b,
                    self._asset.data.root_ang_vel_b,
                    q_rel,
                    self._asset.data.joint_vel[:, self._joint_ids],
                    self._teacher_last_action,
                ),
                dim=1,
            )
            with torch.inference_mode():
                teacher_action = self._teacher(obs)
            self._teacher_last_action.copy_(teacher_action)

            walk_default = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
            self._teacher_target.copy_(walk_default + teacher_action)
            self._processed_actions[walk_active] = self._teacher_target[walk_active]

            if self._transition_active.any():
                active = self._transition_active
                phase = (
                    self._transition_elapsed[active] / self.cfg.transition_duration_s
                ).clamp(0.0, 1.0)
                alpha = phase.square() * (3.0 - 2.0 * phase)
                self._processed_actions[active] = torch.lerp(
                    self._teacher_target[active],
                    self._student_processed_actions[active],
                    alpha.unsqueeze(1),
                )
                self._transition_elapsed[active] += self._env.step_dt

        self._process_kick_teacher()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._teacher_last_action[env_ids] = 0.0
        self._teacher_target[env_ids] = 0.0
        self._kick_teacher_last_action[env_ids] = 0.0
        self._kick_teacher_target[env_ids] = 0.0
        self._kick_teacher_active[env_ids] = False
        self._student_processed_actions[env_ids] = 0.0
        self._walk_teacher_reference_active[env_ids] = True
        self._gait_phase[env_ids] = 0.0
        self._adjust_latched[env_ids] = not self.cfg.execute_walk_teacher
        self._transition_elapsed[env_ids] = 0.0
        self._transition_active[env_ids] = False
        self._sample_handoff_distance(env_ids)


@configclass
class FrozenWalkAdjustActionCfg(JointPositionActionCfg):
    class_type: type = FrozenWalkAdjustAction
    teacher_path: str = MISSING
    teacher_env_var: str = "ADJUST_KICK_WALK_TEACHER_JIT"
    kick_teacher_path: str = MISSING
    kick_teacher_env_var: str = "ADJUST_KICK_KICK_TEACHER_JIT"
    kick_teacher_blend_steps: tuple[int, int, int] = (40_000, 80_000, 140_000)
    kick_teacher_blend: tuple[float, float, float, float] = (1.0, 0.67, 0.33, 0.0)
    handoff_stage_steps: tuple[int, int, int] = (100_000, 250_000, 500_000)
    handoff_distance_ranges: tuple[tuple[float, float], ...] = (
        (0.55, 0.65),
        (0.50, 0.70),
        (0.45, 0.75),
        (0.45, 0.80),
    )
    slowdown_distance: float = 0.90
    handoff_heading_tolerance_deg: float = 20.0
    transition_duration_s: float = 0.50
    execute_walk_teacher: bool = True
    walk_full_speed_heading_deg: float = 15.0
    walk_stop_translation_heading_deg: float = 45.0
    command_gain_x: float = 1.4
    command_gain_y: float = 1.8
    command_gain_yaw: float = 2.5
    gait_frequency: float = 2.0
