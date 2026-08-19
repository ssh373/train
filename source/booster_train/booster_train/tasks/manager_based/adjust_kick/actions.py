"""Frozen adjust/kick teacher composition for transition distillation.

Both supplied teachers use the same 49-observation and 12-action contract. The
action term keeps their ``last_action`` slots independent, produces a smooth
teacher target at the adjust-to-kick boundary, and optionally rolls that target
into the simulator while a single student actor learns it.
"""

from __future__ import annotations

import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .mdp import _adjust_geometry, _ready_latch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FrozenAdjustKickTransitionAction(JointPositionAction):
    """Distill two frozen 49-to-12 policies into one feed-forward actor."""

    cfg: "FrozenAdjustKickTransitionActionCfg"

    ADJUST = 0
    TRANSITION = 1
    KICK = 2

    def __init__(self, cfg: "FrozenAdjustKickTransitionActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        if cfg.control_mode not in ("student", "transition"):
            raise ValueError(
                "FrozenAdjustKickTransitionActionCfg.control_mode must be "
                "'student' or 'transition'."
            )
        self._adjust_teacher = self._load_teacher(
            cfg.adjust_teacher_path, cfg.adjust_teacher_env_var, "adjust"
        )
        self._kick_teacher = self._load_teacher(
            cfg.kick_teacher_path, cfg.kick_teacher_env_var, "kick"
        )

        shape = (self.num_envs, self.action_dim)
        self._adjust_last_action = torch.zeros(shape, device=self.device)
        self._kick_last_action = torch.zeros(shape, device=self.device)
        self._teacher_action = torch.zeros(shape, device=self.device)
        self._adjust_teacher_action = torch.zeros(shape, device=self.device)
        self._kick_teacher_action = torch.zeros(shape, device=self.device)
        self._student_action = torch.zeros(shape, device=self.device)
        self._student_processed_actions = torch.zeros(shape, device=self.device)
        self._teacher_processed_actions = torch.zeros(shape, device=self.device)
        self._transition_residual = torch.zeros(shape, device=self.device)
        self._applied_action = torch.zeros(shape, device=self.device)
        self._applied_action_rate = torch.zeros(shape, device=self.device)
        self._previous_applied_action = torch.zeros(shape, device=self.device)
        self._phase = torch.full(
            (self.num_envs,), self.ADJUST, dtype=torch.long, device=self.device
        )
        self._transition_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._transition_alpha = torch.zeros(self.num_envs, device=self.device)
        self._env._adjust_transition_active = self._phase == self.TRANSITION

    def _load_teacher(self, configured_path: str, env_var: str, label: str):
        path = os.path.abspath(os.path.expanduser(os.environ.get(env_var, configured_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Frozen {label} teacher not found at {path!r}. Set {env_var} "
                "to a compatible 49-to-12 TorchScript policy."
            )
        teacher = torch.jit.load(path, map_location=self.device).eval()
        with torch.inference_mode():
            probe = teacher(torch.zeros(1, 49, device=self.device))
        if tuple(probe.shape) != (1, self.action_dim):
            raise ValueError(
                f"Frozen {label} teacher must map (N, 49) to "
                f"(N, {self.action_dim}); got {tuple(probe.shape)}."
            )
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    @property
    def teacher_action(self) -> torch.Tensor:
        """Raw 12-D composite teacher action used by distillation."""
        return self._teacher_action

    @property
    def adjust_teacher_action(self) -> torch.Tensor:
        return self._adjust_teacher_action

    @property
    def kick_teacher_action(self) -> torch.Tensor:
        return self._kick_teacher_action

    @property
    def student_action(self) -> torch.Tensor:
        return self._student_action

    @property
    def student_processed_actions(self) -> torch.Tensor:
        return self._student_processed_actions

    @property
    def teacher_processed_actions(self) -> torch.Tensor:
        return self._teacher_processed_actions

    @property
    def teacher_target(self) -> torch.Tensor:
        """Compatibility alias for older task reward code."""
        return self._teacher_processed_actions

    @property
    def kick_teacher_target(self) -> torch.Tensor:
        return self._offset + self._kick_teacher_action * self._scale

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    @property
    def transition_alpha(self) -> torch.Tensor:
        return self._transition_alpha

    @property
    def transition_progress_for_next_action(self) -> torch.Tensor:
        """Smoothstep progress that will be applied to the next PPO action."""
        if self.cfg.transition_duration_s <= 0.0:
            return torch.ones_like(self._transition_elapsed)
        normalized = (
            self._transition_elapsed / self.cfg.transition_duration_s
        ).clamp(0.0, 1.0)
        return normalized.square() * (3.0 - 2.0 * normalized)

    @property
    def transition_active(self) -> torch.Tensor:
        return self._phase == self.TRANSITION

    @property
    def applied_action(self) -> torch.Tensor:
        """Normalized action actually converted to joint targets."""
        return self._applied_action

    @property
    def applied_action_rate(self) -> torch.Tensor:
        return self._applied_action_rate

    @property
    def transition_residual(self) -> torch.Tensor:
        return self._transition_residual

    @property
    def adjust_teacher_active(self) -> torch.Tensor:
        return self._phase != self.KICK

    @property
    def kick_teacher_active(self) -> torch.Tensor:
        return self._phase != self.ADJUST

    @property
    def frozen_walk_active(self) -> torch.Tensor:
        """Compatibility alias: pre-kick adjust teacher owns these environments."""
        return self._phase != self.KICK

    @property
    def walk_teacher_reference_active(self) -> torch.Tensor:
        return self._phase == self.ADJUST

    def _policy_observation(self) -> torch.Tensor:
        """Read the exact 49-D actor observation contract used by both teachers."""
        observation = self._env.observation_manager.compute_group(
            "policy", update_history=False
        )
        if not isinstance(observation, torch.Tensor) or observation.shape[1] not in (49, 50):
            shape = getattr(observation, "shape", None)
            raise RuntimeError(
                "Adjust-kick teachers require the original 49-D observation, "
                "optionally followed by one transition-progress value; "
                f"got {shape}."
            )
        # The unified PPO actor receives the appended phase scalar, while both
        # frozen teacher references retain their exact original 49-value input.
        return observation[:, :49]

    def _teacher_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        observation = self._policy_observation()
        adjust_observation = observation.clone()
        kick_observation = observation.clone()
        # Each frozen expert sees its own previous action, as in standalone use.
        adjust_observation[:, -self.action_dim :] = self._adjust_last_action
        kick_observation[:, -self.action_dim :] = self._kick_last_action
        with torch.inference_mode():
            adjust_action = self._adjust_teacher(adjust_observation)
            kick_action = self._kick_teacher(kick_observation)
        self._adjust_teacher_action.copy_(adjust_action)
        self._kick_teacher_action.copy_(kick_action)
        # Do not pre-run the kick expert during a potentially long alignment.
        # Its first transition action therefore starts with the same zero
        # previous-action state as the standalone kick episode. Likewise, stop
        # advancing adjust memory after the transition is complete.
        adjust_active = self._phase != self.KICK
        kick_active = self._phase != self.ADJUST
        self._adjust_last_action[adjust_active] = adjust_action[adjust_active]
        self._kick_last_action[kick_active] = kick_action[kick_active]
        return adjust_action, kick_action

    def _advance_phase(self) -> None:
        adjust_mask = self._phase == self.ADJUST
        if adjust_mask.any():
            ready = _ready_latch(
                self._env,
                standoff=self.cfg.target_ball_distance,
                position_tolerance=self.cfg.ready_position_tolerance,
                heading_tolerance_deg=self.cfg.ready_heading_tolerance_deg,
                ball_speed_tolerance=self.cfg.ready_ball_speed_tolerance,
                min_robot_ball_distance=self.cfg.minimum_ball_distance,
                max_robot_ball_distance=self.cfg.maximum_ball_distance,
                max_ball_displacement=self.cfg.maximum_ball_displacement,
                lateral_tolerance=self.cfg.ready_lateral_tolerance,
                minimum_ball_forward_distance=self.cfg.minimum_ball_forward_distance,
            )
            start_transition = adjust_mask & ready
            self._phase[start_transition] = self.TRANSITION
            self._transition_elapsed[start_transition] = 0.0

        transition = self._phase == self.TRANSITION
        if transition.any():
            if self.cfg.transition_duration_s <= 0.0:
                self._phase[transition] = self.KICK
                self._transition_alpha[transition] = 1.0
            else:
                normalized = (
                    self._transition_elapsed[transition] / self.cfg.transition_duration_s
                ).clamp(0.0, 1.0)
                self._transition_alpha[transition] = normalized.square() * (
                    3.0 - 2.0 * normalized
                )
                finished = transition & (
                    self._transition_elapsed >= self.cfg.transition_duration_s
                )
                self._phase[finished] = self.KICK
                self._transition_alpha[finished] = 1.0
                # Keep the final in-window action at smoothstep(0.9) for a
                # 10-step/0.20-s transition, then enter the kick phase on
                # the following control step. This matches the exported
                # stateful composite exactly.
                still_transition = self._phase == self.TRANSITION
                self._transition_elapsed[still_transition] += self._env.step_dt

        self._env._adjust_transition_active = self._phase == self.TRANSITION

    def _teacher_control_blend(self) -> float:
        """Curriculum from teacher roll-in to independently executed student."""
        step = int(getattr(self._env, "common_step_counter", 0))
        stage = sum(step >= threshold for threshold in self.cfg.rollin_stage_steps)
        return self.cfg.teacher_control_blend[
            min(stage, len(self.cfg.teacher_control_blend) - 1)
        ]

    def _print_debug(self) -> None:
        if not self.cfg.debug_transition:
            return
        step = int(getattr(self._env, "common_step_counter", 0))
        if step % self.cfg.debug_interval_steps != 0:
            return
        _, position_error, heading_error = _adjust_geometry(
            self._env, standoff=self.cfg.target_ball_distance
        )
        counts = [int((self._phase == value).sum().item()) for value in range(3)]
        imitation_rmse = torch.mean(
            (self._student_action - self._teacher_action).square()
        ).sqrt()
        print(
            "[ADJUST_KICK_TRANSITION] "
            f"step={step} adjust={counts[0]} blend={counts[1]} kick={counts[2]} "
            f"position_error={position_error.mean().item():.3f}m "
            f"heading_error={torch.rad2deg(heading_error.abs()).mean().item():.1f}deg "
            f"student_teacher_rmse={imitation_rmse.item():.4f}",
            flush=True,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._student_action.copy_(actions)
        self._student_processed_actions.copy_(self._processed_actions)

        self._advance_phase()
        adjust_action, kick_action = self._teacher_actions()
        alpha = self._transition_alpha.unsqueeze(1)
        composite = torch.lerp(adjust_action, kick_action, alpha)
        composite[self._phase == self.ADJUST] = adjust_action[self._phase == self.ADJUST]
        composite[self._phase == self.KICK] = kick_action[self._phase == self.KICK]
        self._teacher_action.copy_(composite)
        self._teacher_processed_actions.copy_(self._offset + composite * self._scale)

        if self.cfg.control_mode == "transition":
            envelope = (
                4.0 * self._transition_alpha * (1.0 - self._transition_alpha)
            ).clamp(0.0, 1.0)
            envelope *= (self._phase == self.TRANSITION).float()
            residual = (
                torch.tanh(self._student_action)
                * self.cfg.transition_residual_scale
                * envelope.unsqueeze(1)
            )
            self._transition_residual.copy_(residual)
            command = composite + residual
        else:
            self._transition_residual.zero_()
            blend = self._teacher_control_blend()
            command = torch.lerp(self._student_action, composite, blend)
        self._processed_actions.copy_(self._offset + command * self._scale)

        self._applied_action.copy_(command)
        self._applied_action_rate.copy_(command - self._previous_applied_action)
        self._previous_applied_action.copy_(command)
        self._print_debug()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._adjust_last_action[env_ids] = 0.0
        self._kick_last_action[env_ids] = 0.0
        self._teacher_action[env_ids] = 0.0
        self._adjust_teacher_action[env_ids] = 0.0
        self._kick_teacher_action[env_ids] = 0.0
        self._student_action[env_ids] = 0.0
        self._student_processed_actions[env_ids] = 0.0
        self._teacher_processed_actions[env_ids] = 0.0
        self._transition_residual[env_ids] = 0.0
        self._applied_action[env_ids] = 0.0
        self._applied_action_rate[env_ids] = 0.0
        self._previous_applied_action[env_ids] = 0.0
        self._phase[env_ids] = self.ADJUST
        self._transition_elapsed[env_ids] = 0.0
        self._transition_alpha[env_ids] = 0.0


@configclass
class FrozenAdjustKickTransitionActionCfg(JointPositionActionCfg):
    class_type: type = FrozenAdjustKickTransitionAction
    adjust_teacher_path: str = MISSING
    adjust_teacher_env_var: str = "ADJUST_KICK_ADJUST_TEACHER_JIT"
    kick_teacher_path: str = MISSING
    kick_teacher_env_var: str = "ADJUST_KICK_KICK_TEACHER_JIT"

    target_ball_distance: float = 0.30
    minimum_ball_distance: float = 0.20
    maximum_ball_distance: float = 0.40
    # Legacy field retained for config compatibility; the gate now uses the
    # kickable sector (distance/lateral/front geometry) instead of this point.
    ready_position_tolerance: float = 0.08
    # Enter kick transition once the robot is inside the safer +/-25 deg
    # portion of the kick teacher's +/-30 deg capability.
    ready_heading_tolerance_deg: float = 25.0
    ready_ball_speed_tolerance: float = 0.08
    maximum_ball_displacement: float = 0.04
    ready_lateral_tolerance: float = 0.18
    minimum_ball_forward_distance: float = 0.10
    # A short transition preserves the validated adjust/kick contact timing
    # while avoiding a discontinuous joint target at handoff.
    transition_duration_s: float = 0.20

    # ``transition`` means PPO controls only a bounded residual in the handoff;
    # adjust and kick remain frozen teacher outputs.
    control_mode: str = "student"
    transition_residual_scale: float = 0.12

    rollin_stage_steps: tuple[int, int, int] = (120_000, 300_000, 600_000)
    teacher_control_blend: tuple[float, float, float, float] = (1.0, 0.99, 0.90, 0.0)

    debug_transition: bool = False
    debug_interval_steps: int = 50


# Backward-compatible names for configs created from the earlier draft.
FrozenWalkAdjustAction = FrozenAdjustKickTransitionAction
FrozenWalkAdjustActionCfg = FrozenAdjustKickTransitionActionCfg
