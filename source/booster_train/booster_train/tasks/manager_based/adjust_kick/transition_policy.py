"""One stateful policy with frozen adjust/kick experts and learned transition."""

from __future__ import annotations

import math

import torch


class FrozenExpertsLearnedTransitionPolicy(torch.nn.Module):
    """Keep both source motions exact and learn only their short handoff."""

    ADJUST = 0
    TRANSITION = 1
    KICK = 2

    def __init__(
        self,
        adjust_teacher,
        kick_teacher,
        transition_actor,
        policy_dt: float = 0.02,
        target_ball_distance: float = 0.30,
        ready_position_tolerance: float = 0.08,
        ready_heading_tolerance_deg: float = 25.0,
        minimum_ball_distance: float = 0.20,
        maximum_ball_distance: float = 0.40,
        transition_duration_s: float = 0.06,
        minimum_ball_confidence: float = 0.05,
        transition_residual_scale: float = 0.03,
        ready_lateral_tolerance: float = 0.18,
        minimum_ball_forward_distance: float = 0.10,
    ) -> None:
        super().__init__()
        self.adjust_teacher = adjust_teacher.eval()
        self.kick_teacher = kick_teacher.eval()
        self.transition_actor = transition_actor.eval()
        for module in (self.adjust_teacher, self.kick_teacher, self.transition_actor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.target_ball_distance = float(target_ball_distance)
        self.ready_position_tolerance = float(ready_position_tolerance)
        self.ready_heading_tolerance = math.radians(ready_heading_tolerance_deg)
        self.minimum_ball_distance = float(minimum_ball_distance)
        self.maximum_ball_distance = float(maximum_ball_distance)
        self.minimum_ball_confidence = float(minimum_ball_confidence)
        self.ready_lateral_tolerance = float(ready_lateral_tolerance)
        self.minimum_ball_forward_distance = float(minimum_ball_forward_distance)
        self.policy_dt = float(policy_dt)
        self.transition_duration_s = float(transition_duration_s)
        self.transition_residual_scale = float(transition_residual_scale)
        self.register_buffer("_phase", torch.zeros(0, dtype=torch.long))
        self.register_buffer("_elapsed", torch.zeros(0))
        self.register_buffer("_adjust_last", torch.zeros(0, 12))
        self.register_buffer("_kick_last", torch.zeros(0, 12))
        self.register_buffer("_transition_last", torch.zeros(0, 12))

    def _ensure_state(self, observation: torch.Tensor) -> None:
        batch = observation.size(0)
        if self._phase.numel() == batch:
            return
        self._phase = torch.zeros(batch, dtype=torch.long, device=observation.device)
        self._elapsed = torch.zeros(batch, dtype=observation.dtype, device=observation.device)
        self._adjust_last = torch.zeros(batch, 12, dtype=observation.dtype, device=observation.device)
        self._kick_last = torch.zeros(batch, 12, dtype=observation.dtype, device=observation.device)
        self._transition_last = torch.zeros(batch, 12, dtype=observation.dtype, device=observation.device)

    @torch.jit.export
    def reset(self, reset_mask: torch.Tensor) -> None:
        if self._phase.numel() == 0:
            return
        mask = reset_mask.to(device=self._phase.device, dtype=torch.bool)
        self._phase[mask] = self.ADJUST
        self._elapsed[mask] = 0.0
        self._adjust_last[mask] = 0.0
        self._kick_last[mask] = 0.0
        self._transition_last[mask] = 0.0

    @torch.jit.export
    def get_phase(self) -> torch.Tensor:
        return self._phase.clone()

    def _ready(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[:, 6:8]
        direction = observation[:, 8:10] / torch.linalg.vector_norm(
            observation[:, 8:10], dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        heading_error = torch.atan2(direction[:, 1], direction[:, 0]).abs()
        distance = torch.linalg.vector_norm(ball, dim=1)
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        lateral_offset = torch.abs(torch.sum(ball * perpendicular, dim=1))
        return (
            (heading_error <= self.ready_heading_tolerance)
            & (distance >= self.minimum_ball_distance)
            & (distance <= self.maximum_ball_distance)
            & (lateral_offset <= self.ready_lateral_tolerance)
            & (ball[:, 0] >= self.minimum_ball_forward_distance)
            & (observation[:, 10] > 0.5)
            & (observation[:, 12] >= self.minimum_ball_confidence)
        )

    def _alpha(self) -> torch.Tensor:
        normalized = (self._elapsed / max(self.transition_duration_s, 1.0e-6)).clamp(0.0, 1.0)
        return normalized.square() * (3.0 - 2.0 * normalized)

    def _advance_phase(self, observation: torch.Tensor) -> None:
        start = (self._phase == self.ADJUST) & self._ready(observation)
        self._phase[start] = self.TRANSITION
        self._elapsed[start] = 0.0
        transition = self._phase == self.TRANSITION
        if self.transition_duration_s <= 0.0:
            self._phase[transition] = self.KICK
            return
        finished = transition & (self._elapsed >= self.transition_duration_s)
        self._phase[finished] = self.KICK
        self._elapsed[self._phase == self.TRANSITION] += self.policy_dt

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.size(1) != 49:
            raise RuntimeError("transition policy requires observation shape (N, 49)")
        self._ensure_state(observation)
        phase_before = self._phase.clone()
        alpha = self._alpha()
        adjust_obs = observation.clone()
        kick_obs = observation.clone()
        transition_obs = observation.clone()
        adjust_obs[:, 37:49] = self._adjust_last
        kick_obs[:, 37:49] = self._kick_last
        transition_obs[:, 37:49] = self._transition_last
        phase_value = torch.where(
            phase_before == self.ADJUST,
            torch.full_like(alpha, -1.0),
            torch.where(phase_before == self.KICK, torch.full_like(alpha, 2.0), alpha),
        )
        transition_obs = torch.cat((transition_obs, phase_value.unsqueeze(1)), dim=1)
        adjust_action = self.adjust_teacher(adjust_obs)
        kick_action = self.kick_teacher(kick_obs)
        transition_raw = self.transition_actor(transition_obs)
        envelope = (4.0 * alpha * (1.0 - alpha)).clamp(0.0, 1.0)
        envelope *= (phase_before == self.TRANSITION).float()
        residual = torch.tanh(transition_raw) * self.transition_residual_scale * envelope.unsqueeze(1)
        action = torch.lerp(adjust_action, kick_action, alpha.unsqueeze(1)) + residual
        action = torch.where((phase_before == self.ADJUST).unsqueeze(1), adjust_action, action)
        action = torch.where((phase_before == self.KICK).unsqueeze(1), kick_action, action)
        self._advance_phase(observation)
        active_adjust = self._phase != self.KICK
        active_kick = self._phase != self.ADJUST
        self._adjust_last[active_adjust] = adjust_action[active_adjust]
        self._kick_last[active_kick] = kick_action[active_kick]
        self._transition_last.copy_(transition_raw)
        return action
