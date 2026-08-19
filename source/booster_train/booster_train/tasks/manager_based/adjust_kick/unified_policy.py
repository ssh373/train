"""Stateful wrapper for the single learned adjust-kick actor.

The actor itself is the only neural network in the exported artifact.  The
small phase state is deployment bookkeeping: it supplies the same one-value
phase input that was present during PPO and prevents the actor from having to
infer whether an identical physical pose is before or after the handoff.
"""

from __future__ import annotations

import math

import torch


class LearnedUnifiedAdjustKickPolicy(torch.nn.Module):
    """One learned 50-to-12 actor with an internal adjust/kick phase latch."""

    ADJUST = 0
    TRANSITION = 1
    KICK = 2

    def __init__(
        self,
        actor,
        policy_dt: float = 0.02,
        target_ball_distance: float = 0.30,
        ready_position_tolerance: float = 0.08,
        ready_heading_tolerance_deg: float = 30.0,
        minimum_ball_distance: float = 0.20,
        maximum_ball_distance: float = 0.40,
        transition_duration_s: float = 0.20,
        minimum_ball_confidence: float = 0.05,
    ) -> None:
        super().__init__()
        self.actor = actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.policy_dt = float(policy_dt)
        self.target_ball_distance = float(target_ball_distance)
        self.ready_position_tolerance = float(ready_position_tolerance)
        self.ready_heading_tolerance = math.radians(ready_heading_tolerance_deg)
        self.minimum_ball_distance = float(minimum_ball_distance)
        self.maximum_ball_distance = float(maximum_ball_distance)
        self.minimum_ball_confidence = float(minimum_ball_confidence)
        self.transition_steps = max(
            1,
            int(round(transition_duration_s / max(policy_dt, 1.0e-6))),
        )
        self.transition_duration_s = float(transition_duration_s)
        self.register_buffer("_phase", torch.zeros(0, dtype=torch.long))
        self.register_buffer("_transition_elapsed", torch.zeros(0))
        self.register_buffer("_previous_action", torch.zeros(0, 12))

    def _ensure_state(self, observation: torch.Tensor) -> None:
        batch_size = observation.size(0)
        if self._phase.numel() == batch_size:
            return
        self._phase = torch.zeros(
            batch_size, dtype=torch.long, device=observation.device
        )
        self._transition_elapsed = torch.zeros(
            batch_size, dtype=observation.dtype, device=observation.device
        )
        self._previous_action = torch.zeros(
            batch_size, 12, dtype=observation.dtype, device=observation.device
        )

    @torch.jit.export
    def reset(self, reset_mask: torch.Tensor) -> None:
        if self._phase.numel() == 0:
            return
        mask = reset_mask.to(device=self._phase.device, dtype=torch.bool)
        self._phase[mask] = self.ADJUST
        self._transition_elapsed[mask] = 0.0
        self._previous_action[mask] = 0.0

    @torch.jit.export
    def get_phase(self) -> torch.Tensor:
        return self._phase.clone()

    def _ready(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[:, 6:8]
        target_direction = observation[:, 8:10]
        direction = target_direction / torch.linalg.vector_norm(
            target_direction, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        desired_base_error = ball - self.target_ball_distance * direction
        position_error = torch.linalg.vector_norm(desired_base_error, dim=1)
        heading_error = torch.atan2(direction[:, 1], direction[:, 0]).abs()
        ball_distance = torch.linalg.vector_norm(ball, dim=1)
        return (
            (position_error <= self.ready_position_tolerance)
            & (heading_error <= self.ready_heading_tolerance)
            & (ball_distance >= self.minimum_ball_distance)
            & (ball_distance <= self.maximum_ball_distance)
            & (observation[:, 10] > 0.5)
            & (observation[:, 12] >= self.minimum_ball_confidence)
        )

    def _phase_input(self, phase: torch.Tensor) -> torch.Tensor:
        normalized = (
            self._transition_elapsed / max(self.transition_duration_s, 1.0e-6)
        ).clamp(0.0, 1.0)
        alpha = normalized.square() * (3.0 - 2.0 * normalized)
        return torch.where(
            phase == self.ADJUST,
            torch.full_like(alpha, -1.0),
            torch.where(phase == self.KICK, torch.full_like(alpha, 2.0), alpha),
        )

    def _advance_phase(self, observation: torch.Tensor) -> None:
        start = (self._phase == self.ADJUST) & self._ready(observation)
        self._phase[start] = self.TRANSITION
        self._transition_elapsed[start] = 0.0

        transition = self._phase == self.TRANSITION
        if self.transition_duration_s <= 0.0:
            self._phase[transition] = self.KICK
            return
        finished = transition & (
            self._transition_elapsed >= self.transition_duration_s
        )
        self._phase[finished] = self.KICK
        still_transition = self._phase == self.TRANSITION
        self._transition_elapsed[still_transition] += self.policy_dt

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.size(1) != 49:
            raise RuntimeError("unified policy requires observation shape (N, 49)")
        self._ensure_state(observation)
        phase_before_advance = self._phase.clone()
        phase_value = self._phase_input(phase_before_advance)
        actor_observation_base = observation.clone()
        actor_observation_base[:, 37:49] = self._previous_action
        actor_observation = torch.cat(
            (actor_observation_base, phase_value.unsqueeze(1)), dim=1
        )
        action = self.actor(actor_observation)
        self._previous_action.copy_(action)
        self._advance_phase(observation)
        return action
