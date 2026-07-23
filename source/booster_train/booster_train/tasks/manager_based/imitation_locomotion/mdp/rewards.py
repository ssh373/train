"""Confidence-gated imitation rewards.

Only leg joint state is imitated.  Root and body rewards are intentionally
excluded because those fields are not reliable across the current datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import ReferenceVelocityCommand
from .curriculums import piecewise_linear_scale

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command_term(env: "ManagerBasedRLEnv", command_name: str) -> ReferenceVelocityCommand:
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, ReferenceVelocityCommand):
        raise TypeError(f"Command term '{command_name}' is not a ReferenceVelocityCommand")
    return term


def _curriculum_scale(
    env: "ManagerBasedRLEnv",
    iteration_knots: tuple[int, ...],
    values: tuple[float, ...],
    steps_per_iteration: int,
) -> float:
    return piecewise_linear_scale(env, iteration_knots, values, steps_per_iteration)


def reference_joint_position_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    iteration_knots: tuple[int, ...],
    curriculum_values: tuple[float, ...],
    steps_per_iteration: int,
) -> torch.Tensor:
    """Reward similarity to the reference pose, gated by data relevance and time."""

    command = _command_term(env, command_name)
    error = torch.mean(torch.square(command.robot_joint_pos - command.reference_joint_pos), dim=1)
    similarity = torch.exp(-error / (std**2))
    scale = _curriculum_scale(env, iteration_knots, curriculum_values, steps_per_iteration)
    return similarity * command.reference_confidence * scale


def reference_joint_velocity_exp(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float,
    iteration_knots: tuple[int, ...],
    curriculum_values: tuple[float, ...],
    steps_per_iteration: int,
) -> torch.Tensor:
    """Weak velocity imitation; recorded finite differences are deliberately secondary."""

    command = _command_term(env, command_name)
    error = torch.mean(torch.square(command.robot_joint_vel - command.reference_joint_vel), dim=1)
    similarity = torch.exp(-error / (std**2))
    scale = _curriculum_scale(env, iteration_knots, curriculum_values, steps_per_iteration)
    return similarity * command.reference_confidence * scale

