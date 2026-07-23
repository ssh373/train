"""Privileged reference observations for the imitation-guided critic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import ReferenceVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _command_term(env: "ManagerBasedEnv", command_name: str) -> ReferenceVelocityCommand:
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, ReferenceVelocityCommand):
        raise TypeError(f"Command term '{command_name}' is not a ReferenceVelocityCommand")
    return term


def reference_joint_pos_rel(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    """Reference leg positions relative to the robot's default joint pose."""

    command = _command_term(env, command_name)
    default = command.robot.data.default_joint_pos[:, command.joint_ids]
    return command.reference_joint_pos - default


def reference_joint_vel(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    command = _command_term(env, command_name)
    return command.reference_joint_vel


def reference_confidence(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    command = _command_term(env, command_name)
    return command.reference_confidence.unsqueeze(-1)


def reference_clip_progress(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    """Normalized position in a recorded clip; this is not a single gait-cycle phase."""

    command = _command_term(env, command_name)
    return command.reference_clip_progress.unsqueeze(-1)

