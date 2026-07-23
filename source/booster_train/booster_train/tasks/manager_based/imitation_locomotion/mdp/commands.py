"""Velocity commands coupled to weak, command-labelled motion references."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .curriculums import stage_index
from .reference_motion import ReferenceMotionLibrary

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReferenceVelocityCommand(CommandTerm):
    """Generate ``[vx, vy, wz]`` while maintaining a reward-only joint reference.

    The policy command is always three dimensional.  Reference joint state,
    confidence, and clip progress are separate buffers, which lets the actor run
    on the robot without loading any motion file.
    """

    cfg: "ReferenceVelocityCommandCfg"

    # 0=stand, 1=x, 2=y, 3=yaw, 4=xy, 5=x+yaw, 6=y+yaw, 7=xyz
    _MODE_AXES = (
        (),
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    )

    def __init__(self, cfg: "ReferenceVelocityCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._validate_cfg()
        self.robot: Articulation = env.scene[cfg.asset_name]
        joint_ids, joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        if list(joint_names) != list(cfg.joint_names):
            raise ValueError(
                "Reference joints did not resolve in the requested order. "
                f"requested={cfg.joint_names}, resolved={joint_names}"
            )
        self.joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)

        self.library = ReferenceMotionLibrary(
            cfg.dataset_dirs,
            cfg.joint_names,
            self.device,
            first_clip_head_trim_s=cfg.first_clip_head_trim_s,
            jump_threshold=cfg.jump_threshold,
            min_segment_frames=cfg.min_segment_frames,
            min_moving_activity=cfg.min_moving_activity,
            max_joint_velocity=cfg.max_reference_joint_velocity,
            command_zero_tol=cfg.command_zero_tolerance,
        )

        self.vel_command_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        default_q = self.robot.data.default_joint_pos[:, self.joint_ids]
        self.reference_joint_pos = default_q.clone()
        self.reference_joint_vel = torch.zeros_like(self.reference_joint_pos)
        self.reference_confidence = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reference_clip_progress = torch.zeros_like(self.reference_confidence)

        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_cursors = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._base_confidence = torch.zeros_like(self.reference_confidence)
        self._transition_elapsed = torch.zeros_like(self.reference_confidence)
        self._transition_start_q = default_q.clone()
        self._transition_start_dq = torch.zeros_like(default_q)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_confidence"] = torch.zeros(self.num_envs, device=self.device)

    def _validate_cfg(self) -> None:
        if self.cfg.steps_per_iteration <= 0:
            raise ValueError("steps_per_iteration must be positive")
        if any(right <= left for left, right in zip(self.cfg.stage_boundaries[:-1], self.cfg.stage_boundaries[1:])):
            raise ValueError("stage_boundaries must be strictly increasing")
        if not self.cfg.mode_probabilities:
            raise ValueError("mode_probabilities cannot be empty")
        for row in self.cfg.mode_probabilities:
            if len(row) != len(self._MODE_AXES) or any(value < 0.0 for value in row) or sum(row) <= 0.0:
                raise ValueError("Each mode_probabilities row must have eight non-negative values and a positive sum")
        if not self.cfg.speed_scales or any(value <= 0.0 for value in self.cfg.speed_scales):
            raise ValueError("speed_scales must contain positive values")
        if not self.cfg.recorded_command_probability or any(
            value < 0.0 or value > 1.0 for value in self.cfg.recorded_command_probability
        ):
            raise ValueError("recorded_command_probability values must lie in [0, 1]")
        if len(self.cfg.compound_reference_support) != 4:
            raise ValueError("compound_reference_support must contain stand, 1-axis, 2-axis, and 3-axis values")
        if len(self.cfg.reference_distance_scales) != 3 or any(
            value <= 0.0 for value in self.cfg.reference_distance_scales
        ):
            raise ValueError("reference_distance_scales must contain three positive values")
        if self.cfg.reference_confidence_sigma <= 0.0:
            raise ValueError("reference_confidence_sigma must be positive")
        if len(self.cfg.play_command) != 3:
            raise ValueError("play_command must contain [vx, vy, wz]")

    def __str__(self) -> str:
        return (
            "ReferenceVelocityCommand:\n"
            f"\tCommand dim: {tuple(self.command.shape[1:])}\n"
            f"\tReference clips: {self.library.num_clips}\n"
            f"\tDatasets: {self.cfg.dataset_dirs}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_ids]

    @property
    def curriculum_stage(self) -> int:
        return stage_index(self._env, self.cfg.stage_boundaries, self.cfg.steps_per_iteration)

    def _update_metrics(self):
        max_steps = max(self.cfg.resampling_time_range[1] / self._env.step_dt, 1.0)
        self.metrics["error_vel_xy"] += (
            torch.linalg.vector_norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=1
            )
            / max_steps
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_steps
        )
        self.metrics["reference_confidence"] += self.reference_confidence / max_steps

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        if self.cfg.play:
            play_command = torch.tensor(self.cfg.play_command, dtype=torch.float32, device=self.device)
            self.vel_command_b[env_ids] = play_command
        else:
            self.vel_command_b[env_ids] = self._sample_commands(env_ids.numel())

        selected_clips, confidence = self.library.select_clips(
            self.vel_command_b[env_ids],
            distance_scales=self.cfg.reference_distance_scales,
            confidence_sigma=self.cfg.reference_confidence_sigma,
            compound_support=self.cfg.compound_reference_support,
            zero_tolerance=self.cfg.command_zero_tolerance,
        )
        self.clip_ids[env_ids] = selected_clips
        self._base_confidence[env_ids] = confidence
        self.clip_cursors[env_ids] = self.library.closest_cursors(
            self.robot_joint_pos[env_ids], selected_clips
        )
        self._begin_transition(env_ids)
        self._refresh_reference(env_ids)

    def _update_command(self):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        clip_ids = self.clip_ids
        lengths = self.library.lengths[clip_ids].to(torch.float32)
        advance = self.library.fps[clip_ids] * self._env.step_dt
        advance = torch.where(self.library.is_static[clip_ids], torch.zeros_like(advance), advance)
        next_cursor = self.clip_cursors + advance
        wrapped = next_cursor >= lengths
        self.clip_cursors = torch.remainder(next_cursor, lengths)

        wrap_ids = torch.nonzero(wrapped, as_tuple=False).flatten()
        if wrap_ids.numel() > 0:
            self._begin_transition(wrap_ids)
        self._transition_elapsed += self._env.step_dt
        self._refresh_reference(env_ids)

    def _begin_transition(self, env_ids: torch.Tensor) -> None:
        self._transition_start_q[env_ids] = self.robot_joint_pos[env_ids]
        self._transition_start_dq[env_ids] = self.robot_joint_vel[env_ids]
        self._transition_elapsed[env_ids] = 0.0

    def _refresh_reference(self, env_ids: torch.Tensor) -> None:
        clip_ids = self.clip_ids[env_ids]
        cursors = self.clip_cursors[env_ids]
        target_q, target_dq = self.library.sample(clip_ids, cursors)

        duration = max(self.cfg.reference_blend_time_s, 1.0e-6)
        alpha = torch.clamp(self._transition_elapsed[env_ids] / duration, 0.0, 1.0)
        # Smoothstep avoids a target-velocity kink at command changes and clip wraps.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self.reference_joint_pos[env_ids] = torch.lerp(
            self._transition_start_q[env_ids], target_q, alpha.unsqueeze(-1)
        )
        self.reference_joint_vel[env_ids] = torch.lerp(
            self._transition_start_dq[env_ids], target_dq, alpha.unsqueeze(-1)
        )

        seam = self.library.seam_confidence(
            clip_ids, cursors, self.cfg.reference_seam_fraction
        )
        self.reference_confidence[env_ids] = self._base_confidence[env_ids] * seam * alpha
        self.reference_clip_progress[env_ids] = self.library.clip_progress(clip_ids, cursors)

    def _sample_commands(self, count: int) -> torch.Tensor:
        stage = min(self.curriculum_stage, len(self.cfg.mode_probabilities) - 1)
        probabilities = torch.tensor(
            self.cfg.mode_probabilities[stage], dtype=torch.float32, device=self.device
        )
        probabilities /= probabilities.sum()
        modes = torch.multinomial(probabilities, count, replacement=True)
        commands = torch.zeros((count, 3), dtype=torch.float32, device=self.device)

        ranges = (
            self.cfg.ranges.lin_vel_x,
            self.cfg.ranges.lin_vel_y,
            self.cfg.ranges.ang_vel_z,
        )
        speed_scale = self.cfg.speed_scales[min(stage, len(self.cfg.speed_scales) - 1)]
        for mode, axes in enumerate(self._MODE_AXES):
            selected = torch.nonzero(modes == mode, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            for axis in axes:
                commands[selected, axis] = self._sample_component(
                    selected.numel(), ranges[axis], speed_scale
                )

        recorded_probability = self.cfg.recorded_command_probability[
            min(stage, len(self.cfg.recorded_command_probability) - 1)
        ]
        use_recorded = torch.rand(count, device=self.device) < recorded_probability
        for axis_mode, axis in ((1, 0), (2, 1), (3, 2)):
            selected = torch.nonzero((modes == axis_mode) & use_recorded, as_tuple=False).flatten()
            if selected.numel() > 0:
                commands[selected] = self._sample_recorded_axis(axis, selected.numel())
        return commands

    def _sample_component(
        self, count: int, value_range: tuple[float, float], speed_scale: float
    ) -> torch.Tensor:
        lower, upper = float(value_range[0]) * speed_scale, float(value_range[1]) * speed_scale
        values = torch.empty(count, device=self.device).uniform_(lower, upper)
        too_small = torch.abs(values) < self.cfg.min_moving_command
        signs = torch.where(
            values >= 0.0,
            torch.ones_like(values),
            -torch.ones_like(values),
        )
        random_signs = torch.where(
            torch.rand_like(values) >= 0.5,
            torch.ones_like(values),
            -torch.ones_like(values),
        )
        signs = torch.where(values == 0.0, random_signs, signs)
        minimum = min(self.cfg.min_moving_command, max(abs(lower), abs(upper)))
        return torch.where(too_small, signs * minimum, values).clamp(lower, upper)

    def _sample_recorded_axis(self, axis: int, count: int) -> torch.Tensor:
        recorded = self.library.recorded_commands
        axis_mask = torch.abs(recorded[:, axis]) > self.cfg.command_zero_tolerance
        other_axes = [index for index in range(3) if index != axis]
        axis_mask &= torch.all(
            torch.abs(recorded[:, other_axes]) <= self.cfg.command_zero_tolerance, dim=1
        )
        candidates = recorded[axis_mask]
        if candidates.shape[0] == 0:
            raise RuntimeError(f"No recorded single-axis command is available for axis {axis}")
        indexes = torch.randint(candidates.shape[0], (count,), device=self.device)
        return candidates[indexes]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # This task intentionally avoids thousands of reference markers.
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class ReferenceVelocityCommandCfg(CommandTermCfg):
    """Configuration for :class:`ReferenceVelocityCommand`."""

    class_type: type = ReferenceVelocityCommand
    asset_name: str = MISSING
    dataset_dirs: list[str] = MISSING
    joint_names: list[str] = MISSING

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    # These are PPO iterations, converted using steps_per_iteration below.
    stage_boundaries: tuple[int, int, int] = (500, 3_000, 10_000)
    steps_per_iteration: int = 24
    mode_probabilities: tuple[tuple[float, ...], ...] = (
        (0.15, 0.35, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0),
        (0.10, 0.35, 0.275, 0.275, 0.0, 0.0, 0.0, 0.0),
        (0.10, 0.15, 0.12, 0.12, 0.18, 0.165, 0.165, 0.0),
        (0.08, 0.12, 0.10, 0.10, 0.17, 0.14, 0.14, 0.15),
    )
    recorded_command_probability: tuple[float, float, float, float] = (0.90, 0.60, 0.20, 0.05)
    speed_scales: tuple[float, float, float, float] = (1.0, 0.50, 0.75, 1.0)
    min_moving_command: float = 0.10
    command_zero_tolerance: float = 1.0e-6

    # Confidence index is the number of non-zero command axes: stand/axis/2-axis/3-axis.
    compound_reference_support: tuple[float, float, float, float] = (0.50, 1.0, 0.35, 0.15)
    reference_distance_scales: tuple[float, float, float] = (0.35, 0.35, 0.50)
    reference_confidence_sigma: float = 1.0
    reference_blend_time_s: float = 0.30
    reference_seam_fraction: float = 0.05

    first_clip_head_trim_s: float = 5.6
    jump_threshold: float = 0.35
    min_segment_frames: int = 50
    min_moving_activity: float = 0.05
    max_reference_joint_velocity: float = 15.0

    play: bool = False
    play_command: tuple[float, float, float] = (0.3, 0.0, 0.0)
