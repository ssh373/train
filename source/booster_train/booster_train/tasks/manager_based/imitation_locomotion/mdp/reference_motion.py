"""Joint-only reference motion loading for imitation-guided locomotion.

The processed walking data has two incompatible NPZ variants.  The canonical
CSV format is common to both variants, so this loader deliberately reads joint
positions from CSV and command labels from each dataset's ``manifest.json``.
Root/body trajectories are not loaded because they are synthetic or invalid in
part of the currently collected data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path, PureWindowsPath
from typing import Sequence

import numpy as np
import torch


_LOGGER = logging.getLogger(__name__)


@dataclass
class _Clip:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    command: tuple[float, float, float]
    fps: float
    is_static: bool
    source: str


class ReferenceMotionLibrary:
    """Load and pack variable-length, command-labelled leg joint clips.

    Args:
        dataset_dirs: Directories containing ``manifest.json`` and ``csv/``.
        target_joint_names: Exact robot joint order used by rewards and commands.
        device: Torch device on which references are sampled.
        first_clip_head_trim_s: Data-specific removal for the incorrectly labelled
            head of clip 1. Set this to zero after regenerating clean datasets.
        jump_threshold: Split a clip when any tracked joint jumps by more than this
            many radians in one source frame.
        min_segment_frames: Discard short pieces created by jump splitting.
        min_moving_activity: Reject non-zero-command clips whose mean absolute
            joint speed is below this value; these are usually command/log errors.
        max_joint_velocity: Clamp finite-difference reference velocity because it
            is only used as a weak imitation target.
    """

    CSV_BASE_COLUMNS = 7  # base xyz + base quaternion xyzw

    def __init__(
        self,
        dataset_dirs: Sequence[str | Path],
        target_joint_names: Sequence[str],
        device: str | torch.device,
        *,
        first_clip_head_trim_s: float = 5.6,
        jump_threshold: float = 0.35,
        min_segment_frames: int = 50,
        min_moving_activity: float = 0.05,
        max_joint_velocity: float = 15.0,
        command_zero_tol: float = 1.0e-6,
    ):
        self.device = torch.device(device)
        self.target_joint_names = tuple(target_joint_names)
        if len(set(self.target_joint_names)) != len(self.target_joint_names):
            raise ValueError("target_joint_names contains duplicates")

        clips: list[_Clip] = []
        rejected: list[str] = []
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir).expanduser().resolve()
            manifest_path = dataset_path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")

            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)

            fps = float(manifest["output_fps"])
            source_joint_names = list(manifest["joint_names"])
            if len(set(source_joint_names)) != len(source_joint_names):
                raise ValueError(f"Duplicate joint names in {manifest_path}")
            missing = [name for name in self.target_joint_names if name not in source_joint_names]
            if missing:
                raise ValueError(f"Missing target joints in {manifest_path}: {missing}")
            joint_columns = [
                self.CSV_BASE_COLUMNS + source_joint_names.index(name) for name in self.target_joint_names
            ]
            expected_columns = self.CSV_BASE_COLUMNS + len(source_joint_names)

            for clip_info in manifest["clips"]:
                source = f"{dataset_path.name}/clip_{int(clip_info['index']):04d}"
                csv_path = self._resolve_csv_path(dataset_path, str(clip_info["csv_path"]))
                raw = np.atleast_2d(np.loadtxt(csv_path, delimiter=",", dtype=np.float64))
                if raw.shape[1] != expected_columns:
                    raise ValueError(
                        f"Unexpected CSV width for {csv_path}: {raw.shape[1]} != {expected_columns}"
                    )

                joint_pos = raw[:, joint_columns].astype(np.float32, copy=False)
                if int(clip_info["index"]) == 1 and first_clip_head_trim_s > 0.0:
                    joint_pos = joint_pos[int(round(first_clip_head_trim_s * fps)) :]

                command = tuple(float(value) for value in clip_info["command"])
                if len(command) != 3:
                    raise ValueError(f"Command must have three values in {manifest_path}: {command}")
                is_stop = np.linalg.norm(command) <= command_zero_tol

                if is_stop:
                    static_pose = self._stable_observed_pose(joint_pos, fps)
                    if static_pose is None:
                        rejected.append(f"{source}: no finite standing frame")
                        continue
                    static_q = np.stack((static_pose, static_pose), axis=0)
                    clips.append(
                        _Clip(
                            joint_pos=static_q,
                            joint_vel=np.zeros_like(static_q),
                            command=command,
                            fps=fps,
                            is_static=True,
                            source=source,
                        )
                    )
                    continue

                segments = self._split_clean_segments(joint_pos, jump_threshold)
                kept_any = False
                for segment_index, segment in enumerate(segments):
                    if segment.shape[0] < min_segment_frames:
                        continue
                    joint_vel = np.gradient(segment, 1.0 / fps, axis=0).astype(np.float32)
                    activity = float(np.mean(np.abs(joint_vel)))
                    if activity < min_moving_activity:
                        continue
                    joint_vel = np.clip(joint_vel, -max_joint_velocity, max_joint_velocity)
                    clips.append(
                        _Clip(
                            joint_pos=segment.astype(np.float32, copy=False),
                            joint_vel=joint_vel,
                            command=command,
                            fps=fps,
                            is_static=False,
                            source=f"{source}/part_{segment_index}",
                        )
                    )
                    kept_any = True
                if not kept_any:
                    rejected.append(f"{source}: no active segment >= {min_segment_frames} frames")

        if not clips:
            raise RuntimeError("No usable reference clips were loaded")

        self._pack(clips)
        _LOGGER.info(
            "Loaded %d imitation reference clips (%d rejected) from %d datasets",
            self.num_clips,
            len(rejected),
            len(dataset_dirs),
        )
        for reason in rejected:
            _LOGGER.warning("Rejected reference %s", reason)

    @staticmethod
    def _resolve_csv_path(dataset_path: Path, recorded_path: str) -> Path:
        """Resolve stale absolute Windows paths through the local dataset folder."""

        local_name = PureWindowsPath(recorded_path).name
        local_path = dataset_path / "csv" / local_name
        if local_path.is_file():
            return local_path
        direct_path = Path(recorded_path)
        if direct_path.is_file():
            return direct_path
        raise FileNotFoundError(f"Reference CSV not found: {local_path} (manifest: {recorded_path})")

    @staticmethod
    def _split_clean_segments(joint_pos: np.ndarray, jump_threshold: float) -> list[np.ndarray]:
        """Split on invalid frames and logger discontinuities."""

        if joint_pos.shape[0] == 0:
            return []
        finite = np.all(np.isfinite(joint_pos), axis=1)
        segments: list[np.ndarray] = []
        run_start = 0
        for index in range(joint_pos.shape[0] + 1):
            at_end = index == joint_pos.shape[0]
            if at_end or not finite[index]:
                if index > run_start:
                    finite_run = joint_pos[run_start:index]
                    jumps = np.flatnonzero(
                        np.max(np.abs(np.diff(finite_run, axis=0)), axis=1) > jump_threshold
                    )
                    split_points = [0, *(int(jump) + 1 for jump in jumps), finite_run.shape[0]]
                    segments.extend(
                        finite_run[start:end] for start, end in zip(split_points[:-1], split_points[1:])
                    )
                run_start = index + 1
        return segments

    @staticmethod
    def _stable_observed_pose(joint_pos: np.ndarray, fps: float) -> np.ndarray | None:
        """Select a low-speed *observed* frame instead of inventing a median pose."""

        joint_pos = joint_pos[np.all(np.isfinite(joint_pos), axis=1)]
        if joint_pos.shape[0] == 0:
            return None
        if joint_pos.shape[0] == 1:
            return joint_pos[0].copy()
        speed = np.mean(np.abs(np.gradient(joint_pos, 1.0 / fps, axis=0)), axis=1)
        candidate_count = max(1, int(round(0.2 * joint_pos.shape[0])))
        candidate_ids = np.argpartition(speed, candidate_count - 1)[:candidate_count]
        candidates = joint_pos[candidate_ids]
        center = np.median(candidates, axis=0)
        selected = candidate_ids[np.argmin(np.mean(np.square(candidates - center), axis=1))]
        return joint_pos[selected].copy()

    def _pack(self, clips: Sequence[_Clip]) -> None:
        self.num_clips = len(clips)
        max_frames = max(clip.joint_pos.shape[0] for clip in clips)
        num_joints = len(self.target_joint_names)
        self.joint_pos = torch.zeros(
            (self.num_clips, max_frames, num_joints), dtype=torch.float32, device=self.device
        )
        self.joint_vel = torch.zeros_like(self.joint_pos)
        for index, clip in enumerate(clips):
            length = clip.joint_pos.shape[0]
            self.joint_pos[index, :length] = torch.as_tensor(clip.joint_pos, device=self.device)
            self.joint_vel[index, :length] = torch.as_tensor(clip.joint_vel, device=self.device)
            if length < max_frames:
                self.joint_pos[index, length:] = self.joint_pos[index, length - 1]

        self.lengths = torch.tensor(
            [clip.joint_pos.shape[0] for clip in clips], dtype=torch.long, device=self.device
        )
        self.fps = torch.tensor([clip.fps for clip in clips], dtype=torch.float32, device=self.device)
        self.commands = torch.tensor([clip.command for clip in clips], dtype=torch.float32, device=self.device)
        self.is_static = torch.tensor([clip.is_static for clip in clips], dtype=torch.bool, device=self.device)
        self.sources = tuple(clip.source for clip in clips)
        self.recorded_commands = torch.unique(self.commands, dim=0)

    def select_clips(
        self,
        desired_commands: torch.Tensor,
        *,
        distance_scales: Sequence[float],
        confidence_sigma: float,
        compound_support: Sequence[float],
        zero_tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a single real clip and return its imitation confidence.

        Compound commands are never synthesized by blending joint poses.  Instead,
        the closest axis clip is selected and its confidence is reduced, leaving
        velocity tracking to solve the unseen command combination.
        """

        scales = torch.as_tensor(distance_scales, dtype=torch.float32, device=self.device)
        delta = (desired_commands[:, None, :] - self.commands[None, :, :]) / scales
        distance_sq = torch.sum(torch.square(delta), dim=-1)

        desired_stop = torch.linalg.vector_norm(desired_commands, dim=1) <= zero_tolerance
        ref_stop = torch.linalg.vector_norm(self.commands, dim=1) <= zero_tolerance
        invalid = torch.where(desired_stop[:, None], ~ref_stop[None, :], ref_stop[None, :])
        distance_sq = distance_sq.masked_fill(invalid, torch.inf)
        # Random tie-breaking distributes duplicate commands across recording sessions.
        distance_sq = distance_sq + torch.rand_like(distance_sq) * 1.0e-6
        selected_distance_sq, clip_ids = torch.min(distance_sq, dim=1)

        nonzero_axes = torch.sum(torch.abs(desired_commands) > zero_tolerance, dim=1).clamp(max=3)
        support_table = torch.as_tensor(compound_support, dtype=torch.float32, device=self.device)
        support = support_table[nonzero_axes]
        confidence = torch.exp(-0.5 * selected_distance_sq / (confidence_sigma**2)) * support
        return clip_ids, confidence.clamp_(0.0, 1.0)

    def sample(self, clip_ids: torch.Tensor, cursors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Linearly interpolate joint position and velocity at fractional frames."""

        lengths = self.lengths[clip_ids]
        wrapped = torch.remainder(cursors, lengths.to(cursors.dtype))
        frame0 = torch.floor(wrapped).long()
        frame1 = torch.remainder(frame0 + 1, lengths)
        blend = (wrapped - frame0.to(wrapped.dtype)).unsqueeze(-1)
        q0 = self.joint_pos[clip_ids, frame0]
        q1 = self.joint_pos[clip_ids, frame1]
        dq0 = self.joint_vel[clip_ids, frame0]
        dq1 = self.joint_vel[clip_ids, frame1]
        return torch.lerp(q0, q1, blend), torch.lerp(dq0, dq1, blend)

    def closest_cursors(
        self, robot_joint_pos: torch.Tensor, clip_ids: torch.Tensor, *, chunk_size: int = 256
    ) -> torch.Tensor:
        """Align each new reference to the nearest observed joint pose."""

        cursors = torch.zeros(robot_joint_pos.shape[0], dtype=torch.float32, device=self.device)
        for clip_id in torch.unique(clip_ids):
            selected = torch.nonzero(clip_ids == clip_id, as_tuple=False).flatten()
            clip_length = int(self.lengths[clip_id].item())
            reference = self.joint_pos[clip_id, :clip_length]
            for start in range(0, selected.numel(), chunk_size):
                batch_ids = selected[start : start + chunk_size]
                error = torch.mean(
                    torch.square(robot_joint_pos[batch_ids, None, :] - reference[None, :, :]), dim=-1
                )
                cursors[batch_ids] = torch.argmin(error, dim=1).to(torch.float32)
        return cursors

    def clip_progress(self, clip_ids: torch.Tensor, cursors: torch.Tensor) -> torch.Tensor:
        lengths = self.lengths[clip_ids]
        denominator = torch.clamp(lengths, min=1).to(cursors.dtype)
        progress = torch.remainder(cursors, denominator) / denominator
        return torch.where(self.is_static[clip_ids], torch.zeros_like(progress), progress)

    def seam_confidence(
        self, clip_ids: torch.Tensor, cursors: torch.Tensor, seam_fraction: float
    ) -> torch.Tensor:
        """Fade imitation around non-periodic clip ends before a wrap occurs."""

        progress = self.clip_progress(clip_ids, cursors)
        edge_distance = torch.minimum(progress, 1.0 - progress)
        confidence = torch.clamp(edge_distance / max(seam_fraction, 1.0e-6), 0.0, 1.0)
        return torch.where(self.is_static[clip_ids], torch.ones_like(confidence), confidence)
