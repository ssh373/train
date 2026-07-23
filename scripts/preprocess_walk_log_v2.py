"""Preprocess merged walk logs into training clips.

This script is designed for Booster K1 locomotion training. It:
- splits merged logs into constant-command clips,
- trims start/end transients,
- resamples to a fixed FPS,
- reorders raw serial joints into K1 joint order,
- optionally fixes arm joints to a safe pose,
- exports clip CSV files that are compatible with scripts/csv_to_npz.py,
- writes a joint mapping table and conversion helper scripts.

The produced clip CSVs are the main output. Run scripts/csv_to_npz.py on each
CSV to generate Isaac Lab motion NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from booster_assets.motions import K1_JOINT_NAMES as DEFAULT_JOINT_NAMES
except Exception:  # pragma: no cover
    # Fallback: load K1_JOINT_NAMES from local repository layout.
    motions_py = Path(__file__).resolve().parents[2] / "booster_assets-main" / "src" / "booster_assets" / "motions.py"
    if motions_py.exists():
        spec = importlib.util.spec_from_file_location("local_booster_assets_motions", motions_py)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        DEFAULT_JOINT_NAMES = tuple(module.K1_JOINT_NAMES)
    else:
        DEFAULT_JOINT_NAMES = tuple(f"joint_{i}" for i in range(24))

DEFAULT_BODY_NAMES = [
    "Trunk",
    "Head_2",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
]


@dataclass(frozen=True)
class ClipInfo:
    index: int
    command: tuple[float, float, float]
    start_time: float
    end_time: float
    frame_count: int
    csv_path: Path
    npz_path: Path


@dataclass(frozen=True)
class JointMappingRow:
    source_index: int
    source_name: str
    target_index: int
    target_name: str
    fixed_value: float | None


def quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    quat = np.empty((roll.shape[0], 4), dtype=np.float64)
    quat[:, 0] = sr * cp * cy - cr * sp * sy
    quat[:, 1] = cr * sp * cy + sr * cp * sy
    quat[:, 2] = cr * cp * sy - sr * sp * cy
    quat[:, 3] = cr * cp * cy + sr * sp * sy
    return quat


def interp_2d(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    out = np.empty((times_dst.shape[0], values_src.shape[1]), dtype=np.float64)
    for col in range(values_src.shape[1]):
        out[:, col] = np.interp(times_dst, times_src, values_src[:, col])
    return out


def segment_by_command(command_array: np.ndarray, tolerance: float = 1e-6) -> list[tuple[int, int, tuple[float, float, float]]]:
    if command_array.shape[0] == 0:
        return []

    segments: list[tuple[int, int, tuple[float, float, float]]] = []
    start_idx = 0
    current = command_array[0]

    def same_command(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.max(np.abs(a - b)) <= tolerance)

    for idx in range(1, command_array.shape[0]):
        if not same_command(command_array[idx], current):
            segments.append((start_idx, idx - 1, tuple(float(v) for v in current)))
            start_idx = idx
            current = command_array[idx]

    segments.append((start_idx, command_array.shape[0] - 1, tuple(float(v) for v in current)))
    return segments


def load_serial_order(serial_order_json: Path | None) -> list[str] | None:
    if serial_order_json is None:
        return None
    with serial_order_json.open("r", encoding="utf-8") as f:
        serial_order = json.load(f)
    if not isinstance(serial_order, list) or not all(isinstance(item, str) for item in serial_order):
        raise ValueError("serial_order_json must contain a JSON list of joint names")
    return serial_order


def parse_fixed_joint_args(names_arg: str | None, values_arg: str | None) -> tuple[list[str], np.ndarray]:
    if not names_arg and not values_arg:
        return [], np.array([], dtype=np.float64)
    if not names_arg or not values_arg:
        raise ValueError("Both --fixed_arm_joint_names and --fixed_arm_joint_values must be provided together")

    names = [item.strip() for item in names_arg.split(",") if item.strip()]
    values = np.array([float(item.strip()) for item in values_arg.split(",") if item.strip()], dtype=np.float64)
    if len(names) != len(values):
        raise ValueError("The number of fixed arm joint names must match the number of fixed arm joint values")
    return names, values


def build_joint_mapping(
    source_joint_names: list[str] | None,
    target_joint_names: list[str],
    fixed_joint_names: list[str],
    fixed_joint_values: np.ndarray,
) -> list[JointMappingRow]:
    source_names = source_joint_names if source_joint_names is not None else list(target_joint_names)
    source_index_by_name = {name: idx for idx, name in enumerate(source_names)}
    target_index_by_name = {name: idx for idx, name in enumerate(target_joint_names)}
    fixed_value_by_name = {name: float(value) for name, value in zip(fixed_joint_names, fixed_joint_values)}

    mapping: list[JointMappingRow] = []
    for target_name in target_joint_names:
        if target_name not in source_index_by_name:
            raise ValueError(f"Joint '{target_name}' missing from source order")
        mapping.append(
            JointMappingRow(
                source_index=source_index_by_name[target_name],
                source_name=target_name,
                target_index=target_index_by_name[target_name],
                target_name=target_name,
                fixed_value=fixed_value_by_name.get(target_name),
            )
        )
    return mapping


def write_joint_mapping_artifacts(
    output_dir: Path,
    source_joint_names: list[str] | None,
    target_joint_names: list[str],
    fixed_joint_names: list[str],
    fixed_joint_values: np.ndarray,
) -> tuple[Path, Path]:
    mapping = build_joint_mapping(source_joint_names, target_joint_names, fixed_joint_names, fixed_joint_values)
    mapping_csv_path = output_dir / "joint_mapping.csv"
    mapping_json_path = output_dir / "joint_mapping.json"

    with mapping_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_index", "source_name", "target_index", "target_name", "fixed_value"])
        for row in mapping:
            writer.writerow([
                row.source_index,
                row.source_name,
                row.target_index,
                row.target_name,
                "" if row.fixed_value is None else row.fixed_value,
            ])

    with mapping_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "source_index": row.source_index,
                    "source_name": row.source_name,
                    "target_index": row.target_index,
                    "target_name": row.target_name,
                    "fixed_value": row.fixed_value,
                }
                for row in mapping
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    return mapping_csv_path, mapping_json_path


def reorder_joint_columns(values: np.ndarray, source_joint_names: list[str] | None, target_joint_names: list[str]) -> np.ndarray:
    if source_joint_names is None:
        if values.shape[1] != len(target_joint_names):
            raise ValueError(
                f"Cannot assume identity joint order because value width ({values.shape[1]}) != target width ({len(target_joint_names)})"
            )
        return values.copy()

    if len(source_joint_names) != values.shape[1]:
        raise ValueError(
            f"Serial order list length ({len(source_joint_names)}) does not match value width ({values.shape[1]})"
        )

    index_by_name = {name: idx for idx, name in enumerate(source_joint_names)}
    missing = [name for name in target_joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing joint names in serial_order_json: {missing}")

    reordered = np.empty((values.shape[0], len(target_joint_names)), dtype=np.float64)
    for target_idx, joint_name in enumerate(target_joint_names):
        reordered[:, target_idx] = values[:, index_by_name[joint_name]]
    return reordered


def compute_base_trajectory(time_src: np.ndarray, cmd_src: np.ndarray, rpy_src: np.ndarray, base_height: float) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.unwrap(rpy_src[:, 2])
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    vx_world = cmd_src[:, 0] * cos_yaw - cmd_src[:, 1] * sin_yaw
    vy_world = cmd_src[:, 0] * sin_yaw + cmd_src[:, 1] * cos_yaw

    base_pos = np.zeros((time_src.shape[0], 3), dtype=np.float64)
    base_pos[:, 2] = base_height
    for idx in range(1, time_src.shape[0]):
        dt = max(time_src[idx] - time_src[idx - 1], 1e-6)
        base_pos[idx, 0] = base_pos[idx - 1, 0] + 0.5 * (vx_world[idx - 1] + vx_world[idx]) * dt
        base_pos[idx, 1] = base_pos[idx - 1, 1] + 0.5 * (vy_world[idx - 1] + vy_world[idx]) * dt
        base_pos[idx, 2] = base_height

    base_quat = quat_from_euler_xyz(rpy_src[:, 0], rpy_src[:, 1], yaw)
    return base_pos, base_quat


def build_motion_arrays(
    time_src: np.ndarray,
    joint_pos_src: np.ndarray,
    joint_vel_src: np.ndarray,
    imu_rpy_src: np.ndarray,
    imu_gyro_src: np.ndarray,
    cmd_src: np.ndarray,
    output_fps: int,
    base_height: float,
) -> dict[str, np.ndarray]:
    base_pos_src, base_quat_src = compute_base_trajectory(time_src, cmd_src, imu_rpy_src, base_height)

    duration = time_src[-1] - time_src[0]
    if duration <= 0.0:
        raise ValueError("Clip duration is too short")

    frame_count = max(2, int(round(duration * output_fps)) + 1)
    time_dst = np.linspace(0.0, duration, frame_count, dtype=np.float64)
    time_src_rel = time_src - time_src[0]

    base_pos = interp_2d(time_src_rel, base_pos_src, time_dst)
    base_quat = interp_2d(time_src_rel, base_quat_src, time_dst)
    base_quat = base_quat / np.linalg.norm(base_quat, axis=1, keepdims=True)
    joint_pos = interp_2d(time_src_rel, joint_pos_src, time_dst)
    joint_vel = interp_2d(time_src_rel, joint_vel_src, time_dst)
    body_lin_vel = np.gradient(base_pos, time_dst, axis=0)
    body_ang_vel = interp_2d(time_src_rel, imu_gyro_src, time_dst)

    return {
        "time": time_dst,
        "base_pos": base_pos,
        "base_quat": base_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_lin_vel": body_lin_vel,
        "body_ang_vel": body_ang_vel,
    }


def export_clip(
    clip_dir: Path,
    clip_name: str,
    motion: dict[str, np.ndarray],
    joint_names: list[str],
    body_names: list[str],
    fixed_joint_names: list[str],
    fixed_joint_values: np.ndarray,
    output_fps: int,
    include_csv: bool,
    include_npz: bool,
) -> tuple[Path | None, Path | None]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    csv_path = clip_dir / f"{clip_name}.csv"
    npz_path = clip_dir / f"{clip_name}.npz"

    joint_pos = motion["joint_pos"].copy()
    joint_vel = motion["joint_vel"].copy()

    if fixed_joint_names:
        joint_index_by_name = {name: idx for idx, name in enumerate(joint_names)}
        for name, value in zip(fixed_joint_names, fixed_joint_values):
            if name not in joint_index_by_name:
                raise ValueError(f"Fixed joint name '{name}' not found in joint_names")
            idx = joint_index_by_name[name]
            joint_pos[:, idx] = value
            joint_vel[:, idx] = 0.0

    motion_csv = np.concatenate([motion["base_pos"], motion["base_quat"], joint_pos], axis=1)
    if include_csv:
        np.savetxt(csv_path, motion_csv, delimiter=",", fmt="%.10f")
    else:
        csv_path = None

    if include_npz:
        body_count = len(body_names)
        body_pos_w = np.repeat(motion["base_pos"][:, None, :], body_count, axis=1)
        body_quat_w = np.repeat(motion["base_quat"][:, None, :], body_count, axis=1)
        body_lin_vel_w = np.repeat(motion["body_lin_vel"][:, None, :], body_count, axis=1)
        body_ang_vel_w = np.repeat(motion["body_ang_vel"][:, None, :], body_count, axis=1)
        np.savez_compressed(
            npz_path,
            fps=np.array([output_fps], dtype=np.int32),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            body_pos_w=body_pos_w.astype(np.float32),
            body_quat_w=body_quat_w.astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.astype(np.float32),
            joint_names=np.array(joint_names, dtype=object),
            body_names=np.array(body_names, dtype=object),
            command=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
    else:
        npz_path = None

    return csv_path, npz_path


def load_merged_rows(input_csv: Path) -> dict[str, np.ndarray]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    serial_q_keys = [name for name in fieldnames if name.startswith("serial_q_")]
    serial_dq_keys = [name for name in fieldnames if name.startswith("serial_dq_")]
    if not serial_q_keys or not serial_dq_keys:
        raise ValueError("Missing serial_q_* or serial_dq_* columns in merged CSV")

    serial_q_keys.sort(key=lambda key: int(key.rsplit("_", 1)[1]))
    serial_dq_keys.sort(key=lambda key: int(key.rsplit("_", 1)[1]))

    time = np.array([float(row["time"]) for row in rows], dtype=np.float64)
    epoch_ns = np.array([int(row["epoch_ns"]) for row in rows], dtype=np.int64)
    imu_rpy = np.array([[float(row["imu_r"]), float(row["imu_p"]), float(row["imu_y"])] for row in rows], dtype=np.float64)
    imu_gyro = np.array([[float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])] for row in rows], dtype=np.float64)
    cmd = np.array([[float(row["cmd_vx"]), float(row["cmd_vy"]), float(row["cmd_vyaw"])] for row in rows], dtype=np.float64)
    joint_pos = np.array([[float(row[key]) for key in serial_q_keys] for row in rows], dtype=np.float64)
    joint_vel = np.array([[float(row[key]) for key in serial_dq_keys] for row in rows], dtype=np.float64)

    order = np.argsort(epoch_ns)
    return {
        "time": time[order],
        "epoch_ns": epoch_ns[order],
        "imu_rpy": imu_rpy[order],
        "imu_gyro": imu_gyro[order],
        "cmd": cmd[order],
        "joint_pos": joint_pos[order],
        "joint_vel": joint_vel[order],
    }


def write_conversion_commands(output_dir: Path, clips: list[ClipInfo], output_fps: int) -> tuple[Path, Path]:
    ps1_path = output_dir / "convert_clips.ps1"
    sh_path = output_dir / "convert_clips.sh"

    ps1_lines = [
        "# Auto-generated by preprocess_walk_log_v2.py",
        "# Run this from the train repository root or adjust $TrainRoot below.",
        "$TrainRoot = \"C:\\Users\\sinsu\\Desktop\\soccer_ws\\train\"",
        "Set-Location $TrainRoot",
        "",
    ]
    sh_lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by preprocess_walk_log_v2.py",
        "set -euo pipefail",
        'TRAIN_ROOT="/mnt/c/Users/sinsu/Desktop/soccer_ws/train"',
        'cd "$TRAIN_ROOT"',
        "",
    ]

    for clip in clips:
        clip_csv_windows = clip.csv_path.as_posix().replace("/", "\\")
        clip_npz_windows = clip.npz_path.as_posix().replace("/", "\\")
        ps1_lines.append(
            "python scripts/csv_to_npz.py --headless "
            f'--input_file "{clip_csv_windows}" --input_fps {output_fps} '
            f'--output_name "{clip_npz_windows}" --output_fps {output_fps}'
        )
        sh_lines.append(
            "python scripts/csv_to_npz.py --headless "
            f'--input_file "{clip.csv_path.as_posix()}" --input_fps {output_fps} '
            f'--output_name "{clip.npz_path.as_posix()}" --output_fps {output_fps}'
        )

    ps1_path.write_text("\n".join(ps1_lines) + "\n", encoding="utf-8")
    sh_path.write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
    return ps1_path, sh_path


def preprocess_walk_log(args: argparse.Namespace) -> list[ClipInfo]:
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    clip_csv_dir = output_dir / "csv"
    clip_npz_dir = output_dir / "npz"
    manifest_path = output_dir / "manifest.json"
    segment_summary_path = output_dir / "segment_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged_rows(input_csv)
    serial_order_names = load_serial_order(Path(args.serial_order_json) if args.serial_order_json else None)
    robot_joint_names = list(DEFAULT_JOINT_NAMES)
    body_names = list(DEFAULT_BODY_NAMES)

    fixed_joint_names, fixed_joint_values = parse_fixed_joint_args(args.fixed_arm_joint_names, args.fixed_arm_joint_values)
    mapping_csv_path, mapping_json_path = write_joint_mapping_artifacts(
        output_dir=output_dir,
        source_joint_names=serial_order_names,
        target_joint_names=robot_joint_names,
        fixed_joint_names=fixed_joint_names,
        fixed_joint_values=fixed_joint_values,
    )
    print(f"Joint mapping: {mapping_csv_path}")
    print(f"Joint mapping: {mapping_json_path}")

    joint_pos_robot = reorder_joint_columns(merged["joint_pos"], serial_order_names, robot_joint_names)
    joint_vel_robot = reorder_joint_columns(merged["joint_vel"], serial_order_names, robot_joint_names)
    segments = segment_by_command(merged["cmd"], tolerance=args.command_tol)
    stop_trim_sec = args.trim_sec if args.stop_trim_sec is None else args.stop_trim_sec

    clips: list[ClipInfo] = []
    summary_rows: list[dict[str, object]] = []

    for _, (start_idx, end_idx, command) in enumerate(segments, start=1):
        command_norm = float(np.linalg.norm(command))
        is_stop_segment = command_norm <= args.command_tol
        if is_stop_segment and not args.include_stop_segments:
            continue

        segment_start_time = merged["time"][start_idx]
        segment_end_time = merged["time"][end_idx]
        trim_sec = stop_trim_sec if is_stop_segment else args.trim_sec
        trimmed_start_time = segment_start_time + trim_sec
        trimmed_end_time = segment_end_time - trim_sec
        if trimmed_end_time <= trimmed_start_time:
            continue

        segment_mask = (merged["time"] >= trimmed_start_time) & (merged["time"] <= trimmed_end_time)
        if int(segment_mask.sum()) < 2:
            continue

        time_src = merged["time"][segment_mask]
        motion = build_motion_arrays(
            time_src=time_src,
            joint_pos_src=joint_pos_robot[segment_mask],
            joint_vel_src=joint_vel_robot[segment_mask],
            imu_rpy_src=merged["imu_rpy"][segment_mask],
            imu_gyro_src=merged["imu_gyro"][segment_mask],
            cmd_src=merged["cmd"][segment_mask],
            output_fps=args.output_fps,
            base_height=args.base_height,
        )

        clip_name = f"clip_{len(clips) + 1:04d}_vx{command[0]:+0.2f}_vy{command[1]:+0.2f}_vyaw{command[2]:+0.2f}".replace(
            "+", "p"
        ).replace("-", "m")

        csv_path, _ = export_clip(
            clip_dir=clip_csv_dir,
            clip_name=clip_name,
            motion=motion,
            joint_names=robot_joint_names,
            body_names=body_names,
            fixed_joint_names=fixed_joint_names,
            fixed_joint_values=fixed_joint_values,
            output_fps=args.output_fps,
            include_csv=args.export_csv,
            include_npz=False,
        )

        npz_path = None
        if args.export_npz:
            _, npz_path = export_clip(
                clip_dir=clip_npz_dir,
                clip_name=clip_name,
                motion=motion,
                joint_names=robot_joint_names,
                body_names=body_names,
                fixed_joint_names=fixed_joint_names,
                fixed_joint_values=fixed_joint_values,
                output_fps=args.output_fps,
                include_csv=False,
                include_npz=True,
            )

        clip_index = len(clips) + 1
        clips.append(
            ClipInfo(
                index=clip_index,
                command=command,
                start_time=float(trimmed_start_time),
                end_time=float(trimmed_end_time),
                frame_count=int(motion["base_pos"].shape[0]),
                csv_path=csv_path if csv_path is not None else Path(),
                npz_path=npz_path if npz_path is not None else Path(),
            )
        )
        summary_rows.append(
            {
                "index": clip_index,
                "command_vx": command[0],
                "command_vy": command[1],
                "command_vyaw": command[2],
                "start_time": float(trimmed_start_time),
                "end_time": float(trimmed_end_time),
                "frame_count": int(motion["base_pos"].shape[0]),
                "csv_path": str(csv_path) if csv_path is not None else "",
                "npz_path": str(npz_path) if npz_path is not None else "",
            }
        )

    with segment_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "command_vx",
                "command_vy",
                "command_vyaw",
                "start_time",
                "end_time",
                "frame_count",
                "csv_path",
                "npz_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "input_csv": str(input_csv),
        "output_fps": args.output_fps,
        "base_height": args.base_height,
        "command_tol": args.command_tol,
        "trim_sec": args.trim_sec,
        "stop_trim_sec": stop_trim_sec,
        "include_stop_segments": args.include_stop_segments,
        "serial_order_json": args.serial_order_json,
        "fixed_arm_joint_names": fixed_joint_names,
        "fixed_arm_joint_values": fixed_joint_values.tolist(),
        "joint_names": robot_joint_names,
        "body_names": body_names,
        "mapping_csv": str(mapping_csv_path),
        "mapping_json": str(mapping_json_path),
        "segment_summary_csv": str(segment_summary_path),
        "clips": [
            {
                "index": clip.index,
                "command": list(clip.command),
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "frame_count": clip.frame_count,
                "csv_path": str(clip.csv_path),
                "npz_path": str(clip.npz_path),
            }
            for clip in clips
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.write_conversion_commands:
        ps1_path, sh_path = write_conversion_commands(output_dir, clips, args.output_fps)
        print(f"Conversion commands: {ps1_path}")
        print(f"Conversion commands: {sh_path}")

    print(f"Saved {len(clips)} clips to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return clips


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess merged walk logs into motion clips.")
    parser.add_argument("--input_csv", required=True, help="Merged walk CSV with cmd columns.")
    parser.add_argument("--output_dir", required=True, help="Directory to write clip CSV/NPZ files.")
    parser.add_argument("--output_fps", type=int, default=50, help="Target FPS for resampled clips.")
    parser.add_argument("--base_height", type=float, default=0.30, help="Trunk/base height used in the exported motion.")
    parser.add_argument("--trim_sec", type=float, default=0.50, help="Seconds trimmed from each side of a segment.")
    parser.add_argument(
        "--stop_trim_sec",
        type=float,
        default=None,
        help="Optional trim for 0,0,0 command segments. Defaults to --trim_sec when omitted.",
    )
    parser.add_argument("--command_tol", type=float, default=1e-6, help="Command equality tolerance for segmenting.")
    parser.add_argument("--include_stop_segments", action="store_true", help="Keep 0,0,0 segments as clips.")
    parser.add_argument("--serial_order_json", type=str, default=None, help="Optional JSON array describing raw serial joint order.")
    parser.add_argument("--fixed_arm_joint_names", type=str, default=None, help="Comma-separated joint names to overwrite with a fixed pose.")
    parser.add_argument("--fixed_arm_joint_values", type=str, default=None, help="Comma-separated joint values matching --fixed_arm_joint_names.")
    parser.add_argument("--export_csv", action="store_true", default=True, help="Write clip CSV files (default: on).")
    parser.add_argument("--no_export_csv", dest="export_csv", action="store_false", help="Disable clip CSV output.")
    parser.add_argument("--export_npz", action="store_true", default=True, help="Write clip NPZ files (default: on).")
    parser.add_argument("--no_export_npz", dest="export_npz", action="store_false", help="Disable clip NPZ output.")
    parser.add_argument("--write_conversion_commands", action="store_true", default=True, help="Write helper scripts for csv_to_npz conversion (default: on).")
    parser.add_argument("--no_write_conversion_commands", dest="write_conversion_commands", action="store_false", help="Do not write helper scripts for csv_to_npz conversion.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    preprocess_walk_log(args)


if __name__ == "__main__":
    main()
