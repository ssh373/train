from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 생존
def survival(env):
    return torch.ones(env.num_envs, device=env.device)

# x velocity 추종
def track_lin_vel_x_exp(env, command_name: str, tracking_sigma: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-error / tracking_sigma)

# y velocity 추종
def track_lin_vel_y_exp(env, command_name: str, tracking_sigma: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-error / tracking_sigma)

# theta velocity 추종
def track_ang_vel_theta_exp(env, command_name: str, tracking_sigma: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error / tracking_sigma)

# x,y velocity 추종
def track_lin_vel_xy_exp(env, command_name: str, tracking_sigma: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / tracking_sigma)

# trunk 목표 높이
def base_height_l2(env, target_height: float = 0.52, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        base_height = base_height - terrain_height

    return torch.square(base_height - target_height)

# 직전 action에서 얼마나 변했는가
def action_rate_l2(env):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

# 발 이외에 접촉 부위 수
def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history # [num_envs, history_length, num_bodies, 3]
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

# 각도 값을 -pi ~ pi 로 변경
def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi

# quaternion -> yaw 
def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(dim=-1)
    return torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

# 몸통의 yaw와 발의 yaw가 얼마나 틀어졌는가
def _feet_raw_rel(asset: Articulation, foot_body_ids: list[int]) -> torch.Tensor:
    foot_quat_w = asset.data.body_quat_w[:, foot_body_ids]
    base_yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w).unsqueeze(-1)
    foot_yaw = _yaw_from_quat_wxyz(foot_quat_w)
    return _wrap_to_pi(foot_yaw - base_yaw)

# 왼쪽 발이 몸통 대비 yaw 각도가 command에서 원하는 값과 얼마나 다른지 
def foot_yaw_l_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    error = _wrap_to_pi(feet_yaw_rel[:, 0] - command[:, 4])
    return torch.square(error)

def foot_yaw_r_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    error = _wrap_to_pi(feet_yaw_rel[:, 1] - command[:, 5])
    return torch.square(error)

# 양발의 상대 위치를 몸통 방향 기준 좌표계로 바꿈
def _feet_offset_b(asset: Articulation, foot_body_ids: list[int], feet_distance_ref: float):
    feet_pos = asset.data.body_pos_w[:, foot_body_ids]
    base_yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)

    dx = feet_pos[:, 0, 0] - feet_pos[:, 1, 0]
    dy = feet_pos[:, 0, 1] - feet_pos[:, 1, 1]

    feet_x_offset = torch.cos(base_yaw) * dx + torch.sin(base_yaw) * dy
    feet_y_offset = -torch.sin(base_yaw) * dx + torch.cos(base_yaw) * dy
    feet_y_offset = feet_y_offset - feet_distance_ref
    return feet_x_offset, feet_y_offset

# 서 있을 때는 발 앞뒤 정렬을 강하게 요구
def feet_offset_x_l1(env, command_name: str, asset_cfg: SceneEntityCfg, max_forward_vel: float=1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    feet_x_offset, _ = _feet_offset_b(asset, asset_cfg.body_ids, feet_distance_ref=0.18)
    
    error = feet_x_offset - command[:, 8]
    penalty = torch.clip(torch.abs(error), min=0.0, max=0.1)
    # 전진 속도에 따른 패널티를 줄임 -> 약 1.0m/s 일 때 0이 될 것
    vel_scale = torch.clamp((1.0 - torch.abs(command[:, 0]) / max_forward_vel) ** 2, min=0.0, max=1.0) 
    return penalty * vel_scale

# 서 있을 때는 발 앞뒤 정렬을 강하게 요구
def feet_offset_y_l1(env, command_name: str, asset_cfg: SceneEntityCfg, feet_distance_ref: float = 0.18, max_lateral_vel: float = 1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, feet_y_offset = _feet_offset_b(asset, asset_cfg.body_ids, feet_distance_ref)

    error = feet_y_offset - command[:, 9]
    penalty = torch.clip(torch.abs(error), min=0.0, max=0.1)
    vel_scale = torch.clamp((1.0 - torch.abs(command[:, 1]) / max_lateral_vel) ** 2, min=0.0, max=1.0)
    return penalty * vel_scale

# 지금 들어야 하는 발이 실제로 지면에서 떨어져 있길 원함
def feet_swing(env, command_name: str, sensor_cfg: SceneEntityCfg, swing_period: float=0.2, threshold: float=1.0):
    command_term = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    phase = command_term.gait_process
    freq = command_term.gait_frequency

    # 왼발 swing 중심을 0.25, 오른발 swing 중심을 0.75로 설정
    left_swing = (torch.abs(phase - 0.25) < 0.5 * swing_period) & (freq > 1.0e-8)
    right_swing = (torch.abs(phase - 0.75) < 0.5 * swing_period) & (freq > 1.0e-8)

    return (left_swing & ~feet_contact[:, 0]).float() + (right_swing & ~feet_contact[:, 1]).float()

# quaternion -> roll pitch 변환
def _roll_pitch_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    qw, qx, qy, qz = quat.unbind(dim=-1)

    roll = torch.atan2(2.0 * (qw * qx + qy * qz),
                       1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    return roll, pitch

# 몸통에 roll, pitch가 얼마나 틀어졌는가
def orientation_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    roll, pitch = _roll_pitch_from_quat_wxyz(asset.data.root_quat_w)
    roll_error = _wrap_to_pi(roll - command[:, 7])
    pitch_error = _wrap_to_pi(pitch - command[:, 6])
    return torch.square(roll_error) + torch.square(pitch_error)

# 회전 명령이 아닐 때 몸통의 yaw가 얼마나 틀어졌는가
def heading_drift_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), deadband: float = 0.1) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    yaw = _yaw_from_quat_wxyz(asset.data.root_quat_w)

    if not hasattr(env, "ref_yaw") or env.ref_yaw.shape != yaw.shape:
        env.ref_yaw = yaw.clone()
    
    turning = torch.abs(command[:, 2]) >= deadband
    reset = env.episode_length_buf <= 1
    update_ref = reset | turning
    env.ref_yaw[update_ref] = yaw[update_ref]

    yaw_error = _wrap_to_pi(yaw - env.ref_yaw)
    no_turn = ~turning

    return torch.square(yaw_error) * no_turn.float()

# 위아래 움직임을 제곱해서 반환
def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

# x,y 축 각속도 제곱 합 -> 몸통 앞뒤 좌우 흔들림 억제
def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

# 선택한 관절에 실제 적용된 토크를 제곱 합 -> 큰 토크에 대한 패널티
def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

# 선택한 모든 관절의 속도를 제곱 합 -> 관절별 회전 속도 -> 관절 빠르게 움직임 방지
def joint_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

# 선택한 모든 관절의 가속도를 제곱 합 -> 갑작스러운 관절 움직임과 진동 방지
def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

# 선택한 관절 중 soft joint limit을 벗어난 관절 수
def joint_pos_limits_count(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]

    lower = limits[..., 0]
    upper = limits[..., 1]
    return torch.sum(((joint_pos < lower) | (joint_pos > upper)).float(), dim=1)

# 각 관절의 기계적 power 계산 후 양수인 power 합
def power_positive(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(power.clip(min=0.0), dim=1)

# 로봇 몸통의 선형가속도 얼마나 큰지 제곱합 -> 각가속도도 넣을지는 고민해보기
def root_lin_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]), dim=(1, 2))

# 발이 지면에 닿아 있는데도 수평 방향으로 미끄러지는 정도를 계산
def feet_slip(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.square(foot_vel_xy), dim=-1)

    return torch.sum(slip * feet_contact.float(), dim=1)

# 양 발의 roll이 0에서 얼마나 벗어났는지
def feet_roll_l2(env, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    roll, _ = _roll_pitch_from_quat_wxyz(asset.data.body_quat_w[:, asset_cfg.body_ids])
    return torch.sum(torch.square(_wrap_to_pi(roll)), dim=1)

# 양 발의 pitch가 0에서 얼마나 벗어났는지
def feet_pitch_l2(env, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    _, pitch = _roll_pitch_from_quat_wxyz(asset.data.body_quat_w[:, asset_cfg.body_ids])
    return torch.sum(torch.square(_wrap_to_pi(pitch)), dim=1)

# 양 발의 yaw 각도 차이
def feet_yaw_diff_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    commanded_diff = command[:, 5] - command[:, 4]
    actual_diff = feet_yaw_rel[:, 1] - feet_yaw_rel[:, 0]

    return torch.square(_wrap_to_pi(actual_diff - commanded_diff))

# 왼발과 오른발 yaw의 평균 방향이 cmd에서 원하는 평균 방향과 얼마나 다른지
def feet_yaw_mean_l2(env, command_name: str, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    feet_yaw_rel = _feet_raw_rel(asset, asset_cfg.body_ids)
    commanded_mean = 0.5 * (command[:, 5] + command[:, 4])
    actual_mean = torch.mean(feet_yaw_rel, dim=1)

    return torch.square(_wrap_to_pi(actual_mean - commanded_mean))

# 관절이 자기 최대 허용 토크 대비 얼마나 세게 쓰이고 있는지
def torque_tiredness(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]

    if hasattr(asset.data, "soft_joint_effort_limits"):
        limits = asset.data.soft_joint_effort_limits[:, asset_cfg.joint_ids]
    elif hasattr(asset.data, "joint_effort_limits"):
        limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    else:
        raise AttributeError("No joint effort limit field found for torque_tiredness.")

    torque_ratio = asset.data.applied_torque[:, asset_cfg.joint_ids] / limits.clamp(min=1.0e-6)
    return torch.sum(torch.square(torque_ratio).clip(max=1.0), dim=1)

# x 명령만 줬는데 y가 움직일 때 패널티
def pure_x_lateral_drift_l2(env, command_name: str, command_deadband: float = 0.05, zero_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    mask = ((torch.abs(command[:, 0]) > command_deadband)
            & (torch.abs(command[:, 1]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband))

    return torch.square(asset.data.root_lin_vel_b[:, 1]) * mask.float()

# x 명령만 줬는데 yaw가 움직일 때 패널티
def pure_x_yaw_drift_l2(env, command_name: str, command_deadband: float = 0.05, zero_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    mask = ((torch.abs(command[:, 0]) > command_deadband)
            & (torch.abs(command[:, 1]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband))

    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

# y 명령만 줬는데 x가 움직일 때 패널티
def pure_y_forward_drift_l2(env, command_name: str, command_deadband: float = 0.05, zero_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    mask = ((torch.abs(command[:, 1]) > command_deadband)
            & (torch.abs(command[:, 0]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband))

    return torch.square(asset.data.root_lin_vel_b[:, 0]) * mask.float()

# y 명령만 줬는데 yaw가 움직일 때 패널티
def pure_y_yaw_drift_l2(env, command_name: str, command_deadband: float = 0.05, zero_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    mask = ((torch.abs(command[:, 1]) > command_deadband)
            & (torch.abs(command[:, 0]) < zero_deadband)
            & (torch.abs(command[:, 2]) < zero_deadband))

    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

# 회전 명령이 0인데 로봇이 실제로 회전하는 경우 그 yaw 각속도를 패널티 
def zero_yaw_command_ang_vel_l2(env, command_name: str, zero_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    mask = torch.abs(command[:, 2]) < zero_deadband
    return torch.square(asset.data.root_ang_vel_b[:, 2]) * mask.float()

# xy 명령 줬을 때 실제 그 벡터로 잘 가는가
def xy_perpendicular_velocity_l2(env, command_name: str, command_deadband: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    cmd_xy = command[:, :2]
    vel_xy = asset.data.root_lin_vel_b[:, :2]

    cmd_norm = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
    active = cmd_norm.squeeze(-1) > command_deadband

    cmd_dir = cmd_xy / cmd_norm.clamp(min=1.0e-6) # 단위 벡터로 변경
    vel_parallel = torch.sum(vel_xy * cmd_dir, dim=1, keepdim=True) * cmd_dir # 실제 속도 중에서 명령 방향과 평행한 성분 계산
    vel_perp = vel_xy - vel_parallel # 로봇이 옆으로 새는 속도

    return torch.sum(torch.square(vel_perp), dim=1) * active.float()


def joint_roll_yaw_symmetry_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    pairs = [
        ("Left_Hip_Roll", "Right_Hip_Roll"),
        ("Left_Hip_Yaw", "Right_Hip_Yaw"),
        ("Left_Ankle_Roll", "Right_Ankle_Roll")
    ]

    penalty = torch.zeros(env.num_envs, device=env.device)

    for left_name, right_name in pairs:
        left_id = asset.joint_names.index(left_name)
        right_id = asset.joint_names.index(right_name)

        q_left = joint_pos[:, left_id]
        q_right = joint_pos[:, right_id]

        penalty += torch.square(q_left + q_right)

        return penalty