# locomotion/mdp/commands.py
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# 1) Null command (옵션: standing만 시키고 싶을 때)
class NullCommand(CommandTerm):
    """Always returns zeros."""
    def __init__(self, cfg: CommandTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._cmd = torch.zeros((self.num_envs, 3), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        self._cmd[:] = 0.0


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""
    class_type: type = NullCommand

    def __post_init__(self):
        # resampling 안 하도록 무한대로
        self.resampling_time_range = (math.inf, math.inf)


# 2) Velocity command (Uniform / Normal)
class UniformVelocityCommand(CommandTerm):
    """Uniformly samples SE(2) velocity command in base frame: (vx, vy, wz).

    - heading_command=True면 wz를 '목표 heading - 현재 heading' 오차 기반 P제어로 생성합니다.
    - standing 확률(rel_standing_envs)에 따라 명령을 0으로 만드는 env도 섞을 수 있습니다.
    """

    cfg: "UniformVelocityCommandCfg"

    def __init__(self, cfg: "UniformVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # heading command 설정 검증
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "heading_command=True 인데 ranges.heading가 None입니다."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                "ranges.heading는 설정되어 있는데 heading_command=False 입니다. heading_command=True 권장."
            )

        # 로봇 아티큘레이션 핸들
        self.robot: Articulation = env.scene[cfg.asset_name]

        # command 버퍼 (vx, vy, wz)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

        # heading 관련 버퍼
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # standing 관련 버퍼
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics (선택)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dim: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 3) = [vx, vy, wz] in base frame."""
        return self.vel_command_b

    # --- 선택: metric 업데이트 (학습 로그용)
    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """resampling_time이 지나면 호출되어 새 명령 샘플링."""
        r = torch.empty(len(env_ids), device=self.device)

        # vx, vy, wz uniform 샘플링
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # heading 환경 여부 및 target heading 샘플링
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # standing env 샘플링
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """resample된 raw command를 최종 command로 후처리."""
        # heading 기반 wz 생성
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        # standing env는 command=0
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    # --- debug vis (옵션)
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_scale, vel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_scale, vel_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_quat, vel_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor):
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Normal distribution velocity command."""
    cfg: "NormalVelocityCommandCfg"

    def __init__(self, cfg: "NormalVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)

        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0


# 3) Command cfg들 (env_cfg에서 instantiate)
@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING

    heading_command: bool = False
    heading_control_stiffness: float = 1.0

    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 1.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = NormalVelocityCommand
    heading_command: bool = False

    @configclass
    class Ranges:
        mean_vel: tuple[float, float, float] = MISSING
        std_vel: tuple[float, float, float] = MISSING
        zero_prob: tuple[float, float, float] = MISSING

    ranges: Ranges = MISSING


# 4) Observation에서 command를 꺼내쓰기 위한 helper
def generated_commands(env, command_name: str) -> torch.Tensor:
    """ObservationTerm에서 command를 읽기 위한 함수."""
    return env.command_manager.get_command(command_name)
