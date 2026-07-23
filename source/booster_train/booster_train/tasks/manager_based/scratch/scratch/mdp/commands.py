from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

## 추가 scratch 용
class INHAWalkCommand(CommandTerm):
    cfg: INHAWalkCommandCfg

    def __init__(self, cfg: INHAWalkCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 10, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command
    @property
    def gait_frequency(self) -> torch.Tensor:
        return self._command[:, 3]
    
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(self.cfg.play_velocity_command, device=self.device)
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
            return
        
        vr = self.cfg.velocity_ranges
        self._command[env_ids, 0] = sample_uniform(*vr.lin_vel_x, (len(env_ids), ), device=self.device)
        self._command[env_ids, 1] = sample_uniform(*vr.lin_vel_y, (len(env_ids),), device=self.device)
        self._command[env_ids, 2] = sample_uniform(*vr.ang_vel_yaw, (len(env_ids),), device=self.device)

        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            self._command[env_ids, 3] = sample_uniform(*pr.gait_frequency, (len(env_ids),), device=self.device)
            self._command[env_ids, 4] = sample_uniform(*pr.foot_yaw_L, (len(env_ids),), device=self.device)
            self._command[env_ids, 5] = sample_uniform(*pr.foot_yaw_R, (len(env_ids),), device=self.device)
            self._command[env_ids, 6] = sample_uniform(*pr.body_pitch_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 7] = sample_uniform(*pr.body_roll_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 8] = sample_uniform(*pr.feet_offset_x_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 9] = sample_uniform(*pr.feet_offset_y_target, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)

        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_still]]
            self._command[still_ids, 0:4] = 0.0
        
    def _update_command(self): # 보행 주기에서 현재 몇 %까지 진행됐는지를 매 스텝마다 업데이트 
        self.gait_process[:] = torch.fmod(
            self.gait_process + self._env.step_dt * self.gait_frequency, 
            1.0,
        )
    
    def _update_metrics(self):
        pass
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
    def _debug_vis_callback(self, event):
        pass

@configclass
class PureCommandModeProbabilities:
    stand: float = 0.15
    x_only: float = 0.35
    y_only: float = 0.30
    yaw_only: float = 0.20

class PureWalkCommand(INHAWalkCommand):
    cfg: PureWalkCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return 
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        
        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(self.cfg.play_velocity_command, device=self.device)
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
            return 
        
        mode_ids = self._sample_pure_modes(len(env_ids))
        self._command[env_ids, 0:3] = 0.0

        vr = self.cfg.velocity_ranges
        x_mask = mode_ids == 1
        y_mask = mode_ids == 2
        yaw_mask = mode_ids == 3

        self._command[env_ids[x_mask], 0] = sample_uniform(*vr.lin_vel_x, (int(x_mask.sum()),), device=self.device)
        self._command[env_ids[y_mask], 1] = sample_uniform(*vr.lin_vel_y, (int(y_mask.sum()),), device=self.device)
        self._command[env_ids[yaw_mask], 2] = sample_uniform(*vr.ang_vel_yaw, (int(yaw_mask.sum()),), device=self.device)

        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            self._command[env_ids, 3] = sample_uniform(*pr.gait_frequency, (len(env_ids),), device=self.device)
            self._command[env_ids, 4] = sample_uniform(*pr.foot_yaw_L, (len(env_ids),), device=self.device)
            self._command[env_ids, 5] = sample_uniform(*pr.foot_yaw_R, (len(env_ids),), device=self.device)
            self._command[env_ids, 6] = sample_uniform(*pr.body_pitch_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 7] = sample_uniform(*pr.body_roll_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 8] = sample_uniform(*pr.feet_offset_x_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 9] = sample_uniform(*pr.feet_offset_y_target, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
        
        stand_ids = env_ids[mode_ids == 0]
        self._command[stand_ids, 0:4] = 0.0

    def _sample_pure_modes(self, n: int) -> torch.Tensor:
        p = self.cfg.pure_command_mode_probabilities
        probs = torch.tensor([p.stand, p.x_only, p.y_only, p.yaw_only], device=self.device, dtype=torch.float32)
        probs = probs / probs.sum()
        return torch.multinomial(probs, n, replacement=True)
    
class StagedWalkCommand(PureWalkCommand):
    cfg: StagedWalkCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        if self._use_pure_command_stage():
            PureWalkCommand._resample_command(self, env_ids)
        else:
            INHAWalkCommand._resample_command(self, env_ids)
    
    def _use_pure_command_stage(self) -> bool:
        if self.cfg.pure_command_iterations <= 0:
            return False
        current_iteration = int(self._env.common_step_counter) // max(self.cfg.num_steps_per_iteration, 1)
        return current_iteration < self.cfg.pure_command_iterations
    
@configclass
class INHAWalkCommandCfg(CommandTermCfg):
    class_type: type = INHAWalkCommand

    @configclass
    class VelocityRanges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: tuple[float, float] = (-1.6, 1.6)

    @configclass
    class ParameterRanges:
        gait_frequency: tuple[float, float] = (1.5, 1.5)
        foot_yaw_L: tuple[float, float] = (-0.7, 0.7)
        foot_yaw_R: tuple[float, float] = (-0.7, 0.7)
        body_pitch_target: tuple[float, float] = (-0.1, 0.3)
        body_roll_target: tuple[float, float] = (-0.1, 0.1)
        feet_offset_x_target: tuple[float, float] = (-0.15, 0.15)
        feet_offset_y_target: tuple[float, float] = (-0.08, 0.15)

    asset_name: str = MISSING
    velocity_ranges: VelocityRanges = VelocityRanges()
    parameter_ranges: ParameterRanges = ParameterRanges()

    still_proportion: float = 0.1
    sample_internal_parameters: bool = False

    play: bool = False
    play_velocity_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_internal_parameters: tuple[float, float, float,float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

@configclass
class PureWalkCommandCfg(INHAWalkCommandCfg):
    class_type: type = PureWalkCommand
    pure_command_mode_probabilities: PureCommandModeProbabilities = PureCommandModeProbabilities()

@configclass
class StagedWalkCommandCfg(PureWalkCommandCfg):
    class_type: type = StagedWalkCommand
    pure_command_iterations: int = 10000
    num_steps_per_iteration: int = 24
