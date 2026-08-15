"""K1 velocity-locomotion + kick conditional student environment."""

from __future__ import annotations

import os

from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.kick.robots.k1.kick_001.env_cfg import (
    K1KickEnvCfg, K1KickPlayEnvCfg, ObservationsCfg as KickObservationsCfg,
    RewardsCfg as KickRewardsCfg,
)
from booster_train.tasks.manager_based.locomotion_kick import mdp


def _required_teacher_path(variable_name: str) -> str:
    path = os.environ.get(variable_name)
    if not path:
        raise RuntimeError(
            f"{variable_name} is required. Set it to the absolute path of the teacher TorchScript JIT."
        )
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{variable_name} does not exist: {path}")
    return path


@configclass
class CommandsCfg:
    walk_kick = mdp.WalkKickCommandCfg(resampling_time_range=(1.0e9, 1.0e9), debug_vis=False)


@configclass
class ObservationsCfg(KickObservationsCfg):
    @configclass
    class PolicyCfg(KickObservationsCfg.PolicyCfg):
        locomotion_command = ObsTerm(func=mdp.locomotion_command)
        gait_phase = ObsTerm(func=mdp.gait_phase)
        skill_mode = ObsTerm(func=mdp.skill_mode)

    @configclass
    class CriticCfg(KickObservationsCfg.CriticCfg):
        locomotion_command = ObsTerm(func=mdp.locomotion_command)
        gait_phase = ObsTerm(func=mdp.gait_phase)
        skill_mode = ObsTerm(func=mdp.skill_mode)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(KickRewardsCfg):
    # Disable kick-only dense terms that would punish the approach phase.
    stationary_xy = None
    stationary_yaw = None
    ball_velocity_target = None
    ball_target_accuracy = None
    ball_lateral_velocity = None
    ball_overspeed = None
    kicking_foot_approach = None
    kicking_foot_progress = None
    preferred_foot_kick = None
    wrong_foot_proximity = None
    inside_foot_kick = None
    ball_acceleration = None
    waiting = None
    no_kick_failure = None
    pre_kick_stability = None
    post_kick_recovery = None
    walk_ready_after_kick = None

    teacher_tracking = RewTerm(
        func=mdp.teacher_joint_target_tracking,
        weight=4.0,
        params={
            "walk_model_path": _required_teacher_path("VELOCITY_TEACHER_JIT"),
            "kick_model_path": _required_teacher_path("KICK_TEACHER_JIT"),
            "std": 0.20,
        },
    )
    walk_velocity = RewTerm(func=mdp.walk_velocity_tracking, weight=2.0, params={"std": 0.25})
    kick_velocity = RewTerm(func=mdp.kick_ball_velocity, weight=14.0)
    kick_accuracy = RewTerm(func=mdp.kick_ball_accuracy, weight=10.0)
    kick_inside = RewTerm(func=mdp.kick_inside_quality, weight=12.0)
    recovery = RewTerm(func=mdp.recovery_stability, weight=6.0)


@configclass
class VelocityKickEnvCfg(K1KickEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Enough time for up to three walk -> kick -> recovery cycles.
        self.episode_length_s = 36.0
        # Initial curriculum stage. Later cycles are re-seeded by
        # locomotion_kick.mdp with progressively wider distributions.
        self.events.reset_ball.params["x_range"] = (1.2, 1.8)
        self.events.reset_ball.params["y_range"] = (-0.15, 0.15)
        self.events.reset_ball.params["angle_range_deg"] = (-30.0, 30.0)
        self.events.reset_target.params["distance_range"] = (3.0, 5.0)
        self.events.reset_target.params["angle_range_deg"] = (-15.0, 15.0)
        self.terminations.kick_success = None
        self.terminations.ball_not_kicked = None


@configclass
class VelocityKickPlayEnvCfg(VelocityKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.push_robot = None
