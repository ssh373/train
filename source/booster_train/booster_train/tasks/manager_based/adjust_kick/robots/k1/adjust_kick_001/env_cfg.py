"""K1 autonomous approach, adjust, kick, and recovery environment."""

import os

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.adjust_kick import mdp
from booster_train.tasks.manager_based.adjust_kick import standalone_mdp as kick_mdp
from booster_train.tasks.manager_based.adjust_kick import task_mdp
from booster_train.tasks.manager_based.adjust_kick.base_env_cfg import (
    EventsCfg as KickEventsCfg,
    K1KickEnvCfg,
    RewardsCfg as KickRewardsCfg,
    TerminationsCfg as KickTerminationsCfg,
)


FEET = ["left_foot_link", "right_foot_link"]
ADJUST_KICK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
WALK_TEACHER_PATH = os.path.abspath(
    os.path.join(ADJUST_KICK_ROOT, "models", "walk_teacher.pt")
)

KICK_TEACHER_PATH = os.path.abspath(
    os.path.join(ADJUST_KICK_ROOT, "models", "kick_teacher.pt")
)


@configclass
class ActionsCfg:
    joint_pos = task_mdp.FrozenWalkAdjustActionCfg(
        asset_name="robot",
        joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
        scale=1.0,
        use_default_offset=True,
        teacher_path=WALK_TEACHER_PATH,
        kick_teacher_path=KICK_TEACHER_PATH,
        kick_teacher_blend_steps=(40_000, 80_000, 140_000),
        kick_teacher_blend=(1.0, 0.67, 0.33, 0.0),
        handoff_stage_steps=(100_000, 250_000, 500_000),
        handoff_distance_ranges=(
            (0.55, 0.65),
            (0.50, 0.70),
            (0.45, 0.75),
            (0.45, 0.80),
        ),
        slowdown_distance=0.90,
        handoff_heading_tolerance_deg=20.0,
        transition_duration_s=0.50,
        walk_full_speed_heading_deg=15.0,
        walk_stop_translation_heading_deg=45.0,
    )


@configclass
class RewardsCfg(KickRewardsCfg):
    # Walking is required during approach, so kick-only standing terms are off.
    stationary_xy = None
    stationary_yaw = None
    pre_kick_stability = None
    waiting = None
    no_kick_failure = None

    # Kick terms are gated by geometric readiness. They cannot reward an early
    # touch while the robot is still approaching or rotating around the ball.
    ball_velocity_target = RewTerm(func=mdp.gated_kick_velocity, weight=10.0)
    ball_target_accuracy = RewTerm(func=mdp.gated_kick_accuracy, weight=20.0, params={"std": 0.25})
    kick_direction_accuracy = RewTerm(
        func=mdp.kick_direction_accuracy, weight=15.0, params={"minimum_speed": 0.10}
    )
    ball_lateral_velocity = RewTerm(func=mdp.gated_kick_lateral_velocity, weight=-8.0)
    kicking_foot_approach = RewTerm(
        func=mdp.gated_kicking_foot_approach,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "proximity_std": 0.25,
            "stationary_speed": 0.1,
            "velocity_weight": 0.3,
            "center_deadband": 0.03,
        },
    )
    kicking_foot_progress = RewTerm(
        func=mdp.gated_kicking_foot_progress,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "center_deadband": 0.03,
            "max_progress_per_step": 0.04,
        },
    )
    preferred_foot_kick = RewTerm(
        func=mdp.gated_foot_kick_event,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "speed_increase_threshold": 0.08,
            "max_contact_distance": 0.25,
        },
    )
    wrong_foot_proximity = None
    inside_foot_kick = None
    ball_acceleration = None

    # Fast approach and fast alignment to the pose behind the ball.
    adjust_position_progress = RewTerm(func=mdp.adjust_position_progress, weight=8.0)
    adjust_heading_progress = RewTerm(func=mdp.adjust_heading_progress, weight=5.0)
    face_ball_alignment = RewTerm(
        func=mdp.face_ball_alignment,
        weight=8.0,
        params={"heading_std_deg": 15.0},
    )
    face_ball_violation = RewTerm(
        func=mdp.face_ball_violation,
        weight=-12.0,
        params={"tolerance_deg": 20.0, "full_penalty_deg": 45.0},
    )
    fast_approach_velocity = RewTerm(
        func=mdp.fast_approach_velocity, weight=3.0, params={"target_speed": 1.2}
    )
    adjust_pose_accuracy = RewTerm(
        func=mdp.adjust_pose_accuracy,
        weight=4.0,
        params={"position_std": 0.20, "heading_std_deg": 15.0},
    )
    approach_time = RewTerm(func=mdp.approach_time, weight=-0.5)

    # During adjustment the ball must remain untouched.
    early_ball_motion = RewTerm(
        func=mdp.early_ball_motion,
        weight=-80.0,
        params={"speed_tolerance": 0.05, "displacement_tolerance": 0.03},
    )
    early_foot_ball_proximity = RewTerm(
        func=mdp.early_foot_ball_proximity,
        weight=-20.0,
        params={
            "safe_distance": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
        },
    )

    # Preserve the validated walk when a 54-observation walk teacher is supplied.
    walk_teacher_tracking = RewTerm(
        func=mdp.walk_teacher_tracking,
        weight=3.0,
        params={"teacher_env_var": "ADJUST_KICK_WALK_TEACHER_JIT", "std": 0.20},
    )

    # Distill the exact deploy kick only after precise behind-ball alignment.
    # Roll-in blending reaches zero, but this reward keeps the original shape.
    kick_teacher_tracking = RewTerm(
        func=mdp.kick_teacher_tracking,
        weight=4.0,
        params={"std": 0.20},
    )

    # Strong post-kick landing and stand-still objective.
    post_kick_recovery = RewTerm(
        func=kick_mdp.post_kick_recovery,
        weight=4.0,
        params={"kick_speed_threshold": 0.2},
    )
    walk_ready_after_kick = RewTerm(
        func=kick_mdp.walk_ready_after_kick,
        weight=2.0,
        params={"return_delay": 0.5},
    )
    post_kick_feet_grounded = RewTerm(
        func=kick_mdp.post_kick_feet_grounded,
        weight=8.0,
        params={
            "return_delay": 0.25,
            "height_std": 0.035,
            "velocity_scale": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
        },
    )


@configclass
class TerminationsCfg(KickTerminationsCfg):
    kick_success = DoneTerm(
        func=mdp.adjusted_ball_success,
        params={
            "target_xy": (4.0, 0.0),
            "target_radius": 0.15,
            "min_direction_score": 0.98,
            "max_speed": 2.5,
            "recovery_time": 0.8,
            "max_base_speed": 0.35,
            "max_tilt": 0.2,
            "max_mean_joint_deviation": 0.35,
        },
    )
    ball_too_far = DoneTerm(func=kick_mdp.ball_too_far, params={"max_distance": 13.0})
    ball_not_kicked = DoneTerm(
        func=mdp.adjusted_ball_not_kicked,
        params={
            "time_limit": 2.5,
            "movement_speed": 0.12,
            "min_direction_cos": 0.9396926208,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    lost_ball_heading = DoneTerm(
        func=mdp.lost_ball_heading,
        params={"max_heading_error_deg": 45.0},
    )


@configclass
class EventsCfg(KickEventsCfg):
    reset_ball = None
    reset_target = None
    reset_scenario = EventTerm(
        func=mdp.reset_adjust_kick_scenario,
        mode="reset",
        params={
            "stage_steps": (100_000, 250_000, 500_000),
            "visualize_target": False,
            "target_radius": 0.15,
        },
    )


@configclass
class K1AdjustKickEnvCfg(K1KickEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.env_spacing = 14.0
        # Preserve the original 49-value kick actor contract while allowing
        # far-away ball and target coordinates to remain unsaturated.
        self.observations.policy.ball_position.clip = (-6.0, 6.0)
        self.observations.policy.target_position.clip = (-3.0, 3.0)
        # Keep the first curriculum deterministic enough to learn the sequence;
        # startup physics randomization remains active for sim-to-real robustness.
        self.events.push_robot = None


@configclass
class K1AdjustKickPlayEnvCfg(K1AdjustKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.reset_scenario.params["stage_steps"] = (0, 0, 0)
        self.events.reset_scenario.params["visualize_target"] = True
        self.actions.joint_pos.handoff_stage_steps = (0, 0, 0)
