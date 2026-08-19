"""K1 autonomous approach, adjust, kick, and recovery environment."""

import os

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
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
ADJUST_TEACHER_PATH = os.path.abspath(
    os.path.join(ADJUST_KICK_ROOT, "models", "adjust_teacher.pt")
)

KICK_TEACHER_PATH = os.path.abspath(
    os.path.join(ADJUST_KICK_ROOT, "models", "kick_teacher.pt")
)


@configclass
class ActionsCfg:
    joint_pos = task_mdp.FrozenAdjustKickTransitionActionCfg(
        asset_name="robot",
        joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
        scale=1.0,
        use_default_offset=True,
        adjust_teacher_path=ADJUST_TEACHER_PATH,
        kick_teacher_path=KICK_TEACHER_PATH,
        target_ball_distance=0.30,
        minimum_ball_distance=0.20,
        maximum_ball_distance=0.40,
        # Kept for compatibility; transition readiness is now a kickable
        # sector, not an exact 0.30 m point.
        ready_position_tolerance=0.08,
        ready_heading_tolerance_deg=25.0,
        ready_ball_speed_tolerance=0.08,
        maximum_ball_displacement=0.04,
        ready_lateral_tolerance=0.18,
        minimum_ball_forward_distance=0.10,
        transition_duration_s=0.20,
        rollin_stage_steps=(120_000, 300_000, 600_000),
        teacher_control_blend=(1.0, 0.99, 0.90, 0.0),
    )


@configclass
class RewardsCfg(KickRewardsCfg):
    # Walking is required during approach, so kick-only standing terms are off.
    stationary_xy = None
    stationary_yaw = None
    pre_kick_stability = None
    waiting = None
    no_kick_failure = None

    # Lower the trunk by about 3 cm for quick contact-free side adjustment and
    # the kick itself, then return upright for walk-ready recovery.
    base_height = RewTerm(
        func=mdp.phase_base_height_l2,
        weight=-200.0,
        params={"pre_kick_target": 0.52, "post_kick_target": 0.55},
    )

    # Kick terms are gated by geometric readiness. They cannot reward an early
    # touch while the robot is still approaching or rotating around the ball.
    ball_velocity_target = RewTerm(func=mdp.gated_kick_velocity, weight=10.0)
    kick_speed_target = RewTerm(
        func=mdp.direction_gated_kick_speed,
        weight=20.0,
        params={"target_speed": 3.0, "min_direction_score": 0.98},
    )
    # Do not reward eventual arrival at the distant target point. Speed and
    # launch direction are evaluated immediately after contact instead.
    ball_overspeed = None
    ball_target_accuracy = None
    kick_direction_accuracy = RewTerm(
        func=mdp.kick_direction_accuracy, weight=35.0, params={"minimum_speed": 0.05}
    )
    ball_lateral_velocity = RewTerm(func=mdp.gated_kick_lateral_velocity, weight=-20.0)
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
    # The ball already starts in the kick-distance band; do not train a
    # separate forward-approach behavior. Position progress below is retained
    # because it also teaches movement around the ball to the aligned pose.
    fast_approach_velocity = None
    adjust_pose_accuracy = RewTerm(
        func=mdp.adjust_pose_accuracy,
        weight=4.0,
        params={"position_std": 0.20, "heading_std_deg": 15.0},
    )
    # Every pre-kick control step costs reward so the shortest valid approach
    # wins once the target is reachable from any side.
    approach_time = RewTerm(func=mdp.approach_time, weight=-2.0)

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

    # The two supplied experts are frozen.  This single loss preserves both
    # normalized action outputs across adjust, handoff, and kick.
    composite_teacher_tracking = RewTerm(
        func=mdp.composite_teacher_tracking,
        weight=12.0,
        params={"std": 0.10, "transition_multiplier": 1.0},
    )
    ball_distance_band = RewTerm(
        func=mdp.ball_distance_band_penalty,
        weight=-8.0,
        params={
            "minimum_distance": 0.20,
            "maximum_distance": 0.40,
            "scale": 0.10,
            "near_gate_distance": 0.60,
        },
    )

    # Strong post-kick landing and stand-still objective.
    post_kick_recovery = RewTerm(
        func=kick_mdp.post_kick_recovery,
        weight=4.0,
        # A valid foot-contact event plus this launch speed is enough to enter
        # recovery; waiting for the ball to reach the distant target is not.
        params={"kick_speed_threshold": 0.20},
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
    # A combined episode is one approach -> one kick -> recovery sequence.
    kick_complete = DoneTerm(
        func=mdp.kick_recovery_complete,
        params={"recovery_time": 0.8},
    )
    kick_success = DoneTerm(
        func=mdp.kick_quality_success,
        params={
            "minimum_speed": 1.5,
            # About +/-11.5 degrees from the desired launch direction.
            "min_direction_score": 0.98,
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
            # The transition trainer counts iterations as common steps. This
            # mirrors adjust's five bands and reaches full 360 degrees in a
            # 20k-iteration run.
            "stage_steps": (4_000, 8_000, 12_000, 16_000),
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
        # Play samples the final full-target-direction stage so the policy is
        # tested over the same 360-degree target task as adjust.
        self.events.reset_scenario.params["stage_steps"] = (
            0,
            0,
            0,
            0,
        )
        self.events.reset_scenario.params["visualize_target"] = True
        # Play defaults to student-only. For an intermediate diagnostic, set
        # ADJUST_KICK_PLAY_TEACHER_BLEND to a value in [0, 1].
        teacher_blend = float(os.environ.get("ADJUST_KICK_PLAY_TEACHER_BLEND", "0.0"))
        if not 0.0 <= teacher_blend <= 1.0:
            raise ValueError("ADJUST_KICK_PLAY_TEACHER_BLEND must be in [0, 1]")
        self.actions.joint_pos.teacher_control_blend = (
            teacher_blend,
            teacher_blend,
            teacher_blend,
            teacher_blend,
        )
        self.actions.joint_pos.debug_transition = True


@configclass
class UnifiedRewardsCfg(RewardsCfg):
    """Rewards for a single actor learned from frozen teachers at train time."""

    composite_teacher_tracking = RewTerm(
        func=mdp.composite_teacher_tracking,
        weight=10.0,
        params={"std": 0.10, "transition_multiplier": 0.35},
    )
    unified_applied_action_rate = RewTerm(
        func=mdp.transition_applied_action_rate_l2,
        weight=-0.5,
    )
    unified_transition_stability = RewTerm(
        func=mdp.transition_stability,
        weight=8.0,
        params={"tilt_scale": 0.08, "angular_velocity_scale": 1.0},
    )


@configclass
class K1AdjustKickUnifiedEnvCfg(K1AdjustKickEnvCfg):
    """Train one full 50-to-12 actor; teachers exist only inside the trainer."""

    rewards: UnifiedRewardsCfg = UnifiedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.control_mode = "student"
        # Teacher roll-in is a training curriculum. It is forced to zero in
        # the Play config, so the exported actor is evaluated by itself.
        self.actions.joint_pos.rollin_stage_steps = (24_000, 72_000, 144_000)
        self.actions.joint_pos.teacher_control_blend = (1.0, 0.98, 0.70, 0.0)
        self.observations.policy.transition_progress = ObsTerm(
            func=kick_mdp.transition_phase_progress,
            clip=(-1.0, 2.0),
        )


@configclass
class K1AdjustKickUnifiedPlayEnvCfg(K1AdjustKickUnifiedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.reset_scenario.params["stage_steps"] = (0, 0, 0, 0)
        self.events.reset_scenario.params["visualize_target"] = True
        teacher_blend = float(os.environ.get("ADJUST_KICK_PLAY_TEACHER_BLEND", "0.0"))
        if not 0.0 <= teacher_blend <= 1.0:
            raise ValueError("ADJUST_KICK_PLAY_TEACHER_BLEND must be in [0, 1]")
        self.actions.joint_pos.teacher_control_blend = (
            teacher_blend,
            teacher_blend,
            teacher_blend,
            teacher_blend,
        )
        self.actions.joint_pos.debug_transition = True


@configclass
class TransitionRewardsCfg(RewardsCfg):
    """Rewards for learning only the teacher-to-teacher handoff residual."""

    action_rate = None
    composite_teacher_tracking = None
    transition_residual = RewTerm(func=mdp.transition_residual_l2, weight=-2.0)
    transition_action_rate = RewTerm(
        func=mdp.transition_applied_action_rate_l2,
        weight=-0.5,
    )
    transition_stability = RewTerm(
        func=mdp.transition_stability,
        weight=8.0,
        params={"tilt_scale": 0.08, "angular_velocity_scale": 1.0},
    )


@configclass
class K1AdjustKickTransitionEnvCfg(K1AdjustKickEnvCfg):
    """Freeze adjust/kick and train a bounded residual only in transition."""

    rewards: TransitionRewardsCfg = TransitionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.control_mode = "transition"
        # Start with a small correction envelope: the frozen teachers must
        # dominate until the transition actor has learned a safe residual.
        self.actions.joint_pos.transition_residual_scale = 0.04
        self.observations.policy.transition_progress = ObsTerm(
            func=kick_mdp.transition_phase_progress,
            clip=(-1.0, 2.0),
        )


@configclass
class K1AdjustKickTransitionPlayEnvCfg(K1AdjustKickTransitionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.reset_scenario.params["stage_steps"] = (0, 0, 0, 0)
        self.events.reset_scenario.params["visualize_target"] = True
        self.actions.joint_pos.debug_transition = True
