"""K1 contact-free target-alignment environment.

This task learns only the approach/alignment phase.  It terminates when the
robot reaches the ball-relative alignment point and holds it briefly without
causing meaningful ball motion.
"""

from __future__ import annotations

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.assets.robots.booster import K1_ACTION_SCALE
from booster_train.tasks.manager_based.adjust import mdp
from booster_train.tasks.manager_based.kick.robots.k1.kick_001.env_cfg import (
    ActionsCfg as KickActionsCfg,
    EventsCfg as KickEventsCfg,
    KickSceneCfg,
)


FEET = ["left_foot_link", "right_foot_link"]
LEGS = [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]


@configclass
class ObservationsCfg:
    """Superset observation contract intended for a future adjust->kick policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        ball_position = ObsTerm(
            func=mdp.ball_position_camera_b,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "base_noise_std": 0.03,
                "distance_noise_ratio": 0.02,
                "dropout_rate_per_s": 0.50,
                "dropout_duration_range": (0.08, 0.30),
            },
            clip=(-3.0, 3.0),
        )
        target_direction = ObsTerm(func=mdp.target_direction_b, clip=(-1.0, 1.0))
        ball_visible = ObsTerm(func=mdp.ball_visible)
        ball_time_since_seen = ObsTerm(func=mdp.ball_time_since_seen)
        ball_confidence = ObsTerm(func=mdp.ball_confidence)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        policy_gravity = ObsTerm(func=mdp.projected_gravity)
        base_linear_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        ball_position = ObsTerm(func=mdp.ball_pos_w, params={"ball_cfg": SceneEntityCfg("ball")})
        ball_velocity = ObsTerm(
            func=mdp.ball_velocity_b,
            params={"ball_cfg": SceneEntityCfg("ball"), "robot_cfg": SceneEntityCfg("robot")},
        )
        ball_displacement = ObsTerm(func=mdp.ball_displacement_obs)
        target_direction = ObsTerm(func=mdp.target_direction_b)
        alignment_error = ObsTerm(func=mdp.alignment_line_band_error_obs)
        feet_position = ObsTerm(
            func=mdp.feet_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    survival = RewTerm(func=mdp.survival, weight=0.05)

    # GoTo-style unified pose objective: exact position and ball-facing yaw.
    alignment_pose = RewTerm(
        func=mdp.alignment_pose_reward,
        weight=6.0,
        params={"heading_radius": 0.28, "gain": 8.0},
    )
    alignment_ready = RewTerm(
        func=mdp.alignment_ready_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "feet_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "position_tolerance": 0.06,
            "heading_tolerance_deg": 15.0,
            "linear_speed_tolerance": 0.10,
            "yaw_speed_tolerance": 0.10,
            "maximum_tilt_deg": 5.0,
            "contact_threshold": 1.0,
            "feet_midpoint_tolerance": 0.10,
            "feet_longitudinal_spread_tolerance": 0.12,
            "minimum_lateral_spacing": 0.14,
            "maximum_lateral_spacing": 0.30,
            "ball_speed_tolerance": 0.05,
            "ball_displacement_tolerance": 0.02,
        },
    )

    # Adjust-specific route constraints: circle around the ball rather than
    # taking the Euclidean shortcut through it.
    orbit_tangent = RewTerm(
        func=mdp.orbit_tangent_velocity_reward,
        weight=2.0,
        params={"speed_scale": 0.25, "angle_deadband": 0.12},
    )
    orbit_radius = RewTerm(
        func=mdp.orbit_radius_reward,
        weight=1.5,
        params={"target_radius": 0.28, "radius_std": 0.08},
    )
    approach_time = RewTerm(func=mdp.alignment_time_penalty, weight=-0.15)
    arrival_stillness = RewTerm(
        func=mdp.alignment_stillness_penalty,
        weight=-0.75,
        params={"alignment_scale": 0.10, "yaw_weight": 1.0},
    )

    # Retain only a light direct anti-drag guard.  Normal walking should now
    # emerge from the GoTo posture/contact structure rather than gait timing.
    low_foot_drag = RewTerm(
        func=mdp.low_foot_drag_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "drag_height": 0.045,
            "speed_deadband": 0.03,
            "speed_scale": 0.15,
            "alignment_gate_error": 0.08,
            "maximum_penalty": 4.0,
        },
    )
    support_transfer = RewTerm(
        func=mdp.support_transfer_reward,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "minimum_air_time": 0.10,
            "maximum_air_time": 0.45,
            "alignment_gate_error": 0.08,
            "midpoint_tolerance": 0.10,
            "longitudinal_spread_tolerance": 0.12,
            "minimum_lateral_spacing": 0.14,
            "maximum_lateral_spacing": 0.30,
        },
    )
    prolonged_support = RewTerm(
        func=mdp.prolonged_support_contact_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "maximum_contact_time": 0.40,
            "alignment_gate_error": 0.08,
            "midpoint_tolerance": 0.10,
            "longitudinal_spread_tolerance": 0.12,
            "minimum_lateral_spacing": 0.14,
            "maximum_lateral_spacing": 0.30,
        },
    )
    final_feet_stance = RewTerm(
        func=mdp.final_feet_stance_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "alignment_scale": 0.10,
            "midpoint_scale": 0.10,
            "longitudinal_scale": 0.12,
            "target_lateral_spacing": 0.22,
            "lateral_scale": 0.08,
        },
    )
    feet_spacing = RewTerm(
        func=mdp.adjust_feet_lateral_spacing_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "target_spacing": 0.22,
        },
    )
    lower_leg_alignment = RewTerm(
        func=mdp.lower_leg_forward_alignment_penalty,
        weight=-0.25,
        params={
            "feet_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "ankle_roll_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_Ankle_Roll"], preserve_order=True
            ),
            "foot_yaw_free_deg": 15.0,
            "foot_yaw_scale_deg": 10.0,
            "ankle_roll_free_deg": 12.0,
            "ankle_roll_scale_deg": 10.0,
            "ankle_roll_weight": 0.5,
        },
    )

    # Contact-free ball handling remains task-specific and strict.
    ball_motion = RewTerm(
        func=mdp.ball_motion_penalty,
        weight=-12.0,
        params={"speed_scale": 0.08, "displacement_scale": 0.02},
    )
    feet_ball_proximity = RewTerm(
        func=mdp.feet_ball_proximity_penalty,
        weight=-2.0,
        params={
            "safe_distance": 0.20,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
        },
    )

    # GoTo locomotion regularization.  These are deliberately much lighter
    # than the previous shaping so lifting a leg is not more expensive than
    # dragging it.
    nominal_pose = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    tilt_limit = RewTerm(
        func=mdp.dynamic_tilt_limit_penalty,
        weight=-3.0,
        params={
            "moving_allowance_deg": 10.0,
            "arrival_allowance_deg": 5.0,
            "upright_error": 0.06,
            "moving_error": 0.10,
            "violation_scale_deg": 5.0,
        },
    )
    arrival_upright = RewTerm(
        func=mdp.arrival_upright_penalty,
        weight=-9.0,
        params={"alignment_scale": 0.10},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )
    alignment_success = DoneTerm(
        func=mdp.alignment_success,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FEET, preserve_order=True
            ),
            "feet_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "position_tolerance": 0.06,
            "heading_tolerance_deg": 15.0,
            "linear_speed_tolerance": 0.10,
            "yaw_speed_tolerance": 0.10,
            "maximum_tilt_deg": 5.0,
            "contact_threshold": 1.0,
            "feet_midpoint_tolerance": 0.10,
            "feet_longitudinal_spread_tolerance": 0.12,
            "minimum_lateral_spacing": 0.14,
            "maximum_lateral_spacing": 0.30,
            "ball_speed_tolerance": 0.05,
            "ball_displacement_tolerance": 0.02,
            "stable_time": 1.50,
        },
    )
    ball_touched = DoneTerm(
        func=mdp.ball_motion_termination,
        params={"speed_threshold": 0.08, "displacement_threshold": 0.02},
    )


@configclass
class EventsCfg(KickEventsCfg):
    # The standalone adjust task owns the ball and target reset together so
    # the alignment point is derived from the same sampled kick direction.
    reset_ball = None
    reset_target = None
    push_robot = None
    reset_scenario = EventTerm(
        func=mdp.reset_adjust_scenario,
        mode="reset",
        params={
            "ball_x_range": (0.20, 0.35),
            "ball_y_range": (-0.15, 0.15),
            "target_distance_range": (4.0, 4.0),
            "visualize_target": False,
            # Absolute-angle bands force meaningful movement before the final
            # all-direction distribution. Transitions are at PPO iterations
            # 2k, 5k, 8k, and 10k.
            "target_angle_magnitude_ranges_deg": (
                (30.0, 60.0),
                (45.0, 90.0),
                (90.0, 135.0),
                (135.0, 180.0),
                (0.0, 180.0),
            ),
            "stage_steps": (48_000, 120_000, 192_000, 240_000),
            "curriculum_stage": -1,
            "ball_height": 0.105,
        },
    )


@configclass
class K1AdjustEnvCfg(ManagerBasedRLEnvCfg):
    scene: KickSceneCfg = KickSceneCfg(num_envs=4096, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: KickActionsCfg = KickActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.decimation * self.sim.dt
        self.actions.joint_pos.scale = {
            pattern: scale
            for pattern, scale in K1_ACTION_SCALE.items()
            if any(name in pattern for name in ("Hip", "Knee", "Ankle"))
        }


@configclass
class K1AdjustPlayEnvCfg(K1AdjustEnvCfg):
    """Deterministic initial-curriculum alignment evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.friction = None
        self.events.body_mass = None
        self.events.body_com = None
        self.events.pd_gains = None
        # Validate the same initial distribution as the first training stage.
        self.events.reset_scenario.params["curriculum_stage"] = 0
        self.events.reset_scenario.params["visualize_target"] = True
