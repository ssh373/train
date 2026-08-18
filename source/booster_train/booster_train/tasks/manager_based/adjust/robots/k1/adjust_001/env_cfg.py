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

    # The main objective is to reach a ball-relative target point quickly.
    alignment_position = RewTerm(
        func=mdp.alignment_position_reward,
        weight=6.0,
        params={"std": 0.12},
    )
    heading_target = RewTerm(
        func=mdp.heading_target_reward,
        weight=2.0,
        params={"std_deg": 15.0},
    )
    alignment_progress = RewTerm(
        func=mdp.alignment_position_progress,
        weight=12.0,
        params={"max_progress_per_step": 0.04},
    )
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
    approach_time = RewTerm(func=mdp.alignment_time_penalty, weight=-0.20)

    # Contact-free alignment is a hard requirement, not just a small shaping
    # preference.  Termination below also catches meaningful ball motion.
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

    # Keep the base and joint behavior compatible with kick_001's action
    # contract while allowing translational movement.
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-200.0,
        params={"target_height": 0.55},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-20.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-3.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.5)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    body_twist = RewTerm(func=mdp.body_twist, weight=-5.0)


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
            "position_tolerance": 0.06,
            "heading_tolerance_deg": 15.0,
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
            # PPO iteration transitions: 2k, 5k, and 10k.
            # Stage 0 matches kick_001's current +/-30-degree target range;
            # later stages expand the final result point toward full 360 degrees.
            "target_angle_ranges_deg": ((-30.0, 30.0), (-60.0, 60.0), (-120.0, 120.0), (-180.0, 180.0)),
            "stage_steps": (48_000, 120_000, 240_000),
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
    """Deterministic full-360-degree alignment evaluation configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.friction = None
        self.events.body_mass = None
        self.events.body_com = None
        self.events.pd_gains = None
        self.events.reset_scenario.params["curriculum_stage"] = 3
        self.events.reset_scenario.params["visualize_target"] = True
