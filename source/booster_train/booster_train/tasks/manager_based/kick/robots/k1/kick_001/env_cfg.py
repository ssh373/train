"""Isaac Lab manager-based K1 ball-kicking environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.kick import mdp


@configclass
class KickSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ROBOT_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.57),
            joint_pos={
                ".*_Hip_Pitch": -0.2,
                ".*_Knee_Pitch": 0.4,
                ".*_Ankle_Pitch": -0.20,
            },
            joint_vel={".*": 0.0},
        ),
        spawn=ROBOT_CFG.spawn.replace(asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_locomotion.urdf"),
        actuators={"legs": ROBOT_CFG.actuators["legs"], "feet": ROBOT_CFG.actuators["feet"]},
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            # Nominal association-football Size 4: about 20.5 cm diameter.
            radius=0.103,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.37),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.4,
                dynamic_friction=0.3,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.105)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        ball_position = ObsTerm(
            # Use the cached camera model, including noise, dropout, age and
            # confidence, instead of the exact simulator ball position.
            func=mdp.camera_ball_pos_b,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "base_noise_std": 0.03,
                "distance_noise_ratio": 0.02,
                "dropout_rate_per_s": 0.50,
                "dropout_duration_range": (0.08, 0.30),
            },
            clip=(-3.0, 3.0),
        )
        target_position = ObsTerm(
            # Direction-only target input.  The target is defined from the
            # ball because the learned objective is the ball's launch angle.
            func=mdp.kick_ball_target_direction_b,
            params={"target_xy": (4.0, 0.0)},
            scale=1.0,
            clip=(-1.0, 1.0),
        )
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        ball_position = ObsTerm(func=mdp.ball_pos_w, params={"ball_cfg": SceneEntityCfg("ball")})
        ball_velocity = ObsTerm(func=mdp.ball_vel_w, params={"ball_cfg": SceneEntityCfg("ball")})
        target_position = ObsTerm(func=mdp.kick_target_pos_b, params={"target_xy": (4.0, 0.0)})
        feet_position = ObsTerm(
            func=mdp.feet_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])},
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
    stationary_xy = RewTerm(func=mdp.stationary_base_xy, weight=0.1, params={"std": 0.5})
    # Yaw motion is allowed: the actor may twist its trunk to cover the full
    # +/-30 degree target range.
    stationary_yaw = RewTerm(func=mdp.stationary_base_yaw, weight=0.10, params={"std": 0.5})

    ball_velocity_target = RewTerm(
        func=mdp.ball_velocity_to_target,
        weight=12.0,
        params={"target_xy": (4.0, 0.0), "decay_distance": 4.0, "max_reward": 10.0},
    )
    ball_target_accuracy = RewTerm(
        func=mdp.ball_target_accuracy,
        weight=10.0,
        params={"target_xy": (4.0, 0.0), "std": 0.25},
    )
    kick_direction_accuracy = RewTerm(
        func=mdp.kick_direction_accuracy,
        weight=30.0,
        # Do not let tiny/noisy ball motion dominate the direction objective.
        params={
            "target_xy": (4.0, 0.0),
            "minimum_speed": 0.20,
            "full_reward_angle_deg": 5.0,
            "zero_reward_angle_deg": 15.0,
            "max_penalty_angle_deg": 30.0,
        },
    )
    ball_overspeed = RewTerm(
        func=mdp.ball_overspeed,
        weight=-1.5,
        params={"max_speed": 3.0},
    )
    ball_lateral_velocity = RewTerm(
        func=mdp.ball_lateral_velocity,
        weight=-30.0,
        params={"target_xy": (4.0, 0.0)},
    )
    kicking_foot_approach = RewTerm(
        func=mdp.kicking_foot_approach_ball,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "proximity_std": 0.25,
            "stationary_speed": 0.1,
            "velocity_weight": 0.3,
            "center_deadband": 0.03,
        },
    )
    kicking_foot_progress = RewTerm(
        func=mdp.kicking_foot_approach_progress,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "center_deadband": 0.03,
            "max_progress_per_step": 0.04,
        },
    )
    preferred_foot_kick = RewTerm(
        func=mdp.preferred_foot_kick_event,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "speed_increase_threshold": 0.08,
            "max_contact_distance": 0.25,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    wrong_foot_proximity = RewTerm(
        func=mdp.wrong_foot_proximity,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "proximity_std": 0.15,
        },
    )
    ball_acceleration = RewTerm(
        func=mdp.ball_acceleration_to_target,
        weight=0.5,
        params={"target_xy": (4.0, 0.0), "scale": 80.0},
    )
    waiting = RewTerm(func=mdp.waiting_penalty, weight=-12.0)
    no_kick_failure = RewTerm(
        func=mdp.no_kick_failure_penalty,
        weight=-25.0,
        params={"time_limit": 2.5, "movement_speed": 0.12},
    )
    pre_kick_stability = RewTerm(
        func=mdp.pre_kick_stability,
        # Make the policy settle before the kick instead of rewarding only
        # the ball-contact event.
        weight=0.25,
        params={"ball_speed_threshold": 0.15},
    )
    post_kick_recovery = RewTerm(
        func=mdp.post_kick_recovery,
        # Keep the robot upright and quiet after the ball has been kicked.
        weight=12.0,
        params={"kick_speed_threshold": 0.2},
    )
    walk_ready_after_kick = RewTerm(
        func=mdp.walk_ready_after_kick,
        weight=6.0,
        params={"return_delay": 0.5},
    )
    post_kick_feet_grounded = RewTerm(
        func=mdp.post_kick_feet_grounded,
        # Explicitly bring both feet back to the pre-kick ground stance.
        weight=4.0,
        params={
            "return_delay": 0.25,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
            ),
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-200.0,
        params={"target_height": 0.55},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-20.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-5.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-3.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # A strong action-rate penalty fights the fast post-kick recovery motion.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.4)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    support_foot_slide = RewTerm(
        func=mdp.support_foot_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )
    hip_roll_spread = RewTerm(
        func=mdp.hip_roll_spread,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Roll"])},
    )
    # Suppress both rapid yaw-rate swings and slow whole-body turns. Small
    # heading changes remain available for balance.
    body_twist = RewTerm(func=mdp.body_twist, weight=-6.0)
    body_yaw_deviation = RewTerm(
        func=mdp.body_yaw_deviation,
        weight=-8.0,
        params={"deadband_deg": 8.0},
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
    kick_success = DoneTerm(
        func=mdp.ball_success,
        params={
            "target_xy": (4.0, 0.0),
            "target_radius": 0.15,
            # cos(theta) > 0.995 corresponds to about +/-5.7 degrees.
            "min_direction_score": 0.995,
            "max_speed": 2.5,
            "recovery_time": 0.8,
            "max_base_speed": 0.35,
            "max_tilt": 0.2,
            "max_mean_joint_deviation": 0.35,
        },
    )
    ball_too_far = DoneTerm(func=mdp.ball_too_far, params={"max_distance": 5.5})
    ball_not_kicked = DoneTerm(
        func=mdp.ball_not_kicked_in_time,
        params={
            "time_limit": 2.5,
            "movement_speed": 0.12,
            # Reject a kick whose initial velocity is more than about 11.5
            # degrees away from the ball-to-target direction.
            "min_direction_cos": 0.98,
        },
    )


@configclass
class EventsCfg:
    # Sim-to-real randomization.  Keep these modest so the learned nominal kick
    # remains intact while the reset/recovery states see hardware variation.
    friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.1, 0.9),
            "num_buckets": 64,
        },
    )
    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
            ),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_ball = EventTerm(
        func=mdp.reset_ball_in_front,
        mode="reset",
        params={"x_range": (0.10, 0.35), "y_range": (-0.15, 0.0), "height": 0.105},
    )
    reset_target = EventTerm(
        func=mdp.reset_kick_target,
        mode="reset",
        params={
            "distance_range": (4.0, 4.0),
            "angle_range_deg": (-30.0, 30.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.external_push,
        mode="interval",
        interval_range_s=(6.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "push_interval_s": 6.0,
            "push_duration_s": 1.0,
            "force_randomization": {
                "range": (0.0, 6.0),
                "operation": "additive",
                "distribution": "gaussian",
            },
            "torque_randomization": {
                "range": (0.0, 0.6),
                "operation": "additive",
                "distribution": "gaussian",
            },
        },
    )


@configclass
class K1KickEnvCfg(ManagerBasedRLEnvCfg):
    scene: KickSceneCfg = KickSceneCfg(num_envs=4096, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 7.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.decimation * self.sim.dt
        # Use the same torque-aware per-joint range as the locomotion tasks instead
        # of allowing the initial stochastic policy to request +/-1 rad everywhere.
        self.actions.joint_pos.scale = {
            pattern: scale
            for pattern, scale in K1_ACTION_SCALE.items()
            if any(name in pattern for name in ("Hip", "Knee", "Ankle"))
        }


@configclass
class K1KickPlayEnvCfg(K1KickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 8.0
        self.events.push_robot = None
        self.events.friction = None
        self.events.body_mass = None
        self.events.body_com = None
        self.events.pd_gains = None
        self.events.reset_target.params["visualize_target"] = True
        self.events.reset_target.params["target_radius"] = 0.15
        # Reproduce the 2026-08-05 checkpoint observation contract in Play.
        self.observations.policy.ball_position.func = mdp.ball_pos_b
        self.observations.policy.ball_position.params = {"ball_cfg": SceneEntityCfg("ball")}
        self.observations.policy.ball_position.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.scene.robot.init_state.joint_pos[".*_Ankle_Pitch"] = -0.20
        self.scene.ball.spawn.physics_material.static_friction = 0.4
        self.scene.ball.spawn.physics_material.dynamic_friction = 0.3
        self.scene.ball.spawn.physics_material.restitution = 0.0


@configclass
class K1KickLegacyPlayEnvCfg(K1KickPlayEnvCfg):
    """Play environment for checkpoints trained with the former physics rate.

    Keeps the current no-action-clip policy path, while restoring the old
    physics integration (200 Hz simulation, four physics steps per action).
    All training randomization and disturbances are disabled for deterministic
    checkpoint comparison.
    """

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        # The old checkpoint was trained with K1_locomotion.urdf, not the
        # newer K1_jy_locomotion.urdf used by the current kick environment.
        self.scene.robot.spawn.asset_path = f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_locomotion.urdf"
        # Restore the former kick play physics shown in the saved config.
        self.scene.terrain.physics_material.static_friction = 0.5
        self.scene.terrain.physics_material.dynamic_friction = 0.5
        self.scene.terrain.physics_material.friction_combine_mode = "average"
        self.scene.ball.spawn.physics_material.static_friction = 1.0
        self.scene.ball.spawn.physics_material.dynamic_friction = 1.0
        self.scene.ball.spawn.physics_material.restitution = 0.0
        self.scene.ball.spawn.physics_material.friction_combine_mode = "multiply"
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.policy.ball_position.noise = Unoise(n_min=-0.01, n_max=0.01)
        self.events.friction = None
        self.events.body_mass = None
        self.events.body_com = None
        self.events.pd_gains = None
        self.events.push_robot = None
