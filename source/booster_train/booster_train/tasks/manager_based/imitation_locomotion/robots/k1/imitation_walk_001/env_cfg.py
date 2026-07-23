"""K1 imitation-guided velocity locomotion environments."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.tasks.manager_based.imitation_locomotion import mdp
from booster_train.tasks.manager_based.imitation_locomotion.agents.rsl_rl_ppo_cfg import (
    PPO_STEPS_PER_ITERATION,
)
from booster_train.tasks.manager_based.locomotion.robots.k1.walk_002.env_cfg import (
    FlatWoStateEstimationEnvCfg as WalkFlatEnvCfg,
)
from booster_train.tasks.manager_based.locomotion.robots.k1.walk_002.env_cfg import (
    RoughWoStateEstimationEnvCfg as WalkRoughEnvCfg,
)


LEG_JOINT_NAMES = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def _default_dataset_dirs() -> list[str]:
    """Use an environment override or the two datasets beside the source tree."""

    override = os.environ.get("BOOSTER_WALK_DATASETS")
    if override:
        return [str(Path(item).expanduser()) for item in override.split(os.pathsep) if item]
    train_root = Path(__file__).resolve().parents[9]
    return [str(train_root / "processed_walks_001"), str(train_root / "processed_walks_002")]


LEG_ASSET_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
FOOT_ASSET_CFG = SceneEntityCfg(
    "robot", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
)
FOOT_SENSOR_CFG = SceneEntityCfg(
    "contact_forces", body_names=["left_foot_link", "right_foot_link"], preserve_order=True
)


@configclass
class CommandsCfg:
    """Direct velocity commands plus an internal, confidence-gated reference."""

    base_velocity = mdp.ReferenceVelocityCommandCfg(
        asset_name="robot",
        dataset_dirs=_default_dataset_dirs(),
        joint_names=LEG_JOINT_NAMES,
        resampling_time_range=(3.0, 8.0),
        ranges=mdp.ReferenceVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
        steps_per_iteration=PPO_STEPS_PER_ITERATION,
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Reference-free actor and asymmetric, reference-aware critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # The inherited state-estimation-free env sets this field to None.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": LEG_ASSET_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": LEG_ASSET_CFG},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        # The inherited flat env expects this attribute and sets it to None.
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_ASSET_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": LEG_ASSET_CFG})
        actions = ObsTerm(func=mdp.last_action)
        reference_joint_pos = ObsTerm(
            func=mdp.reference_joint_pos_rel, params={"command_name": "base_velocity"}
        )
        reference_joint_vel = ObsTerm(
            func=mdp.reference_joint_vel, params={"command_name": "base_velocity"}
        )
        reference_confidence = ObsTerm(
            func=mdp.reference_confidence, params={"command_name": "base_velocity"}
        )
        reference_clip_progress = ObsTerm(
            func=mdp.reference_clip_progress, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


IMITATION_ITERATION_KNOTS = (0, 500, 3_000, 10_000)
IMITATION_CURRICULUM_VALUES = (1.0, 0.75, 0.40, 0.20)


@configclass
class RewardsCfg:
    """Command tracking dominates; imitation supplies an early gait prior."""

    termination_penalty = RewTerm(func=mdp.is_terminated_term, weight=-200.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": LEG_ASSET_CFG}
    )
    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": LEG_ASSET_CFG}
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": LEG_ASSET_CFG}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={"sensor_cfg": FOOT_SENSOR_CFG, "asset_cfg": FOOT_ASSET_CFG},
    )

    reference_joint_position = RewTerm(
        func=mdp.reference_joint_position_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.35,
            "iteration_knots": IMITATION_ITERATION_KNOTS,
            "curriculum_values": IMITATION_CURRICULUM_VALUES,
            "steps_per_iteration": PPO_STEPS_PER_ITERATION,
        },
    )
    reference_joint_velocity = RewTerm(
        func=mdp.reference_joint_velocity_exp,
        weight=0.10,
        params={
            "command_name": "base_velocity",
            "std": 3.0,
            "iteration_knots": IMITATION_ITERATION_KNOTS,
            "curriculum_values": IMITATION_CURRICULUM_VALUES,
            "steps_per_iteration": PPO_STEPS_PER_ITERATION,
        },
    )


@configclass
class CurriculumCfg:
    """Command modes and imitation weights are scheduled inside their terms."""

    pass


def _restore_stable_k1_pose(env_cfg) -> None:
    """Undo walk_002's all-zero override and restore the K1 crouched default."""

    env_cfg.scene.robot.init_state.joint_pos = {
        ".*_Hip_Pitch": -0.2,
        ".*_Knee_Pitch": 0.4,
        ".*_Ankle_Pitch": -0.25,
    }


@configclass
class FlatEnvCfg(WalkFlatEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _restore_stable_k1_pose(self)


@configclass
class RoughEnvCfg(WalkRoughEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _restore_stable_k1_pose(self)


@configclass
class PlayEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 30
        self.scene.env_spacing = 1.5
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.debug_vis = False
        self.commands.base_velocity.play = True
        self.commands.base_velocity.debug_vis = False
        self.events.push_robot = None
        self.events.reset_robot_joints = None
        self.observations.policy.enable_corruption = False

