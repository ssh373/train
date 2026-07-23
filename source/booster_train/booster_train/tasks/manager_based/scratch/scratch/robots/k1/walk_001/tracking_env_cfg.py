from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg # env.step을 구현하는 대신 lab의 ManagerBasedRLEnv에 구성요소를 등록
from isaaclab.managers import EventTermCfg as EventTerm # 랜덤화/reset/외란 이벤트
from isaaclab.managers import ObservationGroupCfg as ObsGroup # 여러 obs term의 묶음
from isaaclab.managers import ObservationTermCfg as ObsTerm # 개별 obs
from isaaclab.managers import RewardTermCfg as RewTerm # 개별 reward
from isaaclab.managers import SceneEntityCfg # robot body / joint, sensor의 특정 부분 선택
from isaaclab.managers import TerminationTermCfg as DoneTerm # termination 조건
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg

import math 

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.scratch.mdp as mdp

K1_LEG_JOINT_NAMES = [
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

WALK_COMMAND_CFG = mdp.StagedWalkCommandCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1, # 모든 환경의 로봇이 이 공통 지면과 충돌할 수 있도록 하는 설정 
        physics_material=sim_utils.RigidBodyMaterialCfg( # 물리 재질
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.3, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = WALK_COMMAND_CFG( # 실제 속도 입력으로 변경  
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),  # 부모 class에서 갖고 있음 -> 3~8초 랜덤 리샘플링
        still_proportion = 0.1, # 10%는 정지 명령을 받도록
        sample_internal_parameters=False, 
        velocity_ranges=mdp.INHAWalkCommandCfg.VelocityRanges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.5, 1.5),
            ang_vel_yaw=(-1.6, 1.6),
        )
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=K1_LEG_JOINT_NAMES, 
                                           scale=1.0, # htwk 형태로 scale 1로 변경
                                           use_default_offset=True)


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # obs
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"command_name": "motion"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
                            noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},                            
                            noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True # obs term의 noise를 실제로 활성화
            self.concatenate_terms = True # 각 term을 하나의 긴 vector로 연결해 actor에 입력

    @configclass
    class PrivilegedCfg(ObsGroup): 
        # actor obs -> rsl_rl에서 최종적으로 actor obs와 합쳐져서 들어가는지 확인해볼 필요가 있음
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"command_name": "motion"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])})
        actions = ObsTerm(func=mdp.last_action)
        # critic obs
        base_mass_scaled = ObsTerm(func=mdp.base_mass_scaled)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=mdp.base_height, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        push_force = ObsTerm(func=mdp.push_force, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")})
        push_torque = ObsTerm(func=mdp.push_torque, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg: # 환경 생성 시 또는 학습 중 특정 시점에 자동 실행되는 동작
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(func=mdp.randomize_base_com_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "base_com_randomization": {
                "range": (-0.1, 0.1),
                "operation": "additive",
                "distribution": "uniform",
            },
            "base_mass_randomization": {
                "range": (0.8, 1.2),
                "operation": "scaling",
                "distribution": "uniform",
            },
        },
    )

    # interval
    kick_robot = EventTerm(func=mdp.kick_by_setting_velocity, mode="interval", interval_range_s=(4.0, 4.0),
                            params={
                                "lin_vel_randomization": {
                                    "range": (0.0, 0.05),
                                    "operation": "additive",
                                    "distribution": "gaussian",
                                },
                                "ang_vel_randomization": {
                                    "range": (0.0, 0.01),
                                    "operation": "additive",
                                    "distribution": "gaussian",
                                }})

    external_push = EventTerm(func=mdp.external_push, mode="interval", interval_range_s=(0.02, 0.02),
                              params={
                                    "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                                    "push_interval_s": 8.0,
                                    "push_duration_s": 0.2,
                                    "force_randomization": {
                                        "range": (0.0, 1.5),
                                        "operation": "additive",
                                        "distribution": "gaussian",
                                    },
                                    "torque_randomization": {
                                        "range": (0.0, 0.15),
                                        "operation": "additive",
                                        "distribution": "gaussian",
                                    }})

    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
                           params={
                               "asset_cfg": SceneEntityCfg("robot"),
                               "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
                               "velocity_range": {
                                    "x": (-0.1, 0.1),
                                    "y": (-0.1, 0.1),
                                    "z": (0.0, 0.0),
                                    "roll": (0.0, 0.0),
                                    "pitch": (0.0, 0.0),
                                    "yaw": (-0.1, 0.1),
                               }
                           })
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                   params={
                                       "asset_cfg": SceneEntityCfg("robot"),
                                       "position_range": (-0.05, 0.05),
                                       "velocity_range": (-0.1, 0.1),
                                   })

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    survival = RewTerm(func=mdp.survival, weight=0.25)
    tracking_lin_vel_x = RewTerm(func=mdp.track_lin_vel_x_exp, 
                                 weight=1.0, 
                                 params={"command_name": "motion", "tracking_sigma": 0.1})
    tracking_lin_vel_y = RewTerm(func=mdp.track_lin_vel_y_exp, 
                                 weight=1.0, 
                                 params={"command_name": "motion", "tracking_sigma": 0.1})
    tracking_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp,
                                  weight=0.5,
                                  params={"command_name": "motion", "tracking_sigma": 0.1})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_theta_exp, 
                                 weight=0.5, 
                                 params={"command_name": "motion", "tracking_sigma": 0.1})
    base_height = RewTerm(func=mdp.base_height_l2, weight=-20.0, params={"target_height": 0.52, "sensor_cfg": SceneEntityCfg("height_scanner")})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    collision = RewTerm(func=mdp.undesired_contacts,
                        weight=-1.0,
                        params={
                            "sensor_cfg": SceneEntityCfg(
                                "contact_forces", body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"]),
                            "threshold": 1.0,
                        })
    
    # 발 관련 reward
    foot_yaw_L = RewTerm(func=mdp.foot_yaw_l_l2, weight=-2.0,
                         params={
                            "command_name": "motion",
                            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})

    foot_yaw_R = RewTerm(func=mdp.foot_yaw_r_l2, weight=-2.0,
                         params={
                            "command_name": "motion",
                            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})

    feet_offset_x = RewTerm(func=mdp.feet_offset_x_l1, weight=-12.0,
                            params={
                                "command_name": "motion",
                                "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
                                "max_forward_vel": 1.0})

    feet_offset_y = RewTerm(func=mdp.feet_offset_y_l1, weight=-12.0,
                            params={
                                "command_name": "motion",
                                "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
                                "feet_distance_ref": 0.18,
                                "max_lateral_vel": 1.0})

    feet_swing = RewTerm(func=mdp.feet_swing, weight=3.0,
                         params={
                            "command_name": "motion",
                            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
                            "swing_period": 0.2,
                            "threshold": 1.0})
    # 안정화 관련 
    orientation = RewTerm(func=mdp.orientation_l2, weight=-5.0, params={"command_name": "motion"})
    heading_drift = RewTerm(func=mdp.heading_drift_l2, weight=-5.0, params={"command_name": "motion", "deadband": 0.1})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits_count, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})
    power = RewTerm(func=mdp.power_positive, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})

    torque_tiredness = RewTerm(func=mdp.torque_tiredness, weight=-1.0e-2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)})
    root_acc = RewTerm(func=mdp.root_lin_acc_l2, weight=-1.0e-4, params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")})
    feet_slip = RewTerm(func=mdp.feet_slip, weight=-0.1,
                        params={
                            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
                            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
                            "threshold": 1.0})

    feet_roll = RewTerm(func=mdp.feet_roll_l2, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})
    feet_pitch = RewTerm(func=mdp.feet_pitch_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})

    feet_yaw_diff = RewTerm(func=mdp.feet_yaw_diff_l2, weight=-1.0,
                            params={
                                "command_name": "motion",
                                "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})

    feet_yaw_mean = RewTerm(func=mdp.feet_yaw_mean_l2, weight=-1.0,
                            params={
                                "command_name": "motion",
                                "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"])})
    # 속도 명령대로 움직이지 않을 때의 패널티들
    pure_x_lateral_drift = RewTerm(func=mdp.pure_x_lateral_drift_l2, weight=-2.0,
                                    params={"command_name": "motion", "command_deadband": 0.05, "zero_deadband": 0.05})

    pure_x_yaw_drift = RewTerm(func=mdp.pure_x_yaw_drift_l2, weight=-4.0,
                                params={"command_name": "motion", "command_deadband": 0.05, "zero_deadband": 0.05})

    pure_y_forward_drift = RewTerm(func=mdp.pure_y_forward_drift_l2, weight=-2.0,
                                    params={"command_name": "motion", "command_deadband": 0.05, "zero_deadband": 0.05})

    pure_y_yaw_drift = RewTerm(func=mdp.pure_y_yaw_drift_l2, weight=-4.0, params={"command_name": "motion", "command_deadband": 0.05, "zero_deadband": 0.05})

    zero_yaw_command_ang_vel = RewTerm(func=mdp.zero_yaw_command_ang_vel_l2, weight=-5.0,
                                        params={"command_name": "motion", "zero_deadband": 0.05})

    xy_perpendicular_velocity = RewTerm(func=mdp.xy_perpendicular_velocity_l2, weight=-1.0,
                                        params={"command_name": "motion", "command_deadband": 0.05})
    joint_roll_yaw_symmetry = RewTerm(func=mdp.joint_roll_yaw_symmetry_l2, weight=-0.2)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum,
                            params={
                                "minimum_height": 0.35,
                                "sensor_cfg": SceneEntityCfg("height_scanner"),
                            })
    root_velocity = DoneTerm(func=mdp.root_velocity_above_threshold, params={"max_velocity_square": 50.0})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


# Environment configuration
@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10 # 같은 action을 decimation step 동안 유지
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.002 # 물리 시뮬레이션 2ms마다 계산
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.origin_type = "world"   # for rotation the view by mouse and keyboard
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
