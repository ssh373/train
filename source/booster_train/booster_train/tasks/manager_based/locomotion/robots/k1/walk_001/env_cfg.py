# env_cfg.py (walk_001)
from __future__ import annotations
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    SceneEntityCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.locomotion.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from booster_train.tasks.manager_based.locomotion import mdp

@configclass
class BaseWalkSceneCfg(InteractiveSceneCfg):
    """기본 씬: terrain(평지) + robot + 센서"""

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
        debug_vis=True,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 4.0),
        heading_command=True,
        rel_standing_envs=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.3),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # joint_names=["^(?!.*(Shoulder_Pitch|Shoulder_Roll|Elbow_Yaw|Hip_Pitch|Knee_Pitch|Ankle_Pitch)).*$"], 
        joint_names=[".*"],
        scale=0.25,              # FlatEnvCfg에서 K1_ACTION_SCALE로 덮어씀
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 안정화/자세
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # 부드러운 액션
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # 관절 제한
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # 접촉 패널티
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

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )


@configclass
class EventsCfg:
    # reset: root state
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
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
            },
        },
    )

    # reset: joints
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # interval push (train 때만)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"asset_cfg": SceneEntityCfg("robot"), "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class BaseWalkEnvCfg(ManagerBasedRLEnvCfg):
    scene: BaseWalkSceneCfg = BaseWalkSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation

        # sensor update period
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

@configclass
class FlatEnvCfg(BaseWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ROBOT_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.57),
                joint_pos={
                    ".*": 0.0, # URDF에 정의된 기본 자세(rpy)를 0점으로 사용
                },
                joint_vel={".*": 0.0},
            ),
            spawn=ROBOT_CFG.spawn.replace(asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf"),
            )
        # exclude_list=["Shoulder_Pitch","Shoulder_Roll","Elbow_Yaw","Hip_Pitch","Knee_Pitch","Ankle_Pitch"]

        # self.actions.joint_pos.scale = {
        #     k: v for k, v in K1_ACTION_SCALE.items() 
        #     if not any(word in k for word in exclude_list)
        # }
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.observations.policy.height_scan = None


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.base_lin_vel = None


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False  # True로 설정하면 지형 분포를 시각화할 수 있음
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 각 지형 블록(Tile)의 크기 (단위: 미터)
            border_width=20.0,            # 전체 지형 외곽의 테두리 너비 (단위: 미터)
            num_rows=5,                   # 지형 그리드의 행(Row) 개수
            num_cols=10,                  # 지형 그리드의 열(Column) 개수
            horizontal_scale=0.1,         # 수평 해상도 (값이 작을수록 지형이 정밀해짐)
            vertical_scale=0.005,         # 수직 해상도 (높이 값의 최소 단위)
            slope_threshold=0.75,         # 메시(Mesh) 생성을 위한 경사도 단순화 임계값
            use_cache=False,              # 실행할 때마다 지형을 새로 생성함
            curriculum=False,             # 난이도 조절(커리큘럼) 학습 사용 안 함
            sub_terrains={
                # 80% 정도 평면에 가까운 지형 (매우 매끄러움)
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,              # 전체 지형 중 80% 비중 차지
                    noise_range=(0.0, 0.005),    # 높이 변화 폭: 0 ~ 0.5cm (거의 평지)
                    noise_step=0.005,            # 노이즈 변화 단계: 0.5cm
                    border_width=0.25,           # 개별 지형 블록의 테두리 너비
                ),
                # 20% 정도의 무작위 거친 지형
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,              # 전체 지형 중 20% 비중 차지
                    noise_range=(-0.015, 0.015), # 높이 변화 폭: ±1.5cm (울퉁불퉁하다...울퉁불퉁한...)
                    noise_step=0.005,            # 노이즈 변화 단계: 0.5cm
                    border_width=0.25,           # 개별 지형 블록의 테두리 너비
                ),
            },
        )


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None