# env_cfg.py (walk_002)
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
        rel_standing_envs=0.05,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # joint_names=[".*"],
        joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"],
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
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.1},  # std 0.25 → 0.15: yaw 추적 정밀도 강화
    )

    # 안정화/자세
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # ── 팔자걸음 방지: Hip_Yaw가 기본값(0)에서 벗어나지 않도록 페널티 ──
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw"])},
    )

    # ── 좌우 비대칭 방지: Hip_Roll, Ankle_Roll 기본값 유지 ──
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Roll"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Roll"])},
    )

    # 부드러운 액션 (급격한 동작 변화 억제 → 발발발 줄임)
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


    # ──────────────────────────────────────────────────────
    # ── 자연스러운 보행 보상 (무릎 들기 + 보폭 + 미끄럼 방지) ──
    # ──────────────────────────────────────────────────────

    # (1) 이족 보행 air-time: 한 발씩 번갈아 들도록 유도
    #     weight=2.0, threshold=0.55 → 더 긴 스윙으로 보폭 확대
    feet_air_time_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.55,           # air-time 보상 상한 (초): 0.55로 상향
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
            "vel_threshold": 0.05,
        },
    )

    # (2) 발 들어올림(swing clearance): 스윙 중 발이 target 높이까지 올라가도록
    #     target_height ↑, weight ↑ → 발을 높이 들어 보폭 확대
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
            ),
            "target_height": 0.08,       # 발을 8cm 들어 올리도록 유도 (5→8cm)
            "std": 0.05,
            "tanh_mult": 2.0,            # 수평 속도에 대한 민감도
        },
    )

    # (3) 발 미끄럼 패널티: 지면 접촉 중 수평 이동 억제 → 찍고 밀지 않도록
    feet_slide_penalty = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )

    # (4) 양발 동시 접촉(double-support) 페널티: 이동 중 한 발은 들어야 함
    double_support = RewTerm(
        func=mdp.double_support_penalty,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
            "vel_threshold": 0.1,
        },
    )

    # (5) 속도 비례 보폭 보상: 빠를수록 긴 스윙을 유도
    stride_length = RewTerm(
        func=mdp.stride_length_reward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "vel_threshold": 0.1,
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
            "position_range": (-0.15, 0.15),
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



# ── 커리큘럼 헬퍼: 속도 범위를 선형 보간 ──
def _lerp_velocity_range(env, env_ids, data, initial_range, final_range, num_steps_start, num_steps_end):
    """학습 진행도에 따라 velocity command 범위를 initial → final로 선형 확장."""
    if env.common_step_counter < num_steps_start:
        return initial_range
    if env.common_step_counter >= num_steps_end:
        return final_range
    progress = (env.common_step_counter - num_steps_start) / (num_steps_end - num_steps_start)
    new_min = initial_range[0] + progress * (final_range[0] - initial_range[0])
    new_max = initial_range[1] + progress * (final_range[1] - initial_range[1])
    return (new_min, new_max)


@configclass
class CurriculumCfg:
    """속도 명령 범위를 점진적으로 확장하는 커리큘럼.

    - 0~3000 iteration (0~72000 steps): 쉬운 범위 → 최종 범위
    - 3000 iteration 이후: 최종 범위 유지
    """

    vel_x_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": _lerp_velocity_range,
            "modify_params": {
                "initial_range": (-0.3, 0.5),
                "final_range": (-1.0, 1.5),
                "num_steps_start": 0,
                "num_steps_end": 72000,   # ≈ 3000 iterations
            },
        },
    )
    vel_y_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": _lerp_velocity_range,
            "modify_params": {
                "initial_range": (-0.3, 0.3),
                "final_range": (-1.0, 1.0),
                "num_steps_start": 0,
                "num_steps_end": 72000,
            },
        },
    )
    vel_yaw_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": _lerp_velocity_range,
            "modify_params": {
                "initial_range": (-0.5, 0.5),
                "final_range": (-1.2, 1.2),
                "num_steps_start": 0,
                "num_steps_end": 72000,
            },
        },
    )



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
                    ".*_Hip_Pitch": -0.2,
                    ".*_Knee_Pitch": 0.4,
                    ".*_Ankle_Pitch": -0.2,
                },
                joint_vel={".*": 0.0},
            ),
            spawn=ROBOT_CFG.spawn.replace(
                asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_locomotion.urdf",
            ),
            actuators={
                "legs": ROBOT_CFG.actuators["legs"],
                "feet": ROBOT_CFG.actuators["feet"],
            },
        )
        
        # self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.actions.joint_pos.scale = {
            k: v for k, v in K1_ACTION_SCALE.items() 
            if any(s in k for s in ["Hip", "Knee", "Ankle"])
        }


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

        #  로봇을 한 대만 소환 
        self.scene.num_envs = 30
        self.scene.env_spacing = 1.5

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.terrain.debug_vis = False 
        self.commands.base_velocity.debug_vis = False

        # 학습용 방해 이벤트 비활성화
        self.events.push_robot = None
        self.events.reset_robot_joints = None

        #조명
        if self.scene.light is not None:
            self.scene.light.spawn.intensity = 5000.0