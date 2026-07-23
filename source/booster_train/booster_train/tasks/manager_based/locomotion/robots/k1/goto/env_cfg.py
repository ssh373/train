"""K1 end-to-end SE(2) pose-goal environment (50 Hz joint-position policy)."""

from __future__ import annotations

import math
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm, SceneEntityCfg, TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.locomotion import mdp as common_mdp
from booster_train.tasks.manager_based.locomotion.goto import mdp

LEGS = [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
FEET = ["left_foot_link", "right_foot_link"]
NOMINAL_BASE_HEIGHT = 0.57  # Confirmed by the existing K1 locomotion task/asset configuration.


class VisualizedSE2GoalCommand(mdp.UniformSE2GoalCommand):
    """GoTo command with current/target constellation markers in the simulator."""

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not hasattr(self, "current_constellation_visualizer"):
            def marker_cfg(prim_path, color, radius):
                return VisualizationMarkersCfg(
                    prim_path=prim_path,
                    markers={"point": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=color, emissive_color=color),
                    )},
                )

            self.current_constellation_visualizer = VisualizationMarkers(marker_cfg(
                "/Visuals/GoTo/current_constellation", (0.1, 0.9, 0.2), 0.035))
            self.goal_constellation_visualizer = VisualizationMarkers(marker_cfg(
                "/Visuals/GoTo/goal_constellation", (0.95, 0.15, 0.1), 0.045))
            self.current_heading_visualizer = VisualizationMarkers(marker_cfg(
                "/Visuals/GoTo/current_heading", (0.0, 0.8, 1.0), 0.075))
            self.goal_heading_visualizer = VisualizationMarkers(marker_cfg(
                "/Visuals/GoTo/goal_heading", (1.0, 0.55, 0.0), 0.085))

        for visualizer in (
            self.current_constellation_visualizer,
            self.goal_constellation_visualizer,
            self.current_heading_visualizer,
            self.goal_heading_visualizer,
        ):
            visualizer.set_visibility(debug_vis)

    @staticmethod
    def _constellation_points(x, y, yaw, radius, z=0.08):
        offsets = torch.tensor(
            ((0.0, 0.0), (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)),
            device=x.device,
        )
        c, s = torch.cos(yaw), torch.sin(yaw)
        px = x[:, None] + c[:, None] * offsets[:, 0] - s[:, None] * offsets[:, 1]
        py = y[:, None] + s[:, None] * offsets[:, 0] + c[:, None] * offsets[:, 1]
        pz = torch.full_like(px, z)
        return torch.stack((px, py, pz), dim=-1).reshape(-1, 3)

    @staticmethod
    def _heading_points(x, y, yaw, radius, z=0.08):
        return torch.stack(
            (x + radius * torch.cos(yaw), y + radius * torch.sin(yaw), torch.full_like(x, z)), dim=-1)

    def _debug_vis_callback(self, event):
        radius = math.sqrt(float(self.cfg.constellation_inertia))
        robot_pos = self.robot.data.root_pos_w
        robot_yaw = self.robot.data.heading_w
        goal = self.goal_pose_w
        self.current_constellation_visualizer.visualize(self._constellation_points(
            robot_pos[:, 0], robot_pos[:, 1], robot_yaw, radius))
        self.goal_constellation_visualizer.visualize(self._constellation_points(
            goal[:, 0], goal[:, 1], goal[:, 2], radius))
        self.current_heading_visualizer.visualize(self._heading_points(
            robot_pos[:, 0], robot_pos[:, 1], robot_yaw, radius))
        self.goal_heading_visualizer.visualize(self._heading_points(
            goal[:, 0], goal[:, 1], goal[:, 2], radius))


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"),
        debug_vis=False,
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class CommandsCfg:
    pose_goal = mdp.UniformSE2GoalCommandCfg(
        class_type=VisualizedSE2GoalCommand,
        asset_name="robot", resampling_time_range=(4.0, 8.0),
        category_probabilities=(0.10, 0.20, 0.20, 0.20, 0.30),
        ranges=mdp.UniformSE2GoalCommandCfg.Ranges(
            delta_x=(-2.0, 2.0), delta_y=(-1.5, 1.5), delta_yaw=(-math.pi, math.pi)),
        constellation_inertia=1.0, debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = common_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEGS, scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=common_mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=common_mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(func=common_mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=common_mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.2, n_max=0.2))
        previous_action = ObsTerm(func=common_mdp.last_action)
        goal = ObsTerm(func=mdp.goal_command, params={"command_name": "pose_goal"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        # Asymmetric simulator-only state. PolicyCfg remains deployable on K1.
        base_lin_vel = ObsTerm(func=common_mdp.base_lin_vel)
        base_height = ObsTerm(func=common_mdp.base_pos_z)
        feet_contact = ObsTerm(func=mdp.feet_grounded, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
        feet_velocity = ObsTerm(func=mdp.body_linear_velocity_w, params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
        applied_torque = ObsTerm(func=common_mdp.joint_effort, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    constellation = RewTerm(func=mdp.constellation_reward, weight=1.0, params={
        "command_name": "pose_goal", "weight": 0.2, "reward_scale": 1.0})
    progress = RewTerm(func=mdp.goal_progress, weight=2.0, params={"command_name": "pose_goal"})
    success = RewTerm(func=mdp.goal_success, weight=2.0, params={
        "command_name": "pose_goal", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    upright = RewTerm(func=common_mdp.flat_orientation_l2, weight=-0.5)
    vertical_velocity = RewTerm(func=common_mdp.lin_vel_z_l2, weight=-0.5)
    roll_pitch_rate = RewTerm(func=common_mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=common_mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(func=common_mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acceleration = RewTerm(func=common_mdp.joint_acc_l2, weight=-2.5e-7)
    torque = RewTerm(func=common_mdp.joint_torques_l2, weight=-1.0e-5)
    mechanical_power = RewTerm(func=mdp.mechanical_power_l1, weight=-2.0e-5)
    joint_limits = RewTerm(func=common_mdp.joint_pos_limits, weight=-2.0)
    nominal_pose = RewTerm(func=common_mdp.joint_deviation_l1, weight=-0.03,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS)})
    foot_slip = RewTerm(func=common_mdp.feet_slide, weight=-0.1, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    undesired_contact = RewTerm(func=common_mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"]),
        "threshold": 1.0})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=common_mdp.time_out, time_out=True)
    fall_height = DoneTerm(func=mdp.base_height_below_ratio, params={
        "nominal_height": NOMINAL_BASE_HEIGHT, "fall_height_ratio": 0.60})
    trunk_contact = DoneTerm(func=common_mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0})


@configclass
class EventsCfg:
    reset_base = EventTerm(func=common_mdp.reset_root_state_uniform, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}})
    reset_joints = EventTerm(func=common_mdp.reset_joints_by_offset, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"), "position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)})
    friction = EventTerm(func=common_mdp.randomize_rigid_body_material, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET), "static_friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.7, 1.1), "restitution_range": (0.0, 0.0), "num_buckets": 32})
    body_mass = EventTerm(func=common_mdp.randomize_rigid_body_mass, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
        "operation": "scale"})
    body_com = EventTerm(func=common_mdp.randomize_rigid_body_com, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)}})
    pd_gains = EventTerm(func=common_mdp.randomize_actuator_gains, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS),
        "stiffness_distribution_params": (0.9, 1.1),
        "damping_distribution_params": (0.9, 1.1), "operation": "scale"})


@configclass
class K1GoToEnvCfg(ManagerBasedRLEnvCfg):
    # Paper defaults. Keep configurable: K1 morphology may require retuning.
    constellation_radius: float = 1.0
    constellation_inertia: float = 1.0
    constellation_reward_scale: float = 1.0
    constellation_reward_weight: float = 0.2
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.actions.joint_pos.scale = {k: v for k, v in K1_ACTION_SCALE.items()
                                        if any(token in k for token in ("Hip", "Knee", "Ankle"))}
        # Keep paper parameters synchronized with the command/reward implementations.
        self.commands.pose_goal.constellation_inertia = self.constellation_inertia
        self.rewards.constellation.params.update(
            weight=self.constellation_reward_weight,
            reward_scale=self.constellation_reward_scale)


@configclass
class K1GoToSmokeEnvCfg(K1GoToEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class K1GoToSim2RealEnvCfg(K1GoToEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.friction.params.update(static_friction_range=(0.6, 1.3), dynamic_friction_range=(0.5, 1.2))
        self.events.body_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.body_com.params["com_range"] = {
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)}


@configclass
class K1GoToPlayEnvCfg(K1GoToEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.friction = None
        self.events.body_mass = None
        self.events.body_com = None
        self.events.pd_gains = None
        self.observations.policy.enable_corruption = False
        self.commands.pose_goal.debug_vis = True
