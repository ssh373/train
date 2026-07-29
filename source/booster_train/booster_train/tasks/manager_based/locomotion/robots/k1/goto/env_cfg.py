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
from isaaclab.utils.math import wrap_to_pi as lab_wrap_to_pi

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.locomotion import mdp as common_mdp
from booster_train.tasks.manager_based.locomotion.goto import mdp

LEGS = [".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"]
FEET = ["left_foot_link", "right_foot_link"]
NOMINAL_BASE_HEIGHT = float(BOOSTER_K1_CFG.init_state.pos[2])


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
        radius = float(self.cfg.constellation_radius)
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


class AStarLikeSE2GoalCommand(VisualizedSE2GoalCommand):
    """Play-only stream of correlated, noisy look-ahead waypoints.

    This is deliberately not part of training.  It approximates the command
    pattern produced by a local A* planner: nearby waypoints progress along a
    route, occasional replans change direction sharply, and the perceived goal
    has small low-rate localization/planner jitter.  Only the command buffer is
    perturbed; the nominal world-frame waypoint remains fixed between events.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.route_heading_w = self.robot.data.heading_w.clone()
        self.goal_jitter_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.jitter_counter = 0

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        n = ids.numel()
        # Most updates are gentle path bends.  A minority mimic an obstacle
        # appearing and forcing A* to choose a substantially different branch.
        turn = torch.empty(n, device=self.device).uniform_(-math.radians(22.0), math.radians(22.0))
        sharp = torch.rand(n, device=self.device) < 0.12
        sharp_turn = torch.empty(n, device=self.device).uniform_(-math.radians(80.0), math.radians(80.0))
        turn = torch.where(sharp, sharp_turn, turn)
        self.route_heading_w[ids] = lab_wrap_to_pi(self.route_heading_w[ids] + turn)

        distance = torch.empty(n, device=self.device).uniform_(0.45, 0.90)
        heading = self.route_heading_w[ids]
        self.goal_pose_w[ids, 0] = self.robot.data.root_pos_w[ids, 0] + distance * torch.cos(heading)
        self.goal_pose_w[ids, 1] = self.robot.data.root_pos_w[ids, 1] + distance * torch.sin(heading)
        self.goal_pose_w[ids, 2] = heading
        self.category[ids] = 4  # combined translation + heading
        self.just_resampled[ids] = True

    def _update_command(self):
        super()._update_command()
        # A real local planner normally advances to the next waypoint after the
        # current one is reached.  Do not keep placing a fresh point in front of
        # a robot that has not had time to reach the previous one.
        reached = torch.linalg.vector_norm(self.goal_b[:, :2], dim=1) < 0.20
        self.time_left[reached] = 0.0

        # Refresh at 10 Hz for the 50 Hz GoTo policy.  The clamp prevents a
        # single noisy sample from becoming a fake replan event.
        if self.jitter_counter % 5 == 0:
            self.goal_jitter_b[:, :2] = torch.randn_like(self.goal_jitter_b[:, :2]).clamp_(-2.5, 2.5) * 0.02
            self.goal_jitter_b[:, 2] = torch.randn_like(self.goal_jitter_b[:, 2]).clamp_(-3.0, 3.0) * math.radians(2.0)
        self.jitter_counter += 1

        self.goal_b[:, :2] += self.goal_jitter_b[:, :2]
        noisy_heading = torch.atan2(self.goal_b[:, 2], self.goal_b[:, 3]) + self.goal_jitter_b[:, 2]
        self.goal_b[:, 2] = torch.sin(noisy_heading)
        self.goal_b[:, 3] = torch.cos(noisy_heading)


class VisualizedMixedDynamicSE2GoalCommand(mdp.MixedDynamicSE2GoalCommand):
    """Dynamic training command with GoTo markers and speed telemetry."""

    _set_debug_vis_impl = VisualizedSE2GoalCommand._set_debug_vis_impl
    _constellation_points = staticmethod(VisualizedSE2GoalCommand._constellation_points)
    _heading_points = staticmethod(VisualizedSE2GoalCommand._heading_points)

    def _curriculum_stage(self) -> int:
        # A fresh play environment starts at step zero; force the final training
        # distribution so the 1.5 m/s target regime is actually evaluated.
        return 3

    def _debug_vis_callback(self, event):
        VisualizedSE2GoalCommand._debug_vis_callback(self, event)
        if self._dynamic_update_counter % 25 != 0 or self.num_envs == 0:
            return
        mode_names = ("static", "moving", "astar")
        mode = mode_names[int(self.goal_mode[0].item())]
        target_speed = float(self.target_motion_speed[0].item())
        robot_speed = float(torch.linalg.vector_norm(self.robot.data.root_lin_vel_b[0, :2]).item())
        goal_distance = float(torch.linalg.vector_norm(self.goal_b[0, :2]).item())
        print(
            f"[DynamicGoTo env0] mode={mode} target_speed={target_speed:.3f}m/s "
            f"robot_speed={robot_speed:.3f}m/s goal_distance={goal_distance:.3f}m",
            flush=True,
        )


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
        class_type=mdp.UniformSE2GoalCommand,
        asset_name="robot", resampling_time_range=(2.0, 6.0),
        category_probabilities=(0.10, 0.20, 0.20, 0.20, 0.30),
        ranges=mdp.UniformSE2GoalCommandCfg.Ranges(
            delta_x=(-2.0, 2.0), delta_y=(-1.5, 1.5), delta_yaw=(-math.pi, math.pi)),
        constellation_radius=1.0, debug_vis=False,
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
    constellation = RewTerm(func=mdp.constellation_reward, weight=3.0, params={
        "command_name": "pose_goal", "radius": 0.7})
    success = RewTerm(func=mdp.goal_success, weight=2.0, params={
        "command_name": "pose_goal", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)})
    upright = RewTerm(func=common_mdp.flat_orientation_l2, weight=-1.0)
    vertical_velocity = RewTerm(func=common_mdp.lin_vel_z_l2, weight=-0.5)
    roll_pitch_rate = RewTerm(func=common_mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=common_mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(func=common_mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acceleration = RewTerm(func=common_mdp.joint_acc_l2, weight=-2.5e-7)
    torque = RewTerm(func=common_mdp.joint_torques_l2, weight=-1.0e-5)
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
        "nominal_height": 0.57, "fall_height_ratio": 0.60})
    trunk_contact = DoneTerm(func=common_mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0})


@configclass
class EventsCfg:
    reset_base = EventTerm(func=common_mdp.reset_root_state_uniform, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                       "roll": (0.0, 0.0), "pitch": (0.0, 0.0),
                       "yaw": (-math.pi, math.pi)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}})
    reset_joints = EventTerm(func=common_mdp.reset_joints_by_offset, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot"), "position_range": (-0.08, 0.08),
        "velocity_range": (0.0, 0.0)})
    recovery_push = EventTerm(func=common_mdp.push_by_setting_velocity, mode="interval",
                              interval_range_s=(6.0, 10.0), params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {"x": (-0.35, 0.35), "y": (-0.30, 0.30),
                           "roll": (-0.25, 0.25), "pitch": (-0.30, 0.30),
                           "yaw": (-0.20, 0.20)}})
    sustained_push = EventTerm(func=mdp.sustained_random_push, mode="interval",
                               interval_range_s=(0.02, 0.02), params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
        "push_interval_s": 10.0,
        "push_duration_s": 0.5,
        "force_magnitude_range": (5.0, 12.0),
        "torque_range": (-1.0, 1.0)})
    friction = EventTerm(func=common_mdp.randomize_rigid_body_material, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET), "static_friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.7, 1.1), "restitution_range": (0.0, 0.0), "num_buckets": 32})
    body_mass = EventTerm(func=common_mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "mass_distribution_params": (0.9, 1.1), "operation": "scale"})
    body_com = EventTerm(func=common_mdp.randomize_rigid_body_com, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)}})
    pd_gains = EventTerm(func=common_mdp.randomize_actuator_gains, mode="reset", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=LEGS),
        "stiffness_distribution_params": (0.9, 1.1),
        "damping_distribution_params": (0.9, 1.1), "operation": "scale"})


@configclass
class K1GoToEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = 42
    constellation_radius: float = 1.0
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.actions.joint_pos.scale = {k: v for k, v in K1_ACTION_SCALE.items()
                                        if any(token in k for token in ("Hip", "Knee", "Ankle"))}
        # Keep paper parameters synchronized with the command/reward implementations.
        self.commands.pose_goal.constellation_radius = self.constellation_radius
        self.rewards.constellation.params["radius"] = self.constellation_radius

        # Benign base-training randomization.  Physical disturbances are added
        # only by the dedicated Sim2Real/fine-tuning configuration.
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.reset_joints.params["position_range"] = (-0.05, 0.05)
        self.events.recovery_push = None
        self.events.sustained_push = None
        self.events.body_com = None
        self.events.pd_gains = None
        self.events.body_mass.params.update({
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        })


@configclass
class K1GoToSmokeEnvCfg(K1GoToEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64


@configclass
class K1GoToSim2RealEnvCfg(K1GoToEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        robust_events = EventsCfg()
        self.events.recovery_push = robust_events.recovery_push
        self.events.sustained_push = robust_events.sustained_push
        self.events.friction = robust_events.friction
        self.events.body_mass = robust_events.body_mass
        self.events.body_com = robust_events.body_com
        self.events.pd_gains = robust_events.pd_gains
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
        self.events.recovery_push = None
        self.events.sustained_push = None
        self.observations.policy.enable_corruption = False
        self.commands.pose_goal.class_type = VisualizedSE2GoalCommand
        self.commands.pose_goal.debug_vis = True


@configclass
class K1GoToAStarPlayEnvCfg(K1GoToPlayEnvCfg):
    """Play scene that stress-tests a trained GoTo policy with A*-like goals."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_goal.class_type = AStarLikeSE2GoalCommand
        # Match INHA's 1.5 s path hold. Reaching 0.20 m requests the next point
        # earlier; this interval is the replan timeout, not a forced fast switch.
        self.commands.pose_goal.resampling_time_range = (1.5, 2.5)


@configclass
class K1GoToDynamicEnvCfg(K1GoToEnvCfg):
    """Feed-forward task for changing BT/A* pose targets."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.pose_goal.class_type = mdp.MixedDynamicSE2GoalCommand
        self.episode_length_s = 30.0


@configclass
class K1GoToDynamicPlayEnvCfg(K1GoToDynamicEnvCfg):
    """Deterministic visualization of the dynamic-goal training distribution."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.events.friction = None
        self.events.body_mass = None
        self.observations.policy.enable_corruption = False
        self.commands.pose_goal.class_type = VisualizedMixedDynamicSE2GoalCommand
        self.commands.pose_goal.debug_vis = True
