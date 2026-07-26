"""Deterministic command-grid evaluation for Booster-K1-GoTo-v0."""

import argparse
import csv
import itertools
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Booster-K1-GoTo-v0-Play")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--output", default="goto_evaluation.csv")
parser.add_argument("--hold-time", type=float, default=0.25)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import booster_train.tasks  # noqa: F401


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    commands = list(itertools.product(
        (0.5, 1.0, 2.0, 4.0),
        (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
        (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4),
        range(args.seeds),
    ))
    env_cfg.scene.num_envs = len(commands)
    env_cfg.commands.pose_goal.resampling_time_range = (math.inf, math.inf)
    env_cfg.seed = 0
    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    term = env.unwrapped.command_manager.get_term("pose_goal")
    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids, _ = robot.find_bodies(["left_foot_link", "right_foot_link"])
    contact_ids, _ = sensor.find_bodies(["left_foot_link", "right_foot_link"])
    root_xy = robot.data.root_pos_w[:, :2]
    root_yaw = robot.data.heading_w
    for i, (distance, approach, heading, _) in enumerate(commands):
        world_angle = root_yaw[i] + approach
        term.goal_pose_w[i, 0] = root_xy[i, 0] + distance * torch.cos(world_angle)
        term.goal_pose_w[i, 1] = root_xy[i, 1] + distance * torch.sin(world_angle)
        term.goal_pose_w[i, 2] = root_yaw[i] + heading
    term.just_resampled[:] = True
    term._update_command()

    n = len(commands)
    energy = torch.zeros(n, device=robot.device)
    footsteps = torch.zeros(n, device=robot.device)
    previous_contact = torch.ones(n, 2, dtype=torch.bool, device=robot.device)
    success_time = torch.full((n,), float("nan"), device=robot.device)
    hold_steps = torch.zeros(n, device=robot.device)
    fallen = torch.zeros(n, dtype=torch.bool, device=robot.device)
    done_once = torch.zeros(n, dtype=torch.bool, device=robot.device)
    episode_return = torch.zeros(n, device=robot.device)
    final_pos = torch.full((n,), float("nan"), device=robot.device)
    final_ori = torch.full((n,), float("nan"), device=robot.device)
    max_steps = math.ceil(env_cfg.episode_length_s / env.unwrapped.step_dt)

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
        active = ~done_once
        episode_return += rewards * active
        torque = robot.data.applied_torque
        energy += torch.sum(torch.abs(torque * robot.data.joint_vel), dim=1) * env.unwrapped.step_dt * active
        forces = sensor.data.net_forces_w_history[:, :, contact_ids].norm(dim=-1).max(dim=1)[0]
        contact = forces > 1.0
        footsteps += ((~previous_contact) & contact).sum(dim=1) * active
        previous_contact = contact
        dx, dy, dtheta = term._relative_pose()
        success = ((dx.square() + dy.square() < 0.05**2) & (dtheta.abs() < 0.1)
                   & (robot.data.root_lin_vel_b[:, :2].norm(dim=1) < 0.1)
                   & (robot.data.root_ang_vel_b[:, 2].abs() < 0.1) & contact.all(dim=1))
        hold_steps = torch.where(success, hold_steps + 1, torch.zeros_like(hold_steps))
        reached = (hold_steps >= math.ceil(args.hold_time / env.unwrapped.step_dt)) & torch.isnan(success_time)
        success_time[reached] = (step + 1) * env.unwrapped.step_dt
        new_done = dones & ~done_once
        final_pos[new_done] = torch.sqrt(dx[new_done].square() + dy[new_done].square())
        final_ori[new_done] = torch.abs(dtheta[new_done])
        fallen |= new_done & (step + 1 < max_steps)
        done_once |= dones
        if hasattr(policy_nn, "reset"):
            policy_nn.reset(dones)

    dx, dy, dtheta = term._relative_pose()
    unfinished = torch.isnan(final_pos)
    final_pos[unfinished] = torch.sqrt(dx[unfinished].square() + dy[unfinished].square())
    final_ori[unfinished] = torch.abs(dtheta[unfinished])
    rows = []
    for i, command in enumerate(commands):
        rows.append({
            "distance": command[0], "approach_angle": command[1], "final_heading": command[2], "seed": command[3],
            "success": not math.isnan(success_time[i].item()), "time_to_target": success_time[i].item(),
            "final_position_error": final_pos[i].item(), "final_orientation_error": final_ori[i].item(),
            "footstep_count": int(footsteps[i].item()), "mechanical_energy_abs_joule": energy[i].item(),
            "energy_per_meter": energy[i].item() / max(command[0] - final_pos[i].item(), 0.05),
            "episode_return": episode_return[i].item(), "fall": bool(fallen[i].item()),
            "timeout": bool(done_once[i].item() and not fallen[i].item()),
        })
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    successes = sum(row["success"] for row in rows)
    print(f"Wrote {len(rows)} trials to {args.output}; success rate={successes / len(rows):.3f}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
