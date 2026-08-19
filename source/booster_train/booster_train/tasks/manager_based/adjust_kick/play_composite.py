"""Play the one-file stateful unified adjust-kick policy."""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Play the unified learned adjust-kick policy."
)
parser.add_argument("--task", default="Booster-K1-Adjust-Kick_001-Play-v0")
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--robot_urdf", default=None)
parser.add_argument("--real_time", action="store_true")
parser.add_argument(
    "--teacher_control",
    action="store_true",
    help="Diagnostic mode: let frozen training teachers control the robot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import booster_train.tasks  # noqa: F401,E402


def _policy_observation(result):
    observation = result[0] if isinstance(result, tuple) else result
    # Unified training adds the internal phase scalar. The exported stateful
    # wrapper reconstructs that scalar, so deployment input remains 49-D.
    if observation.dim() == 2 and observation.size(1) == 50:
        return observation[:, :49]
    return observation


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.robot_urdf is not None:
        env_cfg.scene.robot.spawn.asset_path = os.path.abspath(
            os.path.expanduser(args_cli.robot_urdf)
        )
    if args_cli.teacher_control:
        env_cfg.actions.joint_pos.teacher_control_blend = (1.0, 1.0, 1.0, 1.0)
    else:
        # The learned unified action must reach the robot unchanged. Teacher
        # computation remains available for diagnostics but has zero blend.
        env_cfg.actions.joint_pos.teacher_control_blend = (0.0, 0.0, 0.0, 0.0)
    env_cfg.actions.joint_pos.debug_transition = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = vec_env.unwrapped
    policy_path = os.path.abspath(os.path.expanduser(args_cli.policy))
    policy = torch.jit.load(policy_path, map_location=raw_env.device).eval()

    observation = _policy_observation(vec_env.get_observations()).to(raw_env.device)
    reset_mask = torch.ones(
        raw_env.num_envs, dtype=torch.bool, device=raw_env.device
    )
    policy(observation)  # allocate scripted state for the current batch size
    policy.reset(reset_mask)

    print(f"[composite-play] policy: {policy_path}")
    print(f"[composite-play] robot asset: {env_cfg.scene.robot.spawn.asset_path}")
    print(
        "[unified-play] phases: 0=adjust, 1=0.2s transition, 2=kick/recovery "
        f"teacher_control={args_cli.teacher_control}"
    )

    step = 0
    dt = raw_env.step_dt
    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        start = time.time()
        with torch.inference_mode():
            action = policy(observation)
            next_observation, _, done, _ = vec_env.step(action)
            policy.reset(done.bool().view(-1))
            observation = _policy_observation(next_observation).to(raw_env.device)
        if step % 100 == 0:
            phase = policy.get_phase()
            counts = [int((phase == value).sum().item()) for value in range(3)]
            print(
                f"[composite-play] step={step} adjust={counts[0]} "
                f"transition={counts[1]} kick={counts[2]}",
                flush=True,
            )
        step += 1
        if args_cli.real_time:
            remaining = dt - (time.time() - start)
            if remaining > 0.0:
                time.sleep(remaining)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
