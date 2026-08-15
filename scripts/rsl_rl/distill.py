"""Supervised walk/kick teacher distillation for the locomotion-kick task."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Distill frozen walk and kick teachers into the student policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--walk_model", type=str, required=True)
parser.add_argument("--kick_model", type=str, required=True)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--output", type=str, default="logs/rsl_rl/locomotion_kick_distilled/student.pt")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

os.environ["VELOCITY_TEACHER_JIT"] = os.path.abspath(os.path.expanduser(args_cli.walk_model))
os.environ["KICK_TEACHER_JIT"] = os.path.abspath(os.path.expanduser(args_cli.kick_model))
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import booster_train.tasks  # noqa: F401, E402
from booster_train.tasks.manager_based.locomotion_kick import mdp  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Some rsl_rl versions only create these attributes when a log directory is
    # provided, although save() reads them unconditionally. Distillation does not
    # use an experiment logger, so make that state explicit for checkpoint saves.
    if not hasattr(runner, "logger_type"):
        runner.logger_type = getattr(agent_cfg, "logger", "tensorboard")
    if not hasattr(runner, "disable_logs"):
        runner.disable_logs = True
    policy = runner.alg.policy
    actor = policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=3.0e-4)
    best_loss = float("inf")
    output = os.path.abspath(args_cli.output)
    best_output = os.path.splitext(output)[0] + "_best.pt"
    raw_env = vec_env.unwrapped
    initial_observations = vec_env.get_observations()
    obs = initial_observations[0] if isinstance(initial_observations, tuple) else initial_observations
    obs = obs.to(raw_env.device)

    for iteration in range(args_cli.iterations):
        teacher_walk, teacher_kick = mdp._teacher_actions(
            raw_env,
            os.environ["VELOCITY_TEACHER_JIT"],
            os.environ["KICK_TEACHER_JIT"],
        )
        mode = raw_env.command_manager.get_command("walk_kick")[:, 5:8]
        target = torch.where(mode[:, 0:1] > 0.5, teacher_walk, teacher_kick)
        student = actor(obs)
        loss = torch.mean((student - target.detach()) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(best_output), exist_ok=True)
            runner.save(best_output)
        if (iteration + 1) % args_cli.save_interval == 0:
            periodic_output = os.path.splitext(output)[0] + f"_iter_{iteration + 1}.pt"
            os.makedirs(os.path.dirname(periodic_output), exist_ok=True)
            runner.save(periodic_output)
            print(f"[distill] saved checkpoint: {periodic_output}", flush=True)
        with torch.no_grad():
            obs, _, done, _ = vec_env.step(student.detach())
            if done.any():
                reset_observations = vec_env.get_observations()
                obs = reset_observations[0] if isinstance(reset_observations, tuple) else reset_observations
                obs = obs.to(raw_env.device)
        if iteration % 100 == 0:
            print(f"[distill] iteration={iteration} loss={loss.item():.6f}", flush=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    runner.save(output)
    print(f"[distill] saved final student checkpoint: {output}")
    print(f"[distill] saved best student checkpoint: {best_output} (loss={best_loss:.6f})")
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
