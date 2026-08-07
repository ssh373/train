# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--checkpoint_videos", action="store_true", default=False,
    help="Pause at saved checkpoints and record a fixed-policy evaluation clip in a separate process.",
)
parser.add_argument(
    "--checkpoint_video_length", type=int, default=250,
    help="Length of each fixed-checkpoint evaluation clip (in policy steps).",
)
parser.add_argument(
    "--checkpoint_video_interval", type=int, default=None,
    help="Checkpoint-video interval in learning iterations (defaults to save_interval).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--reset_optimizer", action="store_true", default=False,
    help="Load checkpoint weights without optimizer state so the selected task's learning rate is used.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video and args_cli.checkpoint_videos:
    parser.error("Use either --video (live training video) or --checkpoint_videos (frozen checkpoint evaluation), not both.")
if args_cli.checkpoint_video_length <= 0:
    parser.error("--checkpoint_video_length must be positive.")
if args_cli.checkpoint_video_interval is not None and args_cli.checkpoint_video_interval <= 0:
    parser.error("--checkpoint_video_interval must be positive.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import subprocess
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class CheckpointVideoRunner(OnPolicyRunner):
    """HTWK-style runner: save, then evaluate that frozen checkpoint in a child process."""

    checkpoint_video_enabled = False
    checkpoint_video_interval = 0
    checkpoint_video_length = 250
    checkpoint_video_task = ""
    checkpoint_video_device = "cuda:0"
    checkpoint_video_script = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorded_video_iterations = set()

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        iteration = int(self.current_learning_iteration)
        if (
            not self.checkpoint_video_enabled
            or iteration <= 0
            or self.checkpoint_video_interval <= 0
            or iteration % self.checkpoint_video_interval != 0
            or iteration in self._recorded_video_iterations
        ):
            return

        self._recorded_video_iterations.add(iteration)
        run_dir = os.path.dirname(os.path.abspath(path))
        video_log_dir = os.path.join(run_dir, "video_logs")
        os.makedirs(video_log_dir, exist_ok=True)
        stdout_path = os.path.join(video_log_dir, f"video_iter_{iteration}_stdout.log")
        stderr_path = os.path.join(video_log_dir, f"video_iter_{iteration}_stderr.log")
        cmd = [
            sys.executable,
            self.checkpoint_video_script,
            "--task", self.checkpoint_video_task,
            "--checkpoint", os.path.abspath(path),
            "--num_envs", "1",
            "--headless",
            "--video",
            "--video_length", str(self.checkpoint_video_length),
            "--tensorboard_video",
            "--video_iteration", str(iteration),
            "--device", self.checkpoint_video_device,
        ]
        print(f"[INFO] Recording frozen checkpoint video for iteration {iteration}.")
        print(f"[INFO] Video command: {' '.join(cmd)}")
        try:
            with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(
                stderr_path, "w", encoding="utf-8"
            ) as stderr_file:
                result = subprocess.run(cmd, stdout=stdout_file, stderr=stderr_file, check=False)
            if result.returncode != 0:
                print(
                    f"[WARNING] Checkpoint video process failed with code {result.returncode}. "
                    f"See {stderr_path}",
                    flush=True,
                )
            else:
                print(f"[INFO] Checkpoint video completed for iteration {iteration}.", flush=True)
        except Exception as exc:
            # Diagnostic recording must not destroy a long training run.
            print(f"[WARNING] Could not launch checkpoint video process: {exc}", flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = CheckpointVideoRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    video_interval = (
        args_cli.checkpoint_video_interval
        if args_cli.checkpoint_video_interval is not None
        else agent_cfg.save_interval
    )
    play_task = args_cli.task if args_cli.task.endswith("-Play") else f"{args_cli.task}-Play"
    runner.checkpoint_video_enabled = args_cli.checkpoint_videos
    runner.checkpoint_video_interval = int(video_interval)
    runner.checkpoint_video_length = int(args_cli.checkpoint_video_length)
    runner.checkpoint_video_task = play_task
    runner.checkpoint_video_device = str(agent_cfg.device)
    runner.checkpoint_video_script = os.path.join(os.path.dirname(__file__), "play.py")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
