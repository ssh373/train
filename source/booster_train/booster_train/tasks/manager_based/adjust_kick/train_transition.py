"""DAgger-style distillation for the unified K1 adjust-to-kick actor.

The simulator is always controlled by the frozen composite teacher during this
stage.  The student sees the ordinary 49-D deployment observation and learns
the adjust expert, the 0.20 s blend, and the kick expert as one 49-to-12 model.
The resulting RSL-RL checkpoint can be PPO fine-tuned with the normal train.py
entry point and exported by play.py.

Launch this through ``scripts/rsl_rl/train_adjust_kick_transition.py`` so
``AppLauncher`` runs before importing the ``booster_train`` package.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Distill adjust and kick teachers into one actor.")
parser.add_argument("--task", default="Booster-K1-Adjust-Kick_001-v0")
parser.add_argument("--iterations", type=int, default=20_000)
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--transition_weight", type=float, default=1.0)
parser.add_argument("--target_preservation", type=float, default=0.99)
parser.add_argument("--preservation_tolerance", type=float, default=0.05)
parser.add_argument(
    "--output",
    default="logs/rsl_rl/k1_adjust_kick_001/distilled/model_distilled.pt",
)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.iterations <= 0:
    parser.error("--iterations must be positive")
if not 0.0 < args_cli.target_preservation <= 1.0:
    parser.error("--target_preservation must be in (0, 1]")

sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import booster_train.tasks  # noqa: F401,E402


def _policy_observation(result):
    return result[0] if isinstance(result, tuple) else result


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # DAgger collection stays on the frozen composite trajectory.  PPO later
    # removes roll-in and validates the independently executed student.
    env_cfg.actions.joint_pos.teacher_control_blend = (1.0, 1.0, 1.0, 1.0)
    # The action term reuses the cached camera state but recomputes the policy
    # group to evaluate both experts. Disable ObservationManager's independent
    # additive-noise draw during supervised matching so student and teachers
    # receive the same 49 values. PPO fine-tuning restores configured noise.
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if not hasattr(runner, "logger_type"):
        runner.logger_type = getattr(agent_cfg, "logger", "tensorboard")
    if not hasattr(runner, "disable_logs"):
        runner.disable_logs = True

    policy = (
        runner.alg.policy
        if hasattr(runner.alg, "policy")
        else runner.alg.actor_critic
    )
    actor = policy.actor
    normalizer = getattr(
        policy,
        "actor_obs_normalizer",
        getattr(runner, "obs_normalizer", None),
    )
    raw_env = vec_env.unwrapped
    action_term = raw_env.action_manager.get_term("joint_pos")

    # Start exactly from the adjust expert instead of random weights. The
    # exported teacher contains both its actor and empirical normalizer, and
    # the supplied source config uses this same [512, 256, 128] architecture.
    adjust_export = action_term._adjust_teacher
    try:
        actor.load_state_dict(adjust_export.actor.state_dict(), strict=True)
        if normalizer is not None and hasattr(adjust_export, "normalizer"):
            normalizer.load_state_dict(adjust_export.normalizer.state_dict(), strict=True)
            normalizer.eval()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not warm-start the unified student from adjust_teacher.pt. "
            "Its actor/normalizer architecture must match [512, 256, 128]."
        ) from exc

    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.learning_rate)
    observation = _policy_observation(vec_env.get_observations()).to(raw_env.device)

    output = os.path.abspath(os.path.expanduser(args_cli.output))
    best_output = os.path.splitext(output)[0] + "_best.pt"
    best_loss = float("inf")
    best_preservation = 0.0
    os.makedirs(os.path.dirname(output), exist_ok=True)

    for iteration in range(1, args_cli.iterations + 1):
        actor_input = normalizer(observation) if normalizer is not None else observation
        student_action = actor(actor_input)

        # process_actions records a teacher target for this exact pre-step
        # observation.  Detached student actions ensure simulator stepping does
        # not retain an Isaac graph.
        next_observation, _, done, _ = vec_env.step(student_action.detach())
        teacher_action = action_term.teacher_action.detach().clone()
        transition = action_term.transition_active.detach().clone()
        valid = ~done.bool()
        squared_error = (student_action - teacher_action).square().mean(dim=1)
        sample_weight = 1.0 + transition.float() * (args_cli.transition_weight - 1.0)
        if valid.any():
            loss = (squared_error[valid] * sample_weight[valid]).sum() / sample_weight[valid].sum()
        else:
            loss = squared_error.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            non_transition = valid & ~transition
            if non_transition.any():
                within_tolerance = (
                    (student_action - teacher_action).abs().amax(dim=1)
                    <= args_cli.preservation_tolerance
                )
                preservation = within_tolerance[non_transition].float().mean().item()
            else:
                preservation = 0.0

        if loss.item() < best_loss or (
            preservation >= args_cli.target_preservation
            and best_preservation < args_cli.target_preservation
        ):
            best_loss = min(best_loss, loss.item())
            best_preservation = max(best_preservation, preservation)
            runner.save(best_output)

        if iteration % args_cli.save_interval == 0:
            periodic = os.path.splitext(output)[0] + f"_iter_{iteration}.pt"
            runner.save(periodic)
            print(
                f"[distill] iteration={iteration} loss={loss.item():.6f} "
                f"preservation={preservation:.2%} saved={periodic}",
                flush=True,
            )

        observation = _policy_observation(next_observation).to(raw_env.device)

    runner.save(output)
    export_dir = os.path.join(os.path.dirname(output), "exported")
    os.makedirs(export_dir, exist_ok=True)
    export_policy_as_jit(
        policy,
        normalizer=normalizer,
        path=export_dir,
        filename="k1_adjust_kick_distilled.pt",
    )
    print(f"[distill] final checkpoint: {output}")
    print(f"[distill] best checkpoint: {best_output}")
    print(f"[distill] single TorchScript policy: {export_dir}")
    print(
        f"[distill] best observed preservation={best_preservation:.2%}; "
        f"requested={args_cli.target_preservation:.2%} at max-error "
        f"tolerance {args_cli.preservation_tolerance:.3f}"
    )
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
