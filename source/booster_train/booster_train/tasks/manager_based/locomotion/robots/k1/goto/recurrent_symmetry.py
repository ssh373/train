"""Recurrent-safe left/right data augmentation for RSL-RL 2.3.1 PPO.

RSL-RL 2.3.1's update-time symmetry helper concatenates the time dimension of
recurrent mini-batches without duplicating masks or LSTM states.  This PPO
variant instead creates a virtual mirrored environment for every real one at
rollout time.  The two trajectories therefore maintain independent recurrent
states and share rewards/dones under the task's left/right symmetry.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from rsl_rl.algorithms import PPO

from .symmetry import mirror_actions, mirror_observations


_ACTOR_OBSERVATION_SLICES = (
    ("base_ang_vel", 0, 3),
    ("projected_gravity", 3, 6),
    ("joint_pos", 6, 18),
    ("joint_vel", 18, 30),
    ("previous_action", 30, 42),
    ("goal", 42, 46),
)


def _check_actor_observations(observations: torch.Tensor) -> None:
    """Fail with the names of actor observation terms containing NaN/Inf."""
    invalid = ~torch.isfinite(observations)
    if not invalid.any():
        return
    invalid_columns = invalid.reshape(-1, observations.shape[-1]).any(dim=0)
    terms = [name for name, start, end in _ACTOR_OBSERVATION_SLICES if invalid_columns[start:end].any()]
    columns = torch.where(invalid_columns)[0].tolist()
    raise FloatingPointError(
        "GoTo policy observations contain NaN/Inf before the LSTM: "
        f"term(s)={terms or ['unknown-layout']}, column(s)={columns}."
    )


def _interleave(original: torch.Tensor, mirrored: torch.Tensor) -> torch.Tensor:
    """Return ``[original_0, mirror_0, original_1, mirror_1, ...]``."""
    return torch.stack((original, mirrored), dim=1).flatten(0, 1)


def _install_positive_std(policy, init_noise_std: float) -> None:
    """Keep this task's learned Normal scale strictly positive.

    RSL-RL 2.3.1's recurrent actor stores the Normal scale directly in
    ``policy.std``.  Since that parameter is unconstrained, a PPO optimizer
    step can make it negative and the following mini-batch then fails in
    ``torch.normal``.  Reinterpret the same checkpoint-compatible parameter as
    an unconstrained value and map it through softplus whenever GoTo creates a
    distribution.
    """
    if not hasattr(policy, "std"):
        raise TypeError("GoTo positive-std guard requires an RSL-RL policy.std parameter")
    if init_noise_std <= 0.0:
        raise ValueError("init_noise_std must be positive")

    # softplus(raw_std) == init_noise_std at the start of a fresh run.  The
    # parameter remains named ``std``, so GoTo checkpoints remain loadable.
    raw_initial_std = math.log(math.expm1(init_noise_std))
    with torch.no_grad():
        policy.std.fill_(raw_initial_std)

    # Device-side counters avoid synchronizing the GPU on every mini-batch.
    # RecurrentSymmetryPPO reads them only once after each PPO iteration.
    policy._goto_invalid_std_grad_events = torch.zeros((), dtype=torch.long, device=policy.std.device)
    policy._goto_invalid_std_grad_values = torch.zeros((), dtype=torch.long, device=policy.std.device)
    policy._goto_std_optimizer_repairs = torch.zeros((), dtype=torch.long, device=policy.std.device)

    # A non-finite std gradient makes Adam's moving averages permanently NaN;
    # global norm clipping cannot repair that because the norm is NaN too.
    # Drop only the invalid entries while leaving all finite exploration
    # gradients untouched.
    def sanitize_std_gradient(gradient: torch.Tensor) -> torch.Tensor:
        invalid = ~torch.isfinite(gradient)
        policy._goto_invalid_std_grad_events.add_(invalid.any())
        policy._goto_invalid_std_grad_values.add_(invalid.sum())
        return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    policy.std.register_hook(sanitize_std_gradient)

    def update_distribution_with_positive_std(self, observations):
        if not torch.isfinite(observations).all():
            memory_parameters_are_finite = all(
                torch.isfinite(parameter).all() for parameter in self.memory_a.parameters())
            hidden_states = self.memory_a.hidden_states
            if hidden_states is None:
                memory_state_is_finite = True
            else:
                states = hidden_states if isinstance(hidden_states, tuple) else (hidden_states,)
                memory_state_is_finite = all(torch.isfinite(state).all() for state in states)
            raise FloatingPointError(
                "GoTo LSTM produced NaN/Inf actor features: "
                f"memory_parameters_finite={memory_parameters_are_finite}, "
                f"hidden_cell_state_finite={memory_state_is_finite}. This is not an std failure."
            )
        mean = self.actor(observations)
        if not torch.isfinite(mean).all():
            actor_parameters_are_finite = all(torch.isfinite(parameter).all() for parameter in self.actor.parameters())
            source = "finite actor parameters produced overflow" if actor_parameters_are_finite else "actor parameters became NaN/Inf"
            raise FloatingPointError(f"GoTo actor mean became NaN/Inf: {source}. This is not an std failure.")
        if not torch.isfinite(self.std).all():
            raise FloatingPointError(
                "GoTo policy raw std became NaN/Inf. Start a fresh run; "
                "do not resume a checkpoint produced before the std-gradient guard."
            )
        scale = F.softplus(self.std).clamp(min=1.0e-4, max=5.0)
        self.distribution = Normal(mean, scale.expand_as(mean))

    policy.update_distribution = types.MethodType(update_distribution_with_positive_std, policy)


def _install_std_optimizer_guard(policy, optimizer) -> None:
    """Repair only GoTo's std parameter/state immediately after an Adam step."""
    # These raw bounds map through softplus to approximately [1e-4, 5.0].
    raw_min = math.log(math.expm1(1.0e-4))
    raw_max = math.log(math.expm1(5.0))
    last_finite_std = policy.std.detach().clone()

    def guard_std_after_step(_optimizer, _args, _kwargs):
        nonlocal last_finite_std
        with torch.no_grad():
            invalid = ~torch.isfinite(policy.std)
            if invalid.any():
                policy.std[invalid] = last_finite_std[invalid]
                policy._goto_std_optimizer_repairs.add_(invalid.sum())

                # Adam's moment buffers can otherwise write NaN back on every
                # following step, even after the parameter itself is repaired.
                for state_value in optimizer.state.get(policy.std, {}).values():
                    if torch.is_tensor(state_value) and state_value.shape == policy.std.shape:
                        state_value[invalid] = 0.0

            policy.std.clamp_(min=raw_min, max=raw_max)
            last_finite_std.copy_(policy.std)

    optimizer.register_step_post_hook(guard_std_after_step)


class RecurrentSymmetryPPO(PPO):
    """PPO with paired mirrored rollout trajectories for recurrent policies."""

    def __init__(self, policy, *args, symmetry_cfg=None, **kwargs):
        if symmetry_cfg is None or "_env" not in symmetry_cfg:
            raise ValueError("RecurrentSymmetryPPO requires symmetry_cfg with the wrapped environment")
        self.symmetry_env = symmetry_cfg["_env"]
        init_noise_std = float(policy.std.detach().mean().item())
        _install_positive_std(policy, init_noise_std)
        # The stock update-time symmetry path is deliberately disabled.  Its
        # recurrent masks/hidden states are incompatible with augmentation.
        super().__init__(policy, *args, symmetry_cfg=None, **kwargs)
        _install_std_optimizer_guard(self.policy, self.optimizer)
        self._last_invalid_std_grad_events = 0
        self._last_std_optimizer_repairs = 0
        if not self.policy.is_recurrent:
            raise ValueError("RecurrentSymmetryPPO is intended for a recurrent policy")
        print("[INFO] GoTo policy std uses a strictly-positive softplus parameterization.")
        print("[INFO] Recurrent symmetry enabled: one mirrored LSTM trajectory per real environment.")

    def update(self):
        loss_dict = super().update()
        total_events = int(self.policy._goto_invalid_std_grad_events.item())
        if total_events != self._last_invalid_std_grad_events:
            new_events = total_events - self._last_invalid_std_grad_events
            total_values = int(self.policy._goto_invalid_std_grad_values.item())
            print(
                "[WARNING] GoTo discarded non-finite std gradients: "
                f"{new_events} update(s) this iteration, {total_events} total "
                f"({total_values} scalar value(s))."
            )
            self._last_invalid_std_grad_events = total_events
        total_repairs = int(self.policy._goto_std_optimizer_repairs.item())
        if total_repairs != self._last_std_optimizer_repairs:
            new_repairs = total_repairs - self._last_std_optimizer_repairs
            print(
                "[WARNING] GoTo repaired non-finite std after Adam step: "
                f"{new_repairs} scalar value(s) this iteration, {total_repairs} total."
            )
            self._last_std_optimizer_repairs = total_repairs
        return loss_dict

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        # One virtual mirrored environment is stored beside every real one.
        super().init_storage(
            training_type,
            2 * num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
        )

    def _augment_observations(self, obs: torch.Tensor, obs_type: str) -> torch.Tensor:
        mirrored = mirror_observations(obs, self.symmetry_env, obs_type)
        return _interleave(obs, mirrored)

    def act(self, obs, critic_obs):
        _check_actor_observations(obs)
        actor_obs = self._augment_observations(obs, "policy")
        critic_obs_aug = self._augment_observations(critic_obs, "critic")
        augmented_actions = super().act(actor_obs, critic_obs_aug)

        # A mirrored transition must contain mirror(action_original), rather
        # than another independent sample from the mirrored distribution.
        original_actions = augmented_actions[0::2]
        paired_actions = augmented_actions.clone()
        paired_actions[1::2] = mirror_actions(original_actions, self.symmetry_env)
        self.transition.actions = paired_actions.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(paired_actions).detach()
        return original_actions

    def process_env_step(self, rewards, dones, infos):
        paired_rewards = _interleave(rewards, rewards)
        paired_dones = _interleave(dones, dones)
        paired_infos = infos.copy()
        if "time_outs" in infos:
            paired_infos["time_outs"] = _interleave(infos["time_outs"], infos["time_outs"])
        super().process_env_step(paired_rewards, paired_dones, paired_infos)

    def compute_returns(self, last_critic_obs):
        super().compute_returns(self._augment_observations(last_critic_obs, "critic"))
