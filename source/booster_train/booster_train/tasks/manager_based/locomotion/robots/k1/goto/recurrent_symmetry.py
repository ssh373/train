"""Recurrent-safe left/right data augmentation for RSL-RL 2.3.1 PPO.

RSL-RL 2.3.1's update-time symmetry helper concatenates the time dimension of
recurrent mini-batches without duplicating masks or LSTM states.  This PPO
variant instead creates a virtual mirrored environment for every real one at
rollout time.  The two trajectories therefore maintain independent recurrent
states and share rewards/dones under the task's left/right symmetry.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

from .symmetry import mirror_actions, mirror_observations


def _interleave(original: torch.Tensor, mirrored: torch.Tensor) -> torch.Tensor:
    """Return ``[original_0, mirror_0, original_1, mirror_1, ...]``."""
    return torch.stack((original, mirrored), dim=1).flatten(0, 1)


class RecurrentSymmetryPPO(PPO):
    """PPO with paired mirrored rollout trajectories for recurrent policies."""

    def __init__(self, policy, *args, symmetry_cfg=None, **kwargs):
        if symmetry_cfg is None or "_env" not in symmetry_cfg:
            raise ValueError("RecurrentSymmetryPPO requires symmetry_cfg with the wrapped environment")
        if not symmetry_cfg.get("use_data_augmentation", False):
            raise ValueError("RecurrentSymmetryPPO requires use_data_augmentation=True")
        self.symmetry_env = symmetry_cfg["_env"]
        # The stock update-time symmetry path is deliberately disabled.  Its
        # recurrent masks/hidden states are incompatible with augmentation.
        super().__init__(policy, *args, symmetry_cfg=None, **kwargs)
        if not self.policy.is_recurrent:
            raise ValueError("RecurrentSymmetryPPO is intended for a recurrent policy")
        print("[INFO] Recurrent symmetry enabled: one mirrored LSTM trajectory per real environment.")

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
