"""Task-specific PPO configuration."""

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.adjust.agents.rsl_rl_ppo_cfg import AdjustPPORunnerCfg


@configclass
class PPORunnerCfg(AdjustPPORunnerCfg):
    experiment_name = "k1_adjust_001"
