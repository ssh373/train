"""RSL-RL PPO configuration for imitation-guided K1 locomotion."""

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


# Command/reward schedules convert environment steps back to PPO iterations.
# Keeping this in one task-local constant prevents those schedules drifting from
# the rollout length when the PPO configuration is tuned.
PPO_STEPS_PER_ITERATION = 24


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    """Keep the proven locomotion PPO defaults and isolate experiment logs."""

    num_steps_per_env = PPO_STEPS_PER_ITERATION
    max_iterations = 30000
    save_interval = 250
    experiment_name = "k1_imitation_locomotion"
    logger = "wandb"
