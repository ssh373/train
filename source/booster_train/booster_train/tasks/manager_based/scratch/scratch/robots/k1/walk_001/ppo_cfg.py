from isaaclab.utils import configclass
from booster_train.tasks.manager_based.scratch.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 30000
    experiment_name = "JY_walk_001"
