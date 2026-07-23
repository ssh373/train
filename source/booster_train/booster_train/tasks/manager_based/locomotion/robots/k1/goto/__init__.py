"""Gym registrations for K1 short-range goal locomotion."""

import gymnasium as gym


def _register(task_id: str, env_cfg: str, agent_cfg: str = "PPORunnerCfg"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{agent_cfg}",
        },
    )


_register("Booster-K1-GoTo-v0", "K1GoToEnvCfg")
_register("Booster-K1-GoTo-Smoke-v0", "K1GoToSmokeEnvCfg", "SmokePPORunnerCfg")
_register("Booster-K1-GoTo-Sim2Real-v0", "K1GoToSim2RealEnvCfg")
_register("Booster-K1-GoTo-v0-Play", "K1GoToPlayEnvCfg")
