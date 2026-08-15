"""MDP terms used by the kicking environments."""

from booster_train.tasks.manager_based.locomotion.mdp import *  # noqa: F401,F403

# Reuse the walk environment's force/torque disturbance implementation.
from .kick_mdp import *  # noqa: F401,F403
