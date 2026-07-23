"""MDP terms for imitation-guided velocity locomotion."""

# Reuse the ordinary locomotion action, event, observation, reward, and
# termination terms.  Existing source files are imported, not copied or edited.
from booster_train.tasks.manager_based.locomotion.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403

