"""Public MDP namespace used only by the standalone adjust-kick task."""

from isaaclab.envs.mdp import *  # noqa: F401,F403

from .actions import *  # noqa: F401,F403
from .standalone_mdp import *  # noqa: F401,F403
# Keep task-local terms explicit when they shadow or extend Isaac Lab's public
# MDP namespace. This also avoids depending on wildcard-export details.
from .standalone_mdp import feet_slide as feet_slide
