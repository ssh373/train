"""Launch adjust-to-kick transition distillation with Isaac Sim initialized first.

This entry point intentionally executes the task-local trainer by file path.
Running it with ``python -m booster_train.tasks...`` imports
``booster_train.__init__`` before the task can create ``AppLauncher``, which is
too early for Omniverse modules such as ``omni.log``.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    trainer = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "booster_train"
        / "booster_train"
        / "tasks"
        / "manager_based"
        / "adjust_kick"
        / "train_transition.py"
    )
    runpy.run_path(str(trainer), run_name="__main__")
