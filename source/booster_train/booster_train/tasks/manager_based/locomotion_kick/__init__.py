"""Conditional locomotion-to-kick training tasks."""
from __future__ import annotations

import os
import sys


def _consume_teacher_args() -> None:
    """Handle teacher paths locally without changing the shared train script."""
    args = sys.argv[1:]
    kept = [sys.argv[0]]
    index = 0
    while index < len(args):
        option = args[index]
        if option in ("--walk_model", "--kick_model"):
            if index + 1 >= len(args):
                raise ValueError(f"{option} requires a TorchScript model path")
            variable = "VELOCITY_TEACHER_JIT" if option == "--walk_model" else "KICK_TEACHER_JIT"
            os.environ[variable] = os.path.abspath(os.path.expanduser(args[index + 1]))
            index += 2
            continue
        kept.append(option)
        index += 1
    sys.argv[:] = kept


_consume_teacher_args()
