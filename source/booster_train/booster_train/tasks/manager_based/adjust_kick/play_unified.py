"""Compatibility entry point for playing the single learned unified policy."""

from __future__ import annotations

import os
import runpy


runpy.run_path(
    os.path.join(os.path.dirname(__file__), "play_composite.py"),
    run_name="__main__",
)
