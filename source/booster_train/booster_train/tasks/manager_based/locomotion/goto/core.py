"""Simulator-independent SE(2) goal and constellation utilities.

This module deliberately has no Isaac Lab or torch dependency so the geometry and
sampler can be unit-tested on development machines without Isaac Sim.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_body_goal(robot: tuple[float, float, float], goal: tuple[float, float, float]) -> tuple[float, float, float]:
    dx, dy = goal[0] - robot[0], goal[1] - robot[1]
    c, s = math.cos(robot[2]), math.sin(robot[2])
    return c * dx + s * dy, -s * dx + c * dy, wrap_to_pi(goal[2] - robot[2])


def constellation_distance(dx: float, dy: float, dtheta: float, inertia: float = 1.0) -> float:
    return dx * dx + dy * dy + inertia * 2.0 * (1.0 - math.cos(dtheta))


def explicit_constellation_distance(dx: float, dy: float, dtheta: float, radius: float = 1.0) -> float:
    """Mean squared displacement of a uniformly sampled circular constellation.

    The closed form is dx^2 + dy^2 + radius^2 * 2(1-cos(dtheta)).
    """
    # Four cardinal points have the exact same second moment as a circle.
    total = 0.0
    c, s = math.cos(dtheta), math.sin(dtheta)
    for px, py in ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)):
        qx, qy = dx + c * px - s * py, dy + s * px + c * py
        total += (qx - px) ** 2 + (qy - py) ** 2
    return total / 4.0


@dataclass(frozen=True)
class GoalSamplingConfig:
    x_range: tuple[float, float] = (-2.0, 2.0)
    y_range: tuple[float, float] = (-1.5, 1.5)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)
    probabilities: tuple[float, float, float, float, float] = (0.10, 0.20, 0.20, 0.20, 0.30)


def sample_relative_goal(rng: random.Random, cfg: GoalSamplingConfig) -> tuple[float, float, float, int]:
    """Sample stand, straight, lateral, turn, or translation+rotation (category 0..4)."""
    if len(cfg.probabilities) != 5 or any(p < 0 for p in cfg.probabilities):
        raise ValueError("probabilities must contain five non-negative values")
    total = sum(cfg.probabilities)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"goal category probabilities must sum to one, got {total}")
    u, acc, category = rng.random(), 0.0, 4
    for i, probability in enumerate(cfg.probabilities):
        acc += probability
        if u < acc:
            category = i
            break
    x = rng.uniform(*cfg.x_range)
    y = rng.uniform(*cfg.y_range)
    yaw = rng.uniform(*cfg.yaw_range)
    if category == 0:
        x = y = yaw = 0.0
    elif category == 1:
        y = yaw = 0.0
    elif category == 2:
        x = yaw = 0.0
    elif category == 3:
        x = y = 0.0
    return x, y, yaw, category
