"""Small curriculum helpers shared by command generation and imitation rewards."""

from __future__ import annotations

from collections.abc import Sequence


def training_iteration(env, steps_per_iteration: int) -> float:
    """Convert Isaac Lab's environment-step counter to an approximate PPO iteration."""

    if steps_per_iteration <= 0:
        raise ValueError("steps_per_iteration must be positive")
    return float(env.common_step_counter) / float(steps_per_iteration)


def stage_index(env, boundaries: Sequence[int], steps_per_iteration: int) -> int:
    """Return the discrete command-curriculum stage."""

    iteration = training_iteration(env, steps_per_iteration)
    return sum(iteration >= boundary for boundary in boundaries)


def piecewise_linear_scale(
    env,
    iteration_knots: Sequence[int],
    values: Sequence[float],
    steps_per_iteration: int,
) -> float:
    """Linearly interpolate a scalar curriculum defined in PPO iterations."""

    if len(iteration_knots) != len(values):
        raise ValueError("iteration_knots and values must have equal lengths")
    if not iteration_knots:
        raise ValueError("At least one curriculum knot is required")
    if any(right <= left for left, right in zip(iteration_knots[:-1], iteration_knots[1:])):
        raise ValueError("iteration_knots must be strictly increasing")

    iteration = training_iteration(env, steps_per_iteration)
    if iteration <= iteration_knots[0]:
        return float(values[0])
    for left, right, left_value, right_value in zip(
        iteration_knots[:-1], iteration_knots[1:], values[:-1], values[1:]
    ):
        if iteration < right:
            alpha = (iteration - left) / float(right - left)
            return float(left_value + alpha * (right_value - left_value))
    return float(values[-1])

