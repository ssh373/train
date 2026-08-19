"""Export frozen adjust/kick plus learned transition as one policy file."""

from __future__ import annotations

import argparse
import os

import torch

try:
    from .transition_policy import FrozenExpertsLearnedTransitionPolicy
except ImportError:
    from transition_policy import FrozenExpertsLearnedTransitionPolicy


parser = argparse.ArgumentParser(description="Export the teacher-preserving transition policy.")
parser.add_argument("--adjust", required=True)
parser.add_argument("--kick", required=True)
parser.add_argument("--transition", required=True, help="50-to-12 transition actor JIT")
parser.add_argument("--output", default="logs/rsl_rl/k1_adjust_kick_transition_001/final/k1_adjust_kick_transition.pt")
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--transition_duration", type=float, default=0.06,
    help="Teacher handoff duration in seconds; use 0 for an immediate kick.",
)
parser.add_argument(
    "--transition_residual_scale", type=float, default=0.03,
    help="Bounded learned residual scale; use 0 for frozen-teacher direct switching.",
)
args = parser.parse_args()


def main() -> None:
    device = torch.device(args.device)
    adjust = torch.jit.load(os.path.abspath(args.adjust), map_location=device).eval()
    kick = torch.jit.load(os.path.abspath(args.kick), map_location=device).eval()
    transition = torch.jit.load(os.path.abspath(args.transition), map_location=device).eval()
    with torch.inference_mode():
        if tuple(adjust(torch.zeros(1, 49, device=device)).shape) != (1, 12):
            raise ValueError("adjust teacher must map (N,49) to (N,12)")
        if tuple(kick(torch.zeros(1, 49, device=device)).shape) != (1, 12):
            raise ValueError("kick teacher must map (N,49) to (N,12)")
        if tuple(transition(torch.zeros(1, 50, device=device)).shape) != (1, 12):
            raise ValueError("transition actor must map (N,50) to (N,12)")
    if args.transition_duration < 0.0:
        raise ValueError("--transition_duration must be non-negative")
    if args.transition_residual_scale < 0.0:
        raise ValueError("--transition_residual_scale must be non-negative")
    policy = FrozenExpertsLearnedTransitionPolicy(
        adjust,
        kick,
        transition,
        transition_duration_s=args.transition_duration,
        transition_residual_scale=args.transition_residual_scale,
    ).to(device)
    scripted = torch.jit.script(policy)
    scripted(torch.zeros(4, 49, device=device))
    scripted.reset(torch.ones(4, dtype=torch.bool, device=device))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    scripted.save(output)
    print(f"[transition] saved one policy with frozen teachers + learned handoff: {output}")


if __name__ == "__main__":
    main()
