"""Export one learned 50-to-12 actor as a stateful 49-to-12 policy."""

from __future__ import annotations

import argparse
import os

import torch

try:
    from .unified_policy import LearnedUnifiedAdjustKickPolicy
except ImportError:
    from unified_policy import LearnedUnifiedAdjustKickPolicy


parser = argparse.ArgumentParser(description="Export the single learned unified actor.")
parser.add_argument("--actor", required=True, help="50-to-12 actor JIT from RSL-RL play.py")
parser.add_argument(
    "--output",
    default="logs/rsl_rl/k1_adjust_kick_unified_001/unified/k1_adjust_kick_unified.pt",
)
parser.add_argument("--device", default="cpu")
parser.add_argument("--policy_dt", type=float, default=0.02)
parser.add_argument("--transition_duration", type=float, default=0.20)
args = parser.parse_args()


def main() -> None:
    actor_path = os.path.abspath(os.path.expanduser(args.actor))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    device = torch.device(args.device)
    actor = torch.jit.load(actor_path, map_location=device).eval()
    with torch.inference_mode():
        probe = actor(torch.zeros(1, 50, device=device))
    if tuple(probe.shape) != (1, 12):
        raise ValueError(f"unified actor must map (N, 50) to (N, 12); got {tuple(probe.shape)}")

    policy = LearnedUnifiedAdjustKickPolicy(
        actor,
        policy_dt=args.policy_dt,
        transition_duration_s=args.transition_duration,
    ).to(device)
    scripted = torch.jit.script(policy)
    scripted(torch.zeros(4, 49, device=device))
    scripted.reset(torch.ones(4, dtype=torch.bool, device=device))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scripted.save(output_path)
    print(f"[unified] saved one-actor policy: {output_path}")
    print("[unified] runtime contract: forward(obs[N,49]); reset(mask[N])")


if __name__ == "__main__":
    main()
