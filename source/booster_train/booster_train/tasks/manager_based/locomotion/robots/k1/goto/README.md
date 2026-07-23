# Booster K1 GoTo

This task implements a single recurrent, goal-conditioned policy. It consumes K1 proprioception and
`[dx_body, dy_body, sin(dyaw), cos(dyaw)]`, and directly emits residual position targets for all 12 actuated leg
joints. There is no velocity command or high-level controller. Physics runs at 200 Hz and the policy at 50 Hz,
matching both the paper and this repository's existing K1 locomotion controller.

The paper defaults are `radius=1 m`, `inertia=1`, and constellation exponent weight `0.2`. They are exposed in
`RewardsCfg`; the radius is documentation/tuning metadata while the analytic implementation uses the independently
configurable inertia. K1 is smaller than Digit, so these parameters should be tuned rather than treated as universal.
The fall threshold is `0.60 * 0.57 m`, based on K1's configured nominal root height instead of Digit's 0.4 m.

RSL-RL 2.3.1's `ActorCriticRecurrent` resets hidden/cell state from the environment `done` mask. A goal resample does
not create a done, so memory persists. Runner checkpoints contain policy, optimizer, iteration, and empirical
normalizer state; inference must call `policy.reset(dones)` when managing the actor module directly. Isaac Lab's
recurrent exporter represents LSTM hidden and cell state in recurrent TorchScript/ONNX exports. Previous action is
part of environment state and is reset by the action manager.

`symmetry.py` provides a K1 reflection resolved from articulation joint names at runtime: pitch joints keep sign,
while roll/yaw joints and lateral/yaw goals change sign. Mirror loss is disabled in the default recurrent config
because RSL-RL 2.3.1 does not augment recurrent masks/hidden states together with sequences. Enabling it safely
requires a recurrent-aware PPO symmetry patch; the default LSTM task remains runnable without that patch.

Mechanical energy evaluation integrates absolute joint power, `sum(abs(tau*qdot))*dt`; this is a stable mechanical
effort measure, not signed regenerative energy. The evaluator counts swing-to-contact transitions. Contact history
and the 50 Hz sampling act as debounce; increase the sensor/contact hold requirement if a noisy asset chatters.

## Commands

```powershell
python scripts/list_envs.py
python scripts/rsl_rl/train.py --task Booster-K1-GoTo-Smoke-v0 --headless --max_iterations 5
python scripts/rsl_rl/train.py --task Booster-K1-GoTo-v0 --headless
python scripts/rsl_rl/play.py --task Booster-K1-GoTo-v0-Play --checkpoint <checkpoint>
python scripts/rsl_rl/evaluate_goto.py --checkpoint <checkpoint> --headless --output goto_eval.csv
python scripts/rsl_rl/play.py --task Booster-K1-GoTo-v0-Play --checkpoint <checkpoint> --headless
```

The final `play.py` command exports recurrent TorchScript and ONNX under the checkpoint run's `exported` directory.
Use the smoke task for wiring checks only; it has 64 environments and five PPO iterations. Full training uses 4096.
