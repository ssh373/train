# Booster K1 GoTo

This task implements a single recurrent, goal-conditioned policy. It consumes K1 proprioception and
`[dx_body, dy_body, sin(dyaw), cos(dyaw)]`, and directly emits residual position targets for all 12 actuated leg
joints. There is no velocity command or high-level controller. Physics runs at 200 Hz and the policy at 50 Hz,
matching both the paper and this repository's existing K1 locomotion controller.

Episodes last 30 seconds while pose goals are still resampled every 4--8 seconds. This exposes the recurrent policy
to several goal transitions and disturbance/recovery cycles without resetting its LSTM state between goals.

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
while roll/yaw joints and lateral/yaw goals change sign. `recurrent_symmetry.py` applies it as rollout-time data
augmentation: every real trajectory is interleaved with a virtual mirrored trajectory that owns its own LSTM
hidden/cell state and done mask. This avoids RSL-RL 2.3.1's update-time recurrent shape mismatch. It is symmetric
data augmentation rather than an additional update-time mirror-loss term.

Training resets intentionally include small non-zero joint/root velocities and roll/pitch offsets, and periodic
velocity pushes exercise recovery from moving states such as a kick-to-walk handoff. The gait-style terms maintain
an 0.18 m nominal lateral foot gap, strongly penalize crossing below 0.10 m, reward roughly 4.5 cm swing clearance,
and weakly regularize left/right contact-time balance. These are regularizers rather than hard references so the
policy can still use asymmetric steps for lateral and turning goals.

Robustness training also applies a sustained random trunk disturbance: after four undisturbed seconds, each
environment receives an independently directed 5--15 N horizontal force and up to 1 N m torque for one second.
The cycle repeats every five seconds and is disabled in the Play configuration.

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
