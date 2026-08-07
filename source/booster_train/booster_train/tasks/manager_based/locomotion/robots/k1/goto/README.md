# Booster K1 GoTo

This task implements a single feed-forward, goal-conditioned policy. It consumes K1 proprioception, previous action, and
`[dx_body, dy_body, sin(dyaw), cos(dyaw)]`, and directly emits residual position targets for all 12 actuated leg
joints. There is no velocity command or high-level controller. Physics runs at 200 Hz and the policy at 50 Hz,
matching both the paper and this repository's existing K1 locomotion controller.

Episodes last 30 seconds while pose goals are still resampled every 4--8 seconds. The policy reacts to each current
goal without recurrent hidden/cell state; joint velocity, body angular velocity, and previous action provide motion context.

The paper defaults are `radius=1 m`, `inertia=1`, and constellation exponent weight `0.2`. They are exposed in
`RewardsCfg`; the radius is documentation/tuning metadata while the analytic implementation uses the independently
configurable inertia. K1 is smaller than Digit, so these parameters should be tuned rather than treated as universal.
The fall threshold is `0.60 * 0.57 m`, based on K1's configured nominal root height instead of Digit's 0.4 m.

`symmetry.py` provides a K1 reflection resolved from articulation joint names at runtime: pitch joints keep sign,
while roll/yaw joints and lateral/yaw goals change sign. The standard feed-forward PPO applies this reflection as
update-time data augmentation. Action standard deviation uses `exp(log_std)`, so it remains positive by construction.

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
python scripts/rsl_rl/play.py --task Booster-K1-GoTo-AStar-v0-Play --checkpoint <checkpoint> --num_envs 1
python scripts/rsl_rl/train.py --task Booster-K1-GoTo-Dynamic-v0 --headless
python scripts/rsl_rl/play.py --task Booster-K1-GoTo-Dynamic-v0-Play --checkpoint <checkpoint> --num_envs 1

# Dynamic curriculum (30K):
#   0-5K:   static/moving/A* = 70/20/10, no jitter
#   5-12K:  40/35/25, moving goals up to 0.6 m/s, 1 cm jitter
#   12-22K: 20/45/35, moving goals up to 1.0 m/s, realistic A* replans
#   22-30K: 10/55/35, moving goals up to 1.5 m/s, rare one-frame flicker
python scripts/rsl_rl/evaluate_goto.py --checkpoint <checkpoint> --headless --output goto_eval.csv
python scripts/rsl_rl/play.py --task Booster-K1-GoTo-v0-Play --checkpoint <checkpoint> --headless
```

The final `play.py` command exports feed-forward TorchScript and ONNX under the checkpoint run's `exported` directory.
Use the smoke task for wiring checks only; it has 64 environments and five PPO iterations. Full training uses 4096.

## Arrival and changing-goal fine-tuning

Start with the independent phase-A task. It always samples stationary pose
goals, so stopping and resuming the run cannot advance it into dynamic-goal
training:

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-GoTo-PhaseA-v0 \
  --resume --reset_optimizer \
  --load_run <existing-k1_goto-run> \
  --checkpoint <model_xxx.pt> \
  --max_iterations 5000 \
  --run_name arrival_phase_a
```

Inspect a saved phase-A checkpoint with:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-GoTo-PhaseA-v0-Play \
  --checkpoint <phase-a-checkpoint> --num_envs 1
```

`Booster-K1-GoTo-FineTune-v0` keeps the 46-observation/12-action policy contract,
adds smooth nominal-pose, zero-action, and stillness costs near stationary goals,
and samples stops plus pre-arrival replans. Start from a validated GoTo checkpoint
without restoring its optimizer, so the task-local fine-tuning learning rate
(`1e-4`) is used. The flag changes behavior only when explicitly supplied:

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-GoTo-FineTune-v0 \
  --resume --reset_optimizer \
  --load_run <existing-k1_goto-run> \
  --checkpoint <model_xxx.pt> \
  --max_iterations 30000 \
  --run_name arrival_dynamic_ft
```

Evaluate/export with the matching play task:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-GoTo-FineTune-v0-Play \
  --checkpoint <fine-tuned-checkpoint> --num_envs 1
```
