# K1 unified adjust -> kick policy

## Single learned unified policy (recommended)

The final deployment artifact is one stateful TorchScript file containing one
learned 50-to-12 actor. The two supplied actors are used only as frozen
training references; they are not packaged into the final file. The actor is
trained with an adjust-to-kick phase input and receives stability, contact,
direction, recovery, and teacher-preservation rewards. The exported wrapper
accepts the original 49 deployment values and maintains the phase internally.

This state is necessary because adjust-before-kick is a sequence with
hysteresis. A stateless 49-to-12 MLP can encounter nearly identical physical
observations before and after handoff while requiring different actions, which
causes supervised distillation to average incompatible leg targets.

Train the integrated actor with the normal RSL-RL trainer:

```bash
cd /home/user/train
python scripts/rsl_rl/train.py \
  --task Booster-K1-Adjust-Kick-Unified_001-v0 \
  --num_envs 4096 \
  --max_iterations 12000 \
  --headless \
  --device cuda:0
```

During roll-in, the environment gradually reduces teacher control. The actor
then produces the complete 12-D joint action during adjust, transition, kick,
and recovery. Teacher tracking preserves the source motions while transition
stability and actual command-rate rewards let the actor learn the missing
boundary behavior.

Load a checkpoint to visually evaluate the single actor and export its 50-to-12
actor JIT:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-Adjust-Kick-Unified_001-Play-v0 \
  --checkpoint /home/user/train/logs/rsl_rl/k1_adjust_kick_unified_001/RUN/model_12000.pt \
  --num_envs 16
```

The exported actor is written next to that checkpoint under `exported/`. Wrap
that actor as the final one-file 49-to-12 stateful policy:

```bash
python source/booster_train/booster_train/tasks/manager_based/adjust_kick/export_unified.py \
  --actor /home/user/train/logs/rsl_rl/k1_adjust_kick_unified_001/RUN/exported/k1_adjust_kick_unified_001_RUN.pt \
  --output logs/rsl_rl/k1_adjust_kick_unified_001/unified/k1_adjust_kick_unified.pt \
  --device cpu
```

The result contains no adjust/kick teacher modules. Test the learned one-file
result in Isaac Sim with the task-local player:

```bash
python source/booster_train/booster_train/tasks/manager_based/adjust_kick/play_unified.py \
  --task Booster-K1-Adjust-Kick-Unified_001-Play-v0 \
  --policy logs/rsl_rl/k1_adjust_kick_unified_001/unified/k1_adjust_kick_unified.pt \
  --num_envs 16
```

The attached source YAMLs use different simulator assets: adjust uses
`K1_jy_locomotion.urdf` and kick uses `K1_locomotion.urdf`. The player accepts
`--robot_urdf /absolute/path.urdf` so both can be evaluated explicitly. The
default unified environment keeps the kick asset because foot-ball contact and
post-kick recovery are the more sensitive phases.

At deployment call `forward(obs)` with the shared 49-value observation and
call `reset(mask)` whenever a new BT adjust-kick behavior starts. If control is
handed over from another leg controller, `reset_with_previous_action(mask,
action)` can seed the adjust expert's autoregressive action history.

`Booster-K1-Adjust-Kick-Unified_001-v0` is the single-actor training task.
`Booster-K1-Adjust-Kick_001-v0` remains the shared legacy environment.

The actor input is unchanged from the two source tasks:

1. projected gravity (3)
2. base angular velocity (3)
3. camera-like robot-frame ball XY (2)
4. requested ball launch direction XY, unit length (2)
5. ball visible / age / confidence (3)
6. leg joint positions (12), velocities (12), previous action (12)

The launch direction is therefore the kick-direction input; no target distance
is encoded in the actor observation.

## Behavior and preservation

The learned controller has three runtime phases:

1. **Adjust:** the unified actor reproduces the adjust teacher behavior while
   orbiting the ball.
2. **Handoff:** the same actor learns the 0.20 s stability transition.
3. **Kick:** the unified actor reproduces kick and post-kick recovery.

The one actor receives the shared 49 values plus one explicit phase/progress
value during training. The exported stateful wrapper appends that value
internally, so deployment still receives only the original 49 values.

The attached source configs used different URDF filenames (`K1_jy_locomotion`
for adjust and `K1_locomotion` for kick). Their joint order, 12-D action scale,
default leg pose, and 49-D actor contract match. The unified task intentionally
uses the kick run's `K1_locomotion.urdf` and ball material because foot-ball
contact is the less forgiving part. Both assets must be checked explicitly in
Play; matching filenames or one-step imitation percentages are not sufficient.

When both motions are later trained on the common `K1_locomotion.urdf`, a new
compatible adjust actor can replace `models/adjust_teacher.pt` (or be selected
with `ADJUST_KICK_ADJUST_TEACHER_JIT`). If the kick actor is unchanged and the
49-observation, 12-action, joint order, default pose, and action scales remain
identical, only the adjust teacher needs replacing for the training reference.
The unified actor must then be retrained and re-exported because its
boundary-state distribution changed. If either contract changes, both the
environment and actor input/output contract must be updated.

The nominal robot-to-ball distance is `0.30 m`. The exported unified policy
enters its transition when the desired behind-ball pose is within `0.08 m`, heading is
within `30 deg`, robot-to-ball distance is `0.20--0.40 m`, and the ball is
visible with non-zero confidence. This uses only values available in the
49-value deployment observation and kicks immediately after entering the
requested kickable geometry.

The reset matches the supplied teacher geometry: the ball is in front of the
robot at `0.20--0.35 m` with lateral offset `+/-0.15 m`. The requested target
direction follows the adjust curriculum from `30--60`, `45--90`, `90--135`,
`135--180`, and finally `0--180` degrees of magnitude with a random left/right
sign, i.e. the full 360-degree task. The `+/-30 deg` value is only the final
behind-ball heading gate that hands the unified actor into its kick phase. Behind-
the-robot or `0.5--1.2 m` starts are intentionally excluded because the
supplied adjust teacher was not trained as a camera-search/long-range
approach controller.

## Included teacher files

- `models/adjust_teacher.pt`: SHA256
  `5AA3E51EDD8403A586A9189A25ED238F51D31BE52583582A91CE6019AE9B3391`
- `models/kick_teacher.pt`: SHA256
  `A8D1A44690F937E79788D6273BA79BA1354EC45FAD40B0BFB13A751B034ACEAC`

Optional overrides:

```bash
export ADJUST_KICK_ADJUST_TEACHER_JIT=/absolute/path/to/adjust.pt
export ADJUST_KICK_KICK_TEACHER_JIT=/absolute/path/to/kick.pt
```

Both must map `(N, 49)` to `(N, 12)`; task startup fails on a mismatch.

## Older experimental paths

The legacy transition trainer learns a new stateless actor from teacher-driven
trajectories. Its preservation percentage is a one-step action metric, not a
closed-loop walking, kick, or recovery guarantee. It is not the recommended
integrated-policy path above.

```bash
cd train
python scripts/rsl_rl/train_adjust_kick_transition.py \
  --task Booster-K1-Adjust-Kick_001-v0 \
  --num_envs 4096 \
  --iterations 20000 \
  --target_preservation 0.99 \
  --headless
```

Outputs are written under:

```text
logs/rsl_rl/k1_adjust_kick_001/distilled/
  model_distilled.pt
  model_distilled_best.pt
  exported/k1_adjust_kick_distilled.pt
```

`target_preservation=0.99` is an acceptance metric, not a claim made before
training: by default a sample counts as preserved only when every one of its 12
normalized actions is within `0.05` of the active frozen teacher.

Then optionally PPO fine-tune task success while rolling teacher control from
100% -> 99% -> 90% -> 0%. Put the distilled checkpoint in a run directory if
needed by your RSL-RL checkpoint resolver, then resume with optimizer reset:

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-Adjust-Kick_001-v0 \
  --resume --load_run distilled --checkpoint model_distilled.pt \
  --reset_optimizer --headless
```

Evaluate/export with no teacher control:

```bash
python scripts/rsl_rl/play.py \
  --task Booster-K1-Adjust-Kick_001-Play-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 16
```

The Play task forces teacher roll-in to zero, so a successful result is the
single student policy rather than hidden switching between the two source
files. The adjust-kick Play environment uses the same source-compatible
near-front ball reset as the teachers (`0.20--0.35 m`, lateral offset
`+/-0.15 m`) and samples the final full-360-degree target stage. The kick
handoff still occurs only after the robot is within the `+/-30 deg` heading
gate.
