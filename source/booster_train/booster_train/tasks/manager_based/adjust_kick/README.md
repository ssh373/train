# K1 unified adjust -> kick policy

`Booster-K1-Adjust-Kick_001-v0` combines the supplied, already-trained K1
adjust and kick policies into one 49-observation, 12-action student actor. The
old policies are loaded as frozen TorchScript teachers and are never optimized.

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

The controller has three internal training phases:

1. **Adjust:** execute and imitate `adjust_teacher.pt` while orbiting the ball.
2. **Handoff:** after the behind-ball ready gate, blend to the kick teacher for
   `0.20 s` with the smoothstep bridge.
3. **Kick:** execute and imitate `kick_teacher.pt` through kick and recovery.

Both teachers keep independent previous-action histories. Outside the 0.20 s
bridge, the composite target is exactly one of the two frozen policies. The
student's imitation loss is measured in their original normalized action
space. The handoff is short because the objective is minimum time from an
arbitrary approach state to a valid kick while avoiding a hard joint-target
jump.

The attached source configs used different URDF filenames (`K1_jy_locomotion`
for adjust and `K1_locomotion` for kick). Their joint order, 12-D action scale,
default leg pose, and 49-D actor contract match. The unified task intentionally
uses the kick run's `K1_locomotion.urdf` and ball material because foot-ball
contact is the less forgiving part; preservation must therefore be accepted by
the measured 99% metric and Play evaluation rather than assumed from filenames.

The nominal robot-to-ball distance is `0.30 m`. Before kick, distances outside
`0.20--0.40 m` receive an explicit penalty. The kick gate requires the desired
behind-ball pose within `0.08 m`, heading within `30 deg`, a nearly stationary
ball, and no more than `0.04 m` ball displacement.

The reset matches the supplied teacher geometry: the ball is in front of the
robot at `0.20--0.35 m` with lateral offset `+/-0.15 m`. The requested target
direction follows the adjust curriculum from `30--60`, `45--90`, `90--135`,
`135--180`, and finally `0--180` degrees of magnitude with a random left/right
sign, i.e. the full 360-degree task. The `+/-30 deg` value is only the final
behind-ball heading gate that hands control to the kick teacher. Behind-
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

## Recommended training sequence

Run the dedicated DAgger-style transition distillation first. It keeps frozen
teacher control enabled, learns one student on the visited states, reports the
fraction of non-transition samples within the requested preservation tolerance,
and exports a single TorchScript actor. The student actor and empirical
normalizer are warm-started exactly from `adjust_teacher.pt`; kick/transition
behavior is then added without beginning from a random network.

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
