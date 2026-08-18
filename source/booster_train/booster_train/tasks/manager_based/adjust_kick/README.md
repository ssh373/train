# K1 autonomous adjust kick

`Booster-K1-Adjust-Kick_001-v0` trains an adjust-first policy: the ball already
starts nearby and in front, then the policy moves around it without contact,
aligns to a 360-degree target, kicks, and recovers. Long-range approach is not
part of this task. There is no walk/kick/phase observation.

The frozen walk teacher is used as the actual controller for the short,
contact-free orbit/side-step alignment around the nearby ball. It hands off
only after the precise behind-ball ready gate. The deploy-identical frozen
kick teacher then preserves the original kick, with a short post-kick teacher
window for stable recovery.

## Independence

This task has private copies of its base environment, kick MDP terms, symmetry,
and PPO configuration. It does not import code from `manager_based/kick`, so
future edits to the old kick task cannot alter this task. The frozen walk and
kick model files are treated as read-only policy assets.

The shared walk/adjust nominal ankle-pitch pose is `-0.20 rad`, with no hidden
`0.05 rad` conversion between the frozen walk and learned branch. Final target
success and target visualization use a `0.15 m` radius. Success also requires
the travelled ball direction to stay within about `5.7 deg` of the initial
ball-to-target vector (`cos > 0.995`). A valid kick must launch the ball above
`0.5 m/s`, but there is no practical speed cap when it reaches the target.

Walk, adjust, kick-teacher conversion, and recovery all use the fixed
`-0.20 rad` ankle-pitch nominal pose. There is no kick-only ankle offset.
The trunk-height target is `0.52 m` during adjustment and through ball contact,
about 3 cm below the upright target. After a valid kick it returns to `0.55 m`
for walk-ready recovery; the fall termination remains at `0.45 m`.

## Curriculum

The reset curriculum uses `common_step_counter` and changes only the reset
distribution. No curriculum-stage value is exposed to the actor.

| Stage | Control steps | Ball distance | Ball bearing | Ball-to-target distance | Target bearing |
|---|---:|---:|---:|---:|---:|
| 0 | 0--99,999 | 0.28--0.35 m | -8--8 deg | 3.5--4.5 m | -30--30 deg |
| 1 | 100,000--249,999 | 0.28--0.35 m | -12--12 deg | 3.5--5.5 m | -90--90 deg |
| 2 | 250,000--499,999 | 0.28--0.35 m | -15--15 deg | 3.5--6.0 m | full 360 deg |
| 3 | 500,000+ | 0.28--0.35 m | -15--15 deg | 3.0--7.0 m | full 360 deg |

The ball remains in the robot's forward sector and inside the kick-distance
band in every stage. There is no long-range forward-approach curriculum. The
target bearing widens to the full 360 degrees, forcing fast contact-free
movement around the ball. The walk teacher controls this short alignment and
hands off over `0.25 s` after the ready gate.

Kick-teacher roll-in has a separate, shorter curriculum:

| Control steps | Approx. PPO iteration | Teacher blend |
|---:|---:|---:|
| 0--199,999 | 0--8,333 | 1.00 |
| 200,000--399,999 | 8,334--16,666 | 0.90 |
| 400,000--649,999 | 16,667--27,083 | 0.70 |
| 650,000+ | 27,084+ | 0.00 |

The blend is active only after the geometric ready gate and remains active for
`0.8 s` after a valid kick so the teacher can guide leg recovery. The kick
imitation reward uses `std=0.25`, allowing small environment-specific changes
such as a deeper knee bend while retaining the validated shape.

## Reward intent

- Fast adjustment around the ball: position progress `+8`; the separate
  forward-approach velocity reward is disabled.
- Fast alignment: heading progress `+5`, precise behind-ball pose `+4`.
- Do not touch during adjustment: early ball motion `-80`, early foot proximity `-20`.
- Accurate kick: target accuracy `+20` (`std=0.25 m`, matching `kick_001`),
  immediate direction accuracy `+15`, target velocity `+10`, direction-gated
  kick speed `+10` up to `3.0 m/s`, valid foot-contact event `+8`, lateral
  velocity `-8`.
- Recovery: stable recovery `+4`, walk-ready pose `+2`, both feet grounded `+8`.
- Existing kick preservation: deploy kick teacher tracking `+4` (`std=0.25`).
- Smoothness and safety terms from the K1 kick task remain active, including
  action-rate penalties. The ball-overspeed penalty is disabled in this task;
  direction and target accuracy are prioritized.

Target distance is sampled from the ball, so the far-ball curriculum cannot
accidentally place the target on top of the ball.

The geometric ready gate is behind-ball position error <= 0.22 m, horizontal
robot-base-to-ball distance 0.10--0.35 m, heading error <= 25 deg, ball speed
<= 0.15 m/s, and ball displacement from reset <= 0.10 m. Kick rewards and the
valid-kick latch are zero before this gate is reached. Ball displacement beyond
0.03 m remains penalized during adjustment, even after the ball stops.

## Walk reference

The 54-observation velocity-walk TorchScript teacher directly controls the
short alignment phase and provides the target used for the smooth handoff:

`adjust_kick/models/walk_teacher.pt`

To override it with another compatible teacher, set:

```bash
export ADJUST_KICK_WALK_TEACHER_JIT=/absolute/path/to/walk_teacher.pt
```

The teacher is frozen. Missing or incompatible teacher files stop environment
startup instead of silently training a different walking style.

The 49-observation kick teacher is loaded from:

`adjust_kick/models/kick_teacher.pt`

It can be overridden with `ADJUST_KICK_KICK_TEACHER_JIT`. Put both models at
the task-local relative paths above before starting training; environment
variables are optional overrides.

Deployment sequence:

1. The BT uses the existing walk policy until the ball is nearby and in front.
2. At approximately `0.30--0.75 m`, the BT stops walk and starts the
   adjust-kick actor.
3. Initialize the actor's previous-action observation from the final walk joint
   target in adjust action coordinates: `(q_target - q_default) / action_scale`.
4. Keep the adjust-kick actor active through kick and recovery.
5. After kick detection and at least 0.8 s of stable recovery, the BT may return
   to the existing walk policy.

For repeated kicks, the BT repeats this same sequence. Walking between balls
always remains the unchanged existing walk policy.

## Train and play

```bash
python scripts/rsl_rl/train.py \
  --task Booster-K1-Adjust-Kick_001-v0 \
  --headless

python scripts/rsl_rl/play.py \
  --task Booster-K1-Adjust-Kick_001-Play-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 1
```

Because the actor and critic observation dimensions match `k1_kick_001`, an
existing kick checkpoint can be used as a warm start. Place or link it under
`logs/rsl_rl/k1_adjust_kick_001/<run>/`, then use `--resume --load_run ...
--checkpoint ... --reset_optimizer`. Resetting the optimizer is recommended
because the horizon and reward structure are substantially different.

```bash
mkdir -p logs/rsl_rl/k1_adjust_kick_001/kick_warmstart
cp logs/rsl_rl/k1_kick_001/<old-run>/model_9999.pt \
  logs/rsl_rl/k1_adjust_kick_001/kick_warmstart/

python scripts/rsl_rl/train.py \
  --task Booster-K1-Adjust-Kick_001-v0 \
  --resume \
  --load_run kick_warmstart \
  --checkpoint model_9999.pt \
  --reset_optimizer \
  --headless
```
