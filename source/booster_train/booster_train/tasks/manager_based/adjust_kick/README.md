# K1 autonomous adjust kick

`Booster-K1-Adjust-Kick_001-v0` trains the policy used after a BT-controlled
walk handoff: close approach, moving around the ball, precise alignment, kick,
and recovery to a walk-ready stance. There is no walk/kick/phase observation.

During training only, the frozen walk teacher rolls the robot into the handoff
state. The deploy-identical frozen kick teacher also rolls in the original
kick after precise alignment. At deployment the BT keeps using its existing walk policy, then starts
the exported adjust-kick actor at its selected handoff distance (for example
0.75 m). No composite walk/adjust export is required.

## Independence

This task has private copies of its base environment, kick MDP terms, symmetry,
and PPO configuration. It does not import code from `manager_based/kick`, so
future edits to the old kick task cannot alter this task. The frozen walk and
kick model files are treated as read-only policy assets.

The shared walk/adjust nominal ankle-pitch pose is `-0.20 rad`, with no hidden
`0.05 rad` conversion between the frozen walk and learned branch. Final target
success and target visualization use a `0.15 m` radius.

Walk, adjust, kick-teacher conversion, and recovery all use the fixed
`-0.20 rad` ankle-pitch nominal pose. There is no kick-only ankle offset.

## Curriculum

The reset curriculum uses `common_step_counter` and changes only the reset
distribution. No curriculum-stage value is exposed to the actor.

| Stage | Control steps | Ball distance | Ball bearing | Handoff distance | Ball-to-target distance | Target bearing |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0--99,999 | 0.25--0.60 m | -45--45 deg | 0.55--0.65 m | 3.5--4.5 m | -30--30 deg |
| 1 | 100,000--249,999 | 0.40--1.20 m | -120--120 deg | 0.50--0.70 m | 3.5--5.5 m | -90--90 deg |
| 2 | 250,000--499,999 | 0.35--2.00 m | full 360 deg | 0.45--0.75 m | 3.5--6.0 m | full 360 deg |
| 3 | 500,000+ | 0.25--3.00 m | full 360 deg | 0.45--0.80 m | 3.0--7.0 m | full 360 deg |

Stage 0 preserves the existing near-ball kick and learns a small adjustment.
Later stages add side/back approaches and arbitrary target geometry. Play
always uses stage 3. Training roll-in is executed by the frozen teacher. A new
handoff distance is sampled independently for every reset, and the learned
branch takes over when planar robot-to-ball distance crosses it, regardless of
heading. It therefore learns the remaining close
approach, including moving around the ball from the opposite side, precise
alignment, kick, and recovery. This switch is latched for the rest of the
episode.

At handoff, joint targets are blended from the frozen walk teacher to the
student with a `0.50 s` smoothstep transition. Kick-ready gating and its
`2.5 s` no-kick timer remain disabled until this transition is complete.

Kick-teacher roll-in has a separate, shorter curriculum:

| Control steps | Approx. PPO iteration | Teacher blend |
|---:|---:|---:|
| 0--39,999 | 0--1,666 | 1.00 |
| 40,000--79,999 | 1,667--3,333 | 0.67 |
| 80,000--139,999 | 3,334--5,833 | 0.33 |
| 140,000+ | 5,834+ | 0.00 |

The blend is active only after the geometric ready gate and stops after the
kick. The kick imitation reward remains active before contact after direct
roll-in reaches zero, so PPO must execute the kick itself while retaining the
validated shape.

## Reward intent

- Fast approach: position progress `+8`, velocity toward the pre-kick pose `+3`.
- Fast alignment: heading progress `+5`, precise behind-ball pose `+4`.
- Do not touch during adjustment: early ball motion `-80`, early foot proximity `-20`.
- Accurate kick: target accuracy `+20`, immediate direction accuracy `+15`,
  target velocity `+10`, valid foot-contact event `+8`, lateral velocity `-8`.
- Recovery: stable recovery `+4`, walk-ready pose `+2`, both feet grounded `+8`.
- Existing kick preservation: deploy kick teacher tracking `+4`.
- Smoothness and safety terms from the K1 kick task remain active, including
  ball overspeed and action-rate penalties.

Target distance is sampled from the ball, so the far-ball curriculum cannot
accidentally place the target on top of the ball.

The geometric ready gate is position error <= 0.16 m, heading error <= 12 deg,
ball speed <= 0.08 m/s, and ball displacement from reset <= 0.05 m. Kick rewards
and the valid-kick latch are zero before this gate is reached. Ball displacement
beyond 0.03 m remains penalized during adjustment, even after the ball stops.

## Walk handoff

The bundled 54-observation velocity-walk TorchScript teacher is enabled during
training and directly supplies joint targets before the sampled handoff:

`adjust_kick/models/walk_teacher.pt`

To override it with another compatible teacher, set:

```bash
export ADJUST_KICK_WALK_TEACHER_JIT=/absolute/path/to/walk_teacher.pt
```

The teacher is frozen. PPO actions are recorded for imitation learning during
approach, but they are not sent to the robot until the close-adjust switch.
Missing or incompatible teacher files stop environment startup instead of
silently training a different walk.

The 49-observation kick teacher is loaded from:

`adjust_kick/models/kick_teacher.pt`

It can be overridden with `ADJUST_KICK_KICK_TEACHER_JIT`. Put both models at
the task-local relative paths above before starting training; environment
variables are optional overrides.

Deployment sequence:

1. The BT commands the existing walk policy toward the ball/approach area.
2. At the configured robot-to-ball XY distance (0.45--0.80 m is supported by the
   final curriculum), the BT stops walk and starts the adjust-kick actor, with
   no heading condition.
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
