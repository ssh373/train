# K1 contact-free target alignment

This package trains only the pre-kick alignment phase.  It is independent of
the existing end-to-end `adjust_kick` task and uses the same 12-joint position
action contract as `kick_001`.

```powershell
python scripts/rsl_rl/train.py --task Booster-K1-Adjust_001-v0 --headless
python scripts/rsl_rl/play.py --task Booster-K1-Adjust_001-Play-v0 --num_envs 1
```

The ball is always reset at positive robot-frame x in the fixed
`0.10--0.35 m` range.  The actual kick target is sampled on a curriculum from
`+/-15` degrees to the full `+/-180` degree range.  The desired robot state is
evaluated as a line segment behind the ball: the robot must be collinear with
the target and ball, and its ball distance must be within `0.15--0.35 m`.
No explicit alignment point is passed to the actor; the environment computes
the line/band error from the ball and target direction.

The angle curriculum starts at ±30° and expands to ±60°, ±120°, and finally
±180° at PPO iterations `2,000`, `5,000`, and `10,000`. With
`num_steps_per_env=24`, these correspond to `common_step_counter` values
`48,000`, `120,000`, and `240,000`. Training is configured for `12,000`
iterations.

The ball is spawned at `x=0.20--0.35 m`; its lateral spawn range is separate
from the target/result-point angle.

The final pose has two separate requirements: the robot root is behind the
ball on the target line, while the robot body faces the ball-to-target
direction within `15°`. Both must remain valid for `1.50 s` before alignment
success.

The alignment reward provides a ball-centered tangent-velocity signal and
prefers an approximately `0.28 m` travel radius, so the robot learns to go
around the ball toward the target line instead of cutting directly through it.
Success must remain valid for `1.50 s` (75 control steps at a `0.02 s` control
period) before the episode is terminated as aligned.

The actor uses the same 49-value observation contract as `kick_001`: robot
proprioception, camera-like ball position, target direction, visibility metadata,
and previous action.  The critic additionally receives exact ball
velocity/displacement, alignment error, and feet state.  A future unified
adjust-to-kick policy can add a phase input to both tasks together.

An episode succeeds only after the line/band error is below `6 cm` for
`120 ms` while ball speed remains below `0.05 m/s` and ball displacement remains
below `2 cm`.  Meaningful ball motion terminates the episode as a touch
failure, while foot proximity receives an additional soft penalty.  The
motion threshold is a simulator/noise tolerance; it is the operational
definition of contact-free behavior for training.
