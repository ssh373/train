# K1 ball kicking

This package ports the direct-joint K1 kicking task from the legacy Isaac Gym
environment into the repository's Isaac Lab manager-based task layout.

Shared RSL-RL actor-critic and PPO settings live under `kick/agents`; each robot
task keeps only its task-specific runner override in its local `ppo_cfg.py`.

## Environments

- `Booster-K1-Kick_001-v0`: 4096-environment training configuration.
- `Booster-K1-Kick_001-Play-v0`: 32-environment play configuration without pushes.

## Run

From an Isaac Lab shell with this extension installed:

```bash
python scripts/rsl_rl/train.py --task Booster-K1-Kick_001-v0 --headless
python scripts/rsl_rl/play.py --task Booster-K1-Kick_001-Play-v0 --num_envs 1
```

The actor observation has 49 values in this order: projected gravity (3), base
angular velocity (3), robot-frame ball XY position (2), robot-frame kick-target XY
position (2), visibility/age/confidence (3), 12 joint position errors, 12 joint
velocities, and 12 previous actions. Ball velocity is intentionally excluded from
the actor input; it remains available to rewards, terminations, and the critic. The
critic additionally receives uncorrupted
robot, ball, foot, and joint state.

The actor sees a camera-like ball observation with distance-dependent position
noise and brief 50--250 ms dropouts. During dropout, the last detected XY is held,
visibility and confidence become zero, and observation age increases up to 0.3 s.
Rewards and the critic continue to use exact simulator state.

The kicking foot is not fixed and is selected from kick geometry. If the target is
left of the ball in the robot frame, the right foot is preferred; if it is right of
the ball, the left foot is preferred. Within a 3 cm lateral deadband either foot is
accepted. This supports inside-foot directional kicks instead of simply matching
the foot to the side on which the ball spawned.

The selected side is latched at reset. At the instant ball speed rises sharply,
the closest foot is treated as the foot that kicked: a correct-foot event receives
`+8`, a wrong-foot event receives `-8`, and approaching a stationary ball with the
non-selected foot receives an additional proximity penalty.

RSL-RL data augmentation mirrors every policy/critic sample and action across the
robot sagittal plane. Ball and target y coordinates, vector signs, feet, and all
left/right leg joints are reflected consistently, preventing a one-foot policy from
receiving more training data than its mirrored counterpart.

Inside-foot contact is encouraged geometrically. The left foot's medial face is
modeled by local `-y` and the right foot's by local `+y`. At a detected kick event,
the reward checks that the ball lies on the medial side, the medial normal faces the
target, and foot velocity points toward the target. High-quality inside contact gets
up to `+8`; a clear non-inside event receives down to `-8`.

Before contact, dense shaping favors a stable base height near 0.53 m and opening
the kicking hip until the medial foot normal points within 10 degrees of the target.
Hip-yaw action scale is 0.16 rad. Excessive swing speed away from the ball,
vertical lift speed, and post-kick landing speed are penalized, while a higher
near-contact allowance preserves an effective pass.

The episode is phase-aware. Before contact, an upright and quiet base is rewarded.
After the ball exceeds 0.2 m/s, the immediate recovery reward favors low body/joint
velocity and small tilt while allowing any stable bracing stance; this initial phase
does not force joints back to the default angles. A successful kick is not allowed
to terminate immediately: the robot must recover for at least 0.8 s and satisfy
base-speed and tilt limits, so the policy learns a stable follow-through and landing.

Recovery then transitions to locomotion readiness in two phases. For the first
0.5 s after a kick, any stable bracing posture is accepted. After 0.5 s,
`walk_ready_after_kick` softly rewards returning toward the robot's default joint
pose with low joint velocity. Successful handoff additionally requires mean joint
deviation below 0.35 rad, keeping the terminal state within a walk-policy-friendly
region without disrupting the immediate kick follow-through.

Actions use K1's torque-aware joint-specific scales (roughly 0.12--0.35 rad), and
the initial policy noise standard deviation is 0.5. These limits prevent the first
iterations from applying +/-1 rad offsets to every leg joint and immediately falling.

The simulated ball uses nominal Size 4 association-football dimensions: 0.103 m
radius (0.206 m diameter) and 0.37 kg mass. Its reset center height is 0.105 m.

Standing rewards are deliberately small. Static foot proximity gives no reward;
only actual closing velocity and step-to-step distance progress are rewarded.
Waiting carries a strong quadratic penalty, and an episode terminates with a
one-step `-25` failure penalty unless a valid foot kick sends the ball within 20
degrees of the target direction within 3.0 s.
This prevents the stable standing pose from becoming more profitable than an
attempted kick.

Each environment places the robot near its origin and samples the ball 0.20--0.35 m
forward and +/-0.15 m laterally in the robot frame. On every reset, the target is
placed 4.0 m from the robot at a uniformly sampled heading from -60 to +60 degrees
relative to its initial forward direction. Success requires entering a 0.25 m target radius
with trajectory-direction cosine above 0.98 and ball speed below 2.5 m/s. Target-distance
reward, lateral-velocity penalty, and overspeed penalty favor accurate placement
over a merely powerful kick. The nominal base-height target is 0.55 m.

No heading-alignment reward forces the torso to face the ball or target. The policy
may use a forward, inside-foot, or side-kick posture as long as the ball reaches the
sampled target accurately and the robot remains stable.
