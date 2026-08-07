# Conditional locomotion + kick

This package trains separate conditional students from interchangeable
locomotion teachers and a shared kick teacher.  Existing locomotion and kick
tasks/checkpoints are not modified.

Shared student PPO settings live in `agents/rsl_rl_ppo_cfg.py`; robot/task
variants only override the experiment name and variant-specific options.

## Implemented task

`Booster-K1-Velocity-Kick-v0` uses the supplied 54-input velocity-locomotion
TorchScript teacher and 49-input kick TorchScript teacher.  The student actor
has 57 observations: the original 49 kick observations plus velocity command
(3), gait phase (2), and WALK/KICK/RECOVERY one-hot mode (3).  Foot contacts
and base linear velocity are deliberately excluded from the actor because the
real K1 interface cannot currently provide them reliably.

The first curriculum uses forward speeds from 0 to 1.0 m/s, a 2.0 Hz gait,
ball spawn 0.8--1.2 m ahead, and target 4 m from the environment origin.  Each
environment samples its WALK-to-KICK threshold uniformly from 0.35--0.50 m
(nominally about 0.4 m), then enters RECOVERY after a valid
foot kick, and finally back to WALK after one second.  This deliberately starts
below the locomotion teacher's full +/-1.5 m/s capability.  Later curricula may
expand to the full 1.5 m/s after transition success is stable.

Teacher outputs are converted to absolute joint targets before comparison:
locomotion uses scale 1.0 and its -0.2 rad ankle default, while kick uses the
per-joint K1 action scale and -0.25 rad ankle default.  The student uses the
kick task's per-joint action scale.

Existing weak randomization is retained: reset pose/velocity/joint noise,
proprioceptive observation noise, intermittent pushes, and ball-camera
noise/dropout.  Strong dynamics and perception disturbances are intentionally
deferred until nominal Isaac Lab and MuJoCo transitions work.

## Planned position variant

`Booster-K1-Position-Kick-v0` will be a separate checkpoint using the same
framework and kick teacher.  Its three command inputs will be robot-frame
target X/Y and target-yaw error; it will not include gait phase, giving 55 actor
observations.  Initial training should sample X/Y goals within +/-2 m and yaw
within +/-60 degrees, then expand only after arrival and kick transitions are
stable.  The position task requires its own source/config and exported JIT
teacher before registration.

## Training

```bash
export VELOCITY_TEACHER_JIT=/absolute/path/to/velocity_teacher.pt
export KICK_TEACHER_JIT=/absolute/path/to/kick_teacher.pt

CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train.py \
  --task Booster-K1-Velocity-Kick-v0 --num_envs 4096
```

This is a new 57-input student, so it must start a new run; neither teacher PPO
checkpoint is resumed.  The selected JIT teachers must include their empirical
normalizers and are frozen during PPO.

Both environment variables are mandatory.  The task refuses to start when a
teacher path is omitted or missing.  The selected velocity teacher must be
54-to-12 and the selected kick teacher must be 49-to-12.
