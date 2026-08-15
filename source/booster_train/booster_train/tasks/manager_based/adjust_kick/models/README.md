# Adjust-kick teacher models

Place the two TorchScript teacher files in this directory:

- `walk_teacher.pt`: input `(N, 54)`, output `(N, 12)`
- `kick_teacher.pt`: input `(N, 49)`, output `(N, 12)`

The files are loaded through paths relative to the standalone `adjust_kick`
task. They may alternatively be overridden with
`ADJUST_KICK_WALK_TEACHER_JIT` and `ADJUST_KICK_KICK_TEACHER_JIT`.
