# Frozen policy assets

- `adjust_teacher.pt`: supplied `k1_adjust_001_2026-08-19_00-50-32.pt`
- `kick_teacher.pt`: supplied `k1_kick_001_2026-08-19_01-35-17.pt`

Both are frozen TorchScript policies with input `(N, 49)` and output `(N, 12)`.
Only the unified student and its adjust-to-kick transition are trained.

Environment overrides are `ADJUST_KICK_ADJUST_TEACHER_JIT` and
`ADJUST_KICK_KICK_TEACHER_JIT`.
