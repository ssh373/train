# K1 imitation-guided locomotion

이 task는 기록 모션을 정확히 재생하는 것이 아니라, 기록 모션을 초기 보행 힌트로 사용하면서 최종적으로
`[vx, vy, wz]` 명령을 잘 추종하는 정책을 PPO로 학습하기 위한 구조다.

## 구조

```text
imitation_locomotion/
├─ agents/
│  └─ rsl_rl_ppo_cfg.py          # PPO rollout/학습 설정
├─ mdp/
│  ├─ commands.py                # 명령 커리큘럼 + reference 선택/진행
│  ├─ curriculums.py             # iteration 기반 스케줄 함수
│  ├─ observations.py            # critic 전용 reference 관측
│  ├─ reference_motion.py        # manifest/CSV 로더와 데이터 정제
│  └─ rewards.py                 # confidence 및 시간 스케줄이 적용된 모방 보상
└─ robots/k1/imitation_walk_001/
   ├─ env_cfg.py                 # K1 scene/observation/reward 연결
   ├─ ppo_cfg.py
   └─ __init__.py                # Gym task 등록
```

등록된 task는 다음과 같다.

- `Booster-K1-Imitation-Locomotion-Flat-v0`: 평지 학습
- `Booster-K1-Imitation-Locomotion-Rough-v0`: 약한 랜덤 지형 학습
- `Booster-K1-Imitation-Locomotion-Play-v0`: 고정 명령으로 확인

## 학습 방식

Actor 입력에는 reference motion이 없다. 최종 actor 입력은 IMU에서 얻을 수 있는 base angular velocity와
projected gravity, 12개 다리 관절 위치/속도, 이전 action, `cmd_vel`뿐이다. 따라서 학습 후 실물에서 NPZ나
reference cursor를 실행할 필요가 없다.

Critic만 `q_ref`, `dq_ref`, reference confidence, clip progress를 추가로 본다. Reference는 reward에도 사용하지만
다음 조건에서 자동으로 약해진다.

- 명령과 기록된 command가 멀 때
- 기록에 없는 2축/3축 조합 명령일 때
- clip 시작/끝 또는 reference 전환 중일 때
- PPO 학습 iteration이 진행되어 imitation curriculum 값이 감소할 때

Command curriculum은 다음 네 단계다.

1. 기록된 정지/앞뒤/좌우/회전 명령 중심
2. 연속적인 단일 축 명령
3. 단일 축과 2축 조합 명령
4. 3축 조합을 포함한 전체 범위

현재 경계는 500, 3,000, 10,000 PPO iteration이다. Imitation scale은 같은 기간에
`1.0 → 0.75 → 0.40 → 0.20`으로 감소한다. 이 값은 reward의 비율이 아니라 모방 reward에 곱하는 계수다.
속도 추종 weight는 선형 4.0, yaw 3.0이고 관절 위치 모방은 2.0, 관절 속도 모방은 0.1이므로 처음부터
명령 추종이 주 목표이며, 학습 후반에는 그 차이가 더 커진다.

## 현재 데이터 처리

기본 데이터는 repository의 `processed_walks_001`, `processed_walks_002`다. 두 세션의 NPZ 구조가 서로 달라
공통으로 신뢰할 수 있는 아래 정보만 사용한다.

- command label: 각 폴더의 `manifest.json`
- 12개 다리 joint position: `csv/*.csv`
- joint velocity: 각 정제된 시퀀스 안에서 50 Hz 중앙 차분으로 다시 계산

Root/body reference는 사용하지 않는다. 첫 clip은 원본 병합 과정에서 실제 첫 명령보다 먼저 command가 붙은
부분이 있어 기본 5.6초를 제거한다. 관절이 한 frame에 0.35 rad보다 크게 튀면 logger discontinuity로 보고
시퀀스를 나누며, 1초보다 짧은 조각은 버린다. Non-zero command인데 관절 활동량이 너무 작은 clip도 버린다.
정지 reference는 정지 구간 전체를 반복하지 않고, 저속 후보 중 실제로 관측된 안정 frame 하나를 선택한다.

새로 정상 전처리한 데이터를 쓸 때는 `CommandsCfg.base_velocity.first_clip_head_trim_s = 0.0`으로 바꿔야 한다.
다른 데이터 경로는 실행 전에 환경 변수로 덮어쓸 수 있다. Windows에서는 여러 경로를 세미콜론으로 구분한다.

```powershell
$env:BOOSTER_WALK_DATASETS="D:\data\walk_a;D:\data\walk_b"
```

## 실행

Isaac Lab Python 환경에서 다음처럼 실행한다.

```bash
python scripts/rsl_rl/train.py --task=Booster-K1-Imitation-Locomotion-Flat-v0 --headless --device=cuda:0
```

학습 결과 확인과 정책 export는 다음과 같다.

```bash
python scripts/rsl_rl/play.py \
  --task=Booster-K1-Imitation-Locomotion-Play-v0 \
  --checkpoint=<CHECKPOINT_PATH>
```

처음에는 Flat task로 보상/데이터 로딩을 확인한 다음 Rough task로 fine-tuning하는 순서를 권장한다.

## Supervised learning과의 구분

현재 CSV에는 로봇이 실제로 수행한 관절 상태 `q`, `dq`는 있지만 teacher가 보낸 policy action 또는
`q_des`가 없다. 관측된 `q(t+1)`을 action label로 간주하면 actuator delay와 PD dynamics가 섞인 잘못된
지도 label이 된다. 따라서 이 구현은 허위 action label을 만든 behavior cloning이 아니라, 기록 관절 상태를
보상 힌트로 쓰는 imitation-guided PPO다.

추후 같은 timestamp의 policy observation, `cmd_vel`, teacher action 또는 `q_des`를 함께 기록하면
`agents/` 아래에 behavior-cloning pretrainer를 추가하고 그 actor checkpoint로 PPO를 시작할 수 있다.

## 참고한 기존 코드와 차이

기존 파일은 수정하지 않았고, 이 폴더 안에만 코드를 추가했다.

- `locomotion/robots/k1/walk_002/env_cfg.py`
  - 참고/상속: K1 scene, 12-leg joint action, contact sensor, reset/push event, termination, 200 Hz simulation과
    decimation 4의 50 Hz policy 주기
  - 이 task에서 변경: direct yaw command, axis-to-compound curriculum, reference-aware critic, 방향 중립적인
    핵심 reward, 안정적인 crouched 초기 자세
- `locomotion/mdp/commands.py`, `locomotion/mdp/rewards.py`
  - 재사용: manager-based command term 형태와 base-frame velocity tracking/stability/contact reward
  - 이 task에서 추가: multi-clip 선택, command confidence, clip crossfade/seam fade, imitation decay
- `locomotion/agents/rsl_rl_ppo_cfg.py`
  - 상속: 검증된 actor/critic network와 PPO 기본 hyperparameter
  - 이 task에서 변경: 별도 experiment 이름, 30,000 iterations, 250-iteration checkpoint 간격
- `beyond_mimic/mdp/commands.py`, `beyond_mimic/mdp/rewards.py`
  - 개념 참고: 이름 기반 reference mapping, motion cursor, exponential imitation error
  - 그대로 쓰지 않은 부분: single NPZ 전용 loader, reset 시 reference state로 robot teleport, full-body/root
    tracking, reference 자체를 actor command로 제공하는 방식

`processed_walks_001/002`와 위 기존 source는 읽기/상속만 하며 이 task 생성 과정에서 수정하지 않는다.

