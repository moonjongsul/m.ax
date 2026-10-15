# max_ai — SmolVLA / MVLA rollout, 노드 네 개

SmolVLA / MVLA 체크포인트로 xArm7을 EEF 속도(twist) 제어한다. 관측·액션
정의는 `max_data_collect`가 녹화한 것과 같다.

`thirdparty/lerobot/ros_inference`를 ROS 패키지로 옮기면서 기능별로 노드를
나눴다. 제어 로직 자체는 그대로다 — 쪼갠 것은 프로세스 경계이고, 로직은
`max_ai/rollout`에 ROS 없이 남아 있다.

```
                        RunRollout.action
                        (사람 / 웹 UI)
                              │
                              v
  MakePlan.srv   ┌──────────────────────┐   RunPolicy.action
  ReplanTask.srv │     manager_node     │──────────────────┐
  ┌──────────────┤  TaskManager          │                  │
  v              │  SubtaskTracker       │<── PolicyState ──┤
┌──────────────┐ │  에피소드 수명·레코더  │    (헤드, 상태,   │
│ planner_node │ └───────┬──────────────┘     그리퍼, 30Hz)  │
│  rules.py    │         │      ^                           │
│  (무상태)     │  TaskState     │ RecognitionVerdict         │
└──────────────┘  (프롬프트,     │                           v
                   subtask t0,   │                  ┌─────────────────┐
                   latched)      │                  │   policy_node   │
                        │        │                  │  30Hz 틱         │
                        └────────┼─────────────────>│  ChunkExecutor   │
                                 │                  │  **안전 제한 전부**│
          ScheduleRecognition.srv│                  └────────┬────────┘
                        ┌────────┴───────┐                   │
                        │ recognizer_node│                   v
                        │  카메라 직접 구독│      /xarm/vc_set_cartesian_velocity
                        │  예약 시각에 판정│      그리퍼 Float32
                        └────────┬───────┘      /data_collection/rollout_action
                                 │
     카메라 x3 ──────────────────┴──────────────> policy_node (eef_pose/twist, 그리퍼도)
```

| 노드 | 역할 | 모델 |
|---|---|---|
| `max_ai_policy` | 관측 → 청크 추론 → **안전 제한** → 로봇. 30Hz | 정책 (GPU) |
| `max_ai_manager` | plan, subtask 전진, 검증, 작업 종료. 사람의 진입점 | 없음 |
| `max_ai_planner` | 인식 상태 + 목표 → subtask 순서. 무상태 | 없음 (테이블) |
| `max_ai_recognizer` | 예약된 시각에 장면 판정 | 인식기 (GPU) |

## 왜 이렇게 나눴나

**정책 노드가 로봇을 직접 잡는다.** 안전 제한(`rollout/action.py`), 청크 실행,
명령 송출이 한 노드 안에 있다. 쪼개면 명령 하나가 노드 경계를 넘을 때마다
변화율 제한의 "틱 사이"가 무슨 뜻인지 흐려지고, `duration=0.2s` 워치독이
지키는 대상이 둘로 늘어난다.

**매니저는 센서도 모델도 없다.** `PolicyState`를 읽고 `TaskState`를 쓴다.
그 사이에 흐르는 것은 프롬프트와 헤드 읽기, 둘뿐이다.

**플래너는 교체될 자리다.** 지금은 시연에서 읽어낸 조회 테이블이다
(`rules.py`: 실패 없는 에피소드 217개 중 216개에서 작업자 본인의 subtask
순서를 재현). 서비스 경계가 있으면 VLM으로 바꾸는 것이 노드 교체로 끝난다.
**재시도 예산은 플래너로 넘어가지 않는다** — 몇 번까지 다시 집을지는
매니저에 남아서, 플래너를 바꿔도 조용히 바뀌지 않는다.

**인식기가 카메라를 직접 구독한다.** 매니저가 예약 시각만 보내고, 그 시각에
프레임을 뜨는 건 인식기다. 캡처 시각이 load-bearing이기 때문이다 — probe는
경계에서 고정 오프셋만큼 떨어진 프레임으로 학습됐고(mat +3/12/21, tray
+8/18/28), 다른 시점의 프레임으로 내린 판정은 학습 때 본 적 없는 장면에 대한
답이다. 매니저가 그 순간에 이미지 3장을 서비스로 보내는 구조였다면 캡처
시각이 큐 지연만큼 밀린다.

**제어 로직은 ROS 없이 돈다.** `max_ai/rollout/`에 ROS import가 없고, 노드는
그 위의 껍데기다. 그래서 `replay`로 로봇 없이 전체 파이프라인을 돌릴 수 있다.
이 경로를 잃으면 체크포인트를 바꿀 때마다 실기에서 처음 확인하게 된다.

## 쪼개면서 새로 생긴 고장 모드 셋

전부 조용히 틀리는 종류다. `test/test_split.py`가 이 셋을 본다.

| | 왜 생기나 | 어떻게 막았나 |
|---|---|---|
| 낡은 프롬프트의 헤드 읽기로 subtask를 공짜로 넘김 | `head_prompt: self`면 헤드는 정책과 같은 프롬프트로 읽으므로 그때의 subtask에 의존한다. 노드 경계를 넘으면 전진 직전 읽기가 전진 직후 도착한다 | `TaskState.seq`를 정책이 echo하고, 매니저가 낡은 seq의 읽기를 버린다 |
| 매니저가 죽으면 팔이 마지막 프롬프트로 계속 움직임 | 쪼개기 전에는 한 프로세스였다 | 정책의 데드맨: `TaskState`가 `task_state_timeout_s`(1초) 끊기면 스스로 정지. 매니저는 `task_state_hz`(5Hz)로 심장박동 |
| 예약이 노드 경계를 넘는 동안 캡처 시각이 지나감 | 예약이 서비스 왕복을 거친다 | `capture_tolerance_s`(0.05초)까지만 받고, 더 지난 것은 **거절**한다. 늦은 프레임으로 판정하면 틀리는 게 아니라 다른 질문에 답하는 것 |

`observation.elapsed_subtask`도 여기에 걸린다. 매니저에게 5Hz로 물어보면 그
사이 청크마다 낡은 값이 들어가므로, `TaskState.subtask_started_at`을 실어
정책이 30Hz로 직접 계산한다.

**시계.** 노드 간에는 ROS 시계를 쓴다(쪼개기 전에는 `time.monotonic()`).
기본 설정에서 ROS 시계는 시스템 시계이므로 NTP 보정에 점프할 수 있고, 큰
점프는 데드맨과 에피소드 타임아웃을 오발동시킨다 — rollout 중에 셀 PC의 시계
동기를 건드리지 말 것.

## 실행

노드 네 개가 **설정 파일 하나**를 읽는다. 노드마다 파라미터를 따로 두면 fps나
카메라 이름이 다른 상태로 뜰 수 있고, 그러면 청크 타이밍과 인식기 캡처 시각이
조용히 틀어진다.

```bash
# 로봇 PC: ROS2 + xarm_ros2 + max_interfaces 워크스페이스를 source 한 뒤
source /opt/ros/jazzy/setup.bash
source ~/workspace/m.ax/ros2/install/setup.bash
# lerobot이 설치된 환경 (정책·인식기·플래너·매니저 모두 lerobot을 import한다)
export PYTHONPATH=~/workspace/m.ax/thirdparty/lerobot/src:$PYTHONPATH

ros2 launch max_ai max_ai.launch.py
ros2 launch max_ai max_ai.launch.py recognizer:=false   # 인식기 없이 (검증 없음)

# 첫 실기는 speed_scale을 낮춰서
ros2 action send_goal /max_ai/rollout/run max_interfaces/action/RunRollout \
    "{goal: kit, speed_scale: 0.3}" --feedback

ros2 topic echo /max_ai/rollout/status          # 1Hz JSON
ros2 topic echo /max_ai/rollout/task_complete   # 종료 리포트 (latched)
ros2 topic echo /max_ai/task/state              # 매니저가 정한 프롬프트
ros2 topic echo /max_ai/policy/state            # 헤드·안전 플래그 30Hz
```

`.venv`의 파이썬이 ROS의 `rclpy`를 import할 수 있어야 한다 (파이썬 버전 일치,
`PYTHONPATH`에 ROS site-packages).

정책만 단독으로 (SmolVLA, 매니저 없이 — 데드맨이 꺼진다):

```bash
ros2 launch max_ai max_ai.launch.py manager:=false recognizer:=false planner:=false
ros2 action send_goal /max_ai/policy/run max_interfaces/action/RunPolicy \
    "{task_instruction: 'pick up the black plastic object, move it to the white tray, and place it in the tray', speed_scale: 0.3}"
```

## 검증 — 로봇 없이

`test/README.md`에 세 경로가 정리돼 있다. 요약:

```bash
# 1. 순수 로직 (39개). GPU도 ROS도 없이
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai python3 -m pytest ros2/src/max_ai/test -q

# 2. 노드 배선. 진짜 manager + 진짜 planner, 정책·인식기는 흉내. GPU 없이
MAX_AI_STUB_CHECKPOINT=/tmp/stub_checkpoint \
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai \
    python3 ros2/src/max_ai/test/integration_rollout.py

# 3. 녹화 에피소드로 전체 파이프라인. GPU 필요, 로봇은 안 씀
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai python3 -m max_ai.rollout.replay \
    --config ros2/src/max_ai/config/xarm7_kitting.yaml --episode 33
```

replay는 **개루프**다(로봇이 명령대로 안 움직임). 단위·카메라 매핑·청크
타이밍·안전 제한이 맞는지를 본다. 노드 분해 뒤에는 헤드 왕복의 **한 틱
지연**도 같이 재현한다 — 실기에서는 헤드 읽기가 policy → manager → TaskState
→ policy로 한 바퀴 돌기 때문에 전진 직후 새 프롬프트는 다음 틱부터 쓰인다.
이 지연을 빼먹으면 replay가 실기보다 나은 조건을 재는 셈이 된다.

ros_inference 시절 val 에피소드 33 결과(50k 체크포인트)는 아래와 같았다.
**노드 분해 뒤에 다시 재지 않았다** — 한 틱 지연이 들어갔으니 subtask 추적
수치는 조금 움직일 수 있다.

| | MVLA s4 | SmolVLA |
|---|---|---|
| 정책 vs 녹화 액션 상관 (vx vy vz wx wy wz) | .77 .64 .91 .92 .96 .64 | .77 .59 .93 .91 .96 .70 |
| 첫 액션까지 | 0.87 s | 0.77 s |
| MVLA 시작 상태 판단 | flip 필요 (정답) | — |
| subtask 추적 정확도 | 91% | — |

## 녹화 연동 (rollout 데이터 쌓기)

`auto_record: true`면 매니저가 에피소드 시작에 START, 끝에 SAVE를 부른다
(실패한 rollout도 저장 — 다음 학습 데이터이자 advantage 추정의 재료다).
액션 토픽은 정책 노드가 퍼블리시한다. 레코더는 `demo_data_collect_config.yaml`을
복사해 두 곳만 바꿔 띄운다:

```yaml
source: rollout
action:
  topic: /data_collection/rollout_action   # policy_node가 퍼블리시
```

`TeleopAction.cartesian_velocity`는 데모 녹화와 같은 **m/s**로 채운다.

## 확인이 필요한 가정 — 첫 실기 전에

| 가정 | 근거 | 틀리면 |
|---|---|---|
| 선속도는 m/s로 녹화됐고 xArm 서비스엔 mm/s로 보낸다 | 데이터셋 action 범위 ±0.25. `TeleopAction.msg` 주석(mm/s)과 다르다 | 1000배 빠르거나 느림 → 첫 실행은 `speed_scale: 0.1`로 |
| 속도는 베이스 좌표(`is_tool_coord=False`) | `is_tool_coord`가 녹화되지 않았다 | 시연 중 툴 좌표를 썼다면 그 구간 액션의 의미가 다름 |
| 명령 경로 = `vc_set_cartesian_velocity` + mode 5 | eef_state_publisher 주석 | teleop 노드 코드가 이 저장소에 없어 서비스 이름·타입은 xarm_ros2 표준으로 가정 |
| 그리퍼 = `/gripper/gripper_client/target_gripper_width_percent` Float32 0/1 | max_server config, TeleopAction 주석 | |
| 카메라 토픽·해상도 640x480, 회전 0 | demo_data_collect_config.yaml | 이미지 분포가 달라지면 조용히 성능 저하 |
| 작업공간 박스 | 녹화 eef 위치 0.1–99.9 퍼센타일 | **z 하한은 셀에서 실측** |
| 정책 노드와 인식기가 한 GPU에 같이 올라간다 | 인식기는 경계에서만 돌아 30Hz를 방해하지 않는다 | 메모리 부족 시 `recognizer.device`를 다른 GPU로 |

xarm_api에서 `vc_set_cartesian_velocity` 서비스가 켜져 있어야 한다.

## 안전 장치

- `speed_scale` — 모든 속도에 곱함. 첫 실기는 0.3 이하로.
- 축별 상한(데이터셋 최대값), 틱 간 변화율 제한(청크 전환 튐 방지)
- 작업공간 박스 — 밖으로 향하는 성분만 0 (안쪽 복귀는 허용)
- 관측이 0.3초 이상 끊기면 정지 (에피소드는 유지, 돌아오면 재개)
- 모든 속도 명령에 `duration=0.2s` — 노드가 죽거나 멈춰도 팔이 스스로 멈춤
- **매니저 데드맨** — `TaskState`가 1초 끊기면 정책이 스스로 정지 (새로 추가)
- 매니저가 예외로 죽어도 BLOCKED를 발행하고 정책 goal을 취소한다
- 에피소드 시간 상한, 매니저의 재시도 상한 → 초과 시 정지
- **비상정지는 하드웨어 버튼이 1순위.** 이 노드들은 그 대체가 아니다.

## MVLA 런타임 동작

세 개의 시계: 정책 30Hz / 헤드 ~5Hz (전진, value) / **인식기 — 경계에서만**

1. **시작**: 그리퍼 effort로 파지 여부. 아니면 인식기 mat 판정 3프레임
   (0.1/0.4/0.7초) → `target`/`flipped` → 플래너가 **메인 prompt(목표)**로
   plan을 세운다. `empty`·`abstain`이면 정지(사람 필요). 인식기가 없으면
   헤드의 approach_flip/pick 확률로 대체 (`TaskState.PROBING`으로 정책에게
   "헤드만 읽고 움직이지 마라"를 알린다)
2. 정책 프롬프트 = `Task: <중립 목표>. Subtask: <plan의 현재>. Quality: 5. Mistake: false.`
3. **전진**: 헤드가 plan의 다음 subtask를 2번 연속 예측
4. **flip 검증**: flip에서 넘어간 직후 +3/12/21프레임에 mat 판정. `flipped`가
   아니면(2회) approach_flip부터 재계획. 판정 동안 정책은 이미 approach_pick으로
   움직인다 — 그래서 프롬프트가 바뀌어도 날아오던 청크는 버리지 않는다
5. **place 검증**: 그리퍼가 실제로 열린 순간 +8/18/28프레임에 tray 판정.
   `ok` 2회 → 작업 완료, `bad` 2회 → **트레이 재시도** `approach_pick → pick → place`
   (데이터에서 place 실패 뒤 작업자는 항상 move 없이 트레이에서 다시 집었다)
6. 실패: 검증 실패 또는 매니저 정체 규칙 → 재계획. 재시도 상한(subtask별 3,
   전체 6) 초과 → 정지

### 메인 prompt 작업 종료

**plan의 마지막 subtask가 끝나면 메인 prompt의 작업이 끝난 것이다.** 목표가
plan을 정하므로(pick → `…pick`에서 끝, kit → `…place`에서 끝) 목표가 바뀌면
끝나는 지점도 바뀐다.

| 마지막 subtask | 완료 판단 | `basis` |
|---|---|---|
| place (kit 목표) | 인식기 tray `ok` 2회 | `recognizer_tray` |
| place인데 릴리스가 안 보임 | value 조건이 5초 유지되면 검증 없이 | `unverified` |
| pick (pick 목표), 인식기 없음 | `value_subtask ≥ -0.05` 또는 `value_episode ≥ -0.05` 3회 연속 | `last_subtask_value` |

`value_episode` 하나로만 판단하지 않는 이유 (`eval_termination`, val 58개):
헤드는 끝 시점을 잘 맞히지만(성공 에피소드 90~95%) **prompt만 pick↔kit로
바꿨을 때 값이 그대로다**(kit 에피소드의 pick 종료: -0.383 vs -0.385). 목표에
따라 종료 지점이 달라지는 걸 보장하는 경로는 plan뿐이다.

종료 시 `RunRollout` 결과와 `/max_ai/rollout/task_complete`(latched)에 같은
리포트가 나간다: `{goal, main_prompt, initial_state, basis, duration_s,
subtasks:[{name,t_start,t_end,outcome,verdict}], retries, value_subtask,
value_episode}`.

SmolVLA에는 헤드가 없어 종료 판단이 없다 (타임아웃 또는 사람이 정지).
