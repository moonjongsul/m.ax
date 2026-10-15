# max_ai 테스트

제어 로직은 전부 `max_ai/rollout`에 ROS 없이 있다. 그래서 체크포인트를 바꿀
때마다 로봇 없이 먼저 확인할 수 있고, 아래 셋이 그 경로다.

## 1. 순수 로직 — GPU도 ROS도 없이

```bash
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai \
    python3 -m pytest ros2/src/max_ai/test -q
```

| 파일 | 보는 것 |
|---|---|
| `test_core.py` | 청크 타이밍, 안전 제한 전부, 관측 변환, 설정 로딩 |
| `test_tracker.py` | plan 진행, flip·place 검증, 트레이 재시도, 종료 판단 |
| `test_split.py` | **노드 경계가 만든 고장 모드**: 프롬프트 seq, 데드맨, 캡처 예약 |

`test_split.py`가 보는 셋은 쪼개기 전에는 존재할 수 없던 것들이고 전부
조용히 틀리는 종류다 — 낡은 프롬프트로 계산된 헤드 읽기가 subtask를 공짜로
넘기는 것, 매니저가 죽었을 때 정책이 마지막 프롬프트로 계속 움직이는 것,
예약이 노드 경계를 넘는 동안 인식기 캡처 시각이 지나가는 것.

## 2. 노드 배선 — ROS는 쓰고 GPU·로봇은 안 쓰고

`policy_node`와 `recognizer_node`를 흉내 내는 노드로 바꿔 끼우고, **진짜
`manager_node`와 진짜 `planner_node`**를 실제 DDS로 돌린다.

```bash
# 매니저는 프롬프트 형태를 체크포인트의 config.json에서 읽는다 (가중치는 안 읽는다)
PYTHONPATH=thirdparty/lerobot/src python3 -c \
  "import lerobot.policies; from lerobot.policies.mvla.configuration_mvla import MVLAConfig; \
   MVLAConfig()._save_pretrained('/tmp/stub_checkpoint')"

source /opt/ros/jazzy/setup.bash && source ros2/install/setup.bash
MAX_AI_STUB_CHECKPOINT=/tmp/stub_checkpoint \
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai \
    python3 ros2/src/max_ai/test/integration_rollout.py
```

네 경우를 돈다: flip이 필요한 kit + 트레이 재시도 한 번, 바로 집는 kit,
pick 목표(value로 종료), 인식기 기권(사람 필요 — 재시도가 아니다).

## 3. 녹화 에피소드로 전체 파이프라인 — GPU 필요, 로봇은 안 씀

```bash
PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai python3 -m max_ai.rollout.replay \
    --config ros2/src/max_ai/config/xarm7_kitting.yaml --episode 33
```

**개루프다** — 로봇이 명령대로 움직이지 않으므로 누적 오차는 재지 못한다.
재는 것은 단위·프레임·카메라 매핑·청크 타이밍·안전 제한이 맞는지다. 노드
분해 뒤에는 헤드 왕복의 **한 틱 지연**도 같이 재현한다(실기와 같게).
