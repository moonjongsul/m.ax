"""MVLA tracker: 인식기 판정·헤드 읽기에 따른 plan 진행. 모델 없이.

노드 분해 뒤에도 이 로직은 ROS 없이 돈다. 달라진 것은 캡처를 직접 뜨지 않고
예약 이벤트(`schedule_recognition`)를 내보낸다는 점뿐이다 -- 그 시각에 프레임을
뜨는 건 카메라를 직접 구독하는 인식기 노드다.
"""

import numpy as np

from lerobot.policies.mvla.configuration_mvla import MVLAConfig
from max_ai.rollout.config import MVLARuntimeConfig, RecognizerRuntimeConfig
from max_ai.rollout.policy_runner import HeadRead
from max_ai.rollout.recognition import Verdict
from max_ai.rollout.subtask import SUBTASK_NAMES, SubtaskTracker

FPS = 30.0


def _tracker(goal="kit"):
    return SubtaskTracker(goal, MVLAConfig(), MVLARuntimeConfig(), RecognizerRuntimeConfig(),
                          recognizer_available=True)


def _mat(label):
    probs = {"empty": 0.0, "target": 0.0, "flipped": 0.0, "unknown_pose": 0.0}
    probs[label] = 1.0
    return {"label": label, "probs": probs, "ood": 1.0, "detected": True}


def _tray(bad):
    return {"label": "bad" if bad > 0.5 else "ok", "probs": {"ok": 1 - bad, "bad": bad}, "ood": 1.0,
            "detected": True}


def _head(subtask, v_sub=-0.5, v_ep=-0.5):
    p = np.zeros(7)
    p[SUBTASK_NAMES.index(subtask)] = 1.0
    return HeadRead(p, v_sub, 0.1, v_ep, np.array([1.0, 0.0, 0.0]))


def _kinds(events):
    return [e.kind for e in events]


def _scheduled(events, purpose):
    """예약 이벤트 하나를 꺼내고, 인식기 노드가 했을 일(request_id 부여)을 흉내."""
    return [e for e in events if e.kind == "schedule_recognition" and e.data["purpose"] == purpose]


def _accept(tr, events, purpose, request_id):
    """예약이 받아들여졌다고 tracker에 알린다 (노드는 서비스 응답으로 받는다)."""
    evs = _scheduled(events, purpose)
    assert evs, f"no {purpose} capture was scheduled"
    tr.note_request(purpose, request_id)
    return evs[0]


def _verdict(purpose, result, request_id, index=0, total=3, t_obs=0.0):
    return Verdict(purpose, t_obs, result, request_id=request_id, index=index, total=total)


# ───────────────────────────────────────────────────────────────── start
def test_start_from_recognizer_builds_flip_plan():
    tr = _tracker()
    ev = tr.schedule_start(0.0)
    sched = _accept(tr, ev, "start", 1)
    # 캡처 시각은 인식기 학습 샘플과 같아야 한다 (recognizer/dataset.py)
    assert sched.data["offsets_s"] == [0.1, 0.4, 0.7]
    assert sched.data["reference"] == 0.0
    assert sched.data["cancel_pending"] is False   # 시작 예약은 아무것도 지우지 않는다

    ev = []
    for i in range(3):
        ev += tr.on_verdict(_verdict("start", _mat("target"), 1, index=i), effort=0.0, t=0.8)
    assert "begin" in _kinds(ev)
    assert tr.remaining == ["approach_flip", "flip", "approach_pick", "pick", "move", "place"]


def test_start_waits_for_every_scheduled_frame():
    """총 개수는 판정 메시지가 알려준다 -- 예약이 일부 거절됐을 수 있다."""
    tr = _tracker()
    _accept(tr, tr.schedule_start(0.0), "start", 1)
    ev = tr.on_verdict(_verdict("start", _mat("target"), 1, index=0, total=2), effort=0.0, t=0.5)
    assert "begin" not in _kinds(ev)
    ev = tr.on_verdict(_verdict("start", _mat("target"), 1, index=1, total=2), effort=0.0, t=0.8)
    assert "begin" in _kinds(ev)


# ─────────────────────────────────────────────────────────── flip verify
def test_flip_verification_failure_replans():
    tr = _tracker()
    tr.begin("target", 0.0)
    tr.observe(_head("flip"), 1.0)
    tr.observe(_head("flip"), 1.2)          # approach_flip -> flip
    tr.observe(_head("approach_pick"), 2.0)
    ev = tr.observe(_head("approach_pick"), 2.2)  # flip -> approach_pick, flip 검증 예약
    assert tr.subtask == "approach_pick"
    sched = _accept(tr, ev, "flip", 7)
    assert sched.data["offsets_s"] == [3 / FPS, 12 / FPS, 21 / FPS]
    assert sched.data["reference"] == 2.2

    ev = []
    for i in range(2):
        ev += tr.on_verdict(_verdict("flip", _mat("target"), 7, index=i), effort=0.0, t=2.5)
    assert "replan" in _kinds(ev)
    assert tr.subtask == "approach_flip"
    assert any(r.name == "flip" and r.outcome == "failed_verification" for r in tr.history)


def test_stale_verdict_from_an_abandoned_boundary_is_dropped():
    """재계획으로 경계가 사라진 뒤 늦게 도착한 판정은 적용되지 않는다.

    노드로 쪼개면서 새로 생긴 경로다: 판정이 토픽으로 오므로 재계획 시점과
    도착 시점이 어긋날 수 있고, 첫 place 실패의 판정이 두 번째 place를
    결정해 버리면 불필요한 재시도가 난다.
    """
    tr = _tracker()
    tr.begin("in_gripper", 0.0)
    tr.observe(_head("place"), 1.0)
    ev = tr.observe(_head("place"), 1.2)    # move -> place
    ev += tr.watch(0.0, 2.0)
    ev += tr.watch(0.5, 2.1)                # 릴리스
    _accept(tr, ev, "place", 11)
    for i in range(2):
        tr.on_verdict(_verdict("place", _tray(0.9), 11, index=i), effort=0.0, t=3.0)
    assert tr.remaining == ["approach_pick", "pick", "place"]   # 트레이 재시도

    # 재계획으로 request 11의 경계는 사라졌다. 남은 세 번째 프레임이 늦게 온다.
    before = list(tr.remaining)
    tr.on_verdict(_verdict("place", _tray(0.9), 11, index=2), effort=0.0, t=3.5)
    assert tr.remaining == before


# ────────────────────────────────────────────────────────── place verify
def test_place_bad_then_tray_retry_then_complete():
    tr = _tracker()
    tr.begin("in_gripper", 0.0)
    tr.observe(_head("place"), 1.0)
    ev = tr.observe(_head("place"), 1.2)    # move -> place
    assert tr.subtask == "place"
    ev += tr.watch(0.0, 2.0)
    ev += tr.watch(0.5, 2.1)                # 그리퍼 열림 -> 릴리스
    assert tr.released_at == 2.1
    sched = _accept(tr, ev, "place", 2)
    assert sched.data["offsets_s"] == [8 / FPS, 18 / FPS, 28 / FPS]
    assert sched.data["cancel_pending"] is True

    ev = []
    for i in range(2):
        ev += tr.on_verdict(_verdict("place", _tray(0.9), 2, index=i), effort=0.0, t=3.0)
    assert "replan" in _kinds(ev)
    assert tr.remaining == ["approach_pick", "pick", "place"]   # move 없이 트레이에서 재시도

    tr.observe(_head("pick"), 4.0)
    tr.observe(_head("pick"), 4.2)
    tr.observe(_head("place"), 5.0)
    ev = tr.observe(_head("place"), 5.2)
    assert tr.subtask == "place"
    ev += tr.watch(0.0, 6.0)
    ev += tr.watch(0.6, 6.1)
    _accept(tr, ev, "place", 3)
    ev = []
    for i in range(2):
        ev += tr.on_verdict(_verdict("place", _tray(0.05), 3, index=i), effort=0.0, t=7.0)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "recognizer_tray"
    assert done[0].data["retries"] == {"place": 1}


# ──────────────────────────────────────────────────────────── completion
def test_unverifiable_last_subtask_completes_on_value():
    tr = _tracker(goal="pick")
    tr.begin("flipped", 0.0)
    tr.observe(_head("pick"), 1.0)
    tr.observe(_head("pick"), 1.2)          # approach_pick -> pick (마지막, 검증 불가)
    ev = []
    for k in range(3):
        ev += tr.observe(_head("pick", v_sub=-0.02), 2.0 + 0.2 * k)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "last_subtask_value"


def test_place_without_release_waits_then_completes_unverified():
    tr = _tracker()
    tr.begin("in_gripper", 0.0)
    tr.observe(_head("place"), 1.0)
    tr.observe(_head("place"), 1.2)
    ev = []
    for k in range(40):                       # 릴리스 없이 value만 높음
        ev += tr.observe(_head("place", v_sub=-0.01), 2.0 + 0.2 * k)
    done = [e for e in ev if e.kind == "task_complete"]
    assert done and done[0].data["basis"] == "unverified"
