"""노드 경계가 만든 새 고장 모드들. 모델도 ROS도 없이.

쪼개기 전에는 `RolloutController.tick()` 하나가 전부를 했다. 쪼개면서 셋이
새로 생겼고, 전부 조용히 틀리는 종류다:

  * 낡은 프롬프트로 계산된 헤드 읽기가 전진을 일으킨다 (seq)
  * 매니저가 죽으면 정책이 마지막 프롬프트로 계속 움직인다 (데드맨)
  * 예약이 노드 경계를 넘는 동안 캡처 시각이 지나간다 (CaptureScheduler)
"""

import numpy as np
import pytest

from lerobot.policies.mvla.configuration_mvla import MVLAConfig
from max_ai.rollout.config import RolloutConfig
from max_ai.rollout.episode import ACTIVE, BLOCKED, DONE, PROBING, EpisodeManager
from max_ai.rollout.executor import HALT, IDLE
from max_ai.rollout.executor import ACTIVE as EXEC_ACTIVE
from max_ai.rollout.executor import PROBING as EXEC_PROBING
from max_ai.rollout.executor import PolicyExecutor, PromptSpec
from max_ai.rollout.observation import RawObservation
from max_ai.rollout.policy_runner import HeadRead
from max_ai.rollout.recognition import CaptureScheduler
from max_ai.rollout.subtask import SUBTASK_NAMES

FPS = 30.0


def _head(subtask, v_sub=-0.5, v_ep=-0.5):
    p = np.zeros(7)
    p[SUBTASK_NAMES.index(subtask)] = 1.0
    return HeadRead(p, v_sub, 0.1, v_ep, np.array([1.0, 0.0, 0.0]))


# ─────────────────────────────────────────────── prompt versioning (seq)
def _manager(goal="kit", recognizer=False, initial_state="target"):
    mgr = EpisodeManager(RolloutConfig(), is_mvla=True, prompt_config=MVLAConfig(),
                         recognizer_available=recognizer)
    mgr.begin(0.0, goal=goal, initial_state=initial_state)
    return mgr


def test_seq_rises_only_when_the_prompt_actually_changes():
    mgr = _manager()
    first = mgr.task_state(0.0)
    assert first.state == ACTIVE and first.subtask == "approach_flip"
    # 전진시키지 않는 헤드 읽기를 여러 번. 프롬프트는 그대로여야 한다.
    for k in range(4):
        mgr.on_head(_head("approach_flip"), 0.5 + 0.2 * k, mgr.seq)
    assert mgr.task_state(1.5).seq == first.seq

    mgr.on_head(_head("flip"), 2.0, mgr.seq)
    mgr.on_head(_head("flip"), 2.2, mgr.seq)     # hysteresis 2 -> 전진
    after = mgr.task_state(2.2)
    assert after.subtask == "flip" and after.seq > first.seq


def test_head_read_from_the_previous_prompt_cannot_advance_the_plan():
    """전진 직전 프롬프트로 계산된 읽기가 전진 직후 도착하는 경우.

    `head_prompt: self`면 헤드는 정책과 같은 프롬프트로 읽으므로, 그 읽기는
    이전 subtask에 대한 판단이다. 그걸로 다음 칸을 넘기면 한 칸을 공짜로
    건너뛴다 -- 쪼개기 전에는 같은 틱 안이라 생길 수 없던 일이다.
    """
    mgr = _manager()
    stale = mgr.seq
    mgr.on_head(_head("flip"), 1.0, stale)
    mgr.on_head(_head("flip"), 1.2, stale)       # approach_flip -> flip
    assert mgr.task_state(1.2).subtask == "flip"
    assert mgr.seq != stale

    # 낡은 seq로 다음 subtask를 두 번 예측 -- 무시돼야 한다.
    mgr.on_head(_head("approach_pick"), 1.3, stale)
    mgr.on_head(_head("approach_pick"), 1.4, stale)
    assert mgr.task_state(1.4).subtask == "flip"

    # 현재 seq로는 전진한다.
    mgr.on_head(_head("approach_pick"), 1.5, mgr.seq)
    mgr.on_head(_head("approach_pick"), 1.6, mgr.seq)
    assert mgr.task_state(1.6).subtask == "approach_pick"


def test_subtask_started_at_travels_with_the_prompt():
    """정책은 elapsed를 직접 계산한다. 그 기준 시각이 TaskState에 실려야 한다."""
    mgr = _manager()
    assert mgr.task_state(0.0).subtask_started_at == 0.0
    mgr.on_head(_head("flip"), 5.0, mgr.seq)
    mgr.on_head(_head("flip"), 5.2, mgr.seq)
    view = mgr.task_state(5.2)
    assert view.subtask == "flip" and view.subtask_started_at == 5.2


def test_recognizer_start_holds_the_policy_in_probing():
    mgr = EpisodeManager(RolloutConfig(), is_mvla=True, prompt_config=MVLAConfig(),
                         recognizer_available=True)
    events = mgr.begin(0.0, goal="kit")
    assert [e.kind for e in events] == ["schedule_recognition"]
    view = mgr.task_state(0.0)
    # 플랜이 없으니 정책은 멈춰 있어야 한다. 인식기 시작이므로 헤드도 안 읽는다.
    assert view.state == PROBING and view.policy_prompt == "" and view.head_prompt == ""


def test_head_start_asks_the_policy_for_heads_only():
    mgr = EpisodeManager(RolloutConfig(), is_mvla=True, prompt_config=MVLAConfig(),
                         recognizer_available=False)
    mgr.begin(0.0, goal="kit")
    view = mgr.task_state(0.0)
    assert view.state == PROBING and view.policy_prompt == "" and view.head_prompt
    # 첫 읽기가 시작 상태를 정한다. approach_flip 확률이 높으면 target.
    mgr.on_head(_head("approach_flip"), 0.1, view.seq)
    assert mgr.status == ACTIVE
    assert mgr.task_state(0.1).subtask == "approach_flip"


def test_a_held_part_skips_probing_entirely():
    mgr = EpisodeManager(RolloutConfig(), is_mvla=True, prompt_config=MVLAConfig(),
                         recognizer_available=True)
    # 파지 여부는 vision이 아니라 그리퍼 effort에서 읽는다.
    events = mgr.begin(0.0, goal="kit", gripper_effort=0.9)
    assert [e.kind for e in events] == ["begin"]
    assert mgr.task_state(0.0).remaining == ["move", "place"]


def test_terminal_states_tell_the_policy_to_halt():
    mgr = _manager(goal="pick", initial_state="flipped")
    mgr.on_head(_head("pick"), 1.0, mgr.seq)
    mgr.on_head(_head("pick"), 1.2, mgr.seq)
    for k in range(3):
        mgr.on_head(_head("pick", v_sub=-0.02), 2.0 + 0.2 * k, mgr.seq)
    assert mgr.status == DONE and mgr.task_state(3.0).state == DONE

    other = _manager()
    other.abort("operator stopped the cell")
    assert other.task_state(1.0).state == BLOCKED


# ──────────────────────────────────────────────────────────────── deadman
class _StubWorker:
    """추론은 하지 않는다. 데드맨과 수명만 본다."""

    busy = False
    submitted = 0

    def submit(self, req, now):
        self.submitted += 1
        return True

    def poll(self, now):
        return []

    def close(self):
        pass


def _executor(cfg=None):
    cfg = cfg or RolloutConfig()
    raw = RawObservation(stamp=0.0, eef_position=np.array([0.1, -0.3, 0.15]), gripper_position=1.0)
    clock = {"t": 0.0}

    def obs(_with_images):
        raw.stamp = clock["t"]
        return raw

    ex = PolicyExecutor(cfg, _Runner(), _StubWorker(), obs, cfg.control.task_state_timeout_s)
    return ex, clock


class _Runner:
    is_mvla = True

    class config:
        type = "mvla"


def test_policy_stops_itself_when_the_task_state_goes_silent():
    ex, clock = _executor()
    ex.begin(0.0)
    prompt = PromptSpec(mode=EXEC_ACTIVE, seq=1, policy_prompt="Task: x.", head_prompt="Task: x.",
                        subtask_started_at=0.0, received_at=0.0)
    clock["t"] = 0.5
    assert ex.tick(0.5, prompt).status == "running"
    # 매니저가 죽었다. received_at이 더 이상 갱신되지 않는다.
    clock["t"] = 1.6
    tick = ex.tick(1.6, prompt)
    assert ex.status == "aborted" and "task state stale" in ex.reason
    assert tick.command.is_zero


def test_standalone_runs_without_a_manager():
    """SmolVLA 단독 실행. 아무도 갱신하지 않는 프롬프트를 끊겼다고 보면 안 된다."""
    ex, clock = _executor()
    ex.begin(0.0)
    prompt = PromptSpec(mode=EXEC_ACTIVE, seq=0, policy_prompt="pick up the black object",
                        subtask_started_at=0.0, received_at=0.0, standalone=True)
    for t in (0.5, 5.0, 60.0):
        clock["t"] = t
        ex.tick(t, prompt)
        assert ex.status == "running", f"stopped at {t}s"


def test_halt_stops_the_arm_and_keeps_the_last_head():
    ex, clock = _executor()
    ex.begin(0.0)
    clock["t"] = 0.5
    ex.tick(0.5, PromptSpec(mode=EXEC_ACTIVE, seq=1, policy_prompt="Task: x.", received_at=0.5))
    clock["t"] = 0.6
    tick = ex.tick(0.6, PromptSpec(mode=HALT, seq=2, received_at=0.6))
    assert ex.status == "stopped" and tick.command.is_zero


def test_probing_requests_heads_but_never_moves():
    ex, clock = _executor()
    ex.begin(0.0)
    clock["t"] = 0.1
    tick = ex.tick(0.1, PromptSpec(mode=EXEC_PROBING, seq=1, head_prompt="Task: x.", received_at=0.1))
    assert tick.command.is_zero and ex.status == "starting"
    assert ex.worker.submitted == 1          # 헤드 읽기 하나
    # 프롬프트가 없으면 청크를 요청하지 않는다 -- 플랜이 선 뒤에.
    clock["t"] = 0.2
    ex.tick(0.2, PromptSpec(mode=EXEC_PROBING, seq=1, head_prompt="", received_at=0.2))
    assert ex.worker.submitted == 1


def test_stale_observation_holds_but_keeps_the_episode():
    ex, clock = _executor()
    ex.begin(0.0)
    prompt = PromptSpec(mode=EXEC_ACTIVE, seq=1, policy_prompt="Task: x.", received_at=0.0)
    clock["t"] = 0.0           # 관측 시각이 멈춘다 (카메라 끊김)
    tick = ex.tick(0.5, prompt)
    assert "stale_observation" in tick.command.flags
    assert ex.active           # 에피소드는 유지. 돌아오면 이어간다.


def test_episode_timeout_aborts():
    cfg = RolloutConfig()
    cfg.control.max_episode_s = 2.0
    ex, clock = _executor(cfg)
    ex.begin(0.0)
    prompt = PromptSpec(mode=EXEC_ACTIVE, seq=1, policy_prompt="Task: x.", received_at=0.0,
                        standalone=True)
    clock["t"] = 2.5
    ex.tick(2.5, prompt)
    assert ex.status == "aborted" and ex.reason == "episode timeout"


def test_idle_prompt_holds_without_ending_the_episode():
    """에피소드 시작 직후, 첫 TaskState가 도착하기 전의 상태."""
    ex, clock = _executor()
    ex.begin(0.0)
    clock["t"] = 0.1
    tick = ex.tick(0.1, PromptSpec(mode=IDLE, received_at=0.1))
    assert tick.command.is_zero and ex.active


# ──────────────────────────────────────────────────── capture scheduling
def test_captures_fire_at_the_scheduled_instants():
    s = CaptureScheduler(tolerance_s=0.05)
    rid, n, late = s.schedule("place", reference=10.0, offsets_s=[8 / FPS, 18 / FPS, 28 / FPS], now=10.0)
    assert (n, late) == (3, [])
    assert s.pop_due(10.0 + 7 / FPS) == []
    due = s.pop_due(10.0 + 8 / FPS)
    assert [c.index for c in due] == [0] and due[0].request_id == rid and due[0].total == 3
    assert len(s.pop_due(10.0 + 30 / FPS)) == 2


def test_a_capture_time_already_gone_is_refused_not_approximated():
    """지연으로 시각이 지나버리면 그 프레임은 포기한다.

    probe는 경계에서 고정 오프셋만큼 떨어진 프레임으로 학습됐다. 늦게 뜬
    프레임으로 판정하면 틀리는 게 아니라 다른 질문에 답하는 것이고, 그걸로
    재시도를 결정하면 안 된다.
    """
    s = CaptureScheduler(tolerance_s=0.05)
    # 예약이 노드 경계를 넘는 데 0.2초 걸렸다. 첫 오프셋(+0.1s)은 지나갔다.
    rid, n, late = s.schedule("start", reference=0.0, offsets_s=[0.1, 0.4, 0.7], now=0.2)
    assert n == 2 and len(late) == 1
    # 남은 둘만 total로 센다 -- tracker가 못 올 프레임을 기다리지 않도록.
    assert [c.total for c in s.pop_due(1.0)] == [2, 2]


def test_everything_past_is_refused_outright():
    s = CaptureScheduler(tolerance_s=0.05)
    _rid, n, late = s.schedule("flip", reference=0.0, offsets_s=[0.1, 0.4], now=5.0)
    assert n == 0 and len(late) == 2


def test_cancel_pending_drops_the_abandoned_boundary():
    s = CaptureScheduler()
    s.schedule("place", reference=0.0, offsets_s=[1.0, 2.0, 3.0], now=0.0)
    assert s.pending == 3
    s.schedule("place", reference=10.0, offsets_s=[1.0], now=10.0, cancel_pending=True)
    assert s.pending == 1
    assert s.pop_due(11.0)[0].due == pytest.approx(11.0)


def test_tolerance_accepts_a_capture_that_just_slipped():
    s = CaptureScheduler(tolerance_s=0.05)
    _rid, n, late = s.schedule("flip", reference=0.0, offsets_s=[0.1], now=0.13)
    assert n == 1 and late == []
