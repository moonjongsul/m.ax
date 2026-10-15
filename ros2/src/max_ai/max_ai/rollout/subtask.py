"""MVLA 런타임: 메인 prompt(목표) -> plan -> subtask 추적 -> 검증 -> 작업 종료.

    메인 prompt ──> 플래너 ──> plan [s1, ..., sN]
                                  │
    헤드(subtask, value) ─────> SubtaskTracker ──> 프롬프트 (Task + 현재 Subtask) ──> 정책
    인식기(mat, tray)   ─────>       │
                                  └── plan의 마지막 subtask 완료 = 메인 prompt 작업 종료

세 개의 시계 (설계 문서 §5.3):
  * 정책 30Hz
  * 헤드 ~5Hz -- subtask 전진, value 정체 감시, 검증 불가 subtask의 완료
  * 인식기, 경계에서만 -- 시작 상태, flip 결과, place 결과. 재시도/완료를 확정한다

규칙:
  * 시작: 그리퍼 effort로 파지 여부. 아니면 인식기 mat 판정(0.1/0.4/0.7초 3프레임)
    -> 플래너. 인식기가 없으면 헤드의 approach_flip/approach_pick 확률로 대체.
  * 전진: 헤드가 plan의 다음 subtask를 `hysteresis`번 연속 예측.
  * flip 검증: flip에서 넘어간 직후 +3/12/21프레임에 mat 판정. `flipped`가 아니면
    (2회 연속) approach_flip부터 재계획. 판정 동안 정책은 이미 approach_pick
    프롬프트로 움직인다 -- flip 프롬프트에 붙잡아 두면 다시 뒤집으려 할 수 있다.
  * place 검증: 그리퍼가 실제로 열린 순간(관측 위치 상승)을 릴리스로 보고
    +8/18/28프레임에 tray 판정. ok 2회 -> 완료, bad 2회 -> approach_pick부터 재시도.
  * 검증할 수 없는 마지막 subtask(pick 목표의 pick, 인식기 없음): value로 완료
    (`value_subtask` 또는 `value_episode` >= 임계가 `completion_consecutive`회).
  * place인데 릴리스가 끝내 안 보이면: value 완료 조건이 `unverified_timeout_s`
    동안 유지될 때 검증 없이 완료하고 근거를 `unverified`로 남긴다.
  * 실패: 매니저의 정체 규칙 또는 검증 실패 -> 재계획. 재시도 상한 초과 -> 정지.

**작업 종료를 plan으로 판단하는 이유:** 헤드의 `value_episode`는 끝 시점을
잘 맞히지만 목표 문장을 바꿔도 값이 그대로다(`eval_termination`). 목표에 따라
끝나는 지점이 바뀐다는 걸 보장하는 경로는 plan뿐이다.

ROS를 import하지 않는다. 노드 분해 뒤에도 그대로다:

  * 플랜은 `PlanSource`로 받는다 -- 실기는 planner 노드의 서비스, replay는
    rules.py 직접 호출.
  * 인식기 캡처는 **예약 이벤트로 내보낸다**. 캡처 시각은 여기서 정하지만
    (학습 샘플과 같은 시점이어야 한다), 그 시각에 프레임을 뜨는 건 카메라를
    직접 구독하는 인식기 노드다. 경계가 지나가는 순간에 왕복이 끼면 캡처
    시각이 큐 지연만큼 밀린다.
  * 프롬프트 조립에 필요한 건 체크포인트 `config.json`의 플래그 몇 개뿐이다
    (`load_prompt_config`). 가중치는 안 읽는다 -- 매니저 노드에 모델이 없다.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lerobot.policies.mvla.data.derive import NEUTRAL_PROMPTS
from lerobot.policies.mvla.planner.manager import VERIFIED_BY, ManagerConfig, Outcome
from lerobot.policies.mvla.planner.rules import (
    APPROACH_FLIP, APPROACH_PICK, FLIP, IN_GRIPPER, MAT_FLIPPED, MAT_TARGET, PLACE,
)
from lerobot.policies.mvla.processor_mvla import PromptBuilder

from .config import MVLARuntimeConfig, RecognizerRuntimeConfig
from .planner import PlanSource, SourcedTaskManager
from .policy_runner import HeadRead, resolve_checkpoint
from .recognition import Verdict, aggregate

logger = logging.getLogger("max_ai.subtask")

# 헤드 클래스 번호 -> 짧은 이름. 데이터셋 meta/subtasks의 순서와 같다.
SUBTASK_NAMES = ["approach_flip", "flip", "approach_pick", "pick", "move", "place"]
# 짧은 이름 -> 학습 때 프롬프트에 들어간 문장 (meta/subtasks.parquet)
SUBTASK_SENTENCES = {
    "approach_flip": "approach to black plastic object for flip",
    "flip": "flip black plastic object upside down",
    "approach_pick": "approach to black plastic object for pick up",
    "pick": "pick up black plastic object",
    "move": "move black plastic object to white tray",
    "place": "place black plastic object in white tray",
}
FPS = 30.0


def load_prompt_config(checkpoint: str | Path):
    """체크포인트의 `config.json`만 읽어 `PromptBuilder`용 설정을 만든다.

    매니저 노드는 모델을 안 올린다. 하지만 프롬프트 형태(`use_subtask_prompt`,
    `quality_bins`, `use_mistake_prompt` ...)는 그 체크포인트가 학습된 형태와
    같아야 하므로, 설정만 디스크에서 그대로 읽는다. 매니저가 자기 기본값으로
    프롬프트를 만들면 학습 때와 다른 문장이 정책에 들어간다.
    """
    import lerobot.policies  # noqa: F401 - config 서브클래스(mvla 포함) 등록
    from lerobot.configs.policies import PreTrainedConfig

    return PreTrainedConfig.from_pretrained(resolve_checkpoint(checkpoint))


@dataclass
class TrackerEvent:
    kind: str       # begin | advance | replan | task_complete | needs_operator | verdict
                    # | schedule_recognition | cancel_recognition
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class SubtaskRecord:
    name: str
    t_start: float
    t_end: float | None = None
    outcome: str = "running"   # running | done | failed | failed_verification
    verdict: dict | None = None


class SubtaskTracker:
    def __init__(self, goal: str, policy_config, cfg: MVLARuntimeConfig,
                 recog: RecognizerRuntimeConfig | None = None, recognizer_available: bool = False,
                 source: PlanSource | None = None):
        if goal not in NEUTRAL_PROMPTS:
            raise ValueError(f"MVLA task must be a goal name {sorted(NEUTRAL_PROMPTS)}, got {goal!r}")
        self.goal = goal
        self.cfg = cfg
        self.rcfg = recog or RecognizerRuntimeConfig()
        self.recognizer = recognizer_available
        self.builder = PromptBuilder(policy_config)
        self.manager = SourcedTaskManager(
            ManagerConfig(goal=goal, tray_bad_threshold=self.rcfg.tray_bad_threshold), source=source
        )
        self.neutral = NEUTRAL_PROMPTS[goal]
        self.t0 = 0.0
        self.t_begin = 0.0
        self.history: list[SubtaskRecord] = []
        self.initial_state: str | None = None
        self.start_verdict: dict | None = None
        self.completion_streak = 0
        self.last_head: HeadRead | None = None
        self._last_value_t: float | None = None
        self._pending = ("", 0)
        self._current: str | None = None
        # 예약해 둔 인식기 요청: purpose -> request_id. 재계획으로 경계가
        # 사라지면 지운다 -- 늦게 도착한 판정을 엉뚱한 경계에 적용하면 안 된다.
        self._requests: dict[str, int] = {}
        self._frames: dict[str, list[dict]] = {"start": [], "flip": [], "place": []}
        self._grip_min: float | None = None
        self.released_at: float | None = None
        self._unverified_since: float | None = None
        self._retries_seen = 0

    # ────────────────────────────────────────────────────────── queries
    @property
    def started(self) -> bool:
        return self.manager.state is not None

    @property
    def subtask(self) -> str | None:
        return self.manager.state.subtask if self.manager.state else None

    @property
    def remaining(self) -> list[str]:
        return list(self.manager.state.plan.subtasks) if self.manager.state else []

    @property
    def outcome(self) -> Outcome:
        return self.manager.state.outcome if self.manager.state else Outcome.RUNNING

    def elapsed(self, t: float) -> float:
        return max(0.0, t - self.t0)

    def main_prompt(self) -> str:
        return self.builder.build_unconditional(self.neutral)

    def policy_prompt(self) -> str:
        sentence = SUBTASK_SENTENCES.get(self.subtask or "")
        return self.builder.build_inference(self.neutral, subtask=sentence, speed_bin_value=self.cfg.speed_bin)

    def head_prompt(self) -> str:
        return self.policy_prompt() if self.cfg.head_prompt == "self" else self.main_prompt()

    def _verifies(self, subtask: str) -> bool:
        if not self.recognizer or subtask not in VERIFIED_BY:
            return False
        return self.rcfg.verify_place if subtask == PLACE else self.rcfg.verify_flip

    # ───────────────────────────────────────────────────────────── start
    def start_mode(self) -> str:
        """recognizer | head | 고정 상태 이름."""
        mode = self.cfg.initial_state
        if mode == "auto":
            return "recognizer" if self.recognizer else "head"
        if mode == "recognizer" and not self.recognizer:
            raise RuntimeError("mvla.initial_state=recognizer but no recognizer is configured")
        return mode

    def schedule_start(self, t: float) -> list[TrackerEvent]:
        return [self._schedule("start", t, list(self.rcfg.start_offsets_s))]

    def begin_from_head(self, head: HeadRead | None, effort: float, t: float) -> list[TrackerEvent]:
        mode = self.start_mode()
        if effort >= self.cfg.holding_effort:
            return self.begin(IN_GRIPPER, t)
        if mode in (MAT_TARGET, MAT_FLIPPED, IN_GRIPPER):
            return self.begin(mode, t)
        if head is None:
            raise RuntimeError("head-based start needs a head read")
        p = head.subtask_probs
        i_flip, i_pick = SUBTASK_NAMES.index(APPROACH_FLIP), SUBTASK_NAMES.index(APPROACH_PICK)
        return self.begin(MAT_TARGET if p[i_flip] >= p[i_pick] else MAT_FLIPPED, t, basis="head")

    def begin(self, mat_label: str, t: float, holding: bool = False, basis: str = "fixed") -> list[TrackerEvent]:
        holding = holding or mat_label == IN_GRIPPER
        st = self.manager.begin(mat_label, holding=holding)
        self.initial_state = st.plan.state
        self.t0 = self.t_begin = t
        self._current = st.subtask
        if st.subtask:
            self.history.append(SubtaskRecord(st.subtask, t))
        logger.info(f"plan: {st.plan}  (start {mat_label}, holding={holding}, basis={basis})")
        if st.outcome is Outcome.NEEDS_OPERATOR:
            return [TrackerEvent("needs_operator", st.reason)]
        if st.outcome is Outcome.DONE:
            return [TrackerEvent("task_complete", "goal already satisfied at start",
                                 self.completion_report(t, "already_satisfied"))]
        return [TrackerEvent("begin", f"{self.initial_state}: {st.plan}", {"basis": basis})]

    # ────────────────────────────────────────────────── recognizer timing
    def _schedule(self, purpose: str, reference: float, offsets_s: list[float]) -> TrackerEvent:
        """캡처 예약 이벤트. 실제 캡처는 인식기 노드가 그 시각에 한다."""
        self._requests[purpose] = -1   # 노드가 request_id를 받아 note_request로 채운다
        return TrackerEvent("schedule_recognition", f"{purpose}: +{offsets_s}",
                            {"purpose": purpose, "offsets_s": offsets_s, "reference": reference,
                             "cancel_pending": purpose != "start"})

    def note_request(self, purpose: str, request_id: int) -> None:
        """예약이 받아들여졌을 때 노드가 알려주는 request_id."""
        self._requests[purpose] = request_id

    def watch(self, gripper_position: float, t: float) -> list[TrackerEvent]:
        """매 틱. place 중 그리퍼가 실제로 열리는 순간을 찾는다.

        학습 샘플의 릴리스는 관측 그리퍼 채널의 가장 큰 상승이었다. 런타임은
        미래를 모르므로 place 중 최저값에서 `release_rise` 이상 오른 첫 순간으로 본다.
        """
        if self.subtask != PLACE or not self._verifies(PLACE) or self.released_at is not None:
            return []
        g = float(gripper_position)
        self._grip_min = g if self._grip_min is None else min(self._grip_min, g)
        if g - self._grip_min < self.rcfg.release_rise:
            return []
        self.released_at = t
        logger.info(f"release detected at {t:.2f}s (gripper {self._grip_min:.2f} -> {g:.2f})")
        return [self._schedule("place", t, [f / FPS for f in self.rcfg.release_offsets_frames])]

    # ──────────────────────────────────────────────────────── verdicts
    def _stale(self, v: Verdict) -> bool:
        """재계획으로 버린 경계의 판정인가.

        request_id로 가린다. 같은 purpose의 판정이 두 경계에 걸쳐 들어올 수
        있고(place 실패 -> 재시도 -> 다시 place), 늦게 온 첫 경계의 판정이
        두 번째를 결정해 버리면 안 된다.
        """
        want = self._requests.get(v.purpose)
        return want is not None and want >= 0 and v.request_id != want

    def on_verdict(self, v: Verdict, effort: float, t: float) -> list[TrackerEvent]:
        if self._stale(v):
            logger.info(f"dropping stale {v.purpose} verdict (request {v.request_id})")
            return []
        res = v.result
        events = [TrackerEvent("verdict", f"{v.purpose}: {res['label']}", {"purpose": v.purpose, **res})]
        if v.purpose == "start":
            self._frames["start"].append(res)
            if len(self._frames["start"]) < v.total or self.started:
                return events
            agg = aggregate(self._frames["start"])
            self.start_verdict = agg
            return events + self.begin(agg["label"], t, holding=effort >= self.cfg.holding_effort,
                                       basis="recognizer")

        st = self.manager.state
        if st is None or st.outcome is not Outcome.RUNNING:
            return events

        if v.purpose == "flip":
            self._frames["flip"].append(res)
            before = self.subtask
            self.manager.verification_failed(FLIP, res)
            if self.subtask != before or self.manager.state.outcome is not Outcome.RUNNING:
                for rec in reversed(self.history):
                    if rec.name == FLIP:
                        rec.outcome, rec.verdict = "failed_verification", aggregate(self._frames["flip"])
                        break
            return events + self._sync(t, basis="recognizer_mat")

        if v.purpose == "place":
            if self.subtask != PLACE:
                return events  # 재계획 뒤 늦게 도착한 판정
            self._frames["place"].append(res)
            self.manager.subtask_finished(verdict=res)
            if self.subtask == PLACE and self.manager.state.outcome is Outcome.RUNNING \
                    and len(self._frames["place"]) >= v.total:
                # 프레임끼리 엇갈려 히스테리시스를 못 채웠다: 평균 판정으로 결정
                agg = aggregate(self._frames["place"])
                self.manager.subtask_finished(verdict=agg)
                self.manager.subtask_finished(verdict=agg)
            if self.history and self.history[-1].name == PLACE and self.subtask != PLACE:
                self.history[-1].verdict = aggregate(self._frames["place"])
            return events + self._sync(t, basis="recognizer_tray")
        return events

    # ─────────────────────────────────────────────────────────── heads
    def observe(self, head: HeadRead, t: float) -> list[TrackerEvent]:
        """헤드 한 번 읽을 때마다."""
        self.last_head = head
        st = self.manager.state
        if st is None or st.outcome is not Outcome.RUNNING:
            return []

        dt = 0.0 if self._last_value_t is None else max(0.0, t - self._last_value_t)
        self._last_value_t = t
        self.manager.observe_value(head.value_subtask, dt)
        if self.subtask != self._current or self.manager.state.outcome is not Outcome.RUNNING:
            return self._sync(t, basis="stall")

        plan = st.plan.subtasks
        basis = "head"
        if len(plan) == 1:
            cfg = self.cfg
            hit = (head.value_subtask >= cfg.completion_value_subtask
                   or head.value_episode >= cfg.completion_value_episode)
            self.completion_streak = self.completion_streak + 1 if hit else 0
            if self.completion_streak >= cfg.completion_consecutive:
                if not self._verifies(plan[0]):
                    self.manager.subtask_finished(verdict=None)
                    basis = "last_subtask_value"
                elif self.released_at is None:
                    # 정책은 끝났다고 보는데 그리퍼가 안 열렸다. 잠시 기다린 뒤 검증 없이 완료.
                    self._unverified_since = self._unverified_since or t
                    if t - self._unverified_since >= self.rcfg.unverified_timeout_s:
                        logger.warning(f"{plan[0]}: no release seen; completing unverified")
                        self.manager.subtask_finished(verdict=None)
                        basis = "unverified"
            else:
                self._unverified_since = None
        else:
            self.completion_streak = 0
            nxt = plan[1]
            pred = SUBTASK_NAMES[head.subtask]
            if pred == nxt:
                last, count = self._pending
                count = count + 1 if last == pred else 1
                self._pending = (pred, count)
                if count >= self.cfg.hysteresis:
                    self._pending = ("", 0)
                    self.manager.subtask_finished(verdict=None)
            else:
                self._pending = ("", 0)
        return self._sync(t, basis=basis)

    # ─────────────────────────────────────────────────── state changes
    def _sync(self, t: float, basis: str) -> list[TrackerEvent]:
        """매니저 상태가 바뀌었으면 기록·타이머·검증 예약을 맞추고 이벤트를 낸다."""
        events: list[TrackerEvent] = []
        st = self.manager.state
        prev, new = self._current, st.subtask
        # 같은 subtask로 재계획되는 경우도 있다(approach_pick 정체 -> approach_pick).
        retried = st.total_retries != self._retries_seen
        self._retries_seen = st.total_retries
        if new != prev or retried:
            # 매니저는 실패(_fail)일 때만 재시도 수를 올린다. 그 외의 변화는 전진이다
            # -- plan 순서는 고정이 아니다(트레이 재시도는 pick -> place).
            advanced = not retried
            kind = "advance" if advanced else "replan"
            if self.history and self.history[-1].t_end is None:
                rec = self.history[-1]
                rec.t_end = t
                if rec.outcome == "running":
                    rec.outcome = "done" if kind == "advance" else "failed"
            if new:
                self.history.append(SubtaskRecord(new, t))
                detail = f"{prev} -> {new}" + (f" ({st.reason})" if kind == "replan" else "")
                events.append(TrackerEvent(kind, detail, {"basis": basis}))
            self._current = new
            self.t0 = t
            self.completion_streak = 0
            self._pending = ("", 0)
            self._grip_min, self.released_at, self._unverified_since = None, None, None
            self._frames["place"] = []
            # 경계가 지나갔다. 아직 안 온 검증 판정은 이제 엉뚱한 경계의 것이다.
            dropped = [p for p in self._requests if p != "start"]
            if dropped:
                for p in dropped:
                    self._requests.pop(p, None)
                events.append(TrackerEvent("cancel_recognition", f"dropped {sorted(dropped)}",
                                           {"purposes": sorted(dropped)}))
            if kind == "advance" and prev == FLIP and self._verifies(FLIP):
                self._frames["flip"] = []
                events.append(self._schedule("flip", t, [f / FPS for f in self.rcfg.flip_offsets_frames]))
        if st.outcome is Outcome.DONE:
            events.append(TrackerEvent("task_complete", f"goal '{self.goal}' reached",
                                       self.completion_report(t, basis)))
        elif st.outcome is Outcome.NEEDS_OPERATOR:
            events.append(TrackerEvent("needs_operator", st.reason))
        return events

    # ────────────────────────────────────────────────────────── report
    def completion_report(self, t: float, basis: str) -> dict:
        h = self.last_head
        return {
            "goal": self.goal,
            "main_prompt": self.main_prompt(),
            "initial_state": self.initial_state,
            "start_verdict": self.start_verdict,
            "basis": basis,
            "duration_s": round(t - self.t_begin, 2),
            "subtasks": [asdict(r) for r in self.history],
            "retries": dict(self.manager.state.retries) if self.manager.state else {},
            "value_subtask": None if h is None else round(h.value_subtask, 3),
            "value_episode": None if h is None else round(h.value_episode, 3),
        }

    def snapshot(self, t: float) -> dict:
        """status 토픽용 현재 상태."""
        h = self.last_head
        st = self.manager.state
        return {
            "goal": self.goal,
            "main_prompt": self.main_prompt(),
            "subtask": self.subtask,
            "remaining": self.remaining,
            "on_last_subtask": len(self.remaining) == 1,
            "subtask_elapsed_s": round(self.elapsed(t), 2),
            "task_elapsed_s": round(t - self.t_begin, 2),
            "completion_streak": self.completion_streak,
            "released": self.released_at is not None,
            "pending_verification": sorted(self._requests),
            "retries": dict(st.retries) if st else {},
            "value_subtask": None if h is None else round(h.value_subtask, 3),
            "value_subtask_spread": None if h is None else round(h.value_subtask_spread, 3),
            "value_episode": None if h is None else round(h.value_episode, 3),
            "head_subtask": None if h is None else SUBTASK_NAMES[h.subtask],
        }
