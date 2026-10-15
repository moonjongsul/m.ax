"""매니저측 에피소드 수명. ROS 없음 -- manager 노드와 replay가 같은 코드를 쓴다.

`ros_inference.controller.RolloutController`를 둘로 가른 쪽 중 매니저측이다.
정책측(`executor.PolicyExecutor`)은 청크를 실행하고 팔을 움직이며, 여기서는
플랜을 세우고 subtask를 전진시키고 작업이 끝났는지 판단한다. 둘 사이에 흐르는
것은 두 가지뿐이다:

    EpisodeManager ──TaskStateView(프롬프트, subtask 시작 시각, seq)──> PolicyExecutor
    EpisodeManager <──헤드 읽기 (task_seq 붙어서)─────────────────────┘

`seq`가 필요한 이유: `head_prompt: self`면 헤드는 정책과 같은 프롬프트로 읽는다.
즉 헤드 읽기는 그때의 subtask에 의존한다. 노드 경계를 넘으면 전진 직전 프롬프트로
계산된 읽기가 전진 직후에 도착할 수 있고, 그 읽기로 "다음 subtask가 보인다"를
판단하면 한 칸을 공짜로 넘긴다. 그래서 낡은 seq의 읽기는 버린다.

상태: idle -> probing -> running -> (done | stopped | aborted)

`done`은 **메인 prompt의 작업이 끝났다**는 뜻이고, 그 판단은 plan의 마지막
subtask가 끝났는지로만 한다 (`subtask.py` 참고 -- value_episode는 끝 시점을
잘 맞히지만 목표 문장을 바꿔도 값이 그대로라서, 목표에 따라 끝이 달라지는 걸
보장하는 경로는 plan뿐이다).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import RolloutConfig
from .planner import PlanSource
from .policy_runner import HeadRead
from .recognition import Verdict
from .subtask import SubtaskTracker, TrackerEvent, load_prompt_config

logger = logging.getLogger("max_ai.episode")

# TaskStateView.state -- max_interfaces/TaskState의 상수와 같은 뜻.
ACTIVE = "active"
IDLE = "idle"
DONE = "done"
BLOCKED = "blocked"
PROBING = "probing"


@dataclass
class TaskStateView:
    """정책에게 보내는 전부. 노드는 이걸 TaskState 메시지로, replay는 PromptSpec으로."""

    seq: int = 0
    state: str = IDLE
    goal: str = ""
    policy_prompt: str = ""
    head_prompt: str = ""
    subtask: str = ""
    remaining: list[str] = field(default_factory=list)
    subtask_started_at: float = 0.0
    speed_bin: int = -1
    on_last_subtask: bool = False


class EpisodeManager:
    def __init__(self, cfg: RolloutConfig, *, is_mvla: bool, prompt_config=None,
                 recognizer_available: bool = False, source: PlanSource | None = None):
        self.cfg = cfg
        self.is_mvla = is_mvla
        self.prompt_config = prompt_config
        self.recognizer_available = recognizer_available
        self.source = source
        self.status = IDLE
        self.reason = ""
        self.goal = ""
        self.task_instruction = ""
        self.tracker: SubtaskTracker | None = None
        self.completion: dict | None = None
        self.start_mode = ""
        self.seq = 0
        self.t_start = 0.0
        self._view = TaskStateView()
        self._last_event = ""

    # ─────────────────────────────────────────────────────── lifecycle
    def begin(self, now: float, goal: str = "", task_instruction: str = "",
              initial_state: str = "", gripper_effort: float = 0.0) -> list[TrackerEvent]:
        """에피소드 시작. 반환 이벤트에 인식기 예약이 섞여 있을 수 있다."""
        self.status = PROBING
        self.reason = ""
        self.completion = None
        self.goal = goal
        self.task_instruction = task_instruction
        self.tracker = None
        self.t_start = now
        self._last_event = ""
        if not self.is_mvla:
            # SmolVLA: 헤드도 플랜도 없다. 학습 때의 문장 하나로 끝까지 간다.
            if not task_instruction:
                raise ValueError("SmolVLA needs task_instruction (the sentence it was trained on)")
            self.status = ACTIVE
            self._bump(state=ACTIVE, goal="", policy_prompt=task_instruction, head_prompt="",
                       subtask="", remaining=[], subtask_started_at=now, on_last_subtask=True)
            logger.info(f"begin: smolvla {task_instruction!r}")
            return []

        mvla = self.cfg.mvla
        if initial_state:
            mvla = type(mvla)(**{**mvla.__dict__, "initial_state": initial_state})
        self.tracker = SubtaskTracker(goal, self.prompt_config, mvla, self.cfg.recognizer,
                                      recognizer_available=self.recognizer_available, source=self.source)
        self.start_mode = self.tracker.start_mode()
        logger.info(f"begin: mvla goal={goal!r} start_mode={self.start_mode} "
                    f"recognizer={self.recognizer_available}")
        if gripper_effort >= mvla.holding_effort or self.start_mode not in ("recognizer", "head"):
            # 쥐고 있거나 시작 상태가 고정이면 바로 plan
            return self._handle(self.tracker.begin_from_head(None, gripper_effort, now), now)
        if self.start_mode == "recognizer":
            events = self.tracker.schedule_start(now)
        else:
            # head: 정책에게 헤드만 읽으라고 한다. 첫 읽기로 시작 상태를 정한다.
            events = []
        self._refresh(now)
        return events

    def stop(self, reason: str = "operator") -> None:
        if self.active:
            self.status, self.reason = "stopped", reason
            self._bump(state=IDLE)
            logger.info(f"stop: {reason}")

    def abort(self, reason: str) -> None:
        if self.active:
            self.status, self.reason = "aborted", reason
            self._bump(state=BLOCKED)
            logger.warning(f"aborted: {reason}")

    @property
    def active(self) -> bool:
        return self.status in (PROBING, ACTIVE)

    @property
    def last_event(self) -> str:
        return self._last_event

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.t_start) if self.status != IDLE else 0.0

    # ───────────────────────────────────────────────────────── inputs
    def on_head(self, head: HeadRead, t_obs: float, task_seq: int,
                gripper_effort: float = 0.0) -> list[TrackerEvent]:
        tr = self.tracker
        if tr is None or not self.active:
            return []
        if not tr.started:
            if self.start_mode != "head":
                return []
            # effort를 다시 본다. 시작 판정 전에 사람이 부품을 쥐여줬을 수 있고,
            # 파지 여부는 vision이 아니라 그리퍼에서 읽는다.
            return self._handle(tr.begin_from_head(head, gripper_effort, t_obs), t_obs)
        if task_seq != self.seq:
            # 전진 직전 프롬프트로 계산된 읽기. 그 subtask에 대한 판단이지
            # 지금 subtask에 대한 판단이 아니다.
            logger.debug(f"dropping head read from seq {task_seq} (current {self.seq})")
            return []
        return self._handle(tr.observe(head, t_obs), t_obs)

    def on_verdict(self, v: Verdict, gripper_effort: float, now: float) -> list[TrackerEvent]:
        tr = self.tracker
        if tr is None or not self.active:
            return []
        return self._handle(tr.on_verdict(v, gripper_effort, now), now)

    def on_tick(self, gripper_position: float, now: float) -> list[TrackerEvent]:
        """정책 상태가 올 때마다. place 중 릴리스 순간을 찾는다."""
        tr = self.tracker
        if tr is None or not self.active or not tr.started:
            return []
        return self._handle(tr.watch(gripper_position, now), now)

    # ────────────────────────────────────────────────────────── output
    def task_state(self, now: float) -> TaskStateView:
        return self._view

    def snapshot(self, now: float) -> dict:
        snap = {
            "status": self.status,
            "reason": self.reason,
            "goal": self.goal or self.task_instruction,
            "episode_elapsed_s": round(self.elapsed(now), 2),
            "task_seq": self.seq,
            "task_complete": self.completion is not None,
        }
        if self.tracker is not None and self.tracker.started:
            snap.update(self.tracker.snapshot(now))
        if self.completion is not None:
            snap["completion"] = self.completion
        return snap

    # ──────────────────────────────────────────────────────── internals
    def _handle(self, events: list[TrackerEvent], now: float) -> list[TrackerEvent]:
        """tracker 이벤트를 매니저 상태로. 인식기 예약 이벤트는 그대로 흘려보낸다."""
        for ev in events:
            if ev.kind in ("schedule_recognition", "cancel_recognition"):
                continue
            self._last_event = f"{ev.kind}: {ev.detail}"
            logger.info(self._last_event)
            if ev.kind == "begin":
                self.status = ACTIVE
            elif ev.kind == "task_complete":
                self.completion = ev.data
                self.status, self.reason = DONE, f"task_complete: {ev.detail}"
                logger.info(f"task_complete {ev.data}")
            elif ev.kind == "needs_operator":
                self.status, self.reason = "needs_operator", f"manager: {ev.detail}"
                logger.warning(self.reason)
        self._refresh(now)
        return events

    def _refresh(self, now: float) -> None:
        """tracker 상태를 TaskStateView로. 바뀐 게 있을 때만 seq를 올린다."""
        tr = self.tracker
        if self.status == DONE:
            self._bump(state=DONE)
            return
        if self.status in ("needs_operator", "aborted"):
            self._bump(state=BLOCKED)
            return
        if self.status == "stopped" or not self.active:
            self._bump(state=IDLE)
            return
        if not self.is_mvla:
            return  # SmolVLA: begin()이 정한 문장 하나로 끝까지 간다
        if tr is None or not tr.started:
            # 플랜 전. head 모드면 정책이 헤드만 읽어야 한다.
            head = tr.main_prompt() if (tr is not None and self.start_mode == "head") else ""
            self._bump(state=PROBING, goal=self.goal, policy_prompt="", head_prompt=head,
                       subtask="", remaining=[], subtask_started_at=now, on_last_subtask=False)
            return
        remaining = tr.remaining
        self._bump(
            state=ACTIVE, goal=self.goal,
            policy_prompt=tr.policy_prompt(), head_prompt=tr.head_prompt(),
            subtask=tr.subtask or "", remaining=remaining,
            subtask_started_at=tr.t0, on_last_subtask=len(remaining) == 1,
            speed_bin=-1 if self.cfg.mvla.speed_bin is None else int(self.cfg.mvla.speed_bin),
        )

    def _bump(self, **fields) -> None:
        """내용이 바뀌었으면 seq를 올린다. 안 바뀌었으면 그대로 둔다.

        seq는 정책이 echo해서 낡은 헤드 읽기를 가리는 데 쓰인다. 내용이 같은데
        올리면 멀쩡한 읽기를 버리게 되고, 바뀌었는데 안 올리면 낡은 읽기로
        한 칸을 넘긴다.
        """
        current = {k: getattr(self._view, k) for k in fields}
        if current == fields:
            return
        new = TaskStateView(**{**{k: v for k, v in self._view.__dict__.items() if k != "seq"}, **fields})
        self.seq += 1
        new.seq = self.seq
        self._view = new
