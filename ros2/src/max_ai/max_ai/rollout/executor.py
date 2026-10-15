"""정책측 30Hz 루프. ROS 없음 -- policy 노드와 replay가 같은 코드를 쓴다.

`ros_inference.controller.RolloutController`를 둘로 가른 쪽 중 정책측이다.
나머지 절반(플랜·subtask 전진·검증·종료)은 `subtask.SubtaskTracker`에 있고
매니저 노드가 돌린다. 경계는 **프롬프트**다:

    매니저 ──PromptSpec(프롬프트, subtask 시작 시각, seq)──> PolicyExecutor
    매니저 <──ExecTick(헤드 읽기, 상태, 안전 플래그)────────┘

구조:

    tick(now, raw, prompt) ── 30Hz, 절대 막히지 않는다
      ├─ worker.poll(now)      끝난 추론 결과 수거 (청크는 실행기로, 헤드는 호출자에게)
      ├─ 정지 조건 판단         데드맨, 관측 끊김, 에피소드 타임아웃
      ├─ worker.submit(...)    다음 청크가 곧 필요하면 / 헤드 읽을 때가 되면
      └─ executor.action_at    지금 시각의 액션 -> 안전 제한 -> 명령

    worker ── 별도 스레드. 관측 -> 정책 -> 결과. 상태를 바꾸지 않는다.

모든 상태 변경은 tick 안에서만 일어난다. 그래서 잠금이 필요 없고, replay는
스레드 대신 가상 시계로 같은 순서를 재현할 수 있다.

**시간 기준 청크 실행.** 청크의 k번째 액션은 그 청크를 만든 관측 시각
`t_obs + k/fps`에 해당한다(데이터셋에서 프레임 t의 액션이 프레임 t의
관측과 같은 시각이듯이). 추론에 L초가 걸리면 새 청크는 인덱스 L*fps부터
쓰인다 -- 지연 보상이 따로 없이 저절로 된다. 다음 청크는 실행 지점
(`execute_steps`)보다 추론 지연만큼 앞서 요청하고, 도착하면 `blend_steps`
동안 이전 청크와 선형 혼합한다. 새 청크가 끝내 안 오면 이전 청크를
끝까지(50스텝) 쓰고, 그것도 다 쓰면 정지한다.

**프롬프트가 바뀌어도 날아오던 청크는 그대로 쓴다.** flip 검증이 그 동작에
기대고 있다 -- 판정이 나오는 ~1초 동안 정책은 이미 approach_pick으로
움직여야 하고, flip 프롬프트에 붙잡아 두면 다시 뒤집으려 한다. 그래서 여기서
프롬프트 변경을 청크 무효화로 취급하지 않는다. 대신 각 결과에 그 프롬프트의
`task_seq`를 달아 돌려주고, 낡은 헤드 읽기를 버리는 건 매니저가 한다.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .action import ActionSafety, Command
from .config import RolloutConfig
from .observation import RawObservation, to_policy_observation
from .policy_runner import HeadRead, PolicyRunner

logger = logging.getLogger("max_ai.executor")

# PromptSpec.mode -- TaskState.state와 같은 뜻이지만 ROS 상수를 안 쓴다.
IDLE = "idle"
ACTIVE = "active"      # 프롬프트대로 움직인다
PROBING = "probing"    # 플랜 전. 헤드만 읽고 팔은 멈춰 있다
HALT = "halt"          # 매니저가 끝났다고 했다 (done | blocked)


# ─────────────────────────────────────────────────────────── chunk timing
class ChunkExecutor:
    def __init__(self, fps: float, execute_steps: int, blend_steps: int):
        self.fps = fps
        self.execute_steps = execute_steps
        self.blend_steps = blend_steps
        self.reset()

    def reset(self) -> None:
        self.chunk: np.ndarray | None = None
        self.t_obs = 0.0
        self._prev: tuple[np.ndarray, float] | None = None
        self._blend_start = 0.0

    def _index(self, now: float, t_obs: float) -> int:
        return int(math.floor((now - t_obs) * self.fps + 1e-6))

    def push(self, chunk: np.ndarray, t_obs: float, now: float) -> None:
        if self.chunk is not None and self.blend_steps > 0:
            self._prev = (self.chunk, self.t_obs)
            self._blend_start = now
        self.chunk, self.t_obs = chunk, t_obs

    def index(self, now: float) -> int:
        return self._index(now, self.t_obs) if self.chunk is not None else -1

    def action_at(self, now: float) -> np.ndarray | None:
        if self.chunk is None:
            return None
        i = max(0, self._index(now, self.t_obs))
        if i >= len(self.chunk):
            return None
        a = self.chunk[i]
        if self._prev is not None:
            w = (now - self._blend_start) * self.fps / self.blend_steps
            prev, prev_t = self._prev
            j = self._index(now, prev_t)
            if w < 1.0 and 0 <= j < len(prev):
                a = w * a + (1.0 - w) * prev[j]
            else:
                self._prev = None
        return a

    def needs_chunk(self, now: float, latency_s: float) -> bool:
        if self.chunk is None:
            return True
        lead = int(math.ceil(latency_s * self.fps))
        return self._index(now, self.t_obs) + lead >= self.execute_steps


# ───────────────────────────────────────────────────────────── inference
@dataclass
class Request:
    raw: RawObservation
    want_chunk: bool
    task: str = ""
    head_task: str | None = None
    elapsed: float | None = None
    task_seq: int = 0
    submitted: float = 0.0


@dataclass
class Result:
    t_obs: float
    chunk: np.ndarray | None
    head: HeadRead | None
    task: str
    compute_s: float
    task_seq: int = 0
    ready: float = 0.0


def compute(runner: PolicyRunner, cfg: RolloutConfig, req: Request) -> Result:
    t0 = time.perf_counter()
    obs = to_policy_observation(
        req.raw, runner.camera_names, tuple(cfg.ros.image_size), cfg.ros.image_rotate, cfg.ros.gripper_effort_max
    )
    head = runner.read_heads(obs, req.head_task, req.elapsed) if req.head_task is not None else None
    chunk = runner.infer_chunk(obs, req.task, req.elapsed) if req.want_chunk else None
    return Result(t_obs=req.raw.stamp, chunk=chunk, head=head, task=req.task,
                  compute_s=time.perf_counter() - t0, task_seq=req.task_seq)


class ThreadedWorker:
    """실기용. 요청 하나를 받아 백그라운드에서 계산한다."""

    def __init__(self, runner: PolicyRunner, cfg: RolloutConfig, clock: Callable[[], float] = time.monotonic):
        self.runner, self.cfg, self.clock = runner, cfg, clock
        self._in: queue.Queue[Request | None] = queue.Queue(maxsize=1)
        self._out: queue.Queue[Result] = queue.Queue()
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self, req: Request, now: float) -> bool:
        if self.busy:
            return False
        req.submitted = now
        self._busy.set()
        self._in.put(req)
        return True

    def poll(self, now: float) -> list[Result]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def _loop(self) -> None:
        while True:
            req = self._in.get()
            if req is None:
                return
            try:
                res = compute(self.runner, self.cfg, req)
                res.ready = self.clock()
                self._out.put(res)
            except Exception:  # noqa: BLE001 - 추론 실패가 제어 루프를 죽이면 안 된다
                logger.exception("inference failed")
            finally:
                self._busy.clear()

    def close(self) -> None:
        self._in.put(None)


class SimulatedWorker:
    """replay용. 즉시 계산하되, 결과는 가상 시계로 지연 후에 내준다.

    `latency_s`가 None이면 실제 계산 시간을 지연으로 쓴다.
    """

    def __init__(self, runner: PolicyRunner, cfg: RolloutConfig, latency_s: float | None = None):
        self.runner, self.cfg, self.latency_s = runner, cfg, latency_s
        self._pending: Result | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def submit(self, req: Request, now: float) -> bool:
        if self.busy:
            return False
        res = compute(self.runner, self.cfg, req)
        res.ready = now + (res.compute_s if self.latency_s is None else self.latency_s)
        self._pending = res
        return True

    def poll(self, now: float) -> list[Result]:
        if self._pending is not None and now >= self._pending.ready:
            res, self._pending = self._pending, None
            return [res]
        return []

    def close(self) -> None:
        pass


# ───────────────────────────────────────────────────────── prompt / tick
@dataclass
class PromptSpec:
    """매니저가 정한 것. policy 노드는 TaskState 메시지에서 이걸 채운다.

    SmolVLA나 매니저 없이 도는 경우는 `standalone=True`로 만들어 데드맨을
    끈다 -- 아무도 갱신하지 않는 프롬프트를 끊겼다고 판단하면 안 된다.
    """

    mode: str = IDLE
    seq: int = 0
    policy_prompt: str = ""
    head_prompt: str | None = None
    # subtask 시작 시각. `observation.elapsed_subtask`를 여기서 직접 계산한다
    # -- 매니저에게 5Hz로 물어보면 그 사이 청크마다 낡은 값이 들어간다.
    subtask_started_at: float = 0.0
    # 이 프롬프트를 받은 시각. 데드맨 판정용.
    received_at: float = 0.0
    standalone: bool = False


@dataclass
class ExecTick:
    command: Command | None       # None = 아무것도 보내지 않는다 (idle)
    status: str
    reason: str = ""
    # 이번 틱에 도착한 헤드 읽기. (head, t_obs, task_seq) -- 매니저가 task_seq로
    # 낡은 읽기를 버린다.
    heads: list[tuple[HeadRead, float, int]] = field(default_factory=list)
    action: np.ndarray | None = None
    chunk_index: int = -1
    latency_s: float = 0.0
    observation_age_s: float = 0.0
    task_seq: int = 0


class PolicyExecutor:
    """상태: idle -> starting -> running -> (stopped | aborted).

    에피소드가 **끝났는지**는 여기서 판단하지 않는다. 그건 플랜의 마지막
    subtask가 끝났는지의 문제이고 매니저의 몫이다. 여기서 끝내는 건 스스로
    멈춰야 하는 경우뿐이다 -- 관측 끊김, 에피소드 타임아웃, 데드맨.
    """

    def __init__(self, cfg: RolloutConfig, runner: PolicyRunner, worker,
                 obs_source: Callable[[bool], RawObservation | None],
                 task_state_timeout_s: float = 1.0):
        self.cfg = cfg
        self.runner = runner
        self.worker = worker
        self.obs_source = obs_source
        self.task_state_timeout_s = task_state_timeout_s
        self.executor = ChunkExecutor(cfg.control.fps, cfg.control.execute_steps, cfg.control.blend_steps)
        self.safety = ActionSafety(cfg.safety, cfg.control.fps)
        self.status = "idle"
        self.reason = ""
        self.latency_s = 0.15
        self.ticks = 0
        self.moving_ticks = 0
        self.log: list[dict] = []
        self.last_head: HeadRead | None = None
        self.head_seq = 0
        self.task_seq = 0
        self._t_start = 0.0
        self._max_episode_s = cfg.control.max_episode_s
        self._last_head_req = -1e9

    # ─────────────────────────────────────────────────────── lifecycle
    def begin(self, now: float, max_episode_s: float | None = None) -> None:
        self.executor.reset()
        raw = self.obs_source(False)
        self.safety.reset(gripper_open=1.0 if raw is None or raw.gripper_position >= 0.5 else 0.0)
        self.status = "starting"
        self.reason = ""
        self.log = []
        self.ticks = self.moving_ticks = 0
        self.last_head = None
        self.head_seq = 0
        self.task_seq = 0
        self._t_start = now
        self._last_head_req = -1e9
        self._max_episode_s = max_episode_s or self.cfg.control.max_episode_s
        logger.info(f"begin: {'mvla' if self.runner.is_mvla else 'smolvla'} "
                    f"speed_scale={self.cfg.safety.speed_scale} max_episode_s={self._max_episode_s}")

    def stop(self, reason: str = "operator") -> None:
        if self.active:
            self.status = "stopped"
            self.reason = reason
            logger.info(f"stop: {reason}")

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")

    def _abort(self, reason: str) -> None:
        self.status, self.reason = "aborted", reason
        logger.warning(f"aborted: {reason}")

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self._t_start) if self.status != "idle" else 0.0

    # ──────────────────────────────────────────────────────────── tick
    def tick(self, now: float, prompt: PromptSpec) -> ExecTick:
        if not self.active:
            for _ in self.worker.poll(now):
                pass  # 멈춘 뒤 도착한 결과는 버린다
            cmd = None if self.status == "idle" else self.safety.stop()
            return ExecTick(cmd, self.status, self.reason, latency_s=self.latency_s)

        self.ticks += 1
        raw = self.obs_source(False)
        heads: list[tuple[HeadRead, float, int]] = []
        for res in self.worker.poll(now):
            self.latency_s = 0.8 * self.latency_s + 0.2 * max(0.0, res.ready - res.t_obs)
            if res.head is not None:
                self.last_head = res.head
                self.head_seq += 1
                heads.append((res.head, res.t_obs, res.task_seq))
            if res.chunk is not None:
                self.executor.push(res.chunk, res.t_obs, now)
                self.task_seq = res.task_seq

        # 매니저가 끝났다고 했으면 바로 멈춘다. 이 틱의 헤드는 그대로 돌려줘서
        # 매니저가 자기 종료 판정의 근거를 잃지 않게 한다.
        if prompt.mode == HALT:
            self.stop("task state halted")
            return self._out(self.safety.stop(), heads, raw, now, prompt)
        # 데드맨. 매니저가 죽으면 마지막 프롬프트로 팔이 계속 움직인다.
        if not prompt.standalone and now - prompt.received_at > self.task_state_timeout_s:
            self._abort(f"task state stale ({now - prompt.received_at:.1f}s)")
            return self._out(self.safety.stop(), heads, raw, now, prompt)

        if raw is None or now - raw.stamp > self.cfg.control.stale_timeout_s:
            # 관측이 끊기면 멈추되 에피소드는 유지한다. 돌아오면 이어간다.
            cmd = self.safety.stop()
            cmd.flags = cmd.flags + ("stale_observation",)
            return self._out(cmd, heads, raw, now, prompt)
        if now - self._t_start > self._max_episode_s:
            self._abort("episode timeout")
            return self._out(self.safety.stop(), heads, raw, now, prompt)

        if prompt.mode == ACTIVE and prompt.policy_prompt:
            self.status = "running"
        self._schedule(now, prompt)

        action = self.executor.action_at(now) if self.status == "running" else None
        cmd = self.safety(action, raw.eef_position) if action is not None else self.safety.stop()
        if not cmd.is_zero:
            self.moving_ticks += 1
        return self._out(cmd, heads, raw, now, prompt, action)

    def _schedule(self, now: float, prompt: PromptSpec) -> None:
        if self.worker.busy:
            return
        if prompt.mode == PROBING:
            # 플랜 전. 시작 상태를 정할 헤드 읽기만. 청크는 플랜이 선 뒤에.
            if prompt.head_prompt:
                self._submit(now, prompt, want_chunk=False, elapsed=0.0)
            return
        if prompt.mode != ACTIVE or not prompt.policy_prompt:
            return
        elapsed = max(0.0, now - prompt.subtask_started_at)
        if self.executor.needs_chunk(now, self.latency_s):
            self._submit(now, prompt, want_chunk=True, elapsed=elapsed)
            return
        if prompt.head_prompt and now - self._last_head_req >= 1.0 / max(self.cfg.mvla.head_hz, 1e-3):
            self._submit(now, prompt, want_chunk=False, elapsed=elapsed)

    def _submit(self, now: float, prompt: PromptSpec, *, want_chunk: bool, elapsed: float) -> None:
        raw = self.obs_source(True)
        if raw is None:
            return
        req = Request(raw=raw, want_chunk=want_chunk, task=prompt.policy_prompt,
                      head_task=prompt.head_prompt or None, elapsed=elapsed, task_seq=prompt.seq)
        if self.worker.submit(req, now) and req.head_task is not None:
            self._last_head_req = now

    def _out(self, cmd: Command, heads, raw, now: float, prompt: PromptSpec,
             action=None) -> ExecTick:
        age = float("inf") if raw is None else max(0.0, now - raw.stamp)
        h = self.last_head
        self.log.append({
            "t": round(now - self._t_start, 4),
            "status": self.status,
            "task_seq": prompt.seq,
            "prompt": prompt.policy_prompt,
            "chunk_index": self.executor.index(now),
            "latency_s": self.latency_s,
            "observation_age_s": None if raw is None else round(age, 4),
            "action": None if action is None else [float(x) for x in action],
            "command": [float(x) for x in cmd.twist],
            "gripper": cmd.gripper_target,
            "flags": list(cmd.flags),
            "value_subtask": None if h is None else h.value_subtask,
            "value_episode": None if h is None else h.value_episode,
            "head_subtask": None if h is None else h.subtask,
        })
        return ExecTick(
            command=cmd, status=self.status, reason=self.reason, heads=heads, action=action,
            chunk_index=self.executor.index(now), latency_s=self.latency_s,
            observation_age_s=age, task_seq=prompt.seq,
        )
