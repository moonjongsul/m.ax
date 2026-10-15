"""상태 인식기 실행: 정책과 별도 모델, 별도 스레드.

설계 문서 §5.3의 세 번째 시계다. 정책(30Hz)과 value 헤드는 "잘 안 되고 있다"를
빠르게 말하고, 인식기는 경계에서 "부품이 제대로 놓였나 / 뒤집혔나"를 확정한다.
재시도나 완료를 결정하는 건 이 판정이다.

인식기는 학습 샘플과 같은 시점의 프레임을 봐야 한다(`recognizer/dataset.py`):

    mat   approach 런 시작 +3/12/21프레임 -- 팔이 아직 부품에서 떨어져 있을 때
    tray  그리퍼가 열린 순간 +8/18/28프레임 -- 손목이 아직 트레이 가까이 있을 때

그래서 매니저가 캡처 시각을 정해 예약하고, **인식기 노드가 그 시각에 자기
카메라 버퍼에서 프레임을 뜬다.** 매니저가 그 순간에 프레임을 보내는 구조였다면
경계가 지나가는 바로 그 순간에 노드 간 왕복이 끼어 캡처 시각이 밀린다.
계산(DINOv2 + 선형 probe, 이미지 2장)은 별도 스레드에서 한다.

ROS를 import하지 않는다. `CaptureScheduler`는 노드와 replay가 같이 쓴다.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .observation import RawObservation, prepare_image

logger = logging.getLogger("max_ai.recognition")

PROBE_OF = {"start": "mat", "flip": "mat", "place": "tray"}


@dataclass
class Verdict:
    purpose: str          # start | flip | place
    t_obs: float
    result: dict          # {"label", "probs", "ood", "detected"}
    # 어느 예약의 몇 번째 캡처인지. 매니저는 request_id로 재계획에 버려진
    # 경계의 판정을 가리고, index+1 == total일 때 집계한다.
    request_id: int = 0
    index: int = 0
    total: int = 1


@dataclass(order=True)
class _Capture:
    """예약된 캡처 하나. `due`로 정렬된다."""

    due: float
    purpose: str = field(compare=False)
    request_id: int = field(compare=False, default=0)
    index: int = field(compare=False, default=0)
    total: int = field(compare=False, default=1)


class CaptureScheduler:
    """예약 시각 관리. 순수 로직 -- 노드와 replay가 같이 쓴다.

    `tolerance_s`는 이미 지나버린 예약을 받아줄 한계다. 그보다 더 지난
    요청은 거절한다 -- 학습 때와 다른 시점의 프레임으로 내린 판정은 틀린 게
    아니라 다른 질문에 대한 답이고, 그걸로 재시도를 결정하면 안 된다.
    """

    def __init__(self, tolerance_s: float = 0.05):
        self.tolerance_s = tolerance_s
        self._pending: list[_Capture] = []
        self._next_id = 1

    def schedule(self, purpose: str, reference: float, offsets_s: list[float], now: float,
                 cancel_pending: bool = False) -> tuple[int, int, list[float]]:
        """예약. (request_id, 받아들인 개수, 버린 시각들)."""
        if cancel_pending:
            self.cancel()
        rid = self._next_id
        self._next_id += 1
        want = [reference + float(o) for o in offsets_s]
        late = [d for d in want if d < now - self.tolerance_s]
        keep = [d for d in want if d >= now - self.tolerance_s]
        for i, due in enumerate(keep):
            self._pending.append(_Capture(due, purpose, rid, i, len(keep)))
        self._pending.sort()
        if late:
            logger.warning(f"{purpose}: dropped {len(late)} capture(s) already past "
                           f"(by up to {now - min(late):.3f}s)")
        return rid, len(keep), late

    def cancel(self, purposes: list[str] | None = None) -> int:
        before = len(self._pending)
        if purposes is None:
            self._pending = []
        else:
            self._pending = [c for c in self._pending if c.purpose not in purposes]
        return before - len(self._pending)

    def pop_due(self, now: float) -> list[_Capture]:
        due = [c for c in self._pending if c.due <= now]
        self._pending = [c for c in self._pending if c.due > now]
        return due

    @property
    def pending(self) -> int:
        return len(self._pending)


class Recognizer:
    def __init__(self, path: str | Path, device: str = "cuda", image_size=(640, 480), rotate=None):
        from lerobot.policies.mvla.recognizer.model import StateRecognizer

        self.model = StateRecognizer.load(Path(path).expanduser())
        self.model.cfg.device = device
        self.image_size = tuple(image_size)
        self.rotate = rotate or {}
        _ = self.model.encoder  # 백본을 미리 올려 첫 판정이 늦지 않게
        logger.info(f"recognizer {path}: probes {sorted(self.model.probes)} cameras {self.model.cfg.cameras}")

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(self.model.cfg.cameras)

    def __call__(self, task: str, raw: RawObservation) -> dict:
        # 녹화와 같은 640x480 BGR로 맞춘 뒤 넘긴다. 크롭은 모델이 한다.
        views = {c: prepare_image(raw.images[c], self.image_size, self.rotate.get(c, 0)) for c in self.cameras}
        out = self.model(task, views)
        if task == "mat" and not out["detected"]:
            # 부품 실루엣을 못 찾으면 크롭이 전체 프레임으로 떨어진다 -- 학습에
            # 없던 입력이므로 판정하지 않는다.
            out["label"] = "abstain"
        out["probs"] = {k: float(v) for k, v in out["probs"].items()}
        return out


class RecognizerWorker:
    """실기용. 캡처된 관측을 받아 백그라운드에서 판정한다."""

    def __init__(self, recognizer: Recognizer):
        self.rec = recognizer
        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue[Verdict] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="recognizer", daemon=True)
        self._thread.start()

    def submit(self, cap: _Capture, raw: RawObservation, now: float) -> None:
        self._in.put((cap, raw))

    def poll(self, now: float) -> list[Verdict]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def _loop(self) -> None:
        while True:
            item = self._in.get()
            if item is None:
                return
            cap, raw = item
            try:
                self._out.put(Verdict(cap.purpose, raw.stamp, self.rec(PROBE_OF[cap.purpose], raw),
                                      request_id=cap.request_id, index=cap.index, total=cap.total))
            except Exception:  # noqa: BLE001
                logger.exception("recognizer failed")

    def close(self) -> None:
        self._in.put(None)


class SyncRecognizerWorker:
    """replay용. 즉시 계산하고 `latency_s` 뒤에 내준다."""

    def __init__(self, recognizer: Recognizer, latency_s: float = 0.05):
        self.rec = recognizer
        self.latency_s = latency_s
        self._pending: list[tuple[float, Verdict]] = []

    def submit(self, cap: _Capture, raw: RawObservation, now: float) -> None:
        v = Verdict(cap.purpose, raw.stamp, self.rec(PROBE_OF[cap.purpose], raw),
                    request_id=cap.request_id, index=cap.index, total=cap.total)
        self._pending.append((now + self.latency_s, v))

    def poll(self, now: float) -> list[Verdict]:
        ready = [v for t, v in self._pending if t <= now]
        self._pending = [(t, v) for t, v in self._pending if t > now]
        return ready

    def close(self) -> None:
        pass


def aggregate(results: list[dict]) -> dict:
    """여러 프레임 판정을 하나로: 확률 평균, 과반이 기권이면 기권."""
    if not results:
        return {"label": "abstain", "probs": {}, "ood": float("nan")}
    abstain = sum(r["label"] == "abstain" for r in results)
    keys = list(results[0]["probs"])
    probs = {k: float(np.mean([r["probs"][k] for r in results])) for k in keys}
    label = "abstain" if abstain * 2 > len(results) else max(probs, key=probs.get)
    return {"label": label, "probs": probs, "ood": float(np.median([r["ood"] for r in results])),
            "frames": len(results)}
