"""인식기 노드: 예약된 시각에 장면을 판정한다.

    ros2 run max_ai max_ai_recognizer --ros-args -p config_file:=<yaml>

    /max_ai/recognizer/schedule   ScheduleRecognition.srv  (매니저가 부른다)
    /max_ai/recognizer/verdict    RecognitionVerdict.msg   (판정 하나당 1회)

세 시계 중 가장 느리고 가장 확실한 쪽이다. 정책의 value 헤드는 "잘 안 되고
있다"를 계속 말하지만, "부품이 포켓이 아니라 리브에 걸터앉았다"를 말할 수
있는 건 이것뿐이고, 재시도와 작업 완료를 결정하는 건 이 판정이다.

**카메라를 직접 구독한다.** 매니저가 예약 시각만 보내고, 그 시각에 프레임을
뜨는 건 여기다. 캡처 시각이 load-bearing이기 때문이다 -- probe는 경계에서
고정 오프셋만큼 떨어진 프레임으로 학습됐고(mat +3/12/21, tray +8/18/28),
다른 시점의 프레임으로 내린 판정은 학습 때 본 적 없는 장면에 대한 답이다.
경계가 지나가는 그 순간에 매니저가 이미지 3장을 서비스로 보내는 구조였다면
캡처 시각이 큐 지연만큼 밀린다.

모델이 없으면(설정의 `recognizer.path`가 비었으면) 예약을 거절한다. 매니저는
그걸 보고 검증 없이 돈다 -- 오지 않을 판정을 기다리게 하지 않는다.
"""

from __future__ import annotations

import threading

import rclpy
from max_interfaces.msg import RecognitionVerdict
from max_interfaces.srv import ScheduleRecognition
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ..rollout.observation import RawObservation, decode_compressed
from ..rollout.recognition import PROBE_OF, CaptureScheduler, Recognizer, RecognizerWorker
from .ros_util import (
    PURPOSE_CODE, PURPOSE_NAME, WarnOnce, declare_config, now_s, sensor_qos, to_sec, to_stamp,
)


class RecognizerNode(Node):
    def __init__(self):
        super().__init__("max_ai_recognizer")
        self.cfg = declare_config(self)
        rc = self.cfg.ros
        self.warn = WarnOnce(self)
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[CompressedImage, float]] = {}

        self.rec = None
        self.worker = None
        if self.cfg.recognizer.path:
            self.rec = Recognizer(self.cfg.recognizer.path, self.cfg.recognizer.device,
                                  tuple(rc.image_size), rc.image_rotate)
            missing = sorted(set(self.rec.cameras) - set(rc.cameras))
            if missing:
                raise ValueError(f"recognizer needs cameras {missing} that the config does not subscribe")
            self.worker = RecognizerWorker(self.rec)
            self._needed = list(self.rec.cameras)
        else:
            self.get_logger().warn("no recognizer configured: every schedule request will be refused")
            self._needed = []

        subs = ReentrantCallbackGroup()
        for name in self._needed:
            topic = rc.cameras[name]
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, _n=name: self._on_image(_n, msg), sensor_qos(rc.qos_depth), callback_group=subs,
            )
            self.get_logger().info(f"subscribe {name:<12} {topic}")

        self.scheduler = CaptureScheduler(self.cfg.recognizer.capture_tolerance_s)
        self.verdict_pub = self.create_publisher(RecognitionVerdict, rc.verdict_topic,
                                                 sensor_qos(rc.qos_depth))
        own = MutuallyExclusiveCallbackGroup()
        self.create_service(ScheduleRecognition, rc.schedule_recognition_service,
                            self._on_schedule, callback_group=own)
        # 예약 시각을 놓치지 않으려면 제어 주기와 같은 해상도로 봐야 한다.
        self.create_timer(1.0 / self.cfg.control.fps, self._on_tick, callback_group=own)
        self.get_logger().info(f"ready: {rc.schedule_recognition_service} -> {rc.verdict_topic}")

    def _on_image(self, name: str, msg: CompressedImage) -> None:
        with self._lock:
            self._frames[name] = (msg, now_s(self))

    # ──────────────────────────────────────────────────────── scheduling
    def _on_schedule(self, req, resp):
        if self.worker is None:
            resp.success, resp.message = False, "no recognizer model loaded"
            return resp
        purpose = PURPOSE_NAME.get(req.purpose)
        if purpose is None:
            resp.success, resp.message = False, f"unknown purpose {req.purpose}"
            return resp
        now = now_s(self)
        ref = to_sec(req.reference) or now
        rid, n, late = self.scheduler.schedule(purpose, ref, list(req.offsets_s), now,
                                               cancel_pending=req.cancel_pending)
        resp.request_id, resp.scheduled = rid, n
        resp.success = n > 0
        if late:
            resp.message = f"{len(late)} capture(s) already past by up to {now - min(late):.3f}s"
            self.get_logger().warn(f"{purpose}: {resp.message}")
        if n == 0:
            resp.message = resp.message or "every requested capture time has already passed"
            return resp
        self.get_logger().info(
            f"scheduled {purpose} request {rid}: {n} capture(s) at "
            f"+{[round(o, 3) for o in req.offsets_s]}s from {ref:.3f}"
        )
        return resp

    def _on_tick(self) -> None:
        if self.worker is None:
            return
        now = now_s(self)
        due = self.scheduler.pop_due(now)
        if due:
            raw = self._capture()
            for cap in due:
                if raw is None:
                    self.warn("capture skipped: cameras not ready")
                    continue
                self.worker.submit(cap, raw, now)
        for v in self.worker.poll(now):
            self._publish(v)

    def _capture(self) -> RawObservation | None:
        """지금 순간의 프레임. 인식기가 쓰는 카메라만 디코드한다."""
        with self._lock:
            latest = {n: self._frames.get(n) for n in self._needed}
        if any(v is None for v in latest.values()):
            return None
        raw = RawObservation(stamp=min(recv for _, recv in latest.values()))
        for name, (msg, _recv) in latest.items():
            img = decode_compressed(bytes(msg.data))
            if img is None:
                self.warn(f"{name}: undecodable frame")
                return None
            raw.images[name] = img
        return raw

    def _publish(self, v) -> None:
        msg = RecognitionVerdict()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.purpose = PURPOSE_CODE[v.purpose]
        msg.probe = PROBE_OF[v.purpose]
        msg.request_id = v.request_id
        msg.index, msg.total = v.index, v.total
        msg.captured_at = to_stamp(v.t_obs)
        msg.label = str(v.result.get("label", "abstain"))
        probs = v.result.get("probs", {}) or {}
        msg.prob_names = list(probs)
        msg.prob_values = [float(probs[k]) for k in msg.prob_names]
        msg.ood = float(v.result.get("ood", float("nan")))
        msg.detected = bool(v.result.get("detected", True))
        self.verdict_pub.publish(msg)
        self.get_logger().info(
            f"verdict {v.purpose} {v.index + 1}/{v.total} (request {v.request_id}): "
            f"{msg.label} {dict(zip(msg.prob_names, [round(p, 3) for p in msg.prob_values]))} "
            f"ood={msg.ood:.1f}"
        )

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = RecognizerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
