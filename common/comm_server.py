"""CommServer — ROS ↔ ZMQ 브릿지 서버

ZMQ 포트 규칙:
  ROS→ZMQ  PUB :5570+domain_id  robot/gripper (pickle, JointState/Float32)
  ROS→ZMQ  PUB :5572            카메라 (lerobot JSON 포맷) ← Streamlit + ModelServer 공용
  ZMQ→ROS  PULL :5590           제어 명령 (web 버튼 / 모델 액션)
  제어권 알림 PUB :5591          granted:<source> / revoked:<source>

카메라 JSON 포맷 (lerobot ZMQCamera 호환):
  {"timestamps": {"cam_name": float, ...}, "images": {"cam_name": "<base64-jpeg>", ...}}

제어 메시지 포맷 (client → CommServer PULL):
  [source, topic, pickled_msg]
  예: [b"web", b"/gello/joint_states", pickled_JointState]
"""

import base64
import json
import threading
import time
import pickle

import cv2
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import zmq

from common.utils.ros_utils import (
    build_ros2zmq_subscriber,
    resolve_msg_type,
    _publisher,
)

ZMQ_BASE_PUB_PORT  = 5570   # ROS → ZMQ PUB (robot/gripper, per domain_id)
ZMQ_CAM_PUB_PORT   = 5572   # ROS → ZMQ PUB (카메라, lerobot JSON 포맷)
ZMQ_BASE_PULL_PORT = 5590   # client → CommServer PULL (제어)
ZMQ_CTRL_PORT      = 5591   # CommServer → client PUB (제어권 알림)

LOCK_TIMEOUT = 5.0


# ─── CameraJsonPublisher ───────────────────────────────────────────────────────

class CameraJsonPublisher:
    """카메라 CompressedImage 토픽들을 수집해 lerobot JSON 포맷으로 PUB 발행.

    포맷: {"timestamps": {name: ts}, "images": {name: base64_jpeg}}
    모든 카메라를 하나의 메시지에 묶어서 발행 (단일 포트 :5572).
    """

    def __init__(self, zmq_context: zmq.Context):
        self._pub = zmq_context.socket(zmq.PUB)
        self._pub.bind(f'tcp://*:{ZMQ_CAM_PUB_PORT}')

        self._lock = threading.Lock()
        self._frames: dict[str, tuple[np.ndarray, float]] = {}  # name → (bgr, ts)
        self._cam_names: list[str] = []

    def register(self, cam_name: str, rotate: int = 0):
        """카메라 이름 등록 (토픽 콜백 연결 전 호출)."""
        self._cam_names.append(cam_name)
        return self._make_callback(cam_name, rotate)

    def _make_callback(self, cam_name: str, rotate: int):
        """CompressedImage ROS 콜백 → 프레임 버퍼에 저장 후 publish."""
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        def callback(msg):
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            rot = rotate_map.get(rotate)
            if rot is not None:
                img = cv2.rotate(img, rot)
            ts = time.time()
            with self._lock:
                self._frames[cam_name] = (img, ts)
            self._publish()

        return callback

    def _publish(self):
        with self._lock:
            if not self._frames:
                return
            frames_copy = dict(self._frames)

        timestamps = {}
        images = {}
        for name, (bgr, ts) in frames_copy.items():
            timestamps[name] = ts
            # BGR 그대로 JPEG 인코딩. 수신 측(웹/ModelServer)이 각자 변환.
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            images[name] = base64.b64encode(buf).decode()

        msg = json.dumps({"timestamps": timestamps, "images": images})
        self._pub.send_string(msg)

    def close(self):
        self._pub.close()


# ─── DomainBridge ─────────────────────────────────────────────────────────────

class DomainBridge:
    """단일 ROS domain_id에 대한 ROS→ZMQ 브릿지."""

    def __init__(self, domain_id: int, zmq_context: zmq.Context,
                 cam_publisher: CameraJsonPublisher | None = None):
        self.domain_id = domain_id
        self._zmq_ctx = zmq_context
        self._cam_pub = cam_publisher  # domain_id=1(카메라)일 때만 사용

        from rclpy._rclpy_pybind11 import SignalHandlerOptions
        self._ros_context = Context()
        rclpy.init(
            context=self._ros_context,
            domain_id=domain_id,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self._node = Node(
            f'comm_server_domain{domain_id}',
            context=self._ros_context,
        )

        # robot/gripper용 pickle PUB (카메라 도메인에서는 바인드하지 않음)
        pub_port = ZMQ_BASE_PUB_PORT + domain_id
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(f'tcp://*:{pub_port}')

        self._ros_publishers: dict[str, object] = {}

    def add_ros2zmq(self, topic: str, msg_type_str: str,
                    cam_name: str | None = None, rotate: int = 0):
        """ROS 토픽 → ZMQ 구독 등록.

        cam_name이 주어지면 CameraJsonPublisher 콜백으로 연결,
        아니면 기존 pickle PUB으로 연결.
        """
        msg_type = resolve_msg_type(msg_type_str)
        if cam_name and self._cam_pub:
            callback = self._cam_pub.register(cam_name, rotate)
            self._node.create_subscription(msg_type, topic, callback, 10)
        else:
            build_ros2zmq_subscriber(self._node, topic, msg_type, self._zmq_pub)

    def add_ros_publisher(self, topic: str, msg_type_str: str):
        msg_type = resolve_msg_type(msg_type_str)
        self._ros_publishers[topic] = _publisher(self._node, topic, msg_type)

    def publish(self, topic: str, raw: bytes):
        pub = self._ros_publishers.get(topic)
        if pub is None:
            return
        msg = pickle.loads(raw)
        pub.publish(msg)

    def start(self):
        self._executor = SingleThreadedExecutor(context=self._ros_context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name=f'bridge-domain{self.domain_id}')
        self._spin_thread.start()

    def shutdown(self):
        self._executor.shutdown()
        self._node.destroy_node()
        rclpy.shutdown(context=self._ros_context)
        self._zmq_pub.close()


# ─── Arbiter ──────────────────────────────────────────────────────────────────

class Arbiter:
    """제어 메시지를 수신하고 제어권을 중재해 ROS publish를 대행."""

    def __init__(self, zmq_context: zmq.Context,
                 bridges: dict[int, DomainBridge],
                 topic_domain_map: dict[str, int]):
        self._ctx = zmq_context
        self._bridges = bridges
        self._topic_domain = topic_domain_map

        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.bind(f'tcp://*:{ZMQ_BASE_PULL_PORT}')
        self._pull.setsockopt(zmq.RCVTIMEO, 200)

        self._ctrl_pub = self._ctx.socket(zmq.PUB)
        self._ctrl_pub.bind(f'tcp://*:{ZMQ_CTRL_PORT}')

        self._lock_source: str | None = None
        self._lock_time: float = 0.0
        self._lock = threading.Lock()
        self._running = False

    def _notify(self, event: str, source: str):
        self._ctrl_pub.send_multipart([
            b"control",
            f"{event}:{source}".encode(),
        ])

    def _acquire(self, source: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if (self._lock_source is not None and
                    now - self._lock_time > LOCK_TIMEOUT):
                print(f"[Arbiter] lock timeout, releasing '{self._lock_source}'")
                self._notify("revoked", self._lock_source)
                self._lock_source = None

            if self._lock_source is None:
                self._lock_source = source
                self._lock_time = now
                self._notify("granted", source)
                print(f"[Arbiter] '{source}' granted control")
                return True

            if self._lock_source == source:
                self._lock_time = now
                return True

            prev = self._lock_source
            self._notify("revoked", prev)
            print(f"[Arbiter] '{prev}' revoked, '{source}' granted control")
            self._lock_source = source
            self._lock_time = now
            self._notify("granted", source)
            return True

    def release(self, source: str):
        """추론 정지 시 즉시 제어권 해제 (ModelServer → stop 명령 수신 후 호출)."""
        with self._lock:
            if self._lock_source == source:
                self._notify("revoked", source)
                self._lock_source = None
                print(f"[Arbiter] '{source}' released control")

    def _run(self):
        while self._running:
            try:
                parts = self._pull.recv_multipart()
            except zmq.Again:
                with self._lock:
                    if (self._lock_source is not None and
                            time.monotonic() - self._lock_time > LOCK_TIMEOUT):
                        print(f"[Arbiter] lock timeout, releasing '{self._lock_source}'")
                        self._notify("revoked", self._lock_source)
                        self._lock_source = None
                continue

            if len(parts) != 3:
                continue

            source, topic_b, raw = parts
            source = source.decode()
            topic  = topic_b.decode()

            if not self._acquire(source):
                continue

            domain_id = self._topic_domain.get(topic)
            if domain_id is None:
                continue

            bridge = self._bridges.get(domain_id)
            if bridge is None:
                continue

            bridge.publish(topic, raw)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name='arbiter')
        self._thread.start()

    def shutdown(self):
        self._running = False
        self._pull.close()
        self._ctrl_pub.close()


# ─── CommServer ───────────────────────────────────────────────────────────────

class CommServer:
    """cfg를 읽어 domain_id별 DomainBridge + CameraJsonPublisher + Arbiter 구성."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._zmq_ctx = zmq.Context()
        self._bridges: dict[int, DomainBridge] = {}
        self._cam_pub: CameraJsonPublisher | None = None
        self._arbiter: Arbiter | None = None
        self._topic_domain: dict[str, int] = {}

    def _get_bridge(self, domain_id: int) -> DomainBridge:
        if domain_id not in self._bridges:
            # 카메라 도메인(domain_id=1)에는 CameraJsonPublisher 연결
            cam_pub = self._cam_pub if domain_id == 1 else None
            self._bridges[domain_id] = DomainBridge(domain_id, self._zmq_ctx, cam_pub)
        return self._bridges[domain_id]

    def _register_robot_gripper(self, topic_cfg):
        domain_id = int(topic_cfg.ros_domain_id)
        bridge = self._get_bridge(domain_id)
        for topic, info in (topic_cfg.get('subscribe') or {}).items():
            bridge.add_ros2zmq(topic, info.type)
        for topic, info in (topic_cfg.get('publish') or {}).items():
            bridge.add_ros_publisher(topic, info.type)
            self._topic_domain[topic] = domain_id

    def _register_camera(self, cam_cfg):
        topic_cfg = cam_cfg.topic
        domain_id = int(topic_cfg.ros_domain_id)
        bridge = self._get_bridge(domain_id)
        cam_name = cam_cfg.name
        rotate = int(cam_cfg.get('rotate', 0))
        for topic, info in (topic_cfg.get('subscribe') or {}).items():
            bridge.add_ros2zmq(topic, info.type, cam_name=cam_name, rotate=rotate)

    def pipeline(self):
        env = self.cfg.env

        # CameraJsonPublisher 먼저 생성 (카메라 DomainBridge 생성 전)
        self._cam_pub = CameraJsonPublisher(self._zmq_ctx)

        for device_name in ('robot', 'gripper'):
            device = env.get(device_name)
            if device and device.get('comm') == 'ros2' and device.get('topic'):
                self._register_robot_gripper(device.topic)

        cameras = env.get('camera') or {}
        for _, cam_cfg in cameras.items():
            if cam_cfg.get('comm') == 'ros2' and cam_cfg.get('topic'):
                self._register_camera(cam_cfg)

        for bridge in self._bridges.values():
            bridge.start()

        self._arbiter = Arbiter(self._zmq_ctx, self._bridges, self._topic_domain)
        self._arbiter.start()

        print(f"[CommServer] {len(self._bridges)} domain bridge(s) started: "
              f"domain_ids={list(self._bridges.keys())}")
        print(f"[CommServer] robot/gripper PUB: :{ZMQ_BASE_PUB_PORT} (domain 0)")
        print(f"[CommServer] camera PUB: :{ZMQ_CAM_PUB_PORT} (lerobot JSON)")
        print(f"[CommServer] Arbiter PULL:{ZMQ_BASE_PULL_PORT}, CTRL PUB:{ZMQ_CTRL_PORT}")
        print(f"[CommServer] publish topics: {list(self._topic_domain.keys())}")

    @property
    def arbiter(self) -> Arbiter | None:
        return self._arbiter

    def shutdown(self):
        if self._arbiter:
            self._arbiter.shutdown()
        for bridge in self._bridges.values():
            bridge.shutdown()
        if self._cam_pub:
            self._cam_pub.close()
        self._zmq_ctx.term()
        print("[CommServer] shutdown complete.")
