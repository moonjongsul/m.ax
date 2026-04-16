"""CommServer — ROS ↔ ZMQ 브릿지 서버

cfg 구조 (proj_gt_kitting.yaml 기반):
  env.robot.topic.ros_domain_id   = 0
  env.gripper.topic.ros_domain_id = 0
  env.camera.*.topic.ros_domain_id = 1

동작:
  - ROS subscribe  → ZMQ PUB  (ros2zmq)
  - ZMQ PUSH(client) → Arbiter → ZMQ SUB → ROS publish

ZMQ 포트 규칙:
  ROS→ZMQ  (PUB)  : ZMQ_BASE_PUB_PORT + domain_id  (5570, 5571, …)
  ZMQ→ROS  (PULL) : ZMQ_BASE_PULL_PORT              (5590)  ← 제어 전용, domain 무관
  제어권 알림 (PUB): ZMQ_BASE_CTRL_PORT              (5591)

제어 메시지 포맷 (client → CommServer PULL):
  [source, topic, pickled_msg]
  예: [b"web", b"/gello/joint_states", pickled_JointState]

제어권 알림 포맷 (CommServer PUB → clients):
  [b"control", b"granted:<source>"]
  [b"control", b"revoked:<source>"]
"""

import threading
import time
import pickle
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

ZMQ_BASE_PUB_PORT  = 5570   # ROS → ZMQ PUB  (per domain_id)
ZMQ_BASE_PULL_PORT = 5590   # client → CommServer PULL (제어 전용)
ZMQ_CTRL_PORT      = 5591   # CommServer → client PUB  (제어권 알림)

LOCK_TIMEOUT = 5.0           # 초: 마지막 메시지 이후 제어권 자동 해제


# ─── DomainBridge ─────────────────────────────────────────────────────────────

class DomainBridge:
    """단일 ROS domain_id에 대한 ROS→ZMQ 브릿지 (subscribe only)."""

    def __init__(self, domain_id: int, zmq_context: zmq.Context):
        self.domain_id = domain_id
        self._zmq_ctx = zmq_context

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

        pub_port = ZMQ_BASE_PUB_PORT + domain_id
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.bind(f'tcp://*:{pub_port}')

        # topic → ROS publisher (Arbiter가 직접 호출)
        self._ros_publishers: dict[str, object] = {}

    # ── 토픽 등록 ──────────────────────────────────────────────────────────

    def add_ros2zmq(self, topic: str, msg_type_str: str):
        msg_type = resolve_msg_type(msg_type_str)
        build_ros2zmq_subscriber(self._node, topic, msg_type, self._zmq_pub)

    def add_ros_publisher(self, topic: str, msg_type_str: str):
        """Arbiter에서 ROS publish 할 publisher 등록."""
        msg_type = resolve_msg_type(msg_type_str)
        self._ros_publishers[topic] = _publisher(self._node, topic, msg_type)

    def publish(self, topic: str, raw: bytes):
        """pickle bytes → ROS publish."""
        pub = self._ros_publishers.get(topic)
        if pub is None:
            return
        msg = pickle.loads(raw)
        pub.publish(msg)

    # ── 실행 / 종료 ────────────────────────────────────────────────────────

    def start(self):
        self._executor = SingleThreadedExecutor(context=self._ros_context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def shutdown(self):
        self._executor.shutdown()
        self._node.destroy_node()
        rclpy.shutdown(context=self._ros_context)
        self._zmq_pub.close()


# ─── Arbiter ──────────────────────────────────────────────────────────────────

class Arbiter:
    """제어 메시지를 수신하고 제어권을 중재해 ROS publish를 대행합니다.

    - PULL socket으로 [source, topic, pickled_msg] 수신
    - 현재 제어권 보유자와 다른 source가 오면 제어권 교체
    - PUB socket으로 granted/revoked 알림 발행
    - LOCK_TIMEOUT 초 동안 메시지가 없으면 제어권 자동 해제
    """

    def __init__(self, zmq_context: zmq.Context,
                 bridges: dict[int, "DomainBridge"],
                 topic_domain_map: dict[str, int]):
        self._ctx = zmq_context
        self._bridges = bridges
        self._topic_domain = topic_domain_map   # topic → domain_id

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
        """b"control", b"<event>:<source>" 발행."""
        self._ctrl_pub.send_multipart([
            b"control",
            f"{event}:{source}".encode(),
        ])

    def _acquire(self, source: str) -> bool:
        """제어권 획득. 다른 source가 갖고 있으면 revoke 후 교체. True 반환."""
        with self._lock:
            now = time.monotonic()
            # 타임아웃으로 자동 해제
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
                self._lock_time = now   # heartbeat
                return True

            # 다른 source가 제어권 요청 → 강제 교체
            prev = self._lock_source
            self._notify("revoked", prev)
            print(f"[Arbiter] '{prev}' revoked, '{source}' granted control")
            self._lock_source = source
            self._lock_time = now
            self._notify("granted", source)
            return True

    def _run(self):
        while self._running:
            try:
                parts = self._pull.recv_multipart()
            except zmq.Again:
                # 타임아웃 — lock 만료 체크만 하고 계속
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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._running = False
        self._pull.close()
        self._ctrl_pub.close()


# ─── CommServer ───────────────────────────────────────────────────────────────

class CommServer:
    """cfg를 읽어 domain_id별 DomainBridge + Arbiter를 구성하고 기동합니다."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._zmq_ctx = zmq.Context()
        self._bridges: dict[int, DomainBridge] = {}
        self._arbiter: Arbiter | None = None
        self._topic_domain: dict[str, int] = {}   # publish 토픽 → domain_id

    def _get_bridge(self, domain_id: int) -> DomainBridge:
        if domain_id not in self._bridges:
            self._bridges[domain_id] = DomainBridge(domain_id, self._zmq_ctx)
        return self._bridges[domain_id]

    def _register_device(self, topic_cfg):
        domain_id = int(topic_cfg.ros_domain_id)
        bridge = self._get_bridge(domain_id)

        for topic, info in (topic_cfg.get('subscribe') or {}).items():
            bridge.add_ros2zmq(topic, info.type)

        for topic, info in (topic_cfg.get('publish') or {}).items():
            bridge.add_ros_publisher(topic, info.type)
            self._topic_domain[topic] = domain_id

    def ros_comm_pipeline(self):
        env = self.cfg.env

        for device_name in ('robot', 'gripper'):
            device = env.get(device_name)
            if device and device.get('comm') == 'ros2' and device.get('topic'):
                self._register_device(device.topic)

        cameras = env.get('camera') or {}
        for _, cam_cfg in cameras.items():
            if cam_cfg.get('comm') == 'ros2' and cam_cfg.get('topic'):
                self._register_device(cam_cfg.topic)

        for bridge in self._bridges.values():
            bridge.start()

        self._arbiter = Arbiter(self._zmq_ctx, self._bridges, self._topic_domain)
        self._arbiter.start()

        print(f"[CommServer] {len(self._bridges)} domain bridge(s) started: "
              f"domain_ids={list(self._bridges.keys())}")
        print(f"[CommServer] Arbiter listening on PULL:{ZMQ_BASE_PULL_PORT}, "
              f"CTRL PUB:{ZMQ_CTRL_PORT}")
        print(f"[CommServer] publish topics: {list(self._topic_domain.keys())}")

    def shutdown(self):
        if self._arbiter:
            self._arbiter.shutdown()
        for bridge in self._bridges.values():
            bridge.shutdown()
        self._zmq_ctx.term()
        print("[CommServer] shutdown complete.")
