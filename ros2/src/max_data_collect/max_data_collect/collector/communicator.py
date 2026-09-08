"""Subscribe every source and cache the latest message per stream.

Everything is subscribed BEST_EFFORT. That is a compatibility
requirement, not a preference: the action and EEF topics are published
BEST_EFFORT, and a RELIABLE subscriber does not connect to a BEST_EFFORT
publisher -- it receives nothing, with no error. The reverse direction is
fine, so one BEST_EFFORT subscription works for every source here
(verified against the RELIABLE joint/gripper/camera publishers).

Every source is stored with the message's own `header.stamp`, which is
when the state was measured. That is the axis the frame builder aligns
on; arrival time at the recorder is not used for anything but staleness.
"""

import threading
import time

from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, JointState

from max_interfaces.msg import TeleopAction


# Stream keys. The frame builder and the writer both index by these, so
# they are defined once here rather than spelled out as strings.
CAM_PREFIX = "camera/"
JOINT = "joint"
EEF_POSE = "eef_pose"
EEF_TWIST = "eef_twist"
GRIPPER = "gripper"
ACTION = "action"


def stamp_to_ns(header) -> int:
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class _Cached:
    """Newest message on one stream."""

    __slots__ = ("msg", "stamp_ns", "recv", "count")

    def __init__(self):
        self.msg = None
        self.stamp_ns = 0       # message header stamp
        self.recv = 0.0         # monotonic arrival, for staleness only
        self.count = 0


class Communicator:
    """Owns every subscription and the latest-message cache."""

    def __init__(self, node, logger, cfg: dict):
        self._log = logger
        self._lock = threading.Lock()
        self._cache: dict[str, _Cached] = {}
        self._subs: list = []

        qos_cfg = cfg["qos"]
        reliability = (
            ReliabilityPolicy.RELIABLE
            if str(qos_cfg.get("reliability", "best_effort")).lower() == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        self._qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=int(qos_cfg.get("depth", 10)),
        )

        self.camera_names = list(cfg["observation_cameras"])
        for name, cam in cfg["observation_cameras"].items():
            self._subscribe(node, CAM_PREFIX + name, cam["topic"], CompressedImage)

        manip = cfg["observation_manipulator"]
        self._subscribe(node, JOINT, manip["joint"]["topic"], JointState)
        self._subscribe(node, EEF_POSE, manip["eef_position"]["topic"], PoseStamped)
        self._subscribe(node, EEF_TWIST, manip["eef_velocity"]["topic"], TwistStamped)
        self._subscribe(node, GRIPPER, cfg["observation_gripper"]["topic"], JointState)
        self._subscribe(node, ACTION, cfg["action"]["topic"], TeleopAction)

    def _subscribe(self, node, key: str, topic: str, msg_type) -> None:
        self._cache[key] = _Cached()

        def _cb(msg, _key=key):
            with self._lock:
                slot = self._cache[_key]
                slot.msg = msg
                slot.stamp_ns = stamp_to_ns(msg.header)
                slot.recv = time.monotonic()
                slot.count += 1

        self._subs.append(node.create_subscription(msg_type, topic, _cb, self._qos))
        self._log.info(f"[collect] subscribe {key:<20} {topic}")

    # ── reading ────────────────────────────────────────────────────────

    def keys(self) -> list[str]:
        return list(self._cache)

    def snapshot(self) -> dict:
        """Read every stream at once: {key: (msg, stamp_ns, recv)}.

        Taking all sources under one lock is what makes a recorded row
        internally consistent -- no stream can update midway through.
        """
        with self._lock:
            return {
                k: (c.msg, c.stamp_ns, c.recv) for k, c in self._cache.items()
            }

    def never_received(self) -> list[str]:
        with self._lock:
            return [k for k, c in self._cache.items() if c.count == 0]

    def stale(self, timeout: float) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return [
                k for k, c in self._cache.items()
                if c.count == 0 or (now - c.recv) > timeout
            ]
