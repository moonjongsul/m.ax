"""노드 네 개가 공유하는 ROS 잡일. 제어 로직은 없다.

**시계.** 노드 간에는 ROS 시계(`node.get_clock()`)를 쓴다. 쪼개기 전에는
`time.monotonic()`이었지만 그건 프로세스마다 원점이 달라서 노드 경계를 넘는
순간 쓸 수 없다. 청크 실행이 관측 시각 기준이고 인식기 캡처가 경계 기준이라
노드들이 같은 시계를 봐야 한다.

기본 설정에서 ROS 시계는 시스템 시계이므로 NTP 보정에 점프할 수 있다. 큰
점프는 데드맨과 에피소드 타임아웃을 오발동시킨다 -- 셀 PC의 시계 동기를
rollout 중에 건드리지 말 것.
"""

from __future__ import annotations

from pathlib import Path

from builtin_interfaces.msg import Time as TimeMsg
from max_interfaces.msg import RecognitionVerdict
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from ..rollout.config import RolloutConfig, load_config

# purpose 상수는 RecognitionVerdict에만 정의하고, 요청(ScheduleRecognition)과
# 판정이 같은 번호를 쓴다. 두 곳에 두면 한쪽만 고쳐질 수 있다.
PURPOSE_NAME = {
    RecognitionVerdict.PURPOSE_START: "start",
    RecognitionVerdict.PURPOSE_FLIP: "flip",
    RecognitionVerdict.PURPOSE_PLACE: "place",
}
PURPOSE_CODE = {v: k for k, v in PURPOSE_NAME.items()}


def declare_config(node: Node) -> RolloutConfig:
    """`config_file` 파라미터로 받은 YAML 하나를 읽는다.

    노드마다 파라미터를 따로 두지 않는 이유는 `rollout/config.py`에 적힌 것과
    같다 -- fps나 카메라 이름이 노드마다 다른 상태로 뜨면 청크 타이밍과 인식기
    캡처 시각이 조용히 틀어진다. 네 노드가 같은 파일을 읽는다.
    """
    path = node.declare_parameter("config_file", "").get_parameter_value().string_value
    if not path:
        raise RuntimeError("config_file parameter is required (the shared rollout YAML)")
    cfg = load_config(Path(path).expanduser())
    node.get_logger().info(f"config: {path}")
    return cfg


def sensor_qos(depth: int = 10) -> QoSProfile:
    """BEST_EFFORT. max_data_collect와 같은 이유로 고른다 -- 액션·EEF 토픽이
    BEST_EFFORT로 퍼블리시되고, RELIABLE 구독자는 거기에 조용히 연결되지 않는다.
    """
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST,
                      durability=DurabilityPolicy.VOLATILE, depth=depth)


def latched_qos(depth: int = 1) -> QoSProfile:
    """TRANSIENT_LOCAL. 늦게 뜬 노드가 현재 상태를 바로 알아야 하는 토픽."""
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=depth)


def now_s(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def to_sec(stamp: TimeMsg) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def to_stamp(seconds: float) -> TimeMsg:
    sec = int(seconds)
    return TimeMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


class WarnOnce:
    """같은 경고를 반복해서 찍지 않는다. 30Hz 루프에서 한 번 틀리면 계속 틀린다."""

    def __init__(self, node: Node):
        self.node = node
        self._seen: set[str] = set()

    def __call__(self, message: str) -> None:
        if message not in self._seen:
            self._seen.add(message)
            self.node.get_logger().warn(message)
