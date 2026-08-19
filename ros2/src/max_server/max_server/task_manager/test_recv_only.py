import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage

IMG_TOPIC_NAME = "/observation/env/color/image_raw/compressed"


class RecvOnlyNode(Node):
    def __init__(self):
        super().__init__("recv_only")
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(
            CompressedImage, IMG_TOPIC_NAME, self.cb, qos
        )
        self.last_recv = None
        self.last_pub_stamp = None
        self.count = 0
        self.stat_timer = self.create_timer(2.0, self.stat)
        self.gaps_recv: list[float] = []
        self.gaps_pub: list[float] = []
        self.lags: list[float] = []

    def cb(self, msg: CompressedImage):
        now = time.time()
        pub_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_recv is not None:
            self.gaps_recv.append(now - self.last_recv)
        if self.last_pub_stamp is not None:
            self.gaps_pub.append(pub_t - self.last_pub_stamp)
        self.lags.append(now - pub_t)
        self.last_recv = now
        self.last_pub_stamp = pub_t
        self.count += 1

    def stat(self):
        if not self.gaps_recv:
            return

        def s(name, arr):
            arr_s = sorted(arr)
            n = len(arr_s)
            return (
                f"{name}: n={n} "
                f"min={arr_s[0] * 1000:.1f}ms "
                f"p50={arr_s[n // 2] * 1000:.1f}ms "
                f"p95={arr_s[int(n * 0.95)] * 1000:.1f}ms "
                f"max={arr_s[-1] * 1000:.1f}ms"
            )

        self.get_logger().info(s("recv_gap", self.gaps_recv))
        self.get_logger().info(s("pub_gap ", self.gaps_pub))
        self.get_logger().info(s("lag     ", self.lags))
        self.gaps_recv.clear()
        self.gaps_pub.clear()
        self.lags.clear()


def main():
    rclpy.init()
    node = RecvOnlyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
