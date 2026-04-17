import rclpy
from rclpy.node import Node
from rclpy.context import Context
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Header, Float32

import zmq
import threading
import pickle


# ─── Message type registry ───────────────────────────────────────────────────

MSG_TYPE_MAP = {
    'sensor_msgs/JointState':           JointState,
    'sensor_msgs/msg/JointState':       JointState,
    'std_msgs/Float32':                 Float32,
    'std_msgs/msg/Float32':             Float32,
    'sensor_msgs/CompressedImage':      CompressedImage,
    'sensor_msgs/msg/CompressedImage':  CompressedImage,
    'geometry_msgs/PoseStamped':        PoseStamped,
    'geometry_msgs/msg/PoseStamped':    PoseStamped,
}

def resolve_msg_type(type_str: str):
    if type_str not in MSG_TYPE_MAP:
        raise ValueError(f"Unsupported message type: '{type_str}'. "
                         f"Available: {list(MSG_TYPE_MAP.keys())}")
    return MSG_TYPE_MAP[type_str]


# ─── Low-level ROS helpers ────────────────────────────────────────────────────

def _subscriber(node: Node, topic: str, msg_type, callback):
    node.create_subscription(msg_type, topic, callback, 10)


def _publisher(node: Node, topic: str, msg_type):
    return node.create_publisher(msg_type, topic, 10)


# ─── Message converters ───────────────────────────────────────────────────────

def convert_joint_states_msg_to_array(msg: JointState):
    return msg.position


def convert_array_to_joint_states_msg(array, joint_names):
    msg = JointState()
    msg.header = Header()
    msg.name = joint_names
    msg.position = array
    msg.velocity = [0.0] * len(array)
    msg.effort = [0.0] * len(array)
    return msg


def convert_float_msg_to_float(msg: Float32):
    return msg.data


def convert_float_to_float_msg(value: float):
    msg = Float32()
    msg.data = value
    return msg


# ─── ZMQ serialization ───────────────────────────────────────────────────────

def ros_msg_to_bytes(msg) -> bytes:
    """ROS 메시지를 pickle로 직렬화합니다."""
    return pickle.dumps(msg)


def bytes_to_ros_msg(data: bytes, msg_type):
    """bytes를 역직렬화합니다."""
    return pickle.loads(data)


# ─── ROS ↔ ZMQ bridge builders ───────────────────────────────────────────────

def build_ros2zmq_subscriber(node: Node, topic: str, msg_type, zmq_pub_socket: zmq.Socket):
    """ROS subscriber → ZMQ PUB: ROS 토픽 수신 후 ZMQ로 전달."""
    def callback(msg):
        zmq_pub_socket.send_multipart([
            topic.encode(),
            ros_msg_to_bytes(msg),
        ])

    _subscriber(node, topic, msg_type, callback)


def build_zmq2ros_publisher(node: Node, topic: str, msg_type, zmq_sub_socket: zmq.Socket):
    """ZMQ SUB → ROS publisher: ZMQ 수신 후 ROS 토픽으로 발행.

    Returns a (publisher, listener_thread) tuple.
    The listener thread must be started by the caller.
    """
    pub = _publisher(node, topic, msg_type)

    def _listen():
        while rclpy.ok(context=node.context):
            try:
                parts = zmq_sub_socket.recv_multipart()
                if len(parts) != 2:
                    continue
                recv_topic, data = parts
                if recv_topic.decode() != topic:
                    continue
                msg = bytes_to_ros_msg(data, msg_type)
                pub.publish(msg)
            except zmq.ZMQError:
                break

    thread = threading.Thread(target=_listen, daemon=True)
    return pub, thread


# ─── Legacy stubs (kept for back-compat) ─────────────────────────────────────

def ros2zmq(topic: str):
    pass


def zmq2ros(value):
    pass
