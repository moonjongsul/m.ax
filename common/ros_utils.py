import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float32


def _subscriber(node: Node, topic: str, msg_type, callback):
    node.create_subscription(
        msg_type,
        topic,
        callback,
        10
    )

def _publisher(node: Node, topic: str, msg_type):
    return node.create_publisher(
        msg_type,
        topic,
        10
    )
    
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