"""Config helpers: parse 'name:topic' entries and resolve msg types by role."""

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped


# Fixed mapping from role name -> ROS message class.
# Entry names in the YAML config are matched against these keys.
ROLE_MSG_TYPES = {
    # robot
    "joint_state": JointState,
    "current_pose": PoseStamped,
    "goal_pose": PoseStamped,
    # gripper
    "gripper_state": JointState,
    "gripper_command": Float32,
}


CAMERA_MSG_TYPE = CompressedImage


def resolve_role_msg_type(role: str):
    """Return the ROS message class for a role name (robot/gripper entries)."""
    if role not in ROLE_MSG_TYPES:
        raise ValueError(
            f"Unknown role '{role}'. Known roles: {list(ROLE_MSG_TYPES)}"
        )
    return ROLE_MSG_TYPES[role]


def parse_entry(entry: str) -> dict:
    """Parse a 'name:topic' string into a dict.

    Note: topic paths may contain slashes but not colons, so a single-colon
    split is unambiguous.
    """
    if ":" not in entry:
        raise ValueError(
            f"Invalid entry format: '{entry}'. Expected 'name:topic'"
        )
    name, topic = entry.split(":", 1)
    return {"name": name, "topic": topic}


def parse_entry_list(entries: list[str]) -> list[dict]:
    return [parse_entry(e) for e in entries]
