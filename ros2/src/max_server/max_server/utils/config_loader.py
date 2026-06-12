"""Config helpers: parse 'name:topic' entries and resolve msg types by role."""

import importlib

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


def resolve_srv_type(srv_type: str):
    """Import a service class from its 'pkg/srv/Type' string.

    e.g. 'std_srvs/srv/Trigger' -> std_srvs.srv.Trigger
    """
    parts = srv_type.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid srv_type '{srv_type}'. Expected 'pkg/srv/Type'"
        )
    pkg, sub, name = parts
    module = importlib.import_module(f"{pkg}.{sub}")
    return getattr(module, name)


def parse_service_client_entry(entry: str) -> dict:
    """Parse a 'name:service:srv_type' string into a dict.

    e.g. 'picking_pick:/picking_cell/pick:std_srvs/srv/Trigger' ->
         {'name': 'picking_pick',
          'service': '/picking_cell/pick',
          'srv_type': 'std_srvs/srv/Trigger'}
    """
    parts = entry.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid service client entry: '{entry}'. "
            "Expected 'name:service:srv_type'"
        )
    name, service, srv_type = parts
    return {"name": name, "service": service, "srv_type": srv_type}


def parse_service_client_list(entries: list[str]) -> list[dict]:
    return [parse_service_client_entry(e) for e in entries]
