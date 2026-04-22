"""Config loader: YAML -> plain dict."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


MSG_TYPE_MAP = {
    "sensor_msgs/msg/JointState": ("sensor_msgs.msg", "JointState"),
    "sensor_msgs/msg/CompressedImage": ("sensor_msgs.msg", "CompressedImage"),
    "geometry_msgs/msg/PoseStamped": ("geometry_msgs.msg", "PoseStamped"),
    "std_msgs/msg/Float32": ("std_msgs.msg", "Float32"),
}


def resolve_msg_type(type_str: str):
    """Resolve ROS 2 message type string like 'sensor_msgs/msg/JointState' to class."""
    if type_str not in MSG_TYPE_MAP:
        raise ValueError(
            f"Unsupported message type: '{type_str}'. "
            f"Available: {list(MSG_TYPE_MAP.keys())}"
        )
    module_name, class_name = MSG_TYPE_MAP[type_str]
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
