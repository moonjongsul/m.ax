"""Communicator: wraps ROS subscriptions/publishers, caches latest observations."""

import threading

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from max_server.utils.config_loader import resolve_msg_type


class Communicator:

    def __init__(self, node: Node, cfg: dict):
        self._node = node
        self._cfg = cfg
        self._lock = threading.Lock()

        self._latest_joint_states = None
        self._latest_current_pose = None
        self._latest_gripper_state = None
        self._latest_images: dict[str, np.ndarray] = {}
        self._camera_names: list[str] = [cam["name"] for cam in cfg.get("cameras", [])]

        self._joint_cmd_pub = None
        self._gripper_cmd_pub = None

        self._setup_robot()
        self._setup_gripper()
        self._setup_cameras()

    # ─── Setup ───────────────────────────────────────────────────────────────

    def _default_qos(self, depth: int = 10) -> QoSProfile:
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            durability=DurabilityPolicy.VOLATILE,
        )

    def _sensor_qos(self, depth: int = 1) -> QoSProfile:
        return QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            durability=DurabilityPolicy.VOLATILE,
        )

    def _setup_robot(self):
        robot = self._cfg.get("robot") or {}
        for sub in robot.get("subscribe") or []:
            msg_cls = resolve_msg_type(sub["type"])
            role = sub["role"]
            topic = sub["topic"]
            cb = self._make_robot_callback(role)
            self._node.create_subscription(msg_cls, topic, cb, self._default_qos())
            self._node.get_logger().info(f"[comm] robot sub: {topic} ({role})")

        for pub in robot.get("publish") or []:
            msg_cls = resolve_msg_type(pub["type"])
            topic = pub["topic"]
            role = pub["role"]
            publisher = self._node.create_publisher(msg_cls, topic, self._default_qos())
            if role == "joint_command":
                self._joint_cmd_pub = publisher
            self._node.get_logger().info(f"[comm] robot pub: {topic} ({role})")

    def _setup_gripper(self):
        gripper = self._cfg.get("gripper") or {}
        for sub in gripper.get("subscribe") or []:
            msg_cls = resolve_msg_type(sub["type"])
            role = sub["role"]
            topic = sub["topic"]
            cb = self._make_gripper_callback(role)
            self._node.create_subscription(msg_cls, topic, cb, self._default_qos())
            self._node.get_logger().info(f"[comm] gripper sub: {topic} ({role})")

        for pub in gripper.get("publish") or []:
            msg_cls = resolve_msg_type(pub["type"])
            topic = pub["topic"]
            role = pub["role"]
            publisher = self._node.create_publisher(msg_cls, topic, self._default_qos())
            if role == "gripper_command":
                self._gripper_cmd_pub = publisher
            self._node.get_logger().info(f"[comm] gripper pub: {topic} ({role})")

    def _setup_cameras(self):
        cameras = self._cfg.get("cameras") or []
        for cam in cameras:
            msg_cls = resolve_msg_type(cam["type"])
            topic = cam["topic"]
            name = cam["name"]
            rotate = int(cam.get("rotate", 0))
            cb = self._make_camera_callback(name, rotate)
            self._node.create_subscription(msg_cls, topic, cb, self._sensor_qos())
            self._node.get_logger().info(f"[comm] camera sub: {topic} ({name})")

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def _make_robot_callback(self, role: str):
        def cb(msg):
            with self._lock:
                if role == "joint_states":
                    self._latest_joint_states = msg
                elif role == "current_pose":
                    self._latest_current_pose = msg
        return cb

    def _make_gripper_callback(self, role: str):
        def cb(msg):
            with self._lock:
                if role == "gripper_state":
                    self._latest_gripper_state = msg
        return cb

    def _make_camera_callback(self, name: str, rotate: int):
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        def cb(msg):
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            rot = rotate_map.get(rotate)
            if rot is not None:
                img = cv2.rotate(img, rot)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest_images[name] = rgb
        return cb

    # ─── Accessors ───────────────────────────────────────────────────────────

    def camera_names(self) -> list[str]:
        return list(self._camera_names)

    def get_latest_observation(self) -> dict | None:
        """Return dict of observations if all required inputs are present, else None."""
        with self._lock:
            if self._latest_joint_states is None:
                return None
            if self._latest_gripper_state is None:
                return None
            if len(self._latest_images) < len(self._camera_names):
                return None
            joints = list(self._latest_joint_states.position)
            gripper = (
                float(self._latest_gripper_state.position[0])
                if self._latest_gripper_state.position else 0.0
            )
            images = dict(self._latest_images)
        return {
            "joint_states": np.array(joints, dtype=np.float32),
            "gripper_state": np.float32(gripper),
            "images": images,
        }

    # ─── Publishers ──────────────────────────────────────────────────────────

    def publish_joint_command(self, msg):
        if self._joint_cmd_pub is None:
            self._node.get_logger().warn("[comm] joint_command publisher not configured")
            return
        self._joint_cmd_pub.publish(msg)

    def publish_gripper_command(self, msg):
        if self._gripper_cmd_pub is None:
            self._node.get_logger().warn("[comm] gripper_command publisher not configured")
            return
        self._gripper_cmd_pub.publish(msg)
