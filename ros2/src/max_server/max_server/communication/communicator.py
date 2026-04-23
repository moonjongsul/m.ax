"""Communicator: per-domain ROS Contexts wrapping subs/pubs, caches latest data.

Robot, gripper, and camera topics may live on different ROS_DOMAIN_IDs (set in
YAML via `<group>.ros_domain_id`). Each unique domain gets its own rclpy
Context + Node + Executor + spin thread; received messages all funnel into
in-process Python state, so the inference loop and downstream consumers see a
single unified view regardless of source domain.

Recognized role names:
  subscribe  : joint_state, current_pose, gripper_state
  publish    : joint_state (robot cmd), goal_pose, gripper_command
"""

import threading

import cv2
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from max_server.utils.config_loader import CAMERA_MSG_TYPE, resolve_role_msg_type


class _DomainEndpoint:
    """rclpy Context/Node/Executor bundle pinned to a single ROS_DOMAIN_ID."""

    def __init__(self, domain_id: int, node_name: str):
        from rclpy._rclpy_pybind11 import SignalHandlerOptions
        self.domain_id = domain_id
        self._ctx = Context()
        rclpy.init(
            context=self._ctx,
            domain_id=domain_id,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self.node = Node(node_name, context=self._ctx)
        self._executor: SingleThreadedExecutor | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._executor = SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self._executor.spin, daemon=True,
            name=f"max-server-domain{self.domain_id}",
        )
        self._thread.start()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown(context=self._ctx)
        except Exception:
            pass


class Communicator:

    def __init__(
        self,
        logger,
        robot_subscribe: list[dict],
        robot_publish: list[dict],
        robot_domain_id: int,
        gripper_subscribe: list[dict],
        gripper_publish: list[dict],
        gripper_domain_id: int,
        cameras: list[dict],
        camera_domain_id: int,
        camera_rotate: dict[str, int] | None = None,
    ):
        self._logger = logger
        self._lock = threading.Lock()

        self._latest_joint_state = None
        self._latest_current_pose = None
        self._latest_gripper_state = None
        self._latest_images: dict[str, np.ndarray] = {}

        # Last values published by the server (for telemetry broadcast).
        self._last_joint_command = None
        self._last_goal_pose = None
        self._last_gripper_command = None

        self._camera_names: list[str] = [c["name"] for c in cameras]
        self._camera_rotate = camera_rotate or {}

        self._joint_cmd_pub = None
        self._goal_pose_pub = None
        self._gripper_cmd_pub = None

        # One endpoint per unique domain id; nodes share names per domain so
        # repeated domains (e.g. robot+gripper both on domain 0) reuse one node.
        self._endpoints: dict[int, _DomainEndpoint] = {}

        self._setup_subscribers(
            robot_subscribe, self._make_robot_callback,
            domain_id=robot_domain_id, label="robot",
        )
        self._setup_publishers(
            robot_publish, domain_id=robot_domain_id, label="robot",
        )
        self._setup_subscribers(
            gripper_subscribe, self._make_gripper_callback,
            domain_id=gripper_domain_id, label="gripper",
        )
        self._setup_publishers(
            gripper_publish, domain_id=gripper_domain_id, label="gripper",
        )
        self._setup_cameras(cameras, domain_id=camera_domain_id)

    # ─── Endpoint management ─────────────────────────────────────────────────

    def _get_endpoint(self, domain_id: int) -> _DomainEndpoint:
        ep = self._endpoints.get(domain_id)
        if ep is None:
            ep = _DomainEndpoint(domain_id, f"max_server_d{domain_id}")
            self._endpoints[domain_id] = ep
            self._logger.info(f"[comm] opened domain {domain_id}")
        return ep

    def start(self) -> None:
        for ep in self._endpoints.values():
            ep.start()

    def shutdown(self) -> None:
        for ep in self._endpoints.values():
            ep.shutdown()
        self._endpoints.clear()

    # ─── QoS ─────────────────────────────────────────────────────────────────

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

    # ─── Setup ───────────────────────────────────────────────────────────────

    def _setup_subscribers(self, entries: list[dict], cb_factory,
                           domain_id: int, label: str):
        ep = self._get_endpoint(domain_id)
        for entry in entries:
            role = entry["name"]
            topic = entry["topic"]
            msg_cls = resolve_role_msg_type(role)
            cb = cb_factory(role)
            ep.node.create_subscription(msg_cls, topic, cb, self._default_qos())
            self._logger.info(
                f"[comm] {label} sub: {topic} ({role}) on domain {domain_id}"
            )

    def _setup_publishers(self, entries: list[dict], domain_id: int, label: str):
        ep = self._get_endpoint(domain_id)
        for entry in entries:
            role = entry["name"]
            topic = entry["topic"]
            msg_cls = resolve_role_msg_type(role)
            publisher = ep.node.create_publisher(msg_cls, topic, self._default_qos())
            if label == "robot" and role == "joint_state":
                self._joint_cmd_pub = publisher
            elif label == "robot" and role == "goal_pose":
                self._goal_pose_pub = publisher
            elif label == "gripper" and role == "gripper_command":
                self._gripper_cmd_pub = publisher
            self._logger.info(
                f"[comm] {label} pub: {topic} ({role}) on domain {domain_id}"
            )

    def _setup_cameras(self, cameras: list[dict], domain_id: int):
        ep = self._get_endpoint(domain_id)
        for cam in cameras:
            name = cam["name"]
            topic = cam["topic"]
            rotate = int(self._camera_rotate.get(name, 0))
            cb = self._make_camera_callback(name, rotate)
            ep.node.create_subscription(CAMERA_MSG_TYPE, topic, cb, self._sensor_qos())
            self._logger.info(
                f"[comm] camera sub: {topic} ({name}, rotate={rotate}) on domain {domain_id}"
            )

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def _make_robot_callback(self, role: str):
        def cb(msg):
            with self._lock:
                if role == "joint_state":
                    self._latest_joint_state = msg
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

    def get_latest_observation(self, expression_type: str) -> dict | None:
        """Return dict of observations if all required inputs are present, else None."""
        with self._lock:
            if self._latest_gripper_state is None:
                return None
            if len(self._latest_images) < len(self._camera_names):
                return None
            
            if expression_type == "joint":
                if self._latest_joint_state is None:
                    return None
                robot_data = list(self._latest_joint_state.position)
            elif expression_type in ("quat", "rot6d"):
                if self._latest_current_pose is None:
                    return None
                p = self._latest_current_pose.pose.position
                o = self._latest_current_pose.pose.orientation
                # [x, y, z, qx, qy, qz, qw] — downstream conversion to rot6d
                # (if needed) is done in the inference side; keep quaternion here.
                robot_data = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
                
            gripper = (
                float(self._latest_gripper_state.position[0])
                if self._latest_gripper_state.position else 0.0
            )
            images = dict(self._latest_images)
        return {
            "robot_state": np.array(robot_data, dtype=np.float32),
            "gripper_state": np.float32(gripper),
            "images": images,
        }

    def get_latest_states(self) -> dict:
        """Snapshot of observed states + last commands, for telemetry broadcast."""
        with self._lock:
            return {
                "joint_state": self._latest_joint_state,
                "current_pose": self._latest_current_pose,
                "gripper_state": self._latest_gripper_state,
                "joint_command": self._last_joint_command,
                "goal_pose": self._last_goal_pose,
                "gripper_command": self._last_gripper_command,
            }

    # ─── Publishers ──────────────────────────────────────────────────────────

    def publish_joint_command(self, msg):
        if self._joint_cmd_pub is None:
            self._logger.warn("[comm] joint_state publisher not configured")
            return
        self._joint_cmd_pub.publish(msg)
        with self._lock:
            self._last_joint_command = msg

    def publish_goal_pose(self, msg):
        if self._goal_pose_pub is None:
            self._logger.warn("[comm] goal_pose publisher not configured")
            return
        self._goal_pose_pub.publish(msg)
        with self._lock:
            self._last_goal_pose = msg

    def publish_gripper_command(self, msg):
        if self._gripper_cmd_pub is None:
            self._logger.warn("[comm] gripper_command publisher not configured")
            return
        self._gripper_cmd_pub.publish(msg)
        with self._lock:
            self._last_gripper_command = msg
