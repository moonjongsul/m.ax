"""MaxServerNode: single ROS 2 Node orchestrating communicator + inference manager.

- Action   /inference/load_policy : load a policy (framework, policy, checkpoint, device, dtype)
- Service  /inference/unload_policy : unload the current policy
- Service  /control/move_to_pose : publish a preset pose
- Action   /inference/run          : start/stop inference loop (task_instruction, expression_type)
- Topic    /inference/status       : periodic status broadcast (1Hz)
- Topic    /control/robot_state_package : periodic robot/gripper telemetry (30Hz)
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from max_interfaces.action import LoadPolicy, RunInference
from max_interfaces.msg import InferenceStatus, RobotStatesPackage
from max_interfaces.srv import MoveToPose, UnloadPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header

from max_server.communication.communicator import Communicator
from max_server.inference.inference_manager import InferenceManager
from max_server.task_manager.task_manager import TaskManager
from max_server.task_planner.task_planner import TaskPlanner
from max_server.utils.config_loader import parse_entry_list


STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_RUNNING = "running"
STATE_ERROR = "error"


class MaxServerNode(Node):

    def __init__(self):
        super().__init__(
            "max_server",
            automatically_declare_parameters_from_overrides=True,
        )

        # ── Parameters (from YAML injected by launch) ─────────────────────
        self._fps = float(self._get_param("inference.fps", 30.0))
        self._robot_joint_list: list[str] = list(self._get_param("robot.joint_list", []))

        robot_sub = parse_entry_list(self._get_param("robot.subscribe_list", []))
        robot_pub = parse_entry_list(self._get_param("robot.publish_list", []))
        gripper_sub = parse_entry_list(self._get_param("gripper.subscribe_list", []))
        gripper_pub = parse_entry_list(self._get_param("gripper.publish_list", []))
        cameras = parse_entry_list(self._get_param("camera.subscribe_list", []))
        camera_rotate = self._get_rotate_map(cameras)

        # Per-group ROS_DOMAIN_IDs. Defaults match the inference domain so that
        # missing fields collapse to single-domain behavior.
        inference_domain = int(self._get_param("inference.ros_domain_id", 0))
        robot_domain = int(self._get_param("robot.ros_domain_id", inference_domain))
        gripper_domain = int(self._get_param("gripper.ros_domain_id", inference_domain))
        camera_domain = int(self._get_param("camera.ros_domain_id", inference_domain))

        # Default values surfaced to the web UI via /inference/status.
        self._defaults = {
            "framework": str(self._get_param("inference.default_framework", "")),
            "policy": str(self._get_param("inference.default_policy", "")),
            "checkpoint": str(self._get_param("inference.default_checkpoint", "")),
            "device": str(self._get_param("inference.default_device", "")),
            "dtype": str(self._get_param("inference.default_dtype", "")),
            "task_instruction": str(
                self._get_param("inference.default_task_instruction", "")
            ),
            "expression_type": str(
                self._get_param("inference.default_expression_type", "joint")
            ),
        }
        self._default_expression_type = self._defaults["expression_type"]

        # Preset poses: {name: [float, ...]} keyed by pose name.
        self._robot_poses: dict[str, list[float]] = self._load_poses("robot.pose")
        self._gripper_poses: dict[str, list[float]] = self._load_poses("gripper.pose")
        # Frame id for PoseStamped publishes.
        self._pose_frame_id = str(
            self._get_param("robot.pose_frame_id", "fr3_link0")
        )

        # ── State ─────────────────────────────────────────────────────────
        self._state_lock = threading.Lock()
        self._server_state = STATE_IDLE
        self._state_detail = ""

        self._task_instruction = ""
        self._expression_type = self._default_expression_type
        self._step = 0
        self._last_action: np.ndarray | None = None

        # ── Callback groups ───────────────────────────────────────────────
        self._sub_group = ReentrantCallbackGroup()
        self._svc_group = MutuallyExclusiveCallbackGroup()
        self._run_action_group = MutuallyExclusiveCallbackGroup()
        self._load_action_group = MutuallyExclusiveCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()

        # ── Components ────────────────────────────────────────────────────
        # Communicator owns its own per-domain rclpy Contexts/Nodes; this main
        # Node only carries action/service/timer interfaces on the inference
        # domain (set by main() via rclpy.init(domain_id=...)).
        self.communicator = Communicator(
            self.get_logger(),
            robot_subscribe=robot_sub,
            robot_publish=robot_pub,
            robot_domain_id=robot_domain,
            gripper_subscribe=gripper_sub,
            gripper_publish=gripper_pub,
            gripper_domain_id=gripper_domain,
            cameras=cameras,
            camera_domain_id=camera_domain,
            camera_rotate=camera_rotate,
        )
        self.communicator.start()
        self.inference_manager = InferenceManager()
        self.task_manager = TaskManager(self)
        self.task_planner = TaskPlanner(self)

        # ── ROS interfaces ────────────────────────────────────────────────
        self.create_service(
            UnloadPolicy, "/inference/unload_policy",
            self._on_unload_policy, callback_group=self._svc_group,
        )
        self.create_service(
            MoveToPose, "/control/move_to_pose",
            self._on_move_to_pose, callback_group=self._svc_group,
        )
        self._load_action_server = ActionServer(
            self,
            LoadPolicy,
            "/inference/load_policy",
            execute_callback=self._execute_load_policy,
            goal_callback=self._accept_load_goal,
            cancel_callback=lambda _goal_handle: CancelResponse.REJECT,
            callback_group=self._load_action_group,
        )
        self._run_action_server = ActionServer(
            self,
            RunInference,
            "/inference/run",
            execute_callback=self._execute_inference,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._run_action_group,
        )
        self._status_pub = self.create_publisher(InferenceStatus, "/inference/status", 10)
        self.create_timer(
            1.0, self._publish_status, callback_group=self._timer_group,
        )

        self._robot_states_pub = self.create_publisher(
            RobotStatesPackage, "/control/robot_state_package", 10,
        )
        self._robot_states_rate_hz = float(
            self._get_param("telemetry.robot_states_rate_hz", 30.0)
        )
        self.create_timer(
            1.0 / self._robot_states_rate_hz,
            self._publish_robot_states,
            callback_group=self._timer_group,
        )

        self.get_logger().info(
            f"[max_server] ready. fps={self._fps}, cameras={self.communicator.camera_names()}"
        )

    # ─── Parameter helpers ────────────────────────────────────────────────

    def _get_param(self, name: str, default):
        """Get parameter value, falling back to `default` if unset or NOT_SET."""
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        p = self.get_parameter(name)
        from rclpy.parameter import Parameter
        if p.type_ == Parameter.Type.NOT_SET:
            return default
        return p.value

    def _get_rotate_map(self, cameras: list[dict]) -> dict[str, int]:
        """Read 'camera.rotate.<name>' parameters for each camera."""
        result = {}
        for cam in cameras:
            name = cam["name"]
            result[name] = int(self._get_param(f"camera.rotate.{name}", 0))
        return result

    def _load_poses(self, prefix: str) -> dict[str, list[float]]:
        """Collect preset poses from parameters under the given prefix.

        e.g. prefix='robot.pose' returns {'home': [...], 'kit': [...]} by
        scanning all auto-declared parameters that start with 'robot.pose.'.
        Insertion order follows parameter name sort order (stable across runs).
        """
        poses: dict[str, list[float]] = {}
        dot_prefix = prefix + "."
        # _parameters is a private-ish dict managed by rclpy; used here because
        # there is no public API to enumerate parameter names with a prefix.
        for name in sorted(self._parameters.keys()):
            if not name.startswith(dot_prefix):
                continue
            leaf = name[len(dot_prefix):]
            # Skip nested keys (e.g. 'pose.home.extra'); we only want direct children.
            if "." in leaf:
                continue
            value = self.get_parameter(name).value
            if isinstance(value, (list, tuple)):
                poses[leaf] = [float(x) for x in value]
        return poses

    # ─── State helpers ────────────────────────────────────────────────────

    def _set_state(self, state: str, detail: str = ""):
        with self._state_lock:
            self._server_state = state
            self._state_detail = detail
        self.get_logger().info(f"[max_server] state -> {state}" + (f" ({detail})" if detail else ""))

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._server_state

    # ─── Status publisher ─────────────────────────────────────────────────

    def _publish_status(self):
        msg = InferenceStatus()
        with self._state_lock:
            msg.server_state = self._server_state
            msg.detail = self._state_detail
        msg.policy_framework = self.inference_manager.framework or ""
        msg.policy_name = self.inference_manager.policy_name or ""
        msg.checkpoint = self.inference_manager.checkpoint or ""
        msg.device = self.inference_manager.device or ""
        msg.dtype = self.inference_manager.dtype or ""
        msg.task_instruction = self._task_instruction
        msg.expression_type = self._expression_type
        msg.step = int(self._step)

        # Defaults from YAML config (for web UI form initialization)
        msg.default_framework = self._defaults["framework"]
        msg.default_policy = self._defaults["policy"]
        msg.default_checkpoint = self._defaults["checkpoint"]
        msg.default_device = self._defaults["device"]
        msg.default_dtype = self._defaults["dtype"]
        msg.default_task_instruction = self._defaults["task_instruction"]
        msg.default_expression_type = self._defaults["expression_type"]

        # Preset pose names (order preserved from parameter name sort)
        msg.robot_pose_names = list(self._robot_poses.keys())
        msg.gripper_pose_names = list(self._gripper_poses.keys())

        self._status_pub.publish(msg)

    # ─── Robot states telemetry publisher ─────────────────────────────────

    def _publish_robot_states(self):
        states = self.communicator.get_latest_states()
        msg = RobotStatesPackage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._pose_frame_id

        js = states["joint_state"]
        if js is not None:
            msg.joint_names = list(js.name) if js.name else list(self._robot_joint_list)
            msg.joint_positions = [float(x) for x in js.position]
        else:
            msg.joint_names = list(self._robot_joint_list)
            msg.joint_positions = []

        cp = states["current_pose"]
        if cp is not None:
            p, o = cp.pose.position, cp.pose.orientation
            msg.current_pose_valid = True
            msg.current_pose_x = float(p.x)
            msg.current_pose_y = float(p.y)
            msg.current_pose_z = float(p.z)
            msg.current_pose_qx = float(o.x)
            msg.current_pose_qy = float(o.y)
            msg.current_pose_qz = float(o.z)
            msg.current_pose_qw = float(o.w)
        else:
            msg.current_pose_valid = False

        gs = states["gripper_state"]
        if gs is not None and gs.position:
            msg.gripper_state_valid = True
            msg.gripper_state = float(gs.position[0])
        else:
            msg.gripper_state_valid = False

        jc = states["joint_command"]
        if jc is not None:
            msg.joint_command_valid = True
            msg.joint_command = [float(x) for x in jc.position]
        else:
            msg.joint_command_valid = False

        gp = states["goal_pose"]
        if gp is not None:
            p, o = gp.pose.position, gp.pose.orientation
            msg.goal_pose_valid = True
            msg.goal_pose_x = float(p.x)
            msg.goal_pose_y = float(p.y)
            msg.goal_pose_z = float(p.z)
            msg.goal_pose_qx = float(o.x)
            msg.goal_pose_qy = float(o.y)
            msg.goal_pose_qz = float(o.z)
            msg.goal_pose_qw = float(o.w)
        else:
            msg.goal_pose_valid = False

        gc = states["gripper_command"]
        if gc is not None:
            msg.gripper_command_valid = True
            msg.gripper_command = float(gc.data)
        else:
            msg.gripper_command_valid = False

        self._robot_states_pub.publish(msg)

    # ─── Action: load policy ──────────────────────────────────────────────

    def _accept_load_goal(self, goal_request: LoadPolicy.Goal):
        if self.state == STATE_RUNNING:
            self.get_logger().warn("[max_server] reject load: inference is running")
            return GoalResponse.REJECT
        if self.state == STATE_LOADING:
            self.get_logger().warn("[max_server] reject load: already loading")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_load_policy(self, goal_handle):
        goal = goal_handle.request
        result = LoadPolicy.Result()

        self._set_state(STATE_LOADING, f"{goal.framework}/{goal.policy}")
        self._publish_load_feedback(goal_handle, "starting",
                                    f"{goal.framework}/{goal.policy}")

        if self.inference_manager.is_loaded():
            self._publish_load_feedback(goal_handle, "unloading_previous", "")

        self._publish_load_feedback(goal_handle, "loading", goal.checkpoint)

        ok, msg = self.inference_manager.load(
            framework=goal.framework,
            policy=goal.policy,
            checkpoint=goal.checkpoint,
            device=goal.device,
            dtype=goal.dtype,
        )

        if ok:
            self._set_state(STATE_READY)
            self._publish_load_feedback(goal_handle, "done", msg)
            goal_handle.succeed()
        else:
            self._set_state(STATE_ERROR, msg)
            self._publish_load_feedback(goal_handle, "done", msg)
            goal_handle.abort()

        result.success = ok
        result.message = msg
        return result

    def _publish_load_feedback(self, goal_handle, stage: str, detail: str):
        fb = LoadPolicy.Feedback()
        fb.stage = stage
        fb.detail = detail
        try:
            goal_handle.publish_feedback(fb)
        except Exception:
            pass

    # ─── Service: unload policy ───────────────────────────────────────────

    def _on_unload_policy(
        self,
        _request: UnloadPolicy.Request,
        response: UnloadPolicy.Response,
    ):
        if self.state == STATE_RUNNING:
            response.success = False
            response.message = "Cannot unload while inference is running"
            return response
        if self.state == STATE_LOADING:
            response.success = False
            response.message = "Policy is currently loading"
            return response
        if not self.inference_manager.is_loaded():
            response.success = True
            response.message = "No policy is loaded"
            self._set_state(STATE_IDLE)
            return response

        self.inference_manager.unload()
        self._set_state(STATE_IDLE)
        response.success = True
        response.message = "Policy unloaded"
        return response

    # ─── Service: move to preset pose ─────────────────────────────────────

    def _on_move_to_pose(
        self,
        request: MoveToPose.Request,
        response: MoveToPose.Response,
    ):
        self.get_logger().info(f"{request}")
        if self.state == STATE_RUNNING:
            response.success = False
            response.message = "Cannot move while inference is running"
            return response

        target = request.target
        pose_location = request.pose_location

        if target == MoveToPose.Request.TARGET_ROBOT:
            pose = self._robot_poses.get(pose_location)
            if pose is None:
                response.success = False
                response.message = (
                    f"Unknown robot pose '{pose_location}'. "
                    f"Available: {list(self._robot_poses.keys())}"
                )
                return response

            expression_type = (request.expression_type or "").strip().lower()
            if expression_type == "joint":
                msg, err = self._build_joint_state_msg(pose)
                if err:
                    response.success = False
                    response.message = err
                    return response
                self.communicator.publish_joint_command(msg)
                response.success = True
                response.message = f"Published joint_state for '{pose_location}'"
                return response

            if expression_type in ("quat", "rot6d"):
                msg, err = self._build_goal_pose_msg(pose)
                self.get_logger().info(f"{msg}")
                if err:
                    response.success = False
                    response.message = err
                    return response
                self.communicator.publish_goal_pose(msg)
                self.get_logger().info(f"--------------------------------")
                response.success = True
                response.message = f"Published goal_pose for '{pose_location}'"
                return response

            response.success = False
            response.message = f"Unsupported expression_type '{request.expression_type}'"
            return response

        if target == MoveToPose.Request.TARGET_GRIPPER:
            pose = self._gripper_poses.get(pose_location)
            if pose is None or not pose:
                response.success = False
                response.message = (
                    f"Unknown gripper pose '{pose_location}'. "
                    f"Available: {list(self._gripper_poses.keys())}"
                )
                return response
            msg = Float32()
            msg.data = float(pose[0])
            self.communicator.publish_gripper_command(msg)
            response.success = True
            response.message = f"Published gripper_command for '{pose_location}'"
            return response

        response.success = False
        response.message = f"Unknown target '{target}'"
        return response

    def _build_joint_state_msg(self, pose: list[float]) -> tuple[JointState | None, str]:
        names = self._robot_joint_list
        n = len(names)
        if len(pose) < n:
            return None, f"joint pose length ({len(pose)}) < joint_list length ({n})"
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(x) for x in pose[:n]]
        msg.velocity = [0.0] * n
        msg.effort = [0.0] * n
        return msg, ""

    def _build_goal_pose_msg(self, pose: list[float]) -> tuple[PoseStamped | None, str]:
        if len(pose) < 7:
            return None, (
                f"goal pose length ({len(pose)}) < 7; expected [x, y, z, qx, qy, qz, qw]"
            )
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._pose_frame_id
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = float(pose[2])
        msg.pose.orientation.x = float(pose[3])
        msg.pose.orientation.y = float(pose[4])
        msg.pose.orientation.z = float(pose[5])
        msg.pose.orientation.w = float(pose[6])
        return msg, ""

    # ─── Action: run inference ────────────────────────────────────────────

    def _accept_goal(self, goal_request):
        if goal_request.command == RunInference.Goal.COMMAND_STOP:
            return GoalResponse.ACCEPT
        if not self.inference_manager.is_loaded():
            self.get_logger().warn("[max_server] reject goal: policy not loaded")
            return GoalResponse.REJECT
        if self.state == STATE_RUNNING:
            self.get_logger().warn("[max_server] reject goal: already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _accept_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _execute_inference(self, goal_handle):
        goal = goal_handle.request

        if goal.command == RunInference.Goal.COMMAND_STOP:
            result = RunInference.Result()
            result.success = True
            result.message = "Stop command acknowledged"
            result.total_steps = int(self._step)
            goal_handle.succeed()
            return result

        # START
        self._task_instruction = goal.task_instruction
        self._expression_type = goal.expression_type or self._expression_type
        self._step = 0
        self._last_action = None
        self.inference_manager.reset()
        self._set_state(STATE_RUNNING)

        period = 1.0 / self._fps if self._fps > 0 else 0.033
        result = RunInference.Result()
        tick_error: str | None = None

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = True
                    result.message = "Inference canceled"
                    break

                started = time.perf_counter()
                error = self._tick()
                if error:
                    tick_error = error
                    goal_handle.abort()
                    result.success = False
                    result.message = error
                    break

                # feedback
                fb = RunInference.Feedback()
                fb.step = int(self._step)
                fb.last_action = (
                    [float(x) for x in self._last_action]
                    if self._last_action is not None else []
                )
                goal_handle.publish_feedback(fb)

                elapsed = time.perf_counter() - started
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            else:
                result.success = True
                result.message = "rclpy shutting down"
                goal_handle.succeed()
        finally:
            result.total_steps = int(self._step)
            if tick_error:
                self._set_state(STATE_ERROR, tick_error)
            else:
                self._set_state(STATE_READY if self.inference_manager.is_loaded() else STATE_IDLE)
            self._task_instruction = ""

        return result

    # ─── Inference tick ───────────────────────────────────────────────────

    def _tick(self) -> str | None:
        """One inference step. Returns error message on failure, None on success/skip."""
        obs = self.communicator.get_latest_observation(expression_type=self._expression_type)
        if obs is None:
            return None  # skip this tick, wait for data
        
        state = np.concatenate([obs["robot_state"], np.array([obs["gripper_state"]], dtype=np.float32)])

        try:
            action = self.inference_manager.predict(
                images=obs["images"],
                state=state,
                task_instruction=self._task_instruction,
                expression_type=self._expression_type,
            )
            
        except Exception as e:
            return f"predict failed: {e}"

        try:
            self._publish_action(action)
        except Exception as e:
            return f"publish failed: {e}"

        self._last_action = action
        self._step += 1
        return None

    def _publish_action(self, action: np.ndarray):
        """Publish action: robot command per expression_type + gripper command (last dim).

        action layout (inference_manager always returns in the "quat" form for
        the Cartesian case, gripper in the last slot):
          joint: [j0..j6, gripper]                     len=8
          quat : [x, y, z, qx, qy, qz, qw, gripper]    len=8
          rot6d: decoded back to quat form inside InferenceManager → same as quat
        """
        action_robot = action[:-1]
        action_gripper = action[-1]
        expr = self._expression_type

        if expr == "joint":
            msg_robot, err = self._build_joint_state_msg(list(action_robot))
            if err:
                self.get_logger().error(f"[max_server] joint action msg error: {err}")
            else:
                self.communicator.publish_joint_command(msg_robot)
        elif expr in ("quat", "rot6d"):
            msg_robot, err = self._build_goal_pose_msg(list(action_robot))
            if err:
                self.get_logger().error(f"[max_server] {expr} action msg error: {err}")
            else:
                self.communicator.publish_goal_pose(msg_robot)
        else:
            self.get_logger().error(f"[max_server] unsupported expression_type '{expr}'")
            return

        msg_gripper = Float32()
        msg_gripper.data = float(action_gripper)
        self.communicator.publish_gripper_command(msg_gripper)


def _resolve_inference_domain(args=None) -> int:
    """Pre-parse the inference ROS_DOMAIN_ID before initializing the main Context.

    rclpy.init(domain_id=...) must be called before constructing the Node, but
    the YAML param is only readable from a Node. So we spin up a throwaway
    Context+Node on the default domain just to read the parameter, then tear
    it down before the real init.
    """
    from rclpy._rclpy_pybind11 import SignalHandlerOptions
    probe_ctx = Context()
    rclpy.init(
        args=args, context=probe_ctx,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    try:
        probe = Node(
            "max_server_probe",
            context=probe_ctx,
            automatically_declare_parameters_from_overrides=True,
        )
        try:
            if probe.has_parameter("inference.ros_domain_id"):
                return int(probe.get_parameter("inference.ros_domain_id").value)
            return 0
        finally:
            probe.destroy_node()
    finally:
        rclpy.shutdown(context=probe_ctx)


def main(args=None):
    inference_domain = _resolve_inference_domain(args=args)

    rclpy.init(args=args, domain_id=inference_domain)
    node = MaxServerNode()
    node.get_logger().info(
        f"[max_server] main context on ROS_DOMAIN_ID={inference_domain}"
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        try:
            node.communicator.shutdown()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
