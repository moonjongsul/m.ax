"""MaxServerNode: single ROS 2 Node orchestrating communicator + inference manager.

- Service  /inference/load_policy : load a policy (framework, policy, checkpoint, device, dtype)
- Action   /inference/run          : start/stop inference loop (task_instruction, expression_type)
- Topic    /inference/status       : periodic status broadcast (1Hz)
"""

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from max_interfaces.action import RunInference
from max_interfaces.msg import InferenceStatus
from max_interfaces.srv import LoadPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header

from max_server.communication.communicator import Communicator
from max_server.inference.inference_manager import InferenceManager
from max_server.task_manager.task_manager import TaskManager
from max_server.task_planner.task_planner import TaskPlanner
from max_server.utils.config_loader import load_config


STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_RUNNING = "running"
STATE_ERROR = "error"


class MaxServerNode(Node):

    def __init__(self):
        super().__init__("max_server")

        cfg_path = self.declare_parameter("config_file", "").value
        if not cfg_path:
            raise RuntimeError("Parameter 'config_file' must be provided")
        self.get_logger().info(f"[max_server] loading config: {cfg_path}")
        self.cfg = load_config(cfg_path)

        self._inference_cfg = self.cfg.get("inference") or {}
        self._fps = float(self._inference_cfg.get("fps", 30.0))

        # ── State ─────────────────────────────────────────────────────────
        self._state_lock = threading.Lock()
        self._server_state = STATE_IDLE
        self._state_detail = ""

        self._task_instruction = ""
        self._expression_type = str(
            self._inference_cfg.get("default_expression_type", "joint")
        )
        self._step = 0
        self._last_action: np.ndarray | None = None

        # ── Callback groups ───────────────────────────────────────────────
        self._sub_group = ReentrantCallbackGroup()
        self._svc_group = MutuallyExclusiveCallbackGroup()
        self._action_group = MutuallyExclusiveCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()

        # ── Components ────────────────────────────────────────────────────
        self.communicator = Communicator(self, self.cfg)
        self.inference_manager = InferenceManager()
        self.task_manager = TaskManager(self)
        self.task_planner = TaskPlanner(self)

        # ── ROS interfaces ────────────────────────────────────────────────
        self.create_service(
            LoadPolicy, "/inference/load_policy",
            self._on_load_policy, callback_group=self._svc_group,
        )
        self._action_server = ActionServer(
            self,
            RunInference,
            "/inference/run",
            execute_callback=self._execute_inference,
            goal_callback=self._accept_goal,
            cancel_callback=self._accept_cancel,
            callback_group=self._action_group,
        )
        self._status_pub = self.create_publisher(InferenceStatus, "/inference/status", 10)
        self.create_timer(
            1.0, self._publish_status, callback_group=self._timer_group,
        )

        self.get_logger().info(
            f"[max_server] ready. fps={self._fps}, cameras={self.communicator.camera_names()}"
        )

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
        self._status_pub.publish(msg)

    # ─── Service: load policy ─────────────────────────────────────────────

    def _on_load_policy(self, request: LoadPolicy.Request, response: LoadPolicy.Response):
        if self.state == STATE_RUNNING:
            response.success = False
            response.message = "Cannot load while inference is running"
            return response
        if self.state == STATE_LOADING:
            response.success = False
            response.message = "Policy is already loading"
            return response

        self._set_state(STATE_LOADING, f"{request.framework}/{request.policy}")
        ok, msg = self.inference_manager.load(
            framework=request.framework,
            policy=request.policy,
            checkpoint=request.checkpoint,
            device=request.device,
            dtype=request.dtype,
        )
        if ok:
            self._set_state(STATE_READY)
        else:
            self._set_state(STATE_ERROR, msg)

        response.success = ok
        response.message = msg
        return response

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
        obs = self.communicator.get_latest_observation()
        if obs is None:
            return None  # skip this tick, wait for data

        state = np.concatenate([obs["joint_states"], np.array([obs["gripper_state"]], dtype=np.float32)])

        try:
            action = self.inference_manager.predict(
                images=obs["images"],
                state=state,
                task_instruction=self._task_instruction,
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
        """Publish action as joint command (first N dims) + gripper command (last dim)."""
        joint_names = (self.cfg.get("robot") or {}).get("joint_list") or []
        n = len(joint_names)
        if len(action) < n + 1:
            raise ValueError(
                f"action dim ({len(action)}) < joints+gripper ({n + 1})"
            )

        # TODO: expression_type-specific decoding (quat/rot6d). For now, joint passthrough.
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = joint_names
        joint_msg.position = [float(x) for x in action[:n]]
        joint_msg.velocity = [0.0] * n
        joint_msg.effort = [0.0] * n
        self.communicator.publish_joint_command(joint_msg)

        gripper_msg = Float32()
        gripper_msg.data = float(action[n])
        self.communicator.publish_gripper_command(gripper_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MaxServerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
