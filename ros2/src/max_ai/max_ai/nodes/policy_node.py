"""정책 노드: 관측 -> 청크 추론 -> 안전 제한 -> 로봇. 제어 로직은 여기 없다.

    ros2 run max_ai max_ai_policy --ros-args -p config_file:=<yaml>

입력 (전부 BEST_EFFORT -- max_data_collect와 같은 이유: 액션·EEF 토픽이
BEST_EFFORT로 퍼블리시되고, RELIABLE 구독자는 거기에 조용히 연결되지 않는다)

    카메라 3대          sensor_msgs/CompressedImage
    /xarm/eef_pose     geometry_msgs/PoseStamped   (xarm_base, TCP)
    /xarm/eef_twist    geometry_msgs/TwistStamped  (xarm_base, TCP, m/s rad/s)
    /xarm/joint_states sensor_msgs/JointState      (joint1..joint7, rad -> state j1..j7)
    그리퍼 상태         sensor_msgs/JointState
    /max_ai/task/state max_interfaces/TaskState    (매니저가 정한 프롬프트)

출력

    /xarm/vc_set_cartesian_velocity   xarm_msgs/srv/MoveVelocity, 30Hz, mm/s + rad/s,
                                      duration = command_duration_s (워치독)
    그리퍼 명령                        std_msgs/Float32, 값이 바뀔 때만
    /data_collection/rollout_action   max_interfaces/msg/TeleopAction, 30Hz (레코더용, m/s)
    /max_ai/policy/state              max_interfaces/PolicyState, 30Hz

    /max_ai/policy/run                RunPolicy.action -- 에피소드 하나 = goal 하나

**안전 제한은 전부 이 노드 안에 있다** (`rollout/action.py`). 쪼개지 않은
이유: 명령 하나가 노드 경계를 넘을 때마다 변화율 제한의 "틱 사이"가 무슨
뜻인지 흐려지고, `duration=0.2s` 워치독이 지키는 대상이 둘로 늘어난다.

**데드맨.** TaskState가 `task_state_timeout_s`보다 오래 끊기면 스스로 멈춘다.
매니저가 죽으면 마지막 프롬프트로 팔이 계속 움직이기 때문이다. `RunPolicy`
goal에 `task_instruction`이 담겨 오면(SmolVLA, 매니저 없이 돌리는 경우)
데드맨을 끈다 -- 아무도 갱신하지 않는 프롬프트를 끊겼다고 볼 수 없다.

시작 시 mode 5(카르테시안 속도) + state 0을 설정하고, 끝나면 0 속도를 보낸다.
노드가 죽어도 duration 워치독으로 팔은 멈춘다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from max_interfaces.action import RunPolicy
from max_interfaces.msg import PolicyState, TaskState
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32

from ..rollout.action import Command, to_xarm_speeds
from ..rollout.executor import ACTIVE, HALT, IDLE, PROBING, PolicyExecutor, PromptSpec, ThreadedWorker
from ..rollout.observation import RawObservation, decode_compressed
from ..rollout.policy_runner import PolicyRunner
from .ros_util import WarnOnce, declare_config, latched_qos, now_s, sensor_qos, to_sec

try:
    from xarm_msgs.srv import MoveVelocity, SetInt16
except ImportError as exc:  # pragma: no cover - 로봇 PC에는 있어야 한다
    raise ImportError("xarm_msgs가 필요하다 (xarm_ros2 워크스페이스를 source 할 것)") from exc

try:
    from max_interfaces.msg import TeleopAction
except ImportError:  # 녹화 연동 없이도 제어는 돈다
    TeleopAction = None

_STATE_CODE = {
    "idle": PolicyState.IDLE,
    "starting": PolicyState.STARTING,
    "running": PolicyState.RUNNING,
    "stopped": PolicyState.STOPPED,
    "aborted": PolicyState.ABORTED,
}
_MODE = {
    TaskState.ACTIVE: ACTIVE,
    TaskState.PROBING: PROBING,
    TaskState.IDLE: IDLE,
    TaskState.DONE: HALT,
    TaskState.BLOCKED: HALT,
}


class _Latest:
    """스트림당 최신 메시지와 수신 시각."""

    def __init__(self):
        self.msg = None
        self.recv = 0.0


class PolicyNode(Node):
    def __init__(self):
        super().__init__("max_ai_policy")
        self.cfg = declare_config(self)
        rc = self.cfg.ros
        self.warn = WarnOnce(self)
        self._lock = threading.Lock()
        self._latest: dict[str, _Latest] = {}
        self._inflight = 0

        qos = sensor_qos(rc.qos_depth)
        subs = ReentrantCallbackGroup()
        for name, topic in rc.cameras.items():
            self._subscribe(f"cam/{name}", topic, CompressedImage, qos, subs)
        self._subscribe("pose", rc.eef_pose, PoseStamped, qos, subs)
        self._subscribe("twist", rc.eef_twist, TwistStamped, qos, subs)
        self._subscribe("gripper", rc.gripper_state, JointState, qos, subs)
        self._subscribe("joint", rc.joint_state, JointState, qos, subs)

        # 명령 경로 -- teleop 노드와 같은 것을 쓴다
        # Reentrant: 명령 경로의 응답끼리 직렬화할 이유가 없다. 워커가 남아
        # 있어도 set_mode/set_state 응답이 vel 응답 뒤로 밀리는 것을 막는다.
        cmd_group = ReentrantCallbackGroup()
        self.vel_client = self.create_client(MoveVelocity, rc.velocity_service, callback_group=cmd_group)
        self.mode_client = self.create_client(SetInt16, rc.set_mode_service, callback_group=cmd_group)
        self.state_client = self.create_client(SetInt16, rc.set_state_service, callback_group=cmd_group)
        self.gripper_pub = self.create_publisher(Float32, rc.gripper_command, 10)
        self.action_pub = (
            self.create_publisher(TeleopAction, rc.rollout_action_topic, qos) if TeleopAction else None
        )
        self.state_pub = self.create_publisher(PolicyState, rc.policy_state_topic, qos)

        # 정책
        self.runner = PolicyRunner(self.cfg.policy.checkpoint, self.cfg.policy.device,
                                   self.cfg.policy.num_inference_steps)
        missing = sorted(set(self.runner.camera_names) - set(rc.cameras))
        if missing:
            raise ValueError(f"policy needs cameras {missing} that the config does not subscribe")
        self.worker = ThreadedWorker(self.runner, self.cfg, clock=lambda: now_s(self))
        self.exec = PolicyExecutor(self.cfg, self.runner, self.worker, self._snapshot,
                                   self.cfg.control.task_state_timeout_s)

        # 매니저가 정한 것. standalone=True는 데드맨 꺼짐을 뜻한다.
        self._prompt = PromptSpec(mode=IDLE, received_at=now_s(self), standalone=True)
        ctl = MutuallyExclusiveCallbackGroup()
        self.create_subscription(TaskState, rc.task_state_topic, self._on_task_state,
                                 latched_qos(), callback_group=ctl)
        self._goal_handle = None
        self._goal_lock = threading.Lock()
        self._standalone = True
        self._log_path: Path | None = None
        self.action_server = ActionServer(
            self, RunPolicy, rc.run_policy_action,
            execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_timer(1.0 / self.cfg.control.fps, self._on_tick, callback_group=ctl)
        self.get_logger().info(
            f"ready: {self.runner.config.type} {self.runner.path} "
            f"speed_scale={self.cfg.safety.speed_scale} action={rc.run_policy_action}"
        )

    # ─────────────────────────────────────────────────────────── inputs
    def _subscribe(self, key, topic, msg_type, qos, group):
        self._latest[key] = _Latest()

        def cb(msg, _key=key):
            with self._lock:
                slot = self._latest[_key]
                slot.msg, slot.recv = msg, now_s(self)

        self.create_subscription(msg_type, topic, cb, qos, callback_group=group)
        self.get_logger().info(f"subscribe {key:<18} {topic}")

    def _snapshot(self, with_images: bool) -> RawObservation | None:
        """최신 메시지들로 RawObservation. 하나라도 없으면 None.

        stamp는 가장 오래된 소스의 수신 시각 -- 정지 판단이 가장 늦은 스트림을
        기준으로 하도록.
        """
        with self._lock:
            latest = {k: (v.msg, v.recv) for k, v in self._latest.items()}
        if any(m is None for m, _ in latest.values()):
            return None
        pose, twist, grip = latest["pose"][0], latest["twist"][0], latest["gripper"][0]
        joints = latest["joint"][0]
        for label, msg in (("eef_pose", pose), ("eef_twist", twist)):
            fid = msg.header.frame_id
            if fid and fid != self.cfg.ros.expected_frame_id:
                self.warn(f"{label} frame_id {fid!r} != {self.cfg.ros.expected_frame_id!r}")
        raw = RawObservation(
            stamp=min(r for _, r in latest.values()),
            eef_twist=np.array([twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z,
                                twist.twist.angular.x, twist.twist.angular.y, twist.twist.angular.z]),
            eef_position=np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]),
            eef_quat=np.array([pose.pose.orientation.x, pose.pose.orientation.y,
                               pose.pose.orientation.z, pose.pose.orientation.w]),
        )
        # 설정된 순서 그대로. 빠진 관절은 0으로 채우지 않고 프레임을 버린다 --
        # 0 rad는 유효한 자세라서, 조용히 틀린 state로 추론하는 것보다 낫다.
        by_name = dict(zip(joints.name, joints.position))
        missing = [j for j in self.cfg.ros.joint_names if j not in by_name]
        if missing:
            self.warn(f"joint_states is missing {missing}")
            return None
        raw.joint_position = np.array([by_name[j] for j in self.cfg.ros.joint_names])
        try:
            i = list(grip.name).index(self.cfg.ros.gripper_joint)
            raw.gripper_position = float(grip.position[i]) if i < len(grip.position) else 0.0
            raw.gripper_effort_raw = float(grip.effort[i]) if i < len(grip.effort) else 0.0
        except ValueError:
            self.warn(f"gripper joint {self.cfg.ros.gripper_joint!r} not in {list(grip.name)}")
            return None
        if with_images:
            for key, (msg, _) in latest.items():
                if key.startswith("cam/"):
                    img = decode_compressed(bytes(msg.data))
                    if img is None:
                        self.warn(f"{key}: undecodable frame")
                        return None
                    raw.images[key[4:]] = img
        return raw

    def _on_task_state(self, msg: TaskState) -> None:
        if self._standalone:
            return  # SmolVLA 단독 실행 중. 매니저 프롬프트는 무시한다.
        self._prompt = PromptSpec(
            mode=_MODE.get(msg.state, IDLE),
            seq=msg.seq,
            policy_prompt=msg.policy_prompt,
            head_prompt=msg.head_prompt or None,
            subtask_started_at=to_sec(msg.subtask_started_at),
            received_at=now_s(self),
            standalone=False,
        )

    # ─────────────────────────────────────────────────────────── action
    def _on_goal(self, request):
        with self._goal_lock:
            if self._goal_handle is not None:
                self.get_logger().warn("RunPolicy rejected: an episode is already running")
                return GoalResponse.REJECT
        if self._snapshot(False) is None:
            waiting = [k for k, v in self._latest.items() if v.msg is None]
            self.get_logger().warn(f"RunPolicy rejected: no data yet from {waiting}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        req = goal_handle.request
        with self._goal_lock:
            self._goal_handle = goal_handle
        try:
            return self._run(goal_handle, req)
        finally:
            with self._goal_lock:
                self._goal_handle = None
            self._send(self.exec.safety.stop())

    def _run(self, goal_handle, req):
        result = RunPolicy.Result()
        if req.checkpoint and str(Path(req.checkpoint).expanduser()) != str(self.runner.path):
            goal_handle.publish_feedback(RunPolicy.Feedback(stage="loading"))
            try:
                self._reload(req.checkpoint)
            except Exception as exc:  # noqa: BLE001
                goal_handle.abort()
                result.success, result.reason = False, f"checkpoint load failed: {exc}"
                result.state = PolicyState.ABORTED
                self.get_logger().error(result.reason)
                return result
        if req.speed_scale > 0:
            # ActionSafety가 들고 있는 것과 같은 객체다. 매 틱 cfg를 다시 읽으므로
            # 여기서 바꾸면 다음 틱부터 적용된다.
            self.cfg.safety.speed_scale = float(req.speed_scale)
            self.get_logger().info(f"speed_scale={self.cfg.safety.speed_scale} (from goal)")
        if self.cfg.ros.setup_velocity_mode and not self._setup_mode():
            goal_handle.abort()
            result.success, result.reason = False, "failed to set xArm mode 5 / state 0"
            result.state = PolicyState.ABORTED
            return result

        # SmolVLA나 매니저 없는 실행: 프롬프트를 goal에서 받고 데드맨을 끈다.
        self._standalone = bool(req.task_instruction)
        if self._standalone:
            self._prompt = PromptSpec(mode=ACTIVE, seq=0, policy_prompt=req.task_instruction,
                                      head_prompt=None, subtask_started_at=now_s(self),
                                      received_at=now_s(self), standalone=True)
            self.get_logger().info(f"standalone: {req.task_instruction!r} (no task-state deadman)")
        elif self.runner.is_mvla:
            self._prompt = PromptSpec(mode=IDLE, received_at=now_s(self), standalone=False)
        else:
            goal_handle.abort()
            result.success, result.reason = False, "SmolVLA needs task_instruction in the goal"
            result.state = PolicyState.ABORTED
            return result

        goal_handle.publish_feedback(RunPolicy.Feedback(stage="ready"))
        self.exec.begin(now_s(self), req.max_episode_s if req.max_episode_s > 0 else None)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = Path(self.cfg.log_dir) / f"rollout_{stamp}.jsonl"

        last_fb = 0.0
        while self.exec.active:
            if goal_handle.is_cancel_requested:
                self.exec.stop("cancelled")
                break
            now = now_s(self)
            if now - last_fb >= 0.2:
                last_fb = now
                goal_handle.publish_feedback(RunPolicy.Feedback(
                    stage="running", state=_STATE_CODE[self.exec.status],
                    elapsed_s=self.exec.elapsed(now), chunk_index=self.exec.executor.index(now),
                    latency_s=self.exec.latency_s,
                    value_episode=0.0 if self.exec.last_head is None else self.exec.last_head.value_episode,
                ))
            time.sleep(0.02)

        self._send(self.exec.safety.stop())
        self._standalone = True   # 다음 goal까지 매니저 프롬프트를 받지 않는다
        result.state = _STATE_CODE[self.exec.status]
        result.reason = self.exec.reason
        result.success = self.exec.status == "stopped"
        result.ticks, result.moving_ticks = self.exec.ticks, self.exec.moving_ticks
        result.log_path = self._write_log()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f"episode end: {self.exec.status} ({self.exec.reason}) "
                               f"{self.exec.moving_ticks}/{self.exec.ticks} moving ticks")
        return result

    def _reload(self, checkpoint: str) -> None:
        self.get_logger().info(f"loading {checkpoint}")
        self.worker.close()
        self.runner = PolicyRunner(checkpoint, self.cfg.policy.device, self.cfg.policy.num_inference_steps)
        missing = sorted(set(self.runner.camera_names) - set(self.cfg.ros.cameras))
        if missing:
            raise ValueError(f"policy needs cameras {missing} that the config does not subscribe")
        self.worker = ThreadedWorker(self.runner, self.cfg, clock=lambda: now_s(self))
        self.exec = PolicyExecutor(self.cfg, self.runner, self.worker, self._snapshot,
                                   self.cfg.control.task_state_timeout_s)
        self.cfg.policy.checkpoint = checkpoint

    # ───────────────────────────────────────────────────────────── tick
    def _on_tick(self) -> None:
        now = now_s(self)
        was_active = self.exec.active
        tick = self.exec.tick(now, self._prompt)
        if tick.command is not None and (was_active or self.exec.active):
            self._send(tick.command)
        self._publish_state(now, tick)

    def _setup_mode(self) -> bool:
        ok = True
        steps = ((self.mode_client, 5, "set_mode(5)"), (self.state_client, 0, "set_state(0)"))
        for client, value, label in steps:
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{label}: service unavailable")
                return False
            req = SetInt16.Request()
            req.data = value
            future = client.call_async(req)
            deadline = time.monotonic() + 3.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            res = future.result() if future.done() else None
            if res is None or res.ret != 0:
                self.get_logger().error(f"{label} failed: {None if res is None else (res.ret, res.message)}")
                ok = False
        time.sleep(0.2)  # 모드 전환 직후의 명령은 무시될 수 있다
        return ok

    # ─────────────────────────────────────────────────────────── output
    def _send(self, cmd: Command) -> None:
        active = self.exec.active
        if self.vel_client.service_is_ready():
            if self._inflight > 5:
                self.warn("velocity service is falling behind (>5 calls in flight)")
            else:
                req = MoveVelocity.Request()
                req.speeds = to_xarm_speeds(cmd)
                req.is_tool_coord = False
                req.duration = float(self.cfg.control.command_duration_s)
                self._inflight += 1
                self.vel_client.call_async(req).add_done_callback(self._on_vel_done)
        else:
            self.warn(f"{self.cfg.ros.velocity_service} not available")

        if cmd.gripper is not None:
            self.gripper_pub.publish(Float32(data=float(cmd.gripper)))

        if self.action_pub is not None:
            msg = TeleopAction()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.space = TeleopAction.CARTESIAN_VELOCITY if active else TeleopAction.IDLE
            # 녹화 스키마와 같은 단위(m/s)로. 데모 녹화의 action 열과 바로 합쳐진다.
            msg.cartesian_velocity = [float(x) for x in cmd.twist]
            msg.is_tool_coord = False
            msg.gripper_width_percent = float(cmd.gripper_target)
            self.action_pub.publish(msg)

    def _on_vel_done(self, future):
        self._inflight = max(0, self._inflight - 1)
        res = future.result()
        if res is not None and res.ret != 0:
            self.warn(f"vc_set_cartesian_velocity ret={res.ret} {res.message}")

    def _publish_state(self, now: float, tick) -> None:
        raw = self._snapshot(False)
        msg = PolicyState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = _STATE_CODE[tick.status]
        msg.reason = tick.reason
        msg.task_seq = tick.task_seq
        msg.policy_type = str(self.runner.config.type)
        msg.checkpoint = str(self.runner.path)
        msg.is_mvla = bool(self.runner.is_mvla)
        msg.chunk_index = tick.chunk_index
        msg.latency_s = float(tick.latency_s)
        msg.observation_age_s = float(min(tick.observation_age_s, 1e6))
        msg.head_seq = self.exec.head_seq
        h = self.exec.last_head
        if h is not None:
            msg.subtask_probs = [float(x) for x in h.subtask_probs]
            msg.value_subtask = float(h.value_subtask)
            msg.value_subtask_spread = float(h.value_subtask_spread)
            msg.value_episode = float(h.value_episode)
            msg.status_probs = [float(x) for x in h.status_probs]
        if raw is not None:
            msg.gripper_position = float(raw.gripper_position)
            msg.gripper_effort = float(raw.effort_normalized(self.cfg.ros.gripper_effort_max))
        if tick.command is not None:
            msg.command_velocity = [float(x) for x in tick.command.twist]
            msg.gripper_target = float(tick.command.gripper_target)
            msg.flags = list(tick.command.flags)
        self.state_pub.publish(msg)

    def _write_log(self) -> str:
        if self._log_path is None:
            return ""
        from dataclasses import asdict

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            f.write(json.dumps({"config": asdict(self.cfg), "status": self.exec.status,
                                "reason": self.exec.reason}, default=float) + "\n")
            for row in self.exec.log:
                f.write(json.dumps(row, default=float) + "\n")
        self.get_logger().info(f"rollout log: {self._log_path}")
        return str(self._log_path)

    def shutdown(self) -> None:
        self.exec.stop("shutdown")
        stop = self.exec.safety.stop()
        if self.vel_client.service_is_ready():
            req = MoveVelocity.Request()
            req.speeds = to_xarm_speeds(stop)
            req.is_tool_coord = False
            req.duration = 0.0
            self.vel_client.call_async(req)
        self.worker.close()


def main(argv=None):
    rclpy.init(args=argv)
    node = PolicyNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        time.sleep(0.2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
