"""pi0.5 단독 노드: 관측 콜백 -> 연속 추론 -> 로봇. 노드 하나로 끝난다.

    ros2 run max_ai max_ai_pi05 --ros-args -p config_file:=<yaml>

`smolvla_node`와 **제어 모델이 같다** -- 액션 큐 + 자기 속도 루프. 그쪽
docstring에 적힌 이유(청크를 벽시계로 인덱싱하지 않아 구조적으로 끊기지
않고, 대신 지연 보상이 없다)가 여기도 그대로 적용된다. 안전 제한도
`rollout/action.py`의 `ActionSafety`를 그대로 쓴다 -- 다시 구현하지 않는다.

**pi0.5가 SmolVLA와 다른 점**은 셋이고, 전부 여기서 흡수한다:

  * **플로우 매칭 스텝 수의 속성 이름.** SmolVLA는 `config.num_steps`,
    pi0.5는 `config.num_inference_steps`다. `PolicyRunner`가
    `set_inference_steps`로 있는 쪽에만 쓴다 -- 없는 쪽에 대입하면 조용히 새
    속성이 생기고 설정값이 무시된다.
  * **액션 큐의 위치.** SmolVLA는 `_queues["action"]`과
    `_check_get_actions_condition()`을 갖지만 pi0.5는 `_action_queue`뿐이고
    그 메서드가 없다. `runner.queue_is_empty()`가 둘 다 처리한다.
  * **dtype.** pi0.5 config에는 `dtype`("float32" | "bfloat16")이 있다.
    float32 체크포인트를 bfloat16 autocast로 돌리면 10스텝 적분이 내내
    저정밀로 누적되므로, `PolicyRunner._autocast`가 선언된 dtype을 따른다.

청크 길이는 `n_action_steps`(pi0.5 기본 50)만큼 소비한다. `chunk_size`가
아니라 이쪽인 이유는 `select_action`이 큐에 채우는 길이가 그것이기 때문이다.

MVLA는 쓰지 않는다 -- `smolvla_node`와 같은 이유로, 헤드 읽기와 subtask
전진은 `policy_node` + `manager_node`의 몫이다.
"""


from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from max_interfaces.action import RunPolicy
from max_interfaces.msg import PolicyState
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32

from ..rollout.action import ActionSafety, Command, to_xarm_speeds
from ..rollout.observation import RawObservation, decode_compressed, to_policy_observation
from ..rollout.policy_runner import PolicyRunner, inference_steps
from .ros_util import WarnOnce, declare_config, now_s, sensor_qos

try:
    from xarm_msgs.srv import MoveVelocity, SetInt16
except ImportError as exc:  # pragma: no cover - 로봇 PC에는 있어야 한다
    raise ImportError("xarm_msgs가 필요하다 (xarm_ros2 워크스페이스를 source 할 것)") from exc

try:
    from max_interfaces.msg import TeleopAction
except ImportError:  # 녹화 연동 없이도 제어는 돈다
    TeleopAction = None


# pi0 계열만 받는다. SmolVLA 체크포인트도 이 노드에서 **돌기는** 하지만,
# 그러면 어느 노드가 무엇을 돌렸는지가 PolicyState와 로그에서만 구분되고
# 설정 파일에서는 구분되지 않는다. 섞어 쓰다 생기는 혼동이 이 셀에서 비싸므로
# 거절하고 max_ai_smolvla로 보낸다.
# LeRobot의 `PreTrainedConfig.register_subclass` 이름 그대로.
_PI0_TYPES = ("pi0", "pi05", "pi0_fast")


def _check_policy_type(runner) -> None:
    if runner.is_mvla:
        raise ValueError(
            "이 노드는 pi0 계열 전용이다. MVLA 체크포인트는 "
            "max_ai_policy + max_ai_manager로 돌릴 것 (헤드와 subtask 프롬프트가 필요하다)"
        )
    kind = str(runner.config.type)
    if kind not in _PI0_TYPES:
        hint = " SmolVLA는 max_ai_smolvla로 돌릴 것." if kind == "smolvla" else ""
        raise ValueError(
            f"이 노드는 pi0 계열{_PI0_TYPES} 전용인데 체크포인트가 {kind!r}이다.{hint}"
        )


class _Latest:
    def __init__(self):
        self.msg = None
        self.recv = 0.0


class PI05Node(Node):
    def __init__(self):
        # 파라미터 서비스 7개를 끈다. `config_file` 하나만 읽고 런타임에 파라미터를
        # 바꾸지 않는데, 이것들이 executor의 wait set에 들어가 매 스핀마다
        # 파이썬에서 순회된다 -- 30Hz 루프에서 그 비용이 실측으로 드러났다.
        super().__init__("max_ai_pi05", start_parameter_services=False)
        self.cfg = declare_config(self)
        rc = self.cfg.ros
        self.warn = WarnOnce(self)
        self._lock = threading.Lock()
        self._latest: dict[str, _Latest] = {}

        qos = sensor_qos(rc.qos_depth)
        # eef_pose/eef_twist는 150Hz로 온다. 제어는 30Hz이고 쓰는 건 최신값
        # 하나뿐이라 초당 300개 중 240개가 버려질 메시지다. 그런데도 rclpy가
        # 매 콜백 사이클마다 wait set을 파이썬에서 재구성하므로 비용은 그대로
        # 든다(py-spy: MainThread의 76%가 wait_for_ready_callbacks). depth 1로
        # 받고 콜백에서 30Hz를 넘는 것은 즉시 버린다.
        fast_qos = sensor_qos(1)
        # 간격을 정확히 1/fps로 두면 150Hz 격자에서 주기마다 한 개씩 더 버려져
        # 실효 26Hz가 된다(150/6 대신 150/5). 소스 주기의 절반을 빼서 30Hz에
        # 맞춘다 -- 제어가 30Hz이므로 그보다 드물게 받으면 안 된다.
        min_interval = 1.0 / self.cfg.control.fps * 0.9
        subs = ReentrantCallbackGroup()
        for name, topic in rc.cameras.items():
            self._subscribe(f"cam/{name}", topic, CompressedImage, qos, subs)
        self._subscribe("pose", rc.eef_pose, PoseStamped, fast_qos, subs,
                        min_interval=min_interval)
        self._subscribe("twist", rc.eef_twist, TwistStamped, fast_qos, subs,
                        min_interval=min_interval)
        self._subscribe("gripper", rc.gripper_state, JointState, qos, subs)
        self._subscribe("joint", rc.joint_state, JointState, qos, subs)

        # 명령 경로는 **별도 노드 + 별도 executor** -- smolvla_node와 같은 이유.
        # `_execute`가 에피소드 내내 자기 executor를 점유하므로, 거기에 달린
        # 클라이언트의 응답 콜백은 돌지 못한다(set_mode 타임아웃, _inflight 누적).
        self._cmd_node = rclpy.create_node(
            "max_ai_pi05_cmd", start_parameter_services=False,
        )
        self._cmd_exec = SingleThreadedExecutor()
        self._cmd_exec.add_node(self._cmd_node)
        threading.Thread(target=self._cmd_exec.spin, name="cmd-exec", daemon=True).start()
        self.vel_client = self._cmd_node.create_client(MoveVelocity, rc.velocity_service)
        self.mode_client = self._cmd_node.create_client(SetInt16, rc.set_mode_service)
        self.state_client = self._cmd_node.create_client(SetInt16, rc.set_state_service)
        self.gripper_pub = self.create_publisher(Float32, rc.gripper_command, 10)
        self.action_pub = (
            self.create_publisher(TeleopAction, rc.rollout_action_topic, qos) if TeleopAction else None
        )
        self.state_pub = self.create_publisher(PolicyState, rc.policy_state_topic, qos)

        self.runner = PolicyRunner(self.cfg.policy.checkpoint, self.cfg.policy.device,
                                   self.cfg.policy.num_inference_steps)
        _check_policy_type(self.runner)
        missing = sorted(set(self.runner.camera_names) - set(rc.cameras))
        if missing:
            raise ValueError(f"policy needs cameras {missing} that the config does not subscribe")

        self.safety = ActionSafety(self.cfg.safety, self.cfg.control.fps)
        self._inflight = 0
        self._goal_handle = None
        self._goal_lock = threading.Lock()
        self._stop_flag = threading.Event()
        # 상태 리포트용. 루프가 쓰고 타이머가 읽는다.
        self._step = 0
        self._infer_count = 0
        self._last_infer_s = 0.0
        self._last_cmd: Command | None = None
        self._status = "idle"
        self._reason = ""

        # 액션 서버도 **별도 노드** -- `_execute`가 에피소드 내내 머무르므로
        # 구독과 같은 executor에 두면 관측이 갱신되지 않는다.
        self.action_node = rclpy.create_node(
            "max_ai_pi05_action", start_parameter_services=False,
        )
        self.action_server = ActionServer(
            self.action_node, RunPolicy, rc.run_policy_action,
            execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=self._on_cancel,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_timer(1.0 / self.cfg.control.fps, self._publish_state,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f"ready: {self.runner.config.type} {self.runner.path} fps={self.cfg.control.fps} "
            f"steps={inference_steps(self.runner.config)} "
            f"queue={getattr(self.runner.config, 'n_action_steps', '?')} "
            f"speed_scale={self.cfg.safety.speed_scale} action={rc.run_policy_action}"
        )

    # ─────────────────────────────────────────────────────────── inputs
    def _subscribe(self, key, topic, msg_type, qos, group, min_interval: float = 0.0):
        """`min_interval`을 주면 그보다 자주 온 메시지는 저장하지 않고 버린다.

        버려도 되는 이유는 `_snapshot`이 최신값 하나만 읽기 때문이다. 게이트를
        콜백 맨 앞에 두어 락도 잡지 않는다 -- 락 경합이 GIL 보유 시간을 늘린다.
        """
        self._latest[key] = _Latest()

        def cb(msg, _key=key):
            now = now_s(self)
            slot = self._latest[_key]
            if min_interval and slot.msg is not None and now - slot.recv < min_interval:
                return
            with self._lock:
                slot.msg, slot.recv = msg, now

        self.create_subscription(msg_type, topic, cb, qos, callback_group=group)
        self.get_logger().info(f"subscribe {key:<18} {topic}")

    def _snapshot(self, with_images: bool) -> RawObservation | None:
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
        # policy_node._snapshot과 같은 처리. 빠진 관절은 0으로 채우지 않는다.
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

    # ─────────────────────────────────────────────────────────── action
    def _on_goal(self, request):
        with self._goal_lock:
            if self._goal_handle is not None:
                self.get_logger().warn("RunPolicy rejected: an episode is already running")
                return GoalResponse.REJECT
        if not request.task_instruction:
            self.get_logger().warn("RunPolicy rejected: pi0.5 needs task_instruction")
            return GoalResponse.REJECT
        if self._snapshot(False) is None:
            waiting = [k for k, v in self._latest.items() if v.msg is None]
            self.get_logger().warn(f"RunPolicy rejected: no data yet from {waiting}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        self._stop_flag.set()
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """에피소드를 **전용 스레드**에 넘기고 끝날 때까지 기다린다.

        `_run`을 여기서 바로 돌리면 rclpy executor 안에서 돌게 되고, 그러면
        executor의 스핀 루프와 GIL을 다툰다 -- py-spy로 재보니 MainThread가
        `wait_for_ready_callbacks`에서 GIL을 84% 쥐고 있어 제어 루프가 30Hz 중
        12Hz밖에 못 돌았다(워커 4개는 전부 유휴였다).

        **executor는 단일 스레드이므로 여기서 블로킹하면 안 된다** -- 그러면
        스핀이 멈춰 카메라·포즈 콜백이 끊기고, 제어 루프가 볼 관측이 영원히
        낡은 값이 된다. 그래서 `thread.join()`을 짧은 타임아웃으로 쪼개고,
        기다리는 동안 executor가 돌 수 있게 GIL을 놓는다. 액션 서버 콜백은
        `ReentrantCallbackGroup`이라 이 대기가 다른 콜백을 막지 않는다.
        """
        with self._goal_lock:
            self._goal_handle = goal_handle
        self._stop_flag.clear()
        try:
            return self._run(goal_handle, goal_handle.request)
        except Exception as exc:  # noqa: BLE001 - 여기서 죽으면 goal이 매달린다
            self.get_logger().error(f"control loop failed: {exc}")
            result = RunPolicy.Result()
            result.success, result.reason = False, f"control loop failed: {exc}"
            result.state = PolicyState.ABORTED
            if goal_handle.is_active:
                goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._goal_handle = None
            self._send(self.safety.stop(), active=False)
            self._status = "idle" if self._status == "running" else self._status

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
            self.cfg.safety.speed_scale = float(req.speed_scale)
            self.get_logger().info(f"speed_scale={self.cfg.safety.speed_scale} (from goal)")
        if self.cfg.ros.setup_velocity_mode and not self._setup_mode():
            goal_handle.abort()
            result.success, result.reason = False, "failed to set xArm mode 5 / state 0"
            result.state = PolicyState.ABORTED
            return result

        raw = self._snapshot(False)
        self.safety.reset(gripper_open=1.0 if raw is None or raw.gripper_position >= 0.5 else 0.0)
        self.runner.policy.reset()   # 액션 큐를 비우고 시작한다
        self._step = self._infer_count = 0
        self._status, self._reason = "running", ""
        task = req.task_instruction
        max_episode_s = req.max_episode_s if req.max_episode_s > 0 else self.cfg.control.max_episode_s
        period = 1.0 / self.cfg.control.fps
        goal_handle.publish_feedback(RunPolicy.Feedback(stage="ready"))
        self.get_logger().info(f"begin: {task!r} max_episode_s={max_episode_s}")

        t_start = now_s(self)
        moving = 0
        last_fb = 0.0
        while rclpy.ok():
            started = time.perf_counter()
            if self._stop_flag.is_set() or goal_handle.is_cancel_requested:
                self._status, self._reason = "stopped", "cancelled"
                break
            now = now_s(self)
            if now - t_start > max_episode_s:
                self._status, self._reason = "aborted", "episode timeout"
                self.get_logger().warning("aborted: episode timeout")
                break

            raw = self._snapshot(True)
            if raw is None or now - raw.stamp > self.cfg.control.stale_timeout_s:
                # 관측이 끊기면 멈추되 에피소드는 유지한다. 돌아오면 이어간다.
                cmd = self.safety.stop()
                cmd.flags = cmd.flags + ("stale_observation",)
                self._send(cmd, active=True)
                self._sleep_rest(started, period)
                continue

            try:
                action = self._infer(raw, task)
            except Exception as exc:  # noqa: BLE001
                self._status, self._reason = "aborted", f"inference failed: {exc}"
                self.get_logger().error(self._reason)
                break

            cmd = self.safety(action, raw.eef_position)
            self._send(cmd, active=True)
            self._step += 1
            if not cmd.is_zero:
                moving += 1

            if now - last_fb >= 0.2:
                last_fb = now
                goal_handle.publish_feedback(RunPolicy.Feedback(
                    stage="running", state=PolicyState.RUNNING,
                    elapsed_s=now - t_start, chunk_index=self._step,
                    latency_s=self._last_infer_s,
                ))
            self._sleep_rest(started, period)

        self._send(self.safety.stop(), active=False)
        result.state = {"stopped": PolicyState.STOPPED, "aborted": PolicyState.ABORTED}.get(
            self._status, PolicyState.STOPPED)
        result.reason = self._reason
        result.success = self._status == "stopped"
        result.ticks, result.moving_ticks = self._step, moving
        result.log_path = ""
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(
            f"episode end: {self._status} ({self._reason}) {moving}/{self._step} moving ticks, "
            f"{self._infer_count} inferences"
        )
        return result

    @staticmethod
    def _sleep_rest(started: float, period: float) -> None:
        """남은 시간만 쉰다. 추론이 period를 넘으면 쉬지 않고 바로 다음 틱으로.

        max_server와 같은 자기 속도 루프다. 느려지더라도 정지 명령을 보내지
        않는 것이 이 노드의 요점이다.
        """
        remaining = period - (time.perf_counter() - started)
        if remaining > 0:
            time.sleep(remaining)

    def _infer(self, raw: RawObservation, task: str) -> np.ndarray:
        """`select_action` 한 번. 큐가 비어 있을 때만 실제 추론이 돈다."""
        obs = to_policy_observation(raw, self.runner.camera_names, tuple(self.cfg.ros.image_size),
                                    self.cfg.ros.image_rotate, self.cfg.ros.gripper_effort_max)
        # 큐가 비어 있으면 이번 호출에서 실제 추론이 돈다. 지연을 그 호출에만
        # 기록하기 위해 미리 본다 -- 나머지 49번은 큐에서 꺼내므로 ~0ms다.
        empty_before = self.runner.queue_is_empty()
        t0 = time.perf_counter()
        batch = self.runner.pre({**obs, "task": task})
        with self.runner._autocast():
            action = self.runner.policy.select_action(batch)
        action = self.runner.post(action.float())
        if empty_before:
            self._infer_count += 1
            self._last_infer_s = time.perf_counter() - t0
        out = action.detach().cpu().numpy().astype(np.float64)
        while out.ndim > 1:
            out = out[0]
        return out

    def _reload(self, checkpoint: str) -> None:
        self.get_logger().info(f"loading {checkpoint}")
        runner = PolicyRunner(checkpoint, self.cfg.policy.device, self.cfg.policy.num_inference_steps)
        _check_policy_type(runner)
        missing = sorted(set(runner.camera_names) - set(self.cfg.ros.cameras))
        if missing:
            raise ValueError(f"policy needs cameras {missing} that the config does not subscribe")
        self.runner = runner
        self.cfg.policy.checkpoint = checkpoint

    def _setup_mode(self) -> bool:
        ok = True
        for client, value, label in ((self.mode_client, 5, "set_mode(5)"),
                                     (self.state_client, 0, "set_state(0)")):
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().error(f"{label}: service unavailable")
                return False
            req = SetInt16.Request()
            req.data = value
            future = client.call_async(req)
            deadline = time.monotonic() + 5.0
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done():
                future.cancel()
                self.get_logger().error(
                    f"{label}: no response in 5s "
                    f"(service_is_ready={client.service_is_ready()}). "
                    "False면 디스커버리가 끊긴 것이므로 노드를 재시작할 것. "
                    "True면 드라이버가 응답하지 않는 것이다 "
                    "(clean_error / clean_warn 후 재시도)."
                )
                ok = False
                continue
            res = future.result()
            if res is None or res.ret != 0:
                self.get_logger().error(
                    f"{label} failed: {None if res is None else (res.ret, res.message)}"
                )
                ok = False
        time.sleep(0.2)  # 모드 전환 직후의 명령은 무시될 수 있다
        return ok

    # ─────────────────────────────────────────────────────────── output
    def _send(self, cmd: Command, *, active: bool) -> None:
        self._last_cmd = cmd
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
            msg.cartesian_velocity = [float(x) for x in cmd.twist]   # 녹화와 같은 m/s
            msg.is_tool_coord = False
            msg.gripper_width_percent = float(cmd.gripper_target)
            self.action_pub.publish(msg)

    def _on_vel_done(self, future):
        self._inflight = max(0, self._inflight - 1)
        res = future.result()
        if res is not None and res.ret != 0:
            self.warn(f"vc_set_cartesian_velocity ret={res.ret} {res.message}")

    def _publish_state(self) -> None:
        raw = self._snapshot(False)
        msg = PolicyState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = {"idle": PolicyState.IDLE, "running": PolicyState.RUNNING,
                     "stopped": PolicyState.STOPPED, "aborted": PolicyState.ABORTED}[self._status]
        msg.reason = self._reason
        msg.policy_type = str(self.runner.config.type)
        msg.checkpoint = str(self.runner.path)
        msg.is_mvla = False
        msg.chunk_index = int(self._step)
        msg.latency_s = float(self._last_infer_s)
        msg.observation_age_s = float("inf") if raw is None else max(0.0, now_s(self) - raw.stamp)
        msg.observation_age_s = float(min(msg.observation_age_s, 1e6))
        if raw is not None:
            msg.gripper_position = float(raw.gripper_position)
            msg.gripper_effort = float(raw.effort_normalized(self.cfg.ros.gripper_effort_max))
        cmd = self._last_cmd
        if cmd is not None:
            msg.command_velocity = [float(x) for x in cmd.twist]
            msg.gripper_target = float(cmd.gripper_target)
            msg.flags = list(cmd.flags)
        self.state_pub.publish(msg)

    def shutdown(self) -> None:
        """정지 명령을 **먼저** 보내고 정리한다 -- `vel_client`가 `_cmd_node`에
        달려 있어, 그 노드를 먼저 내리면 정지 명령이 나가지 못한다."""
        self._stop_flag.set()
        stop = self.safety.stop()
        if self.vel_client.service_is_ready():
            req = MoveVelocity.Request()
            req.speeds = to_xarm_speeds(stop)
            req.is_tool_coord = False
            req.duration = 0.0
            self.vel_client.call_async(req)
            time.sleep(0.1)   # 정지 명령이 나갈 틈
        try:
            self._cmd_exec.shutdown()
        except Exception:  # noqa: BLE001 - 종료 경로
            pass
        for nd in (self._cmd_node, self.action_node):
            try:
                nd.destroy_node()
            except Exception:  # noqa: BLE001 - 종료 경로
                pass


def main(argv=None):
    rclpy.init(args=argv)
    node = PI05Node()
    # 제어 루프가 전용 스레드로 빠졌으니 executor는 콜백 저장과 명령 송출만
    # 한다. 멀티스레드 executor의 wait set 재구성이 더 비싸고, 그 비용이 바로
    # 30Hz 루프를 굶기던 것이다.
    # **executor 셋, 전부 단일 스레드** -- smolvla_node의 main() 주석에 이유와
    # 실측치가 있다. 요약: 멀티스레드는 GIL 보유 시간을 늘려 제어 루프를 굶고,
    # 하나로 묶으면 `_execute`가 점유해 구독과 서비스 응답이 멈춘다.
    #   obs_exec : 구독 + PolicyState 타이머 (절대 막히면 안 된다)
    #   act_exec : 액션 서버 (`_execute`가 에피소드 내내 점유)
    #   node._cmd_exec : 로봇 명령 클라이언트 (생성자에서 띄운다)
    obs_exec = SingleThreadedExecutor()
    obs_exec.add_node(node)
    # act_exec는 **스레드 2개** -- `_execute`가 에피소드 내내 하나를 점유하므로,
    # 하나뿐이면 `_on_cancel`이 실행될 수 없어 Ctrl-C 취소가 먹지 않는다
    # (정지 수단이 사라진다). 이 노드에는 고빈도 구독이 없어 GIL 문제는 없다.
    act_exec = MultiThreadedExecutor(num_threads=2)
    act_exec.add_node(node.action_node)
    threading.Thread(target=obs_exec.spin, name="obs-exec", daemon=True).start()
    try:
        act_exec.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        time.sleep(0.2)
        for ex in (act_exec, obs_exec):
            try:
                ex.shutdown()
            except Exception:  # noqa: BLE001 - 종료 경로
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
