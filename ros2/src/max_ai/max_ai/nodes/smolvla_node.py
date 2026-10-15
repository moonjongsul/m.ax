"""SmolVLA 단독 노드: 관측 콜백 -> 연속 추론 -> 로봇. 노드 하나로 끝난다.

    ros2 run max_ai max_ai_smolvla --ros-args -p config_file:=<yaml>

`policy_node`와 **제어 모델이 다르다.** 그쪽은 청크를 벽시계로 인덱싱해서
`t_obs + k/fps` 시각의 액션을 꺼내고, 그 시각이 청크를 벗어나면 정지 명령을
보낸다. 추론이 느린 셀에서는 그게 "끊김"으로 나타난다 -- 측정된 지연
0.67초에 `execute_steps`가 25면 청크가 바닥나서 60% 틱이 정지였다.

이 노드는 max_server가 쓰던 방식을 그대로 쓴다:

  * **액션 큐.** `select_action`이 LeRobot 내부 큐를 쓴다. 큐가 비는
    50스텝마다 한 번만 추론하고, 나머지는 캐시에서 즉시 꺼낸다. 큐가 빌 때만
    추론하므로 **빈 적이 없다** -- 구조적으로 끊기지 않는다.
  * **자기 속도로 도는 루프.** 남은 시간만 쉰다. 추론이 120ms 걸리면 그 틱은
    120ms 걸리고 루프가 느려질 뿐, 정지 명령을 보내지 않는다.

그래서 **지연 보상이 없다.** 큐의 k번째 액션은 "추론 시점 + k/fps"가 아니라
"꺼낸 그 순간"에 실행된다. 루프가 30Hz를 못 지키면 액션이 녹화 때보다 느리게
재생되고, 속도 명령이므로 **이동 거리가 줄어든다**(위치 제어라면 목표가
그대로라 안 보이는 차이다). 그게 이 방식의 대가다 -- 끊김과 바꾼 것이다.

MVLA는 쓰지 않는다. 헤드 읽기, subtask 전진, 검증, 재계획은 전부 매니저가
있어야 하고, 그건 `policy_node` + `manager_node`의 몫이다. 여기서 MVLA
체크포인트를 받으면 거절한다 -- 헤드 없이 MVLA를 돌리면 프롬프트에 subtask가
빠진 채로 움직이고, 그건 학습 때 본 적 없는 입력이다.

안전 제한은 `rollout/action.py`의 `ActionSafety`를 그대로 쓴다. 다시
구현하지 않는다 -- 두 노드가 다른 상한을 갖게 되는 것이 이 셀에서 가장
위험한 종류의 차이다.
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
from ..rollout.policy_runner import PolicyRunner
from .ros_util import WarnOnce, declare_config, now_s, sensor_qos

try:
    from xarm_msgs.srv import MoveVelocity, SetInt16
except ImportError as exc:  # pragma: no cover - 로봇 PC에는 있어야 한다
    raise ImportError("xarm_msgs가 필요하다 (xarm_ros2 워크스페이스를 source 할 것)") from exc

try:
    from max_interfaces.msg import TeleopAction
except ImportError:  # 녹화 연동 없이도 제어는 돈다
    TeleopAction = None


class _Latest:
    def __init__(self):
        self.msg = None
        self.recv = 0.0


class SmolVLANode(Node):
    def __init__(self):
        # 파라미터 서비스 7개를 끈다. `config_file` 하나만 읽고 런타임에 파라미터를
        # 바꾸지 않는데, 이것들이 executor의 wait set에 들어가 매 스핀마다
        # 파이썬에서 순회된다 -- 30Hz 루프에서 그 비용이 실측으로 드러났다.
        super().__init__("max_ai_smolvla", start_parameter_services=False)
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

        # Reentrant: 응답을 기다리는 쪽(`_setup_mode`)이 executor 밖의 제어
        # 스레드다. MutuallyExclusive로도 동작하는 것은 확인했지만, 명령 경로의
        # 응답끼리 직렬화할 이유가 없고 `_on_vel_done`이 mode 응답 뒤로 밀리는
        # 것도 막는다.
        # 명령 경로 전체를 **별도 노드 + 별도 executor**에 둔다.
        #
        # 제어 루프(`_run`)는 전용 스레드에서 돌고, `_execute`는 그 스레드가
        # 끝날 때까지 메인 executor의 하나뿐인 스레드를 `thread.join()`으로
        # 쥔다. 그래서 메인 executor에 달린 클라이언트의 응답 콜백은 에피소드
        # 내내 실행되지 않는다. 증상이 둘로 나타났다:
        #   - set_mode/set_state가 매번 5초 타임아웃 (`failed: None`)
        #   - `_on_vel_done`이 돌지 않아 `_inflight`가 줄지 않고, 6틱 뒤부터
        #     "velocity service is falling behind"로 모든 속도 명령이 버려짐
        #     (실측: 15초 에피소드에서 1틱만 실행)
        # 메인 executor를 멀티스레드로 바꾸면 교착은 풀리지만 GIL 경합으로
        # 제어 루프가 30Hz -> 9.7Hz로 무너진다(main() 주석의 실측치).
        # 그래서 교착을 executor 분리로 푼다.
        self._cmd_node = rclpy.create_node(
            "max_ai_smolvla_cmd", start_parameter_services=False,
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
        if self.runner.is_mvla:
            raise ValueError(
                "이 노드는 SmolVLA 전용이다. MVLA 체크포인트는 "
                "max_ai_policy + max_ai_manager로 돌릴 것 (헤드와 subtask 프롬프트가 필요하다)"
            )
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

        # 액션 서버는 **별도 노드**에 둔다. `_execute`가 에피소드가 끝날 때까지
        # 자기 콜백 안에 머무르므로, 그 노드를 도는 executor는 그동안 멈춘다.
        # 구독과 같은 executor에 두면 관측이 갱신되지 않는다 -- main() 주석의
        # 실측 증상 참고. 이름만 다르고 토픽/액션 이름은 그대로다.
        self.action_node = rclpy.create_node(
            "max_ai_smolvla_action", start_parameter_services=False,
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
            f"ready: smolvla {self.runner.path} fps={self.cfg.control.fps} "
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
            self.get_logger().warn("RunPolicy rejected: SmolVLA needs task_instruction")
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

        **이 콜백 안에서 블로킹하면 그 동안 다른 콜백이 하나도 돌지 않는다.**
        `ReentrantCallbackGroup`으로도 해결되지 않는다 -- 그룹은 "동시에 실행될
        수 있는가"만 정하고 실행할 **스레드를 만들어 주지는 않는다.** 단일
        스레드 executor에서 그 스레드가 이 콜백에 들어와 있으면 구독·서비스
        응답 콜백이 전부 멈춘다. 실측으로 두 가지 증상이 나왔다:

          - `set_mode`/`set_state`가 매번 5초 타임아웃 (`failed: None`)
          - 모든 구독이 정지해 관측 나이가 단조 증가하고, 매 틱이
            `stale_observation`으로 버려져 15초 에피소드에서 1틱만 실행

        멀티스레드 executor는 답이 아니다 -- 교착은 풀리지만
        `wait_for_ready_callbacks`의 GIL 보유 시간이 늘어 제어 루프의
        `time.sleep`이 늦게 깨어난다(main() 주석의 실측치: 2스레드에서 최대
        66ms 지연, 30Hz -> 9.7Hz).

        그래서 **여기서 에피소드를 직접 돌린다.** rclpy의 스핀은 `main()`이
        전용 스레드에서 돌리므로, 이 콜백이 얼마나 오래 머물러도 구독과 서비스
        응답은 계속 처리된다. 액션 서버 콜백은 `ReentrantCallbackGroup`이라
        이 대기가 취소 요청을 막지도 않는다.
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
        if runner.is_mvla:
            raise ValueError("이 노드는 SmolVLA 전용이다 (MVLA는 max_ai_policy + max_ai_manager)")
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
                # 무응답의 원인이 둘인데 메시지가 같으면 구분할 수 없다:
                #   (a) 드라이버가 요청을 받고도 답을 안 준다 (컨트롤러 에러 등)
                #   (b) 디스커버리가 반쪽이다 -- 서비스는 보이는데 이 노드의
                #       클라이언트가 서버와 매칭되지 않았다. 이 경우
                #       `service_is_ready()`가 False로 떨어지는 것이 보인다.
                # CLI(`ros2 service call`)는 ros2cli 데몬을 재사용해 이미
                # 매칭된 상태로 호출하므로 (b)에서도 성공한다 -- 그래서 CLI는
                # 되는데 노드만 안 되는 증상이 나온다.
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
        msg.policy_type = "smolvla"
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
        """정지 명령을 **먼저** 보내고, 그 다음에 정리한다.

        순서가 중요하다 -- `vel_client`는 `_cmd_node`에 달려 있으므로 그 노드나
        executor를 먼저 내리면 정지 명령이 나가지 못한다.
        """
        self._stop_flag.set()
        stop = self.safety.stop()
        if self.vel_client.service_is_ready():
            req = MoveVelocity.Request()
            req.speeds = to_xarm_speeds(stop)
            req.is_tool_coord = False
            req.duration = 0.0
            self.vel_client.call_async(req)
            time.sleep(0.1)   # 정지 명령이 나갈 틈 (_cmd_exec가 아직 돌고 있다)
        try:
            self._cmd_exec.shutdown()
        except Exception:  # noqa: BLE001 - 종료 경로에서 더 할 수 있는 것이 없다
            pass
        for nd in (self._cmd_node, self.action_node):
            try:
                nd.destroy_node()
            except Exception:  # noqa: BLE001 - 종료 경로
                pass


def main(argv=None):
    rclpy.init(args=argv)
    node = SmolVLANode()
    # **executor 세 개, 전부 단일 스레드.** 멀티스레드로 묶으면 안 된다 --
    # `wait_for_ready_callbacks`가 파이썬에서 wait set을 재구성하며 GIL을 길게
    # 쥐고, 그러면 제어 루프의 `time.sleep`이 늦게 깨어난다. 실측(이 셀,
    # 카메라 3대 + 150Hz pose/twist, sleep(18ms)의 복귀 지연):
    #   1 스레드 : 중앙값 0.1ms (최대 0.6ms)
    #   2 스레드 : 중앙값 1.1ms, **최대 66ms**
    #   6 스레드 : 중앙값 2.9ms, 최대 78ms
    # 2스레드만 써도 30Hz(33ms 예산)가 깨져 실측 9.7Hz까지 떨어졌다.
    #
    # 대신 **역할별로 executor를 나눈다.** `_execute`는 에피소드가 끝날 때까지
    # 자기 콜백 안에 머무르므로, 그것이 점유하는 executor에서는 다른 콜백이
    # 하나도 돌지 못한다(단일 스레드에서는 CallbackGroup이 이를 바꾸지 못한다).
    # 한 executor에 다 몰아넣었을 때의 실측 증상:
    #   - set_mode/set_state가 매번 5초 타임아웃 (`failed: None`)
    #   - 구독이 전부 정지해 7개 소스의 관측 나이가 똑같이 단조 증가하고,
    #     매 틱이 stale_observation으로 버려져 15초에 1틱만 실행
    #
    #   obs_exec : 구독 + PolicyState 타이머. 제어 루프가 읽는 관측이 여기서
    #              갱신되므로 **절대 막히면 안 된다.** 150Hz 토픽이 여기 달려
    #              있으므로 단일 스레드여야 한다(위 실측치).
    #   act_exec : 액션 서버. **스레드 2개** -- `_execute`가 에피소드 내내 하나를
    #              점유하므로, 하나뿐이면 `_on_cancel`이 실행될 수 없어 Ctrl-C로
    #              취소가 되지 않는다(정지 수단이 사라진다). 나머지 하나가 취소
    #              요청을 받는다. 이 노드에는 고빈도 구독이 없으므로 위의 GIL
    #              문제는 생기지 않는다 -- 그것은 obs_exec 쪽 이야기다.
    #   node._cmd_exec : 로봇 명령 클라이언트(노드 생성자에서 띄운다).
    obs_exec = SingleThreadedExecutor()
    obs_exec.add_node(node)
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
