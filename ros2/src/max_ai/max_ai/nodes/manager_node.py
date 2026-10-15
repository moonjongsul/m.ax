"""매니저 노드: 목표 하나를 받아 끝까지 끌고 간다. 사람의 진입점.

    ros2 run max_ai max_ai_manager --ros-args -p config_file:=<yaml>

    /max_ai/rollout/run             RunRollout.action  (사람/웹 UI가 부른다)
    /max_ai/task/state              TaskState          (정책이 읽는 프롬프트, latched)
    /max_ai/rollout/status          std_msgs/String    (JSON, 1Hz + 상태 변화 시)
    /max_ai/rollout/task_complete   std_msgs/String    (JSON, 종료 시 1회, latched)

하는 일: 시작 장면 상태를 정하고, 플래너에게 plan을 받고, plan의 subtask를
따라 정책 프롬프트를 바꾸고, 인식기가 검증할 수 있는 경계를 검증하고, 실패한
것을 재시도하고, 목표에 도달했는지 판단한다.

**모든 상태 변경은 액션 실행 스레드 한 곳에서만 일어난다.** 구독 콜백은 큐에
넣기만 한다. 쪼개기 전 `RolloutController.tick()`이 가졌던 성질 그대로다 --
잠금이 필요 없고, 같은 로직을 replay가 가상 시계로 재현할 수 있다. 플래너
서비스를 블로킹으로 부르는 것도 이 덕분에 안전하다(구독 콜백 안이 아니다).

**작업이 끝나는 지점은 plan이 정한다.** 목표가 plan을 정하므로 `pick`은
`pick`에서, `kit`은 `place`에서 끝난다. value 헤드는 에피소드의 끝을 잘
맞히지만 목표 문장을 바꿔도 값이 그대로여서(-0.383 vs -0.385), 목표에 따라
끝이 달라지는 걸 보장하는 경로는 plan뿐이다.
"""

from __future__ import annotations

import json
import queue
import threading
import time

import rclpy
from max_interfaces.action import RunPolicy, RunRollout
from max_interfaces.msg import PolicyState, RecognitionVerdict, TaskState
from max_interfaces.srv import MakePlan, RecordCommand, ReplanTask, ScheduleRecognition
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from ..rollout.episode import ACTIVE, BLOCKED, DONE, IDLE, PROBING, EpisodeManager
from ..rollout.planner import Plan, UnplannableState
from ..rollout.recognition import Verdict
from ..rollout.subtask import load_prompt_config
from .ros_util import (
    PURPOSE_CODE, PURPOSE_NAME, WarnOnce, declare_config, latched_qos, now_s, sensor_qos,
    to_sec, to_stamp,
)

_STATE_CODE = {
    ACTIVE: TaskState.ACTIVE, IDLE: TaskState.IDLE, DONE: TaskState.DONE,
    BLOCKED: TaskState.BLOCKED, PROBING: TaskState.PROBING,
}
_POLICY_NAME = {
    PolicyState.IDLE: "idle", PolicyState.STARTING: "starting", PolicyState.RUNNING: "running",
    PolicyState.STOPPED: "stopped", PolicyState.ABORTED: "aborted",
}


class _HeadRead:
    """PolicyState에서 복원한 헤드 읽기. `policy_runner.HeadRead`와 같은 모양."""

    def __init__(self, msg: PolicyState):
        import numpy as np

        self.subtask_probs = np.asarray(msg.subtask_probs, dtype=float)
        self.value_subtask = float(msg.value_subtask)
        self.value_subtask_spread = float(msg.value_subtask_spread)
        self.value_episode = float(msg.value_episode)
        self.status_probs = np.asarray(msg.status_probs, dtype=float)

    @property
    def subtask(self) -> int:
        return int(self.subtask_probs[:6].argmax())


class RosPlanSource:
    """플래너 노드의 서비스로 플랜을 받는다. 호출은 액션 스레드에서만."""

    def __init__(self, node: "ManagerNode", timeout_s: float = 3.0):
        self.node = node
        self.timeout_s = timeout_s

    def plan(self, goal: str, object_state: str) -> Plan:
        req = MakePlan.Request(goal=goal, state=object_state)
        res = self.node.call(self.node.plan_client, req, self.timeout_s, "make_plan")
        if res is None:
            raise UnplannableState(f"planner did not answer within {self.timeout_s}s")
        if not res.success:
            raise UnplannableState(res.reason)
        return Plan(goal=goal, state=object_state, task_prompt=res.task_prompt,
                    subtasks=list(res.subtasks))

    def replan(self, plan: Plan, failed_subtask: str) -> Plan:
        req = ReplanTask.Request(goal=plan.goal, state=plan.state, task_prompt=plan.task_prompt,
                                 current_subtasks=list(plan.subtasks), failed_subtask=failed_subtask)
        res = self.node.call(self.node.replan_client, req, self.timeout_s, "replan")
        if res is None:
            raise UnplannableState(f"planner did not answer within {self.timeout_s}s")
        if not res.success:
            raise UnplannableState(res.reason)
        return Plan(goal=plan.goal, state=plan.state, task_prompt=plan.task_prompt,
                    subtasks=list(res.subtasks))


class ManagerNode(Node):
    def __init__(self):
        super().__init__("max_ai_manager")
        self.cfg = declare_config(self)
        rc = self.cfg.ros
        self.warn = WarnOnce(self)

        clients = ReentrantCallbackGroup()
        self.plan_client = self.create_client(MakePlan, rc.make_plan_service, callback_group=clients)
        self.replan_client = self.create_client(ReplanTask, rc.replan_service, callback_group=clients)
        self.schedule_client = self.create_client(ScheduleRecognition, rc.schedule_recognition_service,
                                                  callback_group=clients)
        self.record_client = self.create_client(RecordCommand, rc.record_service, callback_group=clients)
        self.policy_client = ActionClient(self, RunPolicy, rc.run_policy_action, callback_group=clients)

        self.task_pub = self.create_publisher(TaskState, rc.task_state_topic, latched_qos())
        self.status_pub = self.create_publisher(String, rc.status_topic, latched_qos())
        self.complete_pub = self.create_publisher(String, rc.task_complete_topic, latched_qos())

        subs = ReentrantCallbackGroup()
        self._policy_q: queue.Queue[PolicyState] = queue.Queue()
        self._verdict_q: queue.Queue[RecognitionVerdict] = queue.Queue()
        self.create_subscription(PolicyState, rc.policy_state_topic, self._policy_q.put,
                                 sensor_qos(rc.qos_depth), callback_group=subs)
        self.create_subscription(RecognitionVerdict, rc.verdict_topic, self._verdict_q.put,
                                 sensor_qos(rc.qos_depth), callback_group=subs)

        self.mgr: EpisodeManager | None = None
        self.policy: PolicyState | None = None
        self._head_seq = 0
        self._goal_lock = threading.Lock()
        self._running = False
        self._policy_goal = None
        self._prompt_config = None
        self._record_enabled = self.cfg.ros.auto_record
        self._last_task_seq: tuple | None = None
        self.action_server = ActionServer(
            self, RunRollout, rc.run_rollout_action,
            execute_callback=self._execute,
            goal_callback=self._on_goal, cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )
        self.create_timer(1.0, self._publish_status, callback_group=MutuallyExclusiveCallbackGroup())
        self._publish_task_state(TaskState(), force=True)
        self.get_logger().info(f"ready: {rc.run_rollout_action}")

    # ───────────────────────────────────────────────── blocking helpers
    def call(self, client, req, timeout_s: float, label: str):
        """서비스 동기 호출. 액션 실행 스레드에서만 부른다.

        구독 콜백 안에서 부르면 응답 콜백이 같은 그룹에 묶여 교착한다. 모든
        상태 변경을 한 스레드에 모아 둔 이유가 이것이기도 하다.
        """
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=timeout_s):
                self.get_logger().error(f"{label}: service unavailable")
                return None
        future = client.call_async(req)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.005)
        if not future.done():
            self.get_logger().error(f"{label}: timed out after {timeout_s}s")
            return None
        return future.result()

    def _record(self, command: int) -> None:
        if not self._record_enabled:
            return
        if not self.record_client.service_is_ready():
            self.warn(f"recorder {self.cfg.ros.record_service} not available -- rollout not recorded")
            return
        self.record_client.call_async(RecordCommand.Request(command=command))

    # ─────────────────────────────────────────────────────────── action
    def _on_goal(self, request):
        with self._goal_lock:
            if self._running:
                self.get_logger().warn("RunRollout rejected: a rollout is already running")
                return GoalResponse.REJECT
            self._running = True
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        try:
            return self._run(goal_handle)
        except Exception as exc:  # noqa: BLE001
            # 여기서 예외가 새어 나가면 TaskState 발행이 멈추고, 정책은 데드맨이
            # 걸리는 1초 동안 마지막 프롬프트로 계속 움직인다. 그러니 직접
            # 정리한다: 매니저를 BLOCKED로 바꿔 알리고 정책 goal을 취소한다.
            self.get_logger().error(f"rollout crashed: {exc!r}", exc_info=True)
            if self.mgr is not None:
                self.mgr.abort(f"manager error: {exc}")
                self._publish_task_state(self._task_msg(now_s(self)), force=True)
            self._stop_policy()
            result = RunRollout.Result()
            result.success, result.outcome = False, "aborted"
            result.reason = f"manager error: {exc}"
            result.completion_json, result.subtasks_json = "{}", "[]"
            if goal_handle.is_active:
                goal_handle.abort()
            return result
        finally:
            with self._goal_lock:
                self._running = False

    def _run(self, goal_handle):
        req = goal_handle.request
        result = RunRollout.Result()
        self._record_enabled = req.record if req.override_record else self.cfg.ros.auto_record

        # 정책이 떠 있고 MVLA인지 확인한다. 프롬프트 설정을 체크포인트에서
        # 읽어야 하고, 그게 실제로 돌 체크포인트와 같아야 한다.
        policy = self._await_policy(5.0)
        if policy is None:
            return self._fail(goal_handle, result, "aborted",
                              f"no PolicyState on {self.cfg.ros.policy_state_topic}")
        if policy.is_mvla and self._prompt_config is None:
            if policy.checkpoint and policy.checkpoint != str(self.cfg.policy.checkpoint):
                self.get_logger().warn(
                    f"policy is running {policy.checkpoint} but config says "
                    f"{self.cfg.policy.checkpoint}; building prompts from the running one"
                )
            # 프롬프트 형태는 돌고 있는 체크포인트에서 읽는다. 매니저가 자기
            # 기본값으로 프롬프트를 만들면 학습 때와 다른 문장이 정책에 들어가고,
            # 그건 에러 없이 성능만 떨어지는 쪽으로 틀린다. 못 읽으면 시작하지 않는다.
            source = policy.checkpoint or self.cfg.policy.checkpoint
            try:
                self._prompt_config = load_prompt_config(source)
            except Exception as exc:  # noqa: BLE001
                return self._fail(goal_handle, result, "aborted",
                                  f"cannot read the prompt config from {source}: {exc}")

        recognizer_available = self.schedule_client.service_is_ready()
        if not recognizer_available:
            self.get_logger().warn("no recognizer service: start state from heads, no verification")
        self.mgr = EpisodeManager(self.cfg, is_mvla=policy.is_mvla, prompt_config=self._prompt_config,
                                  recognizer_available=recognizer_available, source=RosPlanSource(self))

        # 큐를 비우고 시작한다. 지난 에피소드의 헤드 읽기가 남아 있으면
        # 플랜이 서기도 전에 전진 판정에 들어간다.
        self._drain()
        now = now_s(self)
        self._record(RecordCommand.Request.START)
        try:
            events = self.mgr.begin(
                now, goal=req.goal if policy.is_mvla else "",
                task_instruction="" if policy.is_mvla else (self.cfg.policy.task or req.goal),
                initial_state=req.initial_state,
                gripper_effort=float(policy.gripper_effort),
            )
        except (ValueError, RuntimeError) as exc:
            self._record(RecordCommand.Request.DELETE)
            return self._fail(goal_handle, result, "aborted", str(exc))
        self._apply(events, now)
        self._publish_task_state(self._task_msg(now))

        if not self._start_policy(req):
            self.mgr.abort("policy did not accept the episode")
            self._record(RecordCommand.Request.SAVE)
            return self._fail(goal_handle, result, "aborted", self.mgr.reason)

        last_fb = last_beat = 0.0
        while self.mgr.active:
            if goal_handle.is_cancel_requested:
                self.mgr.stop("cancelled")
                break
            now = now_s(self)
            self._pump(now)
            if self.policy is not None and _POLICY_NAME[self.policy.state] in ("aborted", "stopped") \
                    and self.mgr.active:
                # 정책이 스스로 멈췄다(관측 끊김, 타임아웃, 데드맨). 에피소드는 끝이다.
                self.mgr.abort(f"policy {_POLICY_NAME[self.policy.state]}: {self.policy.reason}")
                break
            if now - last_beat >= 1.0 / max(self.cfg.control.task_state_hz, 1e-3):
                last_beat = now
                self._publish_task_state(self._task_msg(now), force=True)
            if now - last_fb >= 1.0:
                last_fb = now
                goal_handle.publish_feedback(self._feedback(now))
            time.sleep(0.005)

        now = now_s(self)
        self._publish_task_state(self._task_msg(now), force=True)
        self._stop_policy()
        self._record(RecordCommand.Request.SAVE)  # 실패한 rollout도 다음 학습 데이터다
        return self._finish(goal_handle, result, now)

    # ──────────────────────────────────────────────────────────── pump
    def _pump(self, now: float) -> None:
        """큐를 비우고 매니저를 전진시킨다. 상태 변경은 전부 이 안에서."""
        events = []
        while True:
            try:
                msg = self._policy_q.get_nowait()
            except queue.Empty:
                break
            self.policy = msg
            stamp = to_sec(msg.header.stamp)
            if msg.is_mvla and msg.head_seq != self._head_seq and msg.subtask_probs:
                self._head_seq = msg.head_seq
                events += self.mgr.on_head(_HeadRead(msg), stamp, msg.task_seq,
                                           float(msg.gripper_effort))
            events += self.mgr.on_tick(float(msg.gripper_position), stamp)
        while True:
            try:
                vmsg = self._verdict_q.get_nowait()
            except queue.Empty:
                break
            effort = 0.0 if self.policy is None else float(self.policy.gripper_effort)
            events += self.mgr.on_verdict(self._verdict(vmsg), effort, now)
        self._apply(events, now)
        if events:
            self._publish_task_state(self._task_msg(now))
            self._publish_status()

    def _verdict(self, msg: RecognitionVerdict) -> Verdict:
        probs = {n: float(v) for n, v in zip(msg.prob_names, msg.prob_values)}
        return Verdict(
            purpose=PURPOSE_NAME[msg.purpose], t_obs=to_sec(msg.captured_at),
            result={"label": msg.label, "probs": probs, "ood": float(msg.ood),
                    "detected": bool(msg.detected)},
            request_id=msg.request_id, index=msg.index, total=msg.total,
        )

    def _apply(self, events, now: float) -> None:
        """인식기 예약 이벤트를 서비스 호출로. 나머지는 매니저가 이미 처리했다."""
        for ev in events:
            if ev.kind == "cancel_recognition":
                # 다음 예약이 cancel_pending으로 지우므로 따로 부를 것은 없다.
                # 예약 없이 경계만 지나간 경우는 어차피 request_id로 걸러진다.
                continue
            if ev.kind != "schedule_recognition":
                continue
            d = ev.data
            res = self.call(self.schedule_client, ScheduleRecognition.Request(
                purpose=PURPOSE_CODE[d["purpose"]], offsets_s=[float(o) for o in d["offsets_s"]],
                reference=to_stamp(d["reference"]), cancel_pending=bool(d["cancel_pending"]),
            ), 1.0, f"schedule {d['purpose']}")
            if res is None or not res.success:
                why = "no response" if res is None else res.message
                self.get_logger().warn(f"schedule {d['purpose']} refused: {why} -- running unverified")
                continue
            if self.mgr.tracker is not None:
                self.mgr.tracker.note_request(d["purpose"], res.request_id)

    def _drain(self) -> None:
        for q in (self._policy_q, self._verdict_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self._head_seq = 0

    # ─────────────────────────────────────────────────── policy episode
    def _await_policy(self, timeout_s: float) -> PolicyState | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            try:
                self.policy = self._policy_q.get(timeout=0.1)
                return self.policy
            except queue.Empty:
                continue
        return self.policy

    def _start_policy(self, req) -> bool:
        if not self.policy_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"{self.cfg.ros.run_policy_action}: action server unavailable")
            return False
        goal = RunPolicy.Goal(speed_scale=req.speed_scale, max_episode_s=req.max_episode_s)
        if self.mgr is not None and not self.mgr.is_mvla:
            goal.task_instruction = self.mgr.task_instruction
        future = self.policy_client.send_goal_async(goal)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.01)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            self.get_logger().error("RunPolicy goal was rejected")
            return False
        self._policy_goal = handle
        self.get_logger().info("policy accepted the episode")
        return True

    def _stop_policy(self) -> None:
        if self._policy_goal is None:
            return
        # TaskState가 이미 DONE/BLOCKED/IDLE이라 정책은 스스로 멈췄다. 취소는
        # goal을 정리하기 위한 것이다.
        try:
            self._policy_goal.cancel_goal_async()
        except Exception:  # noqa: BLE001
            self.get_logger().warn("failed to cancel the policy goal")
        self._policy_goal = None

    # ────────────────────────────────────────────────────────── output
    def _task_msg(self, now: float) -> TaskState:
        v = self.mgr.task_state(now) if self.mgr is not None else None
        msg = TaskState()
        msg.header.stamp = self.get_clock().now().to_msg()
        if v is None:
            msg.state = TaskState.IDLE
            return msg
        msg.seq = v.seq
        msg.state = _STATE_CODE.get(v.state, TaskState.IDLE)
        msg.goal = v.goal
        msg.policy_prompt = v.policy_prompt
        msg.head_prompt = v.head_prompt
        msg.subtask = v.subtask
        msg.remaining = list(v.remaining)
        msg.subtask_started_at = to_stamp(v.subtask_started_at)
        msg.speed_bin = int(v.speed_bin)
        msg.on_last_subtask = bool(v.on_last_subtask)
        return msg

    def _publish_task_state(self, msg: TaskState, force: bool = False) -> None:
        if not force and self._last_task_seq == (msg.seq, msg.state):
            return
        self._last_task_seq = (msg.seq, msg.state)
        self.task_pub.publish(msg)

    def _feedback(self, now: float) -> RunRollout.Feedback:
        snap = self.mgr.snapshot(now) if self.mgr is not None else {}
        fb = RunRollout.Feedback()
        fb.goal_state = str(snap.get("status", ""))
        fb.subtask = str(snap.get("subtask") or "")
        fb.remaining = list(snap.get("remaining") or [])
        fb.on_last_subtask = bool(snap.get("on_last_subtask", False))
        fb.subtask_elapsed_s = float(snap.get("subtask_elapsed_s") or 0.0)
        fb.task_elapsed_s = float(snap.get("task_elapsed_s") or 0.0)
        fb.completion_streak = int(snap.get("completion_streak") or 0)
        fb.total_retries = int(sum((snap.get("retries") or {}).values()))
        fb.value_subtask = float(snap.get("value_subtask") or 0.0)
        fb.value_episode = float(snap.get("value_episode") or 0.0)
        fb.event = self.mgr.last_event if self.mgr is not None else ""
        return fb

    def _publish_status(self) -> None:
        snap = self.mgr.snapshot(now_s(self)) if self.mgr is not None else {"status": "idle"}
        if self.policy is not None:
            snap["policy"] = {
                "state": _POLICY_NAME[self.policy.state], "reason": self.policy.reason,
                "chunk_index": self.policy.chunk_index, "latency_s": round(self.policy.latency_s, 3),
                "observation_age_s": round(self.policy.observation_age_s, 3),
                "flags": list(self.policy.flags), "task_seq": self.policy.task_seq,
            }
        self.status_pub.publish(String(data=json.dumps(snap, default=float)))

    def _fail(self, goal_handle, result, outcome: str, reason: str):
        self.get_logger().error(f"rollout {outcome}: {reason}")
        result.success, result.outcome, result.reason = False, outcome, reason
        result.completion_json = "{}"
        result.subtasks_json = "[]"
        goal_handle.abort()
        return result

    def _finish(self, goal_handle, result, now: float):
        mgr = self.mgr
        completion = mgr.completion or {}
        result.outcome = mgr.status if mgr.status != ACTIVE else "stopped"
        result.reason = mgr.reason
        result.success = mgr.status == DONE
        result.basis = str(completion.get("basis", ""))
        result.initial_state = str(completion.get("initial_state") or
                                   (mgr.tracker.initial_state if mgr.tracker else "") or "")
        result.main_prompt = str(completion.get("main_prompt") or
                                 (mgr.tracker.main_prompt() if mgr.tracker else ""))
        result.duration_s = float(completion.get("duration_s") or mgr.elapsed(now))
        history = completion.get("subtasks")
        if history is None and mgr.tracker is not None:
            from dataclasses import asdict

            history = [asdict(r) for r in mgr.tracker.history]
        result.subtasks_json = json.dumps(history or [], default=float)
        result.completion_json = json.dumps(completion, default=float)
        retries = completion.get("retries") or (
            dict(mgr.tracker.manager.state.retries)
            if mgr.tracker is not None and mgr.tracker.manager.state else {}
        )
        result.retry_subtasks = list(retries)
        result.retry_counts = [int(retries[k]) for k in result.retry_subtasks]

        if mgr.completion is not None:
            # 래치 토픽이라 늦게 붙은 구독자도 받는다.
            self.complete_pub.publish(String(data=json.dumps(mgr.completion, default=float)))
            self.get_logger().info(f"task complete: {mgr.completion}")
        self._publish_status()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f"rollout end: {result.outcome} ({result.reason})")
        return result


def main(argv=None):
    rclpy.init(args=argv)
    node = ManagerNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
