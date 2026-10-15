"""노드 분해 전체를 실제 DDS로 한 바퀴 돌린다. GPU도 로봇도 없이.

    source /opt/ros/<distro>/setup.bash
    source ros2/install/setup.bash
    PYTHONPATH=thirdparty/lerobot/src:ros2/src/max_ai \
        python3 ros2/src/max_ai/test/integration_rollout.py

진짜 `policy_node`와 `recognizer_node`는 체크포인트와 GPU가 필요하고, 그 둘이
없는 곳에서는 배선이 맞는지 확인할 방법이 없다. 그래서 여기서는 그 둘을
흉내 내는 노드로 바꿔 끼우고 **진짜 `manager_node`와 진짜 `planner_node`**를
돌린다. 재는 것은 추론 품질이 아니라 배선이다:

    RunRollout -> 매니저 -> MakePlan -> TaskState -> (정책) -> PolicyState
                                 ^                                  |
                                 +-- 헤드 읽기로 subtask 전진 <------+
    릴리스 감지 -> ScheduleRecognition -> (인식기) -> verdict -> 작업 완료

pytest가 자동으로 줍지 않도록 이름을 `test_`로 시작하지 않는다 -- rclpy와
DDS가 필요해서 순수 로직 테스트와 같이 돌 수 없다.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from max_interfaces.action import RunPolicy, RunRollout
from max_interfaces.msg import PolicyState, RecognitionVerdict, TaskState
from max_interfaces.srv import ScheduleRecognition
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from max_ai.nodes.manager_node import ManagerNode
from max_ai.nodes.planner_node import PlannerNode
from max_ai.nodes.ros_util import PURPOSE_NAME, latched_qos, now_s, sensor_qos
from max_ai.rollout.recognition import CaptureScheduler
from max_ai.rollout.subtask import SUBTASK_NAMES

CONFIG = os.environ.get("MAX_AI_CONFIG", "ros2/src/max_ai/config/xarm7_kitting.yaml")
# 매니저는 프롬프트 형태를 돌고 있는 체크포인트의 config.json에서 읽는다 (가중치는
# 안 읽는다). GPU 없는 곳에서는 기본값으로 저장한 config.json 하나만 있으면 된다:
#
#   python3 -c "import lerobot.policies; \
#     from lerobot.policies.mvla.configuration_mvla import MVLAConfig; \
#     MVLAConfig()._save_pretrained('/tmp/stub_checkpoint')"
#
#   MAX_AI_STUB_CHECKPOINT=/tmp/stub_checkpoint python3 .../integration_rollout.py
STUB_CHECKPOINT = os.environ.get("MAX_AI_STUB_CHECKPOINT", "")
FPS = 30.0


class StubPolicy(Node):
    """정책 노드 흉내. 추론 대신 TaskState를 보고 "다음 subtask가 보인다"고 말한다.

    `advance_after`번째 틱부터 `remaining[1]`을 예측한다. 진짜 정책이 그 구간을
    실제로 수행하는 데 걸리는 시간 대신이다. 그리퍼는 place에 들어간 뒤
    `release_after` 틱에 열린다 -- 매니저의 릴리스 감지가 이걸 본다.
    """

    def __init__(self, advance_after: int = 8, release_after: int = 10):
        super().__init__("stub_policy")
        self.advance_after = advance_after
        self.release_after = release_after
        self.task: TaskState | None = None
        self.seq_ticks = 0
        self.place_ticks = 0
        self.gripper = 0.0
        self.episode = False
        self.halted = False
        self.prompts: list[str] = []
        group = ReentrantCallbackGroup()
        self.create_subscription(TaskState, "/max_ai/task/state", self._on_task, latched_qos(),
                                 callback_group=group)
        self.pub = self.create_publisher(PolicyState, "/max_ai/policy/state", sensor_qos())
        self.server = ActionServer(self, RunPolicy, "/max_ai/policy/run",
                                   execute_callback=self._execute,
                                   goal_callback=lambda _r: GoalResponse.ACCEPT,
                                   cancel_callback=lambda _g: CancelResponse.ACCEPT,
                                   callback_group=group)
        self.create_timer(1.0 / FPS, self._tick, callback_group=group)

    def _on_task(self, msg: TaskState):
        if self.task is None or msg.seq != self.task.seq:
            self.seq_ticks = 0
            if msg.policy_prompt and msg.policy_prompt not in self.prompts:
                self.prompts.append(msg.policy_prompt)
        if msg.state in (TaskState.DONE, TaskState.BLOCKED):
            self.halted = True
        self.task = msg

    def _execute(self, goal_handle):
        self.episode = True
        while not goal_handle.is_cancel_requested and not self.halted and rclpy.ok():
            time.sleep(0.01)
        self.episode = False
        goal_handle.canceled() if goal_handle.is_cancel_requested else goal_handle.succeed()
        return RunPolicy.Result(success=True, state=PolicyState.STOPPED, reason="stub done")

    def _tick(self):
        msg = PolicyState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.is_mvla = True
        msg.policy_type = "mvla"
        msg.checkpoint = STUB_CHECKPOINT
        msg.state = PolicyState.RUNNING if self.episode else PolicyState.IDLE
        t = self.task
        msg.task_seq = 0 if t is None else t.seq
        if self.episode and t is not None and t.state == TaskState.ACTIVE:
            self.seq_ticks += 1
            remaining = list(t.remaining)
            # 아직 할 게 남았으면 다음 칸을, 마지막이면 지금 칸을 예측한다.
            want = remaining[1] if len(remaining) > 1 else (remaining[0] if remaining else "place")
            predict = want if self.seq_ticks >= self.advance_after else (remaining[0] if remaining else want)
            probs = np.zeros(7)
            probs[SUBTASK_NAMES.index(predict)] = 1.0
            msg.subtask_probs = [float(x) for x in probs]
            # 마지막 subtask에서는 value를 올려 완료 조건을 건드린다.
            done_ish = len(remaining) == 1 and self.seq_ticks >= self.advance_after
            msg.value_subtask = -0.02 if done_ish else -0.5
            msg.value_episode = -0.02 if done_ish else -0.5
            msg.status_probs = [1.0, 0.0, 0.0]
            msg.head_seq = self.seq_ticks          # 틱마다 새 읽기
            if remaining and remaining[0] == "place":
                self.place_ticks += 1
                if self.place_ticks >= self.release_after:
                    self.gripper = 1.0             # 그리퍼가 열렸다
            else:
                self.place_ticks = 0
                self.gripper = 0.0
        msg.gripper_position = float(self.gripper)
        msg.gripper_effort = 0.0
        self.pub.publish(msg)


class StubRecognizer(Node):
    """인식기 노드 흉내. 예약을 받아 예약 시각에 미리 정한 라벨을 낸다."""

    def __init__(self, mat="target", tray_sequence=("bad", "ok")):
        super().__init__("stub_recognizer")
        self.mat = mat
        self.tray_sequence = list(tray_sequence)
        self.scheduler = CaptureScheduler(0.05)
        self.requests: list[tuple[str, int, int]] = []
        self._tray_round = 0
        group = ReentrantCallbackGroup()
        self.pub = self.create_publisher(RecognitionVerdict, "/max_ai/recognizer/verdict", sensor_qos())
        self.create_service(ScheduleRecognition, "/max_ai/recognizer/schedule", self._on_schedule,
                            callback_group=group)
        self.create_timer(1.0 / FPS, self._tick, callback_group=group)

    def _on_schedule(self, req, resp):
        purpose = PURPOSE_NAME[req.purpose]
        now = now_s(self)
        ref = req.reference.sec + req.reference.nanosec * 1e-9
        rid, n, _late = self.scheduler.schedule(purpose, ref or now, list(req.offsets_s), now,
                                                cancel_pending=req.cancel_pending)
        resp.success, resp.request_id, resp.scheduled = n > 0, rid, n
        self.requests.append((purpose, rid, n))
        if purpose == "place":
            self._tray_round += 1
        return resp

    def _tick(self):
        for cap in self.scheduler.pop_due(now_s(self)):
            msg = RecognitionVerdict()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.purpose = {"start": 0, "flip": 1, "place": 2}[cap.purpose]
            msg.probe = "tray" if cap.purpose == "place" else "mat"
            msg.request_id, msg.index, msg.total = cap.request_id, cap.index, cap.total
            msg.captured_at = self.get_clock().now().to_msg()
            msg.detected = True
            msg.ood = 1.0
            if cap.purpose == "place":
                i = min(self._tray_round - 1, len(self.tray_sequence) - 1)
                bad = 0.95 if self.tray_sequence[i] == "bad" else 0.02
                msg.label = "bad" if bad > 0.5 else "ok"
                msg.prob_names, msg.prob_values = ["ok", "bad"], [1 - bad, bad]
            else:
                label = "flipped" if cap.purpose == "flip" else self.mat
                msg.label = label
                names = ["empty", "target", "flipped", "unknown_pose"]
                msg.prob_names = names
                msg.prob_values = [1.0 if n == label else 0.0 for n in names]
            self.pub.publish(msg)


def run_case(label: str, goal: str, mat: str, tray_sequence, expect_success: bool,
             expect_basis: str | None, timeout_s: float = 90.0) -> bool:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    rclpy.init(args=["--ros-args", "-p", f"config_file:={CONFIG}"])
    planner = PlannerNode()
    policy = StubPolicy()
    recognizer = StubRecognizer(mat=mat, tray_sequence=tray_sequence)
    manager = ManagerNode()
    watcher = Node("watcher")
    completions: list[dict] = []
    watcher.create_subscription(String, "/max_ai/rollout/task_complete",
                                lambda m: completions.append(json.loads(m.data)), latched_qos())

    ex = MultiThreadedExecutor(num_threads=12)
    for n in (planner, policy, recognizer, manager, watcher):
        ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    client = ActionClient(watcher, RunRollout, "/max_ai/rollout/run")
    ok = client.wait_for_server(timeout_sec=15.0)
    result = None
    if not ok:
        print("  FAIL: RunRollout action server never appeared")
    else:
        feedback: list[str] = []

        def on_fb(fb):
            line = (f"{fb.feedback.goal_state:9} {fb.feedback.subtask:14} "
                    f"remaining={len(fb.feedback.remaining)} retries={fb.feedback.total_retries} "
                    f"{fb.feedback.event}")
            if not feedback or feedback[-1] != line:
                feedback.append(line)
                print(f"  fb  {line}")

        future = client.send_goal_async(
            RunRollout.Goal(goal=goal, record=False, override_record=True, max_episode_s=60.0),
            feedback_callback=on_fb)
        deadline = time.monotonic() + 15
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        handle = future.result() if future.done() else None
        if handle is None or not handle.accepted:
            print("  FAIL: goal rejected")
        else:
            rfut = handle.get_result_async()
            deadline = time.monotonic() + timeout_s
            while not rfut.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            result = rfut.result().result if rfut.done() else None

    passed = True
    if result is None:
        print("  FAIL: no result within the timeout")
        passed = False
    else:
        print(f"\n  outcome={result.outcome!r} success={result.success} basis={result.basis!r}")
        print(f"  reason={result.reason!r}")
        print(f"  initial_state={result.initial_state!r} duration={result.duration_s:.1f}s")
        print(f"  retries={dict(zip(result.retry_subtasks, result.retry_counts))}")
        subs = json.loads(result.subtasks_json or "[]")
        print(f"  subtasks: {[(s['name'], s['outcome']) for s in subs]}")
        print(f"  prompts seen by the policy ({len(policy.prompts)}):")
        for p in policy.prompts:
            print(f"    {p}")
        print(f"  recognizer requests: {recognizer.requests}")
        if result.success != expect_success:
            print(f"  FAIL: expected success={expect_success}")
            passed = False
        if expect_basis is not None and result.basis != expect_basis:
            print(f"  FAIL: expected basis={expect_basis!r}")
            passed = False
        if expect_success and not completions:
            print("  FAIL: nothing published on task_complete")
            passed = False
        if policy.episode:
            print("  FAIL: the policy episode was never released")
            passed = False

    ex.shutdown()
    for n in (watcher, manager, recognizer, policy, planner):
        n.destroy_node()
    rclpy.shutdown()
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def main():
    if not STUB_CHECKPOINT:
        print("MAX_AI_STUB_CHECKPOINT is not set -- see the comment at the top of this file")
        return 2
    cases = [
        # 매트에 뒤집어야 하는 부품 -> flip 검증 통과 -> place 첫 판정 bad ->
        # 트레이 재시도 -> ok -> 완료. README의 113/280/282 에피소드와 같은 흐름.
        ("kit, needs flip, one tray retry", "kit", "target", ("bad", "ok"), True, "recognizer_tray"),
        # 바로 집을 수 있는 상태, 한 번에 성공
        ("kit, ready to pick, clean run", "kit", "flipped", ("ok",), True, "recognizer_tray"),
        # pick 목표: 마지막 subtask가 검증 불가 -> value로 완료
        ("pick goal, value completion", "pick", "flipped", ("ok",), True, "last_subtask_value"),
        # 인식기가 기권 -> 플랜 없음 -> 사람 필요. 재시도가 아니다.
        ("recognizer abstains, needs operator", "kit", "abstain", ("ok",), False, None),
    ]
    results = [run_case(*c) for c in cases]
    print(f"\n{'=' * 72}")
    print(f"{sum(results)}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
