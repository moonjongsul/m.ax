"""플래너 노드: 인식된 상태 + 목표 -> subtask 순서. 상태를 갖지 않는다.

    ros2 run max_ai max_ai_planner --ros-args -p config_file:=<yaml>

지금 몸통은 시연에서 읽어낸 조회 테이블이다 (`lerobot/policies/mvla/planner/
rules.py`: 실패 없는 에피소드 217개 중 216개에서 작업자 본인의 subtask 순서를
재현). 노드로 떼어 둔 건 이걸 나중에 VLM으로 바꿀 자리이기 때문이고, 서비스
경계가 있으면 그 교체가 노드 교체로 끝난다.

**재시도 예산은 여기 없다.** 이 노드는 "다음에 뭘 할지"만 답한다. 몇 번까지
다시 시도할지는 매니저의 `TaskManager`에 남아 있어서, 플래너를 바꿨을 때 셀이
부품을 몇 번 다시 집을지가 조용히 바뀌지 않는다.

플랜을 못 세우는 상태(`unknown_pose`, 기권, 빈 매트)는 실패로 답한다. 재시도
조건이 아니라 사람이 필요한 상황이다 -- 두 정규 자세 중 어느 쪽도 아닌 부품에
대고 추측하는 것이 셀이 부품을 깨는 방식이다.
"""

from __future__ import annotations

import rclpy
from max_interfaces.srv import MakePlan, ReplanTask
from rclpy.node import Node

from ..rollout.planner import LocalPlanSource, Plan, UnplannableState
from .ros_util import declare_config


class PlannerNode(Node):
    def __init__(self):
        super().__init__("max_ai_planner")
        self.cfg = declare_config(self)
        self.source = LocalPlanSource()
        rc = self.cfg.ros
        self.create_service(MakePlan, rc.make_plan_service, self._on_make_plan)
        self.create_service(ReplanTask, rc.replan_service, self._on_replan)
        self.get_logger().info(f"ready: {rc.make_plan_service}  {rc.replan_service}")

    def _on_make_plan(self, req, resp):
        try:
            plan = self.source.plan(req.goal, req.state)
        except UnplannableState as exc:
            resp.success, resp.reason = False, str(exc)
            self.get_logger().warn(f"make_plan({req.goal!r}, {req.state!r}): {exc}")
            return resp
        resp.success = True
        resp.task_prompt = plan.task_prompt
        resp.subtasks = list(plan.subtasks)
        self.get_logger().info(f"make_plan({req.goal!r}, {req.state!r}) -> {plan}")
        return resp

    def _on_replan(self, req, resp):
        # 현재 플랜을 그대로 재구성해서 넘긴다. 트레이 재시도 판정이 남은
        # 순서의 모양에 걸려 있기 때문이다 -- `place`는 남아 있는데 `move`가
        # 없는 `pick`은 그 자체가 이미 트레이 재시도다.
        current = Plan(goal=req.goal, state=req.state, task_prompt=req.task_prompt,
                       subtasks=list(req.current_subtasks))
        try:
            plan = self.source.replan(current, req.failed_subtask)
        except UnplannableState as exc:
            resp.success, resp.reason = False, str(exc)
            self.get_logger().warn(f"replan after {req.failed_subtask!r}: {exc}")
            return resp
        resp.success = True
        resp.subtasks = list(plan.subtasks)
        self.get_logger().info(f"replan after {req.failed_subtask!r} -> {plan}")
        return resp


def main(argv=None):
    rclpy.init(args=argv)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
