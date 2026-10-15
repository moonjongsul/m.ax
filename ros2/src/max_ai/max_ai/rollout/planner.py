"""플랜 공급자: 매니저가 플래너를 어디서 얻는지만 추상화한다.

`TaskManager`는 플랜을 `_plan_for` / `_replan_for` 훅으로 얻는다. 여기서
그 훅을 구현해 주입한다. 두 가지 경로:

    LocalPlanSource   lerobot의 rules.py를 직접 호출 (replay, 테스트)
    RosPlanSource     planner 노드의 MakePlan / ReplanTask 서비스 (실기)

노드로 쪼갠 이유는 rules.py 주석에 적힌 의도 그대로다 -- 지금은 시연에서
읽어낸 조회 테이블이고(216/217), 나중에 VLM으로 바꿀 자리다. 서비스 경계를
두면 그 교체가 노드 교체로 끝난다.

**재시도 예산은 여기로 넘어오지 않는다.** 플래너는 "다음에 뭘 할지"만
답하고, 몇 번까지 다시 시도할지는 `TaskManager`에 남는다. 플래너를 바꿨을 때
셀이 부품을 몇 번 다시 집을지가 조용히 바뀌면 안 된다.
"""

from __future__ import annotations

from typing import Protocol

from lerobot.policies.mvla.planner.manager import ManagerConfig, TaskManager
from lerobot.policies.mvla.planner.rules import (
    Plan,
    UnplannableState,
    plan as local_plan,
    replan_after_failure as local_replan,
)


class PlanSource(Protocol):
    """플랜 두 질문. 둘 다 플랜을 못 세우면 `UnplannableState`를 던진다."""

    def plan(self, goal: str, object_state: str) -> Plan: ...

    def replan(self, plan: Plan, failed_subtask: str) -> Plan: ...


class LocalPlanSource:
    """rules.py 직접 호출. ROS 없이 도는 경로(replay, 테스트)의 기본값."""

    def plan(self, goal: str, object_state: str) -> Plan:
        return local_plan(goal, object_state)

    def replan(self, plan: Plan, failed_subtask: str) -> Plan:
        return local_replan(plan, failed_subtask)


class SourcedTaskManager(TaskManager):
    """플랜만 외부에서 받는 `TaskManager`. 나머지 동작은 전부 그대로."""

    def __init__(self, config: ManagerConfig | None = None, source: PlanSource | None = None):
        super().__init__(config)
        self.source: PlanSource = source or LocalPlanSource()

    def _plan_for(self, object_state: str) -> Plan:
        return self.source.plan(self.config.goal, object_state)

    def _replan_for(self, plan: Plan, failed_subtask: str) -> Plan:
        return self.source.replan(plan, failed_subtask)


__all__ = ["LocalPlanSource", "Plan", "PlanSource", "SourcedTaskManager", "UnplannableState"]
