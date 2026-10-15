"""ROS 노드 네 개. 전부 `max_ai.rollout`의 얇은 껍데기다.

    planner_node      rules.py -> MakePlan / ReplanTask 서비스 (무상태)
    recognizer_node   카메라 직접 구독, 예약 시각에 판정 -> verdict 토픽
    policy_node       30Hz 틱, 청크 실행, 안전 제한, xArm/그리퍼 명령
    manager_node      plan, subtask 전진, 검증, 작업 종료. 사람의 진입점

여기에는 메시지 변환과 수명 관리만 있다. 제어 로직을 끌어오면 `replay`로
로봇 없이 검증하는 경로가 끊긴다.
"""
