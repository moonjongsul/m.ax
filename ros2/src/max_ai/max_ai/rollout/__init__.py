"""rollout 제어 로직. ROS를 import하지 않는다.

노드 네 개(policy, manager, planner, recognizer)는 이 모듈들의 얇은 껍데기다.
여기에 ROS가 들어오는 순간 `replay`로 로봇 없이 검증하는 경로가 끊긴다 --
체크포인트를 바꿀 때마다 실기에서 처음 확인하게 된다는 뜻이다.
"""
