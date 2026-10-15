"""SmolVLA / MVLA rollout을 노드 네 개로 나눠 돌린다.

    max_ai.rollout   제어 로직 전부. ROS를 import하지 않는다.
    max_ai.nodes     그 위의 ROS 껍데기.

관측·액션 정의는 `max_data_collect`가 녹화한 것과 같아야 한다 -- 정책은 그
녹화를 보고 학습했으므로 프레임·단위·카메라 매핑이 하나라도 어긋나면 정책은
에러 없이 엉뚱하게 움직인다.
"""
