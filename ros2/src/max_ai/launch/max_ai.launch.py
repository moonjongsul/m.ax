"""노드 네 개를 같은 설정 파일 하나로 띄운다.

    ros2 launch max_ai max_ai.launch.py
    ros2 launch max_ai max_ai.launch.py config_file:=/path/to/other.yaml
    ros2 launch max_ai max_ai.launch.py recognizer:=false      # 인식기 없이
    ros2 launch max_ai max_ai.launch.py manager:=false         # 정책만 (SmolVLA 단독)

네 노드가 **같은 파일**을 읽는다. 노드마다 파라미터를 따로 두면 fps나 카메라
이름이 다른 상태로 뜰 수 있고, 그러면 청크 타이밍과 인식기 캡처 시각이 조용히
틀어진다.

정책과 인식기는 각각 GPU에 모델을 올린다. 한 GPU에 둘이 같이 올라가는 것을
전제로 설정돼 있다 -- 인식기는 경계에서만 돌아서 정책의 30Hz를 방해하지
않는다. 메모리가 부족하면 설정에서 `recognizer.device`를 다른 GPU로 돌릴 것.

이 셀은 전부 ROS_DOMAIN_ID 0이라 노드는 환경에서 도메인을 물려받는다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_cfg = PathJoinSubstitution([
        FindPackageShare("max_ai"), "config", "xarm7_kitting.yaml",
    ])
    args = [
        DeclareLaunchArgument("config_file", default_value=default_cfg,
                              description="네 노드가 공유하는 rollout YAML"),
        DeclareLaunchArgument("policy", default_value="true"),
        DeclareLaunchArgument("manager", default_value="true"),
        DeclareLaunchArgument("planner", default_value="true"),
        DeclareLaunchArgument("recognizer", default_value="true"),
    ]
    cfg = LaunchConfiguration("config_file")

    def node(name, executable, condition_arg):
        return Node(
            package="max_ai", executable=executable, name=name,
            output="screen", emulate_tty=True,
            parameters=[{"config_file": cfg}],
            condition=IfCondition(LaunchConfiguration(condition_arg)),
        )

    return LaunchDescription(args + [
        # 플래너와 인식기가 먼저 떠 있어야 매니저가 시작 판정을 할 수 있다.
        # 서비스가 없으면 매니저는 검증 없이 돌기 때문에 순서를 강제하지는
        # 않는다 -- 다만 인식기 없이 돌고 있다는 경고가 뜬다.
        node("max_ai_planner", "max_ai_planner", "planner"),
        node("max_ai_recognizer", "max_ai_recognizer", "recognizer"),
        node("max_ai_policy", "max_ai_policy", "policy"),
        node("max_ai_manager", "max_ai_manager", "manager"),
    ])
