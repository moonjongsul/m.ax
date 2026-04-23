"""Full bringup: cameras + domain_bridge + max_server (+ rosbridge + web_video_server)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_cameras_arg = DeclareLaunchArgument(
        "use_cameras", default_value="true",
        description="Launch camera drivers (RealSense + Orbbec)",
    )
    use_domain_bridge_arg = DeclareLaunchArgument(
        "use_domain_bridge", default_value="false",
        description="Launch domain_bridge (forwards camera topics from domain 1 -> 0)",
    )

    cameras = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("max_bringup"), "launch", "camera.launch.py"])
        ]),
        condition=IfCondition(LaunchConfiguration("use_cameras")),
    )

    domain_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("max_bringup"), "launch", "domain_bridge.launch.py"])
        ]),
        condition=IfCondition(LaunchConfiguration("use_domain_bridge")),
    )

    max_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare("max_server"), "launch", "max_server.launch.py"])
        ]),
    )

    return LaunchDescription([
        use_cameras_arg,
        use_domain_bridge_arg,
        
        cameras,
        max_server,
        # domain_bridge,
        
    ])
