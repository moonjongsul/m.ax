"""Launch max_server_node with config yaml + rosbridge + web_video_server."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("max_server")
    default_cfg = PathJoinSubstitution([pkg_share, "config", "gt_kitting.yaml"])

    cfg_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_cfg,
        description="Path to max_server YAML config",
    )

    max_server_node = Node(
        package="max_server",
        executable="max_server_node",
        name="max_server",
        output="screen",
        emulate_tty=True,
        parameters=[{"config_file": LaunchConfiguration("config_file")}],
    )

    rosbridge = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        output="screen",
        emulate_tty=True,
    )

    web_video = Node(
        package="web_video_server",
        executable="web_video_server",
        name="web_video_server",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        cfg_arg,
        max_server_node,
        rosbridge,
        web_video,
    ])
