"""Launch max_server_node with config yaml + rosbridge + web_video_server.

The max_server process opens its own per-domain rclpy contexts (driven by
`<group>.ros_domain_id` in the YAML), but rosbridge and web_video_server are
external nodes that pick a single domain from their environment.

- rosbridge runs on `bridge_domain_id` (web UI's command/telemetry domain)
- web_video_server runs on `video_domain_id` (camera publish domain)

The picking cell's vision result is consumed by the web UI directly over
rosbridge (CompressedImage subscription), so it needs no extra web_video_server
here — it shares domain 0 and is reachable through the rosbridge above.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("max_server")
    default_cfg = PathJoinSubstitution([pkg_share, "config", "kitting_config.yaml"])

    cfg_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_cfg,
        description="Path to max_server YAML config",
    )
    bridge_domain_arg = DeclareLaunchArgument(
        "bridge_domain_id",
        default_value="0",
        description="ROS_DOMAIN_ID for rosbridge (must match inference.ros_domain_id)",
    )
    video_domain_arg = DeclareLaunchArgument(
        "video_domain_id",
        default_value="1",
        description="ROS_DOMAIN_ID for web_video_server (must match camera.ros_domain_id)",
    )
    call_service_timeout_arg = DeclareLaunchArgument(
        "call_service_timeout",
        default_value="60.0",
        description=(
            "rosbridge default service-call timeout (sec). Raised from the "
            "rosbridge default of 5s so long-running services (e.g. a picking "
            "cell pick cycle) don't time out."
        ),
    )
    max_server_node = Node(
        package="max_server",
        executable="max_server",
        name="max_server",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config_file")],
    )

    rosbridge = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "default_call_service_timeout": LaunchConfiguration("call_service_timeout"),
        }],
        additional_env={"ROS_DOMAIN_ID": LaunchConfiguration("bridge_domain_id")},
    )

    web_video = Node(
        package="web_video_server",
        executable="web_video_server",
        name="web_video_server",
        output="screen",
        emulate_tty=True,
        additional_env={"ROS_DOMAIN_ID": LaunchConfiguration("video_domain_id")},
    )

    return LaunchDescription([
        cfg_arg,
        bridge_domain_arg,
        video_domain_arg,
        call_service_timeout_arg,
        max_server_node,
        rosbridge,
        web_video,
    ])
