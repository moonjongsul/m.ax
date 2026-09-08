"""Launch the recorder with its config yaml.

Everything in this cell is on ROS_DOMAIN_ID 0, so no per-source domain
handling is needed -- the node inherits the domain from its environment.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_cfg = PathJoinSubstitution([
        FindPackageShare("max_data_collect"),
        "config", "demo_data_collect_config.yaml",
    ])
    cfg_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_cfg,
        description="Path to the max_data_collect YAML config",
    )
    recorder = Node(
        package="max_data_collect",
        executable="max_data_collect",
        name="max_data_collect",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config_file")],
    )
    return LaunchDescription([cfg_arg, recorder])
