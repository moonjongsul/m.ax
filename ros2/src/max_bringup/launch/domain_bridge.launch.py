from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('max_bringup'),
        'config',
        'domain_config.yaml',
    ])

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to domain_bridge YAML config',
    )

    bridge = ExecuteProcess(
        cmd=['ros2', 'run', 'domain_bridge', 'domain_bridge',
             LaunchConfiguration('config_file')],
        name='domain_bridge',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([config_file_arg, bridge])
