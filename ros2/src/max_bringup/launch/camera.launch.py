import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_domain_arg = DeclareLaunchArgument(
        "video_domain_id",
        default_value="1",
        description="ROS_DOMAIN_ID for web_video_server (must match camera.ros_domain_id)",
    )

    # Pass ROS_DOMAIN_ID explicitly to each Node via additional_env — relying
    # on SetEnvironmentVariable alone is unreliable because the parent shell's
    # exported ROS_DOMAIN_ID can still leak into TimerAction-delayed spawns.
    cam_env = {'ROS_DOMAIN_ID': LaunchConfiguration("video_domain_id")}

    # ── RealSense ───────────────────────────────────────────────────────
    # realsense D405
    rs_wrist_front = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/observation',
        name='wrist_front',
        output='screen',
        emulate_tty=True,
        additional_env=cam_env,
        parameters=[{
            'serial_no': '_315122272391',
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.color_profile': '640x480x30',
        }],
    )

    # realsense D405
    rs_wrist_rear = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/observation',
        name='wrist_rear',
        output='screen',
        emulate_tty=True,
        additional_env=cam_env,
        parameters=[{
            'serial_no': '_335122271613',
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.color_profile': '640x480x30',
            'rotation_filter.enable': True,
            'rotation_filter.rotation': 180.0,
        }],
    )

    # realsense D435
    rs_front_view = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/observation',
        name='front_view',
        output='screen',
        emulate_tty=True,
        additional_env=cam_env,
        parameters=[{
            'serial_no': '_233622071056',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
        }],
    )

    # realsense D435I (has IMU/Motion module; keep depth enabled at low rate
    # so base_stream can be resolved — depth topic can be ignored downstream)
    rs_side_view = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/observation',
        name='side_view',
        output='screen',
        emulate_tty=True,
        additional_env=cam_env,
        parameters=[{
            'serial_no': '_238722071506',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '640x480x30',
        }],
    )

    # ── domain_bridge ───────────────────────────────────────────────────
    domain_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('max_bringup'),
                'launch',
                'domain_bridge.launch.py',
            ])
        ]),
    )

    # USB 경합 방지: 순차 기동
    # RealSense front → rear → Orbbec secondary → Orbbec primary → domain_bridge
    # (Orbbec primary는 secondary 이후에 기동해야 sync trigger가 맞음)
    staged = [
        TimerAction(period=0.0, actions=[rs_wrist_front]),
        TimerAction(period=2.0, actions=[rs_wrist_rear]),
        TimerAction(period=4.0, actions=[rs_front_view]),
        TimerAction(period=6.0, actions=[rs_side_view]),
        # TimerAction(period=8.0, actions=[domain_bridge_launch]),
    ]

    return LaunchDescription([
        video_domain_arg,
        *staged,
    ])
