import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_domain_arg = DeclareLaunchArgument(
        "video_domain_id",
        default_value="0",
        description="ROS_DOMAIN_ID for web_video_server (must match camera.ros_domain_id)",
    )

    env_tray_span_arg = DeclareLaunchArgument(
        "env_tray_span",
        default_value="true",
        description="Run max_ai env_tray_span (env camera -> top-down tray view)",
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
        # additional_env=cam_env,
        parameters=[{
            'serial_no': '_335122271613',
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
            'serial_no': '_315122272391',
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.color_profile': '640x480x30',
            'rotation_filter.enable': True,
            'rotation_filter.rotation': 180.0,
        }],
    )

    # ── env tray span ───────────────────────────────────────────────────
    # 다른 PC가 내는 env 영상을 받아 트레이만 펴서 /observation/env_tray/... 로
    # 낸다. 카메라 장치를 쓰지 않으므로 순차 기동에 넣지 않고, 도메인은 셸의
    # ROS_DOMAIN_ID를 그대로 쓴다 (env 토픽이 있는 쪽).
    env_tray_span = Node(
        package='max_ai',
        executable='max_ai_env_tray_span',
        name='max_ai_env_tray_span',
        output='screen',
        emulate_tty=True,
        parameters=[PathJoinSubstitution([
            FindPackageShare('max_ai'), 'config', 'env_tray_span_config.yaml',
        ])],
        condition=IfCondition(LaunchConfiguration("env_tray_span")),
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
    # RealSense wrist_front → wrist_rear → webcam → domain_bridge
    staged = [
        TimerAction(period=0.0, actions=[rs_wrist_front]),
        TimerAction(period=2.0, actions=[rs_wrist_rear]),
        # TimerAction(period=6.0, actions=[domain_bridge_launch]),
    ]

    return LaunchDescription([
        video_domain_arg,
        env_tray_span_arg,
        env_tray_span,
        *staged,
    ])
