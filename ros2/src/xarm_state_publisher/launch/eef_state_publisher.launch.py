#!/usr/bin/env python3
"""Publish the xArm end-effector pose and twist.

Attaches to an already-running xarm driver, so it can be started and
stopped independently of teleop or a VLA policy:

    ros2 launch xarm_state_publisher eef_state_publisher.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration('config')

    declared_args = [
        DeclareLaunchArgument(
            'config',
            default_value=PathJoinSubstitution([
                FindPackageShare('xarm_state_publisher'),
                'config', 'eef_state_publisher.yaml']),
            description='Parameter file for the eef_state_publisher node'),
    ]

    node = Node(
        package='xarm_state_publisher',
        executable='eef_state_publisher',
        name='eef_state_publisher',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription(declared_args + [node])
