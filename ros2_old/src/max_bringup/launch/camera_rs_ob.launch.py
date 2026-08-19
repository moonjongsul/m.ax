import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cam_domain_id_arg = DeclareLaunchArgument(
        'cam_domain_id',
        default_value='1',
        description='ROS_DOMAIN_ID used by camera nodes',
    )
    cam_domain_id = LaunchConfiguration('cam_domain_id')

    set_domain_env = SetEnvironmentVariable(
        name='ROS_DOMAIN_ID',
        value=cam_domain_id,
    )

    # ── RealSense ───────────────────────────────────────────────────────
    rs_front = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/wrist',
        name='front',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'serial_no': '_315122272391',
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'depth_module.color_profile': '640x480x30',
        }],
    )

    rs_rear = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='/wrist',
        name='rear',
        output='screen',
        emulate_tty=True,
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

    # ── Orbbec Gemini2 멀티카메라 (primary/secondary sync) ───────────────
    orbbec_share = get_package_share_directory('orbbec_camera')
    gemini2_launch = os.path.join(orbbec_share, 'launch', 'gemini2.launch.py')
    primary_config = os.path.join(orbbec_share, 'config', 'camera_params.yaml')
    secondary_config = os.path.join(
        orbbec_share, 'config', 'camera_secondary_params.yaml'
    )

    common_color_args = {
        'color_width': '640',
        'color_height': '480',
        'color_fps': '30',
        'enable_color_auto_exposure': 'true',
        'color_ae_max_exposure': '800',
        'color_brightness': '85',
        'enable_depth': 'false',
        'enable_ir': 'false',
        'enable_pointcloud': 'false',
        'device_num': '2',
    }

    orbbec_primary = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gemini2_launch),
        launch_arguments={
            **common_color_args,
            'camera_name': 'front_view',
            'serial_number': 'AY35C3200EM',
            'usb_port': '6-1.3.3',
            'sync_mode': 'primary',
            'config_file_path': primary_config,
            'trigger_out_enabled': 'true',
        }.items(),
    )

    orbbec_secondary = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gemini2_launch),
        launch_arguments={
            **common_color_args,
            'camera_name': 'side_view',
            'serial_number': 'AY3794301V0',
            'usb_port': '6-1.3.2',
            'sync_mode': 'secondary_synced',
            'config_file_path': secondary_config,
            'trigger_out_enabled': 'false',
        }.items(),
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
        TimerAction(period=0.0, actions=[rs_front]),
        TimerAction(period=2.0, actions=[rs_rear]),
        TimerAction(period=4.0, actions=[GroupAction([orbbec_secondary])]),
        TimerAction(period=6.0, actions=[GroupAction([orbbec_primary])]),
        # TimerAction(period=8.0, actions=[domain_bridge_launch]),
    ]

    return LaunchDescription([
        cam_domain_id_arg,
        set_domain_env,
        *staged,
    ])
