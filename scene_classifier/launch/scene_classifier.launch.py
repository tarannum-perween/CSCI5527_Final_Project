import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Absolute path to the PyTorch model file (.pth)'
    )

    classifier_node = Node(
        package='scene_classifier',
        executable='scene_classifier_node.py',
        name='scene_classifier_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'range_m': 5.0,
            'grid_size': 64
        }],
        output='screen'
    )

    manager_node = Node(
        package='scene_classifier',
        executable='nav2_context_manager_node.py',
        name='nav2_context_manager_node',
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        classifier_node,
        manager_node
    ])
