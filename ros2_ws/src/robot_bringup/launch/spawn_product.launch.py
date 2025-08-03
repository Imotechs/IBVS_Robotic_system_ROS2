from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('robot_bringup')
    urdf_path = os.path.join(pkg_dir, 'urdf', 'product.urdf')

    # Publish the URDF to a topic
    product_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='product_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path])
        }],
        output='screen'
    )

    # Spawn the robot in Ignition
    spawn_product_box = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'product_box',
            '-x', '1.5', '-y', '1.0', '-z', '0.6',
            '-topic', 'robot_description'  # must match what the publisher publishes
        ],
        output='screen'
    )

    return LaunchDescription([
        product_state_publisher,
        spawn_product_box
    ])
