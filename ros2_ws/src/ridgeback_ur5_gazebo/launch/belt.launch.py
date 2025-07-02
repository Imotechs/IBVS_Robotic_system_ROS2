
# import os
# from launch import LaunchDescription
# from launch.actions import ExecuteProcess, IncludeLaunchDescription
# from launch_ros.actions import Node
# from ament_index_python.packages import get_package_share_directory
# from launch.substitutions import Command
# from ament_index_python.packages import get_package_share_path

# def generate_launch_description():
#     # Path to your conveyor belt xacro file
#     pkg_name = 'ridgeback_ur5_gazebo'  # Change this to your actual package
#     conveyor_belt_xacro = os.path.join(get_package_share_path(pkg_name),'urdf','conveyor_belt.xacro')  
#     # Convert xacro to URDF
#     conveyor_belt_urdf = Command(['xacro ', conveyor_belt_xacro])
#     # Robot description parameter
#     conveyor_belt_description = {'robot_description': conveyor_belt_urdf}
    
#     # Robot State Publisher for conveyor belt
#     conveyor_state_publisher = Node(
#         package='robot_state_publisher',
#         executable='robot_state_publisher',
#         name='conveyor_state_publisher',
#         output='screen',
#         parameters=[conveyor_belt_description],
#         namespace='conveyor_belt'  # Optional namespace
#     )
    
#     # Spawn conveyor belt in Gazebo
#     spawn_conveyor_belt = Node(
#         package='ros_gz_sim',
#         executable='create',
#         arguments=[
#             '-topic', '/conveyor_belt/robot_description',
#             '-name', 'conveyor_belt',
#             '-x', '1.5',  # Position in front of the robot
#             '-y', '1.0',
#             '-z', '0.0',
#             '-Y', '0.0'
#         ],
#         output='screen'
#     )
    
#     return LaunchDescription([
#         conveyor_state_publisher,
#         spawn_conveyor_belt
#     ])

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, PathJoinSubstitution
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Package and file paths
    pkg_name = 'ridgeback_ur5_gazebo'
    bringup_pkg = 'robot_bringup'
    
    # Get world file (create a basic one if you don't have it)
    world_file = os.path.join(
        get_package_share_directory(bringup_pkg),
        'worlds', 'empty.world'  # or whatever your world file is called
    )
    
    conveyor_belt_xacro = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf', 'conveyor_belt.xacro'
    )
    
    # Controller configuration
    controller_yaml = os.path.join(
        get_package_share_directory(bringup_pkg),
        'config', 'conveyor_controller.yaml'
    )
    
    # Convert xacro to URDF
    conveyor_belt_urdf = Command([
        'xacro ', conveyor_belt_xacro, ' prefix:=belt_'
    ])
    
    # Robot description parameter
    conveyor_belt_description = {
        'robot_description': conveyor_belt_urdf,
        'use_sim_time': True
    }
    
    # 1. START GAZEBO SIMULATION
    
    
    # 2. START ROS-GAZEBO BRIDGE
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Bridge clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Bridge spawn service - THIS IS THE KEY PART!
            '/spawn_entity@ros_gz_interfaces/srv/SpawnEntity@gz.msgs.EntityFactory',
            # Bridge delete service
            '/delete_entity@ros_gz_interfaces/srv/DeleteEntity@gz.msgs.Entity',
            # Add other bridges as needed
        ],
        output='screen'
    )
    
    # Load and start controllers
    def load_controller(controller_name):
        return ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                 controller_name, '-c', '/conveyor_belt/controller_manager'],
            output='screen'
        )
    
    # Robot State Publisher for conveyor belt
    conveyor_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='conveyor_state_publisher',
        output='screen',
        parameters=[conveyor_belt_description],
        namespace='conveyor_belt'
    )
    
    # Controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[controller_yaml],
        output='screen',
        namespace='conveyor_belt'
    )
    
    # Wait for Gazebo to be ready before spawning
    spawn_conveyor_belt = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/conveyor_belt/robot_description',
            '-name', 'conveyor_belt',
            '-x', '1.5',
            '-y', '1.0',
            '-z', '0.0',
            '-Y', '0.0'
        ],
        output='screen'
    )
    
    conveyor_controller_name = Node(
        package=bringup_pkg,
        executable='conveyor_controller',
        name='conveyor_controller',
        output='screen',
        parameters=[{
            'default_speed': 0.5,
            'max_speed': 1.0,
            'acceleration': 0.2
        }]
    )
    
    # Product spawner - now it should work!
    product_spawner_node = Node(
        package='robot_bringup',
        executable='product_spawner',
        name='product_spawner',
        output='screen',
        parameters=[
            {'spawn_interval': 3.0},
            {'auto_spawn': False},  # Start with manual spawning for testing
            {'use_sim_time': True}
        ]
    )
    
    return LaunchDescription([

        
        # 2. Start the bridge (with small delay)
        TimerAction(
            period=3.0,
            actions=[bridge]
        ),
        
        # 3. Start conveyor belt components (with more delay)
        TimerAction(
            period=5.0,
            actions=[
                conveyor_state_publisher,
                spawn_conveyor_belt,
            ]
        ),
        
        # 4. Start controllers after conveyor is spawned
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_conveyor_belt,
                on_exit=[controller_manager]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=controller_manager,
                on_exit=[load_controller('belt_velocity_controller')]
            )
        ),
        
        # 5. Start other nodes
        TimerAction(
            period=8.0,
            actions=[
                conveyor_controller_name,
                product_spawner_node,
            ]
        ),
    ])