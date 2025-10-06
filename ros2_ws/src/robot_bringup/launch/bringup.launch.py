import os
import yaml
import time
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess, OpaqueFunction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import rclpy.logging
logger = rclpy.logging.get_logger("ur5_2f85.launch")

# LOAD FILE:
def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        # parent of IOError, OSError *and* WindowsError where available.
        return None

def rewrite_yaml(source_file: str, root_key: str):
    if not root_key:
        return source_file

    with open(source_file, 'r') as file:
        ori_data = yaml.safe_load(file)

    updated_yaml = {root_key: ori_data}
    dst_path = f"/tmp/{time.time()}.yaml"
    with open(dst_path, 'w') as file:
        yaml.dump(updated_yaml, file)
    return dst_path

def load_yaml(package_path, file_path):

    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def load_robot(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    robot_name = LaunchConfiguration('robot_name', default='ridgeback_ur5')
    namespace = LaunchConfiguration('namespace', default='')
    gripper_name = LaunchConfiguration('gripper_name', default='robotiq_2f_85')
    position_x = LaunchConfiguration('position_x', default='0.2')
    position_y = LaunchConfiguration('position_y', default='0.0')
    orientation_yaw = LaunchConfiguration('orientation_yaw', default='3.5')

    robot_name_val = robot_name.perform(context)
    namespace_val = namespace.perform(context)
    gripper_name_val = gripper_name.perform(context)
    logger.info(f"Loading Robot [{gripper_name_val}] with name: {robot_name_val}")


    controller_yaml = "config/ur5_2f85_controllers.yaml"
    moveit_pkg = "ridgeback_ur5_moveit_config"

    # Updating controller parameters
    sim_dir = get_package_share_directory('ridgeback_ur5_gazebo')
    pkg_dir = get_package_share_directory('ur5_robotiq_description')
    controller_file = rewrite_yaml(
        source_file=os.path.join(sim_dir, controller_yaml),
        root_key=namespace_val
    )

    # print("controller file: "+str(controller_file))

    # Loading Robot Model
    robot_xacro = Command([
        'xacro ', os.path.join(pkg_dir, 'urdf/ur5_2f85/ur5_2f85_main.urdf.xacro'),
        ])
    robot_description = {"robot_description": robot_xacro}

    # Publish TF
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        remappings=remappings,
        output="both",
        parameters=[robot_description]
    )

    # Spawn Robot in Gazebo
    spawn_robot_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", f"{namespace_val}/robot_description",
            "-entity", robot_name_val,
            "-x", position_x,
            "-y", position_y,
            "-z", "0.8",
            "-Y", orientation_yaw,
        ],
        output="screen"
    )

    # Load Controllers
    joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster', '-c', f"{namespace_val}/controller_manager"],
        output="screen"
    )

    ur5_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'manipulator_controller', '-c', f"{namespace_val}/controller_manager"],
        output="screen"
    )

    robotiq_controller = None
    if gripper_name_val == "robotiq_2f_85":
        robotiq_controller = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'robotiq_2f85_controller', '-c', f"{namespace_val}/controller_manager"],
            output="screen"
        )
    elif gripper_name_val == "robotiq_2f_140":
        robotiq_controller = ExecuteProcess(
            cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'robotiq_2f140_controller', '-c', f"{namespace_val}/controller_manager"],
            output="screen"
        )
    else:
        robotiq_controller = ExecuteProcess(
            cmd=['echo', '"No grippers loaded"'],
            output="screen"
        )


    load_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot_node,
            on_exit=[
                joint_state_controller,
                ur5_controller,
                robotiq_controller
            ]
        )
    )

    return [
        DeclareLaunchArgument(
            name="use_sim_time",
            default_value=use_sim_time,
            description="Use simulator time",
            choices=["True", "False"]
        ),
        DeclareLaunchArgument(
            name="robot_name",
            default_value=robot_name,
            description="Name of the robot"
        ),
        DeclareLaunchArgument(
            name="namespace",
            default_value=namespace,
            description="Namespace for the robot"
        ),
        DeclareLaunchArgument(
            name="gripper_name",
            default_value=gripper_name,
            description="Name of gripper to use",
            choices=["", "robotiq_2f_85", "robotiq_2f_140"]
        ),
        DeclareLaunchArgument(
            name="position_x",
            default_value=position_x,
            description="X position to spawn the robot"
        ),
        DeclareLaunchArgument(
            name="position_y",
            default_value=position_y,
            description="Y position to spawn the robot"
        ),
        DeclareLaunchArgument(
            name="orientation_yaw",
            default_value=orientation_yaw,
            description="Yaw angle to spawn robot"
        ),

        robot_state_publisher,
        spawn_robot_node,
        load_controllers
        # load_move_group,
        # MoveInterface
    ]



def generate_launch_description():
    ign_gz = LaunchConfiguration('ign_gz', default='false')
    
    # Loading Gazebo
    world = os.path.join(get_package_share_directory("ridgeback_ur5_gazebo"), "worlds/empty.world")
    gazebo_classic_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [os.path.join(get_package_share_directory("gazebo_ros"), "launch"), "/gazebo.launch.py"]
                    ),
                    launch_arguments={
                        'world': world,
                        'verbose': 'true',
                        'pause': 'false'
                    }.items(),
                )


    pkg_dir = get_package_share_directory('ur5_robotiq_description')

    robot_xacro = Command([
        'xacro ', os.path.join(pkg_dir, 'urdf/ur5_2f85/ur5_2f85_main.urdf.xacro'),
        ])
    robot_description = {"robot_description": robot_xacro}

    controller_config = PathJoinSubstitution([
                    FindPackageShare("ridgeback_ur5_gazebo"),
                    "config",
                    "ur5_2f85_controllers.yaml"
                        ])
    manager_node =Node(
            package='controller_manager',
            executable='ros2_control_node',
                    parameters=[robot_description, controller_config],
                    output='screen'
                )

    return LaunchDescription([
        
        manager_node,
        gazebo_classic_node,
        OpaqueFunction(function=load_robot)
    ])
