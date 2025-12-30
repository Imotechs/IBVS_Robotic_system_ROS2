cmake_minimum_required(VERSION 3.5)
project(robot_bringup_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate message interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IBVSState.msg"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
