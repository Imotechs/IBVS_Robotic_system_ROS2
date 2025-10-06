ros2 topic pub /robotiq_2f85_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "header:
  stamp:
    sec: 0
    nanosec: 0
  frame_id: ''
joint_names: ['robotiq_85_left_knuckle_joint']
points:
- positions: [0.8] # or 0.0 to close
  time_from_start:
    sec: 1
    nanosec: 0"


ros2 service call /conveyor/control ifra_conveyorbelt/srv/SetConveyorPower "{power: 50.0}"
