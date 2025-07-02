#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import math

class RobotWarmUp(Node):
    def __init__(self):
        super().__init__('ur5_warm_up_routine')
        self._client = ActionClient(self, FollowJointTrajectory, '/manipulator_controller/follow_joint_trajectory')
        
        # UR5 joint names
        self.joint_names = [
            'ur_arm_shoulder_pan_joint',
            'ur_arm_shoulder_lift_joint',
            'ur_arm_elbow_joint',
            'ur_arm_wrist_1_joint',
            'ur_arm_wrist_2_joint',
            'ur_arm_wrist_3_joint'
        ]
        
        # Standing position (common home position for UR5)
        self.standing_position = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        # Stretched position (horizontal stretch)
        self.stretched_position = [
            math.pi/2,  # shoulder_pan
            math.pi/3,  # shoulder_lift (horizontal)
            0.0,  # elbow (straight out)
            math.pi/3,  # wrist_1
            1.57/2,  # wrist_2
            0.0   # wrist_3
        ]
        
        self.current_goal = "stretch"  # Start with stretching first
        self.timer = self.create_timer(5.0, self.execute_warmup_movement)
        
    def execute_warmup_movement(self):
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Action server not available.')
            return
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        
        if self.current_goal == "stretch":
            point.positions = self.stretched_position
            self.current_goal = "stand"
            self.get_logger().warn('Performing warm up...')
        else:
            point.positions = self.standing_position
            self.current_goal = "stretch"
            self.get_logger().info('Returning to home position...')
        
        # Set movement duration (2 seconds)
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        
        goal_msg.trajectory.points.append(point)
        
        self._client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotWarmUp()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()