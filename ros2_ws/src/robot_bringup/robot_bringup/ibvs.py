import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

HOME_Q = np.array([
    0.000068,
    -1.570000,
    1.570029,
    -1.570040,
    -1.570000,
    0.00000022
])
class UR5IBVSMotion(Node):
    def __init__(self):
        super().__init__('ibvs_motion_control')

        self.q = HOME_Q.copy()
        self.last_time = self.get_clock().now()
        self.v_cam = np.zeros(6)
        self.timeout = 2.0  # seconds
        self.pub_traj = self.create_publisher(
                JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
            )

        self.timer = self.create_timer(0.03, self.update_motion)
        self.get_logger().info("UR5 IBVS Motion Node started")
        self.is_home = True

        # send home once
        self.move_to_home()
        
    def move_to_home(self):
        current_q = self.q.copy()
        steps = 20
        for i in range(1, steps + 1):
            q_interp = current_q + (HOME_Q - current_q) * (i / steps)
            traj = JointTrajectory()
            traj.joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint',
                'elbow_joint', 'wrist_1_joint',
                'wrist_2_joint', 'wrist_3_joint'
            ]
            point = JointTrajectoryPoint()
            point.positions = q_interp.tolist()
            point.velocities = [0.0] * 6
            point.time_from_start = Duration(sec=int(0.1 * i))
            traj.points.append(point)
            self.pub_traj.publish(traj)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.q = HOME_Q.copy()
        self.get_logger().info("Smoothly returned to home position")


def main():
    rclpy.init()
    node = UR5IBVSMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()