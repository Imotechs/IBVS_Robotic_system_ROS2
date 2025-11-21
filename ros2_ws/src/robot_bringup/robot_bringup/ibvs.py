#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft

# ---- Simple joint limits for UR5 (rad, rad/s) ----
Q_MIN = np.array([-6.28] * 6)
Q_MAX = np.array([ 6.28] * 6)
QD_MAX = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # conservative

HOME_Q = np.array([0.0, -1.87, 1.57, -1.65, -2.30, 0.0])

DAMP = 1e-4          # DLS damping for J⁺
SERVO_GAIN = 1.8
CMD_HZ = 120.0       # how fast we publish joint velocities
TWIST_TIMEOUT = 0.15 # seconds without twist → fall back to hold/home

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def adjoint_from_tf(transform):
    """Build 6x6 adjoint matrix Ad(^baseTcam) from a geometry_msgs/TransformStamped."""
    t = transform.transform.translation
    q = transform.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [t.x, t.y, t.z]
    R = T[:3, :3]
    p = T[:3, 3]
    px = np.array([[0, -p[2], p[1]],
                   [p[2], 0, -p[0]],
                   [-p[1], p[0], 0]])
    Ad = np.block([[R, np.zeros((3, 3))],
                   [px @ R, R]])
    return Ad

def safe_pinv(J, lamb=DAMP):
    """Damped least-squares pseudoinverse."""
    return J.T @ np.linalg.inv(J @ J.T + (lamb**2) * np.eye(J.shape[0]))

def ur5_jacobian_identity(_q):
    """Placeholder Jacobian. Replace with real UR5 Jacobian later."""
    return np.eye(6)

# ---------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------
class UR5IBVSVelocityController(Node):
    def __init__(self):
        super().__init__('ur5_ibvs_velocity_controller')

        # --- Parameters ---
        self.declare_parameter('twist_topic', '/ibvs/target_twist')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('cmd_topic', '/forward_velocity_controller/commands')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('cmd_hz', CMD_HZ)

        self.twist_topic = self.get_parameter('twist_topic').value
        self.joint_topic = self.get_parameter('joint_topic').value
        self.cmd_topic   = self.get_parameter('cmd_topic').value
        self.base_frame  = self.get_parameter('base_frame').value
        self.cmd_dt      = 1.0 / float(self.get_parameter('cmd_hz').value)

        # Joint state
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.q = np.zeros(6)
        self.have_joints = False
        self.joint_indices = None

        # IBVS / mode state
        self.last_twist_time = 0.0
        self.qd_cmd = np.zeros(6)
        self.mode = "hold"  # "hold", "ibvs", "go_home"

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        self.create_subscription(JointState,   self.joint_topic, self.on_joint_state, 10)
        self.create_subscription(TwistStamped, self.twist_topic, self.on_twist,       10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        # Timer to continuously stream commands
        self.timer = self.create_timer(self.cmd_dt, self.on_timer)

        self.get_logger().info("✅ UR5 IBVS velocity controller ready.")
        self.get_logger().info(f"  listening: {self.twist_topic}")
        self.get_logger().info(f"  joints: {self.joint_topic}")
        self.get_logger().info(f"  publishing velocities to: {self.cmd_topic}")

    # -----------------------------------------------------------------
    # Joint States
    # -----------------------------------------------------------------
    def on_joint_state(self, msg: JointState):
        if self.joint_indices is None:
            name_to_index = {n: i for i, n in enumerate(msg.name)}
            try:
                self.joint_indices = [name_to_index[n] for n in self.joint_names]
                self.get_logger().info(f"Joint index map: {self.joint_indices}")
            except KeyError:
                # joint names not ready yet
                return

        self.q = np.array([msg.position[i] for i in self.joint_indices])
        self.have_joints = True

    # -----------------------------------------------------------------
    # Twist from IBVS
    # -----------------------------------------------------------------
    def on_twist(self, msg: TwistStamped):
        if not self.have_joints:
            return

        # Special command: go home
        if msg.header.frame_id == "go_home":
            self.mode = "go_home"
            self.last_twist_time = time.time()
            self.get_logger().info("Received go_home command from IBVS.")
            return

        # Get TF camera -> base
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, rclpy.time.Time()
            )
            Ad = adjoint_from_tf(tf)
        except Exception as e:
            self.get_logger().warn(f"TF {msg.header.frame_id}->{self.base_frame} failed: {e}")
            return

        v_cam = np.array([
            msg.twist.linear.x,  msg.twist.linear.y,  msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
        ], dtype=float)

        # Camera twist -> base twist
        v_base = SERVO_GAIN * (Ad @ v_cam)

        # IK: q̇ = J⁺ * twist
        J = ur5_jacobian_identity(self.q)
        J_p = safe_pinv(J)
        qd = J_p @ v_base
        qd = np.clip(qd, -QD_MAX, QD_MAX)

        # Store command
        self.qd_cmd = qd
        self.last_twist_time = time.time()
        self.mode = "ibvs"

        # Optional throttled logging
        if not hasattr(self, "_last_log"):
            self._last_log = 0.0
        now = time.time()
        if now - self._last_log > 1.0:
            self.get_logger().info(
                f"IBVS twist → qdot: {np.round(qd, 3)}"
            )
            self._last_log = now

    # -----------------------------------------------------------------
    # Timer: continuously publish velocity commands
    # -----------------------------------------------------------------
    def on_timer(self):
        if not self.have_joints:
            return

        now = time.time()
        dt_since_twist = now - self.last_twist_time

        # Decide mode / command
        if self.mode == "go_home":
            # PD-like velocity towards HOME_Q
            err = HOME_Q - self.q
            # simple proportional gain
            qd = 2.0 * err
            qd = np.clip(qd, -QD_MAX, QD_MAX)

            # Stop when close enough
            if np.linalg.norm(err) < 0.02:
                self.get_logger().info("Reached HOME, switching to hold.")
                self.mode = "hold"
                qd = np.zeros(6)

        elif self.mode == "ibvs":
            # If twist stopped updating, switch to hold
            if dt_since_twist > TWIST_TIMEOUT:
                qd = np.zeros(6)
                self.mode = "hold"
            else:
                qd = self.qd_cmd.copy()

        else:  # "hold" or unknown
            qd = np.zeros(6)

        # Always clamp
        qd = np.clip(qd, -QD_MAX, QD_MAX)

        # Publish to forward_velocity_controller
        msg = Float64MultiArray()
        msg.data = qd.tolist()
        self.pub_cmd.publish(msg)

# ---------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = UR5IBVSVelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
