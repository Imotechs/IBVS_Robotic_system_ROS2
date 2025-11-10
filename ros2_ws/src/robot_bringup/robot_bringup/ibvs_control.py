#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
import math

# -------------------- Joint configurations --------------------
HOME_Q = np.array([0.0, -1.87, 1.57, -1.65, -2.30, 0.0])
VIEW_Q = np.array([0.0, -1.40, 1.47, -1.65, -1.87, 0.0])

# -------------------- UR5 DH parameters (meters) --------------------
DH_PARAMS = [
    (0,          0,          0.089159),
    (-0.42500,   math.pi/2,  0.0),
    (-0.39225,   0.0,        0.0),
    (0.0,        math.pi/2,  0.10915),
    (0.0,       -math.pi/2,  0.09465),
    (0.0,        0.0,        0.0823),
]

def near(q, qref, tol=1e-2):
    return float(np.linalg.norm(q - qref)) < tol


class UR5IBVSMotion(Node):
    def __init__(self):
        super().__init__('ur5_ibvs_motion')

        # Internal state
        self.q = HOME_Q.copy()
        self.v_cam = np.zeros(6)
        self.last_twist_time = None
        self.last_target_ok_time = None
        self.target_visible = False

        # Control settings
        self.dt = 0.03                 # 33 Hz
        self.timeout_s = 2.0           # hard timeout if no target_ok updates
        self.max_dq = 0.8              # rad/s
        self.max_twist_norm = 2.0      # safety guard
        self.lost_hysteresis_ticks = 20  # require N consecutive "lost" ticks
        self._lost_ticks = 0
        self._homing = False
        self._in_home = False

        # ROS I/O
        self.pub_traj = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )
        self.sub_twist = self.create_subscription(
            TwistStamped, '/ibvs/target_twist', self.on_ibvs_twist, qos_profile_sensor_data
        )
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.on_joint_state, qos_profile_sensor_data
        )
        self.sub_target_ok = self.create_subscription(
            Bool, '/ibvs/target_ok', self.on_target_status, qos_profile_sensor_data
        )

        # Timer
        self.timer = self.create_timer(self.dt, self.update_motion)

        self.get_logger().info("✅ UR5 IBVS Motion Node started.")
        self.move_to_pose(VIEW_Q, label="Camera View Pose")
        self._in_home = near(self.q, HOME_Q)

    # -------------------- Subscribers --------------------
    def on_ibvs_twist(self, msg: TwistStamped):
        self.v_cam = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
        ], dtype=np.float64)
        self.last_twist_time = self.get_clock().now()

    def on_joint_state(self, msg: JointState):
        try:
            joint_names = [
                'shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                'wrist_1_joint','wrist_2_joint','wrist_3_joint'
            ]
            idx = [msg.name.index(j) for j in joint_names]
            self.q = np.array([msg.position[i] for i in idx], dtype=np.float64)
        except Exception:
            pass

    def on_target_status(self, msg: Bool):
        self.target_visible = bool(msg.data)
        self.last_target_ok_time = self.get_clock().now()

    # -------------------- Control loop --------------------
    def update_motion(self):
        now = self.get_clock().now()

        # 1) LOST detection with hysteresis + hard timeout
        lost_by_timeout = (
            self.last_target_ok_time is None or
            (now - self.last_target_ok_time).nanoseconds * 1e-9 > self.timeout_s
        )
        if self.target_visible:
            self._lost_ticks = 0
        else:
            self._lost_ticks += 1

        really_lost = lost_by_timeout or (self._lost_ticks >= self.lost_hysteresis_ticks)

        # 2) If really lost → go HOME (but do it once, and only if not already home)
        if really_lost:
            if not self._homing and not near(self.q, HOME_Q, tol=5e-2):
                self._homing = True
                self.get_logger().warn("Target lost (debounced). Going to HOME once.")
                self.move_to_pose(HOME_Q, label="Home")
                self._in_home = True
                self._homing = False
            return

        # If target (re)appeared and we are at home, rise to VIEW once
        if self._in_home and self.target_visible:
            self.get_logger().info("Target visible again. Going to VIEW pose.")
            self.move_to_pose(VIEW_Q, label="Camera View Pose")
            self._in_home = False

        # 3) If no twist yet, do nothing
        if self.last_twist_time is None:
            return

        # 4) Resolved-rate control: dq = J^+ * v
        v = self.v_cam.copy()
        v_norm = float(np.linalg.norm(v))
        if v_norm > self.max_twist_norm:
            self.get_logger().warn_throttle(2.0, f"Twist too large: ||v||={v_norm:.3f}, clamping.")
            v *= (self.max_twist_norm / (v_norm + 1e-9))

        q = self.q.copy()
        J = self.ur5_jacobian(q)
        # Diagnostic: Jacobian condition
        try:
            s = np.linalg.svd(J, compute_uv=False)
            cond = float((s.max() / max(s.min(), 1e-9)))
            if cond > 1e5:
                self.get_logger().warn_throttle(2.0, f"Jacobian ill-conditioned: cond={cond:.2e}")
        except Exception:
            pass

        dq = np.linalg.pinv(J) @ v
        dq = np.clip(dq, -self.max_dq, self.max_dq)

        q_next = q + dq * self.dt
        self.publish_joint_trajectory(q_next)
        self.q = q_next

    # -------------------- Analytic UR5 Jacobian --------------------
    def ur5_jacobian(self, q):
        """Geometric Jacobian in base frame using DH_PARAMS."""
        a = [p[0] for p in DH_PARAMS]
        alpha = [p[1] for p in DH_PARAMS]
        d = [p[2] for p in DH_PARAMS]

        T = np.eye(4)
        z_axes = []
        origins = [np.zeros(3)]
        for i in range(6):
            ca, sa = math.cos(alpha[i]), math.sin(alpha[i])
            cq, sq = math.cos(q[i]), math.sin(q[i])
            A = np.array([
                [cq, -sq*ca,  sq*sa, a[i]*cq],
                [sq,  cq*ca, -cq*sa, a[i]*sq],
                [0.0,   sa,      ca,    d[i]],
                [0.0,  0.0,     0.0,    1.0]
            ], dtype=np.float64)
            T = T @ A
            z_axes.append(T[0:3, 2])
            origins.append(T[0:3, 3])

        o_n = origins[-1]
        J = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            z = z_axes[i]
            o = origins[i]
            J[0:3, i] = np.cross(z, (o_n - o))
            J[3:6, i] = z
        return J

    # -------------------- Trajectory Publisher --------------------
    def publish_joint_trajectory(self, q_next):
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
            'wrist_1_joint','wrist_2_joint','wrist_3_joint'
        ]
        pt = JointTrajectoryPoint()
        pt.positions = q_next.tolist()
        pt.time_from_start = Duration(sec=0, nanosec=int(self.dt * 1e9))
        traj.points.append(pt)
        self.pub_traj.publish(traj)

    # -------------------- Smooth Pose Transitions --------------------
    def move_to_pose(self, target_q, label="Pose"):
        self.get_logger().info(f"Moving to {label}...")
        current_q = self.q.copy()
        steps = 40
        for i in range(1, steps + 1):
            q_interp = current_q + (target_q - current_q) * (i / steps)
            traj = JointTrajectory()
            traj.joint_names = [
                'shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                'wrist_1_joint','wrist_2_joint','wrist_3_joint'
            ]
            pt = JointTrajectoryPoint()
            pt.positions = q_interp.tolist()
            pt.velocities = [0.0] * 6
            pt.time_from_start = Duration(sec=0, nanosec=int(0.1 * i * 1e9))
            traj.points.append(pt)
            self.pub_traj.publish(traj)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.q = target_q.copy()
        self.get_logger().info(f"✅ Reached {label}.")


def main(args=None):
    rclpy.init(args=args)
    node = UR5IBVSMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
