#!/usr/bin/env python3
"""
UR5 IBVS Controller - Eye-in-hand QR tracking (simplified & stable)

Assumptions:
- Eye-in-hand camera rigidly attached to the wrist.
- Vision node publishes a desired camera twist in camera_optical_frame.
- Conveyor runs roughly along +Y in base_link.
"""

import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray, Bool
from tf2_ros import Buffer, TransformListener
import tf_transformations as tft
from controller_manager_msgs.srv import SwitchController

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

Q_MIN = np.array([-6.28, -6.28, -6.28, -6.28, -6.28, -6.28])
Q_MAX = np.array([ 6.28,  6.28,  6.28,  6.28,  6.28,  6.28])
QD_MAX = np.array([1.8, 1.8, 1.8, 2.5, 2.5, 2.5])   # joint velocity limits

HOME_Q = np.array([0.0, -1.87, 1.57, -1.65, -2.0, 0.0])

CMD_HZ = 100.0       # controller frequency
CMD_DT = 1.0 / CMD_HZ

TWIST_TIMEOUT = 0.6  # if no twist for this long, start decaying / go home
TARGET_LOST_GRACE = 1.5

DEBUG = True
#CONVEYOR_VELOCITY = np.array([0.0, -0.05, 0.0, 0, 0, 0])
CONVEYOR_VELOCITY = np.array([0.0, -0.02, 0.0, 0, 0, 0])

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def adjoint_from_tf(transform):
    """6x6 adjoint matrix from geometry_msgs/TransformStamped."""
    t = transform.transform.translation
    q = transform.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [t.x, t.y, t.z]
    R = T[:3, :3]
    p = T[:3, 3]

    px = np.array([
        [0,     -p[2],  p[1]],
        [p[2],   0,    -p[0]],
        [-p[1],  p[0],  0  ],
    ])

    Ad = np.block([[R,           np.zeros((3, 3))],
                   [px @ R,      R]])
    return Ad


def ur5_jacobian_analytical(q):
    """Analytical geometric Jacobian for UR5 at joint positions q."""
    d     = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
    a     = np.array([0, -0.425, -0.39225, 0, 0, 0])
    alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

    q = np.array(q, dtype=np.float64)

    T = [np.eye(4)]
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])

        T_i = np.array([
            [ct, -st * ca,  st * sa, a[i] * ct],
            [st,  ct * ca, -ct * sa, a[i] * st],
            [0,   sa,       ca,      d[i]],
            [0,   0,        0,       1]
        ])
        T.append(T[-1] @ T_i)

    z_axes   = [T[i][:3, 2] for i in range(6)]
    pos      = [T[i][:3, 3] for i in range(7)]
    p_ee     = pos[6]

    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        J[:3, i] = np.cross(z_axes[i], p_ee - pos[i])
        J[3:, i] = z_axes[i]

    return J


def damped_pseudoinverse(J, lam=0.01):
    """Standard damped least-squares pseudoinverse."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))


# ---------------------------------------------------------------------------
# Controller node
# ---------------------------------------------------------------------------

class EyeInHandQRTrackingController(Node):
    def __init__(self):
        super().__init__("eye_in_hand_qr_tracking_controller")

        # Parameters
        self.declare_parameter("twist_topic", "/ibvs/target_twist")
        self.declare_parameter("target_ok_topic", "/ibvs/target_ok")
        self.declare_parameter("joint_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/forward_velocity_controller/commands")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_optical_frame")

        self.twist_topic   = self.get_parameter("twist_topic").value
        self.target_ok_topic = self.get_parameter("target_ok_topic").value
        self.joint_topic   = self.get_parameter("joint_topic").value
        self.cmd_topic     = self.get_parameter("cmd_topic").value
        self.base_frame    = self.get_parameter("base_frame").value
        self.camera_frame  = self.get_parameter("camera_frame").value

        # State
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_indices = None
        self.q = np.zeros(6)
        self.have_joints = False

        self.last_twist_time = 0.0
        self.tracking_locked = False
        self.mode = "initializing"  # "ready", "tracking", "returning_home"

        self.qd_cmd = np.zeros(6)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cached_adjoint = None

        # ROS interfaces
        self.create_subscription(JointState, self.joint_topic, self.on_joint_state, 10)
        self.create_subscription(TwistStamped, self.twist_topic, self.on_twist, 10)
        self.create_subscription(Bool, self.target_ok_topic, self.on_target_ok, 10)

        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.cmd_msg = Float64MultiArray()
        self.cmd_msg.data = [0.0] * 6
        self.v_cam_filtered = np.zeros(6)
        self.alpha = 0.3   # smoothing factor
        self.v_base_filtered = np.zeros(6)


        self.timer = self.create_timer(CMD_DT, self.control_loop)

        # Controller manager switch
        self.switch_to_velocity_controller()
        self.move_to_home()
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ Eye-in-hand QR tracking controller (simplified)")
        self.get_logger().info(f"  base_frame   : {self.base_frame}")
        self.get_logger().info(f"  camera_frame : {self.camera_frame}")
        self.get_logger().info("=" * 70)

    # ------------- callbacks -------------------------------------------------

    def switch_to_velocity_controller(self):
        client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Controller manager not available!")
            return

        req = SwitchController.Request()
        req.activate_controllers   = ["forward_velocity_controller"]
        req.deactivate_controllers = ["joint_trajectory_controller"]
        req.strictness = SwitchController.Request.BEST_EFFORT

        future = client.call_async(req)

        def _done(f):
            try:
                ok = f.result().ok
            except Exception as e:
                self.get_logger().error(f"Switch controller call failed: {e}")
                return
            if ok:
                self.get_logger().info("✅ forward_velocity_controller active")
            else:
                self.get_logger().warn("⚠️ Controller switch reported failure")

        future.add_done_callback(_done)

    def on_joint_state(self, msg: JointState):
        if self.joint_indices is None:
            name_to_index = {n: i for i, n in enumerate(msg.name)}
            try:
                self.joint_indices = [name_to_index[n] for n in self.joint_names]
            except KeyError:
                return

        self.q = np.array([msg.position[i] for i in self.joint_indices])

        if not self.have_joints:
            self.have_joints = True
            self.mode = "ready"
            self.get_logger().info("✅ Joint states ready, waiting for QR...")

    def on_target_ok(self, msg: Bool):
        # currently just tracked indirectly via twist / go_home
        pass

    def get_adjoint_camera_to_base(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            self.cached_adjoint = adjoint_from_tf(tf)
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"TF lookup failed: {e}")
        return self.cached_adjoint
    def get_adjoint_base_to_camera(self):
        """
        6x6 adjoint mapping twists from base frame to camera frame:
            v_cam = Ad_c_b @ v_base
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame,  # target
                self.base_frame,    # source
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            Ad_c_b = adjoint_from_tf(tf)
            return Ad_c_b
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"TF lookup (base->camera) failed: {e}")
            return None

    def on_twist_old(self, msg: TwistStamped):
        if not self.have_joints:
            return

        now = time.time()

        # "go_home" sentinel from vision node
        if msg.header.frame_id == "go_home":
            self.mode = "returning_home"
            self.tracking_locked = False
            if DEBUG:
                self.get_logger().info("📍 Target lost -> going home")
            return

        # camera twist (already IBVS-controlled, in camera_optical_frame)
       
        
        v_cam = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ], dtype=np.float64)

        Ad = self.get_adjoint_camera_to_base()
        if Ad is None:
            self.get_logger().error("No camera→base TF available")
            return
       
        v_base = Ad @ v_cam
        v_base[2] = np.clip(v_base[2], -0.05, 0.05)
        if DEBUG:
            self.get_logger().info(
                f"v_cam  lin[{v_cam[0]:+.3f},{v_cam[1]:+.3f},{v_cam[2]:+.3f}] "
                f"ang[{v_cam[3]:+.3f},{v_cam[4]:+.3f},{v_cam[5]:+.3f}]"
            )
            self.get_logger().info(
                f"v_base lin[{v_base[0]:+.3f},{v_base[1]:+.3f},{v_base[2]:+.3f}] "
                f"ang[{v_base[3]:+.3f},{v_base[4]:+.3f},{v_base[5]:+.3f}]"
            )

        # 3. If you want to reduce rotation *a bit*, scale instead of killing:
        v_base[3:6] *= 0.5  # keep orientation control, just weaker

        # 4. Very mild Z safety, but don't zero it out:
        v_base[2] = np.clip(v_base[2], -0.05, 0.05)

        # 5. IK: qdot = J^+ * v_base
        J = ur5_jacobian_analytical(self.q)
        sigma = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(sigma)
        lam = 0.02 if min_sv > 0.1 else 0.2    # stronger damping near singularities
        J_pinv = damped_pseudoinverse(J, lam)
        qd = J_pinv @ v_base

        # 6. Joint velocity limits
        # If you want softer behavior, limit by norm as well:
        qd = np.clip(qd, -QD_MAX, QD_MAX)
        max_norm = 1.5  # rad/s
        n = np.linalg.norm(qd)
        if n > max_norm:
            qd *= max_norm / (n + 1e-6)

        self.qd_cmd = qd
        self.last_twist_time = now
        self.tracking_locked = True
        self.mode = "tracking"

        if DEBUG:
            self.get_logger().info(f"qdot: {np.round(qd, 3)}")
    def on_twist(self, msg: TwistStamped):
        if not self.have_joints:
            return

        now = time.time()

        # Handle "go_home" sentinel
        if msg.header.frame_id == "go_home":
            self.mode = "returning_home"
            self.tracking_locked = False
            if DEBUG:
                self.get_logger().info("📍 Target lost -> going home")
            return

        # 1. Desired camera twist (in camera_optical_frame)
        v_cam = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ], dtype=np.float64)

        # 5. Compute base-frame Jacobian for UR5
        J_b = ur5_jacobian_analytical(self.q)  # 6x6, expressed in base_link

        Ad_c_b = self.get_adjoint_base_to_camera()
        if Ad_c_b is None:
            self.get_logger().error("No base→camera TF available")
            return

        J_cam = Ad_c_b @ J_b  # 6x6

        # 7. Damped least-squares pseudoinverse in the CAMERA frame
        sv = np.linalg.svd(J_cam, compute_uv=False)
        min_sv = float(np.min(sv))

        # Simple, robust damping schedule:
        if min_sv < 0.05:
            lam = 0.3
        elif min_sv < 0.1:
            lam = 0.12
        else:
            lam = 0.03

        J_cam_pinv = damped_pseudoinverse(J_cam, lam)

        # 8. Joint velocities directly from camera-frame IBVS
        qd = J_cam_pinv @ v_cam

        # 9. Joint limits + overall norm limit
        qd = np.clip(qd, -QD_MAX, QD_MAX)
        norm_qd = np.linalg.norm(qd)
        if norm_qd > 1.2:
            qd *= 1.2 / (norm_qd + 1e-6)

        self.qd_cmd = qd
        self.last_twist_time = now
        self.tracking_locked = True
        self.mode = "tracking"

        if DEBUG:
            self.get_logger().info(f"v_cam  = {np.round(v_cam, 3)}")
            self.get_logger().info(f"qdot   = {np.round(qd, 3)}")





    # ------------- higher-level behaviours ----------------------------------

    def move_to_home(self):
        err = HOME_Q - self.q
        err_norm = np.linalg.norm(err)

        if err_norm < 0.02:
            self.mode = "ready"
            self.tracking_locked = False
            if DEBUG:
                self.get_logger().info("✅ Reached home pose.")
            return np.zeros(6)

        qd = 1.2 * err
        qd = np.clip(qd, -QD_MAX * 0.6, QD_MAX * 0.6)
        return qd

    # ------------- control loop ---------------------------------------------
    def control_loop_new(self):
        if not self.have_joints:
            return

        now = time.time()
        dt = now - self.last_twist_time

        # CASE 1 — RETURNING HOME
        if self.mode == "returning_home":
            qd = self.move_to_home()
            qd = np.clip(qd, -QD_MAX, QD_MAX)
            self.cmd_msg.data = qd.tolist()
            self.pub_cmd.publish(self.cmd_msg)
            return

        # CASE 2 — ACTIVE TRACKING
        if self.mode == "tracking":

            if dt < 0.12:
                # fresh twist, use directly
                qd = self.qd_cmd.copy()

            elif dt < 0.30:
                # smooth decay
                decay = 1.0 - (dt / 0.30)
                qd = decay * self.qd_cmd

            elif dt < 0.40:
                # stop motion completely
                qd = np.zeros(6)

            else:
                # twist lost — go home
                self.mode = "returning_home"
                qd = self.move_to_home()
        elif self.mode == "ready" and dt > 0.4:
            self.mode = "returning_home"
            qd = self.move_to_home()
        # CASE 3 — READY (not tracking yet)
        else:
            qd = np.zeros(6)

        # FINAL LIMITS & SEND
        qd = np.clip(qd, -QD_MAX, QD_MAX)
        self.cmd_msg.data = qd.tolist()
        self.pub_cmd.publish(self.cmd_msg)


    def control_loop(self):
        if not self.have_joints:
            return

        now = time.time()
        dt_since_twist = now - self.last_twist_time
        if self.mode == "tracking":
            if dt_since_twist < TWIST_TIMEOUT:
                # use last commanded qdot
                qd = self.qd_cmd.copy()
            elif dt_since_twist < TARGET_LOST_GRACE:
                # no new twists, just stop where we are (hold pose)
                qd = np.zeros(6)
            else:
                # only after a long time with no target, go home
                self.mode = "returning_home"
                self.tracking_locked = False
                if DEBUG:
                    self.get_logger().info("⏰ Long time without twists -> going home")
                qd = self.move_to_home()


        elif self.mode == "returning_home":
            qd = self.move_to_home()
        else:  # "ready" or "initializing"
            qd = np.zeros(6)

        qd = np.clip(qd, -QD_MAX, QD_MAX)
        self.cmd_msg.data = qd.tolist()
        self.pub_cmd.publish(self.cmd_msg)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = EyeInHandQRTrackingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # send stop command a few times
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 6
        for _ in range(20):
            node.pub_cmd.publish(stop_msg)
            time.sleep(0.01)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
