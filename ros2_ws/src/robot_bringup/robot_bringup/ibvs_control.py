#!/usr/bin/env python3
"""
UR5 IBVS Controller - Eye-in-hand QR tracking on CONVEYOR
Optimized for tracking items moving on conveyor belt (positive Y direction)
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
# PARAMETERS - TUNED FOR CONVEYOR TRACKING
# ---------------------------------------------------------------------------
Q_MIN = np.array([-6.28, -6.28, -6.28, -6.28, -6.28, -6.28])
#Q_MAX = np.array([ 6.28,  6.28,  6.28,  6.28,  6.28,  6.28])
QD_MAX = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
#QD_MAX = np.array([2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
# QD_MAX = np.array([3, 3, 3, 4, 4, 4])
MAX_JOINT_VEL_NORM = 1.0

HOME_Q = np.array([0.0, -1.87, 1.57, -1.65, -2.0, 0.0])

CMD_HZ = 200.0
CMD_DT = 1.0 / CMD_HZ
TWIST_TIMEOUT = 0.6
TARGET_LOST_GRACE = 1.8

HOLD_KP = 0.6        # small posture stiffness
HOLD_QD_MAX = 0.15  # very small velocities

# CONVEYOR TRACKING - Key to success!
CONVEYOR_ENABLED = True
# Conveyor moves at ~0.02 m/s in +Y (base frame) based on your code
CONVEYOR_VELOCITY_BASE = np.array([0.0, -0.16, 0.0, 0.0, 0.0, 0.0])

# IBVS GAINS - Critical tuning
# The vision node already applies LAMBDA=1.0, so we don't scale again
VELOCITY_SCALE = 0.5  # Direct pass-through of vision velocities

# Filtering for smooth tracking
FILTER_ALPHA = 0.8  # Balance between responsiveness and smoothness

# Damping for singularity handling
DAMPING_NORMAL = 0.05
DAMPING_NEAR_SINGULARITY = 0.2

# Safety limits
MAX_LIN_VEL = 0.80  # m/s
MAX_ANG_VEL = 0.60  # rad/s
#MAX_JOINT_VEL_NORM = 1.5  # rad/s

DEBUG = False  # Set to True for detailed logging

# ---------------------------------------------------------------------------
# Helper Functions
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
    d    = [0.089159,  0.00000,  0.00000,  0.10915,  0.09465,  0.0903 + 0.1628]
    a    = [0.00000,  -0.42500, -0.39225,  0.00000,  0.00000,  0.0000]
    alpha = [np.pi/2,      0,        0,        np.pi/2,     -np.pi/2,    0]

    # d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
    # a = np.array([0, -0.425, -0.39225, 0, 0, 0])
    # alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])

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

    z_axes = [T[i][:3, 2] for i in range(6)]
    pos = [T[i][:3, 3] for i in range(7)]
    p_ee = pos[6]

    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        J[:3, i] = np.cross(z_axes[i], p_ee - pos[i])
        J[3:, i] = z_axes[i]

    return J

def damped_pseudoinverse(J, lam=0.01):
    """Damped least-squares pseudoinverse."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))

def check_singularity(J):
    """Check for singularities and return appropriate damping."""
    sigma = np.linalg.svd(J, compute_uv=False)
    min_sv = np.min(sigma)
    max_sv = np.max(sigma)
    condition = max_sv / (min_sv + 1e-10)
    
    if min_sv < 0.05 or condition > 500:
        return DAMPING_NEAR_SINGULARITY, "NEAR_SING", min_sv
    else:
        return DAMPING_NORMAL, "OK", min_sv

# ---------------------------------------------------------------------------
# Controller Node
# ---------------------------------------------------------------------------
class ConveyorTrackingController(Node):
    def __init__(self):
        super().__init__("conveyor_tracking_controller")

        # Parameters
        self.declare_parameter("twist_topic", "/ibvs/target_twist")
        self.declare_parameter("target_ok_topic", "/ibvs/target_ok")
        self.declare_parameter("joint_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/forward_velocity_controller/commands")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_optical_frame")

        self.twist_topic = self.get_parameter("twist_topic").value
        self.target_ok_topic = self.get_parameter("target_ok_topic").value
        self.joint_topic = self.get_parameter("joint_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        # State
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_indices = None
        self.q = np.zeros(6)
        self.have_joints = False
        self.last_twist_time = 0.0
        self.mode = "initializing"
        self.target_visible = False
        self.qd_cmd = np.zeros(6)
        
        # Filtered velocities for smooth tracking
        self.v_cam_filtered = np.zeros(6)
        self.v_base_filtered = np.zeros(6)
        
        # Statistics
        self.iteration_count = 0
        self.last_log_time = time.time()
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cached_adjoint = None
        
        self.ibvs_disabled = False
        self.create_subscription(Bool, "/ibvs/disable_control", self.on_disable_ibvs, 10)

        # ROS interfaces
        self.create_subscription(JointState, self.joint_topic, self.on_joint_state, 10)
        self.create_subscription(TwistStamped, self.twist_topic, self.on_twist, 10)
        self.create_subscription(Bool, self.target_ok_topic, self.on_target_ok, 10)
        
        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.cmd_msg = Float64MultiArray()
        self.cmd_msg.data = [0.0] * 6
        
        self.timer = self.create_timer(CMD_DT, self.control_loop)
        
        # Switch controller
        self.switch_to_velocity_controller()
        
        self.get_logger().info("=" * 70)
        self.get_logger().info("✅ CONVEYOR TRACKING IBVS Controller")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Conveyor velocity: +Y @ {CONVEYOR_VELOCITY_BASE[1]:.3f} m/s")
        self.get_logger().info(f"  Filter alpha: {FILTER_ALPHA}")
        self.get_logger().info(f"  Max velocities: lin={MAX_LIN_VEL}, ang={MAX_ANG_VEL}")
        self.get_logger().info("=" * 70)

    def switch_to_velocity_controller(self):
        """Switch to velocity controller."""
        client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Controller manager not available!")
            return

        req = SwitchController.Request()
        req.activate_controllers = ["forward_velocity_controller"]
        req.deactivate_controllers = ["joint_trajectory_controller"]
        req.strictness = SwitchController.Request.BEST_EFFORT

        future = client.call_async(req)
        
        def _done(f):
            try:
                ok = f.result().ok
                if ok:
                    self.get_logger().info("✅ Velocity controller active")
                    self.mode = "returning_home"
                else:
                    self.get_logger().warn("⚠️ Controller switch failed")
            except Exception as e:
                self.get_logger().error(f"Switch error: {e}")
        
        future.add_done_callback(_done)

    def on_disable_ibvs(self,msg):
        self.mode = "picking"
        self.ibvs_disabled=msg.data
        
        
    def on_joint_state(self, msg: JointState):
        """Update joint states."""
        if self.joint_indices is None:
            name_to_index = {n: i for i, n in enumerate(msg.name)}
            try:
                self.joint_indices = [name_to_index[n] for n in self.joint_names]
            except KeyError as e:
                self.get_logger().error(f"Joint not found: {e}")
                return

        self.q = np.array([msg.position[i] for i in self.joint_indices])
        
        if not self.have_joints:
            self.have_joints = True
            self.mode = "returning_home"
            self.get_logger().info("✅ Joint states received. Moving to home...")

    def on_target_ok(self, msg: Bool):
        """Update target visibility."""
        self.target_visible = msg.data
    def hold_posture(self):
        err = HOME_Q - self.q
        qd = HOLD_KP * err
        qd = np.clip(qd, -HOLD_QD_MAX, HOLD_QD_MAX)
        return qd

    def get_point_offset_matrix(self, ee_frame="tool0"):
        """
        Build A such that v_c^b = A · v_e^b, relating twists at ee->camera
        in the *same* base frame.
        """
        try:
            # T_base_ee
            tf_be = self.tf_buffer.lookup_transform(
                self.base_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # T_base_cam
            tf_bc = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            def tf_to_pos(tf_msg):
                t = tf_msg.transform.translation
                return np.array([t.x, t.y, t.z], dtype=np.float64)

            p_e = tf_to_pos(tf_be)  # ee position in base
            p_c = tf_to_pos(tf_bc)  # cam position in base
            p_ec = p_c - p_e        # from ee to cam in base

            px = np.array([
                [0,      -p_ec[2],  p_ec[1]],
                [p_ec[2], 0,       -p_ec[0]],
                [-p_ec[1], p_ec[0], 0],
            ], dtype=np.float64)

            # A = [[I, 0], [px, I]]
            A = np.block([
                [np.eye(3), np.zeros((3, 3))],
                [px,        np.eye(3)]
            ])
            return A

        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"TF lookup (point offset) failed: {e}")
            return None



    def get_adjoint_camera_to_base(self):
        """Get transform from camera to base frame."""       
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.cached_adjoint = adjoint_from_tf(tf)
            return self.cached_adjoint
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return None

    def on_twist(self, msg: TwistStamped):
        """Process camera twist from vision node."""
        if not self.have_joints:
            return

        now = time.time()
        
        # Handle go_home signal
        if msg.header.frame_id == "go_home":
            self.mode = "returning_home"
            if DEBUG:
                self.get_logger().info("📍 Target lost -> going home")
            return
        
        # Extract camera velocity from vision node
        v_cam_raw = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ], dtype=np.float64)
        
        # Apply filtering
        self.v_cam_filtered = (
            FILTER_ALPHA * v_cam_raw + 
            (1 - FILTER_ALPHA) * self.v_cam_filtered
        )
        
        v_cam = self.v_cam_filtered.copy()
        
        # Scale angular components
        LIN_GAIN = 1.0
        ANG_GAIN = 0.3
        v_cam[:3] *= LIN_GAIN
        v_cam[3:] *= ANG_GAIN
        v_cam *= VELOCITY_SCALE
        # After v_cam is computed
        if abs(v_cam[2]) < 0.052:
            v_cam[2] = 0.0

        
        # Safety limits in camera frame
        lin_norm = np.linalg.norm(v_cam[:3])
        if lin_norm > MAX_LIN_VEL:
            v_cam[:3] *= MAX_LIN_VEL / lin_norm
        
        ang_norm = np.linalg.norm(v_cam[3:])
        if ang_norm > MAX_ANG_VEL:
            v_cam[3:] *= MAX_ANG_VEL / ang_norm
        
        # Get transform from camera to base
        Ad_cam_to_base = self.get_adjoint_camera_to_base()
        if Ad_cam_to_base is None:
            self.get_logger().error("No camera→base TF available")
            return
        
        # Transform camera twist to base frame
        v_cam_in_base = Ad_cam_to_base @ v_cam
        
        # Get point offset matrix (EE to camera)
        A_ee_to_cam = self.get_point_offset_matrix(ee_frame="tool0")
        if A_ee_to_cam is None:
            self.get_logger().error("No camera→ee offset available")
            return
        
        # Calculate A_inv (from camera to EE)
        # Extract p_ec from A_ee_to_cam
        # A_ee_to_cam = [[I, 0], [px, I]]
        px = A_ee_to_cam[3:, :3]  # Get the px matrix from A
        
        # Build A_inv = [[I, 0], [-px, I]]
        A_inv = np.block([[np.eye(3), np.zeros((3, 3))],
                        [-px,       np.eye(3)]])
        
        # Convert camera velocity to EE velocity in base frame
        v_ee_in_base = A_inv @ v_cam_in_base
        
        # Add conveyor feedforward velocity to EE
        if CONVEYOR_ENABLED:
            v_ee_in_base += CONVEYOR_VELOCITY_BASE
        
        # Apply Z safety constraint
        try:
            tf_ee = self.tf_buffer.lookup_transform(
                self.base_frame,
                "tool0",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            ee_z_position = tf_ee.transform.translation.z
            
            # FIX THIS VALUE! Should be based on your actual setup
            MIN_SAFE_Z = 0.05  # 5cm above conveyor surface
            
            if ee_z_position < MIN_SAFE_Z:
                if v_ee_in_base[2] < 0:  # Prevent downward movement
                    v_ee_in_base[2] = 0.0
                    if DEBUG:
                        self.get_logger().info(f"⚠️ Z constraint: z={ee_z_position:.3f}m")
                        
        except Exception as e:
            if DEBUG:
                self.get_logger().warn(f"Could not check Z position: {e}")
        
        # Filter EE velocity
        self.v_base_filtered = 0.3 * v_ee_in_base + 0.7 * self.v_base_filtered
        v_ee_filtered = self.v_base_filtered.copy()
        
        # Safety limits for EE velocity
        lin_norm = np.linalg.norm(v_ee_filtered[:3])
        if lin_norm > MAX_LIN_VEL * 1.2:
            v_ee_filtered[:3] *= (MAX_LIN_VEL * 1.2) / lin_norm
        
        ang_norm = np.linalg.norm(v_ee_filtered[3:])
        if ang_norm > MAX_ANG_VEL * 1.2:
            v_ee_filtered[3:] *= (MAX_ANG_VEL * 1.2) / ang_norm
        
        # Get EE Jacobian (at tool0)
        J_ee = ur5_jacobian_analytical(self.q)
        
        # Check for singularities
        damping, sing_status, min_sv = check_singularity(J_ee)
        
        # Damped pseudoinverse of EE Jacobian
        J_ee_pinv = damped_pseudoinverse(J_ee, damping)
        
        # Compute joint velocities: qd = J_ee^+ · v_ee_filtered
        qd = J_ee_pinv @ v_ee_filtered
        
        # Apply joint velocity limits
        qd = np.clip(qd, -QD_MAX, QD_MAX)
        
        # Limit overall joint velocity magnitude
        qd_norm = np.linalg.norm(qd)
        if qd_norm > MAX_JOINT_VEL_NORM:
            qd *= MAX_JOINT_VEL_NORM / qd_norm
        
        # Store for control loop
        self.qd_cmd = qd
        self.last_twist_time = now
        self.mode = "tracking"
        
        # Logging
        self.iteration_count += 1
        if DEBUG and self.iteration_count % 50 == 0:
            self.get_logger().info(
                f"🎯 v_cam[{v_cam[0]:+.3f},{v_cam[1]:+.3f},{v_cam[2]:+.3f}] "
                f"v_ee[{v_ee_filtered[0]:+.3f},{v_ee_filtered[1]:+.3f},{v_ee_filtered[2]:+.3f}] "
                f"|qd|={qd_norm:.2f} {sing_status}"
            )
        
        # Periodic status
        if time.time() - self.last_log_time > 2.0:
            self.last_log_time = time.time()
            self.get_logger().info(
                f"📊 Tracking: v_ee_Y={v_ee_filtered[1]:+.3f} (conveyor+tracking), "
                f"|qd|={qd_norm:.2f}, {sing_status}"
            )

    def move_to_home(self):
            """Compute velocity to return to home position."""
            err = HOME_Q - self.q
            err_norm = np.linalg.norm(err)
            
            if err_norm < 0.05:
                self.mode = "ready"
                self.get_logger().info("✅ At home pose. Ready for tracking.")
                return np.zeros(6)
            
            # Proportional control with limits
            Kp = 3.0
            qd_home = Kp * err
            qd_home = np.clip(qd_home, -QD_MAX * 0.6, QD_MAX * 0.6)
            
            return qd_home

    def control_loop(self):
        """Main control loop at 100 Hz."""
        if not self.have_joints:
            return
        if self.ibvs_disabled:
             qd = self.hold_posture()
            
        now = time.time()
        dt_since_twist = now - self.last_twist_time
        
        # State machine
        if self.mode == "tracking":
            if dt_since_twist < TWIST_TIMEOUT:
                # Use last commanded velocity
                qd = self.qd_cmd.copy()
            elif dt_since_twist < TARGET_LOST_GRACE:
                # Gracefully decay velocity
                decay = max(0.0, 1.0 - (dt_since_twist - TWIST_TIMEOUT) / 
                          (TARGET_LOST_GRACE - TWIST_TIMEOUT))
                qd = self.qd_cmd * decay + self.hold_posture() * (1.0 - decay)
            else:
                # Target lost for too long, go home
                self.mode = "returning_home"
                self.get_logger().info("⏰ Target lost. Returning home.")
                qd = self.move_to_home()
                
        elif self.mode == "returning_home":
            qd = self.move_to_home()
            
        elif self.mode == "ready":
            qd = self.hold_posture()
            if self.target_visible and dt_since_twist < TWIST_TIMEOUT:
                self.mode = "tracking"
                self.get_logger().info("🎯 Target acquired. Starting tracking...")
                
        else:  # initializing
            qd = np.zeros(6)
        
        # Final safety
        qd = np.clip(qd, -QD_MAX, QD_MAX)
        
        # Publish command
        self.cmd_msg.data = qd.tolist()
        self.pub_cmd.publish(self.cmd_msg)

# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ConveyorTrackingController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down conveyor tracking controller...")
    finally:
        # Send multiple stop commands
        stop_msg = Float64MultiArray()
        stop_msg.data = [0.0] * 6
        for _ in range(30):
            node.pub_cmd.publish(stop_msg)
            time.sleep(0.01)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()