#!/usr/bin/env python3
"""
Velocity-Only Pick-Place Executor
Uses ONLY velocity control for ALL motions - no controller switching needed!
Simple, reliable, and accurate.
"""

import rclpy
from rclpy.node import Node
from collections import deque
from enum import Enum, auto
import numpy as np
from std_msgs.msg import Bool, Float64MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robot_bringup_interfaces.msg import IBVSState

class PPState(Enum):
    TRACKING = auto()          # IBVS enabled, tracking target
    DISABLE_IBVS = auto()      # Disable IBVS before pick
    APPROACH = auto()          # Slow approach to object
    PAUSE_BEFORE_GRIP = auto() # Brief pause before gripping
    GRIP_CLOSE = auto()        # Close gripper
    LIFT = auto()              # Lift object
    MOVE_TO_DROP = auto()      # Move to drop position
    PAUSE_BEFORE_RELEASE = auto() # Brief pause before release
    RELEASE = auto()           # Open gripper
    RETURN_HOME = auto()       # Return to home position
    ENABLE_IBVS = auto()       # Re-enable IBVS
    COOLDOWN = auto()          # Wait before next pick
    ERROR = auto()             # Error recovery state


class VelocityOnlyPickPlace(Node):
    def __init__(self):
        super().__init__("velocity_pickplace")
        
        # ------------------------
        # Parameters - TUNE THESE!
        # ------------------------
        # Topics
        self.declare_parameter("ibvs_state_topic", "/ibvs/state")
        self.declare_parameter("disable_ibvs_topic", "/ibvs/disable_control")
        self.declare_parameter("velocity_cmd_topic", "/forward_velocity_controller/commands")
        self.declare_parameter("gripper_cmd_topic", "/robotiq_2f85_controller/joint_trajectory")
        self.declare_parameter("joint_states_topic", "/joint_states")
        
        # Gripper
        self.declare_parameter("gripper_joint_name", "robotiq_85_left_knuckle_joint")
        self.declare_parameter("gripper_open_pos", 0.8)    # Fully open
        self.declare_parameter("gripper_close_pos", 0.0)   # Fully closed
        self.declare_parameter("gripper_move_time", 1.5)   # Slow for reliability
        
        # Pick detection (LOOSE for testing)
        self.declare_parameter("pixel_error_trigger", 25.0)     # px
        self.declare_parameter("depth_trigger", 0.35)           # meters
        self.declare_parameter("stable_window_sec", 0.8)        # Must be stable this long
        self.declare_parameter("stable_min_samples", 10)        # Samples in window
        self.declare_parameter("error_std_max", 10.0)           # Max std dev
        
        # Velocity control parameters (CRITICAL - go SLOW!)
        self.declare_parameter("approach_velocity", [0.0, 0.0, -0.05, 0.0, 0.0, 0.0])  # Move down (m/s, rad/s)
        self.declare_parameter("approach_duration", 1.5)        # seconds
        self.declare_parameter("lift_velocity", [0.0, 0.0, 0.08, 0.0, 0.0, 0.0])       # Lift up
        self.declare_parameter("lift_duration", 2.0)
        
        # Drop motion - rotate shoulder pan joint SLOWLY
        self.declare_parameter("rotate_velocity", [0.15, 0.0, 0.0, 0.0, 0.0, 0.0])     # Rotate right
        self.declare_parameter("rotate_duration", 4.0)
        self.declare_parameter("rotate_back_velocity", [-0.15, 0.0, 0.0, 0.0, 0.0, 0.0]) # Rotate back
        self.declare_parameter("rotate_back_duration", 4.0)
        
        # Return home - move shoulder and elbow joints
        self.declare_parameter("return_home_velocity", [0.0, 0.1, -0.08, 0.0, 0.0, 0.0]) # Adjust to your home
        self.declare_parameter("return_home_duration", 3.0)
        
        # Safety
        self.declare_parameter("max_pick_attempts", 3)
        self.declare_parameter("cooldown_sec", 2.0)
        self.declare_parameter("pause_before_grip", 0.3)        # Pause before gripping
        self.declare_parameter("pause_before_release", 0.3)     # Pause before releasing
        
        # Joint limits for safety
        self.declare_parameter("joint_vel_limits", [3.0, 3.0, 3.0, 4.0, 4.0, 4.0])
        self.declare_parameter("joint_pos_limits_min", [-6.28, -6.28, -6.28, -6.28, -6.28, -6.28])
        self.declare_parameter("joint_pos_limits_max", [6.28, 6.28, 6.28, 6.28, 6.28, 6.28])
        
        # State machine
        self.declare_parameter("sm_hz", 50.0)
        
        # Read parameters
        self._read_parameters()
        
        # ------------------------
        # Publishers / Subscribers
        # ------------------------
        self.pub_disable_ibvs = self.create_publisher(Bool, self.disable_ibvs_topic, 10)
        self.pub_velocity = self.create_publisher(Float64MultiArray, self.velocity_cmd_topic, 10)
        self.pub_gripper = self.create_publisher(JointTrajectory, self.gripper_cmd_topic, 10)
        
        self.create_subscription(IBVSState, self.ibvs_state_topic, self.on_ibvs_state, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.on_joint_state, 10)
        
        # ------------------------
        # Internal State
        # ------------------------
        self.state = PPState.TRACKING
        self.pick_attempts = 0
        self.state_start_time = 0.0
        self.current_velocity = np.zeros(6, dtype=np.float64)
        
        # Data buffers
        self.latest_ibvs = None
        self.ibvs_buffer = deque(maxlen=200)
        self.q = np.zeros(6, dtype=np.float64)
        self.have_joints = False
        self.joint_index = None
        
        # Motion timing
        self.motion_start_time = 0.0
        self.motion_duration = 0.0
        self.motion_velocity = np.zeros(6, dtype=np.float64)
        
        # State tracking
        self.last_state = None
        self.cooldown_end_time = 0.0
        self.q_at_pick = None  # Store joint positions at pick time
        
        # Joint names
        self.arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        # State machine timer
        self.sm_timer = self.create_timer(self.sm_dt, self.step)
        
        self.get_logger().info("="*60)
        self.get_logger().info("✅ VELOCITY-ONLY PICK-PLACE EXECUTOR")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Pick triggers: error<{self.pixel_error_trigger}px, depth<{self.depth_trigger}m")
        self.get_logger().info(f"Approach: {self.approach_velocity} for {self.approach_duration}s")
        self.get_logger().info(f"Lift: {self.lift_velocity} for {self.lift_duration}s")
        self.get_logger().info("="*60)
    
    def _read_parameters(self):
        """Read all parameters."""
        # Topics
        self.ibvs_state_topic = self.get_parameter("ibvs_state_topic").value
        self.disable_ibvs_topic = self.get_parameter("disable_ibvs_topic").value
        self.velocity_cmd_topic = self.get_parameter("velocity_cmd_topic").value
        self.gripper_cmd_topic = self.get_parameter("gripper_cmd_topic").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        
        # Gripper
        self.gripper_joint_name = self.get_parameter("gripper_joint_name").value
        self.gripper_open_pos = float(self.get_parameter("gripper_open_pos").value)
        self.gripper_close_pos = float(self.get_parameter("gripper_close_pos").value)
        self.gripper_move_time = float(self.get_parameter("gripper_move_time").value)
        
        # Pick detection
        self.pixel_error_trigger = float(self.get_parameter("pixel_error_trigger").value)
        self.depth_trigger = float(self.get_parameter("depth_trigger").value)
        self.stable_window_sec = float(self.get_parameter("stable_window_sec").value)
        self.stable_min_samples = int(self.get_parameter("stable_min_samples").value)
        self.error_std_max = float(self.get_parameter("error_std_max").value)
        
        # Velocities
        self.approach_velocity = np.array(self.get_parameter("approach_velocity").value, dtype=np.float64)
        self.approach_duration = float(self.get_parameter("approach_duration").value)
        self.lift_velocity = np.array(self.get_parameter("lift_velocity").value, dtype=np.float64)
        self.lift_duration = float(self.get_parameter("lift_duration").value)
        self.rotate_velocity = np.array(self.get_parameter("rotate_velocity").value, dtype=np.float64)
        self.rotate_duration = float(self.get_parameter("rotate_duration").value)
        self.rotate_back_velocity = np.array(self.get_parameter("rotate_back_velocity").value, dtype=np.float64)
        self.rotate_back_duration = float(self.get_parameter("rotate_back_duration").value)
        self.return_home_velocity = np.array(self.get_parameter("return_home_velocity").value, dtype=np.float64)
        self.return_home_duration = float(self.get_parameter("return_home_duration").value)
        
        # Safety
        self.max_pick_attempts = int(self.get_parameter("max_pick_attempts").value)
        self.cooldown_sec = float(self.get_parameter("cooldown_sec").value)
        self.pause_before_grip = float(self.get_parameter("pause_before_grip").value)
        self.pause_before_release = float(self.get_parameter("pause_before_release").value)
        
        # Joint limits
        self.joint_vel_limits = np.array(self.get_parameter("joint_vel_limits").value, dtype=np.float64)
        self.joint_pos_limits_min = np.array(self.get_parameter("joint_pos_limits_min").value, dtype=np.float64)
        self.joint_pos_limits_max = np.array(self.get_parameter("joint_pos_limits_max").value, dtype=np.float64)
        
        # State machine
        sm_hz = float(self.get_parameter("sm_hz").value)
        self.sm_dt = 1.0 / max(1.0, sm_hz)
    
    def ros_now_sec(self) -> float:
        """Get current ROS time in seconds."""
        return self.get_clock().now().nanoseconds * 1e-9
    
    # ------------------------
    # Callbacks
    # ------------------------
    def on_joint_state(self, msg: JointState):
        """Update joint positions."""
        if self.joint_index is None:
            name_to_i = {n: i for i, n in enumerate(msg.name)}
            try:
                self.joint_index = [name_to_i[n] for n in self.arm_joint_names]
            except KeyError as e:
                self.get_logger().error(f"Joint missing: {e}")
                return
        
        self.q = np.array([msg.position[i] for i in self.joint_index], dtype=np.float64)
        self.have_joints = True
    
    def on_ibvs_state(self, msg: IBVSState):
        """Update IBVS state."""
        self.latest_ibvs = msg
        
        t = self.ros_now_sec()
        self.ibvs_buffer.append({
            "t": t,
            "visible": bool(msg.target_visible),
            "centered": bool(msg.centered),
            "close": bool(msg.close),
            "err": float(msg.pixel_error),
            "depth": float(msg.depth),
        })
    
    # ------------------------
    # Stability Detection
    # ------------------------
    def is_stable_for_pick(self) -> bool:
        """
        Check if target is stable enough for picking.
        Returns True if:
        1. Target visible
        2. Within error and depth thresholds
        3. Stable for required time window
        """
        if self.latest_ibvs is None:
            return False
        
        now = self.ros_now_sec()
        
        # Basic checks
        if not self.latest_ibvs.target_visible:
            return False
        
        # Threshold checks
        error_ok = self.latest_ibvs.pixel_error < self.pixel_error_trigger
        depth_ok = self.latest_ibvs.depth < self.depth_trigger
        
        if not (error_ok and depth_ok):
            return False
        
        # Check stability window
        recent = [s for s in self.ibvs_buffer 
                 if (now - s["t"]) <= self.stable_window_sec]
        
        if len(recent) < self.stable_min_samples:
            return False
        
        # All recent samples must be good
        if not all(s["visible"] and (s["err"] < self.pixel_error_trigger) 
                   and (s["depth"] < self.depth_trigger) for s in recent):
            return False
        
        # Check error stability (low variance)
        errs = np.array([s["err"] for s in recent], dtype=np.float64)
        if float(np.std(errs)) > self.error_std_max:
            return False
        
        return True
    
    # ------------------------
    # Motion Control
    # ------------------------
    def send_velocity_command(self, velocity: np.ndarray):
        """Send velocity command to controller with safety checks."""
        # Apply joint velocity limits
        velocity = np.clip(velocity, -self.joint_vel_limits, self.joint_vel_limits)
        
        # Check for singularities (very simple check)
        # You might want to add more sophisticated singularity avoidance
        
        # Publish command
        msg = Float64MultiArray()
        msg.data = velocity.tolist()
        self.pub_velocity.publish(msg)
        
        # Store current velocity
        self.current_velocity = velocity.copy()
    
    def send_stop_command(self):
        """Send zero velocity command."""
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.pub_velocity.publish(msg)
        self.current_velocity = np.zeros(6)
    
    def send_gripper_command(self, position: float):
        """Send gripper command."""
        traj = JointTrajectory()
        traj.joint_names = [self.gripper_joint_name]
        
        pt = JointTrajectoryPoint()
        pt.positions = [position]
        pt.time_from_start.sec = int(self.gripper_move_time)
        pt.time_from_start.nanosec = int((self.gripper_move_time % 1.0) * 1e9)
        
        traj.points = [pt]
        self.pub_gripper.publish(traj)
        
        action = "CLOSING" if position < 0.1 else "OPENING"
        self.get_logger().info(f"🤏 Gripper {action} to position {position}")
    
    def start_motion(self, velocity: np.ndarray, duration: float):
        """Start a timed motion."""
        self.motion_start_time = self.ros_now_sec()
        self.motion_duration = duration
        self.motion_velocity = velocity.copy()
        
        # Send initial velocity
        self.send_velocity_command(velocity)
        
        self.get_logger().info(f"▶️ Starting motion: vel={velocity}, duration={duration:.1f}s")
    
    def is_motion_complete(self) -> bool:
        """Check if current motion is complete."""
        elapsed = self.ros_now_sec() - self.motion_start_time
        return elapsed >= self.motion_duration
    
    def update_motion_velocity(self):
        """Update velocity command during motion (for smooth ramping)."""
        if self.motion_duration <= 0:
            return
        
        elapsed = self.ros_now_sec() - self.motion_start_time
        progress = elapsed / self.motion_duration
        
        # Simple trapezoidal profile: ramp up, constant, ramp down
        if progress < 0.1:  # Ramp up
            scale = progress / 0.1
        elif progress > 0.9:  # Ramp down
            scale = (1.0 - progress) / 0.1
        else:  # Constant
            scale = 1.0
        
        scaled_velocity = self.motion_velocity * scale
        self.send_velocity_command(scaled_velocity)
    
    # ------------------------
    # State Machine
    # ------------------------
    def step(self):
        """Main state machine step."""
        if not self.have_joints:
            return
        
        now = self.ros_now_sec()
        
        # Log state transitions
        if self.last_state != self.state:
            self.get_logger().info(f"STATE: {self.state.name}")
            self.last_state = self.state
            self.state_start_time = now
        
        # Execute current state
        if self.state == PPState.TRACKING:
            self._state_tracking(now)
        
        elif self.state == PPState.DISABLE_IBVS:
            self._state_disable_ibvs(now)
        
        elif self.state == PPState.APPROACH:
            self._state_approach(now)
        
        elif self.state == PPState.PAUSE_BEFORE_GRIP:
            self._state_pause_before_grip(now)
        
        elif self.state == PPState.GRIP_CLOSE:
            self._state_grip_close(now)
        
        elif self.state == PPState.LIFT:
            self._state_lift(now)
        
        elif self.state == PPState.MOVE_TO_DROP:
            self._state_move_to_drop(now)
        
        elif self.state == PPState.PAUSE_BEFORE_RELEASE:
            self._state_pause_before_release(now)
        
        elif self.state == PPState.RELEASE:
            self._state_release(now)
        
        elif self.state == PPState.RETURN_HOME:
            self._state_return_home(now)
        
        elif self.state == PPState.ENABLE_IBVS:
            self._state_enable_ibvs(now)
        
        elif self.state == PPState.COOLDOWN:
            self._state_cooldown(now)
        
        elif self.state == PPState.ERROR:
            self._state_error(now)
    
    def _state_tracking(self, now):
        """TRACKING: IBVS enabled, looking for pick opportunity."""
        # Ensure IBVS is enabled
        self.pub_disable_ibvs.publish(Bool(data=False))
        
        # Check cooldown
        if now < self.cooldown_end_time:
            return
        
        # Check pick attempts
        if self.pick_attempts >= self.max_pick_attempts:
            self.get_logger().error("❌ Max pick attempts reached")
            return
        
        # Check stability
        if self.is_stable_for_pick():
            self.get_logger().info(
                f"🎯 PICK TRIGGERED! Attempt {self.pick_attempts + 1}/{self.max_pick_attempts}\n"
                f"   Error: {self.latest_ibvs.pixel_error:.1f}px (threshold: {self.pixel_error_trigger}px)\n"
                f"   Depth: {self.latest_ibvs.depth:.3f}m (threshold: {self.depth_trigger}m)"
            )
            
            # Store current joint positions
            self.q_at_pick = self.q.copy()
            
            # Move to next state
            self.state = PPState.DISABLE_IBVS
    
    def _state_disable_ibvs(self, now):
        """DISABLE_IBVS: Stop IBVS and ensure robot is stopped."""
        # Disable IBVS
        self.pub_disable_ibvs.publish(Bool(data=True))
        
        # Send stop command to ensure no residual motion
        self.send_stop_command()
        
        # Wait briefly for everything to settle
        if (now - self.state_start_time) > 0.5:
            self.state = PPState.APPROACH
    
    def _state_approach(self, now):
        """APPROACH: Move slowly toward object."""
        if self.state_start_time == now:  # First time in this state
            self.start_motion(self.approach_velocity, self.approach_duration)
        
        # Update velocity (for smooth ramping)
        self.update_motion_velocity()
        
        # Check if motion complete
        if self.is_motion_complete():
            self.send_stop_command()
            self.get_logger().info("✅ Approach complete")
            self.state = PPState.PAUSE_BEFORE_GRIP
    
    def _state_pause_before_grip(self, now):
        """PAUSE_BEFORE_GRIP: Brief pause before gripping."""
        # Ensure stopped
        self.send_stop_command()
        
        if (now - self.state_start_time) > self.pause_before_grip:
            self.state = PPState.GRIP_CLOSE
    
    def _state_grip_close(self, now):
        """GRIP_CLOSE: Close gripper."""
        if self.state_start_time == now:  # First time
            self.send_gripper_command(self.gripper_close_pos)
            self.gripper_end_time = now + self.gripper_move_time
        
        # Wait for gripper to close
        if now >= self.gripper_end_time:
            self.get_logger().info("✅ Gripper closed")
            self.state = PPState.LIFT
    
    def _state_lift(self, now):
        """LIFT: Lift object up."""
        if self.state_start_time == now:  # First time
            self.start_motion(self.lift_velocity, self.lift_duration)
        
        # Update velocity
        self.update_motion_velocity()
        
        # Check if motion complete
        if self.is_motion_complete():
            self.send_stop_command()
            self.get_logger().info("✅ Lift complete")
            self.state = PPState.MOVE_TO_DROP
    
    def _state_move_to_drop(self, now):
        """MOVE_TO_DROP: Rotate to drop position."""
        if self.state_start_time == now:  # First time
            self.start_motion(self.rotate_velocity, self.rotate_duration)
        
        # Update velocity
        self.update_motion_velocity()
        
        # Check if motion complete
        if self.is_motion_complete():
            self.send_stop_command()
            self.get_logger().info("✅ At drop position")
            self.state = PPState.PAUSE_BEFORE_RELEASE
    
    def _state_pause_before_release(self, now):
        """PAUSE_BEFORE_RELEASE: Brief pause before releasing."""
        # Ensure stopped
        self.send_stop_command()
        
        if (now - self.state_start_time) > self.pause_before_release:
            self.state = PPState.RELEASE
    
    def _state_release(self, now):
        """RELEASE: Open gripper to drop object."""
        if self.state_start_time == now:  # First time
            self.send_gripper_command(self.gripper_open_pos)
            self.gripper_end_time = now + self.gripper_move_time
        
        # Wait for gripper to open
        if now >= self.gripper_end_time:
            self.get_logger().info("✅ Object released")
            self.state = PPState.RETURN_HOME
    
    def _state_return_home(self, now):
        """RETURN_HOME: Return to home position."""
        if self.state_start_time == now:  # First time
            self.start_motion(self.rotate_back_velocity, self.rotate_back_duration)
        
        # Update velocity
        self.update_motion_velocity()
        
        # Check if motion complete
        if self.is_motion_complete():
            self.send_stop_command()
            
            # Optional: Second stage of return home
            # You could add more motions here if needed
            
            self.get_logger().info("✅ Returned to home")
            self.state = PPState.ENABLE_IBVS
    
    def _state_enable_ibvs(self, now):
        """ENABLE_IBVS: Re-enable IBVS for next pick."""
        # Ensure stopped
        self.send_stop_command()
        
        # Re-enable IBVS
        self.pub_disable_ibvs.publish(Bool(data=False))
        
        # Set cooldown
        self.cooldown_end_time = now + self.cooldown_sec
        self.pick_attempts += 1
        
        self.get_logger().info(f"✅ Pick #{self.pick_attempts} complete. Cooldown for {self.cooldown_sec}s")
        self.state = PPState.COOLDOWN
    
    def _state_cooldown(self, now):
        """COOLDOWN: Wait before next pick."""
        if now >= self.cooldown_end_time:
            self.get_logger().info("🔄 Cooldown complete, ready for next pick")
            self.state = PPState.TRACKING
    
    def _state_error(self, now):
        """ERROR: Error recovery state."""
        self.get_logger().error("⚠️ In ERROR state - attempting recovery")
        
        # Stop all motion
        self.send_stop_command()
        
        # Try to re-enable IBVS
        self.pub_disable_ibvs.publish(Bool(data=False))
        
        # After 3 seconds, return to tracking
        if (now - self.state_start_time) > 3.0:
            self.get_logger().info("Recovering to TRACKING state")
            self.state = PPState.TRACKING


def main(args=None):
    rclpy.init(args=args)
    node = VelocityOnlyPickPlace()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down velocity pick-place executor...")
    finally:
        # Ensure clean shutdown
        node.send_stop_command()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()