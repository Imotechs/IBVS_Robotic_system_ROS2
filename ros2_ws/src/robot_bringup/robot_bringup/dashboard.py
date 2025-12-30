#!/usr/bin/env python3
"""
IBVS Dashboard Node - Standalone visualization
Subscribes to vision and controller data, displays comprehensive plots
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from collections import deque
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from robot_bringup_interfaces.msg import IBVSState
from std_msgs.msg import Float64MultiArray, Bool
from cv_bridge import CvBridge
import time
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io

PICK_DEPTH_THRESHOLD =2
class IBVSDashboard(Node):
    def __init__(self):
        super().__init__('ibvs_dashboard')
        
        # Subscribers
        self.create_subscription(IBVSState, "/ibvs/state", self.on_vision_state, 10)
        self.create_subscription(TwistStamped, "/ibvs/target_twist", self.on_twist, 10)
        self.create_subscription(Float64MultiArray, "/forward_velocity_controller/commands", 
                                self.on_joint_velocities, 10)
        self.create_subscription(Bool, "/ibvs/disable_control", self.on_control_state, 10)
        
        # For camera image visualization
        self.create_subscription(Image, "/ibvs/visualization", self.on_camera_image, 10)
        
        # Data buffers
        self.error_history = deque(maxlen=200)
        self.depth_history = deque(maxlen=200)
        self.twist_history = deque(maxlen=200)
        self.joint_vel_history = deque(maxlen=200)
        self.timestamps = deque(maxlen=200)
        
        # Current state
        self.current_error = 0.0
        self.current_depth = 0.0
        self.current_twist = np.zeros(6)
        self.current_joint_vel = np.zeros(6)
        self.control_enabled = True
        self.last_update = time.time()
        self.camera_image = None
        
        # Bridge for image conversion
        self.bridge = CvBridge()
        
        # Visualization publisher
        self.viz_pub = self.create_publisher(Image, "/dashboard/visualization", 10)
        
        # Timer for updating dashboard
        self.create_timer(0.05, self.update_dashboard)  # 20 Hz update
        
        self.get_logger().info("📊 IBVS Dashboard started")

    def on_vision_state(self, msg):
        self.current_error = msg.pixel_error
        self.current_depth = msg.depth
        self.error_history.append(msg.pixel_error)
        self.depth_history.append(msg.depth)
        self.timestamps.append(time.time())

    def on_twist(self, msg):
        twist = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
        ])
        self.current_twist = twist
        self.twist_history.append(twist)

    def on_joint_velocities(self, msg):
        self.current_joint_vel = np.array(msg.data)
        self.joint_vel_history.append(self.current_joint_vel)

    def on_control_state(self, msg):
        self.control_enabled = not msg.data

    def on_camera_image(self, msg):
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass
    def plot_depth_opencv(self, size):
        """Create depth plot using OpenCV."""
        w, h = size
        plot = np.ones((h, w, 3), dtype=np.uint8) * 30
        
        if len(self.depth_history) < 2:
            cv2.putText(plot, "Collecting depth data...", 
                    (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return plot
        
        # Plot depth curve
        depths = list(self.depth_history)
        x_vals = np.linspace(0, w-1, len(depths))
        
        # Normalize for display (assume depth 0-1m)
        y_vals = h - 20 - (np.array(depths) / 1.0 * (h-40))
        
        # Draw curve
        for i in range(len(x_vals)-1):
            cv2.line(plot,
                    (int(x_vals[i]), int(y_vals[i])),
                    (int(x_vals[i+1]), int(y_vals[i+1])),
                    (0, 255, 0), 2)  # Green for depth
        
        # Draw pick threshold
        threshold_y = h - 20 - (PICK_DEPTH_THRESHOLD / 1.0 * (h-40))
        cv2.line(plot, (0, int(threshold_y)), (w, int(threshold_y)), 
                (0, 255, 255), 2, cv2.LINE_AA)  # Yellow threshold line
        
        # Labels
        cv2.putText(plot, f"Current: {self.current_depth:.3f}m", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(plot, f"Pick at: {PICK_DEPTH_THRESHOLD}m", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Safe working distance warning
        if self.current_depth < 0.1:  # Too close!
            cv2.putText(plot, "⚠️ TOO CLOSE!", 
                    (w//2-80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return plot
    def create_matplotlib_plot(self, size=(400, 300)):
        """Create a matplotlib figure and convert to OpenCV image."""
        dpi = 100
        fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi)
        
        # Plot 1: Error over time
        ax1 = plt.subplot(2, 2, 1)
        if len(self.error_history) > 1:
            ax1.plot(list(self.error_history), 'b-', linewidth=2)
            ax1.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='Pick threshold')
        ax1.set_title('Pixel Error')
        ax1.set_ylabel('Pixels')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Depth over time
        ax2 = plt.subplot(2, 2, 2)
        if len(self.depth_history) > 1:
            ax2.plot(list(self.depth_history), 'g-', linewidth=2)
            ax2.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='Pick depth')
        ax2.set_title('Depth')
        ax2.set_ylabel('Meters')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Current twist commands
        ax3 = plt.subplot(2, 2, 3)
        if len(self.current_twist) == 6:
            labels = ['Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz']
            colors = plt.cm.Set3(np.linspace(0, 1, 6))
            bars = ax3.bar(labels, self.current_twist, color=colors)
            for bar, val in zip(bars, self.current_twist):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax3.set_title('Current Twist')
        ax3.set_ylabel('Vel (m/s, rad/s)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Current joint velocities
        ax4 = plt.subplot(2, 2, 4)
        if len(self.current_joint_vel) == 6:
            joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
            colors = plt.cm.viridis(np.linspace(0, 1, 6))
            bars = ax4.bar(joint_labels, self.current_joint_vel, color=colors)
            for bar, val in zip(bars, self.current_joint_vel):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        ax4.set_title('Joint Velocities')
        ax4.set_ylabel('Rad/s')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Convert to OpenCV image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        cv_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        plt.close(fig)
        
        return cv_image

    def create_opencv_dashboard(self, size=(1200, 800)):
        """Create dashboard using OpenCV only (no matplotlib)."""
        dashboard = np.ones((size[1], size[0], 3), dtype=np.uint8) * 40
        
        # Layout: 2x2 grid
        cell_w = size[0] // 2
        cell_h = size[1] // 2
        
        # --- TOP LEFT: Camera feed or status ---
        if self.camera_image is not None:
            # Resize camera image to fit
            cam_resized = cv2.resize(self.camera_image, (cell_w-20, cell_h-20))
            dashboard[10:10+cam_resized.shape[0], 
                     10:10+cam_resized.shape[1]] = cam_resized
        else:
            # Draw status info
            status_y = 50
            cv2.putText(dashboard, "Camera feed not available", 
                       (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status_y += 40
            cv2.putText(dashboard, f"Control: {'ENABLED' if self.control_enabled else 'DISABLED'}", 
                       (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.control_enabled else (0, 0, 255), 2)
        
        # --- TOP RIGHT: Error plot ---
        error_plot = self.plot_error_opencv((cell_w-20, cell_h-20))
        dashboard[10:10+error_plot.shape[0], 
                 cell_w+10:cell_w+10+error_plot.shape[1]] = error_plot
        
        # --- BOTTOM LEFT: Twist commands ---
        twist_plot = self.plot_twist_opencv((cell_w-20, cell_h-20))
        dashboard[cell_h+10:cell_h+10+twist_plot.shape[0],
                 10:10+twist_plot.shape[1]] = twist_plot
        
        # --- BOTTOM RIGHT: System info ---
        info_plot = self.plot_system_info((cell_w-20, cell_h-20))
        dashboard[cell_h+10:cell_h+10+info_plot.shape[0],
                 cell_w+10:cell_w+10+info_plot.shape[1]] = info_plot
        
        return dashboard

    def plot_error_opencv(self, size):
        """Create error plot using OpenCV."""
        w, h = size
        plot = np.ones((h, w, 3), dtype=np.uint8) * 30
        
        if len(self.error_history) < 2:
            cv2.putText(plot, "Collecting data...", 
                       (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return plot
        
        # Plot error curve
        errors = list(self.error_history)
        x_vals = np.linspace(0, w-1, len(errors))
        y_vals = h - 20 - (np.array(errors) / max(max(errors), 50) * (h-40))
        
        # Draw curve
        for i in range(len(x_vals)-1):
            cv2.line(plot,
                    (int(x_vals[i]), int(y_vals[i])),
                    (int(x_vals[i+1]), int(y_vals[i+1])),
                    (0, 255, 255), 2)
        
        # Draw thresholds
        cv2.line(plot, (0, h-20-15), (w, h-20-15), 
                (0, 255, 0), 1, cv2.LINE_AA)  # Pick threshold
        
        # Labels
        cv2.putText(plot, f"Current: {self.current_error:.1f}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(plot, f"Min: {min(errors):.1f}px", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return plot

    def plot_twist_opencv(self, size):
        """Create twist visualization using OpenCV."""
        w, h = size
        plot = np.ones((h, w, 3), dtype=np.uint8) * 30
        
        labels = ['Vx', 'Vy', 'Vz', 'Wx', 'Wy', 'Wz']
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255)
        ]
        
        bar_w = w // 10
        max_val = max(np.max(np.abs(self.current_twist)), 0.1)
        
        for i in range(6):
            x = 20 + i * (bar_w + 20)
            value = self.current_twist[i]
            
            # Bar height (normalized)
            height = int(abs(value) / max_val * (h - 100))
            height = min(height, h - 100)
            
            # Draw bar
            if value >= 0:
                cv2.rectangle(plot,
                            (x, h-50),
                            (x + bar_w, h-50 - height),
                            colors[i], -1)
            else:
                cv2.rectangle(plot,
                            (x, h-50),
                            (x + bar_w, h-50 + height),
                            tuple(c//2 for c in colors[i]), -1)
            
            # Label
            cv2.putText(plot, f"{labels[i]}: {value:.3f}",
                       (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 1)
        
        cv2.putText(plot, f"Magnitude: {np.linalg.norm(self.current_twist):.3f}",
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return plot

    def plot_system_info(self, size):
        """Create system information panel."""
        w, h = size
        plot = np.ones((h, w, 3), dtype=np.uint8) * 30
        
        y = 40
        line_height = 35
        
        # System status
        cv2.putText(plot, "SYSTEM STATUS", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y += line_height
        
        # Control state
        control_color = (0, 255, 0) if self.control_enabled else (0, 0, 255)
        cv2.putText(plot, f"Control: {'ENABLED' if self.control_enabled else 'DISABLED'}",
                   (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, control_color, 1)
        y += line_height
        
        # Error status
        error_color = (0, 255, 0) if self.current_error < 15 else (255, 255, 0)
        cv2.putText(plot, f"Pixel Error: {self.current_error:.1f}px",
                   (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 1)
        y += line_height
        
        # Depth status
        depth_color = (0, 255, 0) if self.current_depth < 0.25 else (255, 255, 0)
        cv2.putText(plot, f"Depth: {self.current_depth:.3f}m",
                   (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_color, 1)
        y += line_height
        
        # Pick readiness
        ready = self.current_error < 15 and self.current_depth < 0.25
        pick_color = (0, 255, 0) if ready else (255, 0, 0)
        cv2.putText(plot, f"Pick Ready: {'YES' if ready else 'NO'}",
                   (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pick_color, 1)
        y += line_height * 2
        
        # Statistics
        cv2.putText(plot, "STATISTICS", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y += line_height
        
        if len(self.error_history) > 1:
            avg_error = np.mean(list(self.error_history))
            std_error = np.std(list(self.error_history))
            cv2.putText(plot, f"Avg Error: {avg_error:.1f}px",
                       (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += line_height
            
            cv2.putText(plot, f"Std Dev: {std_error:.1f}px",
                       (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += line_height
        
        # Update rate
        dt = time.time() - self.last_update
        fps = 1.0 / dt if dt > 0 else 0
        self.last_update = time.time()
        cv2.putText(plot, f"Update Rate: {fps:.1f} Hz",
                   (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return plot

    def update_dashboard(self):
        """Main update function called by timer."""
        # Create dashboard
        dashboard = self.create_opencv_dashboard((1200, 800))
        
        # Convert to ROS message and publish
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(dashboard, "bgr8")
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            viz_msg.header.frame_id = "dashboard"
            self.viz_pub.publish(viz_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish dashboard: {e}")

def main(args=None):
    rclpy.init(args=args)
    dashboard = IBVSDashboard()
    
    try:
        rclpy.spin(dashboard)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()