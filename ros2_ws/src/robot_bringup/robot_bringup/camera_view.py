#!/usr/bin/env python3

import os, sys, threading, time
import numpy as np
import cv2
import torch
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from robot_bringup_interfaces.msg import IBVSState

# ============================================================================
# OPTIMIZED FOR HIGH-RATE QR TRACKING
# ============================================================================

# Camera intrinsics
FX, FY = 525.0, 525.0
CX, CY = 320.0, 240.0

# Target size
QR_WIDTH_M = 0.05

# AGGRESSIVE GAINS for keeping QR in view
LAMBDA = 0.2  #0.8 # Higher for aggressive centering

# Damping
DAMP = 0.01

# Velocity limits - HIGH
LIN_MAX = 0.250
ANG_MAX = 0.8

# Detection
CONF_MIN = 0.25  # Lower to detect farther
ERR_STOP_PX = 20.0  # Larger deadband
LOST_HYST_TICKS = 10
MIN_BOX_PIX = 5

# MINIMAL SMOOTHING for fast response
EMA_ALPHA_POSITION = 0.30  # Low smoothing
EMA_ALPHA_DEPTH = 0.35
EMA_ALPHA_TWIST = 0.70  # Very low

# Depth
Z_MIN, Z_MAX = 0.08, 3.0
Z_NOMINAL = 0.50

# CRITICAL: Fast processing
YOLO_INPUT_SIZE = 320  # Balanced speed/accuracy
SKIP_FRAMES = 0  # Process EVERY frame when QR visible
PUBLISH_EVEN_IF_CENTERED = True  # Always publish to maintain tracking

# Disable viz during tracking
DISABLE_VIZ_WHILE_TRACKING = False
LOCK_LOST_THRESHOLD = 5   # frames




from ultralytics import YOLO
pkg_share = get_package_share_directory('robot_bringup')
# Update model path if needed (YOLOv8 models have .pt extension too)
model_path = os.path.join(pkg_share, 'models', 'best.pt')  

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading YOLOv8 model on {device}...")

try:
    model = YOLO(model_path).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    # Try with a different approach if needed
    model = YOLO(model_path)

# YOLOv8 inference settings
model.conf = 0.20  # Confidence threshold
model.iou = 0.45   # NMS IoU threshold
print(f"✅ YOLOv8 Model ready")

# Warmup (different for YOLOv8)
with torch.inference_mode():
    dummy = torch.zeros((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), device=device)
    for _ in range(3):
        # YOLOv8 has different prediction method
        _ = model(dummy, verbose=False)

# ============================================================================
# HIGH-RATE VISION NODE
# ============================================================================

class HighRateQRVision(Node):
    """
    Vision node optimized for continuous QR tracking.
    Publishes at high rate to help controller maintain visual contact.
    """
    
    def __init__(self):
        super().__init__('high_rate_qr_vision')

        # Parameters
        self.declare_parameter('camera_topic', '/ur5/tcp/image_raw')
        self.declare_parameter('twist_topic', '/ibvs/target_twist')
        self.declare_parameter('target_ok_topic', '/ibvs/target_ok')
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('show_visualization', True)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.target_ok_topic = self.get_parameter('target_ok_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.show_viz = self.get_parameter('show_visualization').value

        # State
        self.u_ema = None
        self.v_ema = None
        self.w_ema = None
        self.z_ema = None
        self.z_history = deque(maxlen=3)
        self.twist_ema = None
        self.last_twist_msg = None
        self._lost_ticks = 0
        self._last_detection_time = 0.0
        self._is_tracking = False
        #werwerw
        self.last_center = None
        self.last_bbox = None
        self.target_locked = False
        self.lost_lock_count = 0

        # Performance
        self.frame_count = 0
        self.detection_count = 0
        self.twist_publish_count = 0
        self.last_fps_time = time.time()
        self.last_twist_time = 0.0
        self.processing_times = deque(maxlen=30)
        
        # Frame skipping
        self.skip_counter = 0
        self.repub_rate_hz = 100.0
        self.repub_timer = self.create_timer(
                        1.0 / self.repub_rate_hz, self.republish_last_twist)        
        
        self.state_pub = self.create_publisher(IBVSState, "/ibvs/state", 10)

        # ROS
        self.bridge = CvBridge()
        
        qos_fast = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        self.sub = self.create_subscription(
            Image, self.camera_topic, self.on_image, qos_fast
        )
        self.pub_twist = self.create_publisher(TwistStamped, self.twist_topic, 10)
        self.pub_target_ok = self.create_publisher(Bool, self.target_ok_topic, 10)

        # Visualization - SINGLE window creation
        self.viz_pub = self.create_publisher(Image, "/ibvs/visualization", 10)

        
        # REMOVED the duplicate window creation section (lines 159-168 in your code)

        # Processing thread
        self.lock = threading.Lock()
        self.latest_frame = None
        self.processing = False
        self.process_thread = None
        
        # Frame queue for high throughput
        self.frame_queue = deque(maxlen=2)

        self.get_logger().info("="*70)
        self.get_logger().info("✅ HIGH-RATE QR VISION NODE")
        self.get_logger().info("="*70)
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Twist: {self.twist_topic}")
        self.get_logger().info(f"  Lambda: {LAMBDA}")
        self.get_logger().info(f"  YOLO size: {YOLO_INPUT_SIZE}")
        self.get_logger().info(f"  Visualization: {'ENABLED' if self.show_viz else 'DISABLED'}")
        self.get_logger().info(f"  Target: 3-5 Hz publishing")
        self.get_logger().info(f"  Continuous publishing: {PUBLISH_EVEN_IF_CENTERED}")
        self.get_logger().info("="*70)

    def on_image(self, msg):
        """Fast callback - queue frame for processing."""
        try:
            #frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            frame = self.bridge.imgmsg_to_cv2(msg)

        except Exception as e:
            self.get_logger().error(f"cv_bridge: {e}")
            return
        
        # Queue frame
        with self.lock:
            self.frame_queue.append((frame.copy(), msg.header))
        
        # Process if not busy
        if not self.processing:
            self.process_thread = threading.Thread(
                target=self.process_frame_from_queue,
                daemon=True
            )
            self.process_thread.start()
    def select_target(self, boxes):
        if not boxes:
            return None

        # If no lock yet → pick highest confidence
        if self.last_center is None:
            best = max(boxes, key=lambda b: b["confidence"])
            self.last_center = (
                0.5 * (best["bbox"][0] + best["bbox"][2]),
                0.5 * (best["bbox"][1] + best["bbox"][3]),
            )
            self.target_locked = True
            self.lost_lock_count = 0
            return best

        # If locked → pick closest in image space
        cx_prev, cy_prev = self.last_center

        def dist2(box):
            x1, y1, x2, y2 = box["bbox"]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            return (cx - cx_prev) ** 2 + (cy - cy_prev) ** 2

        best = min(boxes, key=dist2)

        # Update lock
        cx = 0.5 * (best["bbox"][0] + best["bbox"][2])
        cy = 0.5 * (best["bbox"][1] + best["bbox"][3])
        self.last_center = (cx, cy)
        self.lost_lock_count = 0

        return best

    def process_frame_from_queue(self):
        """Process frame from queue."""
        self.processing = True
        
        try:
            with self.lock:
                if not self.frame_queue:
                    return
                # Get latest frame
                frame, header = self.frame_queue[-1]
                self.frame_queue.clear()
            
            self.process_frame(frame, header)
            
        finally:
            self.processing = False

    def detect_objects(self, frame):
        """Run YOLOv8 detection - simpler version."""
        t0 = time.time()
        
        # YOLOv8 inference
        results = model.predict(
            frame, 
            imgsz=YOLO_INPUT_SIZE,
            conf=CONF_MIN,
            verbose=False
        )
        
        self.processing_times.append(time.time() - t0)
        
        boxes = []
        
        if len(results) == 0:
            return boxes
        
        result = results[0]
        
        for box in result.boxes:
            conf = float(box.conf)
            if conf < CONF_MIN:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            
            w = x2 - x1
            h = y2 - y1
            
            if w < MIN_BOX_PIX or h < MIN_BOX_PIX:
                continue
            
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "size": max(w, h)
            })
        
        return boxes

    def process_frame(self, frame, header):
        """Process detection and publish twist."""
        self.frame_count += 1        
        # DEBUG: Check frame
        if frame is None:
            self.get_logger().error("Frame is None!")
            return
        
        # Detect
        boxes = self.detect_objects(frame)
        self.get_logger().error(f"DETECTIONS: {len(boxes)}")

        if not boxes:
            self._lost_ticks += 1
            lost = self._lost_ticks >= LOST_HYST_TICKS
            self.pub_target_ok.publish(Bool(data=False))
            self._is_tracking = False
            self.lost_lock_count+=1
            if self.lost_lock_count >= LOCK_LOST_THRESHOLD:
                self.last_center = None
                self.last_bbox = None
                self.target_locked = False
            if lost and self._lost_ticks == LOST_HYST_TICKS:
                self.publish_zero_twist(header, go_home=True)
                #QR IS LOST
            # Show viz when not tracking
            if self.show_viz and not DISABLE_VIZ_WHILE_TRACKING:
                self.draw_frame(frame, None)
            return
        
        # QR FOUND!
        self._lost_ticks = 0
        self._last_detection_time = time.time()
        self.detection_count += 1
        self._is_tracking = True
        self.pub_target_ok.publish(Bool(data=True))
        
        # Best detection
        #best = max(boxes, key=lambda b: b["confidence"])
        best = self.select_target(boxes)
        if best is None:
            return
        x1, y1, x2, y2 = best["bbox"]
        
        # Center and size
        u_raw = 0.5 * (x1 + x2)
        v_raw = 0.5 * (y1 + y2)
        w_pix_raw = best["size"]
        
        # Light smoothing
        self.u_ema = self.ema(self.u_ema, u_raw, EMA_ALPHA_POSITION)
        self.v_ema = self.ema(self.v_ema, v_raw, EMA_ALPHA_POSITION)
        self.w_ema = self.ema(self.w_ema, w_pix_raw, EMA_ALPHA_POSITION)
        
        u = self.u_ema
        v = self.v_ema
        w_pix = self.w_ema
        
        # Desired position
        img_h, img_w = frame.shape[:2]
        desired_uv = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
        
        # Compute twist
        v_cam, Z, err_px = self.compute_ibvs_twist(u, v, w_pix, desired_uv)
                # Decide if centered + close enough for picking
        CENTER_THRESHOLD = 30.0     # pixels (tune)
        DEPTH_THRESHOLD = 0.18      # meters (tune)

        state = IBVSState()
        state.pixel_error = float(err_px)
        state.depth = float(Z)
        state.centered = (err_px < CENTER_THRESHOLD)
        state.close = (Z < DEPTH_THRESHOLD)
        state.target_visible = True
        self.last_state_msg = state  # Store for republishing

        self.state_pub.publish(state)
        
        # ALWAYS publish when QR visible (critical for tracking!)
        should_publish = True
        
        if not PUBLISH_EVEN_IF_CENTERED:
            should_publish = err_px >= ERR_STOP_PX
        
        if should_publish:
            self.publish_twist(header, v_cam)
            self.twist_publish_count += 1
            
            now = time.time()
            twist_hz = 1.0 / max(now - self.last_twist_time, 1e-6)
            self.last_twist_time = now
        else:
            self.publish_zero_twist(header)
        
        status = "LOCKED" if err_px < ERR_STOP_PX else "TRACKING"
        
        # Viz only if enabled and not tracking
        show_viz_now = self.show_viz
        # if DISABLE_VIZ_WHILE_TRACKING and self._is_tracking:
        #     show_viz_now = False
        
        if self.show_viz:
            self.draw_frame(frame, {
                'bbox': (x1, y1, x2, y2),
                'center': (int(u), int(v)),
                'conf': best['confidence'],
                'depth': Z,
                'error': err_px,
                'status': status
            })
        
        # Periodic stats
        if self.frame_count % 100 == 0:
            now = time.time()
            fps = 100 / max(now - self.last_fps_time, 1e-6)
            twist_rate = self.twist_publish_count / max(now - self.last_fps_time, 1e-6)
            det_rate = 100.0 * self.detection_count / self.frame_count
            avg_proc = np.mean(list(self.processing_times)) * 1000
            
            self.last_fps_time = now
            
            self.get_logger().info(
                f"📊 Vision FPS: {fps:.1f}, Twist Hz: {twist_rate:.2f}, "
                f"Det rate: {det_rate:.1f}%, Proc: {avg_proc:.0f}ms, "
                f"Error: {err_px:.1f}px"
            )
            
            self.twist_publish_count = 0

    def compute_ibvs_twist(self, u, v, w_pix, desired_uv):
        """Compute twist."""
        # Depth
        Z_raw = (FX * QR_WIDTH_M) / max(w_pix, 1.0)
        
        self.z_history.append(Z_raw)
        
        if self.z_ema is None:
            self.z_ema = Z_raw
        else:
            Z_median = np.median(list(self.z_history))
            self.z_ema = EMA_ALPHA_DEPTH * self.z_ema + (1 - EMA_ALPHA_DEPTH) * Z_median
        
        Z = float(np.clip(self.z_ema, Z_MIN, Z_MAX))
        
        # Normalized coords
        x = (u - CX) / FX
        y = (v - CY) / FY
        
        # Jacobian
        L = np.array([
            [-1.0/Z,     0.0,    x/Z,   x*y,        -(1.0 + x*x),  y],
            [0.0,    -1.0/Z,    y/Z,   1.0 + y*y,  -x*y,         -x]
        ], dtype=np.float32)
        
        # Error
        e = desired_uv - np.array([u, v], dtype=np.float32)
        err_px = float(np.linalg.norm(e))
        
        # Twist
        Lt = L.T
        A = Lt @ L + DAMP * np.eye(6, dtype=np.float32)
        b = Lt @ e
        
        try:
            v_cam_raw = -LAMBDA * np.linalg.solve(A, b)
        except:
            v_cam_raw = np.zeros(6, dtype=np.float32)
        
        # Minimal smoothing
        if self.twist_ema is None:
            self.twist_ema = v_cam_raw
        else:
            self.twist_ema = EMA_ALPHA_TWIST * self.twist_ema + \
                             (1 - EMA_ALPHA_TWIST) * v_cam_raw
        
        v_cam = self.twist_ema.copy()
        
        
        Z_NOMINAL = 0.5  # Reference depth
        SCALE_FACTOR = min(1.0, Z / Z_NOMINAL)  # Reduce gain when close
    
        v_cam = self.twist_ema.copy() * SCALE_FACTOR
        lin_max = LIN_MAX 
        ang_max = ANG_MAX
        if Z < 0.3:  # When very close (<30cm)
            lin_max = LIN_MAX * 0.3  # Reduce to 30% of max
            ang_max = ANG_MAX * 0.3
        elif Z < 0.5:  # When moderately close (<50cm)
            lin_max = LIN_MAX * 0.6  # Reduce to 60% of max
            ang_max = ANG_MAX * 0.6
        else:
            lin_max = LIN_MAX
            ang_max = ANG_MAX
        # Limits
        v_cam[0:3] = np.clip(v_cam[0:3], -lin_max, lin_max)
        v_cam[3:6] = np.clip(v_cam[3:6], -ang_max, ang_max)
        
        return v_cam, Z, err_px

    def ema(self, prev, new, alpha):
        if prev is None:
            return new
        return (1.0 - alpha) * prev + alpha * new

    def publish_twist(self, header, v_cam):
        msg = TwistStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.camera_frame
        msg.twist.linear.x = float(v_cam[0])
        msg.twist.linear.y = float(v_cam[1])
        msg.twist.linear.z = float(v_cam[2])
        msg.twist.angular.x = float(v_cam[3])
        msg.twist.angular.y = float(v_cam[4])
        msg.twist.angular.z = float(v_cam[5])
        
        # store & publish
        self.last_twist_msg = msg
        self.get_logger().warn(f"Publishing twist{msg}")
        self.pub_twist.publish(msg)
        
    def republish_last_twist(self):
        """Publish last twist at a fixed high rate."""
        # Only do this while we believe we’re tracking
        #self.get_logger().warning(f"Republishing last twist{self.last_twist_msg}")
        if not self._is_tracking:
            return

        if self.last_twist_msg is None:
            return

        # Update timestamp to 'now' so controller doesn't think it's stale
        self.last_twist_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_twist.publish(self.last_twist_msg)


    def publish_zero_twist(self, header, go_home=False):
        msg = TwistStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "go_home" if go_home else self.camera_frame
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.pub_twist.publish(msg)

    def draw_frame(self, frame, detection_info):
        """Draw YOLO detections with error visualization."""
        if not self.show_viz:
            return
        viz = frame.copy()
        h, w = viz.shape[:2]
        
        # Draw image center crosshair
        center_color = (0, 255, 0)  # Green
        cv2.line(viz, (w//2 - 25, h//2), (w//2 + 25, h//2), center_color, 3)
        cv2.line(viz, (w//2, h//2 - 25), (w//2, h//2 + 25), center_color, 3)
        cv2.circle(viz, (w//2, h//2), 4, (0, 255, 255), -1)  # Yellow center dot
        
        if detection_info:
            x1, y1, x2, y2 = detection_info['bbox']
            cx, cy = detection_info['center']
            conf = detection_info['conf']
            Z = detection_info['depth']
            err = detection_info['error']
            status = detection_info['status']
            
            # Draw detection bounding box
            box_color = (0, 255, 0) if status == "locked" else (255, 128, 0)  # Green/Orange
            cv2.rectangle(viz, (x1, y1), (x2, y2), box_color, 3)
            
            # Draw detection center
            cv2.circle(viz, (cx, cy), 8, (0, 0, 255), -1)  # Red detection center
            
            # Draw error line from detection center to image center
            cv2.line(viz, (cx, cy), (w//2, h//2), (255, 255, 0), 2)  # Cyan line
            
            # Draw error vector with arrow
            cv2.arrowedLine(viz, (cx, cy), (w//2, h//2), (255, 100, 100), 2, tipLength=0.1)
            
            # Calculate error in x and y directions
            error_x = w//2 - cx
            error_y = h//2 - cy
            
            # Enhanced info display
            info_y = 30
            line_height = 30
            
            # Status
            cv2.putText(viz, f"Status: {status}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            info_y += line_height
            
            # Confidence
            cv2.putText(viz, f"Confidence: {conf:.2f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            info_y += line_height
            
            # Total error
            cv2.putText(viz, f"Total Error: {err:.1f} px", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            info_y += line_height
            
            # X/Y errors
            cv2.putText(viz, f"X Error: {error_x:+d} px", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            info_y += line_height
            
            cv2.putText(viz, f"Y Error: {error_y:+d} px", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            info_y += line_height
            
            # Depth
            cv2.putText(viz, f"Depth: {Z:.3f} m", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 150, 0), 2)
            info_y += line_height
            
            # Box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            cv2.putText(viz, f"Box: {box_width}x{box_height} px", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
            
            # Draw error values near the line
            mid_x = (cx + w//2) // 2
            mid_y = (cy + h//2) // 2
            cv2.putText(viz, f"{err:.0f}px", (mid_x - 20, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw coordinate info at detection center
            cv2.putText(viz, f"({cx},{cy})", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            
            # Draw target coordinate
            cv2.putText(viz, f"Target: ({w//2},{h//2})", (w//2 + 10, h//2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
            
        else:
            # No detection - show warning
            cv2.putText(viz, "NO DETECTION", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(viz, "Searching for QR code...", (w//2 - 150, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            
            # Show detection parameters
            info_y = 30
            cv2.putText(viz, f"YOLOv8 Detection Monitor", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            info_y += 30
            cv2.putText(viz, f"Model: best.pt", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            info_y += 25
            cv2.putText(viz, f"Confidence threshold: {CONF_MIN}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Show FPS if available
        if hasattr(self, 'processing_times') and len(self.processing_times) > 0:
            avg_proc = np.mean(list(self.processing_times)) * 1000
            fps = 1000 / avg_proc if avg_proc > 0 else 0
            cv2.putText(viz, f"FPS: {fps:.1f} | Proc: {avg_proc:.0f}ms", 
                       (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            viz_msg.header.stamp = self.get_clock().now().to_msg()
            viz_msg.header.frame_id = "camera_optical_frame"
            self.viz_pub.publish(viz_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish visualization: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = HighRateQRVision()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.frame_count > 0:
            det_rate = 100.0 * node.detection_count / node.frame_count
            node.get_logger().info("="*70)
            node.get_logger().info("📊 VISION STATISTICS")
            node.get_logger().info(f"  Frames: {node.frame_count}")
            node.get_logger().info(f"  Detections: {node.detection_count} ({det_rate:.1f}%)")
            node.get_logger().info(f"  Twists published: {node.twist_publish_count}")
            if node.processing_times:
                node.get_logger().info(f"  Avg proc: {np.mean(list(node.processing_times))*1000:.0f}ms")
            node.get_logger().info("="*70)
    finally:
        if node.show_viz:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()