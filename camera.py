#!/usr/bin/env python3
"""
IBVS Vision Node - QR CODE TRACKING OPTIMIZED
Designed to maintain high publishing rate to keep QR in camera view.
Key: Publish FAST and CONTINUOUSLY when QR detected.
"""
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

# ============================================================================
# OPTIMIZED FOR HIGH-RATE QR TRACKING
# ============================================================================

# Camera intrinsics
FX, FY = 525.0, 525.0
CX, CY = 320.0, 240.0

# Target size
QR_WIDTH_M = 0.05

# AGGRESSIVE GAINS for keeping QR in view
LAMBDA = 1.0  # Higher for aggressive centering

# Damping
DAMP = 0.008

# Velocity limits - HIGH
LIN_MAX = 0.250
ANG_MAX = 0.8

# Detection
CONF_MIN = 0.25  # Lower to detect farther
ERR_STOP_PX = 8.0  # Larger deadband
LOST_HYST_TICKS = 10
MIN_BOX_PIX = 5

# MINIMAL SMOOTHING for fast response
EMA_ALPHA_POSITION = 0.25  # Low smoothing
EMA_ALPHA_DEPTH = 0.35
EMA_ALPHA_TWIST = 0.20  # Very low

# Depth
Z_MIN, Z_MAX = 0.08, 3.0
Z_NOMINAL = 0.50

# CRITICAL: Fast processing
YOLO_INPUT_SIZE = 320  # Balanced speed/accuracy
SKIP_FRAMES = 0  # Process EVERY frame when QR visible
PUBLISH_EVEN_IF_CENTERED = True  # Always publish to maintain tracking

# Disable viz during tracking
DISABLE_VIZ_WHILE_TRACKING = True

# ============================================================================
# YOLO SETUP
# ============================================================================

pkg_share = get_package_share_directory('robot_bringup')
yolo_dir = os.path.join(pkg_share, 'models', 'yolov5')
model_path = os.path.join(pkg_share, 'models', 'content', 'qrcode_model.pt')

if not os.path.exists(yolo_dir):
    raise FileNotFoundError(f"YOLOv5 directory not found: {yolo_dir}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

os.environ.setdefault('YOLOV5_SUPPRESS', '1')
os.environ.setdefault('YOLOV5_NO_AUTOINSTALL', '1')
os.environ.setdefault('GIT_DISCOVERY_ACROSS_FILESYSTEM', '1')

if yolo_dir not in sys.path:
    sys.path.insert(0, yolo_dir)

try:
    from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
    torch.serialization.add_safe_globals([Sequential, ModuleList, ModuleDict])
    from models.yolo import DetectionModel, Detect
    from models.common import Conv, Bottleneck, C3, SPPF, Concat, BottleneckCSP, Focus
    torch.serialization.add_safe_globals([
        DetectionModel, Detect, Conv, Bottleneck, C3, SPPF, Concat, BottleneckCSP, Focus
    ])
except Exception as e:
    print(f"⚠️ Safe globals: {e}", file=sys.stderr)

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

try:
    model = torch.hub.load(
        yolo_dir, 'custom', path=model_path, source='local', force_reload=False
    ).to(device)
except:
    model = torch.hub.load(
        yolo_dir, 'custom', path=model_path, source='local', force_reload=True
    ).to(device)

model.conf = 0.20
model.iou = 0.45
if device == 'cuda':
    try:
        model.half()
    except:
        pass
model.eval()
print(f"✅ Model ready")

# Warmup
with torch.inference_mode():
    dummy = torch.zeros((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), device=device)
    if device == 'cuda':
        try:
            dummy = dummy.half()
        except:
            pass
    for _ in range(3):
        _ = model(dummy)

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
        self.declare_parameter('show_visualization', False)

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
        
        # Performance
        self.frame_count = 0
        self.detection_count = 0
        self.twist_publish_count = 0
        self.last_fps_time = time.time()
        self.last_twist_time = 0.0
        self.processing_times = deque(maxlen=30)
        
        # Frame skipping
        self.skip_counter = 0
        self.repub_rate_hz = 50.0
        self.repub_timer = self.create_timer(
                        1.0 / self.repub_rate_hz, self.republish_last_twist)        
        
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

        # Visualization
        if self.show_viz:
            self.window_name = "QR Tracking"
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, 960, 720)
            except:
                self.show_viz = False

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
        self.get_logger().info(f"  Target: 3-5 Hz publishing")
        self.get_logger().info(f"  Continuous publishing: {PUBLISH_EVEN_IF_CENTERED}")
        self.get_logger().info("="*70)

    def on_image(self, msg):
        """Fast callback - queue frame for processing."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
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
        """Run YOLO."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        with torch.inference_mode():
            results = model(img, size=YOLO_INPUT_SIZE)

        self.processing_times.append(time.time() - t0)
        
        boxes = []
        if len(results.xyxy) == 0:
            return boxes
        
        dets = results.xyxy[0]
        if dets is None or dets.shape[0] == 0:
            return boxes
        
        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            conf = float(conf)
            
            if conf < CONF_MIN:
                continue
            
            w = float(x2 - x1)
            h = float(y2 - y1)
            
            if w < MIN_BOX_PIX or h < MIN_BOX_PIX:
                continue
            
            boxes.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class_id": int(cls),
                "size": max(w, h)
            })
        
        return boxes

    def process_frame(self, frame, header):
        """Process detection and publish twist."""
        self.frame_count += 1
        
        # Detect
        boxes = self.detect_objects(frame)
        
        if not boxes:
            self._lost_ticks += 1
            lost = self._lost_ticks >= LOST_HYST_TICKS
            self.pub_target_ok.publish(Bool(data=False))
            self._is_tracking = False
            
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
        best = max(boxes, key=lambda b: b["confidence"])
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
        if DISABLE_VIZ_WHILE_TRACKING and self._is_tracking:
            show_viz_now = False
        
        if show_viz_now:
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
        
        # Limits
        v_cam[0:3] = np.clip(v_cam[0:3], -LIN_MAX, LIN_MAX)
        v_cam[3:6] = np.clip(v_cam[3:6], -ANG_MAX, ANG_MAX)
        
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
        self.pub_twist.publish(msg)
        
    def republish_last_twist(self):
        """Publish last twist at a fixed high rate."""
        # Only do this while we believe we’re tracking
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
        if not self.show_viz:
            return
        
        viz = frame.copy()
        h, w = viz.shape[:2]
        
        cv2.line(viz, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
        cv2.line(viz, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)
        
        if detection_info:
            x1, y1, x2, y2 = detection_info['bbox']
            cx, cy = detection_info['center']
            conf = detection_info['conf']
            Z = detection_info['depth']
            err = detection_info['error']
            status = detection_info['status']
            
            color = (0, 255, 0) if status == "LOCKED" else (255, 128, 0)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
            cv2.circle(viz, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(viz, (cx, cy), (w//2, h//2), (255, 255, 0), 2)
            
            cv2.putText(viz, f"{status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(viz, f"Err: {err:.0f}px", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(viz, f"Z: {Z:.2f}m", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(viz, "NO QR", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, viz)
        cv2.waitKey(1)


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