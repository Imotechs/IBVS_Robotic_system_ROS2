#!/usr/bin/env python3
import os, sys
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# ---------------- Camera intrinsics + target size ----------------
FX, FY = 525.0, 525.0
CX, CY = 320.0, 240.0
QR_WIDTH_M = 0.05   # meters

# IBVS control parameters
LAMBDA = 1.5
DAMP = 1e-3
LIN_MAX = 0.35
ANG_MAX = 0.8

# ---------------- YOLOv5 path & environment ----------------
pkg_share = get_package_share_directory('robot_bringup')
yolo_dir = os.path.join(pkg_share, 'models', 'yolov5')
if yolo_dir not in sys.path:
    sys.path.insert(0, yolo_dir)

os.environ.setdefault('YOLOV5_SUPPRESS', '1')
os.environ.setdefault('YOLOV5_NO_AUTOINSTALL', '1')
os.environ.setdefault('GIT_DISCOVERY_ACROSS_FILESYSTEM', '1')

# Torch safe load tweaks
try:
    from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
    torch.serialization.add_safe_globals([Sequential, ModuleList, ModuleDict])
    try:
        from models.yolo import DetectionModel, Detect
        from models.common import Conv, Bottleneck, C3, SPPF, Concat, BottleneckCSP, Focus
        torch.serialization.add_safe_globals(
            [DetectionModel, Detect, Conv, Bottleneck, C3, SPPF, Concat, BottleneckCSP, Focus]
        )
    except Exception:
        pass
    _orig_torch_load = torch.load
    def _torch_load_no_weights_only(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_no_weights_only
except Exception:
    pass


class CameraQRCodeIBVS(Node):
    def __init__(self):
        super().__init__('camera_qrcode_ibvs')

        # Parameters
        self.declare_parameter('camera_topic', '/ur5/tcp/image_raw')
        self.declare_parameter('twist_topic', '/ibvs/target_twist')
        self.declare_parameter('camera_frame', 'camera_lens_link')

        self.camera_topic = self.get_parameter('camera_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # State
        self.frame_count = 0
        self.u_ema = self.v_ema = self.w_ema = None
        self.alpha = 0.25  # smoothing factor
        self.last_seen_time = self.get_clock().now()
        self.lost_timeout = 2.0  # seconds to consider target lost

        # Load YOLO model
        model_path = os.path.join(pkg_share, 'models', 'content', 'qrcode_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.get_logger().info(f'Loading YOLOv5 weights: {model_path}')
        self.model = torch.hub.load(
            yolo_dir, 'custom', path=model_path, source='local',
            force_reload=False, trust_repo=True
        )
        self.model.conf = 0.25
        self.model.iou = 0.45

        # ROS I/O
        self.bridge = CvBridge()
        qos_fast = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, self.camera_topic, self.on_image, qos_fast)
        self.pub = self.create_publisher(TwistStamped, self.twist_topic, 1)
        self.pub_target_ok = self.create_publisher(Bool, '/ibvs/target_ok', 1)

        # Visualization window
        self.window_name = 'IBVS Control'
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass

        self.get_logger().info(f'Subscribed: {self.camera_topic}')
        self.get_logger().info(f'Publishing twists: {self.twist_topic} (frame: {self.camera_frame})')

        # Warm up the model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, size=640)
        self.get_logger().info("Model warmed up and ready.")

    # -------------------- Utility --------------------
    def ema(self, prev, new):
        return new if prev is None else (1.0 - self.alpha) * prev + self.alpha * new

    def desired_uv_from_msg(self, msg):
        return np.array([msg.width * 0.5, msg.height * 0.5], dtype=np.float32)

    # -------------------- Detection --------------------
    @torch.inference_mode()
    def detect_best_box(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)
        if len(results.xyxy) == 0 or results.xyxy[0].shape[0] == 0:
            return None
        dets = results.xyxy[0].detach().cpu().numpy()
        best = dets[np.argmax(dets[:, 4])]
        x1, y1, x2, y2, conf, cls_id = best
        return int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)

    # -------------------- IBVS Core --------------------
    def ibvs_twist_from_box(self, u, v, w_pix, desired_uv):
        Z = (FX * QR_WIDTH_M) / max(w_pix, 1.0)
        Z = float(np.clip(Z, 0.10, 2.50))  # safety clamp
        x = (u - CX) / FX
        y = (v - CY) / FY

        L = np.array([
            [-1.0/Z, 0.0, x/Z, x*y, -(1.0 + x*x), y],
            [0.0, -1.0/Z, y/Z, 1.0 + y*y, -x*y, -x]
        ], dtype=np.float32)

        e = np.array([u, v], dtype=np.float32) - desired_uv
        Lt = L.T
        A = Lt @ L + DAMP * np.eye(6, dtype=np.float32)
        b = Lt @ e
        v_cam = -LAMBDA * np.linalg.solve(A, b)

        v_cam[0:3] = np.clip(v_cam[0:3], -LIN_MAX, LIN_MAX)
        v_cam[3:6] = np.clip(v_cam[3:6], -ANG_MAX, ANG_MAX)
        err_px = float(np.linalg.norm(e))
        return v_cam, Z, err_px

    # -------------------- ROS Interface --------------------
    def publish_twist(self, header, v_cam):
        msg = TwistStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = self.camera_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = [float(x) for x in v_cam[0:3]]
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = [float(x) for x in v_cam[3:6]]
        self.pub.publish(msg)

    def publish_zero(self, header, go_home=False):
        msg = TwistStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "go_home" if go_home else self.camera_frame
        self.pub.publish(msg)

    # -------------------- Main Image Callback --------------------
    def on_image(self, msg):
        now = self.get_clock().now()
        frame_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_s = now.nanoseconds * 1e-9
        self.get_logger().info(f"Frame lag: {now_s - frame_time:.3f}s")

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        det = self.detect_best_box(frame)
        desired_uv = self.desired_uv_from_msg(msg)

        # --- No detection ---
        if det is None:
            if (now - self.last_seen_time).nanoseconds * 1e-9 > self.lost_timeout:
                self.publish_zero(msg.header, go_home=True)
                self.pub_target_ok.publish(Bool(data=False))
            else:
                self.publish_zero(msg.header)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        x1, y1, x2, y2, conf, cls_id = det
        if conf < 0.65:
            self.publish_zero(msg.header)
            self.pub_target_ok.publish(Bool(data=False))
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return

        self.last_seen_time = now
        self.pub_target_ok.publish(Bool(data=True))

        # --- Compute feature center ---
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        w_pix = max(x2 - x1, y2 - y1)

        # --- Smooth values ---
        u = self.ema(self.u_ema, u); self.u_ema = u
        v = self.ema(self.v_ema, v); self.v_ema = v
        w_pix = self.ema(self.w_ema, w_pix); self.w_ema = w_pix

        # --- Compute IBVS control ---
        v_cam, Z, err_px = self.ibvs_twist_from_box(u, v, w_pix, desired_uv)

        if err_px < 1.5:
            self.publish_zero(msg.header)
        else:
            self.publish_twist(msg.header, v_cam)

        # --- Visualization ---
        try:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(u), int(v)), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"conf:{conf:.2f} Z~{Z:.3f}m err:{err_px:.1f}px",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraQRCodeIBVS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
