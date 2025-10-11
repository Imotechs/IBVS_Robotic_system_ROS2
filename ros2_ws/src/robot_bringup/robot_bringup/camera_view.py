#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import pathlib
import sys
import os

# ---------------- CONFIG ----------------
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath






# ---------------- ROS2 NODE ----------------
class CameraQRCodeViewer(Node):
    def __init__(self):
        super().__init__('camera_qrcode_viewer')
        # Paths (adjust to your actual model locations)
        pkg_share = get_package_share_directory('robot_bringup')
        self.get_logger().info(f"Package share directory: {pkg_share}")

        yolo_dir = os.path.join(pkg_share, 'models','yolo', 'yolov5')
        model_path = os.path.join(pkg_share, 'models','yolo', 'content', 'qrcode_model.pt')

        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(model_path)
        
        # ---------------------------------------------------------------------
        # 🔥 Load YOLOv5 model
        # ---------------------------------------------------------------------
        self.get_logger().info("Loading YOLOv5 model...")
        self.model = torch.hub.load(
            yolo_dir,
            'custom',
            path=model_path,
            source='local'
        )
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.get_logger().info("YOLOv5 model loaded successfully.")

        self.subscription = self.create_subscription(
            Image,
            '/ur5/camera/image_raw',  
            self.listener_callback,
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Camera QRCode Viewer node initialized.")
        self.window_name = "QR/Barcode Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    # ---------------- DETECTION FUNCTION ----------------
    def detect_objects_in_frame(self,frame):
        """Run YOLO detection on a frame."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img, size=640)
        boxes = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            boxes.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class_id": int(cls)
            })
        return boxes
    def listener_callback(self, msg):
        """Handle incoming camera frames."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        boxes = self.detect_objects_in_frame(frame)

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            conf = box["confidence"]
            cls_id = box["class_id"]

            if conf < 0.5:
                continue

            label = "qrcode" if cls_id == 1 else "barcode"
            color = (0, 255, 0) if label == "qrcode" else (255, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show detection results
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


# ---------------- MAIN ENTRY ----------------
def main(args=None):
    rclpy.init(args=args)
    node = CameraQRCodeViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
