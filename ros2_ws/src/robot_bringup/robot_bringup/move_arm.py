
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()

        # Subscribe to your camera topic
        self.subscription = self.create_subscription(
            Image,
            '/ur5/tcp/image_raw',   # <-- matches your Gazebo plugin
            self.image_callback,
            10
        )

        self.get_logger().info("Camera Viewer Node Started. Waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS image → OpenCV (BGR format)
            frame = self.bridge.imgmsg_to_cv2(msg)

            # Display the frame
            cv2.imshow("UR5 Camera View", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
