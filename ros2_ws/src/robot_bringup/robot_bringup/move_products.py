import rclpy
from rclpy.node import Node
from threading import Lock
import subprocess
import os
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from conveyorbelt_msgs.srv import ConveyorBeltControl  # custom service

class ConveyorBeltNode(Node):
    def __init__(self):
        super().__init__('conveyor_belt_node')

        # URDF file path
        self.urdf_path = os.path.join(
            get_package_share_directory('robot_bringup'),
            'urdf', 'product.urdf'
        )

        # Parameters
        self.prefix = self.declare_parameter('product_name_prefix', 'product').value
        self.interval = self.declare_parameter('spawn_interval', 20.0).value
        self.end_x = self.declare_parameter('belt_end_x', 5.0).value
        self.vel = self.declare_parameter('belt_velocity', 0.1).value

        # State
        self.counter = 0
        self.node_publishers = {}  # name: publisher
        self.lock = Lock()
        self.speed = self.declare_parameter('belt_speed', 40.0).value
          # Control the belt speed
        self.cli = self.create_client(ConveyorBeltControl, '/CONVEYORPOWER')
        while not self.cli.wait_for_service(timeout_sec=2.0):
             self.get_logger().info('Waiting for /CONVEYORPOWER service...')

        # Start spawning loop and movement
        self.create_timer(self.interval, self.spawn_product)
        self.set_belt_speed(self.speed)

    def spawn_product(self):
        self.counter += 1
        name = f"{self.prefix}_{self.counter:03d}"
        self.get_logger().info(f"Spawning {name}...")

        try:
            subprocess.Popen([
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', name,
                '-file', self.urdf_path,
                '-x', '0.8', '-y', '-5.0', '-z', '0.78'
            ])
        except Exception as e:
            self.get_logger().error(f"Failed to spawn {name}: {e}")
            return

        # Delay to allow spawning
        time.sleep(1.0)

      
    def set_belt_speed(self,speed):
        req = ConveyorBeltControl.Request()
        req.power = float(speed)

        future = self.cli.call_async(req)
        def done_cb(fut):
            try:
                res = fut.result()
                if hasattr(res, 'success') and res.success:
                    self.get_logger().info(f"Conveyor set to {speed:.1f}% power.")
                else:
                    self.get_logger().warn("Failed to set conveyor speed.")
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")

        future.add_done_callback(done_cb)
        
def main(args=None):
    rclpy.init(args=args)
    node = ConveyorBeltNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
