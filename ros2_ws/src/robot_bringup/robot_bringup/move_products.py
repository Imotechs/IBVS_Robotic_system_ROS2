import rclpy
from rclpy.node import Node
from threading import Lock
import subprocess
import os
import random
import time
from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32MultiArray
from conveyorbelt_msgs.srv import ConveyorBeltControl
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Point

@dataclass
class Product:
    name: str
    spawn_time: float
    x: float
    y: float
    z: float
    is_active: bool = True

class ConveyorBeltNode(Node):
    def __init__(self):
        super().__init__('conveyor_belt_node')

        # Product SDF file
        self.urdf_path = os.path.join(
            get_package_share_directory('robot_bringup'),
            'urdf', 'product.sdf'
        )
        self.get_logger().info(f'Product SDF path: {self.urdf_path}')

        # Parameters
        self.prefix = self.declare_parameter('product_name_prefix', 'product').value
        self.spawn_interval = self.declare_parameter('spawn_interval', 50.0).value
        self.end_x = self.declare_parameter('belt_end_x', 5.0).value
        self.vel = self.declare_parameter('belt_velocity', 0.1).value
        self.last_spawn_time = time.time()
        # State
        self.counter = random.choice([0,50,200,500, 60, 30, 896])
        self.lock = Lock()
        self.belt_speed = self.declare_parameter('belt_speed', 20.0).value
        

        # Conveyor control client
        self.cli = self.create_client(ConveyorBeltControl, '/CONVEYORPOWER')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /CONVEYORPOWER service...')
        self.set_belt_speed(self.belt_speed)
        
        self.get_model_cli = None
        self.service_ready = False
        
        # Keep track of spawned products with their spawn time
        self.product_list = []  # {name: spawn_time}
        self.spawned_products = []  # List of product names for Gazebo queries

        # Timer for spawning
        self.create_timer(1.0, self.update_products)
        


    def spawn_product(self):
        self.counter += 1
        name = f"{self.prefix}_{self.counter:03d}"
        self.get_logger().info(f"Spawning {name}...")

        try:
            result = subprocess.Popen([
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', name,
                '-file', self.urdf_path,
                '-x', '0.65', '-y', '-5.0', '-z', '0.79'
            ])
            
            # Add to tracking immediately with spawn time
            with self.lock:
                product = Product(
                    name=name,
                    spawn_time=time.time(),
                    x=0.95,      # Initial X position
                    y=-5.0,     # Initial Y position  
                    z=0.79     # Initial Z position
                )
                self.product_list.append(product)
                self.spawned_products.append(name)
            
            self.get_logger().info(f"Added {name} to tracking. Total: {len(self.spawned_products)}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to spawn {name}: {e}")

    def set_belt_speed(self, speed):
        req = ConveyorBeltControl.Request()
        req.power = float(speed)
        future = self.cli.call_async(req)
        future.add_done_callback(self._belt_speed_done_cb)

    def _belt_speed_done_cb(self, fut):
        try:
            res = fut.result()
            if hasattr(res, 'success') and res.success:
                self.get_logger().info(f"Conveyor set to {self.belt_speed:.1f}% power.")
            else:
                self.get_logger().warn("Failed to set conveyor speed.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def update_products(self):
            current_time = time.time()
            if current_time - self.last_spawn_time > self.spawn_interval:
                self.spawn_product()
                self.last_spawn_time = current_time

           
                    
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