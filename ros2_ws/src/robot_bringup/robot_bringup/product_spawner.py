#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SpawnEntity
from std_srvs.srv import Trigger
import random
import math
import os
from ament_index_python.packages import get_package_share_directory

class ProductSpawner(Node):
    def __init__(self):
        super().__init__('product_spawner')
        
        # Parameters
        self.declare_parameter('auto_spawn', False)
        self.declare_parameter('spawn_interval', 2.0)
        
        # Get model path
        pkg_path = get_package_share_directory('robot_bringup')
        self.model_path = os.path.join(pkg_path, 'urdf', 'product.sdf')
        
        # Verify model file exists
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Service client with persistent connection
        #self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.spawn_client = self.create_client(SpawnEntity, 'spawn_entity')

        self.connected = False
        self.connection_timer = self.create_timer(1.0, self.check_connection)
        
        # Spawn service
        self.srv = self.create_service(Trigger, 'spawn_product', self.spawn_callback)
        
        # Auto-spawn timer
        if self.get_parameter('auto_spawn').value:
            self.spawn_timer = self.create_timer(
                self.get_parameter('spawn_interval').value,
                self.auto_spawn
            )
    
    def check_connection(self):
        if not self.connected:
            if self.spawn_client.wait_for_service(timeout_sec=0.5):
                self.connected = True
                self.get_logger().info("Connected to spawn service!")
                self.destroy_timer(self.connection_timer)
            else:
                self.get_logger().warn("Waiting for spawn service...", throttle_duration_sec=5.0)
    
    async def spawn_callback(self, request, response):
        if not self.connected:
            response.success = False
            response.message = "Not connected to spawn service"
            return response
        
        success = await self.spawn_product()
        response.success = success
        response.message = "Spawned product" if success else "Failed to spawn"
        return response
    
    async def auto_spawn(self):
        if self.connected:
            await self.spawn_product()
    
    async def spawn_product(self):
        try:
            with open(self.model_path, 'r') as f:
                sdf = f.read()
            
            req = SpawnEntity.Request()
            req.entity_factory.sdf = sdf
            req.entity_factory.name = f"product_{random.randint(0,1000)}"
            req.entity_factory.relative_to = "world"
            
            # Set random pose
            req.entity_factory.pose.position.x = random.uniform(-0.2, 0.2)
            req.entity_factory.pose.position.y = 1.0
            req.entity_factory.pose.position.z = 0.2
            
            future = self.spawn_client.call_async(req)
            await future
            
            if future.result().success:
                self.get_logger().info(f"Spawned {req.entity_factory.name}")
                return True
            else:
                self.get_logger().error(f"Failed to spawn: {future.result().message}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error in spawn_product: {str(e)}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = ProductSpawner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()