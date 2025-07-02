#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_srvs.srv import SetBool
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class ConveyorController(Node):
    def __init__(self):
        super().__init__('conveyor_controller')
        
        # Parameters
        self.declare_parameter('default_speed', 0.5, 
                              ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                description='Default conveyor speed in m/s'))
        self.declare_parameter('max_speed', 1.0,
                              ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                description='Maximum conveyor speed'))
        self.declare_parameter('acceleration', 0.1,
                              ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                description='Acceleration rate in m/s²'))
        
        # Get parameters
        self.default_speed = self.get_parameter('default_speed').value
        self.max_speed = self.get_parameter('max_speed').value
        self.acceleration = self.get_parameter('acceleration').value
        
        # Current state
        self.target_speed = 0.0
        self.current_speed = 0.0
        self.is_running = False
        
        # Publisher for velocity commands
        self.velocity_pub = self.create_publisher(
            Float64,
            '/conveyor_belt/belt_velocity_controller/commands',  # Updated topic
            10
        )
        
        # Service to start/stop conveyor
        self.service = self.create_service(
            SetBool,
            '/conveyor_belt/control',
            self.control_callback
        )
        
        # Timer for smooth acceleration
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        # For dynamic reconfigure (optional)
        self.get_logger().info("Conveyor controller ready")
    
    def control_callback(self, request, response):
        """Handle start/stop service requests"""
        self.is_running = request.data
        if self.is_running:
            self.target_speed = self.default_speed
            self.get_logger().info("Starting conveyor at speed: %.2f m/s" % self.target_speed)
        else:
            self.target_speed = 0.0
            self.get_logger().info("Stopping conveyor")
        
        response.success = True
        response.message = "Success" if request.data else "Stopped"
        return response
    
    def set_speed(self, speed):
        """Set target speed with safety limits"""
        self.target_speed = max(-self.max_speed, min(speed, self.max_speed))
        self.get_logger().info("Target speed set to: %.2f m/s" % self.target_speed)
    
    def control_loop(self):
        """Handle smooth acceleration and velocity commands"""
        if abs(self.current_speed - self.target_speed) > 0.01:
            # Apply acceleration
            if self.current_speed < self.target_speed:
                self.current_speed = min(self.current_speed + self.acceleration/20.0, self.target_speed)
            else:
                self.current_speed = max(self.current_speed - self.acceleration/20.0, self.target_speed)
            
            # Publish velocity command
            msg = Float64()
            msg.data = self.current_speed
            self.velocity_pub.publish(msg)
            
            # Debug logging
            if abs(self.current_speed - self.target_speed) < 0.02:
                self.get_logger().debug("Conveyor at target speed: %.2f m/s" % self.current_speed)

def main(args=None):
    rclpy.init(args=args)
    
    controller = ConveyorController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        # Send stop command before shutdown
        stop_msg = Float64()
        stop_msg.data = 0.0
        controller.velocity_pub.publish(stop_msg)
        controller.get_logger().info("Shutting down, stopping conveyor...")
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()