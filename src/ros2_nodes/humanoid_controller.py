# ROS 2 Humanoid Robot Controller Node

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class HumanoidController(Node):
    """
    A ROS 2 node that controls a humanoid robot by publishing joint commands
    and subscribing to sensor data.
    """
    
    def __init__(self):
        super().__init__('humanoid_controller')
        
        # Publishers for robot commands
        self.joint_cmd_publisher = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.status_subscriber = self.create_subscription(
            String,
            '/robot_status',
            self.status_callback,
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        # Robot state variables
        self.current_joint_states = JointState()
        self.robot_status = "idle"
        
        self.get_logger().info('Humanoid Controller node initialized')

    def joint_state_callback(self, msg):
        """Callback for receiving joint state messages"""
        self.current_joint_states = msg
        self.get_logger().debug(f'Received joint states for {len(msg.position)} joints')

    def status_callback(self, msg):
        """Callback for receiving robot status messages"""
        self.robot_status = msg.data
        self.get_logger().info(f'Robot status: {self.robot_status}')

    def control_loop(self):
        """Main control loop for the humanoid robot"""
        # Create joint command message
        joint_cmd_msg = JointState()
        joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
        joint_cmd_msg.header.frame_id = "base_link"
        
        # Define joint names for a typical humanoid (20 DOF example)
        joint_cmd_msg.name = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'torso_joint', 'neck_joint',
            'left_eyebrow_joint', 'right_eyebrow_joint',  # For expression
            'left_arm_supination_joint', 'right_arm_supination_joint',
            'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
        ]
        
        # Calculate desired joint positions based on current state and goals
        # For now, we'll just send neutral positions as an example
        neutral_positions = [0.0] * 20  # 20 joints
        joint_cmd_msg.position = neutral_positions
        
        # Add some dynamic movement based on robot status
        if self.robot_status == "walking":
            # Add walking gait pattern
            for i in range(len(joint_cmd_msg.position)):
                joint_cmd_msg.position[i] += 0.1 * np.sin(self.get_clock().now().nanoseconds / 1e9 + i)
        
        self.joint_cmd_publisher.publish(joint_cmd_msg)
        
        # Publish velocity command if needed
        if self.robot_status == "move_forward":
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = 0.2  # Move forward at 0.2 m/s
            cmd_vel_msg.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd_vel_msg)

def main(args=None):
    rclpy.init(args=args)
    
    humanoid_controller = HumanoidController()
    
    try:
        rclpy.spin(humanoid_controller)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_controller.get_logger().info('Shutting down Humanoid Controller')
        humanoid_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()