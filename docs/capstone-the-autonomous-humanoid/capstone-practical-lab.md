---
id: capstone-practical-lab
title: 'Capstone — The Autonomous Humanoid | Chapter 4 — Practical Lab'
sidebar_label: 'Chapter 4 — Practical Lab'
sidebar_position: 4
---

# Chapter 4 — Practical Lab

## Capstone Integration: Building the Autonomous Humanoid

In this practical lab, we'll integrate all components developed in previous modules into a unified autonomous humanoid system. This will involve connecting the ROS 2 nervous system, digital twin simulation, AI-robot brain, and vision-language-action systems into a cohesive architecture that demonstrates advanced autonomous capabilities.

### Prerequisites

Before starting this lab, ensure you have:
- Completed all previous modules (Modules 1-4)
- NVIDIA GPU with CUDA support (RTX 4090 or equivalent recommended)
- Ubuntu 22.04 with ROS 2 Humble
- All Isaac ROS packages installed
- Isaac Sim and Isaac Lab installed
- Access to a humanoid robot model or simulation environment
- All packages from previous modules properly built

## Lab 1: System Architecture Implementation

### Creating the Integration Framework

1. **Create the main integration package**:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python integration_pkg --dependencies rclpy sensor_msgs geometry_msgs std_msgs nav_msgs visualization_msgs tf2_ros tf2_geometry_msgs message_filters
```

2. **Create the central integration node** (`~/ros2_ws/src/integration_pkg/integration_pkg/autonomous_humanoid.py`):
```python
#!/usr/bin/env python3
"""
Main integration node for the autonomous humanoid system
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from std_msgs.msg import String, Bool, Float32
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading
import time
import numpy as np
from collections import deque

class AutonomousHumanoidNode(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid')
        
        # System state
        self.system_state = 'idle'  # idle, active, emergency_stop
        self.emergency_stop_active = False
        
        # Subsystem states
        self.perception_ready = False
        self.ai_brain_ready = False
        self.vla_ready = False
        self.navigation_ready = False
        
        # Data storage
        self.robot_pose = None
        self.odom_data = None
        self.camera_data = None
        self.laser_data = None
        self.joint_states = None
        self.imu_data = None
        
        # Integration state
        self.current_task = None
        self.task_queue = deque()
        self.last_command_time = time.time()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        # Subscribers for subsystem outputs
        self.perception_sub = self.create_subscription(String, '/perception/status', self.perception_status_callback, 10)
        self.ai_brain_sub = self.create_subscription(String, '/ai_brain/status', self.ai_brain_status_callback, 10)
        self.vla_sub = self.create_subscription(String, '/vla/status', self.vla_status_callback, 10)
        self.navigation_sub = self.create_subscription(String, '/navigation/status', self.navigation_status_callback, 10)
        
        # Subscribers for high-level commands
        self.command_sub = self.create_subscription(String, '/command', self.command_callback, 10)
        
        # Timer for system monitoring
        self.monitor_timer = self.create_timer(0.1, self.system_monitor)
        
        # Safety timer
        self.safety_timer = self.create_timer(0.05, self.safety_check)
        
        # Control loop timer
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Autonomous Humanoid Integration Node started')

    def odom_callback(self, msg):
        """Handle odometry data"""
        self.odom_data = msg
        self.robot_pose = msg.pose.pose

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def joint_callback(self, msg):
        """Handle joint state data"""
        self.joint_states = msg

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def perception_status_callback(self, msg):
        """Handle perception system status"""
        if msg.data == "ready":
            self.perception_ready = True
        elif msg.data == "not_ready":
            self.perception_ready = False
        self.get_logger().info(f'Perception status: {msg.data}')

    def ai_brain_status_callback(self, msg):
        """Handle AI brain system status"""
        if msg.data == "ready":
            self.ai_brain_ready = True
        elif msg.data == "not_ready":
            self.ai_brain_ready = False
        self.get_logger().info(f'AI Brain status: {msg.data}')

    def vla_status_callback(self, msg):
        """Handle VLA system status"""
        if msg.data == "ready":
            self.vla_ready = True
        elif msg.data == "not_ready":
            self.vla_ready = False
        self.get_logger().info(f'VLA status: {msg.data}')

    def navigation_status_callback(self, msg):
        """Handle navigation system status"""
        if msg.data == "ready":
            self.navigation_ready = True
        elif msg.data == "not_ready":
            self.navigation_ready = False
        self.get_logger().info(f'Navigation status: {msg.data}')

    def command_callback(self, msg):
        """Handle high-level commands"""
        command = msg.data.lower()
        
        if command.startswith('navigate to'):
            # Extract destination from command
            destination = command.replace('navigate to', '').strip()
            self.get_logger().info(f'Received navigation command: {destination}')
            
            # In a real system, this would be more sophisticated
            # For now, we'll just navigate to a predefined location
            self.queue_navigation_task(destination)
            
        elif command == 'emergency stop':
            self.activate_emergency_stop()
            
        elif command == 'resume':
            self.deactivate_emergency_stop()
            
        else:
            self.get_logger().info(f'Unknown command: {command}')

    def queue_navigation_task(self, destination):
        """Queue a navigation task"""
        task = {
            'type': 'navigation',
            'destination': destination,
            'status': 'pending'
        }
        self.task_queue.append(task)
        self.get_logger().info(f'Queued navigation task to: {destination}')

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.system_state = 'emergency_stop'
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        
        # Send stop command to all actuators
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        # Publish emergency stop message
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        self.system_state = 'idle'
        self.get_logger().info('Emergency stop deactivated')

    def system_monitor(self):
        """Monitor the overall system status"""
        ready_subsystems = sum([
            self.perception_ready,
            self.ai_brain_ready,
            self.vla_ready,
            self.navigation_ready
        ])
        
        status_msg = String()
        status_msg.data = f"State: {self.system_state} | Ready subsystems: {ready_subsystems}/4 | Tasks: {len(self.task_queue)}"
        self.status_pub.publish(status_msg)
        
        # Log system status periodically
        if int(time.time()) % 5 == 0:  # Log every 5 seconds
            self.get_logger().info(f"System Status - State: {self.system_state}, Subsystems Ready: {ready_subsystems}/4")

    def safety_check(self):
        """Perform safety checks"""
        if self.emergency_stop_active:
            # Already in emergency stop state, ensure safety
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            return
        
        # Check for safety violations
        if self.laser_data:
            # Check for obstacles in front of robot
            front_scans = self.laser_data.ranges[:10] + self.laser_data.ranges[-10:]
            min_distance = min([d for d in front_scans if not np.isnan(d)], default=float('inf'))
            
            if 0 < min_distance < 0.3:  # Obstacle within 30cm
                self.get_logger().warn(f'Safety violation: Obstacle at {min_distance:.2f}m')
                # Implement safety response
                self.activate_emergency_stop()

    def control_loop(self):
        """Main control loop"""
        if self.emergency_stop_active:
            return
            
        if self.task_queue and self.system_state != 'active':
            # Process next task in queue
            task = self.task_queue.popleft()
            self.current_task = task
            self.system_state = 'active'
            
            if task['type'] == 'navigation':
                self.execute_navigation_task(task)
                
        elif not self.task_queue:
            # No tasks, return to idle
            self.system_state = 'idle'
            self.current_task = None

    def execute_navigation_task(self, task):
        """Execute a navigation task"""
        destination = task['destination']
        self.get_logger().info(f'Executing navigation to: {destination}')
        
        # In a real system, this would send a goal to the navigation stack
        # For this example, we'll just move forward for demonstration
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # Move forward at 0.2 m/s
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Simulate task completion after 10 seconds
        def complete_task():
            time.sleep(10)
            if self.current_task == task:
                self.current_task = None
                self.system_state = 'idle'
                self.get_logger().info('Navigation task completed')
        
        task_thread = threading.Thread(target=complete_task, daemon=True)
        task_thread.start()

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousHumanoidNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down autonomous humanoid system')
    finally:
        # Ensure safety on shutdown
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. **Update the setup.py file**:
```python
from setuptools import setup
import os
from glob import glob

package_name = 'integration_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Integration package for autonomous humanoid system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'autonomous_humanoid = integration_pkg.autonomous_humanoid:main',
        ],
    },
)
```

4. **Build the integration package**:
```bash
cd ~/ros2_ws
colcon build --packages-select integration_pkg
source ~/ros2_ws/install/setup.bash
```

## Lab 2: Subsystem Status Monitoring

### Creating a System Status Dashboard

1. **Create a system status node** (`~/ros2_ws/src/integration_pkg/integration_pkg/system_status.py`):
```python
#!/usr/bin/env python3
"""
System status monitoring node for the autonomous humanoid
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import time
import threading

class SystemStatusNode(Node):
    def __init__(self):
        super().__init__('system_status')
        
        # System status storage
        self.subsystem_statuses = {
            'perception': {'status': 'unknown', 'last_update': 0},
            'ai_brain': {'status': 'unknown', 'last_update': 0},
            'vla': {'status': 'unknown', 'last_update': 0},
            'navigation': {'status': 'unknown', 'last_update': 0}
        }
        
        self.emergency_stop_active = False
        self.robot_pose = None
        self.joint_states = None
        self.last_command_time = time.time()
        
        # Publishers
        self.status_viz_pub = self.create_publisher(MarkerArray, '/system_status_viz', 10)
        self.performance_pub = self.create_publisher(String, '/performance_summary', 10)
        
        # Subscribers
        self.perception_sub = self.create_subscription(String, '/perception/status', self.perception_status_callback, 10)
        self.ai_brain_sub = self.create_subscription(String, '/ai_brain/status', self.ai_brain_status_callback, 10)
        self.vla_sub = self.create_subscription(String, '/vla/status', self.vla_status_callback, 10)
        self.navigation_sub = self.create_subscription(String, '/navigation/status', self.navigation_status_callback, 10)
        self.emergency_stop_sub = self.create_subscription(Bool, '/emergency_stop', self.emergency_stop_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # Timer for status visualization
        self.viz_timer = self.create_timer(1.0, self.publish_status_viz)
        
        # Timer for performance summary
        self.perf_timer = self.create_timer(5.0, self.publish_performance_summary)
        
        self.get_logger().info('System Status Node started')

    def perception_status_callback(self, msg):
        self.subsystem_statuses['perception']['status'] = msg.data
        self.subsystem_statuses['perception']['last_update'] = time.time()

    def ai_brain_status_callback(self, msg):
        self.subsystem_statuses['ai_brain']['status'] = msg.data
        self.subsystem_statuses['ai_brain']['last_update'] = time.time()

    def vla_status_callback(self, msg):
        self.subsystem_statuses['vla']['status'] = msg.data
        self.subsystem_statuses['vla']['last_update'] = time.time()

    def navigation_status_callback(self, msg):
        self.subsystem_statuses['navigation']['status'] = msg.data
        self.subsystem_statuses['navigation']['last_update'] = time.time()

    def emergency_stop_callback(self, msg):
        self.emergency_stop_active = msg.data

    def odom_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def joint_callback(self, msg):
        self.joint_states = msg

    def publish_status_viz(self):
        """Publish visualization markers for system status"""
        marker_array = MarkerArray()
        
        # Create a text marker for each subsystem
        y_offset = 0
        for i, (subsystem, status_info) in enumerate(self.subsystem_statuses.items()):
            marker = Marker()
            marker.header.frame_id = "map"  # Use appropriate frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "system_status"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # Position markers in a column
            marker.pose.position.x = -2.0
            marker.pose.position.y = y_offset
            marker.pose.position.z = 1.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.z = 0.2  # Text size
            
            # Color based on status
            if status_info['status'] == 'ready':
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            elif status_info['status'] == 'not_ready':
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            
            marker.text = f"{subsystem}: {status_info['status']}"
            marker_array.markers.append(marker)
            
            y_offset -= 0.3  # Space between markers
        
        # Add emergency stop indicator
        em_marker = Marker()
        em_marker.header.frame_id = "map"
        em_marker.header.stamp = self.get_clock().now().to_msg()
        em_marker.ns = "system_status"
        em_marker.id = len(self.subsystem_statuses)
        em_marker.type = Marker.TEXT_VIEW_FACING
        em_marker.action = Marker.ADD
        
        em_marker.pose.position.x = -2.0
        em_marker.pose.position.y = y_offset
        em_marker.pose.position.z = 1.0
        em_marker.pose.orientation.w = 1.0
        
        em_marker.scale.z = 0.2
        em_marker.color.r = 1.0 if self.emergency_stop_active else 0.5
        em_marker.color.g = 0.0 if self.emergency_stop_active else 0.5
        em_marker.color.b = 0.0 if self.emergency_stop_active else 0.5
        em_marker.color.a = 1.0
        
        em_marker.text = f"EMERGENCY: {'ACTIVE' if self.emergency_stop_active else 'INACTIVE'}"
        marker_array.markers.append(em_marker)
        
        self.status_viz_pub.publish(marker_array)

    def publish_performance_summary(self):
        """Publish performance summary"""
        summary_msg = String()
        
        # Calculate subsystem health
        ready_count = sum(1 for status in self.subsystem_statuses.values() 
                         if status['status'] == 'ready')
        total_count = len(self.subsystem_statuses)
        
        # Calculate system uptime
        uptime_minutes = int(time.time() / 60)
        
        summary_msg.data = (f"System Health: {ready_count}/{total_count} subsystems ready, "
                           f"Uptime: {uptime_minutes} min, "
                           f"Emergency: {'ACTIVE' if self.emergency_stop_active else 'OK'}")
        
        self.performance_pub.publish(summary_msg)
        self.get_logger().info(summary_msg.data)

def main(args=None):
    rclpy.init(args=args)
    node = SystemStatusNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 3: Command and Control Interface

### Creating a High-Level Command Interface

1. **Create a command interface node** (`~/ros2_ws/src/integration_pkg/integration_pkg/command_interface.py`):
```python
#!/usr/bin/env python3
"""
Command interface for the autonomous humanoid system
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time

class CommandInterfaceNode(Node):
    def __init__(self):
        super().__init__('command_interface')
        
        # Publishers
        self.command_pub = self.create_publisher(String, '/command', 10)
        self.status_sub = self.create_subscription(String, '/system_status', self.status_callback, 10)
        
        # Store recent status
        self.last_status = "Unknown"
        
        self.get_logger().info('Command Interface Node started')
        self.get_logger().info('Ready to accept commands. Type "help" for command list.')

    def status_callback(self, msg):
        """Store the latest system status"""
        self.last_status = msg.data

    def process_command(self, command):
        """Process and send a command to the humanoid system"""
        if command.lower() in ['quit', 'exit']:
            self.get_logger().info('Command interface shutting down...')
            return False
            
        elif command.lower() == 'help':
            self.show_help()
            
        elif command.lower() == 'status':
            self.get_logger().info(f'Current system status: {self.last_status}')
            
        elif command.lower().startswith('navigate to '):
            # Process navigation command
            destination = command[12:]  # Remove 'navigate to ' prefix
            if destination.strip():
                cmd_msg = String()
                cmd_msg.data = command.lower()
                self.command_pub.publish(cmd_msg)
                self.get_logger().info(f'Sent navigation command: {command}')
            else:
                self.get_logger().warn('Please specify a destination for navigation.')
                
        elif command.lower() == 'emergency stop':
            cmd_msg = String()
            cmd_msg.data = 'emergency stop'
            self.command_pub.publish(cmd_msg)
            self.get_logger().warn('Sent emergency stop command!')
            
        elif command.lower() == 'resume':
            cmd_msg = String()
            cmd_msg.data = 'resume'
            self.command_pub.publish(cmd_msg)
            self.get_logger().info('Sent resume command.')
            
        else:
            self.get_logger().warn(f'Unknown command: {command}. Type "help" for command list.')
            
        return True

    def show_help(self):
        """Display available commands"""
        help_text = """
Available Commands:
  navigate to [location] - Navigate to specified location
  emergency stop         - Activate emergency stop
  resume                 - Resume operation after emergency stop
  status                 - Show current system status
  help                   - Show this help message
  quit/exit              - Exit the command interface
        """
        self.get_logger().info(help_text)

def main(args=None):
    rclpy.init(args=args)
    node = CommandInterfaceNode()
    
    try:
        # Run command loop in a separate thread to allow ROS spinning
        import threading
        command_thread = threading.Thread(target=command_loop, args=(node,), daemon=True)
        command_thread.start()
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down command interface')
    finally:
        node.destroy_node()
        rclpy.shutdown()

def command_loop(node):
    """Command input loop"""
    while rclpy.ok():
        try:
            command = input("Enter command: ").strip()
            if not node.process_command(command):
                break
        except EOFError:
            break
        except Exception as e:
            node.get_logger().error(f'Error processing command: {e}')

if __name__ == '__main__':
    main()
```

## Lab 4: Creating a Launch File for Integrated System

### Creating a Comprehensive Launch File

1. **Create a launch directory and file** (`~/ros2_ws/src/integration_pkg/launch/integrated_humanoid.launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart, OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_dashboard = LaunchConfiguration('enable_dashboard', default='true')
    
    # Main integration node
    autonomous_humanoid_node = Node(
        package='integration_pkg',
        executable='autonomous_humanoid',
        name='autonomous_humanoid',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # System status node
    system_status_node = Node(
        package='integration_pkg',
        executable='system_status',
        name='system_status',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Command interface node (only if needed for interactive use)
    command_interface_node = Node(
        package='integration_pkg',
        executable='command_interface',
        name='command_interface',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen',
        condition=launch.conditions.IfCondition(enable_dashboard)
    )
    
    # Placeholder for perception system (would be real perception package in actual implementation)
    perception_node = Node(
        package='dummy_perception_pkg',  # Replace with real perception package
        executable='perception_node',
        name='perception_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Placeholder for AI brain (would be real AI package in actual implementation)
    ai_brain_node = Node(
        package='dummy_ai_pkg',  # Replace with real AI package
        executable='ai_brain_node',
        name='ai_brain_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Placeholder for VLA system (would be real VLA package in actual implementation)
    vla_node = Node(
        package='dummy_vla_pkg',  # Replace with real VLA package
        executable='vla_node',
        name='vla_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Placeholder for navigation system (would be real navigation package in actual implementation)
    navigation_node = Node(
        package='nav2_bringup',  # Using nav2 as example
        executable='navigation_node',
        name='navigation_node',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Return the launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'enable_dashboard',
            default_value='true',
            description='Enable command dashboard interface'
        ),
        autonomous_humanoid_node,
        system_status_node,
        command_interface_node,
        # These would be connected to real subsystem implementations:
        # perception_node,
        # ai_brain_node,
        # vla_node,
        # navigation_node,
    ])
```

## Lab 5: Performance Monitoring and Diagnostics

### Creating a Performance Monitoring Node

1. **Create a performance monitoring node** (`~/ros2_ws/src/integration_pkg/integration_pkg/performance_monitor.py`):
```python
#!/usr/bin/env python3
"""
Performance monitoring for the autonomous humanoid system
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, LaserScan
import time
from collections import deque
import threading

class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Performance metrics
        self.fps_metrics = {
            'camera': deque(maxlen=100),
            'laser': deque(maxlen=100),
            'control': deque(maxlen=100)
        }
        self.latency_metrics = {
            'perception_to_action': deque(maxlen=50),
            'command_response': deque(maxlen=50)
        }
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Timestamp tracking
        self.last_camera_time = None
        self.last_laser_time = None
        self.last_command_time = None
        
        # Publishers
        self.fps_pub = self.create_publisher(Float32, '/performance/fps', 10)
        self.latency_pub = self.create_publisher(Float32, '/performance/latency', 10)
        self.status_pub = self.create_publisher(String, '/performance/status', 10)
        
        # Subscribers for timing
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 1)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 1)
        self.command_sub = self.create_subscription(String, '/command', self.command_callback, 10)
        
        # Timer for publishing metrics
        self.metrics_timer = self.create_timer(1.0, self.publish_metrics)
        self.status_timer = self.create_timer(2.0, self.publish_status)
        
        # Start system resource monitoring
        self.resource_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        self.resource_thread.start()
        
        self.get_logger().info('Performance Monitor Node started')

    def camera_callback(self, msg):
        """Handle camera message timing"""
        if self.last_camera_time:
            dt = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - \
                 (self.last_camera_time.sec + self.last_camera_time.nanosec * 1e-9)
            fps = 1.0 / dt if dt > 0 else 0
            self.fps_metrics['camera'].append(fps)
        self.last_camera_time = msg.header.stamp

    def laser_callback(self, msg):
        """Handle laser scan message timing"""
        if self.last_laser_time:
            dt = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - \
                 (self.last_laser_time.sec + self.last_laser_time.nanosec * 1e-9)
            fps = 1.0 / dt if dt > 0 else 0
            self.fps_metrics['laser'].append(fps)
        self.last_laser_time = msg.header.stamp

    def command_callback(self, msg):
        """Handle command timing"""
        if self.last_command_time:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            last_time = self.last_command_time.sec + self.last_command_time.nanosec * 1e-9
            latency = current_time - last_time
            self.latency_metrics['command_response'].append(latency)
        self.last_command_time = self.get_clock().now().to_msg()

    def monitor_resources(self):
        """Monitor CPU and memory usage"""
        import psutil
        
        while rclpy.ok():
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)

    def publish_metrics(self):
        """Publish performance metrics"""
        # Average FPS for camera
        if self.fps_metrics['camera']:
            avg_camera_fps = sum(self.fps_metrics['camera']) / len(self.fps_metrics['camera'])
            fps_msg = Float32()
            fps_msg.data = avg_camera_fps
            self.fps_pub.publish(fps_msg)

    def publish_status(self):
        """Publish system performance status"""
        status_parts = []
        
        # FPS status
        if self.fps_metrics['camera']:
            avg_camera_fps = sum(self.fps_metrics['camera']) / len(self.fps_metrics['camera'])
            status_parts.append(f"Camera FPS: {avg_camera_fps:.1f}")
        
        # Latency status
        if self.latency_metrics['command_response']:
            avg_latency = sum(self.latency_metrics['command_response']) / len(self.latency_metrics['command_response'])
            status_parts.append(f"Command Latency: {avg_latency:.3f}s")
        
        # Resource usage
        if self.cpu_usage:
            avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
            status_parts.append(f"CPU: {avg_cpu:.1f}%")
        
        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            status_parts.append(f"Memory: {avg_memory:.1f}%")
        
        status_msg = String()
        status_msg.data = " | ".join(status_parts)
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(f"Performance: {status_msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 6: Testing the Integrated System

### Creating a Basic Integration Test

1. **Create an integration test script** (`~/ros2_ws/src/integration_pkg/integration_pkg/integration_test.py`):
```python
#!/usr/bin/env python3
"""
Integration test for the autonomous humanoid system
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time

class IntegrationTestNode(Node):
    def __init__(self):
        super().__init__('integration_test')
        
        # Publishers for sending test commands
        self.command_pub = self.create_publisher(String, '/command', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers for monitoring system responses
        self.status_sub = self.create_subscription(String, '/system_status', self.status_callback, 10)
        self.emergency_sub = self.create_subscription(Bool, '/emergency_stop', self.emergency_callback, 10)
        
        # Test state
        self.test_phase = 0
        self.test_results = {}
        self.system_ready = False
        self.emergency_triggered = False
        
        # Start test sequence
        self.test_timer = self.create_timer(2.0, self.run_test_sequence)
        
        self.get_logger().info('Integration Test Node started')

    def status_callback(self, msg):
        """Monitor system status"""
        if "Ready subsystems: 4/4" in msg.data:
            self.system_ready = True
            self.get_logger().info('All subsystems are ready')

    def emergency_callback(self, msg):
        """Detect emergency stop"""
        if msg.data:
            self.emergency_triggered = True
            self.get_logger().warn('Emergency stop was triggered during test')

    def run_test_sequence(self):
        """Run the integration test sequence"""
        if self.test_phase == 0:
            self.get_logger().info('Test Phase 0: Verifying system readiness')
            if self.system_ready:
                self.test_results['system_readiness'] = True
                self.test_phase = 1
                self.get_logger().info('System readiness verified, moving to next phase')
            else:
                self.test_results['system_readiness'] = False
                self.get_logger().warn('System not ready, continuing anyway')
                self.test_phase = 1
                
        elif self.test_phase == 1:
            self.get_logger().info('Test Phase 1: Sending navigation command')
            cmd_msg = String()
            cmd_msg.data = 'navigate to kitchen'
            self.command_pub.publish(cmd_msg)
            self.test_results['navigation_command'] = True
            self.test_phase = 2
            
        elif self.test_phase == 2:
            self.get_logger.info('Test Phase 2: Monitoring navigation execution')
            # Check if emergency was triggered during navigation
            self.test_results['navigation_safety'] = not self.emergency_triggered
            self.test_phase = 3
            
        elif self.test_phase == 3:
            self.get_logger.info('Test Phase 3: Testing emergency stop')
            # Trigger emergency stop
            cmd_msg = String()
            cmd_msg.data = 'emergency stop'
            self.command_pub.publish(cmd_msg)
            self.test_results['emergency_triggered'] = True
            self.test_phase = 4
            
        elif self.test_phase == 4:
            self.get_logger.info('Test Phase 4: Verifying emergency stop response')
            self.test_results['emergency_response'] = self.emergency_triggered
            self.test_phase = 5
            
        elif self.test_phase == 5:
            self.get_logger.info('Test Phase 5: Resuming after emergency')
            cmd_msg = String()
            cmd_msg.data = 'resume'
            self.command_pub.publish(cmd_msg)
            self.test_phase = 6
            
        elif self.test_phase == 6:
            self.get_logger().info('Test Phase 6: Final system check')
            # Wait a bit and check final status
            self.test_results['final_state'] = True
            self.test_phase = 7
            self.report_test_results()
            
        elif self.test_phase == 7:
            # Test complete, stop timer
            self.test_timer.cancel()

    def report_test_results(self):
        """Report the results of the integration test"""
        self.get_logger().info('--- INTEGRATION TEST RESULTS ---')
        for test, result in self.test_results.items():
            status = 'PASS' if result else 'FAIL'
            self.get_logger().info(f'{test}: {status}')
        
        # Overall result
        all_passed = all(self.test_results.values())
        overall = 'PASS' if all_passed else 'FAIL'
        self.get_logger().info(f'Overall Integration Test: {overall}')
        self.get_logger().info('--- END TEST RESULTS ---')

def main(args=None):
    rclpy.init(args=args)
    node = IntegrationTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Integration test interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 7: Building and Running the Integrated System

### Final Steps to Build and Test

1. **Update the setup.py to include all new executables**:
```python
from setuptools import setup
import os
from glob import glob

package_name = 'integration_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Integration package for autonomous humanoid system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'autonomous_humanoid = integration_pkg.autonomous_humanoid:main',
            'system_status = integration_pkg.system_status:main',
            'command_interface = integration_pkg.command_interface:main',
            'performance_monitor = integration_pkg.performance_monitor:main',
            'integration_test = integration_pkg.integration_test:main',
        ],
    },
)
```

2. **Build the complete integration package**:
```bash
cd ~/ros2_ws
colcon build --packages-select integration_pkg
source ~/ros2_ws/install/setup.bash
```

3. **Create a simple test launch file to test individual components**:
```bash
# Terminal 1 - Start the main integration node
source ~/ros2_ws/install/setup.bash
ros2 run integration_pkg autonomous_humanoid

# Terminal 2 - Start the system status monitor
source ~/ros2_ws/install/setup.bash
ros2 run integration_pkg system_status

# Terminal 3 - Start the command interface
source ~/ros2_ws/install/setup.bash
ros2 run integration_pkg command_interface
```

4. **Run the integration test**:
```bash
# Run integration test after starting the system
source ~/ros2_ws/install/setup.bash
ros2 run integration_pkg integration_test
```

## Summary

In this practical lab, we've created a comprehensive integration framework for the autonomous humanoid system. We developed:

1. **Main Integration Node**: The central hub that coordinates all subsystems and manages the overall system state
2. **System Status Monitoring**: Real-time visualization and monitoring of subsystem statuses
3. **Command Interface**: Human-friendly interface for controlling the robot
4. **Performance Monitoring**: Metrics collection for system performance and health
5. **Integration Testing**: Automated testing framework to validate the integration

This integration framework demonstrates the key challenges of connecting multiple complex robotic subsystems:

- **Communication coordination**: Ensuring all subsystems can communicate effectively
- **Timing synchronization**: Managing different update rates across subsystems
- **Safety integration**: Implementing safety checks across all components
- **Performance optimization**: Monitoring and optimizing system performance
- **State management**: Tracking the state of complex integrated systems

The framework provides a solid foundation for connecting the ROS 2 nervous system, digital twin, AI brain, and VLA systems into a unified autonomous humanoid platform. Each component can be extended and customized based on specific requirements and available subsystem implementations.