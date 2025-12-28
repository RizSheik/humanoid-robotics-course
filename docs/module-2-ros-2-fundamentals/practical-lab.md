# Module 2: Practical Lab - ROS 2 Communication Patterns for Robotics

## Lab Overview

This practical lab provides hands-on experience with ROS 2 communication patterns in a simulated humanoid robotics environment. Students will implement nodes using different communication patterns (topics, services, and actions) to coordinate robot subsystems, focusing on the specific needs of humanoid robots.

### Learning Objectives

After completing this lab, students will be able to:
1. Create and configure ROS 2 nodes using topics for asynchronous communication
2. Implement service servers and clients for synchronous operations
3. Develop action servers and clients for long-running tasks
4. Debug and optimize ROS 2 communication in simulated environments
5. Design communication architectures for complex robotic systems

### Required Software/Tools

- ROS 2 Humble Hawksbill or later
- Gazebo Harmonic or Isaac Sim
- Python 3.11+ or C++17
- Basic understanding of robotics kinematics and control

### Lab Duration

This lab is designed for 15-18 hours of work, typically spread over 3 weeks.

## Lab 1: Topic-Based Communication for Sensor Data

### Objective
Implement a publisher-subscriber system for sharing sensor data from multiple sensors on a humanoid robot simulation.

### Setup
1. Launch a humanoid robot model in Gazebo with multiple sensors:
   - Joint position sensors
   - IMU (Inertial Measurement Unit)
   - Camera feeds
   - Force/torque sensors

### Implementation Steps
1. Create a sensor fusion node that subscribes to multiple sensor topics
2. Implement appropriate QoS settings for different sensor types
3. Publish combined sensor data with proper timestamps and frame IDs
4. Visualize sensor data using RViz2

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image, WrenchStamped
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
from message_filters import ApproximateTimeSync, Subscriber

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Set up QoS profiles for different sensors
        # High-frequency sensors: Joint states, IMU
        sensor_qos = rclpy.qos.QoSProfile(
            depth=5,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        # Lower frequency sensors: Images
        image_qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        # Create subscribers for different sensor types
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, sensor_qos)
        
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, sensor_qos)
        
        self.force_sub = self.create_subscription(
            WrenchStamped, 'left_foot/force_torque', self.force_callback, sensor_qos)
        
        # Publisher for fused sensor data
        self.fused_pub = self.create_publisher(PoseStamped, 'robot_pose', 10)
        
        # Store sensor data
        self.joint_positions = {}
        self.imu_data = None
        self.force_data = None
        
        # Timer for sensor fusion
        self.fusion_timer = self.create_timer(0.01, self.fusion_callback)  # 100Hz

    def joint_callback(self, msg):
        """Process joint state messages."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
        self.get_logger().debug(f'Updated joint positions for {len(msg.name)} joints')

    def imu_callback(self, msg):
        """Process IMU messages."""
        self.imu_data = msg
        self.get_logger().debug('Updated IMU data')

    def force_callback(self, msg):
        """Process force/torque messages."""
        self.force_data = msg
        self.get_logger().debug('Updated force/torque data')

    def fusion_callback(self):
        """Perform sensor fusion to estimate robot pose."""
        if self.imu_data is not None:
            # Create a PoseStamped message based on sensor fusion
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'odom'
            
            # For simplicity, use orientation from IMU and set position to origin
            # In a real implementation, this would integrate multiple sensors
            pose_msg.pose.orientation = self.imu_data.orientation
            
            # Publish fused data
            self.fused_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion_node = SensorFusionNode()
    
    try:
        rclpy.spin(sensor_fusion_node)
    except KeyboardInterrupt:
        sensor_fusion_node.get_logger().info('Interrupted by user')
    finally:
        sensor_fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Monitor data rates and latencies for different sensor types
- Evaluate the impact of QoS settings on communication reliability
- Analyze the computational overhead of the fusion process

## Lab 2: Service-Based Configuration and Control

### Objective
Implement a service-based system for configuring and controlling humanoid robot parameters, such as joint limits, PID gains, or operational modes.

### Setup
1. Use the same humanoid robot simulation from Lab 1
2. Create a configuration service that allows changing robot parameters
3. Implement an emergency stop service

### Implementation Steps
1. Define custom service messages for robot configuration
2. Create a service server that handles configuration requests
3. Implement a service client for configuration changes
4. Add an emergency stop service with appropriate safety measures

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import JointState
from control_msgs.srv import JointTrajectoryControllerState

class RobotConfigurationService(Node):
    def __init__(self):
        super().__init__('robot_configuration_service')
        
        # Service for joint limit configuration
        self.joint_limit_service = self.create_service(
            JointTrajectoryControllerState,  # Using this as an example
            'set_joint_limits',
            self.set_joint_limits_callback)
        
        # Emergency stop service
        self.emergency_stop_service = self.create_service(
            Trigger,
            'emergency_stop',
            self.emergency_stop_callback)
        
        # Mode change service
        self.mode_change_service = self.create_service(
            SetBool,
            'set_robot_mode',
            self.set_robot_mode_callback)
        
        # Robot state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Current robot state
        self.current_mode = "idle"  # idle, active, error
        self.emergency_stopped = False
        
        # Define joint limits
        self.joint_limits = {
            'hip_joint': (-1.57, 1.57),  # min, max in radians
            'knee_joint': (0, 2.5),
            'ankle_joint': (-0.78, 0.78)
        }

    def set_joint_limits_callback(self, request, response):
        """Handle joint limit configuration requests."""
        try:
            # Validate request
            if not self.validate_limits_request(request):
                response.success = False
                response.error_string = "Invalid joint limit request"
                return response
            
            # Update joint limits
            self.joint_limits.update(request.joint_names)
            self.get_logger().info(f'Updated joint limits: {self.joint_limits}')
            
            response.success = True
            response.error_string = "Joint limits updated successfully"
            
        except Exception as e:
            self.get_logger().error(f'Error setting joint limits: {str(e)}')
            response.success = False
            response.error_string = f'Error: {str(e)}'
        
        return response

    def validate_limits_request(self, request):
        """Validate joint limit request."""
        # In a real implementation, this would validate:
        # - All joint names are valid
        # - Limits are within physical constraints
        # - No conflicting constraints
        return True

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop requests."""
        if self.emergency_stopped:
            response.success = True
            response.message = "Robot already in emergency stop"
            return response
        
        try:
            # Execute emergency stop procedure
            self.execute_emergency_stop()
            
            response.success = True
            response.message = "Emergency stop executed successfully"
            self.get_logger().warn("EMERGENCY STOP ACTIVATED")
            
        except Exception as e:
            self.get_logger().error(f'Error during emergency stop: {str(e)}')
            response.success = False
            response.message = f'Emergency stop failed: {str(e)}'
        
        return response

    def execute_emergency_stop(self):
        """Execute the actual emergency stop procedure."""
        # Stop all joint movements
        stop_msg = JointState()
        stop_msg.name = list(self.joint_limits.keys())
        stop_msg.position = [0.0] * len(stop_msg.name)  # Stop at current position
        self.joint_pub.publish(stop_msg)
        
        self.emergency_stopped = True
        self.current_mode = "error"

    def set_robot_mode_callback(self, request, response):
        """Handle robot mode change requests."""
        new_mode = "active" if request.data else "idle"
        
        # Validate mode change
        if self.emergency_stopped and new_mode == "active":
            response.success = False
            response.message = "Cannot activate robot in emergency stop state"
            return response
        
        # Execute mode change
        self.current_mode = new_mode
        response.success = True
        response.message = f"Robot mode changed to {new_mode}"
        
        self.get_logger().info(f'Robot mode changed to: {new_mode}')
        return response

def main(args=None):
    rclpy.init(args=args)
    config_service = RobotConfigurationService()
    
    try:
        rclpy.spin(config_service)
    except KeyboardInterrupt:
        config_service.get_logger().info('Interrupted by user')
    finally:
        config_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Measure service response times for different operations
- Evaluate the reliability of service-based communication
- Test error handling and recovery mechanisms

## Lab 3: Action-Based Task Execution

### Objective
Implement action-based communication for long-running humanoid robot tasks such as walking, manipulation, or complex motion patterns.

### Setup
1. Create a humanoid robot simulation capable of walking or manipulation
2. Design an action interface for commanding complex tasks
3. Implement both action server and client

### Implementation Steps
1. Define an action interface for humanoid motion tasks
2. Implement an action server that executes motions with feedback
3. Create an action client that sends motion commands and monitors progress
4. Handle cancellation and preemption of ongoing tasks

### Code Template
```python
#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class HumanoidMotionActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_motion_action_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'humanoid_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        
        # Robot state publisher
        self.joint_pub = self.create_publisher(
            JointTrajectoryPoint, 'simulated_joint_commands', 10)
        
        # Balance control publisher
        self.balance_pub = self.create_publisher(
            JointTrajectoryPoint, 'balance_control_commands', 10)

    def goal_callback(self, goal_request):
        """Accept or reject goal requests."""
        # Validate trajectory
        trajectory = goal_request.trajectory
        
        if len(trajectory.points) == 0:
            self.get_logger().warn('Goal rejected: No trajectory points')
            return GoalResponse.REJECT
        
        # Check if trajectory joints match robot's joints
        required_joints = set(trajectory.joint_names)
        robot_joints = set(self.get_robot_joint_names())
        
        if not required_joints.issubset(robot_joints):
            self.get_logger().warn(f'Goal rejected: Unknown joints: {required_joints - robot_joints}')
            return GoalResponse.REJECT
        
        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept cancellation requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the motion goal."""
        self.get_logger().info('Executing motion goal')
        
        # Initialize feedback
        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = goal_handle.request.trajectory.joint_names
        result_msg = FollowJointTrajectory.Result()
        
        # Get trajectory from goal
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points
        
        # Start balance control for humanoid walking
        self.start_balance_control()
        
        try:
            # Execute trajectory point by point
            for i, point in enumerate(points):
                # Check for cancellation
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Motion execution canceled')
                    result_msg.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    goal_handle.canceled()
                    return result_msg
                
                # Update feedback
                feedback_msg.actual.positions = [0.0] * len(joint_names)
                feedback_msg.desired = point
                feedback_msg.error.positions = [0.0] * len(joint_names)
                
                # Calculate progress percentage
                progress = (i + 1) / len(points) * 100.0
                feedback_msg.progress = progress
                
                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)
                
                # Execute the motion
                self.execute_trajectory_point(point, joint_names)
                
                # Log progress
                self.get_logger().info(f'Progress: {progress:.1f}% ({i+1}/{len(points)})')
                
                # Small delay to simulate movement
                time.sleep(0.1)
        finally:
            # Stop balance control regardless of outcome
            self.stop_balance_control()
        
        # Check for final cancellation
        if goal_handle.is_cancel_requested:
            result_msg.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.canceled()
            return result_msg
        
        # Set result and succeed
        result_msg.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        self.get_logger().info('Motion execution completed successfully')
        
        return result_msg

    def execute_trajectory_point(self, point, joint_names):
        """Execute a single trajectory point."""
        # Publish joint commands to simulation
        cmd_msg = JointTrajectoryPoint()
        cmd_msg.positions = point.positions
        cmd_msg.velocities = point.velocities if point.velocities else [0.0] * len(point.positions)
        cmd_msg.accelerations = point.accelerations if point.accelerations else [0.0] * len(point.positions)
        
        # Add timestamp
        cmd_msg.time_from_start = Duration(sec=0, nanosec=100000000)  # 100ms
        
        self.joint_pub.publish(cmd_msg)
        
        # In a real implementation, this would interface with the robot controller
        # and potentially adjust based on sensor feedback

    def start_balance_control(self):
        """Start balance control for humanoid motion."""
        # In a real robot, this would start the balance controller
        self.get_logger().info("Balance control started")

    def stop_balance_control(self):
        """Stop balance control after motion."""
        # In a real robot, this would stop the balance controller
        self.get_logger().info("Balance control stopped")

    def get_robot_joint_names(self):
        """Get the joint names of the simulated robot."""
        # Define the joints for our simulated humanoid
        return [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

def main(args=None):
    rclpy.init(args=args)
    action_server = HumanoidMotionActionServer()
    
    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate action execution performance under different conditions
- Test preemption and cancellation functionality
- Analyze feedback mechanisms and their effectiveness

## Lab 4: Multi-Node Communication Architecture

### Objective
Design and implement a complete communication architecture for a humanoid robot system with multiple autonomous nodes.

### Setup
1. Create multiple nodes representing different robot subsystems:
   - Perception node
   - Planning node
   - Control node
   - State estimation node
2. Connect them using appropriate ROS 2 communication patterns

### Implementation Steps
1. Design a communication architecture for the robot system
2. Implement each subsystem as a separate node
3. Connect the nodes using topics, services, and actions
4. Test the integrated system with simulation scenarios

### Architecture Design
```
Perception Node → Topic: sensor_data → State Estimation Node
Planning Node → Action: motion_plan → Control Node
High-level Controller → Service: execute_behavior → Planning Node
```

### Code Template
```python
#!/usr/bin/env python3

# This file contains the high-level architecture node that coordinates
# the other subsystems

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory

class HumanoidRobotController(Node):
    def __init__(self):
        super().__init__('humanoid_robot_controller')
        
        # Publishers for commanding other nodes
        self.state_pub = self.create_publisher(String, 'robot_state', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'navigation_goal', 10)
        
        # Subscribers for monitoring system state
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, 'robot_pose', self.pose_callback, 10)
        
        # Service client for emergency stop
        self.emergency_stop_client = self.create_client(
            Trigger, 'emergency_stop')
        
        # Action client for motion execution
        self.motion_action_client = ActionClient(
            self, FollowJointTrajectory, 'humanoid_controller/follow_joint_trajectory')
        
        # Timer for main control loop
        self.control_timer = self.create_timer(1.0, self.main_control_loop)
        
        # Robot state
        self.current_state = "idle"
        self.current_pose = None
        self.joint_states = None

    def joint_state_callback(self, msg):
        """Update joint states."""
        self.joint_states = msg

    def pose_callback(self, msg):
        """Update robot pose."""
        self.current_pose = msg

    def main_control_loop(self):
        """Main control loop for the robot."""
        # Update robot state based on sensors and internal state
        if self.joint_states is not None and self.current_pose is not None:
            self.current_state = "active"
            self.state_pub.publish(String(data=self.current_state))
            
            # Example: if robot is idle and a goal is available, plan a motion
            if self.current_state == "idle":
                self.plan_and_execute_motion()
        else:
            self.current_state = "waiting_for_sensors"
            self.state_pub.publish(String(data=self.current_state))

    def plan_and_execute_motion(self):
        """Plan and execute a motion."""
        # This would call the planning service in a real implementation
        self.get_logger().info('Planning motion...')
        
        # Check if motion action server is available
        if not self.motion_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Motion action server not available')
            return

        # Create motion goal
        goal_msg = FollowJointTrajectory.Goal()
        # In a real system, this would be populated with a planned trajectory
        # For simulation, we'll use a simple example
        
        # Send the goal
        future = self.motion_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.motion_feedback_callback)
        
        future.add_done_callback(self.motion_goal_response_callback)

    def motion_goal_response_callback(self, future):
        """Handle motion goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Motion goal rejected')
            return

        self.get_logger().info('Motion goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.motion_result_callback)

    def motion_feedback_callback(self, feedback_msg):
        """Handle motion feedback."""
        feedback = feedback_msg.feedback
        # Process feedback from motion execution
        self.get_logger().info(f'Motion progress: {feedback.progress}%')

    def motion_result_callback(self, future):
        """Handle motion result."""
        result = future.result().result
        self.get_logger().info(f'Motion completed with result: {result.error_code}')

def main(args=None):
    rclpy.init(args=args)
    robot_controller = HumanoidRobotController()
    
    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        robot_controller.get_logger().info('Interrupted by user')
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate the overall system performance and responsiveness
- Analyze communication bottlenecks and potential improvements
- Test the system's ability to handle multiple concurrent operations

## Lab Report Requirements

For each lab exercise, students must submit:

1. **Implementation Documentation** (25%):
   - Code with proper documentation
   - Explanation of design decisions
   - Description of QoS settings and communication pattern choices

2. **Performance Analysis** (40%):
   - Measurements of latency, throughput, and reliability
   - Analysis of resource utilization
   - Comparison of different QoS configurations

3. **System Integration Report** (25%):
   - How different communication patterns work together
   - Challenges encountered and solutions
   - Recommendations for improvements

4. **Reflection and Learning** (10%):
   - What was learned about ROS 2 communication
   - How concepts apply to real-world robotics
   - Future directions for improvement

## Assessment Criteria

- Implementation quality and correctness (40%)
- Understanding of ROS 2 communication patterns (30%)
- Performance analysis and optimization (20%)
- Documentation and code quality (10%)

## Troubleshooting Tips

1. **Topic Connection Issues**: Use `ros2 topic list` and `ros2 node info` to verify connections
2. **QoS Mismatch**: Ensure publishers and subscribers have compatible QoS settings
3. **Performance Issues**: Monitor CPU and memory usage with `ros2 topic hz` and system tools
4. **Action Cancellation**: Implement proper cancellation handling in action servers
5. **Service Timeouts**: Add appropriate timeout handling in service clients

## Extensions and Advanced Challenges

1. **Real-time Performance**: Configure the system for real-time operation
2. **Multi-robot Communication**: Extend to multiple robots communicating with each other
3. **Security Implementation**: Add ROS 2 security features to the system
4. **Distributed Computing**: Implement the system across multiple computers
5. **Formal Verification**: Verify properties of the communication system

## References and Further Reading

- ROS 2 Documentation: https://docs.ros.org/
- DDS Specification: https://www.omg.org/spec/DDS/
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.
- Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS.