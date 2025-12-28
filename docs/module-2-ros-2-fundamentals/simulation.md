# Module 2: Simulation - ROS 2 Communication in Robotic Environments

## Simulation Overview

This simulation module focuses on implementing and testing ROS 2 communication patterns in realistic robotic environments. Students will work with simulation tools to understand how topics, services, and actions function in complex robotic systems, with particular emphasis on humanoid robot applications.

### Learning Objectives

After completing this simulation module, students will be able to:
1. Set up and configure simulation environments with ROS 2 integration
2. Implement and test topic-based communication for sensor data streaming
3. Deploy service-based systems for robot configuration and control
4. Execute long-running tasks using action-based communication
5. Analyze communication performance in simulated robotic environments
6. Integrate multiple communication patterns for complex robot behaviors

### Required Simulation Tools

- **Gazebo Harmonic** or **Isaac Sim** for physics simulation
- **ROS 2 Humble Hawksbill** for robot control and communication
- **RViz2** for visualization
- **Python 3.11+** or **C++17** for implementing nodes
- **rqt** tools for monitoring and debugging
- **rosbags2** for data recording and playback

## Simulation Environment Configuration

### Gazebo Integration with ROS 2

For Gazebo-based simulations, configure the environment with proper ROS 2 bridges:

```xml
<!-- Example Gazebo model configuration with ROS 2 plugins -->
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Robot definition -->
    <link name="base_link">
      <sensor name="imu_sensor" type="imu">
        <plugin filename="libignition-gazebo-imu-system.so" name="ignition::gazebo::systems::Imu">
          <ros>
            <namespace>/robot1</namespace>
            <remapping>~/out@sensor_msgs/msg/Imu@ignition.msgs.IMU</remapping>
          </ros>
        </plugin>
      </sensor>
    </link>
    
    <!-- Joint state publisher -->
    <plugin filename="libignition-gazebo-joint-state-publisher-system.so" 
            name="ignition::gazebo::systems::JointStatePublisher">
      <ros>
        <namespace>/robot1</namespace>
      </ros>
    </plugin>
    
    <!-- Joint position controller -->
    <plugin filename="libignition-gazebo-joint-position-controller-system.so"
            name="ignition::gazebo::systems::JointPositionController">
      <ros>
        <namespace>/robot1</namespace>
      </ros>
    </plugin>
  </model>
</sdf>
```

### Isaac Sim Configuration

For Isaac Sim-based simulations:

1. Configure the robotic platform with appropriate sensors
2. Set up ROS 2 bridges for each sensor type
3. Configure QoS settings for different data streams
4. Implement appropriate control interfaces

## Simulation 1: Topic-Based Sensor Data Streaming

### Objective
Implement and test high-frequency topic-based communication for streaming sensor data in a humanoid robot simulation.

### Setup
1. Configure a humanoid robot model in simulation with multiple sensors:
   - 20+ joint position sensors
   - IMU (3-axis accelerometer, 3-axis gyroscope)
   - Multiple cameras (head, chest, wrist-mounted)
   - Force/torque sensors in feet and hands
   - LIDAR for environment perception

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import time

class RobotSensorSimulator(Node):
    def __init__(self):
        super().__init__('robot_sensor_simulator')
        
        # Create publishers for different sensor types
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.camera_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)
        
        # QoS configuration for different sensor types
        self.sensor_qos = rclpy.qos.QoSProfile(
            depth=5,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
        
        # Publishers with specific QoS for different sensors
        self.hf_publisher = self.create_publisher(
            JointState, 'high_freq_sensor', self.sensor_qos)
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        
        # Initialize joint names for humanoid robot
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'neck_joint', 'waist_joint'
        ]
        
        # Initialize simulation parameters
        self.sim_time = 0.0
        self.time_step = 0.01  # 100 Hz simulation
        
        # Create timers for different sensor frequencies
        self.joint_timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz
        self.imu_timer = self.create_timer(0.005, self.publish_imu_data)      # 200 Hz
        self.camera_timer = self.create_timer(0.1, self.publish_camera_data)  # 10 Hz
        
        self.get_logger().info('Robot sensor simulator initialized')

    def publish_joint_states(self):
        """Publish joint state data at 100Hz."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # Simulate joint positions (oscillating for demonstration)
        positions = []
        for i, _ in enumerate(self.joint_names):
            # Create oscillating joint positions
            pos = 0.1 * np.sin(2 * np.pi * 0.5 * self.sim_time + i * 0.5)
            positions.append(pos)
        
        msg.position = positions
        msg.velocity = [0.0] * len(positions)  # Simplified
        msg.effort = [0.0] * len(positions)    # Simplified
        
        self.joint_pub.publish(msg)
        self.sim_time += self.time_step

    def publish_imu_data(self):
        """Publish IMU data at 200Hz."""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Simulate IMU readings
        msg.linear_acceleration.x = 0.1 * np.sin(2 * np.pi * 2 * self.sim_time)
        msg.linear_acceleration.y = 0.1 * np.cos(2 * np.pi * 2 * self.sim_time)
        msg.linear_acceleration.z = 9.8 + 0.05 * np.sin(2 * np.pi * 5 * self.sim_time)
        
        # Angular velocity
        msg.angular_velocity.x = 0.5 * np.sin(2 * np.pi * 1 * self.sim_time)
        msg.angular_velocity.y = 0.3 * np.cos(2 * np.pi * 1.5 * self.sim_time)
        msg.angular_velocity.z = 0.2 * np.sin(2 * np.pi * 0.8 * self.sim_time)
        
        # Orientation (simplified)
        msg.orientation.w = 1.0  # Perfectly upright for now
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        
        self.imu_pub.publish(msg)

    def publish_camera_data(self):
        """Publish camera data at 10Hz."""
        # Create a simple simulated image (grayscale gradient)
        height, width = 640, 480
        # Create a gradient image for simulation
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                img_array[i, j, 0] = int(255 * i / height)  # Red channel gradient
                img_array[i, j, 1] = int(255 * j / width)   # Green channel gradient
                img_array[i, j, 2] = 128  # Constant blue
        
        # Convert to ROS image message
        img_msg = self.bridge.cv2_to_imgmsg(img_array, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        
        # Publish camera info
        info_msg = CameraInfo()
        info_msg.header = img_msg.header
        info_msg.width = width
        info_msg.height = height
        info_msg.k = [500.0, 0.0, width/2, 0.0, 500.0, height/2, 0.0, 0.0, 1.0]  # Simplified camera matrix
        
        self.camera_pub.publish(img_msg)
        self.camera_info_pub.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_simulator = RobotSensorSimulator()
    
    try:
        rclpy.spin(sensor_simulator)
    except KeyboardInterrupt:
        sensor_simulator.get_logger().info('Simulation stopped by user')
    finally:
        sensor_simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Monitor topic data rates using `ros2 topic hz`
- Analyze the effect of QoS settings on sensor data delivery
- Test performance under different simulation loads
- Visualize sensor data using RViz2

## Simulation 2: Service-Based Robot Configuration and Control

### Objective
Implement and test service-based communication for robot configuration and control tasks that require synchronous responses.

### Setup
1. Create a humanoid robot simulation with configurable parameters
2. Implement services for:
   - Changing robot operational modes
   - Setting joint parameters (limits, gains)
   - Calibrating sensors
   - Emergency stop functionality

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger, SetBool
from control_msgs.srv import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

class RobotConfigurationSimulator(Node):
    def __init__(self):
        super().__init__('robot_config_simulator')
        
        # Service servers
        self.mode_service = self.create_service(
            SetBool, 'set_robot_mode', self.set_robot_mode_callback)
        
        self.calibration_service = self.create_service(
            Trigger, 'calibrate_robot', self.calibrate_robot_callback)
        
        self.emergency_stop_service = self.create_service(
            Trigger, 'emergency_stop', self.emergency_stop_callback)
        
        # Publisher for joint commands (for demonstration)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Robot state
        self.robot_mode = "idle"
        self.calibration_status = {"completed": False, "timestamp": None}
        self.emergency_stopped = False
        self.joint_limits = self.initialize_joint_limits()
        
        self.get_logger().info('Robot configuration simulator initialized')

    def initialize_joint_limits(self):
        """Initialize default joint limits for the robot."""
        return {
            'left_hip_joint': (-1.57, 1.57),   # min, max in radians
            'left_knee_joint': (0.0, 2.5), 
            'left_ankle_joint': (-0.78, 0.78),
            'right_hip_joint': (-1.57, 1.57),
            'right_knee_joint': (0.0, 2.5),
            'right_ankle_joint': (-0.78, 0.78),
            # Add more joints as needed
        }

    def set_robot_mode_callback(self, request, response):
        """Handle robot mode change requests."""
        mode_name = "active" if request.data else "idle"
        
        if self.emergency_stopped and request.data:
            response.success = False
            response.message = "Cannot activate robot in emergency stop state"
            self.get_logger().warn('Rejected activation request: robot in emergency stop')
            return response
        
        # Perform mode change
        old_mode = self.robot_mode
        self.robot_mode = mode_name
        
        # Send confirmation command to simulated robot
        if request.data:  # Activating
            self.activate_robot_systems()
        else:  # Deactivating
            self.deactivate_robot_systems()
        
        response.success = True
        response.message = f"Mode changed from {old_mode} to {mode_name}"
        
        self.get_logger().info(f'Robot mode changed: {old_mode} → {mode_name}')
        return response

    def activate_robot_systems(self):
        """Activate robot systems."""
        # In simulation, we might enable controllers, etc.
        self.get_logger().info('Activating robot systems...')
        # Simulate system activation
        time.sleep(0.1)  # Simulated activation time

    def deactivate_robot_systems(self):
        """Deactivate robot systems."""
        self.get_logger().info('Deactivating robot systems...')
        # In a real system, this would safely shut down controllers

    def calibrate_robot_callback(self, request, response):
        """Handle robot calibration requests."""
        if self.robot_mode != "idle":
            response.success = False
            response.message = f"Cannot calibrate: robot is in {self.robot_mode} mode"
            self.get_logger().warn('Calibration rejected: robot not in idle mode')
            return response
        
        if self.emergency_stopped:
            response.success = False
            response.message = "Cannot calibrate: robot in emergency stop"
            self.get_logger().warn('Calibration rejected: robot in emergency stop')
            return response
        
        try:
            # Simulate calibration process
            self.get_logger().info('Starting robot calibration...')
            
            # Simulate calibration steps
            for step in range(5):
                self.get_logger().info(f'Calibration step {step + 1}/5')
                time.sleep(0.2)  # Simulate calibration time
            
            # Update calibration status
            self.calibration_status = {
                "completed": True,
                "timestamp": self.get_clock().now().to_msg()
            }
            
            response.success = True
            response.message = "Calibration completed successfully"
            
            self.get_logger().info('Robot calibration completed')
            
        except Exception as e:
            response.success = False
            response.message = f"Calibration failed: {str(e)}"
            self.get_logger().error(f'Calibration error: {str(e)}')
        
        return response

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
            self.get_logger().warn('EMERGENCY STOP ACTIVATED')
            
        except Exception as e:
            self.get_logger().error(f'Error during emergency stop: {str(e)}')
            response.success = False
            response.message = f'Emergency stop failed: {str(e)}'
        
        return response

    def execute_emergency_stop(self):
        """Execute the actual emergency stop procedure."""
        # Stop all joint movements immediately
        stop_msg = JointState()
        stop_msg.name = list(self.joint_limits.keys())
        stop_msg.position = [0.0] * len(stop_msg.name)  # Stop at current position
        self.joint_pub.publish(stop_msg)
        
        # Update robot state
        self.emergency_stopped = True
        self.robot_mode = "emergency"
        
        # Log the event
        self.get_logger().warn('All joint movements stopped by emergency stop')

def main(args=None):
    rclpy.init(args=args)
    config_simulator = RobotConfigurationSimulator()
    
    try:
        rclpy.spin(config_simulator)
    except KeyboardInterrupt:
        config_simulator.get_logger().info('Simulation stopped by user')
    finally:
        config_simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import time  # Required for the implementation above
    main()
```

### Analysis
- Test service response times under different simulation loads
- Evaluate reliability of service-based communication
- Analyze behavior during error conditions
- Test concurrent service requests

## Simulation 3: Action-Based Complex Task Execution

### Objective
Implement and test action-based communication for long-running tasks that require feedback and cancellation, such as humanoid walking or manipulation.

### Implementation
```python
#!/usr/bin/env python3

import time
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class HumanoidMotionSimulator(Node):
    def __init__(self):
        super().__init__('humanoid_motion_simulator')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'humanoid_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        
        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectoryPoint, 'simulated_joint_commands', 10)
        
        # Balance controller publisher
        self.balance_cmd_pub = self.create_publisher(
            JointTrajectoryPoint, 'balance_control_commands', 10)
        
        self.get_logger().info('Humanoid motion simulator initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal requests."""
        trajectory = goal_request.trajectory
        
        # Validate trajectory
        if len(trajectory.points) == 0:
            self.get_logger().warn('Goal rejected: No trajectory points')
            return GoalResponse.REJECT
        
        if len(trajectory.joint_names) == 0:
            self.get_logger().warn('Goal rejected: No joint names specified')
            return GoalResponse.REJECT
        
        # Check joint validity (in a real system, compare with robot's joints)
        required_joints = set(trajectory.joint_names)
        if len(required_joints) == 0:
            return GoalResponse.REJECT
        
        self.get_logger().info(f'Goal accepted with {len(trajectory.points)} points')
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
        
        # Initialize feedback message
        feedback_msg.desired = JointTrajectoryPoint()
        feedback_msg.actual = JointTrajectoryPoint()
        feedback_msg.error = JointTrajectoryPoint()
        
        # Start balance control for humanoid motion
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
                feedback_msg.desired = point
                feedback_msg.actual.positions = [0.0] * len(joint_names)
                feedback_msg.error.positions = [0.0] * len(joint_names)
                
                # Calculate progress percentage
                progress = (i + 1) / len(points) * 100.0
                feedback_msg.progress = progress
                
                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)
                
                # Execute the motion point
                self.execute_trajectory_point(point, joint_names)
                
                # Log progress
                self.get_logger().info(f'Progress: {progress:.1f}% ({i+1}/{len(points)})')
                
                # Simulate execution time based on time_from_start
                if i > 0:
                    prev_point_time = points[i-1].time_from_start.sec + points[i-1].time_from_start.nanosec / 1e9
                    curr_point_time = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
                    sleep_time = max(0, curr_point_time - prev_point_time)
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.01)  # Small delay for first point
        
        except Exception as e:
            self.get_logger().error(f'Error during motion execution: {str(e)}')
            result_msg.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            goal_handle.abort()
            return result_msg
            
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
        # Create and publish joint command
        cmd_msg = JointTrajectoryPoint()
        cmd_msg.positions = point.positions
        cmd_msg.velocities = point.velocities if point.velocities else [0.0] * len(point.positions)
        cmd_msg.accelerations = point.accelerations if point.accelerations else [0.0] * len(point.positions)
        
        # Set execution time
        cmd_msg.time_from_start = Duration(sec=0, nanosec=int(10000000))  # 10ms
        
        # Publish command to simulation
        self.joint_cmd_pub.publish(cmd_msg)
        
        # In a real implementation, this would interface with the robot controller
        # and potentially adjust based on sensor feedback

    def start_balance_control(self):
        """Start balance control for humanoid motion."""
        # In a real robot, this would start the balance controller
        self.get_logger().info("Balance control started for humanoid motion")

    def stop_balance_control(self):
        """Stop balance control after motion."""
        # In a real robot, this would stop the balance controller
        self.get_logger().info("Balance control stopped after motion")

def main(args=None):
    rclpy.init(args=args)
    motion_simulator = HumanoidMotionSimulator()
    
    try:
        rclpy.spin(motion_simulator)
    except KeyboardInterrupt:
        motion_simulator.get_logger().info('Simulation stopped by user')
    finally:
        motion_simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Measure action execution performance and feedback frequency
- Test cancellation and preemption functionality
- Evaluate the effectiveness of progress reporting
- Analyze communication overhead for long-running tasks

## Simulation 4: Multi-Robot Communication Simulation

### Objective
Simulate a multi-robot environment where ROS 2 communication patterns are used for coordination and collaboration between humanoid robots.

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterEvent

class MultiRobotCoordinatorSimulator(Node):
    def __init__(self):
        super().__init__('multirobot_coordinator_sim')
        
        # Robot ID for this instance (passed as parameter)
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        
        # Create namespaced topics for multi-robot communication
        self.status_pub = self.create_publisher(
            String, f'/{self.robot_id}/status', 10)
        
        self.pose_pub = self.create_publisher(
            PoseStamped, f'/{self.robot_id}/pose', 10)
        
        # Subscription to other robots' poses
        self.other_poses_sub = self.create_subscription(
            PoseStamped, '/other_robot/pose', self.other_pose_callback, 10)
        
        # Latched topic for robot status using TRANSIENT_LOCAL durability
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.status_latched_pub = self.create_publisher(
            String, f'/{self.robot_id}/init_status', latched_qos)
        
        # Timer for status updates
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        # Robot state
        self.current_pose = PoseStamped()
        self.other_robot_pose = None
        self.status = "idle"
        
        # Publish initial status as latched message
        init_status_msg = String()
        init_status_msg.data = f"{self.robot_id} initialized and ready"
        self.status_latched_pub.publish(init_status_msg)
        
        self.get_logger().info(f'Multi-robot coordinator for {self.robot_id} initialized')

    def publish_status(self):
        """Publish robot status periodically."""
        status_msg = String()
        status_msg.data = f"{self.robot_id}: {self.status}"
        self.status_pub.publish(status_msg)
        
        # Also publish pose
        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = 'map'
        # Simulate movement
        self.current_pose.pose.position.x = 1.0 * abs(hash(self.robot_id)) % 10 / 10.0
        self.current_pose.pose.position.y = 2.0 * abs(hash(self.robot_id)) % 10 / 10.0
        self.current_pose.pose.position.z = 0.0
        self.pose_pub.publish(self.current_pose)

    def other_pose_callback(self, msg):
        """Handle pose messages from other robots."""
        self.other_robot_pose = msg
        self.get_logger().debug(f'Received pose from another robot at: ({msg.pose.position.x}, {msg.pose.position.y})')

def main(args=None):
    rclpy.init(args=args)
    coordinator = MultiRobotCoordinatorSimulator()
    
    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        coordinator.get_logger().info('Multi-robot simulation stopped by user')
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate performance of multi-robot communication
- Analyze namespace management and topic organization
- Test scalability with increasing number of robots
- Measure communication overhead in multi-robot systems

## Simulation Evaluation and Performance Metrics

### Key Performance Indicators

1. **Communication Latency**:
   - Message publishing-to-receiving latency
   - Service response times
   - Action goal-to-result times

2. **Throughput**:
   - Messages per second for different topics
   - Bandwidth utilization
   - CPU and memory usage

3. **Reliability**:
   - Message delivery rates
   - Error rates under load
   - Recovery from communication failures

### Measurement Tools

1. **ros2 topic hz**: Measure message frequency
2. **ros2 topic delay**: Measure message delay
3. **rqt_plot**: Visualize numeric data over time
4. **ros2 bag**: Record data for offline analysis
5. **Custom monitoring nodes**: Track specific metrics

## Advanced Simulation Concepts

### Real-time Simulation

Configure simulation for real-time performance with ROS 2:

```python
# Configuration for real-time simulation
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Define real-time QoS profile
rt_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    deadline=Duration(seconds=0, nanoseconds=10000000)  # 10ms deadline
)
```

### Deterministic Simulation

For testing and verification, create deterministic simulations:

```python
import random

class DeterministicSimulator(Node):
    def __init__(self):
        super().__init__('deterministic_sim')
        # Set random seed for reproducible results
        random.seed(42)
        # All random operations will now be deterministic
```

## Simulation Troubleshooting

### Common Issues and Solutions

1. **Topic Connection Problems**:
   - Use `ros2 topic list` and `ros2 node info` to verify connections
   - Check ROS_DOMAIN_ID if running multiple systems
   - Verify QoS compatibility between publishers and subscribers

2. **Performance Bottlenecks**:
   - Monitor CPU and memory usage with system tools
   - Use `ros2 topic hz` to check topic frequencies
   - Profile nodes using `tracetools` or similar tools

3. **Timing Issues**:
   - Ensure proper synchronization in multi-node systems
   - Use appropriate timer periods for different tasks
   - Check for blocking operations in callback functions

## Simulation Extensions

### Integration with Real Hardware in the Loop

Extend simulations to include real hardware components:

1. **Hardware Abstraction Layer**: Interface with real sensors/actuators
2. **Mixed Reality**: Combine real and simulated environments
3. **Remote Operation**: Control simulated robots from real interfaces

### Cloud-Based Simulation

Deploy simulations in cloud environments:

1. **Distributed Simulation**: Run simulation across multiple machines
2. **Scalable Testing**: Test with many robots simultaneously
3. **Resource Management**: Efficiently allocate computing resources

## Chapter Summary

This simulation module provided comprehensive hands-on experience with ROS 2 communication patterns in realistic robotic environments. Students implemented and tested topic, service, and action-based communication systems, with particular focus on humanoid robot applications. The module emphasized performance evaluation, multi-robot coordination, and best practices for simulation-based testing of robotic systems.

## Key Terms
- Simulation Environment Configuration
- Multi-Robot Communication
- Quality of Service (QoS) in Simulation
- Topic-Based Streaming
- Service-Based Configuration
- Action-Based Execution
- Performance Metrics in Simulation

## References
- ROS 2 Documentation: https://docs.ros.org/
- Gazebo Documentation: http://gazebosim.org/
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.