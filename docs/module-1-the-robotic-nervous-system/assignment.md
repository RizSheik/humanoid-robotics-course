---
id: module-1-assignment
title: Module 1 — The Robotic Nervous System | Chapter 6 — Assignment
sidebar_label: Chapter 6 — Assignment
sidebar_position: 6
---

# Module 1 — The Robotic Nervous System

## Chapter 6 — Assignment

### Assignment Overview

This assignment integrates all the concepts learned in Module 1: The Robotic Nervous System. You will create a complete ROS 2-based communication system for a humanoid robot, implementing all the core components including nodes, topics, services, actions, TF transforms, Quality of Service (QoS) policies, and simulation integration.

### Learning Objectives

By completing this assignment, you will demonstrate:
1. Proficiency in ROS 2 programming with Python and/or C++
2. Understanding of communication patterns in robotic systems
3. Ability to create and integrate multiple ROS 2 nodes
4. Knowledge of TF transforms and coordinate systems
5. Application of appropriate QoS policies for different data types
6. Integration with robotics simulation environments

### Assignment Requirements

#### Core Components to Implement

1. **Robot State Publisher Node**:
   - Publish joint states for all robot joints
   - Publish TF transforms for the robot's kinematic chain
   - Use appropriate QoS policies for robot state information

2. **Control Command Processor**:
   - Subscribe to joint trajectory commands
   - Validate incoming commands for safety
   - Publish processed commands to simulation/hardware interface

3. **Sensor Data Aggregator**:
   - Subscribe to multiple sensor streams (IMU, cameras, etc.)
   - Process and fuse sensor data
   - Publish processed sensor information with appropriate QoS

4. **Parameter Management Service**:
   - Implement a service for changing robot parameters at runtime
   - Include validation for parameter limits
   - Log parameter changes for debugging

5. **System Monitor Action Server**:
   - Implement an action server for complex robot operations
   - Provide feedback during execution
   - Handle preemption and error recovery

6. **Simulation Integration**:
   - Create or modify URDF model for the humanoid robot
   - Configure Gazebo plugins for simulation
   - Implement controller interfaces for simulated hardware

#### Additional Requirements

1. **Documentation**:
   - Comprehensive package documentation
   - Node interfaces and message types
   - Launch file descriptions

2. **Testing**:
   - Unit tests for critical functions
   - Integration tests for node communication
   - Performance tests for real-time constraints

3. **Configuration**:
   - YAML configuration files for parameters
   - Launch files for different operational modes
   - Environment-specific configurations

### Detailed Implementation Steps

#### Step 1: Project Setup and Repository Structure

1. **Create a new ROS 2 package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python humanoid_robot_comm
   ```

2. **Set up the package structure**:
   ```
   humanoid_robot_comm/
   ├── humanoid_robot_comm/
   │   ├── __init__.py
   │   ├── robot_state_publisher.py
   │   ├── control_command_processor.py
   │   ├── sensor_data_aggregator.py
   │   ├── parameter_manager.py
   │   ├── system_monitor.py
   │   └── utils/
   │       └── common_functions.py
   ├── launch/
   │   ├── full_system.launch.py
   │   ├── simulation.launch.py
   │   └── hardware.launch.py
   ├── config/
   │   ├── robot_parameters.yaml
   │   ├── joint_limits.yaml
   │   └── qos_profiles.yaml
   ├── test/
   │   ├── test_robot_state_publisher.py
   │   └── test_sensor_aggregator.py
   ├── worlds/
   │   └── humanoid_test_world.world
   ├── urdf/
   │   └── humanoid_robot.urdf
   ├── rviz/
   │   └── humanoid_robot.rviz
   ├── CMakeLists.txt
   ├── package.xml
   └── setup.py
   ```

#### Step 2: Implement the Robot State Publisher

Create `humanoid_robot_comm/robot_state_publisher.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing state
        self.timer = self.create_timer(0.05, self.publish_robot_state)  # 20 Hz

        # Robot configuration
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'head_pan', 'head_tilt'
        ]

        # Initialize joint positions (default positions)
        self.joint_positions = [0.0] * len(self.joint_names)

        # Initialize base position
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_theta = 0.0

    def publish_robot_state(self):
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Publish joint states
        self.joint_pub.publish(msg)

        # Publish transforms
        self.publish_transforms(msg.header.stamp)

    def publish_transforms(self, time_stamp):
        # Base to world transform
        t = TransformStamped()
        t.header.stamp = time_stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.base_x
        t.transform.translation.y = self.base_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self.base_theta / 2.0)
        t.transform.rotation.w = math.cos(self.base_theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

        # Additional transforms based on joint positions would go here
        # For each joint, publish the appropriate transform
        # This is a simplified example with just the base transform

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Implement the Control Command Processor

Create `humanoid_robot_comm/control_command_processor.py`:

```python
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
import time

class ControlCommandProcessor(Node):
    def __init__(self):
        super().__init__('control_command_processor')

        # Publisher for processed commands
        self.command_pub = self.create_publisher(
            JointTrajectory,
            '/processed_joint_commands',
            10
        )

        # Subscriber for incoming commands
        self.command_sub = self.create_subscription(
            JointTrajectory,
            '/joint_commands',
            self.command_callback,
            10
        )

        # Publisher for status updates
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # Robot joint definitions
        self.valid_joints = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'head_pan', 'head_tilt'
        ]

        # Joint limits (example values)
        self.joint_limits = {
            'left_hip': {'min': -1.57, 'max': 1.57},
            'left_knee': {'min': 0.0, 'max': 2.35},
            'left_ankle': {'min': -0.78, 'max': 0.78},
            'right_hip': {'min': -1.57, 'max': 1.57},
            'right_knee': {'min': 0.0, 'max': 2.35},
            'right_ankle': {'min': -0.78, 'max': 0.78},
            'left_shoulder': {'min': -1.57, 'max': 1.57},
            'left_elbow': {'min': 0.0, 'max': 2.35},
            'left_wrist': {'min': -1.57, 'max': 1.57},
            'right_shoulder': {'min': -1.57, 'max': 1.57},
            'right_elbow': {'min': 0.0, 'max': 2.35},
            'right_wrist': {'min': -1.57, 'max': 1.57},
            'head_pan': {'min': -1.57, 'max': 1.57},
            'head_tilt': {'min': -0.78, 'max': 0.78}
        }

        self.get_logger().info('Control Command Processor initialized')

    def command_callback(self, msg):
        self.get_logger().info(f'Received trajectory with {len(msg.points)} points for joints: {msg.joint_names}')

        # Validate command
        if not self.validate_command(msg):
            self.get_logger().error('Invalid command received')
            self.publish_status('ERROR: Invalid command')
            return

        # Process command (in real system, this would send to hardware)
        processed_msg = self.process_command(msg)

        # Publish processed command
        self.command_pub.publish(processed_msg)
        self.publish_status('Command processed and published')

        self.get_logger().info('Command processed successfully')

    def validate_command(self, traj_msg):
        # Check if all joints in the command are valid
        for joint_name in traj_msg.joint_names:
            if joint_name not in self.valid_joints:
                self.get_logger().error(f'Invalid joint name: {joint_name}')
                return False

        # Check joint limits for all points
        for point in traj_msg.points:
            for i, joint_name in enumerate(traj_msg.joint_names):
                if i < len(point.positions):
                    pos = point.positions[i]
                    limits = self.joint_limits.get(joint_name)
                    if limits and (pos < limits['min'] or pos > limits['max']):
                        self.get_logger().error(f'Joint {joint_name} position {pos} out of limits [{limits["min"]}, {limits["max"]}]')
                        return False

        # Check timing constraints
        for i in range(1, len(traj_msg.points)):
            prev_time = traj_msg.points[i-1].time_from_start.sec + traj_msg.points[i-1].time_from_start.nanosec * 1e-9
            curr_time = traj_msg.points[i].time_from_start.sec + traj_msg.points[i].time_from_start.nanosec * 1e-9
            if curr_time <= prev_time:
                self.get_logger().error('Trajectory points not in chronological order')
                return False

        return True

    def process_command(self, traj_msg):
        # In a real system, this would process the command based on robot dynamics
        # For this assignment, we'll just pass through with safety checks
        processed_msg = JointTrajectory()
        processed_msg.joint_names = traj_msg.joint_names
        processed_msg.points = []

        for point in traj_msg.points:
            processed_point = JointTrajectoryPoint()
            processed_point.positions = point.positions
            processed_point.velocities = point.velocities
            processed_point.accelerations = point.accelerations
            processed_point.effort = point.effort
            processed_point.time_from_start = point.time_from_start
            processed_msg.points.append(processed_point)

        processed_msg.header.stamp = self.get_clock().now().to_msg()
        processed_msg.header.frame_id = 'base_link'

        return processed_msg

    def publish_status(self, status_msg):
        msg = String()
        msg.data = status_msg
        self.status_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControlCommandProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Implement the Parameter Management Service

Create `humanoid_robot_comm/parameter_manager.py`:

```python
import rclpy
from rclpy.node import Node
from humanoid_robot_comm.srv import SetRobotParams  # Custom service
from std_msgs.msg import String

class ParameterManager(Node):
    def __init__(self):
        super().__init__('parameter_manager')

        # Create service server
        self.srv = self.create_service(
            SetRobotParams,
            'set_robot_params',
            self.set_params_callback
        )

        # Publisher for parameter change notifications
        self.param_change_pub = self.create_publisher(
            String,
            '/parameter_changes',
            rclpy.qos.QoSProfile(
                depth=10,
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        )

        # Initialize robot parameters
        self.robot_params = {
            'control_loop_rate': 100.0,  # Hz
            'max_velocity': 1.0,         # rad/s
            'max_acceleration': 2.0,     # rad/s^2
            'safety_margin': 0.1,        # unitless safety factor
            'torso_height': 0.8,         # meters
            'arm_length': 0.7,           # meters
            'leg_length': 1.0,           # meters
        }

        self.get_logger().info('Parameter Manager initialized')

    def set_params_callback(self, request, response):
        self.get_logger().info(f'Received parameter update request: {request.param_name} = {request.param_value}')

        # Validate parameter name and value
        if request.param_name not in self.robot_params:
            response.success = False
            response.message = f'Unknown parameter: {request.param_name}'
            return response

        # Validate parameter value based on parameter type
        if request.param_name in ['control_loop_rate', 'max_velocity', 'max_acceleration']:
            if request.param_value <= 0:
                response.success = False
                response.message = f'Invalid value for {request.param_name}: must be positive'
                return response

        elif request.param_name in ['safety_margin']:
            if request.param_value < 0 or request.param_value > 1.0:
                response.success = False
                response.message = f'Invalid value for {request.param_name}: must be between 0 and 1'
                return response

        elif request.param_name in ['torso_height', 'arm_length', 'leg_length']:
            if request.param_value <= 0:
                response.success = False
                response.message = f'Invalid value for {request.param_name}: must be positive'
                return response

        # Update parameter
        old_value = self.robot_params[request.param_name]
        self.robot_params[request.param_name] = request.param_value

        # Log parameter change
        change_msg = String()
        change_msg.data = f'Parameter {request.param_name} changed from {old_value} to {request.param_value}'
        self.param_change_pub.publish(change_msg)

        self.get_logger().info(f'Parameter {request.param_name} updated from {old_value} to {request.param_value}')

        response.success = True
        response.message = f'Successfully updated {request.param_name} to {request.param_value}'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ParameterManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 5: Create Custom Service Definition

Create the service definition file `humanoid_robot_comm/srv/SetRobotParams.srv`:

```
# Request
string param_name
float64 param_value

---
# Response
bool success
string message
```

#### Step 6: Update setup.py

Update the `setup.py` file to include the service definition:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'humanoid_robot_comm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include service definitions
        (os.path.join('share', package_name, 'srv'), glob('humanoid_robot_comm/srv/*.srv')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        # Include RViz config
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 package for humanoid robot communication system',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_state_publisher = humanoid_robot_comm.robot_state_publisher:main',
            'control_command_processor = humanoid_robot_comm.control_command_processor:main',
            'parameter_manager = humanoid_robot_comm.parameter_manager:main',
        ],
    },
)
```

#### Step 7: Create Configuration Files

Create `config/robot_parameters.yaml`:

```yaml
# Robot-specific parameters
robot_config:
  control:
    loop_rate: 100.0  # Hz
    position_tolerance: 0.01  # rad
    velocity_tolerance: 0.1   # rad/s
    acceleration_limit: 5.0   # rad/s^2
    effort_limit: 100.0       # Nm

  safety:
    collision_threshold: 0.05  # meters
    velocity_limit: 2.0        # rad/s
    acceleration_limit: 5.0    # rad/s^2
    position_bounds:
      min: -3.14
      max: 3.14

  hardware:
    encoder_resolution: 4096   # counts per revolution
    motor_gear_ratio: 100.0
    max_torque: 50.0           # Nm
    torque_constant: 0.1       # Nm/A
    voltage_limit: 24.0        # V
```

### Testing and Validation

#### Unit Tests

1. **Test Robot State Publisher**:
   - Verify joint state messages are published at correct rate
   - Check TF transforms are published correctly
   - Validate message formats

2. **Test Control Command Processor**:
   - Test command validation logic
   - Verify joint limits enforcement
   - Check trajectory processing

3. **Test Parameter Management**:
   - Verify all parameters can be set
   - Test parameter validation
   - Check persistence of parameter changes

#### Integration Tests

1. **Communication Test**:
   - Verify all nodes can communicate properly
   - Test different QoS policies
   - Validate message passing under load

2. **Simulation Integration Test**:
   - Run system with Gazebo simulation
   - Verify robot responds to commands
   - Check sensor data flow

#### Performance Tests

1. **Real-time Performance**:
   - Measure message latency
   - Test system under computational load
   - Verify timing constraints are met

2. **Communication Throughput**:
   - Test system with high-frequency data
   - Validate no message loss under normal conditions

### Package Documentation

Create comprehensive documentation for your package:

1. **README.md**: Overview of the package and its components
2. **Node Documentation**: Detailed description of each node
3. **Message/Service Documentation**: Description of custom messages and services
4. **Configuration Guide**: How to configure and customize the system
5. **Troubleshooting Guide**: Common issues and solutions

### Submission Requirements

Submit the following components:

1. **Complete Source Code**: All ROS 2 packages with proper structure
2. **Documentation**: Comprehensive documentation for the system
3. **Configuration Files**: All YAML configs and launch files
4. **Test Results**: Output from unit and integration tests
5. **Performance Analysis**: Results from performance tests
6. **Video Demonstration**: Short video showing the system in simulation

### Grading Criteria

- **Functionality (40%)**: All components work as specified
- **Code Quality (25%)**: Clean, well-documented, and maintainable code
- **System Integration (20%)**: All components work together properly
- **Testing (10%)**: Comprehensive test coverage and validation
- **Documentation (5%)**: Clear and comprehensive documentation

### Additional Resources

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- Robot Operating System (ROS) tutorials
- Gazebo simulation tutorials
- TF2 tutorials and documentation

### Conclusion

This assignment provides a comprehensive implementation of the robotic nervous system for humanoid robots. Successfully completing this assignment will demonstrate your understanding of ROS 2 concepts, implementation of communication patterns, and integration of multiple robotic components into a cohesive system. The system you build will serve as a foundation for more advanced modules in the course.