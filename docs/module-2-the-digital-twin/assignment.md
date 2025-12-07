---
id: module-2-assignment
title: Module 2 — The Digital Twin | Chapter 6 — Assignment
sidebar_label: Chapter 6 — Assignment
sidebar_position: 6
---

# Module 2 — The Digital Twin

## Chapter 6 — Assignment

### Assignment Overview

This assignment focuses on implementing a complete digital twin system for a humanoid robot using Gazebo as the primary simulation environment. The goal is to create a synchronized system that maintains an accurate virtual representation of a physical robot, enabling testing, monitoring, and validation of humanoid robotics applications.

### Learning Objectives

Through this assignment, you will demonstrate:
1. Proficiency in creating and configuring Gazebo simulation environments
2. Understanding of digital twin architecture and synchronization mechanisms
3. Ability to implement realistic sensor and actuator simulation
4. Skills in performance monitoring and validation of digital twin systems
5. Integration of simulation with real-time control systems

### Assignment Requirements

#### Core Components to Implement

1. **Digital Twin Architecture**:
   - Create a comprehensive digital twin system for a humanoid robot
   - Implement real-time synchronization between physical and virtual systems
   - Design data flow architecture for efficient communication

2. **Physics Simulation Model**:
   - Develop a realistic humanoid robot model in Gazebo
   - Implement accurate physical properties (mass, inertia, friction, etc.)
   - Include appropriate sensors (IMU, cameras, force/torque sensors)

3. **Synchronization System**:
   - Implement state estimation and prediction algorithms
   - Handle network latency and communication delays
   - Ensure accurate mirroring of physical robot state

4. **Monitoring and Analytics**:
   - Create performance metrics for tracking digital twin accuracy
   - Implement visualization tools for monitoring synchronization
   - Develop anomaly detection capabilities

5. **Validation Framework**:
   - Design testing procedures to validate digital twin accuracy
   - Compare simulation behavior to expected physical behavior
   - Document simulation-to-reality gap analysis

#### Additional Requirements

1. **Documentation**:
   - Comprehensive system architecture documentation
   - Setup and configuration guides
   - Validation results and analysis

2. **Testing**:
   - Unit tests for critical components
   - Integration tests for the complete system
   - Performance benchmarks

3. **Scalability Considerations**:
   - Design for potential multi-robot digital twin systems
   - Optimize for real-time performance
   - Consider resource usage efficiency

### Detailed Implementation Steps

#### Step 1: Project Setup and Repository Structure

1. **Create a new ROS 2 package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python digital_twin_system
   ```

2. **Set up the package structure**:
   ```
   digital_twin_system/
   ├── digital_twin_system/
   │   ├── __init__.py
   │   ├── digital_twin_manager.py
   │   ├── state_synchronizer.py
   │   ├── sensor_simulator.py
   │   ├── validation_monitor.py
   │   └── utils/
   │       ├── transforms.py
   │       └── filters.py
   ├── launch/
   │   ├── digital_twin_system.launch.py
   │   ├── simulation.launch.py
   │   └── monitoring.launch.py
   ├── worlds/
   │   └── humanoid_lab.world
   ├── models/
   │   └── digital_twin_humanoid/
   │       ├── model.sdf
   │       ├── materials/
   │       └── meshes/
   ├── config/
   │   ├── robot_properties.yaml
   │   ├── sync_parameters.yaml
   │   └── monitoring_config.yaml
   ├── test/
   │   ├── test_synchronizer.py
   │   └── test_sensor_simulator.py
   ├── scripts/
   │   └── validation_tool.py
   ├── rviz/
   │   └── digital_twin_system.rviz
   ├── CMakeLists.txt
   ├── package.xml
   └── setup.py
   ```

#### Step 2: Implement the Digital Twin Manager

Create `digital_twin_system/digital_twin_system/digital_twin_manager.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from digital_twin_system.msg import DigitalTwinState  # Custom message
import numpy as np
import time
import math
from collections import deque

class DigitalTwinManager(Node):
    def __init__(self):
        super().__init__('digital_twin_manager')

        # Publishers
        self.digital_joint_pub = self.create_publisher(
            JointState,
            '/digital_twin/joint_states',
            10
        )

        self.digital_odom_pub = self.create_publisher(
            Odometry,
            '/digital_twin/odom',
            10
        )

        self.sync_error_pub = self.create_publisher(
            Float32,
            '/digital_twin/sync_error',
            10
        )

        self.state_pub = self.create_publisher(
            DigitalTwinState,
            '/digital_twin/state',
            10
        )

        # Subscribers
        self.physical_joint_sub = self.create_subscription(
            JointState,
            '/physical_robot/joint_states',
            self.physical_joint_callback,
            10
        )

        self.physical_odom_sub = self.create_subscription(
            Odometry,
            '/physical_robot/odom',
            self.physical_odom_callback,
            10
        )

        # Timer for main control loop
        self.timer = self.create_timer(0.05, self.main_loop)  # 20 Hz

        # Digital twin state tracking
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'head_pan', 'head_tilt'
        ]

        self.physical_joint_positions = {name: 0.0 for name in self.joint_names}
        self.digital_joint_positions = {name: 0.0 for name in self.joint_names}
        self.physical_odom = Odometry()
        self.digital_odom = Odometry()

        # Synchronization tracking
        self.sync_error_history = deque(maxlen=100)
        self.last_sync_time = time.time()

        # Performance metrics
        self.update_count = 0
        self.avg_sync_error = 0.0

        self.get_logger().info('Digital Twin Manager initialized')

    def physical_joint_callback(self, msg):
        """Callback for physical robot joint states"""
        for i, name in enumerate(msg.name):
            if name in self.physical_joint_positions:
                self.physical_joint_positions[name] = msg.position[i]

        # Update digital twin with physical values (with some delay simulation)
        self.update_digital_twin_from_physical()

    def physical_odom_callback(self, msg):
        """Callback for physical robot odometry"""
        self.physical_odom = msg

        # Update digital twin odometry
        self.digital_odom = msg  # For this example, direct copy with delay simulation

    def update_digital_twin_from_physical(self):
        """Update digital twin based on physical robot data with simulated delay"""
        # Simulate network delay
        delay = 0.05  # 50ms delay

        # Apply delay by updating digital twin positions to match physical ones
        for name in self.joint_names:
            # In a real system, this would account for the delay in network communication
            self.digital_joint_positions[name] = self.physical_joint_positions[name]

    def main_loop(self):
        """Main control loop for digital twin management"""
        self.update_count += 1

        # Publish synchronized digital twin states
        self.publish_digital_states()

        # Calculate and publish synchronization error
        sync_error = self.calculate_sync_error()
        self.publish_sync_metrics(sync_error)

        # Update performance statistics
        self.update_performance_metrics(sync_error)

        if self.update_count % 100 == 0:  # Log every 100 updates
            self.get_logger().info(
                f'Digital Twin Status: Avg Sync Error = {self.avg_sync_error:.4f}, '
                f'Update Rate = {self.update_count / (time.time() - self.last_sync_time):.2f} Hz'
            )
            self.update_count = 0
            self.last_sync_time = time.time()

    def calculate_sync_error(self):
        """Calculate the difference between physical and digital states"""
        total_error = 0.0
        error_count = 0

        for name in self.joint_names:
            error = abs(
                self.physical_joint_positions[name] -
                self.digital_joint_positions[name]
            )
            total_error += error
            error_count += 1

        if error_count > 0:
            return total_error / error_count
        else:
            return 0.0

    def publish_digital_states(self):
        """Publish digital twin joint states and odometry"""
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'digital_twin_base'

        joint_msg.name = self.joint_names
        joint_msg.position = [self.digital_joint_positions[name] for name in self.joint_names]
        joint_msg.velocity = [0.0] * len(self.joint_names)  # Simplified
        joint_msg.effort = [0.0] * len(self.joint_names)    # Simplified

        self.digital_joint_pub.publish(joint_msg)

        # Publish odometry
        self.digital_odom.header.stamp = self.get_clock().now().to_msg()
        self.digital_odom.header.frame_id = 'digital_twin_odom'
        self.digital_odom.child_frame_id = 'digital_twin_base_footprint'
        self.digital_odom_pub.publish(self.digital_odom)

    def publish_sync_metrics(self, error):
        """Publish synchronization metrics"""
        error_msg = Float32()
        error_msg.data = error
        self.sync_error_pub.publish(error_msg)

        # Update state message with metrics
        state_msg = DigitalTwinState()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.synchronization_error = error
        state_msg.performance_score = max(0.0, 1.0 - error)  # Inverse of error
        state_msg.status = 'ACTIVE' if error < 0.1 else 'WARNING'
        self.state_pub.publish(state_msg)

    def update_performance_metrics(self, error):
        """Update performance tracking metrics"""
        self.sync_error_history.append(error)

        # Calculate average error over recent history
        if self.sync_error_history:
            self.avg_sync_error = sum(self.sync_error_history) / len(self.sync_error_history)

def main(args=None):
    rclpy.init(args=args)
    node = DigitalTwinManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Digital Twin Manager stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Implement the State Synchronizer

Create `digital_twin_system/digital_twin_system/state_synchronizer.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
from digital_twin_system.utils.filters import KalmanFilter
import numpy as np
import math
from collections import deque

class StateSynchronizer(Node):
    def __init__(self):
        super().__init__('state_synchronizer')

        # TF broadcaster for transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.sync_joint_pub = self.create_publisher(
            JointState,
            '/synchronized_joint_states',
            10
        )

        # Subscribers
        self.physical_joint_sub = self.create_subscription(
            JointState,
            '/physical_robot/joint_states',
            self.physical_joint_callback,
            10
        )

        self.digital_joint_sub = self.create_subscription(
            JointState,
            '/digital_twin/joint_states',
            self.digital_joint_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/physical_robot/imu/data',
            self.imu_callback,
            10
        )

        # Timer for synchronization loop
        self.timer = self.create_timer(0.01, self.sync_loop)  # 100 Hz

        # State estimation components
        self.kalman_filter = KalmanFilter(state_dim=13, measurement_dim=7)
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist',
            'head_pan', 'head_tilt'
        ]

        # State buffers
        self.physical_joints = {name: 0.0 for name in self.joint_names}
        self.digital_joints = {name: 0.0 for name in self.joint_names}
        self.estimated_state = np.zeros(13)  # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]

        # Prediction buffers
        self.state_history = deque(maxlen=50)  # Store past states for prediction
        self.time_history = deque(maxlen=50)   # Store corresponding timestamps

        self.get_logger().info('State Synchronizer initialized')

    def physical_joint_callback(self, msg):
        """Update physical joint state"""
        for i, name in enumerate(msg.name):
            if name in self.physical_joints:
                self.physical_joints[name] = msg.position[i]

    def digital_joint_callback(self, msg):
        """Update digital joint state"""
        for i, name in enumerate(msg.name):
            if name in self.digital_joints:
                self.digital_joints[name] = msg.position[i]

    def imu_callback(self, msg):
        """Process IMU data for state estimation"""
        # Extract orientation from IMU
        orientation = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])

        # Extract angular velocity
        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Create measurement vector [orientation, angular_velocity]
        measurement = np.concatenate([orientation, angular_vel])

        # Update Kalman filter with measurement
        self.estimated_state = self.kalman_filter.update(measurement)

    def sync_loop(self):
        """Main synchronization loop"""
        current_time = self.get_clock().now().nanoseconds * 1e-9

        # Store current state in history
        self.state_history.append(self.estimated_state.copy())
        self.time_history.append(current_time)

        # Predict current state based on dynamics model
        self.predict_state(current_time)

        # Publish synchronized joint states
        self.publish_synchronized_states()

        # Publish transforms
        self.publish_transforms(current_time)

    def predict_state(self, current_time):
        """Predict state forward in time accounting for delays"""
        # Simplified prediction based on current velocity
        dt = 0.01  # Time step
        state_dot = np.zeros_like(self.estimated_state)

        # Position update based on velocity
        state_dot[0:3] = self.estimated_state[7:10]  # velocity -> position change
        # Orientation update based on angular velocity
        omega = self.estimated_state[10:13]  # angular velocity
        q = self.estimated_state[3:7]  # orientation quaternion

        # Calculate quaternion derivative
        q_dot = self._quat_derivative(q, omega)
        state_dot[3:7] = q_dot

        # Update state prediction
        self.estimated_state += state_dot * dt

    def _quat_derivative(self, q, omega):
        """Compute quaternion derivative from angular velocity"""
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        return 0.5 * self._quat_multiply(omega_quat, q)

    def _quat_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def publish_synchronized_states(self):
        """Publish synchronized joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'synchronized_frame'

        # Use synchronized joint positions
        msg.name = list(self.physical_joints.keys())
        msg.position = list(self.physical_joints.values())
        msg.velocity = [0.0] * len(msg.position)  # Simplified
        msg.effort = [0.0] * len(msg.position)    # Simplified

        self.sync_joint_pub.publish(msg)

    def publish_transforms(self, timestamp):
        """Publish synchronized transforms"""
        # Publish base transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        # Set position and orientation from estimated state
        t.transform.translation.x = self.estimated_state[0]
        t.transform.translation.y = self.estimated_state[1]
        t.transform.translation.z = self.estimated_state[2]

        t.transform.rotation.w = self.estimated_state[3]
        t.transform.rotation.x = self.estimated_state[4]
        t.transform.rotation.y = self.estimated_state[5]
        t.transform.rotation.z = self.estimated_state[6]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = StateSynchronizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('State Synchronizer stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Create the Sensor Simulator

Create `digital_twin_system/digital_twin_system/sensor_simulator.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image, LaserScan
from geometry_msgs.msg import PointStamped, Vector3Stamped
import numpy as np
import math

class SensorSimulator(Node):
    def __init__(self):
        super().__init__('sensor_simulator')

        # Publishers for simulated sensors
        self.imu_pub = self.create_publisher(Imu, '/digital_twin/imu/data', 10)
        self.camera_pub = self.create_publisher(Image, '/digital_twin/camera/image_raw', 10)
        self.laser_pub = self.create_publisher(LaserScan, '/digital_twin/laser_scan', 10)
        self.ft_sensor_pub = self.create_publisher(
            Vector3Stamped,
            '/digital_twin/force_torque',
            10
        )

        # Subscriber for robot state
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/digital_twin/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for sensor simulation
        self.timer = self.create_timer(0.02, self.simulate_sensors)  # 50 Hz
        self.imu_timer = self.create_timer(0.01, self.simulate_imu)  # 100 Hz for IMU

        # Robot state tracking
        self.joint_positions = {}
        self.joint_velocities = {}

        # Noise parameters for sensors
        self.imu_noise = {
            'accel': 0.01,    # m/s²
            'gyro': 0.001,    # rad/s
            'mag': 0.1        # µT
        }

        self.camera_params = {
            'width': 640,
            'height': 480,
            'fov': 1.047,     # 60 degrees in radians
            'noise': 2.0      # pixel noise std
        }

        self.laser_params = {
            'angle_min': -math.pi/2,
            'angle_max': math.pi/2,
            'angle_increment': math.pi/180,  # 1 degree
            'range_min': 0.1,
            'range_max': 10.0,
            'noise': 0.02    # 2cm noise
        }

        self.get_logger().info('Sensor Simulator initialized')

    def joint_state_callback(self, msg):
        """Update robot state from joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]

    def simulate_sensors(self):
        """Simulate various sensors based on robot state"""
        # Simulate camera data
        self.simulate_camera()

        # Simulate LIDAR data
        self.simulate_laser()

        # Simulate force/torque sensors
        self.simulate_force_torque()

    def simulate_imu(self):
        """Simulate IMU data"""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'

        # Simulate linear acceleration (including gravity)
        msg.linear_acceleration.x = np.random.normal(0, self.imu_noise['accel'])
        msg.linear_acceleration.y = np.random.normal(0, self.imu_noise['accel'])
        msg.linear_acceleration.z = -9.8 + np.random.normal(0, self.imu_noise['accel'])

        # Simulate angular velocity
        msg.angular_velocity.x = np.random.normal(0, self.imu_noise['gyro'])
        msg.angular_velocity.y = np.random.normal(0, self.imu_noise['gyro'])
        msg.angular_velocity.z = np.random.normal(0, self.imu_noise['gyro'])

        # Simulate orientation (in a real system, this would be computed from integration)
        msg.orientation.w = 1.0  # Simplified - no rotation
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0

        # Add covariance
        for i in range(3):
            msg.linear_acceleration_covariance[i*4] = self.imu_noise['accel']**2
            msg.angular_velocity_covariance[i*4] = self.imu_noise['gyro']**2

        self.imu_pub.publish(msg)

    def simulate_camera(self):
        """Simulate camera data"""
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height = self.camera_params['height']
        msg.width = self.camera_params['width']
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3  # 3 bytes per pixel (RGB)

        # Generate a simple test image (in reality, this would come from rendering)
        total_pixels = msg.height * msg.width
        image_data = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

        # Add some pattern to make it more realistic
        for i in range(msg.height):
            for j in range(msg.width):
                # Create a gradient pattern
                image_data[i, j, 0] = int(128 + 100 * math.sin(i * 0.1))  # Red channel
                image_data[i, j, 1] = int(128 + 100 * math.cos(j * 0.1))  # Green channel
                image_data[i, j, 2] = int(128 + 50 * math.sin((i+j) * 0.05))  # Blue channel

        # Add noise
        noise = np.random.normal(0, self.camera_params['noise'], image_data.shape)
        image_data = np.clip(image_data + noise, 0, 255).astype(np.uint8)

        # Flatten image data
        msg.data = image_data.tobytes()

        self.camera_pub.publish(msg)

    def simulate_laser(self):
        """Simulate LIDAR data"""
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_link'

        msg.angle_min = self.laser_params['angle_min']
        msg.angle_max = self.laser_params['angle_max']
        msg.angle_increment = self.laser_params['angle_increment']
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = self.laser_params['range_min']
        msg.range_max = self.laser_params['range_max']

        # Calculate number of points
        num_points = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
        msg.ranges = [0.0] * num_points
        msg.intensities = [0.0] * num_points

        # Simulate ranges (in a real system, this would come from raycasting)
        for i in range(num_points):
            # Simulate different distances based on angle
            angle = msg.angle_min + i * msg.angle_increment
            # Simple environment with obstacles at different distances
            distance = 2.0 + math.sin(angle * 3) * 1.5  # Varying distance
            # Add noise
            distance += np.random.normal(0, self.laser_params['noise'])
            # Ensure valid range
            distance = max(msg.range_min, min(msg.range_max, distance))

            msg.ranges[i] = distance
            msg.intensities[i] = 100.0  # Constant intensity for simplicity

        self.laser_pub.publish(msg)

    def simulate_force_torque(self):
        """Simulate force/torque sensor data"""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'force_torque_sensor_frame'

        # Simulate forces based on robot state
        # In a real system, this would come from physics simulation
        msg.vector.x = np.random.normal(0, 0.5)  # Force in X direction
        msg.vector.y = np.random.normal(0, 0.5)  # Force in Y direction
        msg.vector.z = np.random.normal(5.0, 1.0)  # Nominal Z force (weight support)

        self.ft_sensor_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorSimulator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Sensor Simulator stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 5: Create Validation and Monitoring Component

Create `digital_twin_system/digital_twin_system/validation_monitor.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import JointState
from digital_twin_system.msg import DigitalTwinState
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

class DigitalTwinValidator(Node):
    def __init__(self):
        super().__init__('digital_twin_validator')

        # Publishers for validation metrics
        self.status_pub = self.create_publisher(String, '/digital_twin/status', 10)
        self.error_pub = self.create_publisher(Float32, '/digital_twin/validation_error', 10)

        # Subscribers
        self.sync_error_sub = self.create_subscription(
            Float32,
            '/digital_twin/sync_error',
            self.sync_error_callback,
            10
        )

        self.digital_joint_sub = self.create_subscription(
            JointState,
            '/digital_twin/joint_states',
            self.digital_joint_callback,
            10
        )

        self.physical_joint_sub = self.create_subscription(
            JointState,
            '/physical_robot/joint_states',
            self.physical_joint_callback,
            10
        )

        self.state_sub = self.create_subscription(
            DigitalTwinState,
            '/digital_twin/state',
            self.state_callback,
            10
        )

        # Timer for validation checks
        self.timer = self.create_timer(1.0, self.validation_check)

        # Data storage for validation
        self.sync_errors = deque(maxlen=100)
        self.digital_joints = {}
        self.physical_joints = {}
        self.performance_scores = deque(maxlen=100)

        # Validation parameters
        self.error_threshold = 0.05  # Radians
        self.warning_threshold = 0.1  # Radians
        self.validation_window = 10  # Number of samples for trend analysis

        # Timing for metrics
        self.last_validation_time = time.time()

        self.get_logger().info('Digital Twin Validator initialized')

    def sync_error_callback(self, msg):
        """Record synchronization error"""
        self.sync_errors.append(msg.data)

    def digital_joint_callback(self, msg):
        """Record digital joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.digital_joints[name] = msg.position[i]

    def physical_joint_callback(self, msg):
        """Record physical joint positions"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.physical_joints[name] = msg.position[i]

    def state_callback(self, msg):
        """Record performance scores"""
        # This would contain validation metrics from the digital twin system
        self.performance_scores.append(msg.performance_score)

    def validation_check(self):
        """Perform comprehensive validation checks"""
        current_time = time.time()

        # Calculate validation metrics
        avg_sync_error = np.mean(self.sync_errors) if self.sync_errors else 0.0
        max_sync_error = max(self.sync_errors) if self.sync_errors else 0.0
        std_sync_error = np.std(self.sync_errors) if len(self.sync_errors) > 1 else 0.0

        # Calculate trend over last few samples
        recent_errors = list(self.sync_errors)[-self.validation_window:]
        if len(recent_errors) >= 2:
            error_trend = recent_errors[-1] - recent_errors[0]
        else:
            error_trend = 0.0

        # Determine system status
        status = self.determine_status(
            avg_sync_error, max_sync_error, std_sync_error, error_trend
        )

        # Publish status
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

        # Log validation results
        self.get_logger().info(
            f'Validation Check: '
            f'Avg Error: {avg_sync_error:.4f}, '
            f'Max Error: {max_sync_error:.4f}, '
            f'Std Error: {std_sync_error:.4f}, '
            f'Trend: {error_trend:.4f}, '
            f'Status: {status}'
        )

        # Publish validation error for monitoring
        error_msg = Float32()
        error_msg.data = avg_sync_error
        self.error_pub.publish(error_msg)

        # Perform additional validation if needed
        self.perform_detailed_validation()

        # Update timing for next check
        self.last_validation_time = current_time

    def determine_status(self, avg_error, max_error, std_error, trend):
        """Determine system status based on validation metrics"""
        # Primary check: average error
        if avg_error > self.warning_threshold:
            return "WARNING"
        elif avg_error > self.error_threshold:
            return "DEGRADED"

        # Secondary checks
        if max_error > self.warning_threshold * 2:
            return "WARNING"
        if std_error > self.warning_threshold / 2:
            return "DEGRADED"
        if trend > self.warning_threshold / 5:  # Error increasing too rapidly
            return "WARNING"

        # If all checks pass
        return "ACTIVE"

    def perform_detailed_validation(self):
        """Perform more detailed validation checks"""
        # Check for joint-specific errors
        joint_errors = []
        common_joints = set(self.digital_joints.keys()) & set(self.physical_joints.keys())

        for joint_name in common_joints:
            d_pos = self.digital_joints.get(joint_name, 0.0)
            p_pos = self.physical_joints.get(joint_name, 0.0)
            error = abs(d_pos - p_pos)
            joint_errors.append((joint_name, error))

        # Log any joints with high errors
        high_error_joints = [j for j in joint_errors if j[1] > self.warning_threshold]
        if high_error_joints:
            joint_str = ", ".join([f"{j[0]}:{j[1]:.4f}" for j in high_error_joints])
            self.get_logger().warning(f'High error joints: {joint_str}')

def main(args=None):
    rclpy.init(args=args)
    node = DigitalTwinValidator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Digital Twin Validator stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 6: Update setup.py

Update the `setup.py` file:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'digital_twin_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        # Include model files
        (os.path.join('share', package_name, 'models/digital_twin_humanoid'),
         glob('models/digital_twin_humanoid/*')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include rviz files
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Digital twin system for humanoid robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'digital_twin_manager = digital_twin_system.digital_twin_manager:main',
            'state_synchronizer = digital_twin_system.state_synchronizer:main',
            'sensor_simulator = digital_twin_system.sensor_simulator:main',
            'validator = digital_twin_system.validation_monitor:main',
        ],
    },
)
```

#### Step 7: Create Configuration Files

Create `config/robot_properties.yaml`:

```yaml
# Robot-specific properties for digital twin
robot_config:
  # Physical properties
  physical:
    mass: 30.0  # kg
    height: 1.5  # meters
    com_height: 0.8  # center of mass height

  # Joint properties
  joints:
    left_hip:
      position_limit: [-1.57, 1.57]
      velocity_limit: 2.0
      effort_limit: 100.0
    left_knee:
      position_limit: [0.0, 2.35]
      velocity_limit: 2.0
      effort_limit: 150.0
    left_ankle:
      position_limit: [-0.78, 0.78]
      velocity_limit: 1.5
      effort_limit: 50.0
    # Add other joints similarly...

  # Sensor properties
  sensors:
    imu:
      rate: 100  # Hz
      noise:
        acceleration: 0.01
        gyroscope: 0.001
        orientation: 0.001
    camera:
      rate: 30  # Hz
      resolution: [640, 480]
      fov: 1.047  # 60 degrees
    lidar:
      rate: 10  # Hz
      range_min: 0.1
      range_max: 10.0
      resolution: 1.0  # degrees
```

Create `launch/digital_twin_system.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_digital_twin = get_package_share_directory('digital_twin_system')

    # World file
    world_file = PathJoinSubstitution([
        get_package_share_directory('digital_twin_system'),
        'worlds',
        'humanoid_lab.world'
    ])

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={
            'world': world_file,
            'verbose': 'false'
        }.items()
    )

    gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # Digital twin manager
    digital_twin_manager = Node(
        package='digital_twin_system',
        executable='digital_twin_manager',
        name='digital_twin_manager',
        parameters=[
            os.path.join(pkg_digital_twin, 'config', 'robot_properties.yaml'),
            os.path.join(pkg_digital_twin, 'config', 'sync_parameters.yaml')
        ],
        output='screen'
    )

    # State synchronizer
    state_synchronizer = Node(
        package='digital_twin_system',
        executable='state_synchronizer',
        name='state_synchronizer',
        parameters=[
            os.path.join(pkg_digital_twin, 'config', 'robot_properties.yaml')
        ],
        output='screen'
    )

    # Sensor simulator
    sensor_simulator = Node(
        package='digital_twin_system',
        executable='sensor_simulator',
        name='sensor_simulator',
        parameters=[
            os.path.join(pkg_digital_twin, 'config', 'robot_properties.yaml')
        ],
        output='screen'
    )

    # Validator
    validator = Node(
        package='digital_twin_system',
        executable='validator',
        name='digital_twin_validator',
        parameters=[
            os.path.join(pkg_digital_twin, 'config', 'robot_properties.yaml')
        ],
        output='screen'
    )

    # Robot state publisher for visualization
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': True,
            'publish_frequency': 50.0
        }],
        arguments=[os.path.join(pkg_digital_twin, 'models', 'digital_twin_humanoid', 'model.sdf')]
    )

    # Return launch description with delayed start
    return LaunchDescription([
        gazebo,
        gazebo_client,
        TimerAction(
            period=5.0,  # Delay 5 seconds after Gazebo starts
            actions=[
                digital_twin_manager,
                state_synchronizer,
                sensor_simulator,
                validator,
                robot_state_publisher,
            ]
        )
    ])
```

#### Step 8: Create a Custom Message

Create the custom message file `digital_twin_system/msg/DigitalTwinState.msg`:

```
# Digital twin state message
std_msgs/Header header

# Synchronization metrics
float32 synchronization_error
float32 prediction_accuracy
float32 network_latency

# Performance metrics
float32 performance_score

# System status
string status

# Timestamps
builtin_interfaces/Time last_update
builtin_interfaces/Time last_sync
```

Update package.xml to include message dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>digital_twin_system</name>
  <version>0.0.0</version>
  <description>Digital twin system for humanoid robotics</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>

  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Testing and Validation

#### Unit Tests
Create `test/test_synchronizer.py`:

```python
import unittest
import rclpy
from digital_twin_system.state_synchronizer import StateSynchronizer
from sensor_msgs.msg import JointState

class TestStateSynchronizer(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = StateSynchronizer()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_joint_mapping(self):
        """Test that joint names are properly initialized"""
        expected_joints = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle'
            # ... other joints
        ]

        for joint in expected_joints:
            self.assertIn(joint, self.node.physical_joints)
            self.assertIn(joint, self.node.digital_joints)

    def test_joint_state_callback(self):
        """Test joint state callback functionality"""
        # Create a mock joint state message
        msg = JointState()
        msg.name = ['left_hip', 'right_knee']
        msg.position = [1.0, -0.5]

        # Call the callback method
        self.node.physical_joint_callback(msg)

        # Check if values were updated
        self.assertEqual(self.node.physical_joints['left_hip'], 1.0)
        self.assertEqual(self.node.physical_joints['right_knee'], -0.5)

if __name__ == '__main__':
    unittest.main()
```

### Build and Run Instructions

1. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select digital_twin_system
   source install/setup.bash
   ```

2. **Run the complete system**:
   ```bash
   ros2 launch digital_twin_system digital_twin_system.launch.py
   ```

3. **Monitor the system**:
   ```bash
   # View the node graph
   ros2 run rqt_graph rqt_graph

   # Monitor topics
   ros2 topic echo /digital_twin/sync_error
   ros2 topic echo /digital_twin/state
   ros2 topic echo /digital_twin/status
   ```

### Expected Results

After completing this assignment, you should have:

1. A complete digital twin system running in Gazebo
2. Synchronization between physical and digital robot models
3. Real-time monitoring of synchronization accuracy
4. Validation metrics for system performance
5. Proper documentation of the system architecture

### Assessment Criteria

- **Implementation Quality (30%)**: Code organization, efficiency, and adherence to ROS 2 standards
- **Synchronization Accuracy (25%)**: How well the digital twin maintains synchronization with physical robot
- **System Integration (20%)**: Proper integration of all components
- **Validation Metrics (15%)**: Comprehensive monitoring and validation capabilities
- **Documentation (10%)**: Clear and complete documentation

### Submission Requirements

Submit the following components:

1. **Complete Source Code**: All ROS 2 packages with proper structure
2. **Configuration Files**: YAML files and launch files
3. **Documentation**: System architecture and setup guide
4. **Test Results**: Output from unit and integration tests
5. **Performance Analysis**: Results from validation and monitoring
6. **Video Demonstration**: Short video showing the digital twin in operation