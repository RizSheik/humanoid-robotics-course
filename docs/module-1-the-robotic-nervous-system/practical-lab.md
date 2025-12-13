---
title: Practical Lab - Robotic Nervous System Implementation
description: Hands-on lab exercises implementing core concepts of robotic nervous systems
sidebar_position: 102
---

# Practical Lab - Robotic Nervous System Implementation

## Lab Overview

This lab provides hands-on experience implementing and experimenting with the core concepts of robotic nervous systems. Students will work with simulation environments and physical robots to understand sensor integration, control systems, and the integration challenges that arise when building sophisticated robotic systems. The lab emphasizes practical implementation skills while reinforcing theoretical concepts.

## Lab Objectives

By completing this lab, students will be able to:
- Implement basic and advanced sensor fusion algorithms
- Design and test control systems for robotic platforms
- Integrate multiple sensors into a cohesive nervous system
- Evaluate system performance in simulated and real environments
- Identify and address integration challenges in robotic systems

## Prerequisites and Setup

### Software Requirements
- Ubuntu 20.04 or ROS 2 Humble Hawksbill
- Gazebo simulation environment
- Python 3.8+ and necessary libraries (numpy, matplotlib, scipy)
- Robot Operating System (ROS 2) with navigation2 stack
- OpenCV for computer vision tasks
- Basic knowledge of C++ and Python programming

### Hardware Requirements (for physical robot component)
- TurtleBot3 Burger or similar mobile robot platform
- Laptop with Ubuntu 20.04 and ROS 2
- Network connection for robot communication
- Obstacle course materials (cones, boxes, markers)

## Lab Exercise 1: Sensor Integration and Calibration

### Objective
Implement sensor integration and calibration for a mobile robot platform, focusing on IMU, wheel encoders, and camera systems.

### Steps
1. Launch the simulated TurtleBot3 environment:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

2. Write a ROS 2 node to collect sensor data from multiple sources:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
import numpy as np

class SensorIntegrationNode(Node):
    def __init__(self):
        super().__init__('sensor_integration')
        
        # Subscribe to sensor topics
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers for fused data
        self.fused_pub = self.create_publisher(Odometry, '/fused_odom', 10)
        
        # Initialize sensor data storage
        self.imu_data = None
        self.joint_data = None
        self.odom_data = None
        
    def imu_callback(self, msg):
        # Store IMU data
        self.imu_data = msg
        
    def joint_callback(self, msg):
        # Store joint encoder data
        self.joint_data = msg
        
    def odom_callback(self, msg):
        # Store odometry data
        self.odom_data = msg
        
        # Process and publish fused data
        self.process_sensor_data()
    
    def process_sensor_data(self):
        # Implement sensor fusion logic here
        if self.imu_data and self.joint_data:
            # Example fusion algorithm
            fused_msg = Odometry()
            # Your fusion implementation here
            self.fused_pub.publish(fused_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorIntegrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. Implement a calibration routine for the IMU sensor to correct for bias and scale factor errors.

4. Test your implementation by moving the robot in a known pattern and comparing fused odometry with ground truth.

### Deliverables
- Completed sensor fusion node with proper calibration
- Analysis comparing wheel odometry, IMU-only, and fused estimates
- Calibration parameters for IMU bias and scale factors
- Report on sensor fusion performance in different motion patterns

## Lab Exercise 2: Control System Implementation

### Objective
Implement and test various control approaches for robot navigation, comparing PID control, model predictive control, and adaptive control strategies.

### Steps
1. Create a PID controller for robot navigation:
```cpp
#include <vector>

class PIDController {
private:
    double kp_, ki_, kd_;
    double error_sum_, last_error_;
    double last_time_;
    
public:
    PIDController(double kp, double ki, double kd) 
        : kp_(kp), ki_(ki), kd_(kd), error_sum_(0.0), last_error_(0.0) {}

    double compute(double setpoint, double measurement, double dt) {
        double error = setpoint - measurement;
        error_sum_ += error * dt;
        double derivative = (error - last_error_) / dt;
        
        double output = kp_ * error + ki_ * error_sum_ + kd_ * derivative;
        
        last_error_ = error;
        last_time_ += dt;
        
        return output;
    }
};
```

2. Implement a path following algorithm using the PID controller:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
import math

class PathFollowerNode(Node):
    def __init__(self):
        super().__init__('path_follower')
        
        self.subscription = self.create_subscription(
            Path, '/path_to_follow', self.path_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.path = []
        self.current_waypoint = 0
        self.linear_pid = PIDController(1.0, 0.0, 0.1)  # kp, ki, kd
        self.angular_pid = PIDController(2.0, 0.0, 0.2)
        
    def path_callback(self, msg):
        self.path = msg.poses
        self.current_waypoint = 0
        
    def follow_path(self):
        if not self.path or self.current_waypoint >= len(self.path):
            return
            
        # Get current robot position (subscribe to odometry)
        # Calculate distance to current waypoint
        # Use PID to compute velocity commands
        cmd_vel = Twist()
        
        # Implement your path following logic here
        # Use PID controllers for linear and angular velocity
        
        self.velocity_publisher.publish(cmd_vel)
```

3. Test the controller on different path types (straight lines, curves, sharp turns).

4. Compare the performance of PID control with a simple model predictive control approach for path following.

### Deliverables
- Working PID-based path follower
- Performance comparison of different control strategies
- Analysis of tracking accuracy for different path types
- Tuned PID parameters with justification

## Lab Exercise 3: Advanced Sensor Fusion

### Objective
Implement an Extended Kalman Filter (EKF) for fusing multiple sensor modalities to improve robot state estimation.

### Steps
1. Implement the EKF algorithm for state estimation:
```python
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # Initialize state vector [x, y, theta, vx, vy, omega]
        self.x = np.zeros(state_dim)
        
        # Initialize covariance matrix
        self.P = np.eye(state_dim) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(state_dim) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(measurement_dim) * 0.1
    
    def predict(self, control_input, dt):
        # State transition model (nonlinear)
        F = self.jacobian_f(self.x, control_input, dt)
        self.x = self.state_transition(self.x, control_input, dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement):
        # Measurement model Jacobian
        H = self.jacobian_h(self.x)
        
        # Innovation
        y = measurement - self.measurement_model(self.x)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
    
    def state_transition(self, x, u, dt):
        # Implement nonlinear state transition
        # x, y, theta, vx, vy, omega
        new_x = x.copy()
        new_x[0] += x[3] * dt  # x position
        new_x[1] += x[4] * dt  # y position
        new_x[2] += x[5] * dt  # theta
        # Add velocity updates based on control input
        return new_x
    
    def jacobian_f(self, x, u, dt):
        # Compute Jacobian of state transition
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dtheta/domega
        return F
    
    def measurement_model(self, x):
        # Measurement function (e.g., [x, y, theta] from IMU and encoders)
        return np.array([x[0], x[1], x[2]])
    
    def jacobian_h(self, x):
        # Jacobian of measurement model
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = 1  # dx/dx
        H[1, 1] = 1  # dy/dy
        H[2, 2] = 1  # dtheta/dtheta
        return H
```

2. Integrate the EKF into a ROS 2 node that receives multiple sensor inputs:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_localization')
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)
        
        # Subscribe to sensors
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        # Publish fused state estimate
        self.pose_pub = self.create_publisher(Odometry, '/ekf_pose', 10)
        
        # Timer for prediction step
        self.timer = self.create_timer(0.05, self.predict_callback)  # 20Hz
        
        self.last_cmd_time = self.get_clock().now()
        
    def imu_callback(self, msg):
        # Extract measurement from IMU
        measurement = np.array([
            msg.orientation.x,
            msg.orientation.y, 
            msg.orientation.z
        ])
        
        # Update EKF with IMU measurement
        self.ekf.update(measurement)
        
    def joint_callback(self, msg):
        # Extract position and velocity from encoders
        # Update EKF with encoder measurement
        pass
        
    def cmd_callback(self, msg):
        # Store control input for prediction step
        self.last_cmd = msg
        self.last_cmd_time = self.get_clock().now()
    
    def predict_callback(self):
        # Get time difference
        current_time = self.get_clock().now()
        dt = (current_time - self.last_cmd_time).nanoseconds / 1e9
        
        # Get control input (if available)
        if hasattr(self, 'last_cmd'):
            control_input = np.array([
                self.last_cmd.linear.x,
                self.last_cmd.angular.z
            ])
        else:
            control_input = np.array([0.0, 0.0])
        
        # Perform prediction step
        self.ekf.predict(control_input, dt)
        
        # Publish current estimate
        self.publish_state()
    
    def publish_state(self):
        # Convert EKF state to ROS message
        msg = Odometry()
        # Fill in the message with EKF state and covariance
        self.pose_pub.publish(msg)
```

3. Test the EKF implementation in simulation with different motion patterns and compare its performance to individual sensor estimates.

4. Add simulated sensor noise to evaluate the filter's robustness.

### Deliverables
- Working EKF implementation integrated with ROS 2
- Performance comparison with individual sensors
- Analysis of filter convergence and stability
- Results with noisy sensor data

## Lab Exercise 4: Integration Challenge

### Objective
Combine all components learned in previous exercises to create a complete robotic nervous system that can navigate to goals while avoiding obstacles and maintaining accurate state estimation.

### Steps
1. Create a complete system node that integrates:
   - Sensor fusion (EKF for state estimation)
   - Path planning and following
   - Obstacle detection and avoidance
   - Higher-level navigation goals

2. Implement a behavior arbitration system that handles conflicts between different goals (reach goal vs. avoid obstacles):
```python
class BehaviorArbitrator:
    def __init__(self):
        self.behaviors = {
            'path_following': PathFollower(),
            'obstacle_avoidance': ObstacleAvoider(),
            'goal_seeking': GoalSeeker()
        }
        
    def arbitrate(self, sensor_data, goal):
        # Get candidate actions from each behavior
        candidates = {}
        for name, behavior in self.behaviors.items():
            candidates[name] = behavior.compute_action(sensor_data, goal)
        
        # Arbitration logic based on situation
        if self.is_obstacle_immediate(sensor_data):
            return candidates['obstacle_avoidance']
        else:
            return self.blend_actions(candidates)
    
    def blend_actions(self, candidates):
        # Blend path following with other objectives
        primary = candidates['path_following']
        secondary = candidates['goal_seeking']
        
        # Weighted combination based on context
        blended = Twist()
        blended.linear.x = 0.7 * primary.linear.x + 0.3 * secondary.linear.x
        blended.angular.z = 0.6 * primary.angular.z + 0.4 * secondary.angular.z
        
        return blended
```

3. Test the complete system in simulation with:
   - Static obstacles
   - Dynamic obstacles
   - Multiple goals in sequence
   - Sensor failures (simulate by disabling certain inputs)

4. Document the trade-offs between different system design choices.

### Deliverables
- Complete integrated system
- Test results across different scenarios
- Analysis of system robustness
- Documentation of design trade-offs

## Assessment Rubric

### Technical Implementation (50%)
- Correct implementation of algorithms
- Proper ROS 2 integration
- Efficient code structure
- Handling of edge cases

### System Performance (30%)
- Accuracy of sensor fusion
- Effectiveness of control algorithms
- Robustness to disturbances
- Real-time performance

### Analysis and Documentation (20%)
- Clear analysis of results
- Proper experimental methodology
- Identification of system limitations
- Suggestions for improvements

## Additional Resources

### Sample Configuration Files
- ROS 2 launch files for simulation environment
- Parameter files for controller tuning
- Gazebo world files for different testing scenarios

### Reference Implementations
- Working examples of sensor fusion algorithms
- Benchmark implementations for comparison
- Common troubleshooting solutions

This lab provides comprehensive hands-on experience with implementing robotic nervous system concepts, from basic sensor integration to advanced multi-modal systems that exhibit sophisticated behaviors.