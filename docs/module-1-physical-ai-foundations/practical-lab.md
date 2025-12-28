# Module 1: Practical Lab - Hands-On Physical AI and Embodied Intelligence

## Lab Overview

This practical lab provides hands-on experience with the principles of Physical AI and embodied intelligence using simulation environments. The lab is structured to progress from simple concepts to more complex applications, allowing students to experience how physical embodiment can enhance robot intelligence.

### Learning Objectives

After completing this lab, students will be able to:
1. Implement and evaluate simple embodied agents in simulation
2. Design control systems that exploit morphological computation
3. Apply sensorimotor learning to robotic tasks
4. Analyze the trade-offs between embodied and traditional approaches
5. Understand the role of physical dynamics in robotic intelligence

### Required Software/Tools

- ROS 2 (Humble Hawksbill or later)
- Gazebo Harmonic or Isaac Sim
- Python 3.11+
- Basic knowledge of ROS 2 packages and nodes

### Lab Duration

This lab is designed for 12-15 hours of work, typically spread over 2-3 weeks.

## Lab 1: Basic Embodied Agent

### Objective
Create a simple wheeled robot that demonstrates basic embodied intelligence by navigating using only local sensory information and simple reactive behaviors.

### Setup
1. Launch a Gazebo simulation with a simple wheeled robot in an environment with obstacles.
2. The robot should have basic sensors (e.g., forward, left, right distance sensors) and differential drive control.

### Implementation Steps
1. Create a ROS 2 node that subscribes to sensor data and publishes velocity commands
2. Implement a simple reactive controller that:
   - Moves forward when no obstacles are near
   - Turns right when obstacle is ahead
   - Turns left when obstacles are ahead and right
   - Turns right when only left has an obstacle

### Analysis
- Observe how the robot behavior emerges from the interaction between the body (wheels), environment (obstacles), and simple control rules
- Compare the robustness of this approach to a path-planning approach
- Discuss how the robot's physical embodiment contributes to its behavior

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class EmbodiedController(Node):
    def __init__(self):
        super().__init__('embodied_controller')
        self.subscription = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.scan_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
    def scan_callback(self, msg):
        # Get distance readings in front, left, right
        front = min(msg.ranges[0:10] + msg.ranges[-10:])
        left = min(msg.ranges[80:100])
        right = min(msg.ranges[260:280])
        
        # Simple reactive controller
        cmd = Twist()
        if front > 1.0:  # No obstacle ahead
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
        elif right > 1.0:  # Turn right if clear
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif left > 1.0:  # Turn left if clear
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:  # Turn left as default
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
            
        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = EmbodiedController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 2: Morphological Computation

### Objective
Demonstrate morphological computation by creating controllers that take advantage of the physical properties of the robot's body rather than relying on complex algorithms.

### Setup
1. Create two simulated robots in Gazebo:
   - A simple manipulator with passive compliance at the end-effector
   - A manipulator with precise, non-compliant joints
2. Set up the same pick-and-place task for both robots

### Implementation Steps
1. Create a simple position controller for the compliant robot that doesn't need to account for object compliance
2. Create a complex force-position controller for the non-compliant robot that must carefully manage contact forces
3. Compare the performance of both approaches

### Analysis
- Document the differences in control complexity between the two approaches
- Explain how the physical compliance of the first robot performs computational work
- Discuss the trade-offs between morphological computation and algorithmic computation

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class MorphologicalComputation(Node):
    def __init__(self):
        super().__init__('morphological_comp')
        self.joint_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.state_sub = self.create_subscription(JointState, 'joint_states', self.state_callback, 10)
        
        # Target position for pick-and-place task
        self.target_pos = np.array([0.5, 0.2, 0.1])  # x, y, z in meters
        
    def state_callback(self, msg):
        # Get current end-effector position (simplified)
        # In practice, you'd use forward kinematics
        current_pos = np.array([msg.position[0], msg.position[1], msg.position[2]])
        
        # Simple PD controller for compliant robot (the compliance handles force control)
        error = self.target_pos - current_pos
        cmd = Float64MultiArray()
        cmd.data = [0.1 * e + 0.02 * de for e, de in zip(error, self.prev_error - error)]
        
        self.joint_pub.publish(cmd)
        self.prev_error = error

def main(args=None):
    rclpy.init(args=args)
    controller = MorphologicalComputation()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 3: Sensorimotor Learning

### Objective
Implement a basic sensorimotor learning system where a robot learns to interact with its environment through trial and error.

### Setup
1. Create a simulation environment with a simple manipulator arm
2. Configure the environment to provide a reward signal based on task performance
3. Set up sensors (position, force, tactile)

### Implementation Steps
1. Implement a simple neural network controller for the robot
2. Use a reinforcement learning algorithm (such as Q-learning or SARSA) to learn a reaching task
3. Compare learning with and without different sensory modalities
4. Analyze how the sensorimotor loop affects learning

### Analysis
- Plot learning curves showing how performance improves over time
- Compare the effectiveness of different sensory inputs for learning
- Discuss how the physical embodiment affects the learning process

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from geometry_msgs.msg import Twist
import numpy as np
import torch
import torch.nn as nn

class SensorimotorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SensorimotorLearner(Node):
    def __init__(self):
        super().__init__('sensorimotor_learner')
        self.subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        
        self.reward_sub = self.create_subscription(
            Twist, 'reward_signal', self.reward_callback, 10)
        
        self.pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Neural network and learning parameters
        self.network = SensorimotorNetwork(input_size=10, hidden_size=20, output_size=3)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    def joint_callback(self, msg):
        # Create state vector from joint states and other sensors
        state = np.array([
            msg.position[0], msg.position[1], msg.position[2],
            msg.velocity[0], msg.velocity[1], msg.velocity[2],
            # Add other sensor values
        ])
        
        # Convert to tensor and get action
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = self.network(state_tensor)
        
        # Convert action to ROS message and publish
        cmd = JointState()
        cmd.position = [0.0, float(action[0]), float(action[1]), float(action[2])]
        self.pub.publish(cmd)
        
        # Store for learning
        self.state_memory.append(state_tensor)
        self.action_memory.append(action)
        
    def reward_callback(self, msg):
        # Store reward for learning
        self.reward_memory.append(msg.linear.x)  # Assuming reward in x component

def main(args=None):
    rclpy.init(args=args)
    learner = SensorimotorLearner()
    
    # Training loop (in practice, this would run periodically)
    # for episode in range(1000):
    #     # Training code here
    #     pass
    
    rclpy.spin(learner)
    learner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 4: Balance and Dynamic Control

### Objective
Implement a simple balance controller for a bipedal robot, demonstrating the principles of dynamic balance using embodied intelligence.

### Setup
1. Load a simplified bipedal robot model in simulation
2. Configure sensors for balance (IMU, joint encoders, force/torque sensors)
3. Set up a challenge course with small obstacles or uneven terrain

### Implementation Steps
1. Implement a basic balance controller using the inverted pendulum model
2. Use sensor feedback to maintain the Zero Moment Point (ZMP) within the support polygon
3. Implement basic stepping strategies to maintain balance when disturbed
4. Test the controller with different disturbances (pushes, uneven terrain)

### Analysis
- Analyze the relationship between sensor feedback and balance maintenance
- Document the robot's response to different types of disturbances
- Discuss how the physical embodiment (compliance, inertia, etc.) contributes to stability

### Code Template
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
import numpy as np

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')
        
        # Subscriptions
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Balance control parameters
        self.com_height = 0.8  # Center of mass height (meters)
        self.gravity = 9.81
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_error_x = 0.0
        self.integral_error_y = 0.0
        
    def imu_callback(self, msg):
        # Extract orientation and angular velocity
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        
        # Calculate roll and pitch angles (simplified)
        pitch = np.arcsin(2.0 * (orientation.w * orientation.y - orientation.z * orientation.x))
        roll = np.arctan2(2.0 * (orientation.w * orientation.x + orientation.y * orientation.z), 
                         1.0 - 2.0 * (orientation.x**2 + orientation.y**2))
        
        # Calculate desired ZMP based on orientation
        desired_zmp_x = -self.com_height / self.gravity * pitch
        desired_zmp_y = -self.com_height / self.gravity * roll
        
        # Current ZMP (simplified - in a real system, you'd use force/torque sensors)
        current_zmp_x = 0.0  # Would come from force sensors
        current_zmp_y = 0.0  # Would come from force sensors
        
        # Control law using PID for ZMP tracking
        error_x = desired_zmp_x - current_zmp_x
        error_y = desired_zmp_y - current_zmp_y
        
        # PID control
        kp = 100.0  # Proportional gain
        ki = 1.0    # Integral gain
        kd = 5.0    # Derivative gain
        
        self.integral_error_x += error_x
        derivative_error_x = (error_x - self.prev_error_x)
        
        self.integral_error_y += error_y
        derivative_error_y = (error_y - self.prev_error_y)
        
        control_output_x = kp * error_x + ki * self.integral_error_x + kd * derivative_error_x
        control_output_y = kp * error_y + ki * self.integral_error_y + kd * derivative_error_y
        
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # Generate joint commands based on control outputs
        cmd = JointState()
        # This is a simplified representation - in reality you'd map these to actual joint torques
        # based on your robot's specific kinematics and dynamics
        cmd.position = [0.0, control_output_x, control_output_y, 0.0]
        self.cmd_pub.publish(cmd)

    def joint_callback(self, msg):
        # Process joint positions and velocities for balance control if needed
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = BalanceController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab Report Requirements

For each lab exercise, students must submit:
1. A written report describing their approach and implementation
2. Code files with proper documentation
3. Analysis of results, including plots where appropriate
4. Discussion of how the experiments demonstrated embodied intelligence principles
5. Suggestions for improvements or extensions

## Assessment Criteria

- Implementation quality and correctness (30%)
- Understanding of embodied intelligence principles (30%)
- Analysis and interpretation of results (25%)
- Documentation and code quality (15%)

## Lab Extension Ideas

1. **Advanced Embodiment:** Implement controllers that use more sophisticated morphological computation techniques
2. **Soft Robotics:** Experiment with soft-bodied robots and their control challenges
3. **Collective Embodiment:** Develop multiple agents that exhibit collective intelligence through embodiment
4. **Learning Algorithms:** Implement more sophisticated learning algorithms for sensorimotor tasks

## Troubleshooting Tips

1. **Simulation Issues:** Ensure ROS 2 and Gazebo are properly installed and sourced
2. **Controller Instability:** Check gains in PID controllers and simulation time constants
3. **Sensor Noise:** Implement basic filtering if sensor signals are too noisy
4. **Performance:** Optimize algorithms if running slowly in simulation

## References and Further Reading

- Dudek, G., & Jenkin, M. (2010). Computational Principles of Mobile Robotics.
- Pfeifer, R., & Bongard, J. (2006). How the Body Shapes the Way We Think.
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.