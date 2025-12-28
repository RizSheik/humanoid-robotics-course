# Module 1: Simulation - Physical AI and Embodied Intelligence in Practice

## Simulation Overview

This simulation module focuses on implementing and testing Physical AI and embodied intelligence concepts in virtual environments. Students will work with simulation tools to understand how physical embodiment, sensorimotor loops, and environmental interaction contribute to intelligent behavior.

### Learning Objectives

After completing this simulation module, students will be able to:
1. Set up and configure simulation environments for embodied robotics
2. Implement embodied agents that demonstrate morphological computation
3. Test sensorimotor learning algorithms in simulated environments
4. Analyze the effects of physical embodiment on robot behavior and learning
5. Transfer knowledge from simulation to potential real-world applications

### Required Simulation Tools

- **Gazebo Harmonic** or **Isaac Sim** for physics simulation
- **ROS 2 Humble Hawksbill** for robot control and communication
- **Python 3.11+** for implementing controllers and learning algorithms
- **PyTorch/TensorFlow** for neural network implementations
- **OpenAI Gym/Gymnasium** or custom environments for learning tasks

## Simulation Environment Setup

### Gazebo Configuration

For Gazebo-based simulations:

1. Create a simulation world with appropriate physics properties:
   - Set gravity (`<gravity>9.81 0 0</gravity>`)
   - Configure physical properties (friction, restitution)
   - Define sensor parameters (noise, update rates)

2. Define robot models with accurate physical properties:
   - Mass, inertia, and center of mass for each link
   - Joint limits and dynamics (friction, damping)
   - Sensor placement and parameters

3. Create appropriate environments for testing:
   - Simple environments for basic testing
   - Complex environments for advanced challenges
   - Environments with varied terrain for locomotion

### Isaac Sim Configuration

For Isaac Sim-based simulations:

1. Set up the environment with accurate physics parameters
2. Import robot models with proper articulation
3. Configure sensors (camera, LiDAR, IMU, force/torque)
4. Set up reward functions for learning tasks

## Simulation 1: Passive Dynamic Walking

### Objective
Demonstrate how physical embodiment can produce complex behaviors (walking) without active control, using only the interaction between body dynamics and environment.

### Implementation
Create a simple 2D passive dynamic walker in simulation:

```xml
<!-- Passive Walker URDF -->
<?xml version="1.0"?>
<robot name="passive_walker">
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.3 0.1 0.5"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.3 0.1 0.5"/>
      </geometry>
    </collision>
  </link>
  
  <link name="left_thigh">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.25"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Similar definition for right leg -->
  <joint name="left_hip" type="continuous">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0 0.1 0.1"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### Simulation Code
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import numpy as np

class PassiveWalkerSim(Node):
    def __init__(self):
        super().__init__('passive_walker_sim')
        
        # Physics parameters
        self.gravity = 9.81
        self.foot_length = 0.1  # Length of feet
        self.time_step = 0.001  # Simulation time step
        
        # For a passive walker, we only need to set initial conditions
        # The walking emerges from the physical interaction
        
        # Initial small push to start walking
        self.initial_push = True
        self.push_time = 0
        
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
        # Timer for simulation loop
        self.timer = self.create_timer(self.time_step, self.simulation_step)
        
    def simulation_step(self):
        # For a truly passive walker, no active control is applied
        # The walking emerges from the interaction between gravity, 
        # the slope of the surface, and the physical configuration
        
        # However, for simulation purposes, we might apply a small initial push
        if self.initial_push and self.push_time < 0.5:  # Push for 0.5 seconds
            self.push_time += self.time_step
            # Apply a small horizontal force to initiate motion
            # In a real simulation, this would be handled by the physics engine
            pass
        elif self.push_time >= 0.5:
            self.initial_push = False

def main(args=None):
    rclpy.init(args=args)
    sim = PassiveWalkerSim()
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Observe how different leg angles and masses affect walking stability
- Analyze the energy efficiency of the passive walk
- Compare with active control approaches

## Simulation 2: Morphological Computation in Manipulation

### Objective
Implement and test how physical properties of a robot can perform computational work that would otherwise require complex algorithms.

### Setup
1. Create two simulated grippers in Gazebo:
   - Rigid gripper with precise force control
   - Underactuated compliant gripper (e.g., tendon-driven or spring-loaded)
2. Place various objects of different shapes and stiffnesses

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, ForceTorqueSensor
from std_msgs.msg import Float64MultiArray
import numpy as np

class MorphologicalComputationSim(Node):
    def __init__(self):
        super().__init__('morph_comp_sim')
        
        # Publishers for each gripper type
        self.rigid_cmd_pub = self.create_publisher(Float64MultiArray, 'rigid_gripper/commands', 10)
        self.compliant_cmd_pub = self.create_publisher(Float64MultiArray, 'compliant_gripper/commands', 10)
        
        # Subscribers for gripper states
        self.rigid_state_sub = self.create_subscription(JointState, 'rigid_gripper/states', self.rigid_state_callback, 10)
        self.compliant_state_sub = self.create_subscription(JointState, 'compliant_gripper/states', self.compliant_state_callback, 10)
        
        # Force feedback for rigid gripper
        self.rigid_force_sub = self.create_subscription(ForceTorqueSensor, 'rigid_gripper/force', self.rigid_force_callback, 10)
        
        self.rigid_force = 0.0
        self.compliant_grasp_success = False
        
        # Simulation step timer
        self.timer = self.create_timer(0.01, self.control_step)
        
    def rigid_state_callback(self, msg):
        # Store rigid gripper state
        self.rigid_position = msg.position
        self.rigid_velocity = msg.velocity
        
    def compliant_state_callback(self, msg):
        # Store compliant gripper state
        self.compliant_position = msg.position
        self.compliant_velocity = msg.velocity
        
    def rigid_force_callback(self, msg):
        # Store force feedback
        self.rigid_force = msg.wrench.force.x  # Simplified to x-axis force
        
    def control_step(self):
        # Control for rigid gripper - precise force control needed
        rigid_cmd = Float64MultiArray()
        
        # More complex control for rigid gripper to manage force
        if self.rigid_force < 5.0:  # Target force
            # Increase grip strength
            rigid_cmd.data = [0.8]  # More aggressive approach needed
        elif self.rigid_force > 10.0:  # Maximum safe force
            # Decrease grip strength
            rigid_cmd.data = [0.2]  # Reduce pressure
        else:
            # Maintain current grip
            rigid_cmd.data = [0.5]
        
        self.rigid_cmd_pub.publish(rigid_cmd)
        
        # Control for compliant gripper - simpler control due to physical compliance
        compliant_cmd = Float64MultiArray()
        
        # Simple position control for compliant gripper
        # The physical compliance handles force regulation
        compliant_cmd.data = [0.7]  # Close to object, let compliance handle force
        
        self.compliant_cmd_pub.publish(compliant_cmd)
        
    def evaluate_grasps(self):
        # Evaluate grasp success based on force and object position
        # This would be called periodically to assess performance
        pass

def main(args=None):
    rclpy.init(args=args)
    sim = MorphologicalComputationSim()
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Compare the control complexity between both approaches
- Measure grasp success rates for different object types
- Analyze energy efficiency of each approach
- Document how physical compliance reduces control requirements

## Simulation 3: Sensorimotor Learning in Navigation

### Objective
Implement and test a sensorimotor learning algorithm where a robot learns to navigate through environmental interaction.

### Setup
1. Create a maze environment in the simulation
2. Configure a mobile robot with proximity sensors
3. Set up reward system based on distance to goal and collision penalties

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float64
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class LinearQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearQNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.fc(x)

class SensorimotorLearningSim(Node):
    def __init__(self):
        super().__init__('sensorimotor_learning_sim')
        
        self.subscription = self.create_subscription(
            LaserScan, 'laser_scan', self.scan_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float64, 'reward', 10)
        
        # RL parameters
        self.state_size = 5  # Number of sensor readings used
        self.action_size = 4  # 4 discrete actions: forward, turn left, turn right, slight turn
        self.learning_rate = 0.001
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.memory = deque(maxlen=10000)
        
        # Neural networks
        self.q_network = LinearQNetwork(self.state_size, self.action_size)
        self.target_network = LinearQNetwork(self.state_size, self.action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Action definitions
        self.actions = {
            0: (0.5, 0.0),    # Forward
            1: (0.2, 0.5),    # Turn left
            2: (0.2, -0.5),   # Turn right
            3: (0.4, 0.1)     # Slight turn
        }
        
        # Game state
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None
        self.steps = 0
        
        # Goal position (simplified - in real sim would get from TF or parameters)
        self.goal_x = 5.0
        self.goal_y = 5.0
        
        # Timer for decision making
        self.timer = self.create_timer(0.1, self.decision_step)
        
    def scan_callback(self, msg):
        # Process laser scan to get state representation
        # Take readings from front, front-left, front-right, left, right
        front = min(min(msg.ranges[0:10]), min(msg.ranges[350:359]))
        front_left = min(msg.ranges[45:55])
        front_right = min(msg.ranges[305:315])
        left = min(msg.ranges[80:100])
        right = min(msg.ranges[260:280])
        
        # Normalize distances to 0-1 range (max sensor range assumed to be 10m)
        state = np.array([
            min(1.0, front/10.0),
            min(1.0, front_left/10.0), 
            min(1.0, front_right/10.0),
            min(1.0, left/10.0),
            min(1.0, right/10.0)
        ])
        
        self.current_state = state
        
    def get_robot_position(self):
        # In real implementation, get from TF or robot state
        # For simulation, we might track this internally or get from odometry
        return (0, 0)  # Placeholder
    
    def calculate_reward(self, action, sensor_data):
        # Calculate reward based on sensor readings and action taken
        # Positive reward for moving toward goal, negative for collisions
        x, y = self.get_robot_position()
        
        # Distance to goal (simplified)
        dist_to_goal = np.sqrt((x - self.goal_x)**2 + (y - self.goal_y)**2)
        
        # Calculate reward
        reward = 0
        
        # Reward for moving closer to goal
        reward += (10 - dist_to_goal) * 0.1  # Higher reward when closer
        
        # Penalty for being close to obstacles
        if sensor_data[0] < 0.3:  # Front sensor close to obstacle
            reward -= 5.0
        elif sensor_data[0] < 0.5:
            reward -= 1.0
            
        # Small penalty for taking any action (to encourage efficiency)
        reward -= 0.01
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decision_step(self):
        if self.current_state is None:
            return
            
        # Choose action using epsilon-greedy
        if random.random() < self.epsilon:
            # Explore
            action = random.randrange(self.action_size)
        else:
            # Exploit
            state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        
        # Execute action
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = self.actions[action]
        self.cmd_pub.publish(cmd)
        
        # Calculate reward
        reward = self.calculate_reward(action, self.current_state)
        
        # Store experience for learning
        if self.previous_state is not None:
            self.remember(
                self.previous_state, 
                self.previous_action, 
                self.previous_reward, 
                self.current_state, 
                False
            )
        
        # Update previous values for next iteration
        self.previous_state = self.current_state.copy()
        self.previous_action = action
        self.previous_reward = reward
        
        # Publish reward for monitoring
        reward_msg = Float64()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)
        
        # Learn from experience
        if len(self.memory) > 1000:
            self.replay()
            if self.steps % 100 == 0:
                self.update_target_network()
                
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps += 1
        
        # Reset if episode ends (simplified - would typically have proper episodes)
        if self.steps > 10000:  # Reset after 10000 steps
            self.steps = 0
            self.reset_environment()

    def reset_environment(self):
        # In a real simulation, reset robot to starting position
        # For this example, we just reset some parameters
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None
        self.get_logger().info("Environment reset")

def main(args=None):
    rclpy.init(args=args)
    sim = SensorimotorLearningSim()
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Plot learning curves showing improvement over time
- Analyze which sensorimotor patterns led to successful navigation
- Compare performance with traditional path-planning approaches
- Discuss the role of embodiment in the learning process

## Simulation 4: Dynamic Balance and Control

### Objective
Simulate dynamic balance control for a bipedal robot, demonstrating how embodied intelligence principles can maintain stability.

### Implementation
```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import Float64MultiArray
import numpy as np

class BalanceControllerSim(Node):
    def __init__(self):
        super().__init__('balance_controller_sim')
        
        # Subscriptions
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        
        # Publishers
        self.joint_cmd_pub = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.zmp_pub = self.create_publisher(Point, 'zmp', 10)
        
        # Balance control parameters
        self.com_height = 0.8  # Center of mass height (meters)
        self.gravity = 9.81
        self.control_frequency = 100  # Hz
        self.time_step = 1.0 / self.control_frequency
        
        # LIPM parameters
        self.desired_com_pos = np.array([0.0, 0.0])  # Desired CoM position in x-y plane
        self.desired_com_vel = np.array([0.0, 0.0])
        
        # PID controller gains for ZMP control
        self.kp = 100.0
        self.ki = 10.0
        self.kd = 50.0
        
        # Integral and derivative terms
        self.zmp_error_integral = np.array([0.0, 0.0])
        self.prev_zmp_error = np.array([0.0, 0.0])
        
        # Timer for control loop
        self.timer = self.create_timer(1.0/self.control_frequency, self.balance_control_step)
        
        # Robot state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.joint_positions = []
        self.joint_velocities = []
        
    def imu_callback(self, msg):
        # Extract orientation from quaternion
        q = msg.orientation
        self.roll = np.arctan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x**2 + q.y**2))
        self.pitch = np.arcsin(2.0 * (q.w * q.y - q.z * q.x))
        self.yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
        
        # Extract angular velocity
        self.angular_velocity[0] = msg.angular_velocity.x
        self.angular_velocity[1] = msg.angular_velocity.y
        self.angular_velocity[2] = msg.angular_velocity.z
        
    def joint_callback(self, msg):
        self.joint_positions = msg.position
        self.joint_velocities = msg.velocity
        
    def balance_control_step(self):
        # Calculate current ZMP (simplified - in reality would use force/torque sensors)
        # For this simulation, we'll estimate ZMP from IMU and kinematic data
        estimated_zmp = self.estimate_zmp()
        
        # Calculate error between desired and actual ZMP
        # For this simple case, we want to keep ZMP at (0,0) - center of support
        zmp_error = self.desired_com_pos - estimated_zmp
        
        # PID control for ZMP tracking
        self.zmp_error_integral += zmp_error * self.time_step
        zmp_error_derivative = (zmp_error - self.prev_zmp_error) / self.time_step
        
        # Calculate control output
        zmp_control_output = (
            self.kp * zmp_error + 
            self.ki * self.zmp_error_integral + 
            self.kd * zmp_error_derivative
        )
        
        self.prev_zmp_error = zmp_error.copy()
        
        # Convert control output to joint commands
        # This is a simplified approach - in reality, this would involve
        # inverse kinematics, whole-body control, or other advanced techniques
        joint_commands = self.compute_joint_commands(zmp_control_output)
        
        # Publish commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_commands
        self.joint_cmd_pub.publish(cmd_msg)
        
        # Publish ZMP for visualization
        zmp_msg = Point()
        zmp_msg.x = estimated_zmp[0]
        zmp_msg.y = estimated_zmp[1]
        zmp_msg.z = 0.0  # ZMP is a point on the ground
        self.zmp_pub.publish(zmp_msg)
        
        # Log balance status
        self.get_logger().debug(f'ZMP Error: {zmp_error}, Control Output: {zmp_control_output}')
    
    def estimate_zmp(self):
        """
        Estimate Zero Moment Point from IMU and kinematic data.
        In a real robot, this would come from foot force/torque sensors.
        """
        # Simplified estimation based on orientation
        # In practice, this requires force/torque sensors in the feet
        estimated_zmp_x = -self.com_height / self.gravity * self.pitch
        estimated_zmp_y = self.com_height / self.gravity * self.roll
        
        return np.array([estimated_zmp_x, estimated_zmp_y])
    
    def compute_joint_commands(self, control_output):
        """
        Convert ZMP control output to joint commands.
        This is a simplified approach - in reality, this would be much more complex.
        """
        # For this simulation, we'll map control outputs to simple hip and ankle adjustments
        # This is greatly simplified - a real implementation would use whole-body control
        
        # Extract control commands
        x_cmd = control_output[0]
        y_cmd = control_output[1]
        
        # Map to joint adjustments (simplified)
        commands = [0.0] * 12  # Assuming 12 joints for a simple biped
        
        # Hip adjustments
        commands[0] = x_cmd * 0.1  # Left hip roll
        commands[1] = y_cmd * 0.1  # Left hip pitch
        commands[6] = -x_cmd * 0.1  # Right hip roll
        commands[7] = y_cmd * 0.1  # Right hip pitch
        
        # Ankle adjustments for balance
        commands[2] = -y_cmd * 0.05  # Left ankle pitch
        commands[3] = x_cmd * 0.05   # Left ankle roll
        commands[8] = -y_cmd * 0.05  # Right ankle pitch
        commands[9] = -x_cmd * 0.05  # Right ankle roll
        
        return commands

def main(args=None):
    rclpy.init(args=args)
    sim = BalanceControllerSim()
    
    # Update target network periodically during simulation
    sim.timer2 = sim.create_timer(1.0, sim.update_target_network)
    
    rclpy.spin(sim)
    sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate balance stability under different perturbations
- Analyze the role of physical parameters (COM height, foot size) on stability
- Test recovery from various disturbances (pushes, uneven terrain)
- Compare with static balance approaches

## Simulation Evaluation and Transfer

### Simulation-to-Reality Gap Analysis

Discuss the challenges and methods for addressing the sim-to-real gap:

1. **Visual Domain Randomization**: Vary lighting, textures, and camera parameters
2. **Dynamics Randomization**: Randomize physical parameters (masses, frictions, etc.)
3. **Noise Injection**: Add realistic sensor and actuator noise
4. **System Identification**: Tune simulation parameters to match real robot behavior

### Metrics for Evaluation

1. **Task Performance Metrics**:
   - Success rate
   - Time to complete tasks
   - Energy efficiency
   - Stability measures

2. **Embodied Intelligence Metrics**:
   - Morphological computation index
   - Sensorimotor loop efficiency
   - Adaptation speed to environmental changes

3. **Learning Performance Metrics**:
   - Convergence rate
   - Final performance level
   - Generalization to new conditions

## Advanced Simulation Techniques

### Differentiable Physics Simulation

Modern simulators support differentiable physics, allowing gradients to flow through the simulation for direct optimization of control policies.

### Hardware-in-the-Loop Simulation

Connect real sensors and computational units to the simulation for more realistic testing while maintaining safety.

### Multi-Physics Simulation

Combine multiple physical models (mechanical, electrical, thermal) for comprehensive system analysis.

## Troubleshooting Simulation Issues

1. **Unstable Dynamics**: Check mass/inertia properties and joint limits
2. **Slow Performance**: Reduce physics update rate or simplify models
3. **Learning Failures**: Verify reward function and state representation
4. **Control Issues**: Check PID gains and control frequency

## Simulation Extensions

1. **Adversarial Environments**: Create challenging scenarios that test the limits of embodied approaches
2. **Multi-Agent Embodiment**: Simulate teams of embodied agents
3. **Soft Body Simulation**: Implement soft robotic systems in simulation
4. **Developmental Learning**: Simulate robots that learn over extended periods

## Chapter Summary

This simulation module provided hands-on experience with embodied intelligence concepts in virtual environments. Students implemented and tested passive dynamic walking, morphological computation, sensorimotor learning, and dynamic balance control. The simulations demonstrated how physical embodiment can enhance robot intelligence and efficiency, while also highlighting the challenges and opportunities in embodied AI.

## Key Terms
- Simulation-to-Reality Gap
- Zero Moment Point (ZMP)
- Passive Dynamic Walking
- Sensorimotor Learning
- Morphological Computation
- Linear Inverted Pendulum Model (LIPM)

## References
- Tedrake, R. (2023). Underactuated Robotics: Algorithms for Walking, Running, Swimming, Flying, and Manipulation.
- Kubricht, J. R., Holyoak, K. J., & Lu, H. (2017). The power of mental simulation. 
- Gazebo Documentation: http://gazebosim.org/tutorials
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/