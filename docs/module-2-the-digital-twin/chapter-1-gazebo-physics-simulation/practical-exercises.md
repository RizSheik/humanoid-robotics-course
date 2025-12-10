---
id: module-2-chapter-1-practical-exercises
title: 'Module 2 — The Digital Twin | Chapter 1 — Practical Exercises'
sidebar_label: 'Chapter 1 — Practical Exercises'
---

# Chapter 1 — Practical Exercises

## Gazebo Physics Simulation: Hands-On Implementation

This practical lab focuses on setting up and configuring Gazebo simulation environments for humanoid robotics. You'll learn to create, configure, and validate physics simulation environments.

### Exercise 1: Basic Gazebo Environment Setup

#### Objective
Create a basic simulation environment with physics parameters optimized for humanoid robots.

#### Steps
1. Create a new ROS 2 package for simulation
2. Create a basic world file with ground plane and lighting
3. Configure physics parameters for humanoid simulation
4. Launch the environment and verify it works

First, create the package structure:

```bash
# Create a new package for simulation files
ros2 pkg create --build-type ament_python my_robot_simulation
mkdir -p my_robot_simulation/worlds
mkdir -p my_robot_simulation/launch
mkdir -p my_robot_simulation/config
```

Create the basic world file (`my_robot_simulation/worlds/basic_humanoid_world.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_humanoid_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Physics engine configuration optimized for humanoid robots -->
    <physics name="humanoid_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>200</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Add a simple humanoid model -->
    <!-- For this exercise, we'll use a simple model made of basic shapes -->
    <model name="simple_humanoid">
      <pose>0 0 1 0 0 0</pose>
      
      <!-- Torso -->
      <link name="torso">
        <pose>0 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>0.3 0.2 0.5</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>0.3 0.2 0.5</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.5</iyy>
            <iyz>0</iyz>
            <izz>0.2</izz>
          </inertia>
        </inertial>
      </link>
      
      <!-- Head -->
      <link name="head">
        <pose>0 0 0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere><radius>0.1</radius></sphere>
          </geometry>
          <material>
            <ambient>0.5 0.5 1 1</ambient>
            <diffuse>0.5 0.5 1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.01</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.01</iyy>
            <iyz>0</iyz>
            <izz>0.01</izz>
          </inertia>
        </inertial>
      </link>
      
      <!-- Joint to connect head to torso -->
      <joint name="neck_joint" type="revolute">
        <parent>torso</parent>
        <child>head</child>
        <pose>0 0 0.25 0 0 0</pose>
        <axis>
          <xyz>0 1 0</xyz>
          <limit>
            <lower>-0.785</lower>
            <upper>0.785</upper>
            <effort>10</effort>
            <velocity>1</velocity>
          </limit>
        </axis>
      </joint>
      
      <!-- Left leg -->
      <link name="left_thigh">
        <pose>0.1 0 -0.3 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <cylinder><radius>0.06</radius><length>0.4</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>0.06</radius><length>0.4</length></cylinder>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.1</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1</iyy>
            <iyz>0</iyz>
            <izz>0.05</izz>
          </inertia>
        </inertial>
      </link>
      
      <joint name="left_hip_joint" type="revolute">
        <parent>torso</parent>
        <child>left_thigh</child>
        <pose>0.1 0 -0.1 0 0 0</pose>
        <axis>
          <xyz>0 1 0</xyz>
          <limit>
            <lower>-1.57</lower>
            <upper>1.57</upper>
            <effort>50</effort>
            <velocity>1</velocity>
          </limit>
        </axis>
      </joint>
    </model>
  </world>
</sdf>
```

Create a launch file to start the simulation (`my_robot_simulation/launch/basic_sim.launch.py`):

```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_my_simulation = FindPackageShare('my_robot_simulation')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_my_simulation, 'worlds', 'basic_humanoid_world.sdf']),
        }.items()
    )

    return LaunchDescription([
        gazebo,
    ])
```

Test the setup:

```bash
# Build the package
colcon build --packages-select my_robot_simulation

# Source the environment
source install/setup.bash

# Launch the simulation
ros2 launch my_robot_simulation basic_sim.launch.py
```

### Exercise 2: Physics Parameter Tuning for Stability

#### Objective
Tune physics parameters to achieve stable simulation for humanoid robots.

#### Steps
1. Create multiple world files with different physics parameters
2. Test robot stability with different configurations
3. Identify optimal parameters for humanoid locomotion

Create a comparative world file (`my_robot_simulation/worlds/comparison_world.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physics_comparison">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Physics configuration 1: Conservative settings -->
    <physics name="conservative_physics" type="ode">
      <max_step_size>0.0005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>2000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>500</iters>
          <sor>1.0</sor>
        </solver>
        <constraints>
          <cfm>1e-06</cfm>
          <erp>0.1</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.0005</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Physics configuration 2: Performance-focused settings -->
    <physics name="performance_physics" type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>500</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>50</iters>
          <sor>1.5</sor>
        </solver>
        <constraints>
          <cfm>1e-04</cfm>
          <erp>0.5</erp>
          <contact_max_correcting_vel>200</contact_max_correcting_vel>
          <contact_surface_layer>0.002</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Robot in conservative physics area -->
    <model name="conservative_robot">
      <pose>-2 0 1 0 0 0</pose>
      <!-- Simple humanoid model -->
      <link name="body">
        <collision><geometry><box><size>0.5 0.2 0.8</size></box></geometry></collision>
        <visual><geometry><box><size>0.5 0.2 0.8</size></box></geometry></visual>
        <inertial><mass>20.0</mass><inertia><ixx>1.0</ixx><iyy>1.0</iyy><izz>1.0</izz></inertia></inertial>
      </link>
    </model>
    
    <!-- Robot in performance physics area -->
    <model name="performance_robot">
      <pose>2 0 1 0 0 0</pose>
      <!-- Simple humanoid model -->
      <link name="body">
        <collision><geometry><box><size>0.5 0.2 0.8</size></box></geometry></collision>
        <visual><geometry><box><size>0.5 0.2 0.8</size></box></geometry></visual>
        <inertial><mass>20.0</mass><inertia><ixx>1.0</ixx><iyy>1.0</iyy><izz>1.0</izz></inertia></inertial>
      </link>
    </model>
  </world>
```

Create a launch file for comparison (`my_robot_simulation/launch/physics_comparison.launch.py`):

```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_my_simulation = FindPackageShare('my_robot_simulation')

    # Gazebo launch with comparison world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_my_simulation, 'worlds', 'comparison_world.sdf']),
        }.items()
    )

    return LaunchDescription([
        gazebo,
    ])
```

### Exercise 3: Complex Environment Setup

#### Objective
Create a complex environment with various obstacles and surfaces for humanoid testing.

#### Steps
1. Create a world with multiple surfaces and obstacles
2. Add sensors to the robot model
3. Test robot interaction with the environment

Create a complex world file (`my_robot_simulation/worlds/complex_environment.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="complex_humanoid_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Physics engine -->
    <physics name="complex_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>200</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    
    <!-- Various surfaces -->
    <model name="grass_area">
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>10 10 0.01</size></box></geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grass</name>
            </script>
          </material>
        </visual>
        <collision name="collision">
          <geometry><box><size>10 10 0.01</size></box></geometry>
        </collision>
        <inertial><mass>100</mass><inertia><ixx>100</ixx><iyy>100</iyy><izz>100</izz></inertia></inertial>
      </link>
    </model>
    
    <!-- Ramp for testing locomotion -->
    <model name="ramp">
      <pose>3 0 0 0 0.3 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>2 1 0.2</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material>
        </visual>
        <collision name="collision">
          <geometry><box><size>2 1 0.2</size></box></geometry>
        </collision>
        <inertial><mass>50</mass><inertia><ixx>20</ixx><iyy>20</iyy><izz>20</izz></inertia></inertial>
      </link>
    </model>
    
    <!-- Obstacles -->
    <model name="cylinder_obstacle">
      <pose>5 2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry></collision>
        <visual name="visual"><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry></visual>
        <inertial><mass>5.0</mass><inertia><ixx>0.2</ixx><iyy>0.2</iyy><izz>0.1</izz></inertia></inertial>
      </link>
    </model>
    
    <model name="box_obstacle">
      <pose>5 -2 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>0.8 0.8 1.0</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>0.8 0.8 1.0</size></box></geometry></visual>
        <inertial><mass>10.0</mass><inertia><ixx>0.8</ixx><iyy>0.8</iyy><izz>0.8</izz></inertia></inertial>
      </link>
    </model>
    
    <!-- Stairs -->
    <model name="stairs">
      <pose>-3 0 0 0 0 0</pose>
      <!-- Step 1 -->
      <link name="step1">
        <pose>0 0 0.1 0 0 0</pose>
        <collision><geometry><box><size>1.0 0.8 0.2</size></box></geometry></collision>
        <visual><geometry><box><size>1.0 0.8 0.2</size></box></geometry></visual>
        <inertial><mass>10.0</mass><inertia><ixx>1.0</ixx><iyy>1.0</iyy><izz>1.0</izz></inertia></link>
      <static>true</static>
      <!-- Step 2 -->
      <link name="step2">
        <pose>0 0 0.3 0 0 0</pose>
        <collision><geometry><box><size>1.0 0.8 0.2</size></box></geometry></collision>
        <visual><geometry><box><size>1.0 0.8 0.2</size></box></geometry></visual>
        <inertial><mass>10.0</mass><inertia><ixx>1.0</ixx><iyy>1.0</iyy><izz>1.0</izz></inertia></link>
      <static>true</static>
      <!-- Step 3 -->
      <link name="step3">
        <pose>0 0 0.5 0 0 0</pose>
        <collision><geometry><box><size>1.0 0.8 0.2</size></box></geometry></collision>
        <visual><geometry><box><size>1.0 0.8 0.2</size></box></geometry></visual>
        <inertial><mass>10.0</mass><inertia><ixx>1.0</ixx><iyy>1.0</iyy><izz>1.0</izz></inertia></link>
      <static>true</static>
    </model>
  </world>
</sdf>
```

### Exercise 4: Robot-Environment Interaction

#### Objective
Implement a simple controller to test robot-environment interactions.

Create a simple controller node (`my_robot_simulation/scripts/simple_controller.py`):

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        
        # Timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)  # 100 Hz
        
        # Joint names for our simple humanoid
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'neck_joint'
        ]
        
        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        # Time counter
        self.time_counter = 0.0
        
        self.get_logger().info('Simple controller started')

    def control_loop(self):
        # Update joint positions using sinusoidal patterns to test movement
        self.time_counter += 0.01
        
        # Create walking pattern for legs
        step_freq = 1.0  # 1 Hz step frequency
        step_amplitude = 0.3  # 30 degree amplitude
        
        # Left leg movement
        self.joint_positions[0] = math.sin(self.time_counter * step_freq) * step_amplitude  # hip
        self.joint_positions[1] = math.sin(self.time_counter * step_freq + 0.5) * step_amplitude * 0.8  # knee
        self.joint_positions[2] = math.sin(self.time_counter * step_freq + 1.0) * step_amplitude * 0.5  # ankle
        
        # Right leg movement (opposite phase)
        self.joint_positions[3] = math.sin(self.time_counter * step_freq + math.pi) * step_amplitude
        self.joint_positions[4] = math.sin(self.time_counter * step_freq + math.pi + 0.5) * step_amplitude * 0.8
        self.joint_positions[5] = math.sin(self.time_counter * step_freq + math.pi + 1.0) * step_amplitude * 0.5
        
        # Arm movements for balance
        self.joint_positions[6] = math.sin(self.time_counter * step_freq + math.pi) * 0.2  # left shoulder
        self.joint_positions[7] = math.sin(self.time_counter * step_freq) * 0.3  # left elbow
        self.joint_positions[8] = math.sin(self.time_counter * step_freq) * 0.2  # right shoulder
        self.joint_positions[9] = math.sin(self.time_counter * step_freq + math.pi) * 0.3  # right elbow
        
        # Head movement
        self.joint_positions[10] = math.sin(self.time_counter * 0.5) * 0.2  # neck
        
        # Create and publish joint state message
        msg = JointState()
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = SimpleController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 5: Validation and Testing

#### Objective
Validate the simulation setup by testing key behaviors.

Create a validation script (`my_robot_simulation/scripts/validation_test.py`):

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import numpy as np

class SimulationValidator(Node):
    def __init__(self):
        super().__init__('simulation_validator')
        
        # Subscribe to model states to monitor robot position
        self.model_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )
        
        # Publisher for validation metrics
        self.metrics_pub = self.create_publisher(Float64, '/validation_metrics', 10)
        
        # Track robot stability metrics
        self.robot_z_positions = []
        self.robot_last_position = None
        self.accumulated_drift = 0.0
        self.timestep_counter = 0
        
        self.get_logger().info('Simulation validator started')

    def model_states_callback(self, msg):
        # Find our robot in the model states
        robot_idx = -1
        for i, name in enumerate(msg.name):
            if 'simple_humanoid' in name or 'conservative_robot' in name or 'performance_robot' in name:
                robot_idx = i
                break
        
        if robot_idx >= 0:
            robot_pose = msg.pose[robot_idx]
            
            # Check for excessive drift in Z direction (vertical)
            z_pos = robot_pose.position.z
            self.robot_z_positions.append(z_pos)
            
            # Calculate drift from expected height (should stay near ground level)
            expected_height = 0.9  # Adjust based on robot's default height
            drift = abs(z_pos - expected_height)
            
            # Accumulate drift over time
            self.accumulated_drift += drift
            self.timestep_counter += 1
            
            # Log if drift is excessive
            if drift > 0.5:  # More than 50cm drift is problematic
                self.get_logger().warn(f'Large Z drift detected: {drift:.3f}m')
            
            # Calculate and publish stability metric
            stability_metric = Float64()
            if len(self.robot_z_positions) > 10:
                # Calculate variance in Z position (lower is better)
                z_variance = np.var(self.robot_z_positions[-10:])
                stability_metric.data = 1.0 / (1.0 + z_variance)  # Higher is better
            else:
                stability_metric.data = 1.0
            
            self.metrics_pub.publish(stability_metric)
            
            self.get_logger().info(f'Robot Z: {z_pos:.3f}, Stability: {stability_metric.data:.3f}')

def main(args=None):
    rclpy.init(args=args)
    validator = SimulationValidator()
    
    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Assessment Criteria

Your implementation will be evaluated based on:

1. **Environment Setup**: Correct configuration of Gazebo world files
2. **Physics Parameters**: Appropriate selection of physics parameters for humanoid simulation
3. **Stability Testing**: Demonstration of stable robot behavior in simulation
4. **Complex Environment**: Creation of a complex environment with obstacles
5. **Validation**: Proper validation of simulation setup and performance

### Troubleshooting Tips

1. **Robot falling through ground**: Check collision geometry and physics parameters
2. **Jittery movement**: Increase constraint solver iterations or reduce time step
3. **Performance issues**: Optimize collision meshes and reduce update rates for non-critical elements
4. **Joint limits**: Ensure joint limits are realistic for humanoid movement
5. **Controller stability**: Fine-tune control parameters for stable operation

### Extensions for Advanced Students

- Implement dynamic environments that change during simulation
- Create scenarios with multiple humanoid robots
- Add advanced sensors (force/torque, IMU, cameras) and simulate their data streams
- Implement physics parameter optimization algorithms
- Create scenario-based tests for humanoid behaviors

This practical exercise provides comprehensive experience with setting up and validating Gazebo physics simulation environments for humanoid robotics.