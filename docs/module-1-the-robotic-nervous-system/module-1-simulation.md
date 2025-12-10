---
id: module-1-simulation
title: 'Module 1 — The Robotic Nervous System | Chapter 5 — Simulation'
sidebar_label: 'Chapter 5 — Simulation'
sidebar_position: 5
---

# Chapter 5 — Simulation

## Introduction to Robot Simulation in ROS 2

Simulation is a critical component of modern robotics development, allowing developers to test algorithms, validate behaviors, and train AI systems in a safe, cost-effective environment before deploying to real hardware. In the context of ROS 2, simulation integrates seamlessly with the rest of the robotics middleware stack, enabling realistic testing of robotic applications.

## Gazebo Simulation Environment

Gazebo is the primary simulation environment used in ROS 2 development. It provides:

- **Physics engine**: Accurate simulation of rigid body dynamics, collisions, and contacts
- **Sensor simulation**: Realistic modeling of cameras, LIDARs, IMUs, and other sensors
- **3D visualization**: Interactive 3D visualization of simulated environments
- **Plugin architecture**: Extensible system for custom sensors and controllers
- **World editor**: Tools for creating complex simulation environments

### Installing Gazebo for ROS 2

Gazebo Harmonic is the recommended version for ROS 2 Humble Hawksbill:

```bash
# Install Gazebo Harmonic
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Install additional dependencies
sudo apt install ros-humble-xacro ros-humble-joint-state-publisher ros-humble-robot-state-publisher
```

## Creating a Simple Robot Model

### URDF (Unified Robot Description Format)

URDF is the standard format for describing robots in ROS. We'll create a simple differential drive robot model:

1. **Create the URDF directory structure**:
```bash
mkdir -p ~/ros2_ws/src/my_robot_pkg/urdf
mkdir -p ~/ros2_ws/src/my_robot_pkg/meshes
mkdir -p ~/ros2_ws/src/my_robot_pkg/materials/textures
mkdir -p ~/ros2_ws/src/my_robot_pkg/config
```

2. **Create the main robot URDF** (save as `~/ros2_ws/src/my_robot_pkg/urdf/my_robot.urdf.xacro`):
```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.016666" ixy="0" ixz="0" iyy="0.041666" iyz="0" izz="0.041666"/>
    </inertial>
  </link>

  <!-- Front caster wheel -->
  <joint name="caster_front_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_front_link"/>
    <origin xyz="0.2 0 -0.05" rpy="0 0 0"/>
  </joint>

  <link name="caster_front_link">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left_link"/>
    <origin xyz="0 0.15 -0.1" rpy="${-PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_left_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.00053" ixy="0" ixz="0" iyy="0.00053" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <joint name="wheel_right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_right_link"/>
    <origin xyz="0 -0.15 -0.1" rpy="${-PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_right_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.00053" ixy="0" ixz="0" iyy="0.00053" iyz="0" izz="0.001"/>
    </inertial>
  </link>

</robot>
```

### Creating a Gazebo Integration File

1. **Create the Gazebo integration file** (save as `~/ros2_ws/src/my_robot_pkg/urdf/my_robot.gazebo.xacro`):
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="wheel_left_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="wheel_right_link">
    <material>Gazebo/Black</material>
  </gazebo>

  <gazebo reference="caster_front_link">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Motor controller -->
  <gazebo>
    <plugin filename="libgazebo_ros_diff_drive.so" name="my_robot_diff_drive">
      <update_rate>30</update_rate>
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_wheel_tf>true</publish_wheel_tf>
      <publish_wheel_joint_state>true</publish_wheel_joint_state>
      <ros>
        <namespace>/my_robot</namespace>
      </ros>
    </plugin>
  </gazebo>

</robot>
```

## Launching the Simulation

### Creating a Launch File for Gazebo Integration

1. **Create a launch file** (save as `~/ros2_ws/src/my_robot_pkg/launch/robot_spawn.launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Get the URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('my_robot_pkg'),
        'urdf',
        'my_robot.urdf.xacro'
    )

    # Define the robot_state_publisher node
    params = {'robot_description': Command(['xacro', ' ', urdf_file])}
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # Define the joint_state_publisher node
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    # Define the Gazebo node to spawn the robot
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0',
            '-y', '0',
            '-z', '0.2'
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
        robot_state_publisher_node,
        joint_state_publisher_node,
        spawn_entity_node,
    ])
```

### Creating a Launch File to Start Gazebo

1. **Create the main Gazebo launch file** (save as `~/ros2_ws/src/my_robot_pkg/launch/gazebo_world.launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    # Get the package share directory
    pkg_dir = get_package_share_directory('my_robot_pkg')
    
    # Define the Gazebo server node
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Define the Gazebo client node
    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    # Define a static transform publisher for the map to odom transformation
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    # Return the launch description
    return LaunchDescription([
        gzserver_cmd,
        gzclient_cmd,
        static_transform_publisher,
    ])
```

## Running the Complete Simulation

### Building the Package

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source ~/ros2_ws/install/setup.bash
```

### Launching the Environment

1. **Start Gazebo with an empty world**:
```bash
ros2 launch my_robot_pkg gazebo_world.launch.py
```

2. **In a new terminal, spawn the robot**:
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch my_robot_pkg robot_spawn.launch.py
```

### Controlling the Robot in Simulation

1. **Send movement commands**:
```bash
# Move forward
ros2 topic pub --once /my_robot/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 1.0}, angular: {z: 0.0}}'

# Rotate in place
ros2 topic pub --once /my_robot/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.0}, angular: {z: 1.0}}'
```

## Advanced Simulation Concepts

### Sensor Integration

Adding sensors to your simulated robot:

1. **Create a sensor configuration** (add to the URDF file):
```xml
<!-- Camera sensor -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>800</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Physics Parameters

Tuning physics parameters for realistic simulation:

- **Mass properties**: Accurate mass and inertia values for each link
- **Friction coefficients**: For realistic surface interactions
- **Damping parameters**: To simulate mechanical losses
- **Contact properties**: To control collision behavior

### Simulation Speed and Real-time Factor

You can adjust simulation speed in Gazebo:
- **Real-time factor**: Controls how fast simulation runs compared to real time
- **Update rate**: Sets how frequently physics updates occur
- **Adaptive time stepping**: Allows Gazebo to adjust time steps based on computational load

## Best Practices for Simulation

### 1. Model Complexity
- Start with simple models and gradually add complexity
- Balance realism with computational efficiency
- Use simplified collision meshes for faster physics simulation

### 2. Sensor Simulation
- Validate sensor outputs against real hardware
- Account for sensor noise and limitations
- Test with various environmental conditions

### 3. Physics Tuning
- Validate physics parameters against real-world behavior
- Test with various terrain and interaction scenarios
- Monitor simulation stability and performance

## Troubleshooting Common Issues

### Issue: Robot falls through the ground
**Solution**: Check collision geometries and ensure proper mass/inertia values are defined.

### Issue: Robot moves jerkily
**Solution**: Verify physics update rates and adjust time stepping parameters in Gazebo.

### Issue: Sensors not publishing data
**Solution**: Check that sensor plugins are properly loaded and that the ROS 2 bridge is running.

## Connecting Simulation to Real Robots

Simulation provides a safe environment to test code before deployment on real hardware:

1. **Same ROS 2 interfaces**: Use identical topics, services, and actions in simulation and reality
2. **Hardware abstraction**: Implement common interfaces that can work with both simulated and real sensors
3. **Configuration management**: Use parameter files to switch between simulation and real hardware settings

## Summary

This simulation chapter has covered the fundamentals of creating and running robotic simulations with ROS 2 and Gazebo. You've learned how to:

- Create robot models using URDF and Xacro
- Integrate robots with Gazebo simulation environment
- Control robots through ROS 2 interfaces in simulation
- Add sensors and other components to your simulation
- Troubleshoot common simulation issues

Simulation is a powerful tool for developing, testing, and validating robotic applications before deployment to real hardware. Mastering these concepts will greatly enhance your ability to develop robust robotic systems.

In the next chapters, we'll explore more advanced topics and put these simulation skills to use.