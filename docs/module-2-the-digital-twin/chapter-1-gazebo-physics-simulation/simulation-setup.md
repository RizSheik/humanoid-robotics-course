---
id: module-2-chapter-1-simulation-setup
title: 'Module 2 — The Digital Twin | Chapter 1 — Simulation Setup'
sidebar_label: 'Chapter 1 — Simulation Setup'
---

# Chapter 1 — Simulation Setup

## Configuring Gazebo for Humanoid Robotics

Setting up a simulation environment for humanoid robotics requires careful configuration of physics parameters, robot models, sensors, and interaction environments. This chapter provides a comprehensive guide to creating effective simulation environments for humanoid robots.

### Gazebo Installation and Configuration

#### Prerequisites
Before setting up humanoid simulations, ensure you have:

```bash
# Install Gazebo Harmonic (or appropriate version for your ROS 2 distro)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control

# Install additional tools
sudo apt install gazebo
```

#### Basic Gazebo Launch
To start Gazebo with default settings:

```bash
# Launch Gazebo with empty world
ros2 launch gazebo_ros empty_world.launch.py

# Or with GUI
ros2 launch gazebo_ros gzserver.launch.py verbose:=true
```

### Basic Simulation Environment

#### World File Structure
Create a SDF world file to configure your simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_simulation">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
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
    
    <!-- Your robot model would be included here -->
    <include>
      <uri>model://my_humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
    
    <!-- Add obstacles and objects for testing -->
    <model name="test_box">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.5</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.5</size>
            </box>
          </geometry>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.083</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.083</iyy>
            <iyz>0</iyz>
            <izz>0.083</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Robot Model Integration

#### URDF to SDF Conversion
For complex robots, you may need to convert URDF to SDF or include URDF directly:

```xml
<!-- Including a URDF robot in a Gazebo world -->
<include>
  <name>humanoid_robot</name>
  <pose>0 0 1 0 0 0</pose>
  <uri>model://humanoid_robot_description</uri>
</include>
```

#### Robot Spawn Parameters
When spawning robots, consider:

```xml
<spawn>
  <name>humanoid_robot</name>
  <param name="robot_description">$(find-pkg-share my_robot_description)/urdf/humanoid.urdf</param>
  <topic>/robot_description</topic>
  <spawn>
    <name>humanoid_robot</name>
    <x>0</x>
    <y>0</y>
    <z>1.0</z>
    <roll>0</roll>
    <pitch>0</pitch>
    <yaw>0</yaw>
  </spawn>
</spawn>
```

### Launch File Configuration

Create a launch file to automate the setup process:

```python
# humanoid_simulation.launch.py
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_robot_description = FindPackageShare('my_robot_description')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'world': PathJoinSubstitution([pkg_robot_description, 'worlds', 'humanoid_world.sdf']),
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': open(PathJoinSubstitution([
                pkg_robot_description, 
                'urdf', 
                'humanoid.urdf'
            ]).perform({})).read()
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0',
            '-y', '0',
            '-z', '1.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

### Sensor Configuration

Configure sensors appropriately for humanoid robots:

```xml
<!-- Gazebo sensor configuration -->
<gazebo reference="head_camera">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>camera</namespace>
        <remapping>~/image_raw:=image</remapping>
        <remapping>~/camera_info:=camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <image_topic_name>image</image_topic_name>
      <camera_info_topic_name>camera_info</camera_info_topic_name>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU sensor -->
<gazebo reference="imu_link">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
    </plugin>
  </sensor>
</gazebo>

<!-- Force/Torque sensor -->
<gazebo reference="left_foot">
  <sensor name="left_foot_force_torque" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <plugin name="left_foot_ft_plugin" filename="libgazebo_ros_ft.so">
      <ros>
        <namespace>left_foot_ft</namespace>
        <remapping>~/wrench:=left_foot_wrench</remapping>
      </ros>
      <frame_name>left_foot</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Control Configuration

Configure control interfaces for the humanoid robot:

```yaml
# controller_config.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    humanoid_controller:
      type: position_controllers/JointGroupPositionController
      
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

humanoid_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint
      - left_shoulder_joint
      - left_elbow_joint
      - right_shoulder_joint
      - right_elbow_joint
      - torso_joint
      - neck_joint
```

### Environment Configuration

Create diverse environments for testing:

```xml
<!-- Complex environment with obstacles -->
<world name="complex_humanoid_world">
  <!-- Include ground plane -->
  <include>
    <uri>model://ground_plane</uri>
  </include>
  
  <!-- Add textured floor -->
  <model name="floor">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box><size>10 10 0.1</size></box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Grass</name>
          </script>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box><size>10 10 0.1</size></box>
        </geometry>
      </collision>
      <inertial>
        <mass>100</mass>
        <inertia><ixx>100</ixx><iyy>100</iyy><izz>100</izz></inertia>
      </inertial>
    </link>
  </model>
  
  <!-- Add obstacles of different shapes -->
  <model name="cylinder_obstacle">
    <pose>3 0 0.5 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry><cylinder><radius>0.2</radius><length>1.0</length></cylinder></geometry>
      </collision>
      <visual name="visual">
        <geometry><cylinder><radius>0.2</radius><length>1.0</length></cylinder></geometry>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia><ixx>0.15</ixx><iyy>0.15</iyy><izz>0.1</izz></inertia>
      </inertial>
    </link>
  </model>
  
  <!-- Stairs for locomotion testing -->
  <model name="stairs">
    <pose>5 -2 0 0 0 0</pose>
    <!-- Multiple steps -->
    <include>
      <uri>model://step</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>
    <include>
      <uri>model://step</uri>
      <pose>0 0 0.2 0 0 0</pose>
    </include>
    <include>
      <uri>model://step</uri>
      <pose>0 0 0.3 0 0 0</pose>
    </include>
  </model>
</world>
```

### Physics Parameter Tuning

Fine-tune physics parameters for humanoid robots:

```xml
<!-- Optimized physics for humanoid simulation -->
<physics name="humanoid_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- Sufficient for humanoid stability -->
      <sor>1.2</sor>
    </solver>
    <constraints>
      <cfm>1e-05</cfm>      <!-- Low constraint force mixing -->
      <erp>0.2</erp>        <!-- Error reduction for stable contacts -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>  <!-- Prevent sinking -->
    </constraints>
  </ode>
</physics>
```

### Debugging and Visualization

Enable debugging features:

```xml
<!-- Add to your world file for debugging -->
<world name="debug_world">
  <!-- Enable physics visualization -->
  <physics type="ode">
    <!-- ... physics config ... -->
  </physics>
  
  <!-- Enable contact visualization -->
  <gazebo>
    <enable_physics_visualization>true</enable_physics_visualization>
  </gazebo>
</world>
```

### Performance Optimization

For large-scale simulations:

1. **Reduce update rates** for non-critical sensors
2. **Optimize collision meshes** (use simpler shapes)
3. **Adjust physics parameters** for performance vs. accuracy
4. **Use appropriate world bounding volumes**

### Launch and Test

Once configured, launch your simulation:

```bash
# Build your packages
colcon build --packages-select my_robot_description my_robot_simulation

# Source the environment
source install/setup.bash

# Launch the simulation
ros2 launch my_robot_simulation humanoid_simulation.launch.py
```

Proper simulation setup is crucial for effective humanoid robotics development, allowing for safe testing of complex behaviors before physical deployment.