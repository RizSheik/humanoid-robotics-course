---
id: module-1-chapter-4-urdf-specification
title: 'Module 1 — The Robotic Nervous System | Chapter 4 — URDF Specification'
sidebar_label: 'Chapter 4 — URDF Specification'
---

# Chapter 4 — URDF Specification

## Unified Robot Description Format for Humanoids

Unified Robot Description Format (URDF) is an XML format for representing a robot model. URDF is used to describe the physical and visual properties of a robot, including kinematic and dynamic properties, visual appearance, and sensors.

### URDF Basics

A URDF file contains elements that describe various aspects of a robot:

- **Links**: Rigid parts of the robot (e.g., chassis, arm links)
- **Joints**: Connections between links (e.g., revolute, prismatic)
- **Materials**: Visual properties like color and texture
- **Transmissions**: Mapping between actuators and joints
- **Gazebo plugins**: Simulation-specific properties

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Joints -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
  
  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>
</robot>
```

### Links in URDF

Links represent rigid bodies in the robot model. Each link can have:

- **Visual**: How the link appears in visualization tools
- **Collision**: The geometry used for collision detection
- **Inertial**: Mass and inertia properties for physics simulation

```xml
<link name="link_name">
  <visual>
    <!-- Visual representation -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Box, cylinder, sphere, or mesh -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="color_name"/>
  </visual>
  
  <collision>
    <!-- Collision geometry -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Usually simplified geometry for collision detection -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  
  <inertial>
    <!-- Physical properties for simulation -->
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

### Joints in URDF

Joints define the connection between links and specify how they can move relative to each other:

- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint
- **Fixed**: No movement between links
- **Floating**: 6DOF movement
- **Planar**: Movement on a plane

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Materials and Colors

Materials define visual properties that can be reused across multiple links:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="black">
  <color rgba="0 0 0 1"/>
</material>
```

### Using Mesh Files

For complex robot shapes, you can use 3D mesh files:

```xml
<visual>
  <geometry>
    <mesh filename="package://my_robot/meshes/link_name.dae" scale="1 1 1"/>
  </geometry>
</visual>
```

### URDF with Xacro

Xacro (XML Macros) allows for more complex URDF definitions with variables, math, and macro-like functionality:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  
  <!-- Define properties -->
  <xacro:property name="base_size" value="0.5 0.5 0.2"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="PI" value="3.1415926535897931"/>
  
  <!-- Define a macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>
    
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_size}"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="${base_size}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_left" parent="base_link" xyz="-0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="back_right" parent="base_link" xyz="-0.2 -0.2 0" rpy="0 0 0"/>
</robot>
```

### URDF for Humanoid Robots

Humanoid robots require special considerations in URDF:

1. **Kinematic Chains**: Multiple limbs with appropriate joint types
2. **Balance**: Proper mass distribution for stable locomotion
3. **Degrees of Freedom**: Sufficient DOF for humanoid-like movements
4. **Sensors**: Integration of IMUs, cameras, and other sensors

Example humanoid torso:

```xml
<!-- Torso of humanoid robot -->
<link name="torso">
  <visual>
    <geometry>
      <box size="0.3 0.2 0.5"/>
    </geometry>
    <material name="grey"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.3 0.2 0.5"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="5.0"/>
    <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.1"/>
  </inertial>
</link>

<!-- Head with camera -->
<joint name="neck_joint" type="revolute">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
</joint>

<link name="head">
  <visual>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
    <material name="pink"/>
  </visual>
  <collision>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>

<!-- Camera sensor -->
<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera"/>
  <origin xyz="0.05 0 0" rpy="0 0 0"/>
</joint>

<link name="camera">
  <visual>
    <geometry>
      <box size="0.02 0.04 0.02"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.04 0.02"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>
```

### Validating URDF

To validate your URDF files:

1. Use `check_urdf` to check for parsing errors
2. Use `urdf_to_graphiz` to visualize the kinematic tree
3. Load in RViz to visualize the robot
4. Test in Gazebo simulation

```bash
# Check URDF validity
check_urdf /path/to/robot.urdf

# Visualize kinematic tree
urdf_to_graphiz /path/to/robot.urdf
```

URDF is fundamental for robot simulation and visualization in ROS 2, providing the geometric, kinematic, and dynamic properties necessary for physics simulation and robot reasoning.