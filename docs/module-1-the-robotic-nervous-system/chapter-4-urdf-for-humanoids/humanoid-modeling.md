---
id: module-1-chapter-4-humanoid-modeling
title: 'Module 1 — The Robotic Nervous System | Chapter 4 — Humanoid Modeling'
sidebar_label: 'Chapter 4 — Humanoid Modeling'
---

# Chapter 4 — Humanoid Modeling

## Designing Humanoid Robot Models with URDF

Humanoid robot modeling requires careful consideration of biomechanics, kinematics, and dynamics to create robots that can effectively interact with human environments and perform human-like movements.

### Humanoid Robot Anatomy

A typical humanoid robot includes:

- **Torso**: Central body with head attachment
- **Head**: With vision sensors and neck joint
- **Arms**: Shoulders, elbows, wrists, and end-effectors (hands)
- **Legs**: Hips, knees, ankles, and feet
- **Hands**: For manipulation tasks

### Design Principles

When modeling humanoid robots, consider:

1. **Degrees of Freedom (DOF)**: Balance between complexity and controllability
2. **Range of Motion**: Mimic human capabilities without exceeding physical constraints
3. **Load Distribution**: Ensure actuators can handle required torques
4. **Stability**: Proper center of mass for balance during locomotion
5. **Safety**: Inherently safe design for human interaction

### Complete Humanoid URDF Example

Here's a simplified but complete humanoid model:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
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
    <color rgba="0.5 0.5 0.5 1.0"/>
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

  <!-- Base and world joint -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Fixed joint to world -->
  <joint name="base_joint" type="fixed">
    <parent link="base_link"/>
    <child link="pelvis"/>
  </joint>

  <!-- PELVIS -->
  <link name="pelvis">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- TORSO -->
  <joint name="torso_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.175"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="30" velocity="1.0"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.2"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2"/>
      <geometry>
        <box size="0.18 0.2 0.4"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.2"/>
      <geometry>
        <box size="0.18 0.2 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- HEAD -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.42"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="5" velocity="2.0"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Cameras -->
  <joint name="left_camera_joint" type="fixed">
    <parent link="head"/>
    <child link="left_camera_frame"/>
    <origin xyz="0.05 0.05 0.05" rpy="0 0 0"/>
  </joint>

  <link name="left_camera_frame"/>

  <joint name="right_camera_joint" type="fixed">
    <parent link="head"/>
    <child link="right_camera_frame"/>
    <origin xyz="0.05 -0.05 0.05" rpy="0 0 0"/>
  </joint>

  <link name="right_camera_frame"/>

  <!-- LEFT ARM -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.15 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.0 0.0 0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0.0 0.0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="left_lower_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.0 0.0 0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- RIGHT ARM -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.0 -0.15 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="right_upper_arm">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.0 0.0 0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.05" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0.0 0.0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="right_lower_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0.0 0.0 0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.1"/>
      <geometry>
        <cylinder radius="0.04" length="0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- LEFT LEG -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="left_thigh"/>
    <origin xyz="0.0 0.08 -0.075"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0.0 0.0 -0.2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0.0 0.0 -0.2"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0.0 -0.05"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- RIGHT LEG -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="right_thigh"/>
    <origin xyz="0.0 -0.08 -0.075"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0.0 0.0 -0.2"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="1.57" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0.0 0.0 -0.2"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 -0.2"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="20" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0.0 -0.05"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### Humanoid Kinematics

Humanoid robots typically have two types of kinematic structures:

1. **Open Chains**: Arms and legs act independently
2. **Closed Chains**: When both feet are on the ground, creating a closed loop

For locomotion, humanoid robots use techniques such as:

- **Zero-Moment Point (ZMP)**: Maintaining balance by keeping ground reaction forces within support polygon
- **Capture Point**: Point where the robot can come to a stop without falling
- **Inverted Pendulum Models**: Simplified models for balance control

### Dynamics Considerations

When modeling humanoid robots, consider:

1. **Mass Distribution**: Centralized mass in torso vs. distributed mass
2. **Moment of Inertia**: Affects stability and required actuator torques
3. **Actuator Limits**: Ensuring sufficient torque for motion
4. **Center of Mass**: Critical for balance during movement

### Sensor Integration

Humanoid robots typically include:

- **IMUs**: For balance and orientation
- **Force/Torque Sensors**: In feet and hands for contact information
- **Cameras**: For vision-based tasks
- **LIDAR**: For navigation
- **Joint Encoders**: For position feedback

### Model Validation

Validate your humanoid model:

1. **Forward Kinematics**: Verify that joint angles correspond to proper end-effector positions
2. **Inverse Kinematics**: Test that reaching tasks can be solved
3. **Balance**: Simulate standing and walking to verify center of mass behavior
4. **Range of Motion**: Ensure movements stay within joint limits
5. **Collision Detection**: Verify no self-collisions during normal movements

### Best Practices for Humanoid Modeling

1. **Modular Design**: Use Xacro to create modular components
2. **Scalability**: Design for easy modification of link lengths and masses
3. **Realistic Inertial Properties**: Use realistic mass and inertia values
4. **Safety Margins**: Include safety factors in actuator sizing
5. **Standard Interfaces**: Use standard joint and sensor interfaces

Humanoid modeling requires balancing mechanical constraints with the need for human-like movement capabilities, resulting in complex but rewarding robotic systems.