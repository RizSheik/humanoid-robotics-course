---
id: module-1-chapter-4-practical-exercises
title: 'Module 1 — The Robotic Nervous System | Chapter 4 — Practical Exercises'
sidebar_label: 'Chapter 4 — Practical Exercises'
---

# Chapter 4 — Practical Exercises

## URDF and Humanoid Modeling: Hands-On Implementation

This practical lab focuses on creating and validating URDF models for humanoid robots. You will learn to build complex robot models and simulate them in a physics environment.

### Exercise 1: Basic Robot Model Creation

#### Objective
Create a simple wheeled robot model using URDF and visualize it in RViz.

#### Steps
1. Create a new ROS 2 package for the robot model
2. Create the URDF file for a differential drive robot
3. Visualize the robot in RViz

```xml
<!-- Create a file called basic_robot.urdf -->
<?xml version="1.0"?>
<robot name="differential_drive_robot">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
  
  <!-- Left wheel -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.18 -0.05" rpy="1.57079632679 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>
  
  <!-- Right wheel -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.18 -0.05" rpy="1.57079632679 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>
  
  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  </joint>
  
  <link name="camera">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>
</robot>
```

To visualize this model:
```bash
# Install joint state publisher GUI
sudo apt-get install ros-humble-joint-state-publisher-gui

# Launch RViz with the robot model
ros2 launch urdf_tutorial display.launch.py model_path:=path_to_your_basic_robot.urdf
```

### Exercise 2: Humanoid Limb Modeling

#### Objective
Create a functional arm model with multiple degrees of freedom using Xacro.

#### Steps
1. Create a Xacro file for a simple humanoid arm
2. Include all necessary joints and links
3. Verify the kinematic structure

```xml
<!-- arm_model.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_arm">
  
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  
  <!-- Arm parameters -->
  <xacro:property name="upper_arm_length" value="0.3"/>
  <xacro:property name="lower_arm_length" value="0.25"/>
  <xacro:property name="arm_radius" value="0.04"/>
  <xacro:property name="hand_mass" value="0.3"/>
  
  <!-- Upper arm macro -->
  <xacro:macro name="upper_arm" params="side parent xyz rpy">
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="100" velocity="2.0"/>
      <dynamics damping="1.0" friction="0.0"/>
    </joint>
    
    <link name="${side}_upper_arm">
      <visual>
        <origin xyz="0 0 ${upper_arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_radius}" length="${upper_arm_length}"/>
        </geometry>
        <material name="arm_color">
          <color rgba="0.7 0.7 0.7 1.0"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${upper_arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_radius}" length="${upper_arm_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 ${upper_arm_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
      </inertial>
    </link>
    
    <!-- Elbow joint -->
    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 ${upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="${0.8*M_PI}" effort="100" velocity="2.0"/>
      <dynamics damping="1.0" friction="0.0"/>
    </joint>
    
    <!-- Lower arm -->
    <link name="${side}_lower_arm">
      <visual>
        <origin xyz="0 0 ${lower_arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_radius*0.8}" length="${lower_arm_length}"/>
        </geometry>
        <material name="arm_color">
          <color rgba="0.5 0.5 0.5 1.0"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 ${lower_arm_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_radius*0.8}" length="${lower_arm_length}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <origin xyz="0 0 ${lower_arm_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.004"/>
      </inertial>
    </link>
    
    <!-- Wrist joint -->
    <joint name="${side}_wrist_joint" type="revolute">
      <parent link="${side}_lower_arm"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 ${lower_arm_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="50" velocity="2.0"/>
      <dynamics damping="0.5" friction="0.0"/>
    </joint>
    
    <!-- Hand -->
    <link name="${side}_hand">
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="0.1 0.08 0.05"/>
        </geometry>
        <material name="hand_color">
          <color rgba="0.8 0.6 0.4 1.0"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="0.1 0.08 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="${hand_mass}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>
  
  <!-- Base link to attach the arm -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.3 0.4"/>
      </geometry>
      <material name="torso_color">
        <color rgba="0.3 0.5 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.3 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Instance the arms -->
  <xacro:upper_arm side="left" parent="torso" 
                  xyz="0 0.15 0.1" rpy="0 0 0"/>
  <xacro:upper_arm side="right" parent="torso" 
                  xyz="0 -0.15 0.1" rpy="0 0 0"/>
</robot>
```

### Exercise 3: Complete Humanoid Model with Gazebo Integration

#### Objective
Create a complete humanoid model that can be simulated in Gazebo with physics properties and sensors.

#### Steps
1. Create a full humanoid model with all limbs
2. Add Gazebo-specific tags for physics simulation
3. Include sensors for perception

```xml
<!-- complete_humanoid.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complete_humanoid">
  
  <!-- Load Gazebo plugins and materials -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/transmission.xacro"/>
  
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  
  <!-- Gazebo Material Definition -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>
  
  <!-- Base and world joint -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Fixed joint to root -->
  <joint name="base_root_joint" type="fixed">
    <parent link="base_link"/>
    <child link="root"/>
  </joint>

  <!-- ROOT LINK - Main body frame -->
  <link name="root">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Pelvis -->
  <joint name="root_to_pelvis" type="fixed">
    <parent link="root"/>
    <child link="pelvis"/>
    <origin xyz="0 0 0"/>
  </joint>

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

  <!-- Torso -->
  <joint name="torso_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.0875"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="30" velocity="1.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.18 0.2 0.2"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.18 0.2 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.21"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="5" velocity="2.0"/>
    <dynamics damping="0.2" friction="0.05"/>
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

  <!-- Sensors -->
  <joint name="imu_joint" type="fixed">
    <parent link="torso"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <!-- Gazebo Sensor Plugins -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="30" velocity="1.0"/>
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
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="30" velocity="1.0"/>
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

  <!-- Right Arm -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.0 -0.1 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="30" velocity="1.0"/>
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
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="30" velocity="1.0"/>
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

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="left_thigh"/>
    <origin xyz="0.0 0.05 -0.075"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="50" velocity="1.0"/>
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
    <limit lower="0" upper="${0.8*M_PI}" effort="50" velocity="1.0"/>
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
    <limit lower="${-0.3*M_PI}" upper="${0.3*M_PI}" effort="20" velocity="1.0"/>
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
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Leg -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="pelvis"/>
    <child link="right_thigh"/>
    <origin xyz="0.0 -0.05 -0.075"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5*M_PI}" upper="${0.5*M_PI}" effort="50" velocity="1.0"/>
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
    <limit lower="0" upper="${0.8*M_PI}" effort="50" velocity="1.0"/>
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
    <limit lower="${-0.3*M_PI}" upper="${0.3*M_PI}" effort="20" velocity="1.0"/>
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
        <box size="0.15 0.1 0.05"/>
      </geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <origin xyz="0.05 0.0 -0.05"/>
      <geometry>
        <box size="0.15 0.1 0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <parameters>$(find my_robot_description)/config/hardware_control.yaml</parameters>
    </plugin>
  </gazebo>
</robot>
```

### Exercise 4: Model Validation and Testing

#### Objective
Validate the robot model by testing kinematic and dynamic properties.

#### Steps
1. Use URDF validation tools
2. Test forward and inverse kinematics
3. Check for collisions and self-collisions

```python
# validation_test.py - Python script to test the robot model
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import numpy as np

class ModelValidator(Node):
    def __init__(self):
        super().__init__('model_validator')
        
        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer to publish joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        
        # Initialize joint names and positions
        self.joint_names = [
            'torso_joint', 'neck_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint',
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        self.get_logger().info('Model validator started')

    def publish_joint_states(self):
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
    validator = ModelValidator()
    
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

### Exercise 5: Advanced Modeling - Adding Actuators and Transmissions

#### Objective
Add transmission information to make the model ready for hardware control.

```xml
<!-- transmissions.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <xacro:macro name="transmission_block" params="joint_name hardware_interface:=PositionJointInterface actuator_name:=${joint_name}_motor reduction:=1">
    <transmission name="${joint_name}_trans">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="${joint_name}">
        <hardwareInterface>${hardware_interface}</hardwareInterface>
      </joint>
      <actuator name="${actuator_name}">
        <mechanicalReduction>${reduction}</mechanicalReduction>
      </actuator>
    </transmission>
  </xacro:macro>
  
  <!-- Apply transmissions to all joints -->
  <xacro:transmission_block joint_name="torso_joint"/>
  <xacro:transmission_block joint_name="neck_joint"/>
  <xacro:transmission_block joint_name="left_shoulder_joint"/>
  <xacro:transmission_block joint_name="left_elbow_joint"/>
  <xacro:transmission_block joint_name="right_shoulder_joint"/>
  <xacro:transmission_block joint_name="right_elbow_joint"/>
  <xacro:transmission_block joint_name="left_hip_joint"/>
  <xacro:transmission_block joint_name="left_knee_joint"/>
  <xacro:transmission_block joint_name="left_ankle_joint"/>
  <xacro:transmission_block joint_name="right_hip_joint"/>
  <xacro:transmission_block joint_name="right_knee_joint"/>
  <xacro:transmission_block joint_name="right_ankle_joint"/>
</robot>
```

### Assessment Criteria

Your implementation will be evaluated based on:

1. **Model Validity**: The URDF files correctly represent the robot structure
2. **Kinematic Consistency**: All joints and links are properly connected
3. **Dynamic Properties**: Appropriate mass and inertia values
4. **Simulation Readiness**: Models work in simulation environments
5. **Documentation**: Clear, well-commented URDF/Xacro files

### Troubleshooting Tips

1. **URDF Validation**: Use `check_urdf` and `urdf_to_graphiz` to validate your models
2. **Joint Limits**: Ensure joint limits are realistic for the intended robot
3. **Inertial Properties**: Use realistic inertial values for accurate simulation
4. **Collision Detection**: Test for self-collisions during normal movement ranges
5. **Gazebo Integration**: Verify that Gazebo plugins are correctly configured

### Extensions for Advanced Students

- Implement custom controllers for the robot in Gazebo
- Add more complex sensors and simulate their data streams
- Create a walking pattern generator for the humanoid model
- Implement a simple controller to make the robot stand up or walk
- Add hand models with grasp capabilities

This practical exercise provides hands-on experience with creating complex humanoid robot models that are ready for simulation and eventually control.