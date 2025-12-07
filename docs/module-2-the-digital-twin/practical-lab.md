---
id: module-2-practical-lab
title: Module 2 — The Digital Twin | Chapter 4 — Practical Lab
sidebar_label: Chapter 4 — Practical Lab
sidebar_position: 4
---

# Module 2 — The Digital Twin

## Chapter 4 — Practical Lab

### Laboratory Setup and Prerequisites

This practical lab focuses on implementing digital twin capabilities for humanoid robotics using Gazebo as the primary simulation platform. Before beginning, ensure you have:

#### Software Requirements
- Ubuntu 22.04 LTS or similar Linux distribution
- ROS 2 Humble Hawksbill installed
- Gazebo Garden or newer version
- Git for version control
- Basic development tools (gcc, g++, cmake)

#### Optional but Recommended
- Unity Hub with Unity Robotics packages (for Unity exercises)
- NVIDIA Isaac Sim (for advanced users with compatible hardware)
- RViz2 for visualization

### Lab Exercise 1: Gazebo Environment Setup and Basic Robot Model

#### Objective
Set up a Gazebo simulation environment and create a basic humanoid robot model.

#### Step-by-Step Instructions

1. **Create a new ROS 2 package for simulation**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python digital_twin_tutorial
   ```

2. **Create the package structure**:
   ```
   digital_twin_tutorial/
   ├── CMakeLists.txt
   ├── package.xml
   ├── launch/
   ├── worlds/
   │   └── simple_room.world
   ├── models/
   │   └── simple_humanoid/
   │       ├── model.sdf
   │       └── meshes/
   ├── config/
   │   └── robot_properties.yaml
   └── scripts/
   ```

3. **Create a basic world file** - Create `worlds/simple_room.world`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_room">
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1.0</real_time_factor>
         <real_time_update_rate>1000.0</real_time_update_rate>
       </physics>

       <include>
         <uri>model://sun</uri>
       </include>

       <include>
         <uri>model://ground_plane</uri>
       </include>

       <!-- Room with walls -->
       <model name="wall_front">
         <static>true</static>
         <pose>0 5 1 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_back">
         <static>true</static>
         <pose>0 -5 1 0 0 3.14159</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_left">
         <static>true</static>
         <pose>-5 0 1 0 0 -1.5708</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <model name="wall_right">
         <static>true</static>
         <pose>5 0 1 0 0 1.5708</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>10 0.1 2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <!-- Object for robot interaction -->
       <model name="table">
         <pose>2 0 0 0 0 0</pose>
         <link name="table_top">
           <collision name="collision">
             <geometry>
               <box>
                 <size>1.0 0.8 0.8</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1.0 0.8 0.8</size>
               </box>
             </geometry>
             <material>
               <ambient>0.6 0.4 0.2 1</ambient>
               <diffuse>0.6 0.4 0.2 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   ```

4. **Create a simple humanoid robot model** - Create `models/simple_humanoid/model.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <model name="simple_humanoid">
       <!-- Torso -->
       <link name="torso">
         <pose>0 0 0.8 0 0 0</pose>
         <inertial>
           <mass>10.0</mass>
           <inertia>
             <ixx>0.5</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.5</iyy>
             <iyz>0.0</iyz>
             <izz>0.5</izz>
           </inertia>
         </inertial>
         <visual name="torso_visual">
           <geometry>
             <box>
               <size>0.3 0.3 0.6</size>
             </box>
           </geometry>
           <material>
             <ambient>0.2 0.2 1.0 1</ambient>
             <diffuse>0.2 0.2 1.0 1</diffuse>
           </material>
         </visual>
         <collision name="torso_collision">
           <geometry>
             <box>
               <size>0.3 0.3 0.6</size>
             </box>
           </geometry>
         </collision>
       </link>

       <!-- Head -->
       <link name="head">
         <pose>0 0 0.3 0 0 0</pose>
         <inertial>
           <mass>2.0</mass>
           <inertia>
             <ixx>0.01</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.01</iyy>
             <iyz>0.0</iyz>
             <izz>0.01</izz>
           </inertia>
         </inertial>
         <visual name="head_visual">
           <geometry>
             <sphere>
               <radius>0.15</radius>
             </sphere>
           </geometry>
           <material>
             <ambient>0.8 0.8 0.8 1</ambient>
             <diffuse>0.8 0.8 0.8 1</diffuse>
           </material>
         </visual>
         <collision name="head_collision">
           <geometry>
             <sphere>
               <radius>0.15</radius>
             </sphere>
           </geometry>
         </collision>
       </link>

       <joint name="torso_head" type="fixed">
         <parent>torso</parent>
         <child>head</child>
         <pose>0 0 0.3 0 0 0</pose>
       </joint>

       <!-- Left Arm -->
       <link name="left_upper_arm">
         <pose>0.15 0 0.1 0 0 0</pose>
         <inertial>
           <mass>1.0</mass>
           <inertia>
             <ixx>0.01</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.01</iyy>
             <iyz>0.0</iyz>
             <izz>0.01</izz>
           </inertia>
         </inertial>
         <visual name="left_upper_arm_visual">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>1.0 0.0 0.0 1</ambient>
             <diffuse>1.0 0.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="left_upper_arm_collision">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="left_shoulder" type="revolute">
         <parent>torso</parent>
         <child>left_upper_arm</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-1.57</lower>
             <upper>1.57</upper>
             <effort>100</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0.15 0 0.1 0 0 0</pose>
       </joint>

       <link name="left_lower_arm">
         <pose>0.15 0 -0.15 0 0 0</pose>
         <inertial>
           <mass>0.5</mass>
           <inertia>
             <ixx>0.005</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.005</iyy>
             <iyz>0.0</iyz>
             <izz>0.005</izz>
           </inertia>
         </inertial>
         <visual name="left_lower_arm_visual">
           <geometry>
             <cylinder>
               <length>0.2</length>
               <radius>0.04</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>1.0 0.0 0.0 1</ambient>
             <diffuse>1.0 0.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="left_lower_arm_collision">
           <geometry>
             <cylinder>
               <length>0.2</length>
               <radius>0.04</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="left_elbow" type="revolute">
         <parent>left_upper_arm</parent>
         <child>left_lower_arm</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>0</lower>
             <upper>1.57</upper>
             <effort>50</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.3 0 0 0</pose>
       </joint>

       <!-- Right Arm (mirror of left) -->
       <link name="right_upper_arm">
         <pose>-0.15 0 0.1 0 0 0</pose>
         <inertial>
           <mass>1.0</mass>
           <inertia>
             <ixx>0.01</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.01</iyy>
             <iyz>0.0</iyz>
             <izz>0.01</izz>
           </inertia>
         </inertial>
         <visual name="right_upper_arm_visual">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>1.0 0.0 0.0 1</ambient>
             <diffuse>1.0 0.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="right_upper_arm_collision">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="right_shoulder" type="revolute">
         <parent>torso</parent>
         <child>right_upper_arm</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-1.57</lower>
             <upper>1.57</upper>
             <effort>100</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>-0.15 0 0.1 0 0 0</pose>
       </joint>

       <link name="right_lower_arm">
         <pose>-0.15 0 -0.15 0 0 0</pose>
         <inertial>
           <mass>0.5</mass>
           <inertia>
             <ixx>0.005</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.005</iyy>
             <iyz>0.0</iyz>
             <izz>0.005</izz>
           </inertia>
         </inertial>
         <visual name="right_lower_arm_visual">
           <geometry>
             <cylinder>
               <length>0.2</length>
               <radius>0.04</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>1.0 0.0 0.0 1</ambient>
             <diffuse>1.0 0.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="right_lower_arm_collision">
           <geometry>
             <cylinder>
               <length>0.2</length>
               <radius>0.04</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="right_elbow" type="revolute">
         <parent>right_upper_arm</parent>
         <child>right_lower_arm</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>0</lower>
             <upper>1.57</upper>
             <effort>50</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.3 0 0 0</pose>
       </joint>

       <!-- Legs -->
       <link name="left_upper_leg">
         <pose>0.05 0 -0.3 0 0 0</pose>
         <inertial>
           <mass>2.0</mass>
           <inertia>
             <ixx>0.05</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.05</iyy>
             <iyz>0.0</iyz>
             <izz>0.05</izz>
           </inertia>
         </inertial>
         <visual name="left_upper_leg_visual">
           <geometry>
             <cylinder>
               <length>0.4</length>
               <radius>0.06</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>0.0 1.0 0.0 1</ambient>
             <diffuse>0.0 1.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="left_upper_leg_collision">
           <geometry>
             <cylinder>
               <length>0.4</length>
               <radius>0.06</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="left_hip" type="revolute">
         <parent>torso</parent>
         <child>left_upper_leg</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-0.78</lower>
             <upper>0.78</upper>
             <effort>200</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0.05 0 -0.3 0 0 0</pose>
       </joint>

       <link name="left_lower_leg">
         <pose>0.05 0 -0.7 0 0 0</pose>
         <inertial>
           <mass>1.5</mass>
           <inertia>
             <ixx>0.03</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.03</iyy>
             <iyz>0.0</iyz>
             <izz>0.03</izz>
           </inertia>
         </inertial>
         <visual name="left_lower_leg_visual">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>0.0 1.0 0.0 1</ambient>
             <diffuse>0.0 1.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="left_lower_leg_collision">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="left_knee" type="revolute">
         <parent>left_upper_leg</parent>
         <child>left_lower_leg</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>0</lower>
             <upper>1.57</upper>
             <effort>150</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.4 0 0 0</pose>
       </joint>

       <link name="left_foot">
         <pose>0.05 0 -0.85 0 0 0</pose>
         <inertial>
           <mass>0.5</mass>
           <inertia>
             <ixx>0.01</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.01</iyy>
             <iyz>0.0</iyz>
             <izz>0.01</izz>
           </inertia>
         </inertial>
         <visual name="left_foot_visual">
           <geometry>
             <box>
               <size>0.15 0.08 0.05</size>
             </box>
           </geometry>
           <material>
             <ambient>0.3 0.3 0.3 1</ambient>
             <diffuse>0.3 0.3 0.3 1</diffuse>
           </material>
         </visual>
         <collision name="left_foot_collision">
           <geometry>
             <box>
               <size>0.15 0.08 0.05</size>
             </box>
           </geometry>
         </collision>
       </link>

       <joint name="left_ankle" type="revolute">
         <parent>left_lower_leg</parent>
         <child>left_foot</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-0.5</lower>
             <upper>0.5</upper>
             <effort>50</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.15 0 0 0</pose>
       </joint>

       <!-- Right leg (mirror of left) -->
       <link name="right_upper_leg">
         <pose>-0.05 0 -0.3 0 0 0</pose>
         <inertial>
           <mass>2.0</mass>
           <inertia>
             <ixx>0.05</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.05</iyy>
             <iyz>0.0</iyz>
             <izz>0.05</izz>
           </inertia>
         </inertial>
         <visual name="right_upper_leg_visual">
           <geometry>
             <cylinder>
               <length>0.4</length>
               <radius>0.06</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>0.0 1.0 0.0 1</ambient>
             <diffuse>0.0 1.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="right_upper_leg_collision">
           <geometry>
             <cylinder>
               <length>0.4</length>
               <radius>0.06</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="right_hip" type="revolute">
         <parent>torso</parent>
         <child>right_upper_leg</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-0.78</lower>
             <upper>0.78</upper>
             <effort>200</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>-0.05 0 -0.3 0 0 0</pose>
       </joint>

       <link name="right_lower_leg">
         <pose>-0.05 0 -0.7 0 0 0</pose>
         <inertial>
           <mass>1.5</mass>
           <inertia>
             <ixx>0.03</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.03</iyy>
             <iyz>0.0</iyz>
             <izz>0.03</izz>
           </inertia>
         </inertial>
         <visual name="right_lower_leg_visual">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
           <material>
             <ambient>0.0 1.0 0.0 1</ambient>
             <diffuse>0.0 1.0 0.0 1</diffuse>
           </material>
         </visual>
         <collision name="right_lower_leg_collision">
           <geometry>
             <cylinder>
               <length>0.3</length>
               <radius>0.05</radius>
             </cylinder>
           </geometry>
         </collision>
       </link>

       <joint name="right_knee" type="revolute">
         <parent>right_upper_leg</parent>
         <child>right_lower_leg</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>0</lower>
             <upper>1.57</upper>
             <effort>150</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.4 0 0 0</pose>
       </joint>

       <link name="right_foot">
         <pose>-0.05 0 -0.85 0 0 0</pose>
         <inertial>
           <mass>0.5</mass>
           <inertia>
             <ixx>0.01</ixx>
             <ixy>0.0</ixy>
             <ixz>0.0</ixz>
             <iyy>0.01</iyy>
             <iyz>0.0</iyz>
             <izz>0.01</izz>
           </inertia>
         </inertial>
         <visual name="right_foot_visual">
           <geometry>
             <box>
               <size>0.15 0.08 0.05</size>
             </box>
           </geometry>
           <material>
             <ambient>0.3 0.3 0.3 1</ambient>
             <diffuse>0.3 0.3 0.3 1</diffuse>
           </material>
         </visual>
         <collision name="right_foot_collision">
           <geometry>
             <box>
               <size>0.15 0.08 0.05</size>
             </box>
           </geometry>
         </collision>
       </link>

       <joint name="right_ankle" type="revolute">
         <parent>right_lower_leg</parent>
         <child>right_foot</child>
         <axis>
           <xyz>0 1 0</xyz>
           <limit>
             <lower>-0.5</lower>
             <upper>0.5</upper>
             <effort>50</effort>
             <velocity>1</velocity>
           </limit>
         </axis>
         <pose>0 0 -0.15 0 0 0</pose>
       </joint>

       <!-- Sensors -->
       <sensor name="camera_left" type="camera">
         <pose>0.05 0.05 0.2 0 0 0</pose>
         <camera>
           <horizontal_fov>1.047</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
           </image>
           <clip>
             <near>0.1</near>
             <far>10</far>
           </clip>
         </camera>
         <plugin name="camera_left_plugin" filename="libgazebo_ros_camera.so">
           <ros>
             <namespace>/simple_humanoid</namespace>
             <remapping>~/image_raw:=left_camera/image_raw</remapping>
             <remapping>~/camera_info:=left_camera/camera_info</remapping>
           </ros>
           <camera_name>left_camera</camera_name>
           <frame_name>left_camera_frame</frame_name>
         </plugin>
       </sensor>

       <sensor name="imu_sensor" type="imu">
         <pose>0 0 0.1 0 0 0</pose>
         <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
           <ros>
             <namespace>/simple_humanoid</namespace>
             <remapping>~/out:=imu/data</remapping>
           </ros>
           <frame_name>torso_imu_frame</frame_name>
         </plugin>
       </sensor>
     </model>
   </sdf>
   ```

5. **Create a launch file** - Create `launch/digital_twin.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get package directories
       pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
       pkg_digital_twin = get_package_share_directory('digital_twin_tutorial')

       # World file
       world = os.path.join(pkg_digital_twin, 'worlds', 'simple_room.world')

       # Launch Gazebo server
       gzserver = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
           ),
           launch_arguments={'world': world}.items(),
       )

       # Launch Gazebo client
       gzclient = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
           ),
       )

       # Robot state publisher node
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           parameters=[{
               'use_sim_time': True,
               'publish_frequency': 50.0
           }],
           arguments=[
               os.path.join(pkg_digital_twin, 'models', 'simple_humanoid', 'model.sdf')
           ]
       )

       return LaunchDescription([
           gzserver,
           gzclient,
           robot_state_publisher,
       ])
   ```

6. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select digital_twin_tutorial
   source install/setup.bash
   ```

7. **Run the simulation**:
   ```bash
   ros2 launch digital_twin_tutorial digital_twin.launch.py
   ```

#### Expected Results
You should see the Gazebo simulation window with a simple humanoid robot in a room environment. The robot should include basic sensors as defined in the model.

### Lab Exercise 2: Implementing Real-time Synchronization

#### Objective
Create a node that demonstrates synchronization between physical and digital systems.

#### Step-by-Step Instructions

1. **Create a synchronization node** - Create `digital_twin_tutorial/digital_twin_tutorial/synchronization_demo.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32MultiArray
   from sensor_msgs.msg import JointState
   from geometry_msgs.msg import Twist
   import time
   import math
   from collections import deque

   class DigitalTwinSynchronizer(Node):
       def __init__(self):
           super().__init__('digital_twin_synchronizer')

           # Publisher for synchronized joint states
           self.joint_pub = self.create_publisher(
               JointState,
               '/digital_twin/joint_states',
               10
           )

           # Subscriber for physical robot joint states
           self.joint_sub = self.create_subscription(
               JointState,
               '/physical_robot/joint_states',
               self.joint_state_callback,
               10
           )

           # Publisher for robot commands (simulated physical robot)
           self.cmd_pub = self.create_publisher(
               Float32MultiArray,
               '/physical_robot/commands',
               10
           )

           # Timer for synchronization
           self.timer = self.create_timer(0.05, self.sync_callback)  # 20 Hz

           # Internal state tracking
           self.physical_joint_positions = {}
           self.digital_joint_positions = {}
           self.joint_names = [
               'left_hip', 'left_knee', 'left_ankle',
               'right_hip', 'right_knee', 'right_ankle',
               'left_shoulder', 'left_elbow',
               'right_shoulder', 'right_elbow'
           ]

           # Initialize all joint positions to 0
           for name in self.joint_names:
               self.physical_joint_positions[name] = 0.0
               self.digital_joint_positions[name] = 0.0

           # Synchronization delay simulation (network latency)
           self.sync_delay = 0.1  # 100ms delay
           self.state_history = deque(maxlen=100)

           self.get_logger().info('Digital Twin Synchronizer initialized')

       def joint_state_callback(self, msg):
           """Callback for physical robot joint states"""
           self.get_logger().info(f'Received {len(msg.name)} joint states')

           # Update internal state with received values
           for i, name in enumerate(msg.name):
               if name in self.physical_joint_positions:
                   self.physical_joint_positions[name] = msg.position[i]

           # Store in history for delay compensation
           current_time = self.get_clock().now().nanoseconds * 1e-9
           self.state_history.append((current_time, dict(self.physical_joint_positions)))

       def sync_callback(self):
           """Main synchronization loop"""
           # Simulate a simple control loop
           current_time = self.get_clock().now().nanoseconds * 1e-9

           # Simulate command generation based on internal state
           command_msg = Float32MultiArray()

           # Generate oscillating motion for demonstration
           t = current_time
           for name in self.joint_names:
               # Different oscillation patterns for different joint types
               if 'hip' in name or 'knee' in name:
                   # Leg movements - slow oscillation
                   cmd = 0.2 * math.sin(0.5 * t + hash(name) % 10)
               else:
                   # Arm movements - faster oscillation
                   cmd = 0.3 * math.sin(1.0 * t + hash(name) % 10)

               command_msg.data.append(cmd)

           # Publish commands to simulated physical robot
           self.cmd_pub.publish(command_msg)

           # Publish synchronized joint states for digital twin visualization
           self.publish_synchronized_states(current_time)

           self.get_logger().debug('Synchronization step completed')

       def publish_synchronized_states(self, current_time):
           """Publish synchronized joint states for visualization"""
           msg = JointState()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.header.frame_id = 'digital_twin_base'

           # Add joint names
           msg.name = self.joint_names

           # Add positions (could include compensation for delay here)
           msg.position = [self.physical_joint_positions[name] for name in self.joint_names]

           # Add velocities and efforts (simulated)
           msg.velocity = [0.0] * len(self.joint_names)
           msg.effort = [0.0] * len(self.joint_names)

           self.joint_pub.publish(msg)

   def main(args=None):
       rclpy.init(args=args)
       synchronizer = DigitalTwinSynchronizer()

       try:
           rclpy.spin(synchronizer)
       except KeyboardInterrupt:
           synchronizer.get_logger().info('Synchronization node stopped')
       finally:
           synchronizer.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Update setup.py** to include the new script:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'digital_twin_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
           (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
           (os.path.join('share', package_name, 'models/simple_humanoid'), glob('models/simple_humanoid/*')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Digital twin tutorial for humanoid robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'synchronization_demo = digital_twin_tutorial.synchronization_demo:main',
           ],
       },
   )
   ```

3. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select digital_twin_tutorial
   source install/setup.bash
   ```

4. **Run the synchronization demo**:
   ```bash
   ros2 run digital_twin_tutorial synchronization_demo
   ```

#### Expected Results
The synchronization node should publish joint states that could represent the digital twin's current state. In a real system, this would be compared with physical robot states to maintain synchronization.

### Lab Exercise 3: Unity Integration (Optional)

#### Objective
Set up basic Unity integration to create a visual digital twin.

#### Step-by-Step Instructions

1. **Set up Unity Hub and install Unity** (if not already done)

2. **Install ROS-TCP-Connector package** through Unity Package Manager

3. **Create a basic Unity scene** that connects to ROS:

   ```csharp
   using UnityEngine;
   using RosSharp.RosBridgeClient;

   public class UnityDigitalTwin : MonoBehaviour
   {
       [Header("ROS Settings")]
       public string rosBridgeServerUrl = "ws://127.0.0.1:9090";
       public float updateRate = 0.02f; // 50 Hz

       [Header("Robot Configuration")]
       public Transform[] jointTransforms;  // Assign in editor
       public string[] jointNames;          // Names for ROS messages

       private RosSocket rosSocket;
       private JointStateSubscriber jointStateSubscriber;
       private float updateTimer;

       void Start()
       {
           // Initialize ROS connection
           WebSocketNativeClient webSocket = new WebSocketNativeClient(rosBridgeServerUrl);
           rosSocket = new RosSocket(webSocket);

           // Subscribe to joint states topic
           jointStateSubscriber = new JointStateSubscriber(rosSocket, "/digital_twin/joint_states",
               UpdateJointPositions);
       }

       void Update()
       {
           // Update timer for potential publishing
           updateTimer += Time.deltaTime;
       }

       private void UpdateJointPositions(float[] positions, string[] names)
       {
           // Update the Unity transforms based on received joint positions
           for (int i = 0; i < jointTransforms.Length && i < positions.Length; i++)
           {
               // Update the rotation of each joint based on received position
               // This assumes revolute joints rotating around Y-axis
               jointTransforms[i].localEulerAngles = new Vector3(0, positions[i] * Mathf.Rad2Deg, 0);
           }
       }
   }
   ```

4. **Build and run the Unity scene** to visualize the digital twin

#### Expected Results
The Unity scene should display a 3D model of the robot that updates in real-time based on ROS joint state messages.

### Lab Exercise 4: Digital Twin Analytics and Monitoring

#### Objective
Create a monitoring system that tracks digital twin performance and provides analytics.

#### Step-by-Step Instructions

1. **Create an analytics node** - Create `digital_twin_tutorial/digital_twin_tutorial/analytics_node.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32
   from sensor_msgs.msg import JointState
   from digital_twin_tutorial.msg import DigitalTwinMetrics  # You would need to define this message
   import time
   import numpy as np
   from collections import deque

   class DigitalTwinAnalytics(Node):
       def __init__(self):
           super().__init__('digital_twin_analytics')

           # Subscribers for various data sources
           self.joint_sub = self.create_subscription(
               JointState,
               '/digital_twin/joint_states',
               self.joint_callback,
               10
           )

           self.physical_joint_sub = self.create_subscription(
               JointState,
               '/physical_robot/joint_states',
               self.physical_joint_callback,
               10
           )

           # Publisher for analytics
           self.analytics_pub = self.create_publisher(
               Float32,
               '/digital_twin/analytics/synchronization_error',
               10
           )

           # Timer for analytics computation
           self.timer = self.create_timer(1.0, self.compute_analytics)

           # Data storage for analytics
           self.joint_sync_errors = {}
           self.performance_history = deque(maxlen=100)
           self.joint_names = []

           # Initialize error tracking
           for joint_name in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
               self.joint_sync_errors[joint_name] = deque(maxlen=50)

           self.get_logger().info('Digital Twin Analytics node initialized')

       def joint_callback(self, msg):
           """Callback for digital twin joint states"""
           if not self.joint_names:
               self.joint_names = msg.name

           # Update internal tracking (for this example, we're just storing the data)
           for i, name in enumerate(msg.name):
               if name in self.joint_sync_errors:
                   # In a real system, we'd compare with physical robot state here
                   pass

       def physical_joint_callback(self, msg):
           """Callback for physical robot joint states"""
           # Calculate synchronization error
           for i, name in enumerate(msg.name):
               if name in self.joint_sync_errors:
                   # In a real system, we'd compare with the corresponding digital twin state
                   # For this example, we'll simulate some error
                   error = abs(msg.position[i]) * 0.01  # Simulate 1% error
                   self.joint_sync_errors[name].append(error)

       def compute_analytics(self):
           """Compute and publish analytics"""
           # Calculate average synchronization error for key joints
           total_error = 0
           error_count = 0

           for joint_name, errors in self.joint_sync_errors.items():
               if errors:
                   avg_error = sum(errors) / len(errors)
                   total_error += avg_error
                   error_count += 1

           if error_count > 0:
               overall_error = total_error / error_count

               # Publish the error metric
               error_msg = Float32()
               error_msg.data = overall_error
               self.analytics_pub.publish(error_msg)

               self.get_logger().info(f'Average synchronization error: {overall_error:.4f}')

               # Store in performance history
               self.performance_history.append({
                   'timestamp': time.time(),
                   'sync_error': overall_error,
                   'computation_load': np.random.uniform(0.1, 0.8)  # Placeholder
               })

   def main(args=None):
       rclpy.init(args=args)
       analytics_node = DigitalTwinAnalytics()

       try:
           rclpy.spin(analytics_node)
       except KeyboardInterrupt:
           analytics_node.get_logger().info('Analytics node stopped')
       finally:
           analytics_node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create a custom message for digital twin metrics** - Create `digital_twin_tutorial/msg/DigitalTwinMetrics.msg`:
   ```
   float32 synchronization_error
   float32 computation_load
   float32 network_latency
   string status
   ```

3. **Update package.xml** to include message dependencies:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>digital_twin_tutorial</name>
     <version>0.0.0</version>
     <description>Digital twin tutorial for humanoid robotics</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>sensor_msgs</depend>
     <depend>geometry_msgs</depend>

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

4. **Update setup.py** to include the message:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'digital_twin_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
           (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
           (os.path.join('share', package_name, 'models/simple_humanoid'), glob('models/simple_humanoid/*')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Digital twin tutorial for humanoid robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'synchronization_demo = digital_twin_tutorial.synchronization_demo:main',
               'analytics_node = digital_twin_tutorial.analytics_node:main',
           ],
       },
   )
   ```

5. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select digital_twin_tutorial
   source install/setup.bash
   ```

#### Expected Results
The analytics node will monitor the digital twin system and publish metrics about synchronization quality, computation load, and other performance indicators.

### Lab Exercise 5: Integration and Testing

#### Objective
Integrate all components and test the complete digital twin system.

#### Step-by-Step Instructions

1. **Create an integration launch file** - Create `launch/integrated_digital_twin.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import IncludeLaunchDescription, TimerAction
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Get package directories
       pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
       pkg_digital_twin = get_package_share_directory('digital_twin_tutorial')

       # World file
       world = os.path.join(pkg_digital_twin, 'worlds', 'simple_room.world')

       # Launch Gazebo server
       gzserver = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
           ),
           launch_arguments={'world': world}.items(),
       )

       # Launch Gazebo client
       gzclient = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
           ),
       )

       # Digital twin synchronizer node
       synchronizer_node = Node(
           package='digital_twin_tutorial',
           executable='synchronization_demo',
           name='digital_twin_synchronizer',
       )

       # Analytics node
       analytics_node = Node(
           package='digital_twin_tutorial',
           executable='analytics_node',
           name='digital_twin_analytics',
       )

       # Robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           parameters=[{
               'use_sim_time': True,
               'publish_frequency': 50.0
           }],
       )

       return LaunchDescription([
           gzserver,
           gzclient,
           TimerAction(
               period=5.0,  # Wait 5 seconds after Gazebo starts
               actions=[
                   synchronizer_node,
                   analytics_node,
                   robot_state_publisher,
               ]
           )
       ])
   ```

2. **Run the complete integrated system**:
   ```bash
   ros2 launch digital_twin_tutorial integrated_digital_twin.launch.py
   ```

3. **Monitor the system** using various tools:
   ```bash
   # Monitor topics
   ros2 topic echo /digital_twin/joint_states

   # Monitor analytics
   ros2 topic echo /digital_twin/analytics/synchronization_error

   # Check the node graph
   ros2 run rqt_graph rqt_graph
   ```

#### Expected Results
All components should work together to create a complete digital twin system with:
- Gazebo simulation of the humanoid robot
- Real-time state synchronization
- Analytics and performance monitoring
- Proper ROS 2 communication patterns

### Troubleshooting Guide

#### Common Issues and Solutions

1. **Gazebo Not Starting**:
   - Check if GPU drivers are properly installed
   - Verify X11 forwarding if running in a container
   - Check for conflicting processes using the same ports

2. **Joint States Not Syncing**:
   - Verify topic names match between publisher and subscriber
   - Check QoS profiles compatibility
   - Confirm both systems are using same time source (sim_time vs real time)

3. **Performance Issues**:
   - Reduce simulation complexity
   - Optimize collision geometry (use simpler shapes)
   - Adjust physics parameters (step size, solver iterations)

4. **Network Communication Problems**:
   - Verify ROS domain ID consistency
   - Check firewall settings
   - Confirm IP addresses and ports are accessible

### Conclusion

This practical lab has walked you through creating a complete digital twin system for humanoid robotics. You've implemented:

1. A Gazebo-based simulation environment
2. Real-time synchronization mechanisms
3. Analytics and monitoring tools
4. Integration with visualization systems

The system you've built provides a foundation for more advanced digital twin applications in humanoid robotics, including predictive maintenance, algorithm testing, and operator training scenarios.