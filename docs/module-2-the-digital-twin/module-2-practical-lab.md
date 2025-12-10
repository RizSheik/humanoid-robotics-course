---
id: module-2-practical-lab
title: 'Module 2 — The Digital Twin | Chapter 4 — Practical Lab'
sidebar_label: 'Chapter 4 — Practical Lab'
sidebar_position: 4
---

# Chapter 4 — Practical Lab

## Setting Up Digital Twin Simulation Environments

In this practical lab, we'll set up and configure simulation environments using both Gazebo and Unity for creating digital twins of robotic systems. We'll start with Gazebo, then explore Unity integration with ROS 2.

### Prerequisites

Before starting this lab, ensure you have:
- Completed Module 1 (The Robotic Nervous System)
- Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill installed
- Basic familiarity with URDF robot modeling
- Sufficient disk space for simulation environments (10+ GB)
- For Unity: Windows or Linux with a capable GPU and Unity Hub installed
- For Isaac Sim: NVIDIA GPU with CUDA support

## Lab 1: Advanced Gazebo Environment Setup

### Installing Gazebo Garden

First, we'll upgrade from the older Gazebo Classic to Gazebo Garden (the newer version):

```bash
# Install Gazebo Garden
sudo apt update
sudo apt install gazebo libgazebo-dev

# Verify installation
gz --version

# Install additional packages for ROS 2 integration
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### Testing Gazebo Installation

1. **Launch Gazebo with an empty world**:
```bash
gz sim
```

2. **Try basic operations**:
   - Click and drag to move the camera
   - Use the Entities panel to examine and modify objects
   - Use the Components panel to view and edit properties

### Creating a Custom World

1. **Create a directory for your custom worlds**:
```bash
mkdir -p ~/ros2_ws/src/my_robot_pkg/worlds
```

2. **Create a simple world file** (`~/ros2_ws/src/my_robot_pkg/worlds/simple_circuit.sdf`):
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_circuit">
    <!-- Include the outdoor environment -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Creating a simple circuit track with blocks -->
    <model name="block_1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="block_2">
      <pose>-2 0 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0 1 0 1</ambient>
            <diffuse>0 1 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="block_3">
      <pose>0 2 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="block_4">
      <pose>0 -2 0.5 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 1 0 1</ambient>
            <diffuse>1 1 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

3. **Launch Gazebo with your custom world**:
```bash
gz sim ~/ros2_ws/src/my_robot_pkg/worlds/simple_circuit.sdf
```

## Lab 2: Advanced Robot Model with Sensors

### Creating a Robot with Multiple Sensors

1. **Update your robot URDF** (`~/ros2_ws/src/my_robot_pkg/urdf/my_advanced_robot.urdf.xacro`):
```xml
<?xml version="1.0"?>
<robot name="my_advanced_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>
  
  <!-- Material definitions -->
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

  <!-- Camera -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- IMU sensor -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- LIDAR -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="lidar_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0002"/>
    </inertial>
  </link>

</robot>
```

### Adding Gazebo Integration

2. **Create Gazebo integration file** (`~/ros2_ws/src/my_robot_pkg/urdf/my_advanced_robot.gazebo.xacro`):
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

  <gazebo reference="camera_link">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="lidar_link">
    <material>Gazebo/Orange</material>
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

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
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
      <plugin filename="libgazebo_ros_camera.so" name="camera_controller">
        <frame_name>camera_link</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>100.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <gazebo reference="imu_link">
    <sensor type="imu" name="imu_sensor">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <topic>__default_topic__</topic>
      <plugin filename="libgazebo_ros_imu_sensor.so" name="imu_plugin">
        <topicName>imu</topicName>
        <bodyName>imu_link</bodyName>
        <updateRateHZ>100.0</updateRateHZ>
        <gaussianNoise>0.01</gaussianNoise>
        <xyzOffset>0 0 0</xyzOffset>
        <rpyOffset>0 0 0</rpyOffset>
        <frameName>imu_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LIDAR sensor -->
  <gazebo reference="lidar_link">
    <sensor type="ray" name="lidar_sensor">
      <pose>0 0 0 0 0 0</pose>
      <visualize>false</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1.0</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin filename="libgazebo_ros_ray_sensor.so" name="lidar_plugin">
        <ros>
          <namespace>/my_robot</namespace>
          <remapping>~/out:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### Creating a Combined URDF File

3. **Create a combined URDF file** (`~/ros2_ws/src/my_robot_pkg/urdf/complete_robot.urdf.xacro`):
```xml
<?xml version="1.0"?>
<robot name="complete_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include the robot model -->
  <xacro:include filename="my_advanced_robot.urdf.xacro"/>
  <xacro:include filename="my_advanced_robot.gazebo.xacro"/>

</robot>
```

## Lab 3: Integrating with ROS 2

### Creating Launch Files

1. **Create a launch file for simulation** (`~/ros2_ws/src/my_robot_pkg/launch/robot_with_sensors.launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world = LaunchConfiguration('world', default='simple_circuit.sdf')
    
    # Get the URDF file path
    urdf_file = os.path.join(
        get_package_share_directory('my_robot_pkg'),
        'urdf',
        'complete_robot.urdf.xacro'
    )

    # Define the robot_state_publisher node
    params = {'robot_description': Command(['xacro', ' ', urdf_file])}
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # Define the Gazebo server node
    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', 
             os.path.join(
                 get_package_share_directory('my_robot_pkg'),
                 'worlds',
                 world
             ),
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Define the Gazebo client node
    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=launch.conditions.IfCondition(
            launch.substitutions.LaunchConfiguration('gui')
        )
    )

    # Define the spawn entity node
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
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from `/my_robot_pkg/worlds`'
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Whether to launch the Gazebo client with a GUI'
        ),
        robot_state_publisher_node,
        gzserver_cmd,
        gzclient_cmd,
        spawn_entity_node,
    ])
```

### Building and Testing the Robot in Gazebo

2. **Build the package**:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source ~/ros2_ws/install/setup.bash
```

3. **Launch the simulation with your robot**:
```bash
# First terminal
ros2 launch my_robot_pkg robot_with_sensors.launch.py

# In another terminal, check the available topics
ros2 topic list

# You should see topics like /my_robot/scan, /my_robot/imu, /camera/image_raw, etc.
```

4. **Test robot movement**:
```bash
# Send a velocity command to make the robot move
ros2 topic pub /my_robot/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

## Lab 4: Unity Integration

### Setting up Unity for Robotics

Unity provides a powerful platform for high-fidelity simulation. Here's how to set it up:

1. **Install Unity Hub and Unity Editor**:
   - Download Unity Hub from the Unity website
   - Install Unity 2022.3 LTS or newer
   - Install the Universal Render Pipeline (URP) package

2. **Install Unity Robotics packages**:
   - Open Unity Hub and create a new 3D project
   - Open the Package Manager (Window > Package Manager)
   - Install the following packages:
     - Unity Robotics Hub
     - ROS-TCP-Connector
     - Unity-Robotics-Tools

3. **Basic Unity Scene Setup**:
   - Create a floor plane for the robot to move on
   - Add lighting to the scene
   - Create a basic robot model using primitives
   - Add cameras and other sensors

### Setting up ROS Communication in Unity

Unity communicates with ROS using TCP/IP. The ROS-TCP-Connector package provides the necessary infrastructure:

1. **Add ROS Connection Manager**:
   - In Unity, create an empty GameObject
   - Add the "ROS Connection" component to it
   - Configure the IP address and port (typically localhost:10000)

2. **Create a robot controller script** (C#):
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using ROS2;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string cmdVelTopicName = "/my_robot/cmd_vel";
    
    // Robot components
    public GameObject leftWheel;
    public GameObject rightWheel;
    public float wheelRadius = 0.1f;
    
    // Movement parameters
    float linearVelocity = 0f;
    float angularVelocity = 0f;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterSubscriber<Unity.Robotics.ROS2Polar.MessageTypes.Std_msgs.Float64MultiArray>(cmdVelTopicName);
    }
    
    void Update()
    {
        // Update wheel rotation based on velocities
        float leftWheelVel = (linearVelocity - angularVelocity * 0.15f) / wheelRadius; // 0.15f is wheel separation/2
        float rightWheelVel = (linearVelocity + angularVelocity * 0.15f) / wheelRadius;
        
        leftWheel.transform.Rotate(Vector3.right, leftWheelVel * Time.deltaTime * Mathf.Rad2Deg);
        rightWheel.transform.Rotate(Vector3.right, rightWheelVel * Time.deltaTime * Mathf.Rad2Deg);
        
        // Move the robot body
        transform.Translate(Vector3.forward * linearVelocity * Time.deltaTime);
        transform.Rotate(Vector3.up, angularVelocity * Time.deltaTime * Mathf.Rad2Deg);
    }
    
    public void OnMessageReceived(Unity.Robotics.ROS2Polar.MessageTypes.Std_msgs.Float64MultiArray cmdVel)
    {
        if (cmdVel.data.Length >= 6)
        {
            linearVelocity = (float)cmdVel.data[0];  // Linear x velocity
            angularVelocity = (float)cmdVel.data[5]; // Angular z velocity
        }
    }
}
```

## Lab 5: NVIDIA Isaac Sim Setup (Optional)

If you have an NVIDIA GPU with CUDA support, you can set up Isaac Sim:

1. **Install Isaac Sim**:
   ```bash
   # Install Isaac Sim from NVIDIA's Omniverse launcher
   # Follow the installation guide from NVIDIA Developer website
   ```

2. **Basic Isaac Sim Usage**:
   - Launch Isaac Sim from the Omniverse launcher
   - Import your robot URDF using the URDF Importer extension
   - Set up sensors and physics properties
   - Create simulation scenarios

## Lab 6: Validation and Comparison

### Comparing Simulation Fidelity

1. **Record sensor data from simulation**:
```bash
# Record topics to a bag file
ros2 bag record /my_robot/scan /my_robot/imu /tf /tf_static
```

2. **Analyze the data**:
```python
# Create a Python script to analyze the recorded data
# This would typically be done in a separate file
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray

class SensorAnalyzer(Node):
    def __init__(self):
        super().__init__('sensor_analyzer')
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/my_robot/scan',
            self.scan_callback,
            10)
        self.imu_subscription = self.create_subscription(
            Float64MultiArray,  # Placeholder for IMU data
            '/my_robot/imu',
            self.imu_callback,
            10)
        
    def scan_callback(self, msg):
        # Process LIDAR data
        self.get_logger().info(f"LIDAR data: {len(msg.ranges)} points, range: {msg.range_min} to {msg.range_max}")
        
    def imu_callback(self, msg):
        # Process IMU data
        self.get_logger().info(f"IMU data: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    analyzer = SensorAnalyzer()
    rclpy.spin(analyzer)
    analyzer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Optimization Techniques

1. **Adjust physics update rates** in your Gazebo models:
```xml
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <update_rate>100</update_rate>  <!-- Adjust based on requirements -->
  ...
</plugin>
```

2. **Use appropriate visual properties**:
```xml
<!-- For performance, consider reducing visual quality in simulation -->
<visual>
  <geometry>
    <mesh filename="package://my_robot_pkg/meshes/low_poly_model.dae"/>
  </geometry>
</visual>
```

## Lab 7: Advanced Simulation Scenarios

### Creating Complex Environments

1. **Build a multi-room world** with obstacles and navigation challenges:
   - Use modeling tools to create complex geometries
   - Add realistic textures and materials
   - Include interactive objects (that can be moved/pushed)

2. **Simulate dynamic environments**:
   - Moving obstacles
   - Changing lighting conditions
   - Weather effects (if supported by your simulation platform)

### Multi-Robot Simulation

1. **Spawn multiple robots** in the same environment:
```xml
<!-- In your world file, define multiple robot instances -->
<model name="robot_1">
  <pose>0 0 0.2 0 0 0</pose>
  <!-- Include robot definition -->
</model>

<model name="robot_2">
  <pose>2 2 0.2 0 0 0</pose>
  <!-- Include robot definition -->
</model>
```

### Hardware-in-the-Loop Setup

1. **Connect simulation to real hardware**:
   - Use ROS 2 bridges to connect simulation and real sensors
   - Implement time synchronization
   - Handle network latency and reliability issues

## Troubleshooting Common Issues

### Issue: Robot falls through the ground
**Solution**: Check that collision geometries are properly defined in your URDF and that mass/inertia values are reasonable.

### Issue: Sensor data appears noisy or unrealistic
**Solution**: Verify your sensor definitions in the Gazebo integration file and ensure appropriate noise models are set.

### Issue: Simulation runs too slowly
**Solution**: Reduce physics update rates, simplify collision meshes, or adjust rendering quality.

### Issue: Joint states not publishing
**Solution**: Make sure `joint_state_publisher` and `robot_state_publisher` are running and properly configured.

## Summary

This practical lab has guided you through setting up comprehensive digital twin environments using Gazebo and Unity. You've learned how to:

- Create complex robot models with multiple sensors
- Integrate these models with Gazebo simulation
- Set up communication between ROS 2 and simulation
- Implement advanced simulation scenarios
- Begin exploration of Unity integration for high-fidelity visualization

These skills form the foundation for creating effective digital twins that can accelerate robotics development while ensuring safe and effective transfer to real-world applications.