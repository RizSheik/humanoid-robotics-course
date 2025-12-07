---
id: module-1-simulation
title: Module 1 — The Robotic Nervous System | Chapter 5 — Simulation
sidebar_label: Chapter 5 — Simulation
sidebar_position: 5
---

# Module 1 — The Robotic Nervous System

## Chapter 5 — Simulation

### Introduction to Robotics Simulation

Simulation is a critical component in the development of robotic systems, providing a safe, cost-effective environment for testing algorithms, validating designs, and training robotic systems before deployment on physical hardware. In the context of humanoid robotics, simulation is especially important due to the complexity and potential safety concerns associated with physical humanoid robots.

### Simulation in the Robotic Nervous System Context

For Module 1, which focuses on the robotic nervous system (ROS 2), simulation serves several key roles:

1. **Communication Testing**: Validating ROS 2 communication patterns without physical hardware
2. **Integration Testing**: Ensuring different ROS 2 nodes work together properly
3. **Algorithm Development**: Testing navigation, perception, and control algorithms
4. **Safety Validation**: Ensuring robot behaviors are safe before deployment

### Gazebo Simulation Environment

Gazebo is the most commonly used physics simulator in the ROS ecosystem. It provides realistic simulation of rigid body dynamics, sensors, and environmental effects.

#### Key Features of Gazebo

1. **Physics Engine**: Supports ODE, Bullet, Simbody, and DART physics engines
2. **Sensor Simulation**: Cameras, LiDAR, IMU, GPS, and other sensors
3. **Realistic Rendering**: High-quality graphics with OGRE rendering engine
4. **Plugin System**: Extensible architecture for custom models and behaviors

#### Gazebo-ROS Integration

The `gazebo_ros` package provides seamless integration between Gazebo and ROS 2:

- **Gazebo Plugins**: ROS interfaces to Gazebo models and sensors
- **URDF Integration**: Direct loading of URDF robot models into Gazebo
- **ROS Services**: Start/stop/reset simulation, spawn/remove models

### Setting Up a Gazebo Simulation Environment

#### Installation and Setup

```bash
# Install Gazebo Garden (or newer version)
sudo apt update
sudo apt install ros-humble-gazebo-*

# Install additional ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-ros-pkgs
```

#### Creating a Simulation Package

Let's create a simulation package for our humanoid robot:

1. **Create the package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_cmake robot_nervous_system_simulation
   ```

2. **Directory Structure**:
   ```
   robot_nervous_system_simulation/
   ├── CMakeLists.txt
   ├── package.xml
   ├── launch/
   │   ├── robot_world.launch.py
   │   └── simulation.launch.py
   ├── worlds/
   │   └── simple_room.world
   ├── models/
   │   └── simple_humanoid/  # Link to URDF from previous chapter
   └── config/
       └── gazebo_params.yaml
   ```

3. **World File** - Create `worlds/simple_room.world`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.6">
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

       <!-- Simple room with walls -->
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

       <!-- Object for robot to interact with -->
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

4. **Launch File** - Create `launch/robot_world.launch.py`:
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
       pkg_simulation = get_package_share_directory('robot_nervous_system_simulation')

       # World file
       world = os.path.join(pkg_simulation, 'worlds', 'simple_room.world')

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

       return LaunchDescription([
           gzserver,
           gzclient,
       ])
   ```

5. **URDF Integration** - Create a URDF with Gazebo plugins:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_humanoid_gazebo" xmlns:xacro="http://www.ros.org/wiki/xacro">
     <!-- Include the original URDF -->
     <xacro:include filename="$(find robot_nervous_system_tutorial)/urdf/simple_humanoid.urdf" />

     <!-- Gazebo-specific configurations -->
     <gazebo>
       <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
         <parameters>$(find robot_nervous_system_simulation)/config/hardware_control.yaml</parameters>
       </plugin>
     </gazebo>

     <!-- Link properties for physics simulation -->
     <gazebo reference="base_link">
       <material>Gazebo/Blue</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
       <kp>1000000.0</kp>
       <kd>1.0</kd>
     </gazebo>

     <gazebo reference="head">
       <material>Gazebo/White</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="left_upper_arm">
       <material>Gazebo/Red</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="left_lower_arm">
       <material>Gazebo/Red</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="right_upper_arm">
       <material>Gazebo/Red</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="right_lower_arm">
       <material>Gazebo/Red</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="left_upper_leg">
       <material>Gazebo/Green</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="left_lower_leg">
       <material>Gazebo/Green</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="right_upper_leg">
       <material>Gazebo/Green</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>

     <gazebo reference="right_lower_leg">
       <material>Gazebo/Green</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
     </gazebo>
   </robot>
   ```

### Robot Simulation Control

#### Joint Control in Simulation

In Gazebo, we can control robot joints using the `gazebo_ros_control` plugin:

1. **Controller Configuration** - Create `config/hardware_control.yaml`:
   ```yaml
   simple_humanoid:
     # Position Controllers
     left_shoulder_position_controller:
       type: position_controllers/JointPositionController
       joint: left_shoulder
       pid: {p: 100.0, i: 0.01, d: 10.0}

     left_elbow_position_controller:
       type: position_controllers/JointPositionController
       joint: left_elbow
       pid: {p: 50.0, i: 0.01, d: 5.0}

     right_shoulder_position_controller:
       type: position_controllers/JointPositionController
       joint: right_shoulder
       pid: {p: 100.0, i: 0.01, d: 10.0}

     right_elbow_position_controller:
       type: position_controllers/JointPositionController
       joint: right_elbow
       pid: {p: 50.0, i: 0.01, d: 5.0}

     left_hip_position_controller:
       type: position_controllers/JointPositionController
       joint: left_hip
       pid: {p: 200.0, i: 0.1, d: 20.0}

     left_knee_position_controller:
       type: position_controllers/JointPositionController
       joint: left_knee
       pid: {p: 100.0, i: 0.05, d: 10.0}

     right_hip_position_controller:
       type: position_controllers/JointPositionController
       joint: right_hip
       pid: {p: 200.0, i: 0.1, d: 20.0}

     right_knee_position_controller:
       type: position_controllers/JointPositionController
       joint: right_knee
       pid: {p: 100.0, i: 0.05, d: 10.0}
   ```

2. **Controller Spawner Node** - Create `robot_nervous_system_simulation/robot_nervous_system_simulation/controller_spawner.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from controller_manager_msgs.srv import SwitchController
   import time

   class ControllerSpawner(Node):
       def __init__(self):
           super().__init__('controller_spawner')
           self.cli = self.create_client(
               SwitchController,
               '/controller_manager/switch_controller'
           )
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')

           self.spawn_controllers()

       def spawn_controllers(self):
           req = SwitchController.Request()
           req.start_controllers = [
               'left_shoulder_position_controller',
               'left_elbow_position_controller',
               'right_shoulder_position_controller',
               'right_elbow_position_controller',
               'left_hip_position_controller',
               'left_knee_position_controller',
               'right_hip_position_controller',
               'right_knee_position_controller'
           ]
           req.stop_controllers = []
           req.strictness = SwitchController.Request.BEST_EFFORT

           future = self.cli.call_async(req)
           rclpy.spin_until_future_complete(self, future)

           if future.result() is not None:
               response = future.result()
               self.get_logger().info(f'Controller spawn result: {response.ok}')
           else:
               self.get_logger().error('Failed to spawn controllers')

   def main(args=None):
       rclpy.init(args=args)
       controller_spawner = ControllerSpawner()
       rclpy.spin(controller_spawner)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

### Simulation with ROS 2 Integration

#### Connecting Simulation to ROS 2

The key to effective simulation is ensuring that the simulated robot behaves as closely as possible to the real robot:

1. **Sensor Simulation**:
   - Simulated sensors publish the same message types as real sensors
   - Sensor noise and latency can be modeled to match real-world conditions
   - TF frames published by simulation match those on the physical robot

2. **Actuator Simulation**:
   - Joint position/velocity/effort controllers respond to the same ROS 2 messages
   - Dynamics modeling approximates real-world physics
   - Controller parameters can be tuned to match physical robot behavior

3. **Communication Layer**:
   - ROS 2 nodes can work with both simulated and real robots without code changes
   - QoS settings can be adjusted for optimal simulation performance

#### Example: Simulated Robot Controller

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class SimulatedRobotController(Node):
    def __init__(self):
        super().__init__('simulated_robot_controller')

        # Publisher for joint trajectory commands
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )

        # Subscriber for joint states from Gazebo
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer to send periodic joint commands
        self.timer = self.create_timer(0.1, self.send_trajectory_command)

        self.joint_names = [
            'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow',
            'left_hip', 'left_knee', 'right_hip', 'right_knee'
        ]

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')
        # Process joint state information here

    def send_trajectory_command(self):
        # Create a simple trajectory to move the robot's arms
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [0.1, 0.2, -0.1, 0.2, 0.0, 0.0, 0.0, 0.0]  # Example positions
        point.velocities = [0.0] * len(point.positions)  # Zero velocities
        point.time_from_start = Duration(sec=1)  # 1 second to reach position

        traj_msg.points = [point]
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'base_link'

        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(f'Published trajectory command with {len(point.positions)} joints')

def main(args=None):
    rclpy.init(args=args)
    controller = SimulatedRobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Simulation Concepts

#### Physics Tuning

For humanoid robotics, accurate physics simulation is crucial:

1. **Mass and Inertia**: Properly calculated mass properties for each link
2. **Friction Parameters**: Realistic friction coefficients between surfaces
3. **Damping**: Proper damping coefficients to match real-world behavior
4. **Contact Properties**: Realistic contact stiffness and dissipation

#### Sensor Simulation

Accurate sensor simulation is essential for developing perception algorithms:

1. **Camera Simulation**:
   - Field of view, resolution, and distortion matching real cameras
   - Noise modeling to simulate real-world imperfections
   - Image delay to simulate processing time

2. **IMU Simulation**:
   - Gyroscope and accelerometer noise modeling
   - Bias and drift simulation
   - Sampling rate matching real hardware

3. **LiDAR Simulation**:
   - Range, resolution, and field of view matching real LiDAR
   - Noise and dropout modeling
   - Multiple return simulation

#### Multi-Robot Simulation

Simulating multiple robots requires additional considerations:

1. **Network Simulation**: Simulating network conditions between robots
2. **Communication Latency**: Adding realistic communication delays
3. **Resource Management**: Managing computational resources for multiple robots
4. **Collision Avoidance**: Ensuring robots don't interfere with each other

### Integration with the Robotic Nervous System

Simulation must integrate seamlessly with the ROS 2 communication framework:

1. **Message Compatibility**: Simulated sensors publish exactly the same message types as real sensors
2. **TF Consistency**: Transform frames in simulation match those on the real robot
3. **Controller Interface**: Joint controllers accept the same commands in both environments
4. **Parameter Consistency**: Robot parameters are the same in simulation and reality

### Best Practices for Simulation

#### Model Validation

1. **Compare with Real Robot**: Validate that simulation behavior matches real-world performance
2. **Parameter Tuning**: Adjust physics and controller parameters to match real robot
3. **Sensory Validation**: Ensure simulated sensors produce data similar to real sensors

#### Performance Optimization

1. **Update Rates**: Balance simulation accuracy with computational performance
2. **Level of Detail**: Reduce complexity for parts of the model not critical to the simulation
3. **Parallel Processing**: Use multi-threaded simulation where appropriate

#### Testing Strategies

1. **Simulation-to-Reality Transfer**: Test algorithms from simulation on real robots
2. **Edge Case Testing**: Use simulation to test dangerous or hard-to-reach real-world scenarios
3. **Fuzz Testing**: Randomize parameters to test algorithm robustness

### Simulation Tools and Frameworks

#### Alternatives to Gazebo

1. **Webots**: General-purpose robot simulation with good humanoid support
2. **PyBullet**: Physics engine with Python API, good for research
3. **MuJoCo**: Advanced physics simulation (commercial)
4. **Unity Robotics Hub**: Game engine-based simulation with realistic graphics

#### Visualization and Debugging

1. **RViz Integration**: Combine Gazebo visualization with ROS 2 data
2. **Gazebo GUI**: Real-time visualization of simulation state
3. **Logging**: Comprehensive logging of simulation data for analysis
4. **Debug Tools**: Visualization of forces, contacts, and other physics properties

### Hardware-in-the-Loop (HIL) Simulation

In some cases, it's beneficial to run some components on real hardware while others are simulated:

1. **Real Controllers**: Use actual robot controllers with simulated environment
2. **Physical Sensors**: Connect real sensors to the simulated robot
3. **Network Simulation**: Simulate network conditions for distributed systems

### Conclusion

Simulation is a vital component of the robotic nervous system development process. It allows for safe testing of communication patterns, validation of control algorithms, and integration testing without the risks and costs associated with physical hardware.

By properly configuring simulation environments, we can create virtual laboratories that closely match real-world conditions, enabling the development of robust, reliable humanoid robotics systems. The integration between simulation and ROS 2 ensures that algorithms developed in simulation can transition to real hardware with minimal code changes.

As we move forward in this course, simulation will continue to be an important tool for testing and validating concepts related to the digital twin, AI-robot brain, and Vision-Language-Action systems.