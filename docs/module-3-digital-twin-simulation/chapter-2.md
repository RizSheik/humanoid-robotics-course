# Chapter 2: Gazebo Harmonic - Physics-Based Simulation


<div className="robotDiagram">
  <img src="../../../img/book-image/Illustration_explaining_Physical_AI_huma_1.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>

## Learning Objectives

After completing this chapter, students will be able to:
- Configure and operate Gazebo Harmonic simulation environment
- Create accurate robot models using URDF and SDF formats
- Implement realistic physics simulation with appropriate parameters
- Develop custom sensors and plugins for specialized applications
- Optimize simulation performance for real-time operation
- Evaluate and validate simulation models against real-world behavior

## 2.1 Introduction to Gazebo Harmonic

Gazebo Harmonic is a state-of-the-art physics-based simulation environment designed for robotics research and development. As part of the Open Robotics ecosystem, it provides a complete solution for simulating robots in 3D environments with realistic physics, sensors, and interaction capabilities.

### 2.1.1 Key Features of Gazebo Harmonic

**Physics Engine**: Based on Ignition Physics, offering multiple backend physics engines (DART, Bullet) with accurate collision detection, contact simulation, and dynamics modeling.

**Sensor Simulation**: Comprehensive support for various sensor types including cameras, LIDAR, IMU, force/torque sensors, GPS, and more, with realistic noise models and parameters.

**Modular Architecture**: Component-based design allowing customization of physics, rendering, and other simulation aspects using plugins.

**ROS 2 Integration**: Native support for ROS 2 through Ignition Transport (which bridges to ROS 2 topics), making it straightforward to connect simulation to ROS 2-based robot code.

**Scalability**: Support for simulating multiple robots simultaneously and distributed simulation across multiple machines.

### 2.1.2 Architecture Overview

Gazebo Harmonic has a modular architecture consisting of:

1. **Gazebo Server**: Core simulation engine that handles physics, sensors, and plugin execution
2. **Gazebo Client**: User interface for visualization and interaction
3. **Fuel Server**: Online model database for sharing robot models and environments
4. **Transport Layer**: Message passing system based on Ignition Transport, bridging to ROS 2

## 2.2 Installation and Configuration

### 2.2.1 System Requirements

To run Gazebo Harmonic effectively:
- **CPU**: Multi-core processor (Intel i5 or equivalent recommended)
- **Memory**: 8GB+ RAM (16GB+ for complex simulations)
- **GPU**: OpenGL 3.3+ capable graphics card with dedicated VRAM
- **OS**: Ubuntu 22.04 (recommended) or other supported Linux distributions

### 2.2.2 Installation Process

Gazebo Harmonic is typically installed as part of a ROS 2 installation or separately:

```bash
# Install via package manager
sudo apt-get update
sudo apt-get install ros-humble-gazebo-*

# Or install as part of ROS 2 desktop installation
sudo apt-get install ros-humble-desktop
```

### 2.2.3 Basic Configuration

Gazebo Harmonic can be configured through environment variables and launch files:

```bash
# Set Gazebo resource paths
export GZ_SIM_RESOURCE_PATH=/path/to/models:/path/to/worlds
export GZ_SIM_SYSTEM_PLUGIN_PATH=/path/to/plugins

# Set specific physics engine (optional)
export GZ_PHYSICS_ENGINE_NAME=libignition-physics-dartsim-plugin.so
```

## 2.3 Robot Model Creation with URDF/SDF

### 2.3.1 Understanding URDF and SDF

**URDF (Unified Robot Description Format)** is an XML-based format primarily used for robot kinematics and basic dynamics. It's well-integrated with ROS 2 tools and is typically used for robot description.

```xml
<!-- Example URDF snippet -->
<robot name="simple_robot">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0.25 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>
</robot>
```

**SDF (Simulation Description Format)** is the native Gazebo format that extends URDF capabilities with simulation-specific features like joints, transmission systems, sensors, and plugins:

```xml
<!-- Example SDF snippet -->
<sdf version="1.7">
  <model name="simple_robot">
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      
      <visual name="visual">
        <geometry>
          <box><size>0.5 0.5 0.5</size></box>
        </geometry>
      </visual>
      
      <collision name="collision">
        <geometry>
          <box><size>0.5 0.5 0.5</size></box>
        </geometry>
      </collision>
      
      <sensor name="imu_sensor" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>2e-4</stddev>
              </noise>
            </x>
          </angular_velocity>
        </imu>
      </sensor>
    </link>
  </model>
</sdf>
```

### 2.3.2 Converting URDF to SDF

Gazebo typically uses SDF for simulation, but URDF can be converted automatically:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or include URDF in SDF world file
<sdf version="1.7">
  <world name="default">
    <include>
      <uri>model://robot_model</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## 2.4 Physics Configuration and Parameters

### 2.4.1 Physics Engine Selection

Gazebo Harmonic supports multiple physics engines:

- **DART (Dynamic Animation and Robotics Toolkit)**: Recommended for most robotics applications, especially for complex contact handling
- **Bullet**: Good for faster simulation, suitable for simpler contact scenarios
- **ODE**: Older but stable, good for basic rigid body simulation

### 2.4.2 Physics Parameters

The physics engine can be configured with parameters to balance accuracy and performance:

```xml
<!-- Physics configuration in world file -->
<physics type="ignored">
  <max_step_size>0.001</max_step_size>      <!-- Simulation time step -->
  <real_time_factor>1.0</real_time_factor>  <!-- Real-time simulation speed -->
  <real_time_update_rate>1000</real_time_update_rate> <!-- Updates per second -->
  
  <!-- Solver settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>        <!-- Iterations for constraint solving -->
      <sor>1.3</sor>           <!-- Successive over-relaxation parameter -->
    </solver>
    <constraints>
      <cfm>0.0</cfm>           <!-- Constraint force mixing -->
      <erp>0.2</erp>           <!-- Error reduction parameter -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### 2.4.3 Material and Surface Properties

Surface properties affect how objects interact:

```xml
<!-- Surface properties -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>            <!-- Static friction coefficient -->
      <mu2>1.0</mu2>          <!-- Secondary friction coefficient -->
      <slip1>0.0</slip1>      <!-- Primary slip coefficient -->
      <slip2>0.0</slip2>      <!-- Secondary slip coefficient -->
    </ode>
  </friction>
  <bounce>
    <restitution_coefficient>0.0</restitution_coefficient>
    <threshold>100000</threshold>
  </bounce>
  <contact>
    <ode>
      <soft_cfm>0.0</soft_cfm>
      <soft_erp>0.2</soft_erp>
      <kp>1000000000000.0</kp>  <!-- Contact stiffness -->
      <kd>1.0</kd>              <!-- Damping coefficient -->
      <max_vel>100.0</max_vel>
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

## 2.5 Sensor Integration and Simulation

### 2.5.1 Camera Simulation

Gazebo provides realistic camera simulation with various parameters:

```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>  <!-- Field of view in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### 2.5.2 LIDAR Simulation

LIDAR sensors can be configured with detailed parameters:

```xml
<sensor name="laser" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin filename="libignition-gazebo-ray-sensor-system.so" name="ignition::gazebo::systems::RaySensor">
    <ros>
      <namespace>/robot1</namespace>
      <remapping>~/out@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan</remapping>
    </ros>
  </plugin>
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### 2.5.3 IMU and Force/Torque Sensors

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## 2.6 Custom Plugins Development

### 2.6.1 Model Plugins

Model plugins extend robot behavior:

```cpp
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/math/Pose3.hh>
#include <sdf/sdf.hh>

namespace gazebo_class
{
  class CustomController : public ignition::gazebo::System,
                          public ignition::gazebo::ISystemConfigure,
                          public ignition::gazebo::ISystemPreUpdate
  {
    public: void Configure(const ignition::gazebo::Entity &_entity,
                          const std::shared_ptr<const sdf::Element> &_sdf,
                          ignition::gazebo::EntityComponentManager &_ecm,
                          ignition::gazebo::EventManager &_eventMgr) override
    {
      // Initialize the controller
      this->model = ignition::gazebo::Model(_entity);
    }

    public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                          ignition::gazebo::EntityComponentManager &_ecm) override
    {
      // Custom control logic executed before each simulation step
      // This is where control algorithms are typically implemented
    }

    private: ignition::gazebo::Model model;
  };
}

// Register the plugin
IGNITION_ADD_PLUGIN(gazebo_class::CustomController,
                  ignition::gazebo::System,
                  CustomController::ISystemConfigure,
                  CustomController::ISystemPreUpdate)
```

### 2.6.2 Sensor Plugins

Sensor plugins process sensor data:

```cpp
#include <ignition/gazebo/System.hh>
#include <ignition/sensors/CameraSensor.hh>
#include <ignition/transport/Node.hh>

namespace gazebo_class
{
  class CustomCameraProcessor : public ignition::gazebo::System,
                                public ignition::gazebo::ISystemPreUpdate
  {
    public: void PreUpdate(const ignition::gazebo::UpdateInfo &_info,
                          ignition::gazebo::EntityComponentManager &_ecm) override
    {
      // Process camera data
      // Apply custom image processing algorithms
    }
  };
}
```

## 2.7 Performance Optimization

### 2.7.1 Simulation Parameters for Performance

Balance accuracy and performance with these settings:

```xml
<!-- Performance-optimized physics settings -->
<physics type="dartsim">
  <max_step_size>0.01</max_step_size>        <!-- Larger steps for performance -->
  <real_time_factor>1.0</real_time_factor>   <!-- Target real-time simulation -->
  <real_time_update_rate>100</real_time_update_rate>  <!-- Lower update rate -->
  
  <dartsim>
    <solver>
      <type>PGS</type>                        <!-- Fast solver type -->
      <iterations>50</iterations>              <!-- Fewer iterations -->
      <sor>1.2</sor>                          <!-- Lower SOR parameter -->
    </solver>
  </dartsim>
</physics>
```

### 2.7.2 Visualization Settings

Optimize rendering for performance:

```bash
# Disable rendering for headless operation
gazebo --headless my_world.sdf

# Or through environment variable
export GZ_GUI_PLUGIN_CAMERA_TRACKER_ENABLED=false
```

### 2.7.3 Model Simplification

For performance-critical applications, simplify models:

```xml
<!-- Simplified collision geometry -->
<link name="simplified_link">
  <!-- Use simple shapes instead of complex meshes -->
  <collision>
    <geometry>
      <cylinder>
        <radius>0.1</radius>
        <length>0.5</length>
      </cylinder>
    </geometry>
  </collision>
  
  <!-- Detailed visual geometry -->
  <visual>
    <geometry>
      <mesh>
        <uri>model://complex_robot/meshes/detailed_link.dae</uri>
      </mesh>
    </geometry>
  </visual>
</link>
```

## 2.8 ROS 2 Integration

### 2.8.1 Ignition Transport to ROS 2 Bridge

Gazebo Harmonic uses Ignition Transport, which bridges to ROS 2:

```xml
<!-- Example of ROS 2 bridge configuration in SDF -->
<model name="robot_with_ros">
  <plugin filename="libignition-gazebo-joint-position-controller-system.so" 
          name="ignition::gazebo::systems::JointPositionController">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>cmd_pos@std_msgs/msg/Float64@ignition.msgs.Double</remapping>
    </ros>
  </plugin>
  
  <plugin filename="libignition-gazebo-joint-state-publisher-system.so"
          name="ignition::gazebo::systems::JointStatePublisher">
    <ros>
      <namespace>/my_robot</namespace>
    </ros>
  </plugin>
</model>
```

### 2.8.2 Launch Files for Gazebo Integration

Creating launch files to integrate Gazebo with ROS 2 systems:

```python
# launch/gazebo_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={'gz_args': ' -r empty.sdf'}.items()
    )
    
    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-file', PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf'
            ])
        ],
        output='screen'
    )
    
    return LaunchDescription([
        gazebo_launch,
        spawn_entity,
    ])
```

## 2.9 Validation and Calibration

### 2.9.1 Model Validation Techniques

Validating simulation models against real robot behavior:

1. **Kinematic Validation**: Verify joint limits and ranges of motion match real robot
2. **Dynamic Validation**: Compare motion with real robot under same control inputs
3. **Sensor Validation**: Ensure simulated sensors match real sensor characteristics

### 2.9.2 Parameter Calibration

Calibrating simulation parameters to match reality:

```python
# Example of system identification to calibrate simulation parameters
import numpy as np
from scipy.optimize import minimize

def simulation_error(params):
    """Calculate error between simulation and real robot"""
    # Update simulation with new parameters
    update_simulation_params(params)
    
    # Run simulation and collect data
    sim_data = run_simulation()
    
    # Compare with real robot data
    real_data = get_real_robot_data()
    
    # Calculate error
    error = np.mean((sim_data - real_data) ** 2)
    return error

# Optimize parameters to minimize simulation error
result = minimize(simulation_error, initial_guess, method='BFGS')
calibrated_params = result.x
```

### 2.9.3 Sim-to-Real Transfer Validation

Testing sim-to-real transfer effectiveness:

1. Train controller in simulation
2. Deploy directly to real robot without modification
3. Measure performance degradation
4. Iterate simulation model improvements

## 2.10 Advanced Topics

### 2.10.1 Multi-Robot Simulation

Simulating multiple robots with proper communication:

```xml
<!-- Multi-robot world file -->
<sdf version="1.7">
  <world name="multi_robot">
    <include>
      <name>robot1</name>
      <uri>model://turtlebot3_waffle</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
    
    <include>
      <name>robot2</name>
      <uri>model://turtlebot3_waffle</uri>
      <pose>1 0 0 0 0 0</pose>
    </include>
    
    <physics type="dartsim">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
  </world>
</sdf>
```

### 2.10.2 Domain Randomization

Implementing domain randomization for robust sim-to-real transfer:

```xml
<!-- Example of domain randomization in Gazebo -->
<model name="randomized_box">
  <link name="link">
    <inertial>
      <mass>__random_mass__</mass>
      <inertia>
        <ixx>__random_ixx__</ixx>
        <iyy>__random_iyy__</iyy>
        <izz>__random_izz__</izz>
        <!-- Other inertia values -->
      </inertia>
    </inertial>
    
    <collision name="collision">
      <geometry>
        <box>
          <size>__random_size_x__ __random_size_y__ __random_size_z__</size>
        </box>
      </geometry>
    </collision>
  </link>
</model>
```

## Chapter Summary

This chapter provided a comprehensive overview of Gazebo Harmonic as a physics-based simulation environment for robotics. We covered installation and configuration, robot model creation using URDF and SDF, physics configuration, sensor integration, custom plugin development, performance optimization, and ROS 2 integration. The chapter concluded with validation techniques and advanced topics including multi-robot simulation and domain randomization.

## Key Terms
- Gazebo Harmonic
- URDF (Unified Robot Description Format)
- SDF (Simulation Description Format)
- Physics Engine
- Sensor Simulation
- Model Plugins
- ROS 2 Integration
- Sim-to-Real Transfer
- Domain Randomization

## Exercises
1. Create a simple robot model in URDF and simulate it in Gazebo
2. Implement a custom controller plugin for a differential drive robot
3. Configure a multi-sensor robot with camera, LIDAR, and IMU
4. Validate a simulation model against real robot data and calibrate parameters

## References
- Gazebo Harmonic Documentation: http://gazebosim.org/
- Ignition Robotics Documentation: https://ignitionrobotics.org/
- Open Source Robotics Foundation. (2023). Gazebo Harmonic Release Notes.