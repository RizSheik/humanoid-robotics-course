# Module 3: Simulation - Digital Twin Environments for Robotics

## Simulation Overview

This simulation module focuses on implementing and testing digital twin simulation environments for robotics using three major platforms: Gazebo, Unity, and Isaac Sim. Students will gain practical experience with creating realistic simulation environments, implementing advanced simulation techniques, and validating simulation models against real-world behavior.

### Learning Objectives

After completing this simulation module, students will be able to:
1. Implement and configure simulation environments in Gazebo, Unity, and Isaac Sim
2. Create realistic robot models with accurate physics and sensor simulation
3. Apply advanced simulation techniques like domain randomization and system identification
4. Validate simulation models and evaluate sim-to-real transfer potential
5. Optimize simulation performance and realism for specific applications
6. Design simulation experiments that accelerate robotics development

### Required Simulation Tools

- **Gazebo Harmonic** or later with ROS 2 integration
- **Unity Hub** with Unity 2021.3 LTS and robotics packages
- **NVIDIA Isaac Sim** with Omniverse platform
- **ROS 2 Humble Hawksbill** for robotics communication
- **Python 3.11+** and **C++17** for custom implementations
- **Docker** for Isaac Sim container deployment
- **RViz2** and **rqt** for visualization and debugging

## Gazebo Simulation Environment

### Gazebo Configuration and Setup

For Gazebo-based simulations, proper configuration is essential for achieving realistic physics and sensor simulation:

```xml
<!-- Example Gazebo world configuration with realistic physics -->
<sdf version="1.7">
  <world name="realistic_world">
    <!-- Physics engine configuration -->
    <physics type="dartsim">
      <max_step_size>0.001</max_step_size>        <!-- Time step: 1ms (1000Hz) -->
      <real_time_factor>1.0</real_time_factor>    <!-- Real-time simulation -->
      <real_time_update_rate>1000</real_time_update_rate>
      
      <dart>
        <solver>
          <type>PGS</type>
          <iterations>100</iterations>
          <collision_filter_mode>all</collision_filter_mode>
        </solver>
        <constraints>
          <contact_surface_layer>0.001</contact_surface_layer>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        </constraints>
        <collision_detector>bullet</collision_detector>
        <impact_capture_model>
          <type>None</type>
        </impact_capture_model>
        <soft_body_solver>
          <velocity_damping>0.0</velocity_damping>
          <position_damping>1.0</position_damping>
          <drift_correction>1.0</drift_correction>
          <cluster_stiffness>1e+06</cluster_stiffness>
        </soft_body_solver>
      </dart>
    </physics>

    <!-- Include a realistic ground plane -->
    <include>
      <uri>model://ground_plane</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- Add lighting for realistic rendering -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add ambient light -->
    <light type="directional" name="ambient_light">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>100</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>
    
    <!-- Include robot model -->
    <include>
      <name>mobile_robot</name>
      <uri>model://differential_drive_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <!-- Add objects for interaction -->
    <model name="table">
      <pose>2 2 0 0 0 0</pose>
      <link name="table_link">
        <pose>0 0 0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box><size>1.0 0.6 0.8</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.0 0.6 0.8</size></box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Robot Model Configuration in Gazebo

Creating realistic robot models for simulation:

```xml
<!-- Advanced robot model with realistic sensors and actuators -->
<sdf version="1.7">
  <model name="advanced_robot">
    <!-- Base chassis -->
    <link name="base_link">
      <inertial>
        <mass>20.0</mass>
        <inertia>
          <ixx>1.0</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
          <iyy>1.5</iyy><iyz>0.0</iyz><izz>2.0</izz>
        </inertia>
      </inertial>
      
      <collision name="collision">
        <geometry><box><size>0.8 0.5 0.3</size></box></geometry>
      </collision>
      
      <visual name="visual">
        <geometry><mesh><uri>model://robot/meshes/base.dae</uri></mesh></geometry>
        <material>
          <ambient>0.1 0.1 0.8 1</ambient>
          <diffuse>0.1 0.1 0.8 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Add realistic sensors -->
    <sensor name="camera_front" type="camera">
      <pose>0.3 0 0.2 0 0 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>30.0</far>
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

    <!-- IMU sensor -->
    <sensor name="imu_sensor" type="imu">
      <pose>0 0 0.1 0 0 0</pose>
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

    <!-- LIDAR sensor -->
    <sensor name="3d_lidar" type="ray">
      <pose>0.2 0 0.4 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
          <vertical>
            <samples>16</samples>
            <resolution>1</resolution>
            <min_angle>-0.2618</min_angle> <!-- -15 degrees -->
            <max_angle>0.2618</max_angle>   <!-- 15 degrees -->
          </vertical>
        </scan>
        <range>
          <min>0.1</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <always_on>true</always_on>
      <update_rate>10</update_rate>
      <visualize>true</visualize>
    </sensor>

    <!-- ROS 2 integration -->
    <plugin filename="libignition-gazebo-diff-drive-system.so" name="ignition::gazebo::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_radius>0.15</wheel_radius>
      <odom_publish_frequency>30</odom_publish_frequency>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <tf_topic>tf</tf_topic>
    </plugin>

    <plugin filename="libignition-gazebo-joint-state-publisher-system.so" name="ignition::gazebo::systems::JointStatePublisher">
      <ros>
        <namespace>/robot1</namespace>
        <remapping>joint_states:=/robot1/joint_states</remapping>
      </ros>
    </plugin>
  </model>
</sdf>
```

### Gazebo Simulation Implementation

Implementing simulation with advanced physics and control:

```python
#!/usr/bin/env python3
# Advanced Gazebo simulation implementation for robotics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, Imu, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class GazeboRobotSimulator(Node):
    def __init__(self):
        super().__init__('gazebo_robot_simulator')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Robot state variables
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Robot physical parameters
        self.wheel_separation = 0.4
        self.wheel_radius = 0.15
        self.robot_radius = 0.25
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.laser_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.camera_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create timer for simulation update
        self.timer = self.create_timer(0.01, self.update_simulation)  # 100Hz
        
        # Initialize random number generator for sensor noise
        self.rng = np.random.default_rng(seed=42)
        
        self.get_logger().info('Gazebo Robot Simulator initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z

    def update_simulation(self):
        """Update robot state based on physics model"""
        dt = 0.01  # Time step based on timer frequency
        
        # Update robot pose using differential drive kinematics
        dx = self.linear_velocity * math.cos(self.theta) * dt
        dy = self.linear_velocity * math.sin(self.theta) * dt
        dtheta = self.angular_velocity * dt
        
        self.x += dx
        self.y += dy
        self.theta += dtheta
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Publish odometry
        self.publish_odometry()
        
        # Publish laser scan with simulated environment
        self.publish_laser_scan()
        
        # Publish IMU data with simulated noise
        self.publish_imu_data()
        
        # Publish camera image
        self.publish_camera_image()
        
        # Publish transform
        self.publish_transform()

    def publish_odometry(self):
        """Publish odometry data"""
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # Orientation (convert from angle to quaternion)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('z', self.theta).as_quat()
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        
        # Velocity
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity
        
        self.odom_pub.publish(odom)

    def publish_laser_scan(self):
        """Publish simulated laser scan"""
        scan = LaserScan()
        scan.header = Header()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'laser_link'
        
        # Laser parameters
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2 * math.pi / 720  # 720 rays
        scan.time_increment = 0.0
        scan.scan_time = 0.1  # 10Hz
        scan.range_min = 0.1
        scan.range_max = 30.0
        
        # Generate simulated ranges based on environment
        ranges = []
        for i in range(720):
            angle = scan.angle_min + i * scan.angle_increment
            
            # Simulate a simple environment with walls and objects
            # In a real implementation, this would be based on the scene geometry
            distance = self.simulate_laser_ray(angle)
            
            # Add realistic noise
            noise = self.rng.normal(0, 0.02)  # 2cm noise
            distance_with_noise = max(scan.range_min, min(scan.range_max, distance + noise))
            
            ranges.append(distance_with_noise)
        
        scan.ranges = ranges
        scan.intensities = []  # No intensity information
        
        self.laser_pub.publish(scan)

    def simulate_laser_ray(self, angle):
        """Simulate a laser ray in the environment"""
        # For this simulation, create a simple environment
        # with walls at x = -5, x = 5, y = -5, y = 5
        robot_x = self.x
        robot_y = self.y
        robot_angle = self.theta
        
        # Global angle of the laser beam
        global_angle = robot_angle + angle
        
        # Calculate intersection with walls
        # Wall at x = -5
        t_x_neg = float('inf')
        if math.cos(global_angle) != 0:
            t_x_neg = (-5 - robot_x) / math.cos(global_angle)
        if t_x_neg > 0:
            y_at_x_neg = robot_y + t_x_neg * math.sin(global_angle)
            if -5 <= y_at_x_neg <= 5:
                return t_x_neg
        
        # Wall at x = 5
        t_x_pos = float('inf')
        if math.cos(global_angle) != 0:
            t_x_pos = (5 - robot_x) / math.cos(global_angle)
        if t_x_pos > 0:
            y_at_x_pos = robot_y + t_x_pos * math.sin(global_angle)
            if -5 <= y_at_x_pos <= 5:
                return t_x_pos
        
        # Wall at y = -5
        t_y_neg = float('inf')
        if math.sin(global_angle) != 0:
            t_y_neg = (-5 - robot_y) / math.sin(global_angle)
        if t_y_neg > 0:
            x_at_y_neg = robot_x + t_y_neg * math.cos(global_angle)
            if -5 <= x_at_y_neg <= 5:
                return t_y_neg
        
        # Wall at y = 5
        t_y_pos = float('inf')
        if math.sin(global_angle) != 0:
            t_y_pos = (5 - robot_y) / math.sin(global_angle)
        if t_y_pos > 0:
            x_at_y_pos = robot_x + t_y_pos * math.cos(global_angle)
            if -5 <= x_at_y_pos <= 5:
                return t_y_pos
        
        # Return max range if no intersection
        return 30.0

    def publish_imu_data(self):
        """Publish simulated IMU data"""
        imu = Imu()
        imu.header = Header()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.header.frame_id = 'imu_link'
        
        # Simulate IMU readings with realistic noise
        # Linear acceleration (simulating gravity and movement)
        imu.linear_acceleration.x = self.rng.normal(0, 0.05)
        imu.linear_acceleration.y = self.rng.normal(0, 0.05)
        imu.linear_acceleration.z = 9.81 + self.rng.normal(0, 0.05)
        
        # Angular velocity (including current rotation and noise)
        imu.angular_velocity.x = self.rng.normal(0, 0.001)
        imu.angular_velocity.y = self.rng.normal(0, 0.001)
        imu.angular_velocity.z = self.angular_velocity + self.rng.normal(0, 0.005)
        
        # Orientation (simplified - in real implementation should be integrated)
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('z', self.theta).as_quat()
        imu.orientation.x = quat[0] + self.rng.normal(0, 0.001)
        imu.orientation.y = quat[1] + self.rng.normal(0, 0.001)
        imu.orientation.z = quat[2] + self.rng.normal(0, 0.001)
        imu.orientation.w = quat[3] + self.rng.normal(0, 0.001)
        
        # Normalize quaternion
        norm = np.linalg.norm([imu.orientation.x, imu.orientation.y, 
                              imu.orientation.z, imu.orientation.w])
        imu.orientation.x /= norm
        imu.orientation.y /= norm
        imu.orientation.z /= norm
        imu.orientation.w /= norm
        
        # Set covariance values (information about measurement uncertainty)
        imu.orientation_covariance[0] = 0.01
        imu.orientation_covariance[4] = 0.01
        imu.orientation_covariance[8] = 0.01
        imu.angular_velocity_covariance[0] = 0.01
        imu.angular_velocity_covariance[4] = 0.01
        imu.angular_velocity_covariance[8] = 0.01
        imu.linear_acceleration_covariance[0] = 0.01
        imu.linear_acceleration_covariance[4] = 0.01
        imu.linear_acceleration_covariance[8] = 0.01
        
        self.imu_pub.publish(imu)

    def publish_camera_image(self):
        """Publish simulated camera image"""
        # Create a synthetic image with simulated environment
        height, width = 480, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add elements to simulate environment
        # Center of image corresponds to robot's forward direction
        center_x, center_y = width // 2, height // 2
        
        # Add simulated obstacles based on position and orientation
        # For this example, let's just add some static features
        cv2.rectangle(img, (center_x-50, center_y-100), (center_x+50, center_y-80), (100, 100, 100), -1)  # Ground feature
        cv2.circle(img, (center_x + int(100*math.cos(self.theta)), center_y + int(100*math.sin(self.theta))), 20, (0, 255, 0), -1)  # Green obstacle
        
        # Add simulated camera distortion and noise
        img = self.add_camera_effects(img)
        
        # Convert to ROS message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        
        self.camera_pub.publish(img_msg)

    def add_camera_effects(self, img):
        """Add realistic camera effects like noise and distortion"""
        # Add noise
        noise = self.rng.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img

    def publish_transform(self):
        """Publish TF transform"""
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('z', self.theta).as_quat()
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    simulator = GazeboRobotSimulator()
    
    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        simulator.get_logger().info('Simulation stopped by user')
    finally:
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    import cv2  # Required for image processing
    main()
```

## Unity Simulation Environment

### Unity Scene Configuration

Configuring Unity for advanced robotics simulation:

```csharp
// UnityRobotSimulation.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;
using System.Collections;

public class UnityRobotSimulation : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotBase;
    public WheelCollider[] wheelColliders;
    public GameObject[] wheelVisuals;
    public Transform cameraMount;
    public float maxMotorTorque = 100f;
    public float maxSteeringAngle = 45f;

    [Header("Physics Configuration")]
    public float robotMass = 20f;
    public float wheelRadius = 0.3f;
    public float wheelWidth = 0.1f;

    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;
    public string cmdVelTopic = "/cmd_vel";
    public string odomTopic = "/odom";
    public string laserTopic = "/scan";

    [Header("Simulation Configuration")]
    public float simulationSpeed = 1.0f;
    public int physicsIterations = 10;
    public int velocityIterations = 10;

    private ROSConnection ros;
    private Rigidbody robotRigidbody;
    private float motorTorque = 0f;
    private float steering = 0f;

    void Start()
    {
        // Configure physics parameters
        ConfigurePhysics();
        
        // Initialize ROS connection
        InitializeROS();
        
        // Initialize robot components
        InitializeRobot();
    }

    void ConfigurePhysics()
    {
        // Set physics parameters for accurate simulation
        Physics.defaultSolverIterations = physicsIterations;
        Physics.defaultSolverVelocityIterations = velocityIterations;
        Physics.sleepThreshold = 0.001f;
        Physics.defaultContactOffset = 0.01f;
        Physics.bounceThreshold = 2f;
    }

    void InitializeROS()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        
        // Subscribe to command topics
        ros.Subscribe<TwistMsg>(cmdVelTopic, ReceiveVelocityCommand);
    }

    void InitializeRobot()
    {
        robotRigidbody = GetComponent<Rigidbody>();
        if (robotRigidbody == null)
        {
            robotRigidbody = gameObject.AddComponent<Rigidbody>();
        }
        
        robotRigidbody.mass = robotMass;
        robotRigidbody.drag = 0.1f;
        robotRigidbody.angularDrag = 0.2f;
        robotRigidbody.interpolation = RigidbodyInterpolation.Interpolate;
    }

    void ReceiveVelocityCommand(TwistMsg cmd)
    {
        // Convert ROS velocity command to Unity controls
        // Linear x becomes forward movement
        motorTorque = cmd.linear.x * maxMotorTorque;
        
        // Angular z becomes steering
        steering = -cmd.angular.z * maxSteeringAngle;  // Negative for correct direction
    }

    void Update()
    {
        // Update wheel rotations for visual representation
        UpdateWheelVisuals();
    }

    void FixedUpdate()
    {
        // Apply physics-based movement
        ApplyWheelForces();
        
        // Publish odometry and sensor data
        PublishSensorData();
    }

    void ApplyWheelForces()
    {
        for (int i = 0; i < wheelColliders.Length; i++)
        {
            if (i < 2) // Front wheels - steering
            {
                wheelColliders[i].steerAngle = steering;
            }
            
            if (i >= 2) // Rear wheels - motor
            {
                wheelColliders[i].motorTorque = motorTorque;
            }
            
            // Apply handbrake effect if needed
            wheelColliders[i].brakeTorque = 0f;
        }
    }

    void UpdateWheelVisuals()
    {
        for (int i = 0; i < wheelColliders.Length && i < wheelVisuals.Length; i++)
        {
            Quaternion q;
            Vector3 p;
            
            wheelColliders[i].GetWorldPose(out p, out q);
            
            // Update visual wheel positions and rotations
            wheelVisuals[i].transform.position = p;
            wheelVisuals[i].transform.rotation = q;
        }
    }

    void PublishSensorData()
    {
        // Publish odometry data
        PublishOdometry();
        
        // In a complete implementation, also publish laser data, IMU, etc.
    }

    void PublishOdometry()
    {
        var odom = new OdometryMsg();
        odom.header = new std_msgs.HeaderMsg();
        odom.header.stamp = new TimeMsg();
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        
        // Position
        odom.pose.pose.position = new Vector3Msg(
            transform.position.x,
            transform.position.y,
            transform.position.z
        );
        
        // Orientation
        odom.pose.pose.orientation = new QuaternionMsg(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        );
        
        // Linear velocity
        odom.twist.twist.linear = new Vector3Msg(
            robotRigidbody.velocity.x,
            robotRigidbody.velocity.y,
            robotRigidbody.velocity.z
        );
        
        // Angular velocity
        odom.twist.twist.angular = new Vector3Msg(
            robotRigidbody.angularVelocity.x,
            robotRigidbody.angularVelocity.y,
            robotRigidbody.angularVelocity.z
        );
        
        ros.Publish(odomTopic, odom);
    }

    // Visualization methods for debugging
    void OnValidate()
    {
        if (wheelColliders.Length != wheelVisuals.Length)
        {
            Debug.LogWarning("Number of wheel colliders doesn't match number of wheel visuals");
        }
    }

    void OnDrawGizmos()
    {
        // Draw robot coordinate system
        Gizmos.color = Color.red;
        Gizmos.DrawLine(transform.position, transform.position + transform.right * 0.5f);
        
        Gizmos.color = Color.green;
        Gizmos.DrawLine(transform.position, transform.position + transform.up * 0.5f);
        
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(transform.position, transform.position + transform.forward * 0.5f);
    }
}
```

### Advanced Unity Perception Simulation

Implementing advanced perception simulation in Unity:

```csharp
// UnityPerceptionSimulator.cs
using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityPerceptionSimulator : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera perceptionCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float cameraFov = 60f;
    public float cameraNear = 0.1f;
    public float cameraFar = 100f;

    [Header("LIDAR Configuration")]
    public Transform lidarSensor;
    public float lidarRange = 30f;
    public int horizontalResolution = 720;
    public int verticalResolution = 1;
    public float lidarUpdateRate = 10f; // Hz

    [Header("Noise Configuration")]
    public float cameraNoiseIntensity = 0.01f;
    public float lidarNoise = 0.02f; // meters

    [Header("ROS Topics")]
    public string imageTopic = "/camera/image_raw";
    public string cameraInfoTopic = "/camera/camera_info";
    public string laserTopic = "/scan";

    private RenderTexture renderTexture;
    private float lidarUpdateInterval;
    private float lastLidarUpdate;
    private float lastImageUpdate;
    
    private ROSConnection ros;
    private float[] lidarData;

    void Start()
    {
        InitializePerception();
    }

    void InitializePerception()
    {
        // Camera setup
        if (perceptionCamera == null)
        {
            perceptionCamera = GetComponent<Camera>();
        }
        
        if (perceptionCamera == null)
        {
            perceptionCamera = gameObject.AddComponent<Camera>();
        }

        perceptionCamera.fieldOfView = cameraFov;
        perceptionCamera.nearClipPlane = cameraNear;
        perceptionCamera.farClipPlane = cameraFar;
        perceptionCamera.enabled = true;

        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        perceptionCamera.targetTexture = renderTexture;

        // LIDAR setup
        lidarUpdateInterval = 1.0f / lidarUpdateRate;
        lastLidarUpdate = 0;
        lidarData = new float[horizontalResolution];

        // ROS setup
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(ROSConnection.Instance.RosIPAddress, ROSConnection.Instance.RosPort);

        // Initialize camera info
        PublishCameraInfo();
    }

    void Update()
    {
        // Update LIDAR simulation
        if (Time.time - lastLidarUpdate >= lidarUpdateInterval)
        {
            SimulateLidarScan();
            lastLidarUpdate = Time.time;
        }
    }

    void SimulateLidarScan()
    {
        float horizontalAngleIncrement = 360.0f / horizontalResolution;

        for (int i = 0; i < horizontalResolution; i++)
        {
            float angle = i * horizontalAngleIncrement * Mathf.Deg2Rad;
            float distance = PerformLidarRaycast(angle);
            
            // Add noise to the measurement
            distance += Random.Range(-lidarNoise, lidarNoise);
            
            // Apply range limits
            lidarData[i] = Mathf.Clamp(distance, cameraNear, lidarRange);
        }

        PublishLaserScan();
    }

    float PerformLidarRaycast(float angle)
    {
        Vector3 direction = new Vector3(
            Mathf.Cos(angle),
            0f, // No elevation in 2D LIDAR
            Mathf.Sin(angle)
        );

        direction = lidarSensor.TransformDirection(direction);

        if (Physics.Raycast(lidarSensor.position, direction, out RaycastHit hit, lidarRange))
        {
            return hit.distance;
        }
        else
        {
            return lidarRange; // Max range if no object detected
        }
    }

    void PublishLaserScan()
    {
        var laserScan = new LaserScanMsg();
        laserScan.header = new std_msgs.HeaderMsg();
        laserScan.header.stamp = new TimeMsg();
        laserScan.header.frame_id = "laser_link";

        laserScan.angle_min = -Mathf.PI;
        laserScan.angle_max = Mathf.PI;
        laserScan.angle_increment = (2 * Mathf.PI) / horizontalResolution;
        laserScan.time_increment = 0.0f; // Not using time increment for this example
        laserScan.scan_time = 1.0f / lidarUpdateRate;
        laserScan.range_min = cameraNear;
        laserScan.range_max = lidarRange;

        laserScan.ranges = new float[horizontalResolution];
        for (int i = 0; i < horizontalResolution; i++)
        {
            laserScan.ranges[i] = lidarData[i];
        }

        laserScan.intensities = new float[horizontalResolution]; // No intensity data

        ros.Publish(laserTopic, laserScan);
    }

    public void CaptureAndPublishImage()
    {
        // Render the camera image to the render texture
        perceptionCamera.Render();

        // Create temporary texture to read from render texture
        Texture2D imageTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        // Remember the currently active render texture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Read pixels from the active render texture
        imageTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        imageTexture.Apply();

        // Restore the original render texture
        RenderTexture.active = currentRT;

        // Add noise to the image
        AddImageNoise(imageTexture);

        // Convert to ROS image message and publish
        PublishImageMessage(imageTexture);

        // Clean up
        Destroy(imageTexture);
    }

    void AddImageNoise(Texture2D texture)
    {
        // Apply noise to the image
        Color[] pixels = texture.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            Color original = pixels[i];
            float noise = Random.Range(-cameraNoiseIntensity, cameraNoiseIntensity);

            pixels[i] = new Color(
                Mathf.Clamp01(original.r + noise),
                Mathf.Clamp01(original.g + noise),
                Mathf.Clamp01(original.b + noise),
                original.a
            );
        }

        texture.SetPixels(pixels);
        texture.Apply();
    }

    void PublishImageMessage(Texture2D imageTexture)
    {
        // In a real implementation, convert Unity texture to ROS sensor_msgs/Image
        // This would involve encoding the image data properly
        
        var imageMsg = new ImageMsg();
        imageMsg.header = new std_msgs.HeaderMsg();
        imageMsg.header.stamp = new TimeMsg();
        imageMsg.header.frame_id = "camera_link";
        
        imageMsg.height = (uint)imageTexture.height;
        imageMsg.width = (uint)imageTexture.width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(imageTexture.width * 3); // 3 bytes per pixel for RGB
        
        // For this example, we're not actually transferring image data
        // In a real implementation, encode the texture data
        
        ros.Publish(imageTopic, imageMsg);
    }

    void PublishCameraInfo()
    {
        var cameraInfo = new CameraInfoMsg();
        cameraInfo.header = new std_msgs.HeaderMsg();
        cameraInfo.header.stamp = new TimeMsg();
        cameraInfo.header.frame_id = "camera_link";
        
        cameraInfo.width = (uint)imageWidth;
        cameraInfo.height = (uint)imageHeight;
        
        // Calculate camera matrix values
        float fx = (imageWidth / 2.0f) / Mathf.Tan(Mathf.Deg2Rad * cameraFov / 2.0f);
        float fy = fx; // Assuming square pixels
        float cx = imageWidth / 2.0f;
        float cy = imageHeight / 2.0f;
        
        cameraInfo.K = new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 };
        cameraInfo.R = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        cameraInfo.P = new double[] { fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 };
        
        cameraInfo.distortion_model = "plumb_bob";
        cameraInfo.D = new double[] { 0, 0, 0, 0, 0 }; // No distortion for this example

        ros.Publish(cameraInfoTopic, cameraInfo);
    }

    // Visualization for debugging LIDAR rays
    void OnDrawGizmosSelected()
    {
        if (lidarSensor == null) return;

        float horizontalAngleIncrement = 360.0f / horizontalResolution;
        Gizmos.color = Color.red;

        for (int i = 0; i < horizontalResolution; i += 20) // Draw every 20th ray for visibility
        {
            float angle = i * horizontalAngleIncrement * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = lidarSensor.TransformDirection(direction);

            if (i < lidarData.Length)
            {
                Gizmos.DrawRay(lidarSensor.position, direction * lidarData[i]);
            }
            else
            {
                Gizmos.DrawRay(lidarSensor.position, direction * lidarRange);
            }
        }
    }
}
```

## Isaac Sim Simulation Environment

### Isaac Sim Robot Configuration

Setting up advanced Isaac Sim robot models:

```python
# isaac_sim_advanced_robot.py
import omni
from pxr import Gf, Usd, UsdGeom, Sdf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import _sensor as _sensor
import numpy as np
import carb

class IsaacSimAdvancedRobot:
    def __init__(self, robot_name: str = "carter", position: np.ndarray = np.array([0, 0, 0.5])):
        self.world = World(stage_units_in_meters=1.0)
        self.robot_name = robot_name
        self.position = position
        self.robot = None
        
        self._setup_scene()
        self._setup_robot()
        
    def _setup_scene(self):
        """Set up the Isaac Sim scene with realistic environment"""
        # Add default ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        self._add_lighting()
        
        # Add textured environment
        self._add_environment_objects()

    def _add_lighting(self):
        """Add realistic lighting to the scene"""
        # Add a dome light for environment illumination
        dome_light_path = "/World/DomeLight"
        create_prim(
            prim_path=dome_light_path,
            prim_type="DomeLight",
            position=np.array([0, 0, 0]),
            orientation=np.array([0, 0, 0, 1]),
            attributes={"color": (0.4, 0.4, 0.4), "intensity": 3000}
        )
        
        # Add a distant light for shadows
        distant_light_path = "/World/DistantLight"
        create_prim(
            prim_path=distant_light_path,
            prim_type="DistantLight",
            position=np.array([0, 0, 10]),
            orientation=np.array([0, 0, 0, 1]),
            attributes={"color": (0.9, 0.9, 0.9), "intensity": 1000, "angle": 0.5}
        )

    def _add_environment_objects(self):
        """Add objects to create a realistic environment"""
        from omni.isaac.core.objects import DynamicCuboid
        
        # Add some static objects
        table = DynamicCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([2, 2, 0.4]),
            size=np.array([1.0, 0.6, 0.8]),
            color=np.array([0.5, 0.5, 0.5])
        )
        self.world.scene.add(table)
        
        # Add a wall
        wall = DynamicCuboid(
            prim_path="/World/Wall",
            name="wall",
            position=np.array([3, 0, 1]),
            size=np.array([0.1, 5, 2]),
            color=np.array([0.7, 0.7, 0.7])
        )
        self.world.scene.add(wall)

    def _setup_robot(self):
        """Set up the robot with realistic configuration"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            
        # Load the Carter robot
        robot_path = f"/World/{self.robot_name}"
        carter_path = assets_root_path + "/Isaac/Robots/Carter/carter_nucleus.usd"
        add_reference_to_stage(usd_path=carter_path, prim_path=robot_path)
        
        # Set the robot position
        robot_prim = get_prim_at_path(robot_path)
        if robot_prim.IsValid():
            # Set position
            import omni.kit.commands
            omni.kit.commands.execute(
                "TransformMultiPrimsSIPrimsCommand",
                count=1,
                paths=[robot_path],
                new_positions=[self.position[0], self.position[1], self.position[2]],
                usd_context=omni.usd.get_context()
            )
        
        # Create robot view for control and observation
        self.robot = ArticulationView(
            prim_path=robot_path,
            name="carter_view",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.robot)

    def setup_sensors(self):
        """Configure advanced sensors on the robot"""
        from omni.isaac.range_sensor import _range_sensor
        from omni.isaac.core.utils.prims import is_prim_path_valid
        
        # Get sensor interface
        self.sensor_interface = _range_sensor.acquire_imu_sensor_interface()
        
        # Add LIDAR sensor
        self._add_lidar_sensor()
        
        # Add camera sensor
        self._add_camera_sensor()
        
        # Add IMU sensor
        self._add_imu_sensor()

    def _add_lidar_sensor(self):
        """Add a 3D LIDAR sensor to the robot"""
        # Use Isaac Sim's built-in LIDAR creation
        lidar_path = f"/World/{self.robot_name}/Lidar"
        
        try:
            # Create LIDAR using Omniverse commands (this is a simplified representation)
            # In actual Isaac Sim, you would use the LIDAR sensor creation tools
            carb.log_info(f"Creating LIDAR sensor at {lidar_path}")
            
            # For this example, we'll note that we would add a LIDAR sensor
            # The actual implementation would depend on the Isaac Sim version
            pass
        except Exception as e:
            carb.log_error(f"Could not create LIDAR sensor: {e}")

    def _add_camera_sensor(self):
        """Add a camera sensor to the robot"""
        # Create camera sensor (simplified for the example)
        camera_path = f"/World/{self.robot_name}/Camera"
        
        # In practice, add a camera using Isaac Sim tools
        carb.log_info(f"Camera sensor placeholder at {camera_path}")

    def _add_imu_sensor(self):
        """Add an IMU sensor to the robot"""
        # Create IMU sensor (simplified for the example)
        imu_path = f"/World/{self.robot_name}/Imu"
        
        # In practice, add an IMU using Isaac Sim tools
        carb.log_info(f"IMU sensor placeholder at {imu_path}")

    def run_simulation(self, num_steps: int = 1000):
        """Run the simulation for a specified number of steps"""
        self.world.reset()
        
        for step in range(num_steps):
            # Apply some simple control (e.g., move forward)
            if step < 200:
                # Move forward for first 200 steps
                self.apply_velocity_commands(1.0, 0.0)  # linear vel = 1.0, angular vel = 0.0
            elif step < 400:
                # Turn left
                self.apply_velocity_commands(0.5, 0.5)  # forward and turn
            else:
                # Stop
                self.apply_velocity_commands(0.0, 0.0)
            
            # Step the world
            self.world.step(render=True)
            
            # Print progress every 100 steps
            if step % 100 == 0:
                robot_pos, robot_orn = self.robot.get_world_poses()
                carb.log_info(f"Step {step}: Position = {robot_pos[0]}, Orientation = {robot_orn[0]}")

    def apply_velocity_commands(self, linear_vel: float, angular_vel: float):
        """Apply velocity commands to the robot"""
        # For Carter robot, we need to convert differential drive commands to joint velocities
        # Wheel separation for Carter is approximately 0.44m, wheel radius 0.115m
        wheel_separation = 0.44  # meters
        wheel_radius = 0.115     # meters
        
        # Convert linear and angular velocities to wheel velocities
        left_wheel_vel = (linear_vel - angular_vel * wheel_separation / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_separation / 2.0) / wheel_radius
        
        # Apply joint velocities
        joint_indices = [0, 1]  # Assuming first two joints are wheels
        velocities = np.array([[left_wheel_vel, right_wheel_vel]])
        
        # In a real implementation, set the joint velocities
        # self.robot.set_joint_velocity_targets(velocities)

    def get_robot_state(self):
        """Get the current state of the robot"""
        positions, orientations = self.robot.get_world_poses()
        linear_vels, angular_vels = self.robot.get_velocities()
        
        return {
            'position': positions[0],
            'orientation': orientations[0],
            'linear_velocity': linear_vels[0],
            'angular_velocity': angular_vels[0]
        }

    def cleanup(self):
        """Clean up resources"""
        self.world.clear()
        carb.log_info("Isaac Sim Advanced Robot cleaned up")

def run_advanced_robot_simulation():
    """Run the advanced Isaac Sim robot simulation"""
    sim_robot = IsaacSimAdvancedRobot(position=np.array([0, 0, 0.5]))
    sim_robot.setup_sensors()
    
    carb.log_info("Starting advanced robot simulation...")
    sim_robot.run_simulation(num_steps=1000)
    sim_robot.cleanup()

if __name__ == "__main__":
    run_advanced_robot_simulation()
```

## Advanced Simulation Techniques

### Domain Randomization Implementation

Implementing domain randomization for robust sim-to-real transfer:

```python
# domain_randomization.py
import numpy as np
import random
from scipy.stats import truncnorm
import carb

class DomainRandomization:
    def __init__(self):
        self.parameters = {
            'robot_mass': (0.8, 1.2),  # 80% to 120% of nominal
            'friction_coeff': (0.1, 1.0),
            'camera_noise_multiplier': (0.5, 2.0),
            'lighting_intensity': (0.5, 2.0),
            'material_roughness': (0.0, 1.0),
            'gravity': (9.7, 9.9),  # Variation in gravity
            'actuator_dynamics': (0.9, 1.1),  # Delay/bandwidth variation
        }
        
        self.randomization_schedule = {
            'initial': 0.1,      # Apply 10% of range initially
            'final': 1.0,        # Apply full range eventually
            'schedule_type': 'linear',  # linear, exponential, etc.
            'episodes_for_full_randomization': 1000
        }
    
    def get_randomized_parameters(self, episode_number: int = 0):
        """Get randomized parameters based on episode number"""
        randomized_params = {}
        
        # Calculate schedule factor
        schedule_factor = min(
            1.0, 
            episode_number / self.randomization_schedule['episodes_for_full_randomization']
        )
        
        if self.randomization_schedule['schedule_type'] == 'linear':
            applied_range = self.randomization_schedule['initial'] + \
                           schedule_factor * (self.randomization_schedule['final'] - 
                                            self.randomization_schedule['initial'])
        elif self.randomization_schedule['schedule_type'] == 'exponential':
            applied_range = self.randomization_schedule['initial'] * \
                           (self.randomization_schedule['final'] / 
                            self.randomization_schedule['initial']) ** schedule_factor
        
        for param_name, (min_val, max_val) in self.parameters.items():
            # Calculate the range to apply based on schedule
            center = (min_val + max_val) / 2.0
            range_to_apply = (max_val - min_val) * applied_range / 2.0
            
            # Sample from the range
            randomized_params[param_name] = np.random.uniform(
                center - range_to_apply,
                center + range_to_apply
            )
        
        return randomized_params
    
    def get_truncated_normal_random(self, mean: float, std: float, min_val: float, max_val: float):
        """Get random value from truncated normal distribution"""
        a, b = (min_val - mean) / std, (max_val - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)
    
    def systematic_randomization(self, episode_number: int, n_dims: int = 10):
        """Use systematic sampling instead of purely random"""
        # Use Halton sequence for good coverage of parameter space
        def halton_sequence(index, base):
            result = 0
            f = 1.0
            i = index
            while i > 0:
                f = f / base
                result = result + f * (i % base)
                i = int(i / base)
            return result
        
        # Generate Halton sequence for each dimension
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:n_dims]  # Prime bases
        
        values = []
        for i, base in enumerate(bases):
            halton_val = halton_sequence(episode_number, base)
            values.append(halton_val)
        
        return values

class PhysicsParameterRandomizer:
    def __init__(self, sim_environment):
        self.sim_env = sim_environment
        self.domain_rand = DomainRandomization()
        
    def apply_randomization(self, episode_number: int):
        """Apply domain randomization to simulation parameters"""
        params = self.domain_rand.get_randomized_parameters(episode_number)
        
        # Apply parameters to simulation
        self._apply_mass_randomization(params)
        self._apply_friction_randomization(params)
        self._apply_dynamics_randomization(params)
        
        return params
    
    def _apply_mass_randomization(self, params):
        """Apply mass randomization to robot links"""
        # In a real implementation, update mass properties of robot
        # This would use the physics API of the simulation environment
        carb.log_info(f"Applying mass multiplier: {params['robot_mass']}")
    
    def _apply_friction_randomization(self, params):
        """Apply friction randomization to contacts"""
        # In a real implementation, update friction coefficients
        carb.log_info(f"Applying friction coefficient: {params['friction_coeff']}")
    
    def _apply_dynamics_randomization(self, params):
        """Apply actuator dynamics randomization"""
        # In a real implementation, update actuator parameters
        carb.log_info(f"Applying actuator dynamics multiplier: {params['actuator_dynamics']}")
```

### System Identification for Simulation Calibration

Implementing system identification to improve simulation accuracy:

```python
# system_identification.py
import numpy as np
from scipy.optimize import minimize, least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt

class SimulationSystemIdentification:
    def __init__(self, simulation_model, real_robot_interface):
        self.sim_model = simulation_model
        self.real_robot = real_robot_interface
        self.collected_data = []
        self.correction_model = None
        
    def collect_training_data(self, n_samples: int = 500):
        """Collect data for system identification"""
        carb.log_info(f"Collecting {n_samples} training data samples...")
        
        for i in range(n_samples):
            # Generate random control input
            control_input = self._generate_random_control()
            
            # Apply to real robot and get response
            real_state = self.real_robot.apply_control_and_measure(control_input)
            
            # Apply to simulation and get response
            sim_state = self.sim_model.apply_control_and_predict(control_input)
            
            # Store the data point with error
            data_point = {
                'control_input': control_input,
                'real_state': real_state,
                'sim_state': sim_state,
                'error': real_state - sim_state
            }
            
            self.collected_data.append(data_point)
            
            if (i + 1) % 100 == 0:
                carb.log_info(f"Collected {i + 1}/{n_samples} samples")
    
    def _generate_random_control(self):
        """Generate random control input for system ID"""
        # For a differential drive robot, random linear and angular velocities
        linear_vel = np.random.uniform(-1.0, 1.0)
        angular_vel = np.random.uniform(-0.5, 0.5)
        return np.array([linear_vel, angular_vel])
    
    def train_correction_model(self, model_type: str = 'gaussian_process'):
        """Train a model to predict simulation corrections"""
        if len(self.collected_data) == 0:
            raise ValueError("No training data collected")
        
        # Prepare training data
        X = np.array([data['sim_state'] for data in self.collected_data])
        y = np.array([data['error'] for data in self.collected_data])
        
        if model_type == 'gaussian_process':
            # Define kernel for Gaussian Process
            kernel = ConstantKernel(1.0) * RBF(1.0)
            self.correction_model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                alpha=1e-6,
                normalize_y=True
            )
        
        # Train the model
        self.correction_model.fit(X, y)
        carb.log_info("Correction model trained successfully")
        
        # Evaluate model performance
        score = self.correction_model.score(X, y)
        carb.log_info(f"Model R² score: {score:.4f}")
    
    def corrected_prediction(self, state, control):
        """Get simulation prediction with learned correction"""
        # Get base simulation prediction
        sim_prediction = self.sim_model.apply_control_and_predict(control)
        
        # Apply learned correction if model exists
        if self.correction_model is not None:
            correction = self.correction_model.predict(sim_prediction.reshape(1, -1))
            corrected_prediction = sim_prediction + correction.flatten()
        else:
            corrected_prediction = sim_prediction
        
        return corrected_prediction
    
    def optimize_simulation_parameters(self):
        """Optimize simulation parameters by minimizing error"""
        def objective_function(params):
            # Apply parameters to simulation
            self.sim_model.set_parameters(params)
            
            # Calculate total error
            total_error = 0
            for data_point in self.collected_data:
                predicted_state = self.sim_model.apply_control_and_predict(
                    data_point['control_input']
                )
                error = np.mean((predicted_state - data_point['real_state']) ** 2)
                total_error += error
            
            # Reset parameters for next iteration
            # self.sim_model.reset_parameters()  # Implement as appropriate
            
            return total_error / len(self.collected_data)
        
        # Get initial parameters
        initial_params = self.sim_model.get_parameters()
        
        # Optimize using scipy
        result = minimize(
            objective_function,
            initial_params,
            method='BFGS',
            options={'disp': True}
        )
        
        # Apply optimized parameters
        self.sim_model.set_parameters(result.x)
        
        carb.log_info(f"Optimization completed. Final parameters: {result.x}")
        carb.log_info(f"Final error: {result.fun}")
        
        return result

class AdaptiveSimulation:
    def __init__(self, base_simulator, system_id_module):
        self.base_sim = base_simulator
        self.sys_id = system_id_module
        self.performance_threshold = 0.05
        
    def adaptive_simulation(self, control_sequence, initial_state):
        """Run simulation with adaptive correction"""
        states = [initial_state]
        current_state = initial_state
        
        for t, control in enumerate(control_sequence):
            # Get prediction with correction
            predicted_state = self.sys_id.corrected_prediction(current_state, control)
            
            states.append(predicted_state)
            current_state = predicted_state
            
            # Periodically validate and update model
            if t % 50 == 0 and t > 0:
                self._validate_and_update_model(states[-50:])
        
        return states
    
    def _validate_and_update_model(self, recent_states):
        """Validate model and retrain if necessary"""
        # Implementation for online model validation and updating
        pass

def run_system_identification_example():
    """Run a system identification example"""
    # This would typically connect to real simulation and robot interfaces
    # sim_model = MySimulationModel()
    # real_robot = MyRealRobotInterface()
    # sys_id = SimulationSystemIdentification(sim_model, real_robot)
    
    # Collect training data
    # sys_id.collect_training_data(n_samples=500)
    
    # Train correction model
    # sys_id.train_correction_model()
    
    # Optimize parameters
    # result = sys_id.optimize_simulation_parameters()
    
    # Evaluate performance
    carb.log_info("System identification example completed")
```

## Simulation Validation and Performance Analysis

### Simulation Validation Techniques

Validating simulation accuracy and performance:

```python
# simulation_validation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SimulationValidator:
    def __init__(self):
        self.metrics = {}
        
    def validate_trajectory_tracking(self, sim_trajectory, real_trajectory):
        """Validate how well simulation matches real trajectory"""
        if len(sim_trajectory) != len(real_trajectory):
            raise ValueError("Trajectory lengths must match")
        
        # Calculate error metrics
        mse = mean_squared_error(real_trajectory, sim_trajectory)
        mae = mean_absolute_error(real_trajectory, sim_trajectory)
        r2 = r2_score(real_trajectory, sim_trajectory)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        # Calculate maximum error
        max_error = np.max(np.abs(real_trajectory - sim_trajectory))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'max_error': max_error,
            'mean_error': np.mean(real_trajectory - sim_trajectory),
            'std_error': np.std(real_trajectory - sim_trajectory)
        }
        
        return metrics
    
    def validate_sensor_data(self, sim_sensor_data, real_sensor_data):
        """Validate sensor data from simulation vs reality"""
        # Apply statistical tests
        ks_statistic, p_value = stats.ks_2samp(
            sim_sensor_data.flatten(), 
            real_sensor_data.flatten()
        )
        
        # Calculate correlation
        correlation = np.corrcoef(
            sim_sensor_data.flatten(), 
            real_sensor_data.flatten()
        )[0, 1]
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'correlation': correlation,
            'sim_mean': np.mean(sim_sensor_data),
            'real_mean': np.mean(real_sensor_data),
            'sim_std': np.std(sim_sensor_data),
            'real_std': np.std(real_sensor_data)
        }
    
    def validate_dynamic_response(self, sim_input_output, real_input_output):
        """Validate dynamic system response"""
        # For each input-output pair, validate the dynamic response
        errors = []
        for (sim_in, sim_out), (real_in, real_out) in zip(sim_input_output, real_input_output):
            # Ensure inputs are the same
            if not np.allclose(sim_in, real_in):
                raise ValueError("Input signals must match for dynamic validation")
            
            error = np.mean(np.abs(sim_out - real_out))
            errors.append(error)
        
        return {
            'mean_response_error': np.mean(errors),
            'std_response_error': np.std(errors),
            'max_response_error': np.max(errors)
        }
    
    def plot_validation_results(self, sim_data, real_data, title="Validation Results"):
        """Plot simulation vs real data for visual validation"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Trajectory comparison
        axes[0, 0].plot(real_data, label='Real', linewidth=2)
        axes[0, 0].plot(sim_data, label='Simulation', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Trajectory Comparison')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Error over time
        error = real_data - sim_data
        axes[0, 1].plot(error)
        axes[0, 1].set_title('Error Over Time')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True)
        
        # Plot 3: Scatter plot (sim vs real)
        axes[1, 0].scatter(sim_data, real_data, alpha=0.5)
        min_val = min(np.min(sim_data), np.min(real_data))
        max_val = max(np.max(sim_data), np.max(real_data))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Simulation')
        axes[1, 0].set_ylabel('Real')
        axes[1, 0].set_title('Simulation vs Real Scatter')
        axes[1, 0].grid(True)
        
        # Plot 4: Histogram of errors
        axes[1, 1].hist(error, bins=50, density=True, alpha=0.7)
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def conduct_comprehensive_validation():
    """Conduct comprehensive simulation validation"""
    validator = SimulationValidator()
    
    # Example: validating a simple trajectory
    time_steps = 100
    time = np.linspace(0, 10, time_steps)
    
    # Real data (simulated for this example)
    real_trajectory = np.sin(time) + 0.1 * np.random.normal(size=time_steps)
    
    # Simulation data (with some error)
    sim_trajectory = np.sin(time) * 0.95 + 0.05  # Simulated with slight offset and scale error
    
    # Validate trajectory tracking
    trajectory_metrics = validator.validate_trajectory_tracking(
        sim_trajectory, real_trajectory
    )
    
    print("Trajectory Validation Metrics:")
    for name, value in trajectory_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Plot results
    validator.plot_validation_results(
        sim_trajectory, real_trajectory, 
        title="Trajectory Validation: Simulation vs Real"
    )
    
    return trajectory_metrics

if __name__ == "__main__":
    conduct_comprehensive_validation()
```

## Chapter Summary

This simulation module provided comprehensive coverage of digital twin simulation environments for robotics using Gazebo, Unity, and Isaac Sim. Students implemented realistic robot models with accurate physics and sensor simulation, applied advanced techniques like domain randomization and system identification, and validated simulation models against real-world behavior. The module emphasized practical implementation skills needed to create effective digital twins that can accelerate robotics development and enable successful sim-to-real transfer.

## Key Terms
- Digital Twin Simulation
- Physics-Based Simulation
- Sensor Simulation and Calibration
- Domain Randomization
- System Identification
- Simulation Validation
- Multi-Platform Simulation
- Sim-to-Real Transfer

## Advanced Exercises
1. Implement a reinforcement learning task in each simulation environment and compare performance
2. Design and apply domain randomization techniques for a specific robotic manipulation task
3. Perform system identification to calibrate simulation parameters based on real robot data
4. Validate sim-to-real transfer by comparing results across all three simulation platforms