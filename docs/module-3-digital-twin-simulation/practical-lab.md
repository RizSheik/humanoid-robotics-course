# Module 3: Practical Lab - Digital Twin Simulation Environments

## Lab Overview

This practical lab provides hands-on experience with implementing digital twin simulation environments using Gazebo, Unity, and Isaac Sim. Students will work with all three platforms to understand their strengths and applications in robotics development, with a focus on creating accurate, efficient, and transferable simulation environments.

### Learning Objectives

After completing this lab, students will be able to:
1. Set up and configure simulation environments in Gazebo, Unity, and Isaac Sim
2. Create accurate robot models with realistic physics and sensor simulation
3. Implement advanced simulation techniques like domain randomization and system identification
4. Compare the strengths and weaknesses of different simulation platforms
5. Validate simulation models against real-world data
6. Design and implement sim-to-real transfer strategies

### Required Software/Tools

- **Gazebo Harmonic** or later with ROS 2 integration
- **Unity Hub** with Unity 2021.3 LTS and robotics packages
- **NVIDIA Isaac Sim** with Omniverse platform
- **ROS 2 Humble Hawksbill** or later
- **Python 3.11+** and **C++17** for custom implementations
- **Git** for version control
- **Docker** for Isaac Sim container deployment

### Lab Duration

This lab is designed for 18-22 hours of work, typically spread over 3-4 weeks.

## Lab 1: Gazebo Simulation Environment Setup

### Objective
Set up a Gazebo simulation environment with realistic physics and sensor simulation for a wheeled robot.

### Setup
1. Install Gazebo Harmonic with ROS 2 Humble
2. Create a simple wheeled robot model (e.g., differential drive)
3. Configure physics parameters and sensor simulation

### Implementation Steps
1. Create URDF model of a differential drive robot with appropriate inertial properties
2. Configure Gazebo plugins for differential drive control
3. Add sensor models (camera, LIDAR, IMU) with realistic noise parameters
4. Create a world file with obstacles and terrain features
5. Test the simulation and ensure proper ROS 2 topic bridging

### Code Template
```xml
<!-- differential_drive_robot.urdf -->
<?xml version="1.0"?>
<robot name="differential_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Left Wheel -->
  <link name="left_wheel">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Right Wheel -->
  <link name="right_wheel">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.15 0.175 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.15 -0.175 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Gazebo Plugins -->
  <gazebo>
    <plugin filename="libignition-gazebo-diff-drive-system.so" name="ignition::gazebo::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.35</wheel_separation>
      <wheel_radius>0.1</wheel_radius>
      <odom_publish_frequency>30</odom_publish_frequency>
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <tf_topic>tf</tf_topic>
    </plugin>
  </gazebo>

  <!-- Camera Sensor -->
  <gazebo reference="base_link">
    <sensor name="camera" type="camera">
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>
  </gazebo>

</robot>
```

### Analysis
- Evaluate the realism of the physics simulation
- Test sensor accuracy and noise characteristics
- Analyze the performance of the simulation (real-time factor)
- Verify ROS 2 topic communication

## Lab 2: Unity Simulation Environment Implementation

### Objective
Create a photorealistic Unity simulation environment with advanced rendering and perception capabilities.

### Setup
1. Install Unity Hub and Unity 2021.3 LTS
2. Install Unity Robotics packages (ROS TCP Connector, Perception)
3. Create a scene with realistic lighting and materials

### Implementation Steps
1. Set up a new Unity scene with outdoor environment
2. Create a 3D model of the robot (or import from assets)
3. Configure advanced lighting with realistic shadows
4. Implement perception pipeline with synthetic data generation
5. Connect to ROS using TCP connector

### Code Template
```csharp
// UnityRobotController.cs
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public Transform robotBase;
    public WheelCollider[] wheels;
    public float maxMotorTorque = 50f;
    public float maxSteeringAngle = 30f;

    [Header("ROS Connection")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    private ROSConnection ros;
    private float motorTorque = 0f;
    private float steering = 0f;

    void Start()
    {
        // Connect to ROS
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
        
        // Subscribe to command topic
        ros.Subscribe<TwistMsg>("/cmd_vel", ReceiveVelocityCommand);
        
        // Publish odometry
        InvokeRepeating("PublishOdometry", 0f, 0.05f);  // 20 Hz
    }

    void ReceiveVelocityCommand(TwistMsg cmd)
    {
        // Convert ROS velocity command to Unity controls
        motorTorque = cmd.linear.x * 50f;  // Scale factor
        steering = cmd.angular.z * 0.5f;   // Scale factor
    }

    void Update()
    {
        // Apply motor and steering to wheels
        foreach (var wheel in wheels)
        {
            wheel.motorTorque = motorTorque;
            
            if (wheel.gameObject.name.Contains("Front"))
            {
                wheel.steerAngle = steering;
            }
        }
    }

    void PublishOdometry()
    {
        // Calculate odometry data
        var odomData = new OdometryMsg();
        odomData.header.stamp = new TimeMsg();
        odomData.header.frame_id = "odom";
        odomData.child_frame_id = "base_link";
        
        // Set position and orientation
        odomData.pose.pose.position = new Vector3Msg(
            transform.position.x,
            transform.position.y,
            transform.position.z
        );
        
        odomData.pose.pose.orientation = new QuaternionMsg(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        );
        
        // Set velocities
        odomData.twist.twist.linear = new Vector3Msg(
            rigidbody.velocity.x,
            rigidbody.velocity.y,
            rigidbody.velocity.z
        );
        
        // Publish odometry
        ros.Publish("/odom", odomData);
    }
}
```

### Analysis
- Evaluate the photorealistic rendering quality
- Test perception pipeline and synthetic data generation
- Analyze the performance of Unity simulation (frame rate)
- Compare Unity's rendering advantages over Gazebo

## Lab 3: Isaac Sim Advanced Environment

### Objective
Implement an AI-powered robotics simulation environment in Isaac Sim with perception and control capabilities.

### Setup
1. Install Isaac Sim using Docker or direct installation
2. Set up Omniverse environment and extensions
3. Create a complex scene with multiple objects and dynamic elements

### Implementation Steps
1. Create a USD scene with advanced materials and lighting
2. Configure robot with accurate kinematics and dynamics
3. Implement Isaac ROS perception pipeline
4. Train a simple navigation policy using Isaac Gym
5. Generate synthetic datasets for computer vision tasks

### Python Template
```python
# isaac_sim_robot.py
import omni
from pxr import Gf, Usd, UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.sensor import _sensor as _sensor
import numpy as np
import carb

class IsaacSimRobotEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self._setup_scene()
        
    def _setup_scene(self):
        """Set up the Isaac Sim environment"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
            
        # Use a simple robot for this example
        robot_path = "/World/Robot"
        robot_prim_path = f"{robot_path}/Robot"
        
        # We'll create a simple cube-based robot structure
        robot_stage_path = "/Isaac/Robots/Carter/carter_nucleus.usd"
        if assets_root_path:
            full_path = assets_root_path + robot_stage_path
            add_reference_to_stage(usd_path=full_path, prim_path=robot_path)
        
        # Add objects for the robot to interact with
        self._add_objects()
        
    def _add_objects(self):
        """Add objects to the scene"""
        # Add a target object
        from omni.isaac.core.objects import DynamicCuboid
        
        self.target = DynamicCuboid(
            prim_path="/World/Target",
            name="target",
            position=np.array([1.5, 1.5, 0.5]),
            size=np.array([0.2, 0.2, 0.2]),
            color=np.array([1.0, 0.0, 0.0])
        )
        self.world.scene.add(self.target)
        
        # Add obstacles
        self.obstacle1 = DynamicCuboid(
            prim_path="/World/Obstacle1",
            name="obstacle1",
            position=np.array([1.0, 0.0, 0.5]),
            size=np.array([0.3, 0.3, 0.3]),
            color=np.array([0.0, 1.0, 0.0])
        )
        self.world.scene.add(self.obstacle1)
        
    def reset(self):
        """Reset the environment to initial state"""
        self.world.reset()
        
        # Reset robot position
        # This would depend on the specific robot implementation
        pass
        
    def step(self, action):
        """Execute an action and return the new state"""
        # Apply action to robot
        # This would involve sending commands to the robot's joints/controllers
        
        # Step the physics world
        self.world.step(render=True)
        
        # Get new observations
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self._is_done()
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        """Get current observation from the environment"""
        # Get robot state
        # Get sensor data (camera, LIDAR, etc.)
        # Format into observation space
        
        return np.zeros(10)  # Placeholder
    
    def _calculate_reward(self):
        """Calculate reward for the current state"""
        # Calculate based on task (navigation, manipulation, etc.)
        return 0.0  # Placeholder
    
    def _is_done(self):
        """Check if the episode is done"""
        return False  # Placeholder

def run_isaac_sim_robot():
    """Main function to run the Isaac Sim environment"""
    env = IsaacSimRobotEnvironment()
    
    # Run simulation for a few steps
    for i in range(100):
        obs, reward, done, info = env.step(np.zeros(2))  # Dummy action
        if done:
            env.reset()
    
    # Clean up
    env.world.clear()
```

### Analysis
- Evaluate the photorealistic rendering and physics simulation
- Test the Isaac ROS integration and perception pipeline
- Analyze the synthetic data generation capabilities
- Compare the AI training capabilities with other platforms

## Lab 4: Cross-Platform Comparison and Validation

### Objective
Compare simulation results across different platforms and validate against real-world data.

### Setup
1. Run the same robot control task in all three simulation environments
2. Collect data from each simulation
3. Compare the results and analyze differences

### Implementation Steps
1. Implement the same navigation task in Gazebo, Unity, and Isaac Sim
2. Use the same control algorithm in each environment
3. Collect performance metrics (trajectory accuracy, timing, resource usage)
4. Compare sensor data quality and realism
5. Analyze the simulation-to-simulation differences

### Analysis Template
```python
# comparison_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SimulationComparisonAnalyzer:
    def __init__(self, gazebo_data, unity_data, isaac_data):
        self.gazebo_data = gazebo_data
        self.unity_data = unity_data
        self.isaac_data = isaac_data
        
    def compare_performance_metrics(self):
        """Compare performance metrics across platforms"""
        metrics = {
            'gazebo': {
                'real_time_factor': self.calculate_rtf(self.gazebo_data),
                'avg_cpu_usage': self.calculate_cpu_usage(self.gazebo_data),
                'avg_memory_usage': self.calculate_memory_usage(self.gazebo_data)
            },
            'unity': {
                'real_time_factor': self.calculate_rtf(self.unity_data),
                'avg_cpu_usage': self.calculate_cpu_usage(self.unity_data),
                'avg_memory_usage': self.calculate_memory_usage(self.unity_data)
            },
            'isaac_sim': {
                'real_time_factor': self.calculate_rtf(self.isaac_data),
                'avg_cpu_usage': self.calculate_cpu_usage(self.isaac_data),
                'avg_memory_usage': self.calculate_memory_usage(self.isaac_data)
            }
        }
        
        return metrics
    
    def compare_sensor_quality(self):
        """Compare sensor data quality across platforms"""
        # Calculate signal-to-noise ratios, accuracy, etc.
        gazebo_camera_quality = self.evaluate_camera_quality(self.gazebo_data['camera'])
        unity_camera_quality = self.evaluate_camera_quality(self.unity_data['camera'])
        isaac_camera_quality = self.evaluate_camera_quality(self.isaac_data['camera'])
        
        return {
            'gazebo': gazebo_camera_quality,
            'unity': unity_camera_quality,
            'isaac_sim': isaac_camera_quality
        }
    
    def visualize_comparison(self):
        """Create comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance comparison
        platforms = ['Gazebo', 'Unity', 'Isaac Sim']
        rtf_values = [
            self.calculate_rtf(self.gazebo_data),
            self.calculate_rtf(self.unity_data),
            self.calculate_rtf(self.isaac_data)
        ]
        
        axes[0, 0].bar(platforms, rtf_values)
        axes[0, 0].set_title('Real-Time Factor Comparison')
        axes[0, 0].set_ylabel('RTF')
        
        # Sensor accuracy comparison
        sensor_acc = [
            self.evaluate_sensor_accuracy(self.gazebo_data),
            self.evaluate_sensor_accuracy(self.unity_data),
            self.evaluate_sensor_accuracy(self.isaac_data)
        ]
        
        axes[0, 1].bar(platforms, sensor_acc)
        axes[0, 1].set_title('Sensor Accuracy Comparison')
        axes[0, 1].set_ylabel('Accuracy (lower is better)')
        
        # Development complexity
        # This would be subjective ratings
        complexity_ratings = [2, 4, 5]  # 1-5 scale, 5 being most complex
        axes[1, 0].bar(platforms, complexity_ratings)
        axes[1, 0].set_title('Development Complexity')
        axes[1, 0].set_ylabel('Complexity Rating (1-5)')
        
        # Rendering quality
        # This would be based on subjective evaluation or metrics like FPS
        rendering_quality = [3, 5, 5]  # 1-5 scale, 5 being best
        axes[1, 1].bar(platforms, rendering_quality)
        axes[1, 1].set_title('Rendering Quality')
        axes[1, 1].set_ylabel('Quality Rating (1-5)')
        
        plt.tight_layout()
        plt.savefig('simulation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def conduct_cross_platform_analysis():
    """Conduct cross-platform simulation analysis"""
    # This function would:
    # 1. Run identical experiments in all three platforms
    # 2. Collect performance and quality metrics
    # 3. Generate comparison reports
    
    # Load data from all platforms
    # gazebo_data = load_gazebo_experiment_data()
    # unity_data = load_unity_experiment_data() 
    # isaac_data = load_isaac_sim_experiment_data()
    
    # analyzer = SimulationComparisonAnalyzer(gazebo_data, unity_data, isaac_data)
    # metrics = analyzer.compare_performance_metrics()
    # sensor_qualities = analyzer.compare_sensor_quality()
    # analyzer.visualize_comparison()
    
    # Generate comparison report
    print("Cross-platform analysis completed")
    print("For full analysis, implement the data loading and processing functions")
```

### Analysis
- Compare the performance metrics (real-time factor, resource usage)
- Evaluate the quality of sensor simulation in each platform
- Analyze the development complexity and time requirements
- Determine the best use cases for each platform

## Lab 5: Advanced Simulation Techniques Implementation

### Objective
Implement advanced digital twin simulation techniques including domain randomization, system identification, and sim-to-real transfer.

### Setup
1. Choose one of the simulation platforms as primary environment
2. Implement domain randomization techniques
3. Apply system identification to improve simulation accuracy

### Implementation Steps
1. Implement domain randomization for the robot simulation
2. Collect data from real robot (or realistic mock data)
3. Apply system identification to calibrate simulation parameters
4. Validate sim-to-real transfer capability
5. Document the process and results

### Code Template
```python
# advanced_techniques.py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class AdvancedSimulationTechniques:
    def __init__(self, simulation_environment):
        self.sim_env = simulation_environment
        self.randomization_params = self._initialize_randomization_params()
        self.system_id_data = []
        
    def _initialize_randomization_params(self):
        """Initialize parameters for domain randomization"""
        return {
            'mass_range': (0.8, 1.2),  # 80% to 120% of nominal
            'friction_range': (0.1, 1.0),
            'inertia_range': (0.9, 1.1),
            'sensor_noise_multiplier': (0.5, 2.0),
            'lighting_variation': (0.5, 2.0),
            'material_properties_range': (0.8, 1.2)
        }
    
    def apply_domain_randomization(self, episode_number=0):
        """Apply domain randomization to simulation parameters"""
        # Randomly select parameters within ranges
        random_params = {}
        for param_name, (min_val, max_val) in self.randomization_params.items():
            if 'range' in param_name or param_name.endswith('_multiplier'):
                # Random sampling
                random_params[param_name] = np.random.uniform(min_val, max_val)
            elif param_name == 'lighting_variation':
                # Apply to lighting if available
                random_params[param_name] = np.random.uniform(min_val, max_val)
        
        # Apply parameters to simulation
        self.sim_env.update_parameters(random_params)
        
        return random_params
    
    def collect_system_identification_data(self, real_robot_interface, n_samples=1000):
        """Collect data for system identification"""
        print("Collecting system identification data...")
        
        for i in range(n_samples):
            # Apply random control input
            control_input = self.generate_random_control()
            
            # Get real robot response
            real_state = real_robot_interface.apply_control_and_get_state(control_input)
            
            # Get simulation prediction
            sim_state = self.sim_env.apply_control_and_get_state(control_input)
            
            # Store data pair
            self.system_id_data.append({
                'control': control_input,
                'real_state': real_state,
                'sim_state': sim_state,
                'error': np.linalg.norm(real_state - sim_state)
            })
            
            if i % 100 == 0:
                print(f"Collected {i}/{n_samples} data points")
    
    def optimize_simulation_parameters(self):
        """Optimize simulation parameters to match real robot"""
        def objective_function(params):
            # Apply parameters to simulation
            self.sim_env.apply_parameter_vector(params)
            
            # Calculate total error across all data points
            total_error = 0
            for data_point in self.system_id_data:
                sim_state = self.sim_env.apply_control_and_get_state(
                    data_point['control']
                )
                error = np.linalg.norm(data_point['real_state'] - sim_state)
                total_error += error
            
            return total_error / len(self.system_id_data)
        
        # Get initial parameter vector from sim_env
        initial_params = self.sim_env.get_parameter_vector()
        
        # Optimize parameters
        result = minimize(
            objective_function,
            initial_params,
            method='BFGS',
            options={'disp': True}
        )
        
        # Apply optimized parameters
        self.sim_env.apply_parameter_vector(result.x)
        
        print(f"Optimization completed. Final error: {result.fun}")
        return result
    
    def validate_sim_to_real_transfer(self, policy, n_trials=10):
        """Validate policy transfer from simulation to reality"""
        sim_successes = 0
        real_successes = 0
        
        print("Validating sim-to-real transfer...")
        
        for trial in range(n_trials):
            print(f"Trial {trial + 1}/{n_trials}")
            
            # Test in simulation
            sim_success = self.test_policy_in_simulation(policy)
            if sim_success:
                sim_successes += 1
            
            # Test on real system (or high-fidelity simulation)
            real_success = self.test_policy_on_real_system(policy)
            if real_success:
                real_successes += 1
        
        sim_rate = sim_successes / n_trials
        real_rate = real_successes / n_trials
        
        print(f"Simulation success rate: {sim_rate:.2%}")
        print(f"Real system success rate: {real_rate:.2%}")
        print(f"Transfer gap: {sim_rate - real_rate:.2%}")
        
        return {
            'sim_success_rate': sim_rate,
            'real_success_rate': real_rate,
            'transfer_gap': sim_rate - real_rate
        }
    
    def generate_random_control(self):
        """Generate random control input for system ID"""
        # Example: random wheel velocities for differential drive
        return np.random.uniform(-1.0, 1.0, size=2)
    
    def test_policy_in_simulation(self, policy):
        """Test policy in simulation environment"""
        # Reset simulation
        self.sim_env.reset()
        
        # Run policy for episode
        for step in range(100):  # 100 steps per episode
            state = self.sim_env.get_state()
            action = policy(state)
            self.sim_env.apply_action(action)
            
            if self.sim_env.is_episode_done():
                break
        
        return self.sim_env.get_reward() > 0.5  # Success threshold
    
    def test_policy_on_real_system(self, policy):
        """Test policy on real system (mock implementation)"""
        # In practice, this would connect to real robot
        # For this exercise, we'll simulate with added realism
        success_prob = 0.7  # Simulated success probability
        return np.random.random() < success_prob

# Example usage
def implement_advanced_techniques_example():
    """Example implementation of advanced simulation techniques"""
    
    # Assuming we have a simulation environment
    # sim_env = initialize_simulation_environment()
    
    # Initialize advanced techniques
    # advanced_sim = AdvancedSimulationTechniques(sim_env)
    
    # Apply domain randomization
    # for episode in range(100):
    #     random_params = advanced_sim.apply_domain_randomization(episode)
    #     # Run training episode with randomized environment
    #     pass
    
    # Collect data for system identification
    # real_robot = initialize_real_robot_interface()  # Mock interface
    # advanced_sim.collect_system_identification_data(real_robot, n_samples=500)
    
    # Optimize simulation parameters
    # optimization_result = advanced_sim.optimize_simulation_parameters()
    
    # Validate sim-to-real transfer
    # dummy_policy = lambda state: np.random.uniform(-1, 1, 2)  # Random policy
    # transfer_results = advanced_sim.validate_sim_to_real_transfer(dummy_policy)
    
    print("Advanced simulation techniques implemented")
    print("For full implementation, connect to actual simulation and real robot environments")
```

## Lab Report Requirements

For each lab exercise, students must submit:

1. **Implementation Documentation** (20%):
   - Complete code with proper documentation
   - Step-by-step implementation guide
   - Configuration files and environment setup

2. **Performance Analysis** (35%):
   - Quantitative analysis of simulation performance
   - Comparison metrics between platforms
   - Resource usage analysis
   - Real-time factor measurements

3. **Validation Results** (25%):
   - Data comparing simulation to expected behavior
   - Analysis of sensor and physics accuracy
   - Sim-to-real transfer validation results

4. **System Design Report** (20%):
   - Architecture of the implemented simulation system
   - Design decisions and rationale
   - Challenges encountered and solutions

## Assessment Criteria

- Implementation quality and correctness (40%)
- Performance analysis and optimization (30%)
- Understanding of different simulation platforms (20%)
- Documentation and code quality (10%)

## Troubleshooting Tips

1. **Gazebo Issues**:
   - Ensure proper ROS 2 environment setup
   - Check plugin compatibility
   - Verify URDF/SDF model validity

2. **Unity Issues**:
   - Verify graphics card compatibility
   - Check ROS TCP Connector setup
   - Ensure proper physics configuration

3. **Isaac Sim Issues**:
   - Verify NVIDIA GPU and drivers
   - Check Omniverse connection
   - Ensure proper asset paths

4. **General Issues**:
   - Monitor system resource usage
   - Check network connectivity for ROS connections
   - Verify correct time synchronization

## Extensions and Advanced Challenges

1. **Multi-Robot Simulation**: Extend to multiple robots with coordination
2. **Complex Environments**: Create dynamic, changing environments
3. **Advanced Physics**: Implement soft-body physics or fluid dynamics
4. **AI Integration**: Train reinforcement learning policies in simulation
5. **Real Robot Integration**: Connect with actual robotic hardware

## References and Further Reading

- Gazebo Harmonic Documentation: http://gazebosim.org/
- Unity Robotics Hub: https://unity.com/solutions/robotics
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- ROS 2 Documentation: https://docs.ros.org/
- CoppeliaSim (V-REP) Documentation for alternative perspectives