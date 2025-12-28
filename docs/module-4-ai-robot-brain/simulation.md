# Module 4: Simulation - Isaac Platform for Perception and Control

## Simulation Environment Configuration

This simulation module provides hands-on experience with the NVIDIA Isaac Platform for developing AI-powered perception and control systems in robotics. Students will work with Isaac Sim for AI training, Isaac ROS for perception pipelines, and integrated control systems.

### Learning Objectives

After completing this simulation, students will be able to:
1. Configure and run Isaac Sim for training AI models
2. Implement GPU-accelerated perception pipelines using Isaac ROS
3. Design and test control systems in Isaac environments
4. Validate AI models trained in simulation for real-world transfer
5. Analyze performance and bottlenecks in AI robotics systems
6. Deploy and optimize AI models for edge robotics platforms

### Isaac Platform Architecture for Simulation

The Isaac Platform simulation environment consists of three main components:

1. **Isaac Sim** - High-fidelity physics simulation and synthetic data generation
2. **Isaac ROS** - GPU-accelerated perception and control packages
3. **Isaac Applications** - Pre-built AI applications for navigation and manipulation

### Required Simulation Tools

- **Isaac Sim** (Omniverse-based simulation platform)
- **Isaac ROS packages** (with GPU acceleration)
- **Isaac Navigation** and **Isaac Manipulation** applications
- **ROS 2 Humble Hawksbill** with Isaac extensions
- **NVIDIA GPU** with CUDA support
- **Python 3.11+** for scripting and development

## Simulation 1: Isaac Sim for AI Training

### Objective
Implement and test Isaac Sim for generating synthetic data and training AI models for robotics applications.

### Setup
1. Launch Isaac Sim with appropriate robot and environment models
2. Configure synthetic data generation pipelines
3. Implement domain randomization for robust model training
4. Set up reinforcement learning environments for control policy learning

### Implementation

```python
# isaac_sim_ai_training.py
import omni
import carb
from pxr import Gf, Usd, UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.core.sensors import ImuSensor
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class IsaacSimAIGym:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.syn_data = SyntheticDataHelper()
        
        # Robot configuration
        self.robot_name = "carter"
        self.robot = None
        self.robot_view = None
        
        # Environment configuration
        self.scene_objects = []
        self.domain_randomization_settings = {
            'lighting_variation': (0.5, 2.0),
            'material_roughness': (0.0, 1.0),
            'texture_randomization': True,
            'dynamic_objects': 3,
            'environment_complexity': 5  # 1-10 scale
        }
        
        # AI training configuration
        self.observation_space = 24  # Example: 24-dimensional state space
        self.action_space = 2        # Example: linear and angular velocity
        self.episode_length = 500    # Steps per episode
        self.current_step = 0
        
    def setup_environment(self):
        """Set up the Isaac Sim environment for AI training"""
        print("Setting up Isaac Sim environment for AI training...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        self._add_lighting()
        
        # Load robot
        self._load_robot()
        
        # Add objects for interaction
        self._add_training_objects()
        
        # Setup sensors
        self._setup_sensors()
        
        # Configure domain randomization
        self._configure_domain_randomization()
        
        print("Isaac Sim environment setup complete!")
    
    def _add_lighting(self):
        """Add realistic lighting to the environment"""
        # Add dome light for ambient illumination
        dome_light_path = "/World/DomeLight"
        carb.kit.commands.execute(
            "CreatePrim",
            prim_type="DomeLight",
            prim_path=dome_light_path,
            attributes={"color": (0.2, 0.2, 0.2), "intensity": 500}
        )
        
        # Add directional light for shadows
        distant_light_path = "/World/DistantLight"
        carb.kit.commands.execute(
            "CreatePrim", 
            prim_type="DistantLight",
            prim_path=distant_light_path,
            attributes={"color": (0.9, 0.9, 0.9), "intensity": 500, "angle": 0.1}
        )
    
    def _load_robot(self):
        """Load and configure the robot for training"""
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        robot_path = f"/World/{self.robot_name}"
        carter_path = assets_root_path + "/Isaac/Robots/Carter/carter_nucleus.usd"
        add_reference_to_stage(usd_path=carter_path, prim_path=robot_path)
        
        # Create robot view for control
        from omni.isaac.core.articulations import ArticulationView
        self.robot_view = ArticulationView(
            prim_path=robot_path,
            name="carter_view",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.robot_view)
        
        print(f"Robot {self.robot_name} loaded and added to scene")
    
    def _add_training_objects(self):
        """Add objects for navigation and manipulation training"""
        # Add target object
        target = DynamicCuboid(
            prim_path="/World/Target",
            name="target",
            position=np.array([3.0, 0.0, 0.2]),
            size=np.array([0.2, 0.2, 0.2]),
            color=np.array([1.0, 0.0, 0.0])  # Red target
        )
        self.world.scene.add(target)
        self.scene_objects.append(target)
        
        # Add obstacles
        for i in range(5):
            obstacle = DynamicCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array([
                    np.random.uniform(1.0, 4.0),
                    np.random.uniform(-2.0, 2.0),
                    0.2
                ]),
                size=np.array([0.3, 0.3, 0.4]),
                color=np.array([0.5, 0.5, 0.5])  # Gray obstacle
            )
            self.world.scene.add(obstacle)
            self.scene_objects.append(obstacle)
        
        print(f"Added target and {len(self.scene_objects)-1} obstacles for training")
    
    def _setup_sensors(self):
        """Set up robot sensors for training"""
        # In a real implementation, we would add sensors to the robot
        # For this simulation, we'll note the sensor configuration
        print("Sensor configuration noted for AI training")
    
    def _configure_domain_randomization(self):
        """Configure domain randomization for robust training"""
        print("Domain randomization configured:")
        for setting, value in self.domain_randomization_settings.items():
            print(f"  {setting}: {value}")
    
    def reset_environment(self):
        """Reset the environment to a new configuration"""
        self.world.reset()
        self.current_step = 0
        
        # Reset robot to random position
        rand_x = np.random.uniform(-2.0, 2.0)
        rand_y = np.random.uniform(-2.0, 2.0)
        robot_position = np.array([rand_x, rand_y, 0.5])
        self.robot_view.set_world_poses(positions=robot_position.reshape(1, 3))
        
        # Randomize other objects in the scene
        self._randomize_scene()
        
        return self.get_observation()
    
    def _randomize_scene(self):
        """Apply domain randomization to the scene"""
        # Randomize lighting
        dome_light = get_prim_at_path("/World/DomeLight")
        if dome_light.IsValid():
            intensity = np.random.uniform(
                self.domain_randomization_settings['lighting_variation'][0],
                self.domain_randomization_settings['lighting_variation'][1]
            ) * 500
            dome_light.GetAttribute("intensity").Set(intensity)
        
        # Randomize object positions
        for i, obj in enumerate(self.scene_objects[1:], start=1):  # Skip target
            new_pos = np.array([
                np.random.uniform(0.5, 4.0),
                np.random.uniform(-3.0, 3.0),
                0.2
            ])
            obj.set_world_poses(positions=new_pos.reshape(1, 3))
    
    def get_observation(self):
        """Get current observation from the environment"""
        # Get robot state
        robot_pos, robot_orn = self.robot_view.get_world_poses()
        robot_lin_vel, robot_ang_vel = self.robot_view.get_velocities()
        
        # Get target position
        target_pos, _ = self.scene_objects[0].get_world_poses()  # First object is target
        
        # Calculate relative position to target
        rel_pos = target_pos[0] - robot_pos[0]
        
        # Combine into observation vector
        observation = np.concatenate([
            robot_pos[0],  # Robot position [x, y, z]
            robot_orn[0],  # Robot orientation [qw, qx, qy, qz]
            robot_lin_vel[0, :2],  # Robot linear velocity [x, y] (ignore z)
            robot_ang_vel[0, 2:3],  # Robot angular velocity [z] (ignore x, y)
            rel_pos[:2]  # Relative position to target [x, y]
        ]).astype(np.float32)
        
        return observation
    
    def apply_action(self, action):
        """Apply action to the robot"""
        # Action format: [linear_velocity, angular_velocity]
        linear_vel = np.clip(action[0], -1.0, 1.0)
        angular_vel = np.clip(action[1], -1.0, 1.0)
        
        # Convert to wheel velocities for differential drive
        wheel_separation = 0.44  # meters
        wheel_radius = 0.115     # meters
        
        left_wheel_vel = (linear_vel - angular_vel * wheel_separation / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_separation / 2.0) / wheel_radius
        
        # Apply velocities to wheels (in real implementation, set joint velocities)
        # self.robot_view.set_joint_velocities(np.array([[left_wheel_vel, right_wheel_vel]]))
        
    def calculate_reward(self, action, obs):
        """Calculate reward for the current step"""
        # Get target and robot positions
        target_pos, _ = self.scene_objects[0].get_world_poses()  # Target is first object
        robot_pos, _ = self.robot_view.get_world_poses()
        
        # Calculate distance to target
        distance = np.linalg.norm(target_pos[0, :2] - robot_pos[0, :2])
        
        # Dense reward based on distance to target
        reward = -distance * 0.01  # Negative because closer is better
        
        # Bonus for getting very close to target
        if distance < 0.3:
            reward += 1.0
        
        # Penalty for large actions to encourage smooth movement
        action_penalty = -0.001 * np.sum(np.abs(action))
        reward += action_penalty
        
        return reward
    
    def is_done(self):
        """Check if the episode is done"""
        # Get positions
        robot_pos, _ = self.robot_view.get_world_poses()
        target_pos, _ = self.scene_objects[0].get_world_poses()
        
        # Calculate distance
        distance = np.linalg.norm(target_pos[0, :2] - robot_pos[0, :2])
        
        # Episode is done if:
        # 1. Reached target
        # 2. Exceeded maximum steps
        # 3. Robot is out of bounds
        return (
            distance < 0.3 or  # Reached target
            self.current_step >= self.episode_length or  # Exceeded max steps
            abs(robot_pos[0, 0]) > 10.0 or  # Out of bounds in X
            abs(robot_pos[0, 1]) > 10.0     # Out of bounds in Y
        )
    
    def step(self, action):
        """Take a step in the environment"""
        # Apply action to robot
        self.apply_action(action)
        
        # Step the physics simulation
        self.world.step(render=True)
        self.current_step += 1
        
        # Get new observation
        observation = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward(action, observation)
        
        # Check if episode is done
        done = self.is_done()
        
        # Additional info
        info = {
            'step': self.current_step,
            'distance_to_target': np.linalg.norm(observation[-2:]),
            'episode_done': done
        }
        
        return observation, reward, done, info
    
    def run_training_episode(self, policy):
        """Run a single training episode using the given policy"""
        obs = self.reset_environment()
        total_reward = 0
        step_count = 0
        
        print(f"Starting episode with initial observation shape: {obs.shape}")
        
        while not self.is_done() and step_count < self.episode_length:
            # Get action from policy
            if hasattr(policy, 'get_action'):
                action = policy.get_action(obs)
            else:
                # Default random policy for simulation
                action = np.random.uniform(-1, 1, size=self.action_space).astype(np.float32)
            
            # Take step in environment
            obs, reward, done, info = self.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward = {reward:.3f}, Total = {total_reward:.3f}")
        
        print(f"Episode completed: Steps = {step_count}, Total Reward = {total_reward:.3f}")
        return total_reward, step_count

class AIPolicy:
    """Simple AI policy for demonstration"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.network = self._build_network()
    
    def _build_network(self):
        """Build a simple neural network for policy"""
        # In practice, this would be a more complex network trained on the environment
        return nn.Sequential(
            nn.Linear(24, 128),  # Match observation space
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
    
    def get_action(self, observation):
        """Get action from policy"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            action = self.network(obs_tensor)
        
        # Convert to numpy and clip to valid range
        action_np = action.numpy()[0]
        return np.clip(action_np, -1.0, 1.0)

def train_ai_model():
    """Train an AI model using Isaac Sim environment"""
    sim_env = IsaacSimAIGym()
    sim_env.setup_environment()
    
    # Create a simple policy for demonstration
    policy = AIPolicy(action_space=2)
    
    print("Starting training loop...")
    
    # Run several episodes to test the environment
    for episode in range(10):
        print(f"\nRunning training episode {episode + 1}/10")
        total_reward, steps = sim_env.run_training_episode(policy)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.3f}, Steps = {steps}")
        
        if episode < 9:  # Don't reset after the last episode
            obs = sim_env.reset_environment()
    
    print("\nTraining simulation completed!")

def main():
    """Main function for Isaac Sim AI Training simulation"""
    carb.log_info("Starting Isaac Sim AI Training Simulation")
    
    try:
        train_ai_model()
    except Exception as e:
        carb.log_error(f"Error in Isaac Sim AI Training: {e}")
    finally:
        carb.log_info("Isaac Sim AI Training Simulation completed")

if __name__ == "__main__":
    main()
```

### Analysis
- Monitor training performance and reward trends
- Evaluate the effectiveness of domain randomization
- Analyze the quality of synthetic data generated
- Assess the transferability of trained policies

## Simulation 2: Isaac ROS Perception Pipeline

### Objective
Implement and evaluate GPU-accelerated perception pipelines using Isaac ROS packages for computer vision tasks.

### Setup
1. Configure Isaac ROS perception packages with GPU acceleration
2. Set up camera and sensor data processing pipelines
3. Implement object detection and tracking systems
4. Validate perception accuracy and performance

### Implementation

```python
# isaac_ros_perception_sim.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Header, String
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
import torch
import time
import threading
from collections import deque

class IsaacROSPipelineSimulator(Node):
    def __init__(self):
        super().__init__('isaac_ros_pipeline_simulator')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers for simulation
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        
        # Perception output publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.visualization_pub = self.create_publisher(Image, '/perception_visualization', 10)
        
        # Control publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Internal state
        self.latest_image = None
        self.latest_imu = None
        self.latest_scan = None
        self.simulated_objects = []
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.detection_rates = deque(maxlen=100)
        
        # Simulation parameters
        self.sim_width = 640
        self.sim_height = 480
        self.sim_fov = 60  # degrees
        self.sim_frame_rate = 30  # Hz
        
        # Perception parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4  # Non-maximum suppression
        
        # Create timer for simulation
        self.timer = self.create_timer(1.0/self.sim_frame_rate, self.simulation_step)
        
        # Initialize perception models
        self.perception_model = self.initialize_perception_model()
        
        self.get_logger().info('Isaac ROS Pipeline Simulator initialized')

    def initialize_perception_model(self):
        """Initialize perception model for simulation"""
        # In a real implementation, this would load an Isaac ROS DNN Inference model
        # For this simulation, we'll create a simple model
        
        # Create a simple CNN for object detection simulation
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 80)  # 80 coco classes
        )
        
        # Set model to evaluation mode
        model.eval()
        
        self.get_logger().info('Perception model initialized')
        return model

    def simulation_step(self):
        """Main simulation step - generate sensor data and run perception"""
        # Generate simulated sensor data
        self.generate_simulated_image()
        self.generate_simulated_imu()
        self.generate_simulated_scan()
        
        # Run perception pipeline
        if self.latest_image is not None:
            self.run_perception_pipeline()
    
    def generate_simulated_image(self):
        """Generate simulated camera image with objects"""
        # Create a synthetic image
        img = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
        
        # Add background elements
        cv2.rectangle(img, (0, 0), (self.sim_width, self.sim_height), (100, 150, 200), -1)  # Sky-like background
        
        # Add ground
        cv2.rectangle(img, (0, int(0.7*self.sim_height)), (self.sim_width, self.sim_height), (50, 150, 50), -1)  # Ground
        
        # Add simulated objects (for perception testing)
        objects = [
            {'name': 'person', 'bbox': [100, 150, 80, 150], 'color': (0, 255, 0)},
            {'name': 'car', 'bbox': [300, 250, 120, 80], 'color': (255, 0, 0)},
            {'name': 'chair', 'bbox': [200, 300, 60, 80], 'color': (0, 0, 255)}
        ]
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(img, (x, y), (x+w, y+h), obj['color'], -1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)  # White border
            cv2.putText(img, obj['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Convert to ROS Image and publish
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        
        self.image_pub.publish(img_msg)
        self.latest_image = img
    
    def generate_simulated_imu(self):
        """Generate simulated IMU data"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        
        # Simulate IMU readings with realistic noise
        imu_msg.linear_acceleration.x = np.random.normal(0.0, 0.1)
        imu_msg.linear_acceleration.y = np.random.normal(0.0, 0.1)
        imu_msg.linear_acceleration.z = 9.8 + np.random.normal(0.0, 0.1)  # Gravity + noise
        
        imu_msg.angular_velocity.x = np.random.normal(0.0, 0.01)
        imu_msg.angular_velocity.y = np.random.normal(0.0, 0.01)
        imu_msg.angular_velocity.z = np.random.normal(0.0, 0.01)
        
        # Orientation (simplified)
        imu_msg.orientation.w = 1.0
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        
        # Set covariance values
        for i in range(9):
            imu_msg.linear_acceleration_covariance[i] = 0.01 if i % 4 == 0 else 0.0
            imu_msg.angular_velocity_covariance[i] = 0.01 if i % 4 == 0 else 0.0
            imu_msg.orientation_covariance[i] = 0.1 if i % 4 == 0 else 0.0
        
        self.imu_pub.publish(imu_msg)
        self.latest_imu = imu_msg

    def generate_simulated_scan(self):
        """Generate simulated LIDAR scan data"""
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_link'
        
        # LIDAR parameters
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 2 * np.pi / 360  # 360 points
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / self.sim_frame_rate
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        
        # Generate simulated ranges with some objects
        ranges = []
        for i in range(360):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            
            # Simulate distance to nearest object
            # Add some simulated objects at specific angles
            distance = scan_msg.range_max  # Default to max range
            
            # Add simulated objects at specific positions
            if 85 < i < 95:  # Front object at 2m
                distance = 2.0 + np.random.normal(0, 0.05)
            elif 265 < i < 275:  # Back object at 1.5m
                distance = 1.5 + np.random.normal(0, 0.05)
            else:
                # Random noise and ground
                if np.random.random() < 0.02:  # 2% chance of obstacle
                    distance = np.random.uniform(0.5, 5.0)
            
            ranges.append(min(distance, scan_msg.range_max))
        
        scan_msg.ranges = ranges
        scan_msg.intensities = []  # No intensity data for simulation
        
        self.scan_pub.publish(scan_msg)
        self.latest_scan = scan_msg

    def run_perception_pipeline(self):
        """Run the Isaac ROS perception pipeline on simulated data"""
        start_time = time.time()
        
        # Convert image to tensor for model input
        image_tensor = torch.from_numpy(self.latest_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Run object detection simulation
        with torch.no_grad():
            # In real Isaac ROS, this would call the actual DNN inference node
            detections = self.simulate_object_detection(image_tensor)
        
        # Process detections and publish results
        detection_array = self.process_detections(detections)
        self.detection_pub.publish(detection_array)
        
        # Create and publish visualization
        vis_image = self.create_visualization(self.latest_image, detections)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
        vis_msg.header.stamp = self.get_clock().now().to_msg()
        vis_msg.header.frame_id = 'camera_link'
        self.visualization_pub.publish(vis_msg)
        
        # Record performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Calculate FPS
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        self.get_logger().debug(f'Perception FPS: {avg_fps:.2f} (Inference time: {inference_time:.4f}s)')
    
    def simulate_object_detection(self, image_tensor):
        """Simulate object detection (in real implementation, use Isaac ROS DNN Inference)"""
        # In a real Isaac ROS system, this would use Isaac ROS DNN Inference package
        # For this simulation, return pseudo-detections
        
        # Simulate detections based on what's in the image
        simulated_detections = []
        
        # Hardcoded detections to match the objects we put in the simulated image
        objects_in_image = [
            {'class': 'person', 'bbox': [100, 150, 180, 300], 'confidence': 0.85},
            {'class': 'car', 'bbox': [300, 250, 420, 330], 'confidence': 0.78},
            {'class': 'chair', 'bbox': [200, 300, 260, 380], 'confidence': 0.65}
        ]
        
        for obj in objects_in_image:
            # Only include if confidence is above threshold
            if obj['confidence'] > self.confidence_threshold:
                detection = {
                    'class': obj['class'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox']  # [x_min, y_min, x_max, y_max]
                }
                simulated_detections.append(detection)
        
        return simulated_detections

    def process_detections(self, detections):
        """Process detections into ROS message format"""
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'
        
        for det in detections:
            detection_msg = Detection2D()
            
            # Calculate center and size from bbox [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = det['bbox']
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            size_x = x_max - x_min
            size_y = y_max - y_min
            
            detection_msg.bbox.center.x = float(center_x)
            detection_msg.bbox.center.y = float(center_y)
            detection_msg.bbox.size_x = float(size_x)
            detection_msg.bbox.size_y = float(size_y)
            
            # Add classification result
            from vision_msgs.msg import ObjectHypothesisWithPose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det['class']
            hypothesis.hypothesis.score = det['confidence']
            
            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)
        
        return detection_array

    def create_visualization(self, image, detections):
        """Create visualization of detections on image"""
        vis_image = image.copy()
        
        for detection in detections:
            if detection['confidence'] > self.confidence_threshold:
                bbox = detection['bbox']
                x_min, y_min, x_max, y_max = bbox
                
                # Draw bounding box
                cv2.rectangle(vis_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                
                # Draw label and confidence
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                label_position = (int(x_min), int(y_min) - 10)
                cv2.putText(vis_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_image

    def get_performance_metrics(self):
        """Get performance metrics for the perception pipeline"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0.0,
                'avg_fps': 0.0,
                'latency_percentiles': [0.0, 0.0]
            }
        
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        times_list = list(self.inference_times)
        p50 = float(np.percentile(times_list, 50)) if len(times_list) > 1 else avg_inference_time
        p95 = float(np.percentile(times_list, 95)) if len(times_list) > 1 else avg_inference_time
        
        return {
            'avg_inference_time': avg_inference_time,
            'avg_fps': avg_fps,
            'latency_percentiles': [p50, p95],
            'inference_count': len(self.inference_times)
        }

class IsaacROSPerceptionValidator(Node):
    def __init__(self):
        super().__init__('isaac_ros_perception_validator')
        
        # Subscribe to perception outputs
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_validation_callback, 10)
        self.vis_sub = self.create_subscription(
            Image, '/perception_visualization', self.visualization_validation_callback, 10)
        
        # Performance monitoring
        self.detected_objects = deque(maxlen=100)
        self.validation_results = {}
        
        # Timer for periodic validation
        self.validation_timer = self.create_timer(5.0, self.periodic_validation)
        
        self.get_logger().info('Isaac ROS Perception Validator initialized')

    def detection_validation_callback(self, msg):
        """Validate incoming detections"""
        # Count detected objects
        num_detections = len(msg.detections)
        self.detected_objects.append(num_detections)
        
        # Validate detection quality
        for detection in msg.detections:
            # Check that bounding boxes are valid
            bbox = detection.bbox
            if bbox.size_x <= 0 or bbox.size_y <= 0:
                self.get_logger().warning('Invalid bounding box detected')
            
            # Check confidence scores
            for result in detection.results:
                if not (0.0 <= result.hypothesis.score <= 1.0):
                    self.get_logger().warning(f'Invalid confidence score: {result.hypothesis.score}')

    def visualization_validation_callback(self, msg):
        """Validate visualization output"""
        # Check image dimensions
        if msg.height != 480 or msg.width != 640:  # Our simulation dimensions
            self.get_logger().warning(f'Unexpected image dimensions: {msg.width}x{msg.height}')

    def periodic_validation(self):
        """Periodically validate perception performance"""
        if not self.detected_objects:
            self.get_logger().info('No detections received for validation')
            return
        
        avg_detections = sum(self.detected_objects) / len(self.detected_objects)
        
        self.get_logger().info(f'Perception Validation - Avg Detections: {avg_detections:.2f}, '
                              f'Buffer Size: {len(self.detected_objects)}')

def run_perception_simulation():
    """Run the Isaac ROS perception simulation"""
    rclpy.init()
    
    # Create simulator and validator nodes
    simulator = IsaacROSPipelineSimulator()
    validator = IsaacROSPerceptionValidator()
    
    try:
        # Combine nodes in single executor
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(simulator)
        executor.add_node(validator)
        
        # Run for a while to collect data
        import signal
        import sys
        
        def signal_handler(sig, frame):
            print('Shutting down perception simulation...')
            simulator.get_logger().info('Performance metrics:')
            metrics = simulator.get_performance_metrics()
            for key, value in metrics.items():
                simulator.get_logger().info(f'  {key}: {value}')
            rclpy.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        simulator.get_logger().info('Starting Isaac ROS perception simulation...')
        executor.spin()
        
    except KeyboardInterrupt:
        simulator.get_logger().info('Shutting down perception simulation...')
        metrics = simulator.get_performance_metrics()
        for key, value in metrics.items():
            simulator.get_logger().info(f'{key}: {value}')
    finally:
        simulator.destroy_node()
        validator.destroy_node()
        rclpy.shutdown()

def main():
    run_perception_simulation()

if __name__ == '__main__':
    main()
```

### Analysis
- Measure perception accuracy and false positive rates
- Evaluate performance metrics (FPS, latency)
- Assess the quality of object detections
- Analyze computational resource usage

## Simulation 3: Isaac ROS Control Systems

### Objective
Implement and test GPU-accelerated control systems using Isaac ROS navigation and manipulation applications.

### Setup
1. Configure Isaac ROS navigation stack with GPU acceleration
2. Set up manipulation control systems with Isaac ROS
3. Implement high-level planning and execution systems
4. Validate control system performance and safety

### Implementation

```python
# isaac_ros_control_sim.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, Imu, LaserScan
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header, String, Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import tf2_ros
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class IsaacNavigationStackSimulator(Node):
    def __init__(self):
        super().__init__('isaac_navigation_stack_simulator')
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.localization_pub = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_command_callback, 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Robot state
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta (radians)
        self.robot_velocity = np.array([0.0, 0.0])    # linear, angular
        self.target_pose = None
        self.navigation_state = 'idle'  # idle, planning, executing, reached_goal
        self.control_frequency = 50  # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Navigation parameters
        self.linear_vel_limit = 1.0  # m/s
        self.angular_vel_limit = 1.0  # rad/s
        self.arrival_tolerance = 0.2  # meters
        self.angle_tolerance = 0.1    # radians
        
        # Control gains
        self.kp_linear = 1.0  # Proportional gain for linear velocity
        self.kp_angular = 2.0  # Proportional gain for angular velocity
        
        # Path planning (simplified for simulation)
        self.current_plan = []
        
        # Create control timer
        self.control_timer = self.create_timer(self.dt, self.navigation_control_loop)
        
        # Performance monitoring
        self.control_loop_times = []
        
        self.get_logger().info('Isaac Navigation Stack Simulator initialized')

    def goal_callback(self, msg):
        """Handle navigation goal requests"""
        # Extract goal pose
        self.target_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            self.quaternion_to_yaw(msg.pose.orientation)
        ])
        
        self.navigation_state = 'planning'
        self.get_logger().info(f'New navigation goal received: ({self.target_pose[0]:.2f}, {self.target_pose[1]:.2f})')
        
        # Simple path planning (in reality, this would use navigation stack)
        self.plan_simple_path()
    
    def plan_simple_path(self):
        """Simple path planning for simulation"""
        if self.target_pose is None:
            return
        
        # Create a simple path from current position to target
        # In a real Isaac Navigation implementation, this would use more sophisticated planners
        current_pos = self.robot_pose[:2]
        target_pos = self.target_pose[:2]
        
        # Simple straight-line path
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        # Generate path points
        steps = max(1, int(distance / 0.5))  # 0.5m between waypoints
        path = []
        
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            point = current_pos + t * direction
            path.append(np.array([point[0], point[1], 0.0]))  # Adding theta=0 as placeholder
        
        self.current_plan = path
        self.navigation_state = 'executing'
        
        # Publish the path for visualization
        self.publish_path()
        
        self.get_logger().info(f'Simple path planned with {len(path)} waypoints')

    def velocity_command_callback(self, msg):
        """Handle velocity commands from navigation stack"""
        # For simulation, we just log the command
        self.get_logger().debug(f'Velocity command: linear={msg.linear.x:.2f}, angular={msg.angular.z:.2f}')
    
    def navigation_control_loop(self):
        """Main navigation control loop"""
        start_time = time.time()
        
        if self.navigation_state == 'idle':
            # Publish current pose but don't move
            self.publish_odometry()
            self.publish_tf()
        
        elif self.navigation_state == 'executing':
            if self.target_pose is not None:
                # Calculate control commands
                cmd_vel = self.calculate_navigation_command()
                
                # Apply commands to simulated robot
                self.apply_velocity_commands(cmd_vel)
                
                # Publish odometry and TF
                self.publish_odometry()
                self.publish_tf()
                
                # Check if reached goal
                distance = np.linalg.norm(self.robot_pose[:2] - self.target_pose[:2])
                angle_diff = abs(self.robot_pose[2] - self.target_pose[2])
                
                if distance < self.arrival_tolerance and angle_diff < self.angle_tolerance:
                    self.navigation_state = 'reached_goal'
                    self.get_logger().info('Navigation goal reached!')
                    self.publish_goal_reached()
            else:
                # No target, stay idle
                self.navigation_state = 'idle'
        
        elif self.navigation_state == 'reached_goal':
            # Stop robot and stay at goal
            stop_cmd = Twist()
            self.apply_velocity_commands(stop_cmd)
            self.publish_odometry()
            self.publish_tf()
        
        # Record timing
        elapsed_time = time.time() - start_time
        self.control_loop_times.append(elapsed_time)
    
    def calculate_navigation_command(self):
        """Calculate navigation commands to reach target"""
        cmd = Twist()
        
        # Current position vs target
        target_pos = self.target_pose[:2]
        current_pos = self.robot_pose[:2]
        
        # Direction to target
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:  # If not very close to target
            # Normalize direction vector
            direction_norm = direction / distance
            
            # Calculate desired heading
            desired_heading = np.arctan2(direction[1], direction[0])
            
            # Calculate heading error
            heading_error = desired_heading - self.robot_pose[2]
            
            # Normalize to [-π, π]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Proportional control
            cmd.linear.x = min(self.linear_vel_limit, distance * self.kp_linear)
            cmd.angular.z = heading_error * self.kp_angular
            
            # Apply velocity limits
            cmd.linear.x = np.clip(cmd.linear.x, -self.linear_vel_limit, self.linear_vel_limit)
            cmd.angular.z = np.clip(cmd.angular.z, -self.angular_vel_limit, self.angular_vel_limit)
        
        return cmd

    def apply_velocity_commands(self, cmd_vel):
        """Apply velocity commands to simulated robot"""
        # Update robot state using simple kinematic model
        linear_vel = cmd_vel.linear.x
        angular_vel = cmd_vel.angular.z
        
        # Update orientation
        self.robot_pose[2] += angular_vel * self.dt
        self.robot_pose[2] = ((self.robot_pose[2] + np.pi) % (2 * np.pi)) - np.pi  # Normalize to [-π, π]
        
        # Update position based on new orientation
        self.robot_pose[0] += linear_vel * np.cos(self.robot_pose[2]) * self.dt
        self.robot_pose[1] += linear_vel * np.sin(self.robot_pose[2]) * self.dt
        
        # Update velocity state
        self.robot_velocity[0] = linear_vel
        self.robot_velocity[1] = angular_vel

    def publish_odometry(self):
        """Publish odometry information"""
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        
        # Position
        odom.pose.pose.position.x = float(self.robot_pose[0])
        odom.pose.pose.position.y = float(self.robot_pose[1])
        odom.pose.pose.position.z = 0.0
        
        # Convert orientation from theta to quaternion
        quat = self.yaw_to_quaternion(self.robot_pose[2])
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        
        # Velocity
        odom.twist.twist.linear.x = float(self.robot_velocity[0])
        odom.twist.twist.angular.z = float(self.robot_velocity[1])
        
        self.odom_pub.publish(odom)

    def publish_tf(self):
        """Publish transform from odom to base_link"""
        t = TransformStamped()
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = float(self.robot_pose[0])
        t.transform.translation.y = float(self.robot_pose[1])
        t.transform.translation.z = 0.0
        
        quat = self.yaw_to_quaternion(self.robot_pose[2])
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def publish_path(self):
        """Publish the current plan for visualization"""
        if not self.current_plan:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'
        
        for waypoint in self.current_plan:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(waypoint[0])
            pose_stamped.pose.position.y = float(waypoint[1])
            pose_stamped.pose.position.z = 0.0
            
            # Simple orientation pointing toward next waypoint
            if np.array_equal(waypoint, self.current_plan[-1]):  # Last waypoint
                quat = self.yaw_to_quaternion(0.0)
            else:
                next_idx = self.current_plan.index(waypoint) + 1
                if next_idx < len(self.current_plan):
                    target = self.current_plan[next_idx][:2]
                    current = waypoint[:2]
                    angle = np.arctan2(target[1] - current[1], target[0] - current[0])
                    quat = self.yaw_to_quaternion(angle)
                else:
                    quat = self.yaw_to_quaternion(0.0)
            
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)

    def publish_goal_reached(self):
        """Publish goal reached notification"""
        self.get_logger().info('Goal reached successfully')

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        sinr_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosr_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return np.arctan2(sinr_cosp, cosr_cosp)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return [0.0, 0.0, sy, cy]

class IsaacManipulationControllerSimulator(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_controller_simulator')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.cartesian_command_pub = self.create_publisher(PoseStamped, '/cartesian_command', 10)
        self.gripper_command_pub = self.create_publisher(Bool, '/gripper_command', 10)
        
        # Subscribers
        self.cartesian_goal_sub = self.create_subscription(
            PoseStamped, '/cartesian_goal', self.cartesian_goal_callback, 10)
        
        # Robot state
        self.arm_joints = {
            'shoulder_pan_joint': 0.0,
            'shoulder_lift_joint': 0.0,
            'elbow_joint': 0.0,
            'wrist_1_joint': 0.0,
            'wrist_2_joint': 0.0,
            'wrist_3_joint': 0.0
        }
        
        self.gripper_open = True
        self.arm_state = 'idle'  # idle, planning, moving_to_waypoints, executing_task
        self.current_cartesian_pose = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0])  # x,y,z,rx,ry,rz
        
        # Control parameters
        self.control_frequency = 100  # Hz
        self.dt = 1.0 / self.control_frequency
        self.joint_velocity_limits = 0.5  # rad/s
        
        # Create control timer
        self.control_timer = self.create_timer(self.dt, self.manipulation_control_loop)
        
        self.get_logger().info('Isaac Manipulation Controller Simulator initialized')

    def cartesian_goal_callback(self, msg):
        """Handle Cartesian pose goal requests"""
        # Extract goal pose
        goal_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        # Convert quaternion to Euler angles for simulation
        r = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, 
                         msg.pose.orientation.z, msg.pose.orientation.w])
        euler = r.as_euler('xyz')
        
        self.current_cartesian_pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            euler[0], euler[1], euler[2]
        ])
        
        self.arm_state = 'moving_to_waypoints'
        self.get_logger().info(f'Cartesian goal received: ({goal_pose[0]:.3f}, {goal_pose[1]:.3f}, {goal_pose[2]:.3f})')
    
    def manipulation_control_loop(self):
        """Main manipulation control loop"""
        if self.arm_state == 'idle':
            # Publish current joint states
            self.publish_joint_states()
        
        elif self.arm_state == 'moving_to_waypoints':
            # Simulate movement to waypoints
            # In a real Isaac Manipulation implementation, this would call the actual manipulation stack
            
            # Publish joint states with updated positions (simplified simulation)
            self.apply_inverse_kinematics()
            self.publish_joint_states()
            
            # Check if reached target (simplified - just simulate movement)
            self.arm_state = 'idle'  # In simulation, assume immediate movement
            self.get_logger().info('Cartesian pose reached')
    
    def apply_inverse_kinematics(self):
        """Simplified inverse kinematics for simulation"""
        # This is a simplified simulation of IK
        # In real Isaac Manipulation, this would use advanced IK solvers
        
        # Update joint positions based on desired Cartesian pose
        # This is a very simplified representation
        self.arm_joints['shoulder_pan_joint'] = self.current_cartesian_pose[0] * 0.5
        self.arm_joints['shoulder_lift_joint'] = self.current_cartesian_pose[1] * 0.3
        self.arm_joints['elbow_joint'] = self.current_cartesian_pose[2] * 0.2
        self.arm_joints['wrist_1_joint'] = self.current_cartesian_pose[3]
        self.arm_joints['wrist_2_joint'] = self.current_cartesian_pose[4]
        self.arm_joints['wrist_3_joint'] = self.current_cartesian_pose[5]

    def publish_joint_states(self):
        """Publish joint state information"""
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'
        
        for joint_name, joint_pos in self.arm_joints.items():
            joint_state.name.append(joint_name)
            joint_state.position.append(joint_pos)
            joint_state.velocity.append(0.0)  # Simplified: no velocity feedback
            joint_state.effort.append(0.0)   # Simplified: no effort feedback
        
        self.joint_state_pub.publish(joint_state)

class IsaacROSControlSystem:
    def __init__(self):
        self.navigation_simulator = None
        self.manipulation_simulator = None
        
    def start_systems(self):
        """Start both navigation and manipulation simulation systems"""
        rclpy.init()
        
        self.navigation_simulator = IsaacNavigationStackSimulator()
        self.manipulation_simulator = IsaacManipulationControllerSimulator()
        
        try:
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(self.navigation_simulator)
            executor.add_node(self.manipulation_simulator)
            
            self.navigation_simulator.get_logger().info('Isaac ROS Control Systems running...')
            executor.spin()
            
        except KeyboardInterrupt:
            self.navigation_simulator.get_logger().info('Shutting down Isaac ROS Control Systems')
        finally:
            self.navigation_simulator.destroy_node()
            self.manipulation_simulator.destroy_node()
            rclpy.shutdown()

def run_control_simulation():
    """Run the Isaac ROS control system simulation"""
    control_system = IsaacROSControlSystem()
    control_system.start_systems()

def main():
    run_control_simulation()

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate navigation accuracy and path planning effectiveness
- Assess manipulation control precision
- Test system integration and coordination
- Monitor resource utilization and performance

## Simulation 4: AI Model Deployment and Performance

### Objective
Deploy and evaluate AI models in Isaac environment, focusing on performance optimization and real-time constraints.

### Setup
1. Configure AI models for Isaac deployment
2. Optimize models for edge GPU deployment
3. Integrate AI with perception and control systems
4. Validate performance against real-time requirements

### Implementation

```python
# isaac_ai_deployment_sim.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

@dataclass
class DeploymentConfig:
    """Configuration for AI model deployment"""
    model_path: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    precision: str = 'fp32'  # fp32, fp16, int8
    batch_size: int = 1
    max_latency: float = 0.05  # 50ms
    min_throughput: float = 20  # FPS
    device: str = 'cuda:0'

class ModelOptimizer:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.original_model = None
        self.optimized_model = None
        self.is_quantized = False
    
    def load_model(self):
        """Load the original model"""
        try:
            # In a real implementation, load the saved model
            # model = torch.jit.load(self.config.model_path)
            # self.original_model = model
            # self.original_model.eval()
            
            # For this simulation, create a mock model
            input_features = np.prod(self.config.input_shape[1:])
            output_features = np.prod(self.config.output_shape[1:])
            
            self.original_model = nn.Sequential(
                nn.Linear(input_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_features)
            ).eval()
            
            print(f"Mock model loaded with input shape {self.config.input_shape} and output shape {self.config.output_shape}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def optimize_for_tensorrt(self):
        """Optimize model for TensorRT deployment"""
        if self.original_model is None:
            print("Model not loaded, cannot optimize")
            return False
        
        try:
            # Create dummy input
            dummy_input = torch.randn(self.config.batch_size, *self.config.input_shape[1:]).cuda()
            
            # Compile with Torch-TensorRT
            compilation_args = {
                "inputs": [torch_tensorrt.Input(
                    shape=[self.config.batch_size, *self.config.input_shape[1:]]
                )],
                "enabled_precisions": {torch.float} if self.config.precision == 'fp32' else {torch.float, torch.half},
                "refit_enabled": True,
                "debug": False
            }
            
            self.optimized_model = torch_tensorrt.compile(
                self.original_model.cuda(),
                **compilation_args
            )
            
            print(f"Model optimized for TensorRT with {self.config.precision} precision")
            return True
            
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            # Fallback to original model if optimization fails
            self.optimized_model = self.original_model
            return False
    
    def quantize_model(self):
        """Apply quantization to reduce model size and improve performance"""
        if self.original_model is None:
            print("Model not loaded, cannot quantize")
            return False
        
        try:
            # Create quantized version
            self.original_model = self.original_model.cpu().eval()
            quantized_model = torch.quantization.quantize_dynamic(
                self.original_model, {nn.Linear}, dtype=torch.qint8
            )
            
            self.optimized_model = quantized_model
            self.is_quantized = True
            
            print(f"Model quantized to INT8. Size reduction: ~75%")
            return True
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return False

class AIModelDeploymentSimulator:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.optimizer = ModelOptimizer(config)
        self.model = None
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Performance monitoring
        self.inference_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        self.power_consumption = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Threading for performance measurement
        self.monitoring_lock = threading.Lock()
        
        # Initialize model
        self.initialize_deployment()
    
    def initialize_deployment(self):
        """Initialize the model deployment"""
        print("Initializing AI model deployment...")
        
        # Load model
        if not self.optimizer.load_model():
            print("Failed to load model, stopping deployment")
            return
        
        # Optimize model
        if self.config.precision.startswith('fp'):  # Use TensorRT for floating point
            if not self.optimizer.optimize_for_tensorrt():
                print("TensorRT optimization failed, using original model")
                self.model = self.optimizer.original_model
            else:
                self.model = self.optimizer.optimized_model
        else:  # Use quantization for integer types
            if not self.optimizer.quantize_model():
                print("Quantization failed, using original model")
                self.model = self.optimizer.original_model
            else:
                self.model = self.optimizer.optimized_model
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model deployment initialized successfully")
    
    def simulate_inference(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Simulate model inference with performance measurement"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Ensure correct input shape and device
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3:  # Add batch dimension if missing
            input_tensor = input_tensor.unsqueeze(0)
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            if self.is_quantized:
                # Quantized model needs different handling
                output = self.model(input_tensor.float())  # Convert to float for quantized model
            else:
                output = self.model(input_tensor)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Record performance metrics
        with self.monitoring_lock:
            self.inference_times.append(inference_time)
            
            # Simulate power consumption (in watts)
            power = 1.0 + 0.5 * (inference_time / self.config.max_latency)  # More inference time = more power
            self.power_consumption.append(power)
            
            # Simulate memory usage (in MB)
            mem_usage = 200 + np.random.uniform(-20, 20)  # Base memory + noise
            self.memory_usage.append(mem_usage)
        
        return output, inference_time
    
    def run_performance_test(self, num_inferences: int = 1000, input_generator=None):
        """Run comprehensive performance test"""
        print(f"Running performance test with {num_inferences} inferences...")
        
        # Default input generator if none provided
        if input_generator is None:
            def input_generator():
                return torch.randn(*self.config.input_shape).to(self.device)
        
        # Warmup inferences
        print("Warming up model...")
        for _ in range(10):
            dummy_input = input_generator()
            _, _ = self.simulate_inference(dummy_input)
        
        # Main test
        inference_times = []
        outputs = []
        
        print("Running performance test...")
        for i in range(num_inferences):
            input_tensor = input_generator()
            output, inf_time = self.simulate_inference(input_tensor)
            
            inference_times.append(inf_time)
            outputs.append(output)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_inferences} inferences")
        
        # Analyze results
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        p95_latency = np.percentile(inference_times, 95)
        
        print(f"\nPerformance Results:")
        print(f"  Average inference time: {avg_time:.6f}s ({avg_fps:.2f} FPS)")
        print(f"  Standard deviation: {std_time:.6f}s")
        print(f"  95th percentile latency: {p95_latency:.6f}s")
        print(f"  Min/Max latency: {min(inference_times):.6f}s / {max(inference_times):.6f}s")
        
        # Check requirements
        meets_latency = avg_time <= self.config.max_latency
        meets_throughput = avg_fps >= self.config.min_throughput
        
        print(f"\nRequirement Compliance:")
        print(f"  Latency requirement (<{self.config.max_latency}s): {'✓' if meets_latency else '✗'}")
        print(f"  Throughput requirement (>{self.config.min_throughput} FPS): {'✓' if meets_throughput else '✗'}")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'avg_fps': avg_fps,
            'p95_latency': p95_latency,
            'min_max': (min(inference_times), max(inference_times)),
            'meets_latency': meets_latency,
            'meets_throughput': meets_throughput,
            'inference_times': inference_times
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        with self.monitoring_lock:
            if not self.inference_times:
                return {'error': 'No data collected yet'}
            
            recent_times = list(self.inference_times)[-100:]  # Last 100 inferences
            avg_time = np.mean(recent_times)
            std_time = np.std(recent_times) if len(recent_times) > 1 else 0
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            p90_latency = np.percentile(recent_times, 90) if len(recent_times) > 1 else avg_time
            
            avg_power = np.mean(self.power_consumption) if self.power_consumption else 0
            avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        
        return {
            'avg_inference_time': float(avg_time),
            'std_inference_time': float(std_time),
            'avg_fps': float(avg_fps),
            'p90_latency': float(p90_latency),
            'avg_power_consumption': float(avg_power),
            'avg_memory_usage': float(avg_memory),
            'sample_count': len(recent_times)
        }
    
    def visualize_performance(self, results: Dict):
        """Create visualization of performance results"""
        if 'inference_times' not in results:
            print("No inference times data to visualize")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Inference time distribution
        plt.subplot(2, 3, 1)
        plt.hist(results['inference_times'], bins=50, alpha=0.7, color='skyblue')
        plt.axvline(results['avg_time'], color='red', linestyle='--', label=f"Avg: {results['avg_time']:.6f}s")
        plt.axvline(self.config.max_latency, color='orange', linestyle='--', label=f"Target: {self.config.max_latency}s")
        plt.xlabel('Inference Time (s)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        
        # Plot 2: Inference time over sequence
        plt.subplot(2, 3, 2)
        times = results['inference_times']
        plt.plot(times, alpha=0.7, color='green')
        plt.axhline(results['avg_time'], color='red', linestyle='--', label=f"Avg: {results['avg_time']:.6f}s")
        plt.axhline(self.config.max_latency, color='orange', linestyle='--', label=f"Target: {self.config.max_latency}s")
        plt.xlabel('Inference Sequence')
        plt.ylabel('Inference Time (s)')
        plt.title('Inference Time Over Sequence')
        plt.legend()
        
        # Plot 3: Performance requirements
        plt.subplot(2, 3, 3)
        req_labels = ['Latency', 'Throughput']
        req_status = [results['meets_latency'], results['meets_throughput']]
        req_colors = ['green' if status else 'red' for status in req_status]
        
        bars = plt.bar(req_labels, [1 if s else 0 for s in req_status], color=req_colors)
        plt.ylim(0, 1.2)
        plt.title('Requirement Compliance')
        
        # Add text labels
        for bar, status in zip(bars, req_status):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    'Met' if status else 'Not Met', ha='center', va='bottom')
        
        # Plot 4: Performance vs requirements
        plt.subplot(2, 3, 4)
        metrics = ['Latency (s)', 'Throughput (FPS)']
        actual = [results['avg_time'], results['avg_fps']]
        requirements = [self.config.max_latency, self.config.min_throughput]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, actual, width, label='Actual', alpha=0.8)
        plt.bar(x + width/2, requirements, width, label='Required', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Performance vs Requirements')
        plt.xticks(x, metrics)
        plt.legend()
        
        # Plot 5: Resource consumption
        plt.subplot(2, 3, 5)
        # We'll simulate resource data for this example
        time_points = np.arange(100)
        power_usage = np.random.normal(2.5, 0.2, 100)  # Simulated power usage
        memory_usage = np.random.normal(800, 50, 100)  # Simulated memory usage (MB)
        
        plt.plot(time_points, power_usage, label='Power Consumption (W)', color='red', alpha=0.7)
        plt.plot(time_points, memory_usage, label='Memory Usage (MB)', color='blue', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Resource Usage')
        plt.title('Resource Consumption Over Time')
        plt.legend()
        
        # Plot 6: Efficiency metrics
        plt.subplot(2, 3, 6)
        efficiency = [results['avg_fps'], 1/results['avg_time'], 1000/results['avg_time']]  # FPS, Inferences/sec, Inferences/second*1000
        eff_labels = ['FPS', '1/Time', 'Scaled Perf']
        
        plt.bar(eff_labels, efficiency, color=['orange', 'purple', 'brown'], alpha=0.7)
        plt.ylabel('Efficiency Metric')
        plt.title('Efficiency Metrics')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.suptitle(f'AI Model Deployment Performance: Precision={self.config.precision}, Batch={self.config.batch_size}', 
                     y=1.02, fontsize=14)
        plt.show()

class AIDeploymentSystem:
    def __init__(self, configs: List[DeploymentConfig]):
        self.configs = configs
        self.simulators = {}
        self.performance_results = {}
        
    def deploy_models(self):
        """Deploy all configured models"""
        print("Starting AI model deployments...")
        
        for i, config in enumerate(self.configs):
            print(f"\nDeploying model {i+1}/{len(self.configs)}: {config.model_path}")
            simulator = AIModelDeploymentSimulator(config)
            self.simulators[f"model_{i}"] = simulator
        
        print(f"\nSuccessfully deployed {len(self.simulators)} models")
    
    def run_comparative_analysis(self):
        """Run comparative performance analysis"""
        print("\nRunning comparative analysis across different configurations...")
        
        results = {}
        
        for name, simulator in self.simulators.items():
            print(f"\nTesting {name}...")
            
            # Generate dummy inputs for testing
            def input_gen():
                return torch.randn(*simulator.config.input_shape).to(simulator.device)
            
            # Run performance test
            result = simulator.run_performance_test(num_inferences=100, input_generator=input_gen)
            results[name] = result
        
        # Compare results
        print("\nComparative Analysis Results:")
        print("-" * 50)
        
        for name, result in results.items():
            config = [c for c in self.configs if f"model_{list(self.simulators.keys()).index(name)}" == name][0]
            print(f"{name} (Precision: {config.precision}):")
            print(f"  Avg FPS: {result['avg_fps']:.2f}")
            print(f"  Avg Latency: {result['avg_time']:.6f}s")
            print(f"  95th %ile: {result['p95_latency']:.6f}s")
            print(f"  Meets Requirements: Latency={result['meets_latency']}, Throughput={result['meets_throughput']}")
            print()
        
        # Visualize comparison
        self.visualize_comparative_results(results)
        
        return results
    
    def visualize_comparative_results(self, results: Dict):
        """Visualize comparative results"""
        names = list(results.keys())
        avg_fps = [results[name]['avg_fps'] for name in names]
        avg_times = [results[name]['avg_time'] for name in names]
        meets_reqs = [results[name]['meets_throughput'] for name in names]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # FPS comparison
        axes[0].bar(names, avg_fps, alpha=0.7, 
                   color=['green' if meets else 'red' for meets in meets_reqs])
        axes[0].set_title('Frames Per Second Comparison')
        axes[0].set_ylabel('FPS')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Latency comparison
        axes[1].bar(names, avg_times, alpha=0.7,
                   color=['green' if meets else 'red' for meets in meets_reqs])
        axes[1].axhline(y=0.05, color='orange', linestyle='--', label='Target Latency (50ms)')
        axes[1].set_title('Average Inference Time Comparison')
        axes[1].set_ylabel('Time (s)')
        axes[1].legend()
        axes[1].tick_params(axis='x', rotation=45)
        
        # Requirement compliance
        req_status = [1 if meets else 0 for meets in meets_reqs]
        axes[2].bar(names, req_status, alpha=0.7, 
                   color=['green' if meets else 'red' for meets in meets_reqs])
        axes[2].set_title('Requirement Compliance')
        axes[2].set_ylabel('Meets Throughput Req')
        axes[2].set_ylim(0, 1.2)
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add labels to requirement compliance chart
        for i, (name, meets) in enumerate(zip(names, meets_reqs)):
            axes[2].text(i, meets + 0.05, 'Yes' if meets else 'No', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle('Comparative Performance Analysis', y=1.02)
        plt.show()

def run_ai_deployment_simulation():
    """Run the AI deployment simulation"""
    print("Starting Isaac AI Deployment Simulation")
    
    # Define different deployment configurations to compare
    configs = [
        DeploymentConfig(
            model_path="./models/perception_fp32.pth",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 80),
            precision='fp32',
            batch_size=1,
            max_latency=0.05,
            min_throughput=20
        ),
        DeploymentConfig(
            model_path="./models/perception_fp16.pth",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 80),
            precision='fp16',
            batch_size=1,
            max_latency=0.03,
            min_throughput=30
        ),
        DeploymentConfig(
            model_path="./models/control_quantized.pth",
            input_shape=(1, 24),
            output_shape=(1, 2),
            precision='int8',
            batch_size=1,
            max_latency=0.01,
            min_throughput=100
        )
    ]
    
    # Create deployment system
    deployment_system = AIDeploymentSystem(configs)
    
    # Deploy models
    deployment_system.deploy_models()
    
    # Run comparative analysis
    results = deployment_system.run_comparative_analysis()
    
    print("\nAI Deployment Simulation Completed")
    return results

def main():
    results = run_ai_deployment_simulation()
    
    # Print summary
    print("\n" + "="*50)
    print("DEPLOYMENT SIMULATION SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Average FPS: {metrics['avg_fps']:.2f}")
        print(f"  Average Latency: {metrics['avg_time']:.6f}s")
        print(f"  Requirement Compliance: {'PASS' if metrics['meets_throughput'] and metrics['meets_latency'] else 'FAIL'}")

if __name__ == '__main__':
    main()
```

### Analysis
- Evaluate model deployment performance across different configurations
- Assess the effectiveness of optimization techniques
- Compare performance between different precision settings
- Analyze resource utilization and efficiency

## Chapter Summary

This simulation module provided comprehensive hands-on experience with the NVIDIA Isaac Platform for AI-powered robotics applications. Students worked with Isaac Sim for generating synthetic data and training AI models, implemented GPU-accelerated perception pipelines using Isaac ROS, developed control systems with navigation and manipulation capabilities, and evaluated AI model deployment performance. The simulations emphasized practical implementation skills needed to develop and validate AI systems for real-world robotics applications using the Isaac Platform's specialized tools and optimizations.

## Key Terms
- Isaac Sim
- Isaac ROS
- Isaac Navigation
- Isaac Manipulation
- TensorRT Optimization
- Domain Randomization
- GPU-Accelerated Perception
- AI Model Deployment

## References
- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Isaac ROS Documentation: https://docs.nvidia.com/isaac/ros/
- Isaac Navigation Documentation
- Isaac Manipulation Documentation