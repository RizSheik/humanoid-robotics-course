# Module 4: Practical Lab - Isaac Platform for Perception and Control

## Lab Overview

This practical lab provides hands-on experience with the NVIDIA Isaac Platform for implementing AI-powered perception and control systems. Students will work with Isaac Sim for AI training, Isaac ROS for perception and control pipelines, and integrate these into complete robotic systems. The lab emphasizes practical implementation of advanced AI techniques for real-world robotics applications.

### Learning Objectives

After completing this lab, students will be able to:
1. Set up and configure Isaac Sim for AI training and validation
2. Implement perception pipelines using Isaac ROS packages
3. Design and deploy control systems with AI integration
4. Integrate perception and control components into complete robotic systems
5. Deploy and validate AI models on edge devices using Isaac Platform
6. Debug and optimize AI robotics systems

### Required Software/Tools

- **NVIDIA Isaac Sim**: Version 2023.1 or later
- **Isaac ROS Packages**: Latest stable release
- **ROS 2 Humble Hawksbill**: With Isaac ROS extensions
- **NVIDIA GPU**: With CUDA 11.4+ support (RTX 3070 or equivalent)
- **Python 3.11+**: For implementing custom nodes
- **Docker Environment**: For Isaac Sim and Isaac ROS deployment
- **Isaac Navigation and Manipulation Apps**: Pre-built applications

### Lab Duration

This lab is designed for 18-20 hours of hands-on work, typically spread over 3-4 weeks with 6 hours per week.

## Lab 1: Isaac Sim Setup and AI Training Environment

### Objective
Set up Isaac Sim and create an AI training environment for a mobile robot navigation task.

### Setup Instructions

1. Install Isaac Sim using Docker:
```bash
# Pull Isaac Sim image
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

# Run Isaac Sim container
docker run --gpus all -it --rm \
  --network=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="${PWD}/projects:/isaac-sim/projects" \
  --volume="${HOME}/.nvidia-omniverse:/root/.nvidia-omniverse" \
  --volume="${HOME}/.cache/ov:/root/.cache/ov" \
  --device=/dev/dri:/dev/dri \
  --name="isaac-sim" \
  nvcr.io/nvidia/isaac-sim:2023.1.1
```

2. Verify Isaac Sim installation by launching:
```bash
./isaac-sim-launch.sh
```

### Implementation Steps

1. **Create a new Isaac Sim project** for mobile robot navigation:
   - Use the Isaac Sim Project Wizard or manually create a new project
   - Select appropriate scene template (e.g., Office, Warehouse)
   - Configure physics properties with realistic parameters

2. **Set up robot model** (using Carter robot as example):
   - Add a differential drive robot to the scene
   - Configure robot with accurate kinematic and dynamic properties
   - Add sensors (camera, IMU, LIDAR) with realistic parameters

3. **Design training environments**:
   - Create multiple environments with varying complexity
   - Include different lighting conditions and textures
   - Add dynamic objects for training robustness

4. **Generate synthetic training data**:
   - Implement data collection pipeline
   - Generate labeled perception data (images, depth, semantic segmentation)
   - Create navigation scenarios for RL training

### Code Template

```python
# lab1_isaac_sim_setup.py
import omni
from pxr import Gf, Usd, UsdGeom
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim
from omni.isaac.sensor import _sensor as _sensor
import numpy as np
import carb

class Lab1IsaacSimEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.robot = None
        self.robot_name = "carter"
        self.setup_complete = False
        
    def setup_environment(self):
        """Set up the initial Isaac Sim environment"""
        print("Setting up Isaac Sim environment...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        self._add_lighting()
        
        # Load Carter robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        # Add Carter robot
        robot_path = f"/World/{self.robot_name}"
        carter_path = assets_root_path + "/Isaac/Robots/Carter/carter_nucleus.usd"
        add_reference_to_stage(usd_path=carter_path, prim_path=robot_path)
        
        # Create robot view for control
        from omni.isaac.core.articulations import ArticulationView
        self.robot = ArticulationView(
            prim_path=robot_path,
            name="carter_view",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.robot)
        
        # Add objects for interaction
        self._add_training_objects()
        
        self.setup_complete = True
        print("Environment setup complete!")
    
    def _add_lighting(self):
        """Add realistic lighting to the environment"""
        # Add dome light for ambient illumination
        dome_light = create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            attributes={"color": (0.2, 0.2, 0.2), "intensity": 500}
        )
        
        # Add distant light for shadows
        distant_light = create_prim(
            prim_path="/World/DistantLight",
            prim_type="DistantLight", 
            attributes={
                "color": (0.9, 0.9, 0.9), 
                "intensity": 500,
                "angle": 0.5
            }
        )
    
    def _add_training_objects(self):
        """Add objects for navigation training"""
        from omni.isaac.core.objects import DynamicCuboid
        
        # Add navigation targets
        target = DynamicCuboid(
            prim_path="/World/Target",
            name="target_cube",
            position=np.array([5.0, 0.0, 0.2]),
            size=np.array([0.2, 0.2, 0.2]),
            color=np.array([1.0, 0.0, 0.0])  # Red
        )
        self.world.scene.add(target)
        
        # Add obstacles
        obstacles = []
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
                color=np.array([0.5, 0.5, 0.5])  # Gray
            )
            self.world.scene.add(obstacle)
            obstacles.append(obstacle)
        
        print(f"Added target and {len(obstacles)} obstacles for training")
    
    def generate_training_scenarios(self):
        """Generate multiple training scenarios"""
        print("Generating training scenarios...")
        
        # For each scenario, randomize:
        # - Robot starting position
        # - Target position
        # - Obstacle positions
        # - Lighting conditions
        # - Floor textures
        
        scenarios = []
        for i in range(20):  # Generate 20 different scenarios
            scenario = self._create_scenario(f"scenario_{i}")
            scenarios.append(scenario)
        
        print(f"Generated {len(scenarios)} training scenarios")
        return scenarios
    
    def _create_scenario(self, name):
        """Create a single training scenario"""
        return {
            "name": name,
            "robot_start_pos": np.array([
                np.random.uniform(-3.0, 3.0), 
                np.random.uniform(-3.0, 3.0), 
                0.5
            ]),
            "target_pos": np.array([
                np.random.uniform(2.0, 6.0),
                np.random.uniform(-2.0, 2.0),
                0.2
            ]),
            "obstacles": [
                {
                    "pos": np.array([
                        np.random.uniform(0.5, 5.0),
                        np.random.uniform(-2.5, 2.5),
                        0.2
                    ]),
                    "size": np.array([0.3, 0.3, 0.4])
                } for _ in range(np.random.randint(3, 8))
            ]
        }
    
    def run_training_simulation(self, num_steps=1000, scenario=None):
        """Run simulation for AI training"""
        if not self.setup_complete:
            print("Environment not set up. Run setup_environment first.")
            return
        
        self.world.reset()
        
        if scenario:
            self._apply_scenario(scenario)
        
        print(f"Running training simulation for {num_steps} steps...")
        
        for step in range(num_steps):
            # In a real implementation, you would:
            # 1. Get robot state and sensor data
            # 2. Apply AI control policy
            # 3. Calculate reward
            # 4. Store experience for training
            
            # For this lab, we'll just step the simulation
            self.world.step(render=True)
            
            if step % 100 == 0:
                robot_pos, robot_orn = self.robot.get_world_poses()
                print(f"Step {step}: Robot at {robot_pos[0]}")
    
    def _apply_scenario(self, scenario):
        """Apply scenario configuration to environment"""
        # Move robot to start position
        self.robot.set_world_poses(positions=scenario["robot_start_pos"].reshape(1, 3))
        
        # Move target to position (this would be implemented based on your scene setup)
        target_prim = get_prim_at_path("/World/Target")
        # Implementation to move target would go here
        
        # Move obstacles
        for i, obstacle_data in enumerate(scenario["obstacles"]):
            obstacle_prim = get_prim_at_path(f"/World/Obstacle_{i}")
            if obstacle_prim.IsValid():
                # Move obstacle to position (implementation depends on your setup)
                pass
    
    def cleanup(self):
        """Clean up the environment"""
        if hasattr(self, 'world'):
            self.world.clear()
        print("Environment cleaned up.")

def main():
    """Main entry point for Lab 1"""
    env = Lab1IsaacSimEnvironment()
    
    try:
        # Set up the environment
        env.setup_environment()
        
        # Generate training scenarios
        scenarios = env.generate_training_scenarios()
        
        # Run a short simulation with a scenario
        if scenarios:
            env.run_training_simulation(num_steps=500, scenario=scenarios[0])
        
        print("Lab 1 completed successfully!")
        
    except Exception as e:
        print(f"Error in Lab 1: {e}")
        carb.log_error(f"Error in Lab 1: {e}")
    finally:
        env.cleanup()

if __name__ == "__main__":
    main()
```

### Analysis and Documentation

Document your results in the lab report:
1. Verify Isaac Sim installation and basic functionality
2. Record the configuration of your robot model (sensors, actuator limits, etc.)
3. Document the variety of training scenarios created
4. Evaluate the realism of your simulation environment

## Lab 2: Isaac ROS Perception Pipeline

### Objective
Implement a perception pipeline using Isaac ROS packages for object detection and environment understanding.

### Setup

1. Verify Isaac ROS installation:
```bash
# Check if Isaac ROS packages are installed
ros2 pkg list | grep isaac
```

2. Source the Isaac ROS overlay:
```bash
source /opt/ros/humble/setup.bash
source /usr/local/share/isaac_ros_common/setup.bash
```

### Implementation Steps

1. **Set up camera calibration**:
   - Use Isaac ROS Image Pipeline for camera calibration
   - Implement rectification and undistortion

2. **Implement object detection**:
   - Use Isaac ROS DNN Inference for object detection
   - Configure model for your specific objects

3. **Create sensor fusion**:
   - Combine data from camera, IMU, and LIDAR
   - Implement spatial and temporal alignment

4. **Implement tracking**:
   - Create object tracking pipeline
   - Associate detections across frames

### Code Template

```python
#!/usr/bin/env python3
# lab2_isaac_ros_perception.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import cv2
import torch

class Lab2IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('lab2_isaac_perception_pipeline')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.visualization_pub = self.create_publisher(Image, '/detection_visualization', 10)
        
        # Subscribers with approximate time synchronization
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.camera_info_sub = Subscriber(self, CameraInfo, '/camera/camera_info') 
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # Synchronize image and camera info
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.camera_info_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.camera_callback)
        
        # Internal state
        self.latest_imu = None
        self.latest_camera_info = None
        self.detection_model = None  # Will be loaded later
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Initialize detection model
        self._load_detection_model()
        
        self.get_logger().info('Isaac ROS Perception Pipeline initialized')

    def _load_detection_model(self):
        """Load object detection model"""
        # In a real implementation, you would load a TensorRT-optimized model
        # For this lab, we'll create a simple placeholder
        self.get_logger().info('Loading object detection model...')
        
        # Create a simple detection model (in practice, use Isaac ROS DNN Inference)
        # self.detection_model = torch_tensorrt.compile(loaded_model, ...)  
        
    def camera_callback(self, image_msg, camera_info_msg):
        """Process synchronized camera image and info"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            
            # Update camera parameters
            self.camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(camera_info_msg.d)
            
            # Run object detection
            detections = self._run_object_detection(cv_image)
            
            # Create and publish detections
            detection_array_msg = self._create_detection_array_msg(detections, image_msg.header)
            self.detection_pub.publish(detection_array_msg)
            
            # Create and publish visualization
            vis_image = self._create_visualization(cv_image, detections)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            vis_msg.header = image_msg.header
            self.visualization_pub.publish(vis_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {str(e)}')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu = msg
        
        # In a real system, you might use IMU data for:
        # - Sensor fusion
        # - Motion compensation
        # - State estimation
        self.get_logger().debug(f'IMU data received: linear_acceleration=({msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f})')

    def _run_object_detection(self, image):
        """Run object detection on image using Isaac ROS DNN Inference"""
        # In real Isaac ROS, this would call the actual detection node
        # For this example, we'll simulate detection
        
        # Convert image to tensor (in real implementation, use Isaac ROS)
        height, width = image.shape[:2]
        
        # Simulate detection results
        # In practice, you'd run the actual model
        simulated_detections = [
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.85,
                'bbox': [int(width*0.3), int(height*0.2), int(width*0.2), int(height*0.4)]  # [x, y, w, h]
            },
            {
                'class_id': 1,
                'class_name': 'chair',
                'confidence': 0.72,
                'bbox': [int(width*0.6), int(height*0.4), int(width*0.25), int(height*0.3)]
            }
        ]
        
        return simulated_detections

    def _create_detection_array_msg(self, detections, header):
        """Create Detection2DArray message from detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for detection in detections:
            if detection['confidence'] > 0.5:  # Confidence threshold
                detection_msg = Detection2D()
                
                # Bounding box center and size
                bbox = detection['bbox']
                detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2  # center_x
                detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2  # center_y
                detection_msg.bbox.size_x = bbox[2]  # width
                detection_msg.bbox.size_y = bbox[3]  # height
                
                # Classification result
                from vision_msgs.msg import ObjectHypothesisWithPose
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(detection['class_id'])
                hypothesis.hypothesis.score = detection['confidence']
                
                detection_msg.results.append(hypothesis)
                detection_array.detections.append(detection_msg)
        
        return detection_array

    def _create_visualization(self, image, detections):
        """Create visualization of detections on image"""
        vis_image = image.copy()
        
        for detection in detections:
            if detection['confidence'] > 0.5:  # Only visualize confident detections
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Draw bounding box
                start_point = (x, y)
                end_point = (x + w, y + h)
                color = (0, 255, 0)  # Green
                thickness = 2
                cv2.rectangle(vis_image, start_point, end_point, color, thickness)
                
                # Draw label and confidence
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                label_position = (x, y - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                cv2.putText(vis_image, label, label_position, font, font_scale, color, 1)
        
        return vis_image

class IsaacPerceptionValidator(Node):
    """Validate perception pipeline performance"""
    def __init__(self):
        super().__init__('isaac_perception_validator')
        
        # Subscribe to perception output
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_validation_callback, 10
        )
        
        # Performance metrics
        self.detection_counts = []
        self.processing_times = []
        self.confidence_scores = []
        
        # Timer for periodic evaluation
        self.eval_timer = self.create_timer(5.0, self.periodic_evaluation)
        
        self.get_logger().info('Perception Validator initialized')

    def detection_validation_callback(self, msg):
        """Validate incoming detections"""
        import time
        start_time = time.time()
        
        # Count detections
        num_detections = len(msg.detections)
        self.detection_counts.append(num_detections)
        
        # Collect confidence scores
        for detection in msg.detections:
            for result in detection.results:
                self.confidence_scores.append(result.hypothesis.score)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        self.get_logger().debug(f'Detections received: {num_detections}, Processing time: {processing_time:.4f}s')

    def periodic_evaluation(self):
        """Periodically evaluate perception performance"""
        if len(self.detection_counts) == 0:
            self.get_logger().info('No detections received yet')
            return
        
        # Calculate metrics
        avg_detections = sum(self.detection_counts) / len(self.detection_counts)
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        
        self.get_logger().info(
            f'Perception Performance - Avg Detections: {avg_detections:.2f}, '
            f'Avg Processing Time: {avg_processing_time:.4f}s, '
            f'Avg Confidence: {avg_confidence:.4f}'
        )
        
        # Reset for next period
        self.detection_counts = []
        self.processing_times = []
        self.confidence_scores = []

def main(args=None):
    rclpy.init(args=args)
    
    # Create perception pipeline and validator
    perception_pipeline = Lab2IsaacPerceptionPipeline()
    validator = IsaacPerceptionValidator()
    
    try:
        # Combine both nodes in a single executor
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(perception_pipeline)
        executor.add_node(validator)
        
        executor.spin()
    except KeyboardInterrupt:
        perception_pipeline.get_logger().info('Shutting down Isaac Perception Pipeline')
    finally:
        perception_pipeline.destroy_node()
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis and Documentation

Document your results:
1. Measure detection accuracy and performance
2. Evaluate the quality of sensor fusion
3. Assess real-time performance (FPS, latency)
4. Document any calibration or alignment issues

## Lab 3: Isaac ROS Control Systems

### Objective
Implement AI-powered control systems using Isaac ROS for navigation and manipulation tasks.

### Setup

1. Ensure Isaac ROS navigation and manipulation packages are installed:
```bash
sudo apt-get install ros-humble-isaac-ros-nav2-benchmarks
sudo apt-get install ros-humble-isaac-ros-manipulation
```

2. Set up robot control interface:
```bash
# Source ROS and Isaac ROS
source /opt/ros/humble/setup.bash
source /usr/local/share/isaac_ros_common/setup.bash
```

### Implementation Steps

1. **Implement navigation stack**:
   - Configure Isaac ROS Navigation with GPU acceleration
   - Set up costmaps and planners
   - Implement dynamic obstacle avoidance

2. **Integrate AI for navigation**:
   - Use Isaac ROS Perception output for navigation decisions
   - Implement learning-based planning
   - Add semantic navigation capabilities

3. **Set up manipulation control**:
   - Implement Isaac ROS manipulation pipeline
   - Integrate perception for object recognition and pose estimation
   - Create grasping and manipulation policies

4. **Implement high-level control**:
   - Create task planning system
   - Integrate multiple AI components
   - Implement human-robot interaction

### Code Template

```python
#!/usr/bin/env python3
# lab3_isaac_control_systems.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf2_ros

class IsaacNavigationController(Node):
    def __init__(self):
        super().__init__('isaac_navigation_controller')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        # TF2 broadcaster and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Robot state
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_velocity = np.array([0.0, 0.0])   # linear, angular
        self.target_pose = np.array([0.0, 0.0, 0.0])
        self.navigation_state = 'idle'  # idle, navigating, avoiding
        self.detected_objects = []
        
        # Control parameters
        self.linear_gain = 1.0
        self.angular_gain = 2.0
        self.collision_threshold = 0.5  # meters
        self.arrival_threshold = 0.2    # meters
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz
        
        self.get_logger().info('Isaac Navigation Controller initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        # Extract position
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        
        # Extract orientation
        quat = msg.pose.pose.orientation
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        euler = rot.as_euler('xyz')
        self.current_pose[2] = euler[2]  # Only care about yaw
        
        # Extract velocity
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.angular.z

    def detection_callback(self, msg):
        """Process object detections"""
        self.detected_objects = []
        
        for detection in msg.detections:
            # In a real implementation, you would need to convert 2D detections to 3D
            # using depth information or other methods
            
            obj_info = {
                'class': detection.results[0].hypothesis.class_id if detection.results else 'unknown',
                'confidence': detection.results[0].hypothesis.score if detection.results else 0.0,
                'bbox_center': (detection.bbox.center.x, detection.bbox.center.y),
                'bbox_size': (detection.bbox.size_x, detection.bbox.size_y)
            }
            
            self.detected_objects.append(obj_info)

    def set_navigation_goal(self, x, y, theta=0.0):
        """Set navigation goal"""
        self.target_pose = np.array([x, y, theta])
        self.navigation_state = 'navigating'
        self.get_logger().info(f'Navigation goal set: ({x}, {y}, {theta})')

    def control_loop(self):
        """Main navigation control loop"""
        if self.navigation_state == 'idle':
            # Stop the robot
            self._stop_robot()
            return
        
        if self.navigation_state == 'navigating':
            # Check for obstacles
            if self._detect_collision_risk():
                self.navigation_state = 'avoiding'
                self.get_logger().info('Collision risk detected, switching to avoidance mode')
            else:
                # Calculate navigation commands
                cmd_vel = self._compute_navigation_command()
                self.cmd_vel_pub.publish(cmd_vel)
                
                # Check if arrived
                distance = np.linalg.norm(self.current_pose[:2] - self.target_pose[:2])
                if distance < self.arrival_threshold:
                    self.navigation_state = 'arrived'
                    self._stop_robot()
                    self.get_logger().info('Destination reached!')

    def _detect_collision_risk(self):
        """Detect collision risk based on object detections"""
        for obj in self.detected_objects:
            if obj['confidence'] > 0.7:  # High confidence detection
                # This is simplified - in real implementation, you'd use depth data
                # to estimate distance to obstacles
                if obj['class'] in ['person', 'obstacle', 'furniture']:
                    # Assume obstacle is close enough to be risky
                    return True
        return False

    def _compute_navigation_command(self):
        """Compute navigation command to reach target"""
        cmd = Twist()
        
        # Calculate direction to target
        direction = self.target_pose[:2] - self.current_pose[:2]
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:  # If not already at target
            # Normalize direction
            direction_norm = direction / distance
            
            # Calculate desired heading
            desired_theta = np.arctan2(direction[1], direction[0])
            
            # Calculate heading error
            heading_error = desired_theta - self.current_pose[2]
            
            # Normalize angle to [-π, π]
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Proportional control
            cmd.linear.x = min(0.5, distance * self.linear_gain)  # Cap speed
            cmd.angular.z = heading_error * self.angular_gain
            
            # Adjust for robot kinematics if needed
            # For differential drive robots, adjust angular velocity based on linear velocity
            cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)  # Cap angular velocity
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        return cmd

    def _stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

class IsaacManipulationController(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_controller')
        
        # Publishers and subscribers for manipulation
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.object_detection_callback, 10)
        
        # Manipulation state
        self.detected_objects = []
        self.grasp_target = None
        self.manipulation_state = 'idle'  # idle, planning_grasp, executing_grasp, holding
        
        # Timer for manipulation control
        self.manip_timer = self.create_timer(0.1, self.manipulation_loop)
        
        self.get_logger().info('Isaac Manipulation Controller initialized')

    def object_detection_callback(self, msg):
        """Process object detections for manipulation"""
        # Process detections for graspable objects
        graspable_objects = []
        
        for detection in msg.detections:
            # Check if this is a graspable object (based on class, size, etc.)
            if detection.results and detection.results[0].hypothesis.score > 0.8:
                class_id = detection.results[0].hypothesis.class_id
                
                # For now, assume anything that's not background could be graspable
                if class_id in ['0', '1', '2']:  # These would be object categories
                    graspable_objects.append({
                        'bbox': detection.bbox,
                        'class': class_id,
                        'confidence': detection.results[0].hypothesis.score
                    })
        
        self.detected_objects = graspable_objects

    def manipulation_loop(self):
        """Main manipulation control loop"""
        if self.manipulation_state == 'idle':
            # Look for graspable objects
            if self.detected_objects:
                # Select highest confidence object to grasp
                best_obj = max(self.detected_objects, key=lambda o: o['confidence'])
                self.grasp_target = best_obj
                self.manipulation_state = 'planning_grasp'
                self.get_logger().info(f'Planning grasp for object {best_obj["class"]}')
        
        elif self.manipulation_state == 'planning_grasp':
            # In a real implementation, plan grasp trajectory
            # For this lab, just transition to execution
            self.manipulation_state = 'executing_grasp'
            self.get_logger().info('Executing grasp plan')
            
        elif self.manipulation_state == 'executing_grasp':
            # Execute grasp (in real implementation)
            # For this lab, just transition to holding
            self.manipulation_state = 'holding'
            self.get_logger().info('Object grasped successfully')
            
        elif self.manipulation_state == 'holding':
            # Hold object until commanded otherwise
            pass

def main(args=None):
    rclpy.init(args=args)
    
    # Create navigation and manipulation controllers
    nav_controller = IsaacNavigationController()
    manip_controller = IsaacManipulationController()
    
    try:
        # Set a navigation goal for testing
        nav_controller.set_navigation_goal(2.0, 2.0)
        
        # Combine nodes in single executor
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(nav_controller)
        executor.add_node(manip_controller)
        
        executor.spin()
    except KeyboardInterrupt:
        nav_controller.get_logger().info('Shutting down Isaac Control Systems')
    finally:
        nav_controller.destroy_node()
        manip_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Analysis and Documentation

Document your results:
1. Measure navigation performance (accuracy, time to goal, collision avoidance)
2. Evaluate manipulation success rates
3. Assess integration between perception and control
4. Document any tuning of control parameters needed

## Lab 4: AI Integration and Deployment

### Objective
Deploy and integrate the AI systems developed in previous labs onto an edge device, simulating real-world deployment conditions.

### Setup

1. Set up Jetson or other edge device:
```bash
# Install Isaac ROS on Jetson
sudo apt-get update
sudo apt-get install ros-humble-isaac-ros-common
```

2. Prepare models for deployment:
```bash
# Convert PyTorch models to TensorRT (simplified)
python3 -c "
import torch
import torch_tensorrt

# Load your trained model
model = torch.load('model.pth')
model.eval()

# Trace the model
traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

# Compile with TensorRT
compiled_model = torch_tensorrt.compile(
    traced_model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float, torch.half},
    refit_enabled=True,
    debug=False
)

# Save compiled model
torch.jit.save(compiled_model, 'compiled_model.ts')
"
```

### Implementation Steps

1. **Optimize models for edge deployment**:
   - Apply quantization
   - Optimize model architecture for real-time performance
   - Test on edge hardware

2. **Implement deployment pipeline**:
   - Create Docker containers for edge deployment
   - Set up remote monitoring and logging
   - Implement automatic model updates

3. **Validate deployment**:
   - Test performance in edge environment
   - Validate accuracy is maintained
   - Monitor resource usage

4. **Implement fallback mechanisms**:
   - Create safety fallbacks for AI failures
   - Implement graceful degradation
   - Add manual override capabilities

### Code Template

```python
# lab4_ai_deployment.py
import torch
import numpy as np
import time
import threading
from collections import deque
from dataclasses import dataclass
import json
import logging

@dataclass
class DeploymentConfig:
    """Configuration for AI model deployment"""
    model_path: str
    input_shape: tuple
    output_shape: tuple
    batch_size: int = 1
    precision: str = 'fp16'  # fp32, fp16, int8
    max_latency: float = 0.1  # seconds
    min_throughput: float = 10  # FPS

class ModelOptimizer:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.original_model = None
        self.optimized_model = None
    
    def load_model(self):
        """Load the original model"""
        self.original_model = torch.load(self.config.model_path)
        self.original_model.eval()
        print(f"Loaded model from {self.config.model_path}")
    
    def optimize_for_jetson(self):
        """Optimize model for Jetson deployment using TensorRT"""
        import torch_tensorrt
        
        # Trace the model
        dummy_input = torch.randn(self.config.batch_size, *self.config.input_shape[1:])
        traced_model = torch.jit.trace(self.original_model, dummy_input)
        
        # Determine precision
        if self.config.precision == 'fp16':
            precision_set = {torch.half}
        else:
            precision_set = {torch.float}
        
        # Compile with TensorRT
        self.optimized_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input(
                shape=[self.config.batch_size, *self.config.input_shape[1:]]
            )],
            enabled_precisions=precision_set,
            refit_enabled=True,
            debug=False
        )
        
        print(f"Model optimized for {self.config.precision} precision")
        return self.optimized_model
    
    def quantize_model(self):
        """Apply post-training quantization"""
        # Create quantizable version of model
        quantized_model = torch.quantization.quantize_dynamic(
            self.original_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        self.optimized_model = quantized_model
        print("Model quantized to INT8")
        return quantized_model

class EdgeDeploymentManager:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.optimizer = ModelOptimizer(config)
        self.model = None
        self.is_ready = False
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=50)
        
        # Logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup performance logger"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('EdgeDeploy')
        
        # File handler for persistent logs
        file_handler = logging.FileHandler('edge_deployment.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def deploy_model(self):
        """Deploy model to edge device"""
        try:
            # Load and optimize model
            self.optimizer.load_model()
            self.model = self.optimizer.optimize_for_jetson()
            
            # Validate model performance
            self._validate_performance()
            
            # Check if meets requirements
            if self._meets_requirements():
                self.is_ready = True
                self.logger.info("Model successfully deployed and validated")
                return True
            else:
                self.logger.error("Model does not meet deployment requirements")
                return False
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {str(e)}")
            return False
    
    def _validate_performance(self):
        """Validate model performance on edge device"""
        # Run multiple inference iterations to measure performance
        test_inputs = torch.randn(100, *self.config.input_shape[1:])
        
        inference_times = []
        for i in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(test_inputs[i:i+1])
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Update performance metrics
            self.inference_times.append(inference_time)
            self.throughput_history.append(1.0 / inference_time if inference_time > 0 else 0)
        
        avg_inference_time = np.mean(inference_times)
        avg_throughput = np.mean(list(self.throughput_history)[-20:])  # Last 20 measurements
        
        self.logger.info(f"Performance validation: avg_inference_time={avg_inference_time:.4f}s, "
                        f"avg_throughput={avg_throughput:.2f} FPS")
    
    def _meets_requirements(self):
        """Check if model meets deployment requirements"""
        if len(self.inference_times) == 0:
            return False
        
        avg_inference_time = np.mean(list(self.inference_times)[-20:])  # Last 20 measurements
        
        meets_latency = avg_inference_time <= self.config.max_latency
        avg_throughput = np.mean(list(self.throughput_history)[-20:]) if self.throughput_history else 0
        meets_throughput = avg_throughput >= self.config.min_throughput
        
        return meets_latency and meets_throughput
    
    def run_inference(self, input_tensor):
        """Run optimized inference on edge device"""
        if not self.is_ready:
            raise RuntimeError("Model not ready for inference")
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
        end_time = time.time()
        
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        
        # Log slow inference times
        if inference_time > self.config.max_latency:
            self.logger.warning(f"Slow inference: {inference_time:.4f}s (threshold: {self.config.max_latency}s)")
        
        return output
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if len(self.inference_times) == 0:
            return {
                'average_inference_time': 0.0,
                'average_throughput': 0.0,
                'latency_percentiles': [0.0, 0.0],
                'last_10_avg': 0.0
            }
        
        times_list = list(self.inference_times)
        avg_time = np.mean(times_list)
        avg_throughput = np.mean(list(self.throughput_history)) if self.throughput_history else 0
        
        p50 = np.percentile(times_list, 50) if len(times_list) > 1 else avg_time
        p95 = np.percentile(times_list, 95) if len(times_list) > 1 else avg_time
        
        last_10_avg = np.mean(times_list[-10:]) if len(times_list) >= 10 else avg_time
        
        return {
            'average_inference_time': avg_time,
            'average_throughput': avg_throughput,
            'latency_percentiles': [float(p50), float(p95)],
            'last_10_avg': last_10_avg
        }

class AIIntegrationValidator:
    def __init__(self, deployment_manager: EdgeDeploymentManager):
        self.deployment_manager = deployment_manager
        self.validation_results = {}
        
    def validate_integration(self):
        """Validate AI system integration"""
        try:
            # Test model with edge-specific inputs
            dummy_input = torch.randn(1, *self.deployment_manager.config.input_shape[1:])
            
            # Run inference
            start_time = time.time()
            output = self.deployment_manager.run_inference(dummy_input)
            end_time = time.time()
            
            # Validate output shape
            expected_shape = (1, *self.deployment_manager.config.output_shape[1:])
            actual_shape = tuple(output.shape)
            
            if actual_shape == expected_shape:
                validation_passed = True
                validation_message = f"Output shape validation passed: {actual_shape}"
            else:
                validation_passed = False
                validation_message = f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"
            
            # Performance validation
            inference_time = end_time - start_time
            latency_met = inference_time <= self.deployment_manager.config.max_latency
            
            # Store validation results
            self.validation_results = {
                'passed': validation_passed,
                'message': validation_message,
                'inference_time': inference_time,
                'latency_requirement_met': latency_met,
                'timestamp': time.time()
            }
            
            return self.validation_results
            
        except Exception as e:
            self.validation_results = {
                'passed': False,
                'message': f"Validation failed with error: {str(e)}",
                'timestamp': time.time()
            }
            return self.validation_results

def run_deployment_lab():
    """Run the AI deployment lab"""
    print("Starting AI Deployment Lab...")
    
    # Create deployment configuration
    config = DeploymentConfig(
        model_path="./models/perception_model.pth",
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 10),  # Example: 10 object classes
        batch_size=1,
        precision='fp16',
        max_latency=0.05,  # 50ms
        min_throughput=15  # 15 FPS
    )
    
    # Create and start deployment manager
    deploy_manager = EdgeDeploymentManager(config)
    
    # Deploy model to edge device
    if deploy_manager.deploy_model():
        print("Model deployed successfully!")
        
        # Validate integration
        validator = AIIntegrationValidator(deploy_manager)
        validation_results = validator.validate_integration()
        
        print(f"Integration validation: {validation_results['message']}")
        print(f"Inference time: {validation_results['inference_time']:.4f}s")
        print(f"Latency requirement met: {validation_results['latency_requirement_met']}")
        
        # Get performance statistics
        perf_stats = deploy_manager.get_performance_stats()
        print(f"Performance Stats: {perf_stats}")
        
        # Simulate running inference for a while
        print("\nRunning inference simulation...")
        for i in range(10):
            dummy_input = torch.randn(1, *config.input_shape[1:])
            try:
                output = deploy_manager.run_inference(dummy_input)
                print(f"Step {i+1}: Inference completed, output shape: {tuple(output.shape)}")
            except Exception as e:
                print(f"Step {i+1}: Inference failed: {str(e)}")
            time.sleep(0.1)  # Small delay to simulate real-time operation
        
    else:
        print("Model deployment failed!")

if __name__ == "__main__":
    run_deployment_lab()
```

### Analysis and Documentation

Document your results:
1. Measure inference performance (latency, throughput)
2. Evaluate resource usage (CPU, GPU, memory)
3. Assess accuracy preservation after optimization
4. Document deployment process and any challenges

## Lab Report Requirements

For each lab exercise, students must submit:

1. **Implementation Documentation** (25%):
   - Complete code with proper documentation
   - Explanation of design decisions
   - Configuration files and environment setup

2. **Performance Analysis** (40%):
   - Measurements of latency, throughput, and accuracy
   - Analysis of resource utilization
   - Comparison to baseline implementations

3. **System Integration Report** (25%):
   - How different Isaac components work together
   - Challenges encountered and solutions
   - Recommendations for improvements

4. **Reflection and Learning** (10%):
   - What was learned about Isaac Platform
   - How concepts apply to real-world robotics
   - Future directions for improvement

## Assessment Criteria

- Implementation quality and correctness (40%)
- Understanding of Isaac Platform components (30%)
- Performance analysis and optimization (20%)
- Documentation and code quality (10%)

## Troubleshooting Tips

1. **Isaac Sim Issues**: Check Omniverse connection and GPU drivers
2. **ROS Integration**: Verify proper namespace and topic mapping
3. **Performance Issues**: Monitor GPU utilization and memory usage
4. **Model Deployment**: Validate TensorRT compatibility with hardware
5. **Calibration Problems**: Re-run sensor calibration procedures
6. **Synchronization Issues**: Check message timestamps and queue sizes

## Extensions and Advanced Challenges

1. **Multi-Robot Coordination**: Extend to multi-robot systems
2. **Advanced Perception**: Implement 3D object detection and tracking
3. **Learning Systems**: Add reinforcement learning for navigation
4. **Human-Robot Interaction**: Implement gesture recognition interfaces
5. **Cloud Integration**: Connect with cloud AI services for heavy computation
6. **Safety Systems**: Implement comprehensive safety monitoring

## References and Further Reading

- NVIDIA Isaac Sim Documentation: https://docs.omniverse.nvidia.com/isaacsim/
- Isaac ROS Documentation: https://docs.nvidia.com/isaac/ros/
- Isaac Navigation Documentation
- Isaac Manipulation Documentation 
- ROS 2 Documentation for reference implementations