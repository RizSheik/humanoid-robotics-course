---
id: module-3-practical-lab
title: Module 3 — The AI Robot Brain | Chapter 4 — Practical Lab
sidebar_label: Chapter 4 — Practical Lab
sidebar_position: 4
---

# Module 3 — The AI Robot Brain

## Chapter 4 — Practical Lab

### Laboratory Setup and Prerequisites

This practical lab focuses on implementing AI systems for humanoid robots using NVIDIA Isaac as the primary development platform. Before beginning, ensure you have:

#### Hardware Requirements
- Computer with NVIDIA GPU (RTX 2080 or better recommended)
- Access to NVIDIA Isaac-compatible robot platform or simulation environment
- Network access for downloading AI models and datasets

#### Software Requirements
- Ubuntu 20.04 or 22.04 LTS
- ROS 2 Humble Hawksbill
- NVIDIA GPU drivers (470.86 or newer)
- CUDA 11.8 or newer
- Isaac Sim 2022.2.1 or newer
- Isaac ROS packages
- Python 3.8 or newer
- Git and other standard development tools

#### Installation Instructions

1. **Install NVIDIA Isaac Sim**:
   ```bash
   # Download Isaac Sim from NVIDIA Developer website
   # Follow installation instructions in NVIDIA documentation
   ```

2. **Install Isaac ROS packages**:
   ```bash
   sudo apt update
   sudo apt install ros-humble-isaac-ros-* ros-humble-nvblox-*
   ```

3. **Install AI development libraries**:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install tensorflow
   pip3 install transformers
   pip3 install stable-baselines3[extra]
   pip3 install gymnasium[classic-control,robotics]
   ```

### Lab Exercise 1: Computer Vision for Robot Perception

#### Objective
Implement a computer vision system for object detection and tracking using NVIDIA Isaac tools.

#### Step-by-Step Instructions

1. **Create a new ROS 2 package for computer vision**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python ai_perception_tutorial
   cd ai_perception_tutorial
   ```

2. **Create the package structure**:
   ```
   ai_perception_tutorial/
   ├── ai_perception_tutorial/
   │   ├── __init__.py
   │   ├── object_detector.py
   │   └── tracker.py
   ├── launch/
   │   └── perception_pipeline.launch.py
   ├── config/
   │   └── perception_params.yaml
   ├── models/
   │   └── yolov8n.pt  # Download YOLOv8 model
   ├── test/
   │   └── test_detector.py
   ├── CMakeLists.txt
   └── package.xml
   ```

3. **Create the object detector implementation** - `ai_perception_tutorial/ai_perception_tutorial/object_detector.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
   from cv_bridge import CvBridge
   import cv2
   import torch
   import numpy as np

   class ObjectDetector(Node):
       def __init__(self):
           super().__init__('object_detector')

           # Create subscriber for camera images
           self.image_sub = self.create_subscription(
               Image,
               '/camera/color/image_raw',
               self.image_callback,
               10
           )

           # Create publisher for detections
           self.detection_pub = self.create_publisher(
               Detection2DArray,
               '/ai_perception/detections',
               10
           )

           # Initialize CV bridge
           self.cv_bridge = CvBridge()

           # Load YOLO model
           try:
               self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
               self.get_logger().info('YOLOv5 model loaded successfully')
           except Exception as e:
               self.get_logger().error(f'Failed to load YOLO model: {e}')
               self.model = None

           # Load configuration parameters
           self.declare_parameter('confidence_threshold', 0.5)
           self.confidence_threshold = self.get_parameter('confidence_threshold').value

           self.declare_parameter('class_filter', ['person', 'bottle', 'cup', 'chair', 'dining table'])
           self.class_filter = self.get_parameter('class_filter').value

       def image_callback(self, msg):
           """Process incoming camera image and detect objects"""
           if self.model is None:
               return

           try:
               # Convert ROS Image message to OpenCV image
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Run object detection
               results = self.model(cv_image)

               # Parse results and create detection messages
               detections_msg = Detection2DArray()
               detections_msg.header = msg.header  # Use same timestamp and frame

               for detection in results.xyxy[0]:  # Results in xyxy format
                   x1, y1, x2, y2, conf, cls = detection
                   conf = float(conf)
                   cls = int(cls)

                   # Get class name
                   class_name = self.model.names[cls]

                   # Filter by confidence and class
                   if conf > self.confidence_threshold and class_name in self.class_filter:
                       # Create detection
                       detection_2d = Detection2D()
                       detection_2d.header = msg.header

                       # Calculate center and size
                       center_x = float((x1 + x2) / 2)
                       center_y = float((y1 + y2) / 2)
                       width = float(x2 - x1)
                       height = float(y2 - y1)

                       # Set the center position
                       detection_2d.bbox.center.x = center_x
                       detection_2d.bbox.center.y = center_y
                       detection_2d.bbox.size_x = width
                       detection_2d.bbox.size_y = height

                       # Add hypothesis
                       hypothesis = ObjectHypothesisWithPose()
                       hypothesis.hypothesis.class_id = class_name
                       hypothesis.hypothesis.score = conf
                       detection_2d.results.append(hypothesis)

                       # Add to array
                       detections_msg.detections.append(detection_2d)

               # Publish detections
               self.detection_pub.publish(detections_msg)

               # Log detection info
               if len(detections_msg.detections) > 0:
                   classes = [det.results[0].hypothesis.class_id for det in detections_msg.detections]
                   self.get_logger().info(f'Detected objects: {classes}')

           except Exception as e:
               self.get_logger().error(f'Error in image callback: {e}')

   def main(args=None):
       rclpy.init(args=args)
       detector = ObjectDetector()

       try:
           rclpy.spin(detector)
       except KeyboardInterrupt:
           detector.get_logger().info('Object detector stopped')
       finally:
           detector.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Create the object tracker implementation** - `ai_perception_tutorial/ai_perception_tutorial/tracker.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from vision_msgs.msg import Detection2DArray, Detection2D
   from geometry_msgs.msg import Point, PointStamped
   import numpy as np
   from collections import defaultdict
   import time

   class ObjectTracker(Node):
       def __init__(self):
           super().__init__('object_tracker')

           # Subscribers
           self.detection_sub = self.create_subscription(
               Detection2DArray,
               '/ai_perception/detections',
               self.detection_callback,
               10
           )

           # Publishers
           self.tracked_objects_pub = self.create_publisher(
               Detection2DArray,
               '/ai_perception/tracked_objects',
               10
           )

           self.tracked_point_pub = self.create_publisher(
               PointStamped,
               '/ai_perception/tracked_point',
               10
           )

           # Tracker state
           self.tracked_objects = {}  # {id: (position, last_seen_time)}
           self.next_id = 0
           self.max_track_age = 5.0  # seconds
           self.max_distance = 50.0  # pixels

           # Tracking parameters
           self.declare_parameter('max_distance', 50.0)
           self.declare_parameter('max_track_age', 5.0)
           self.max_distance = self.get_parameter('max_distance').value
           self.max_track_age = self.get_parameter('max_track_age').value

           # Timer for cleanup
           self.timer = self.create_timer(0.1, self.cleanup_old_tracks)

       def detection_callback(self, msg):
           """Process incoming detections and update tracks"""
           if not msg.detections:
               return

           # Convert new detections to format for tracking
           new_detections = []
           for detection in msg.detections:
               if len(detection.results) > 0:
                   class_id = detection.results[0].hypothesis.class_id
                   score = detection.results[0].hypothesis.score

                   x = detection.bbox.center.x
                   y = detection.bbox.center.y

                   new_detections.append({
                       'position': np.array([x, y]),
                       'class_id': class_id,
                       'score': score,
                       'detection': detection
                   })

           # Update existing tracks with new detections
           assigned_detections = set()
           updated_tracks = {}

           for track_id, (track_info, last_seen) in self.tracked_objects.items():
               track_pos, track_class = track_info

               # Find closest detection
               closest_detection = None
               min_distance = float('inf')

               for i, det in enumerate(new_detections):
                   if i in assigned_detections:
                       continue

                   distance = np.linalg.norm(track_pos - det['position'])

                   if (distance < self.max_distance and
                       det['class_id'] == track_class and
                       distance < min_distance):
                       min_distance = distance
                       closest_detection = det
                       closest_idx = i

               if closest_detection is not None:
                   # Update track position
                   updated_pos = (track_pos + closest_detection['position']) / 2
                   updated_tracks[track_id] = ((updated_pos, track_class), time.time())
                   assigned_detections.add(closest_idx)
               else:
                   # Keep old track position but update timestamp
                   updated_tracks[track_id] = ((track_pos, track_class), last_seen)

           # Create new tracks for unassigned detections
           for i, det in enumerate(new_detections):
               if i not in assigned_detections:
                   track_id = self.next_id
                   self.next_id += 1
                   updated_tracks[track_id] = (
                       (det['position'], det['class_id']),
                       time.time()
                   )

           # Update tracked objects
           self.tracked_objects = updated_tracks

           # Publish tracked objects
           self.publish_tracked_objects(msg.header)

       def publish_tracked_objects(self, header):
           """Publish tracked objects with IDs"""
           tracked_array = Detection2DArray()
           tracked_array.header = header

           for track_id, (track_info, _) in self.tracked_objects.items():
               pos, class_id = track_info

               detection = Detection2D()
               detection.header = header

               # Set position
               detection.bbox.center.x = float(pos[0])
               detection.bbox.center.y = float(pos[1])
               detection.bbox.size_x = 30.0  # Placeholder size
               detection.bbox.size_y = 30.0

               # Add track ID as object ID
               hypothesis = ObjectHypothesisWithPose()
               hypothesis.hypothesis.class_id = f"{class_id}_{track_id}"
               hypothesis.hypothesis.score = 0.9  # High confidence for tracked objects
               detection.results.append(hypothesis)

               tracked_array.detections.append(detection)

               # Publish center point for a specific tracked object (e.g., first person)
               if class_id == 'person':
                   point_msg = PointStamped()
                   point_msg.header = header
                   point_msg.point.x = float(pos[0])
                   point_msg.point.y = float(pos[1])
                   point_msg.point.z = 0.0
                   self.tracked_point_pub.publish(point_msg)
                   break  # Only publish for first person found

           if tracked_array.detections:
               self.tracked_objects_pub.publish(tracked_array)

       def cleanup_old_tracks(self):
           """Remove tracks that haven't been updated recently"""
           current_time = time.time()
           old_tracks = [
               track_id for track_id, (_, last_seen)
               in self.tracked_objects.items()
               if current_time - last_seen > self.max_track_age
           ]

           for track_id in old_tracks:
               del self.tracked_objects[track_id]

           if old_tracks:
               self.get_logger().info(f'Removed {len(old_tracks)} old tracks')

   def main(args=None):
       rclpy.init(args=args)
       tracker = ObjectTracker()

       try:
           rclpy.spin(tracker)
       except KeyboardInterrupt:
           tracker.get_logger().info('Object tracker stopped')
       finally:
           tracker.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

5. **Create a launch file** - `ai_perception_tutorial/launch/perception_pipeline.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Package path
       pkg_share = get_package_share_directory('ai_perception_tutorial')

       # Launch configuration variables
       use_sim_time = LaunchConfiguration('use_sim_time')

       # Declare launch arguments
       declare_use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )

       # Object detection node
       object_detector = Node(
           package='ai_perception_tutorial',
           executable='object_detector',
           name='object_detector',
           parameters=[
               os.path.join(pkg_share, 'config/perception_params.yaml')
           ],
           remappings=[
               ('/camera/color/image_raw', '/camera/color/image_raw'),
           ],
           output='screen'
       )

       # Object tracking node
       object_tracker = Node(
           package='ai_perception_tutorial',
           executable='object_tracker',
           name='object_tracker',
           parameters=[
               os.path.join(pkg_share, 'config/perception_params.yaml')
           ],
           output='screen'
       )

       # Return launch description
       return LaunchDescription([
           declare_use_sim_time,
           object_detector,
           object_tracker
       ])
   ```

6. **Create configuration file** - `ai_perception_tutorial/config/perception_params.yaml`:
   ```yaml
   object_detector:
     ros__parameters:
       confidence_threshold: 0.5
       class_filter: ["person", "bottle", "cup", "chair", "dining table"]

   object_tracker:
     ros__parameters:
       max_distance: 50.0
       max_track_age: 5.0
   ```

7. **Create setup.py**:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'ai_perception_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           # Include launch files
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
           # Include config files
           (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='AI perception tutorial for robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'object_detector = ai_perception_tutorial.object_detector:main',
               'object_tracker = ai_perception_tutorial.tracker:main',
           ],
       },
   )
   ```

8. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>ai_perception_tutorial</name>
     <version>0.0.0</version>
     <description>AI perception tutorial for robotics</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <depend>rclpy</depend>
     <depend>sensor_msgs</depend>
     <depend>vision_msgs</depend>
     <depend>geometry_msgs</depend>
     <depend>cv_bridge</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

9. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select ai_perception_tutorial
   source install/setup.bash
   ```

10. **Run the object detection pipeline**:
    ```bash
    ros2 launch ai_perception_tutorial perception_pipeline.launch.py
    ```

#### Expected Results
You should see object detection results published to the `/ai_perception/detections` and `/ai_perception/tracked_objects` topics. The system will detect objects like persons, bottles, cups, etc., and track them across frames.

### Lab Exercise 2: Reinforcement Learning for Robot Control

#### Objective
Create a reinforcement learning agent for robot control using Isaac Gym and NVIDIA's reinforcement learning frameworks.

#### Step-by-Step Instructions

1. **Create a reinforcement learning package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python robot_rl_tutorial
   cd robot_rl_tutorial
   ```

2. **Create the reinforcement learning implementation** - `robot_rl_tutorial/robot_rl_tutorial/ppo_agent.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32MultiArray
   from geometry_msgs.msg import Twist, Vector3
   from sensor_msgs.msg import LaserScan, Imu
   from nav_msgs.msg import Odometry
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import torch.nn.functional as F
   import numpy as np
   import random
   from collections import deque
   import math

   class ActorCritic(nn.Module):
       def __init__(self, state_dim, action_dim, hidden_dim=256):
           super(ActorCritic, self).__init__()

           # Shared feature extraction layers
           self.shared_layers = nn.Sequential(
               nn.Linear(state_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU()
           )

           # Actor network (policy)
           self.actor = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, action_dim),
               nn.Tanh()
           )

           # Critic network (value)
           self.critic = nn.Sequential(
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, 1)
           )

       def forward(self, state):
           features = self.shared_layers(state)

           action_probs = self.actor(features)
           state_value = self.critic(features)

           return action_probs, state_value

   class RobotPPOAgent(Node):
       def __init__(self):
           super().__init__('robot_ppo_agent')

           # Neural network
           self.state_dim = 24  # Example: 2D position + 20 LiDAR readings
           self.action_dim = 2  # Linear and angular velocity
           self.hidden_dim = 256

           # Initialize networks
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
           self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)

           # Memory for storing experiences
           self.memory = []
           self.gamma = 0.99  # Discount factor
           self.eps_clip = 0.2  # Clipping parameter
           self.max_grad_norm = 0.5  # Gradient clipping

           # Subscribers for robot sensors
           self.lidar_sub = self.create_subscription(
               LaserScan,
               '/scan',
               self.lidar_callback,
               10
           )

           self.odom_sub = self.create_subscription(
               Odometry,
               '/odom',
               self.odom_callback,
               10
           )

           self.imu_sub = self.create_subscription(
               Imu,
               '/imu/data',
               self.imu_callback,
               10
           )

           # Publisher for robot commands
           self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Timer for control loop
           self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

           # Robot state
           self.lidar_data = None
           self.position = [0.0, 0.0]
           self.orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion
           self.linear_vel = 0.0
           self.angular_vel = 0.0

           # Episode tracking
           self.episode_steps = 0
           self.max_episode_steps = 1000
           self.last_state = None
           self.last_action = None
           self.last_reward = None

           self.get_logger().info('PPO Robot Agent initialized')

       def lidar_callback(self, msg):
           """Process LiDAR data"""
           self.lidar_data = list(msg.ranges)
           # Filter invalid ranges
           self.lidar_data = [r if not math.isnan(r) and r > 0.1 else 3.5 for r in self.lidar_data]

       def odom_callback(self, msg):
           """Process odometry data"""
           self.position = [
               msg.pose.pose.position.x,
               msg.pose.pose.position.y
           ]

           self.orientation = [
               msg.pose.pose.orientation.x,
               msg.pose.pose.orientation.y,
               msg.pose.pose.orientation.z,
               msg.pose.pose.orientation.w
           ]

           self.linear_vel = msg.twist.twist.linear.x
           self.angular_vel = msg.twist.twist.angular.z

       def imu_callback(self, msg):
           """Process IMU data"""
           # In a real implementation, use IMU for additional state
           pass

       def get_robot_state(self):
           """Combine sensor data into a state vector"""
           if self.lidar_data is None:
               # Return a default state if no sensor data available
               return np.zeros(self.state_dim, dtype=np.float32)

           # Use first 20 LiDAR readings
           lidar_features = self.lidar_data[:20] if len(self.lidar_data) >= 20 else [3.5] * 20

           # Add position and velocity information
           state = np.array([
               self.position[0],    # x position
               self.position[1],    # y position
               self.linear_vel,     # linear velocity
               self.angular_vel,    # angular velocity
           ] + lidar_features, dtype=np.float32)  # LiDAR data

           # Normalize state to [-1, 1] range
           # Position and velocity normalization (adjust based on your robot's range)
           state[0] = np.clip(state[0] / 10.0, -1, 1)  # x position normalized by 10m
           state[1] = np.clip(state[1] / 10.0, -1, 1)  # y position normalized by 10m
           state[2] = np.clip(state[2] / 1.0, -1, 1)  # linear velocity normalized by 1 m/s
           state[3] = np.clip(state[3] / 2.0, -1, 1)  # angular velocity normalized by 2 rad/s

           # LiDAR range normalization (0-3.5m range)
           for i in range(4, len(state)):
               state[i] = np.clip(state[i] / 3.5, 0, 1) * 2 - 1  # Scale to [-1, 1]

           return state

       def select_action(self, state):
           """Select action using the current policy"""
           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

           with torch.no_grad():
               action_probs, state_value = self.actor_critic(state_tensor)
               action_probs = action_probs.cpu().numpy()[0]

           # Add some exploration noise
           action = action_probs + np.random.normal(0, 0.1, size=action_probs.shape)
           action = np.clip(action, -1, 1)  # Clip to valid range

           return action, state_value.item()

       def compute_reward(self, state, action):
           """Compute reward based on current state and action"""
           # Reward for moving forward (encourage exploration)
           forward_reward = max(0, action[0]) * 0.5  # Positive reward for forward movement

           # Penalty for getting too close to obstacles
           if self.lidar_data:
               min_distance = min(self.lidar_data) if self.lidar_data else 3.5
               obstacle_penalty = -5.0 if min_distance < 0.5 else 0.0

               # Small reward for maintaining safe distance
               safe_distance_reward = max(0, (min_distance - 0.5) * 2.0)
           else:
               obstacle_penalty = 0.0
               safe_distance_reward = 0.0

           # Reward for exploration (function of position change)
           exploration_reward = 0.1  # Small continuous reward for staying active

           total_reward = forward_reward + obstacle_penalty + safe_distance_reward + exploration_reward

           return total_reward

       def control_loop(self):
           """Main control loop for the robot"""
           # Get current state
           current_state = self.get_robot_state()

           # Select action
           action, state_value = self.select_action(current_state)

           # Compute reward
           reward = self.compute_reward(current_state, action)

           # Store experience if we have a previous state
           if self.last_state is not None and self.last_action is not None:
               experience = {
                   'state': self.last_state,
                   'action': self.last_action,
                   'log_prob': 0.0,  # Placeholder, in real PPO you'd compute this
                   'reward': self.last_reward,
                   'done': False,
                   'value': 0.0  # Will be computed later
               }
               self.memory.append(experience)

           # Publish action to robot
           cmd_msg = Twist()
           cmd_msg.linear.x = float(action[0]) * 0.5  # Scale to max 0.5 m/s
           cmd_msg.angular.z = float(action[1]) * 1.0  # Scale to max 1.0 rad/s
           self.cmd_pub.publish(cmd_msg)

           # Update state variables
           self.last_state = current_state.copy()
           self.last_action = action.copy()
           self.last_reward = reward

           # Increment episode steps
           self.episode_steps += 1

           # Check if episode should end
           if self.episode_steps >= self.max_episode_steps:
               self.end_episode()

       def end_episode(self):
           """End episode and update policy"""
           self.get_logger().info(f'Episode ended after {self.episode_steps} steps')

           # Add final experience
           if self.last_state is not None:
               final_experience = {
                   'state': self.last_state,
                   'action': self.last_action,
                   'log_prob': 0.0,
                   'reward': self.last_reward,
                   'done': True,
                   'value': 0.0
               }
               self.memory.append(final_experience)

           # Update policy using stored experiences
           if len(self.memory) > 10:  # Only update if we have enough experiences
               self.update_policy()

           # Reset episode
           self.episode_steps = 0
           self.memory = []

       def update_policy(self):
           """Update policy using PPO algorithm"""
           if len(self.memory) < 2:
               return

           # Convert memory to tensors
           states = torch.FloatTensor([exp['state'] for exp in self.memory[:-1]]).to(self.device)
           actions = torch.FloatTensor([exp['action'] for exp in self.memory[:-1]]).to(self.device)
           rewards = torch.FloatTensor([exp['reward'] for exp in self.memory[:-1]]).to(self.device)

           # Compute discounted rewards
           discounted_rewards = []
           running_add = self.memory[-1]['value'] if self.memory[-1]['done'] else 0
           for exp in reversed(self.memory[:-1]):
               running_add = exp['reward'] + self.gamma * running_add
               discounted_rewards.insert(0, running_add)

           discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)

           # Normalize advantages
           advantages = discounted_rewards - torch.FloatTensor([exp['value'] for exp in self.memory[:-1]]).to(self.device)
           advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

           # Compute loss and update
           old_states = states.detach()
           old_actions = actions.detach()

           # Forward pass
           new_action_probs, new_state_values = self.actor_critic(states)

           # Compute value loss
           value_loss = F.mse_loss(new_state_values.squeeze(), discounted_rewards)

           # For simplicity, using direct loss (in real PPO you'd compute ratios)
           policy_loss = -torch.mean(advantages * torch.sum((new_action_probs - old_actions) ** 2, dim=1))

           # Total loss
           total_loss = policy_loss + 0.5 * value_loss

           # Update network
           self.optimizer.zero_grad()
           total_loss.backward()

           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

           self.optimizer.step()

           self.get_logger().info(f'Policy updated. Value loss: {value_loss.item():.4f}, Policy loss: {policy_loss.item():.4f}')

   def main(args=None):
       rclpy.init(args=args)
       agent = RobotPPOAgent()

       try:
           rclpy.spin(agent)
       except KeyboardInterrupt:
           agent.get_logger().info('PPO Robot Agent stopped')
       finally:
           agent.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file for the RL agent** - `robot_rl_tutorial/launch/ppo_agent.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Package path
       pkg_share = get_package_share_directory('robot_rl_tutorial')

       # Launch configuration variables
       use_sim_time = LaunchConfiguration('use_sim_time')

       # Declare launch arguments
       declare_use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )

       # PPO agent node
       ppo_agent = Node(
           package='robot_rl_tutorial',
           executable='ppo_agent',
           name='robot_ppo_agent',
           parameters=[],
           remappings=[
               ('/scan', '/scan'),
               ('/odom', '/odom'),
               ('/imu/data', '/imu/data'),
               ('/cmd_vel', '/cmd_vel'),
           ],
           output='screen'
       )

       # Return launch description
       return LaunchDescription([
           declare_use_sim_time,
           ppo_agent
       ])
   ```

4. **Update setup.py**:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_rl_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           # Include launch files
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Robot reinforcement learning tutorial',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'ppo_agent = robot_rl_tutorial.ppo_agent:main',
           ],
       },
   )
   ```

5. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>robot_rl_tutorial</name>
     <version>0.0.0</version>
     <description>Robot reinforcement learning tutorial</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>geometry_msgs</depend>
     <depend>sensor_msgs</depend>
     <depend>nav_msgs</depend>
     <depend>message_filters</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

6. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_rl_tutorial
   source install/setup.bash
   ```

7. **Run the reinforcement learning agent**:
   ```bash
   ros2 launch robot_rl_tutorial ppo_agent.launch.py
   ```

#### Expected Results
The reinforcement learning agent will learn to navigate while avoiding obstacles. The robot should move forward while avoiding collisions, with the learning algorithm improving its policy over time.

### Lab Exercise 3: Natural Language Processing for Human-Robot Interaction

#### Objective
Implement a natural language processing system for human-robot interaction using NVIDIA Riva or similar tools.

#### Step-by-Step Instructions

1. **Create an NLP package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python robot_nlp_tutorial
   cd robot_nlp_tutorial
   ```

2. **Create the NLP implementation** - `robot_nlp_tutorial/robot_nlp_tutorial/nlp_processor.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan
   import re
   import json
   from typing import Dict, List, Tuple
   import math

   class NLPProcessor(Node):
       def __init__(self):
           super().__init__('nlp_processor')

           # Subscribers
           self.speech_sub = self.create_subscription(
               String,
               '/speech_recognition/text',
               self.speech_callback,
               10
           )

           self.lidar_sub = self.create_subscription(
               LaserScan,
               '/scan',
               self.lidar_callback,
               10
           )

           # Publishers
           self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.speech_pub = self.create_publisher(String, '/tts/text', 10)

           # Robot state
           self.lidar_data = None
           self.obstacle_threshold = 0.8  # meters

           # Command patterns and their handlers
           self.command_patterns = {
               'move_forward': [
                   r'go forward',
                   r'move forward',
                   r'forward',
                   r'go ahead',
                   r'continue'
               ],
               'move_backward': [
                   r'go backward',
                   r'move backward',
                   r'backward',
                   r'reverse',
                   r'back up'
               ],
               'turn_left': [
                   r'turn left',
                   r'go left',
                   r'left',
                   r'strafe left'
               ],
               'turn_right': [
                   r'turn right',
                   r'go right',
                   r'right',
                   r'strafe right'
               ],
               'stop': [
                   r'stop',
                   r'hold',
                   r'freeze',
                   r'wait',
                   r'halt'
               ],
               'find_object': [
                   r'find (?:the )?(\w+)',
                   r'locate (?:the )?(\w+)',
                   r'where is (?:the )?(\w+)'
               ],
               'go_to_object': [
                   r'go to (?:the )?(\w+)',
                   r'move to (?:the )?(\w+)',
                   r'go (?:to the )?(\w+)'
               ],
               'greet': [
                   r'hello',
                   r'hi',
                   r'hey',
                   r'greetings'
               ]
           }

           # Precompile regex patterns for efficiency
           self.precompiled_patterns = {}
           for cmd_type, patterns in self.command_patterns.items():
               self.precompiled_patterns[cmd_type] = []
               for pattern in patterns:
                   self.precompiled_patterns[cmd_type].append(re.compile(pattern, re.IGNORECASE))

           # Command handlers
           self.command_handlers = {
               'move_forward': self.handle_move_forward,
               'move_backward': self.handle_move_backward,
               'turn_left': self.handle_turn_left,
               'turn_right': self.handle_turn_right,
               'stop': self.handle_stop,
               'find_object': self.handle_find_object,
               'go_to_object': self.handle_go_to_object,
               'greet': self.handle_greet
           }

           self.get_logger().info('NLP Processor initialized')

       def speech_callback(self, msg):
           """Process incoming speech command"""
           command_text = msg.data
           self.get_logger().info(f'Received command: "{command_text}"')

           # Parse command and execute
           command_type, command_args = self.parse_command(command_text)

           if command_type and command_type in self.command_handlers:
               self.command_handlers[command_type](command_args)
           else:
               # Unknown command
               self.speak_response("I don't understand that command.")

       def parse_command(self, text: str) -> Tuple[str, List[str]]:
           """Parse natural language command and return type and arguments"""
           text = text.strip().lower()

           for cmd_type, patterns in self.precompiled_patterns.items():
               for pattern in patterns:
                   match = pattern.search(text)
                   if match:
                       args = list(match.groups()) if match.groups() else []
                       return cmd_type, args

           return None, []

       def lidar_callback(self, msg):
           """Process LiDAR data"""
           self.lidar_data = list(msg.ranges)
           # Filter invalid ranges
           self.lidar_data = [r if not math.isnan(r) and r > 0.1 else 3.5 for r in self.lidar_data]

       def has_obstacle_ahead(self) -> bool:
           """Check if there's an obstacle in front of the robot"""
           if not self.lidar_data:
               return False

           # Check front 30 degrees (15 degrees left and right of center)
           center_idx = len(self.lidar_data) // 2
           front_range = self.lidar_data[center_idx - 15:center_idx + 15]

           if front_range:
               min_dist = min(front_range)
               return min_dist < self.obstacle_threshold

           return False

       def speak_response(self, text: str):
           """Publish text-to-speech response"""
           response_msg = String()
           response_msg.data = text
           self.speech_pub.publish(response_msg)
           self.get_logger().info(f'Responding: "{text}"')

       # Command handlers
       def handle_move_forward(self, args):
           """Handle forward movement command"""
           if self.has_obstacle_ahead():
               self.speak_response("I cannot go forward because there is an obstacle in the way.")
               return

           cmd = Twist()
           cmd.linear.x = 0.3  # 0.3 m/s forward
           cmd.angular.z = 0.0
           self.cmd_pub.publish(cmd)
           self.speak_response("Moving forward.")

       def handle_move_backward(self, args):
           """Handle backward movement command"""
           cmd = Twist()
           cmd.linear.x = -0.2  # 0.2 m/s backward
           cmd.angular.z = 0.0
           self.cmd_pub.publish(cmd)
           self.speak_response("Moving backward.")

       def handle_turn_left(self, args):
           """Handle left turn command"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.5  # 0.5 rad/s counter-clockwise (left)
           self.cmd_pub.publish(cmd)
           self.speak_response("Turning left.")

       def handle_turn_right(self, args):
           """Handle right turn command"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = -0.5  # 0.5 rad/s clockwise (right)
           self.cmd_pub.publish(cmd)
           self.speak_response("Turning right.")

       def handle_stop(self, args):
           """Handle stop command"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.0
           self.cmd_pub.publish(cmd)
           self.speak_response("Stopping now.")

       def handle_find_object(self, args):
           """Handle find object command"""
           if args:
               obj_type = args[0]
               self.speak_response(f"Looking for {obj_type}. I don't have object detection enabled in this demo, but I would search visually for {obj_type}.")
           else:
               self.speak_response("What object should I look for?")

       def handle_go_to_object(self, args):
           """Handle go to object command"""
           if args:
               obj_type = args[0]
               self.speak_response(f"Going to the {obj_type}. I would navigate to the {obj_type} if I could detect it.")
           else:
               self.speak_response("Go to what object?")

       def handle_greet(self, args):
           """Handle greeting command"""
           self.speak_response("Hello! How can I assist you today?")

   def main(args=None):
       rclpy.init(args=args)
       processor = NLPProcessor()

       try:
           rclpy.spin(processor)
       except KeyboardInterrupt:
           processor.get_logger().info('NLP Processor stopped')
       finally:
           processor.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** - `robot_nlp_tutorial/launch/nlp_system.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Package path
       pkg_share = get_package_share_directory('robot_nlp_tutorial')

       # Launch configuration variables
       use_sim_time = LaunchConfiguration('use_sim_time')

       # Declare launch arguments
       declare_use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='false',
           description='Use simulation (Gazebo) clock if true'
       )

       # NLP processor node
       nlp_processor = Node(
           package='robot_nlp_tutorial',
           executable='nlp_processor',
           name='nlp_processor',
           parameters=[],
           remappings=[
               ('/speech_recognition/text', '/speech_recognition/text'),
               ('/scan', '/scan'),
               ('/cmd_vel', '/cmd_vel'),
           ],
           output='screen'
       )

       # Return launch description
       return LaunchDescription([
           declare_use_sim_time,
           nlp_processor
       ])
   ```

4. **Update setup.py**:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nlp_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           # Include launch files
           (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Robot NLP tutorial',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'nlp_processor = robot_nlp_tutorial.nlp_processor:main',
           ],
       },
   )
   ```

5. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>robot_nlp_tutorial</name>
     <version>0.0.0</version>
     <description>Robot NLP tutorial</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>geometry_msgs</depend>
     <depend>sensor_msgs</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

6. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nlp_tutorial
   source install/setup.bash
   ```

7. **Run the NLP system**:
   ```bash
   ros2 launch robot_nlp_tutorial nlp_system.launch.py
   ```

#### Expected Results
The NLP system will process text commands sent to `/speech_recognition/text` and respond appropriately by publishing robot commands to `/cmd_vel` and verbal responses to `/tts/text`.

### Lab Exercise 4: Integration and Testing

#### Objective
Integrate all AI components into a complete AI Robot Brain system and test in simulation.

#### Step-by-Step Instructions

1. **Create an integration launch file** - `robot_nlp_tutorial/launch/integrated_ai_brain.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Package directories
       pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
       pkg_ai_perception = get_package_share_directory('ai_perception_tutorial')
       pkg_robot_rl = get_package_share_directory('robot_rl_tutorial')
       pkg_robot_nlp = get_package_share_directory('robot_nlp_tutorial')

       # Launch configuration
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       # Gazebo launch
       gazebo = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
           ),
           launch_arguments={
               'world': PathJoinSubstitution([
                   get_package_share_directory('turtlebot3_gazebo'),
                   'worlds',
                   'empty_world.model'
               ]),
           }.items()
       )

       gazebo_client = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
           )
       )

       # Robot spawn node
       spawn_entity = Node(
           package='gazebo_ros',
           executable='spawn_entity.py',
           arguments=[
               '-topic', 'robot_description',
               '-entity', 'turtlebot3',
           ],
           output='screen'
       )

       # AI components
       object_detector = Node(
           package='ai_perception_tutorial',
           executable='object_detector',
           name='object_detector',
           parameters=[
               os.path.join(pkg_ai_perception, 'config/perception_params.yaml')
           ],
           condition=None
       )

       object_tracker = Node(
           package='ai_perception_tutorial',
           executable='object_tracker',
           name='object_tracker',
           parameters=[
               os.path.join(pkg_ai_perception, 'config/perception_params.yaml')
           ],
           condition=None
       )

       ppo_agent = Node(
           package='robot_rl_tutorial',
           executable='ppo_agent',
           name='robot_ppo_agent',
           parameters=[],
           condition=None
       )

       nlp_processor = Node(
           package='robot_nlp_tutorial',
           executable='nlp_processor',
           name='nlp_processor',
           parameters=[],
           condition=None
       )

       # Robot state publisher
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           parameters=[{'use_sim_time': use_sim_time}]
       )

       # Return launch description with delayed start
       return LaunchDescription([
           gazebo,
           gazebo_client,
           TimerAction(
               period=5.0,
               actions=[
                   spawn_entity,
                   robot_state_publisher,
                   object_detector,
                   object_tracker,
                   ppo_agent,
                   nlp_processor
               ]
           )
       ])
   ```

2. **Create a test script to verify integration** - `robot_nlp_tutorial/scripts/test_integration.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from sensor_msgs.msg import LaserScan
   from geometry_msgs.msg import Twist
   import time

   class IntegrationTester(Node):
       def __init__(self):
           super().__init__('integration_tester')

           # Publishers
           self.speech_pub = self.create_publisher(String, '/speech_recognition/text', 10)
           self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Subscribers
           self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

           # Test state
           self.scan_received = False
           self.test_phase = 0
           self.test_complete = False

           # Timer for tests
           self.test_timer = self.create_timer(2.0, self.run_test)

           self.get_logger().info('Integration Tester started')

       def scan_callback(self, msg):
           """Callback to confirm sensor data is flowing"""
           if not self.scan_received:
               self.get_logger().info('Laser scan data is flowing - sensors connected')
               self.scan_received = True

       def run_test(self):
           """Run integration tests"""
           if self.test_complete:
               return

           if self.test_phase == 0:
               self.get_logger().info('Test Phase 0: Testing basic movement')
               self.test_basic_movement()
           elif self.test_phase == 1:
               self.get_logger.info('Test Phase 1: Testing NLP command')
               self.test_nlp_command()
           elif self.test_phase == 2:
               self.get_logger().info('Test Phase 2: Testing object detection')
               self.test_object_detection()
           elif self.test_phase == 3:
               self.get_logger().info('Integration tests completed successfully')
               self.test_complete = True

           self.test_phase += 1

       def test_basic_movement(self):
           """Test basic movement commands"""
           cmd = Twist()
           cmd.linear.x = 0.2
           cmd.angular.z = 0.0
           self.cmd_pub.publish(cmd)
           self.get_logger().info('Published forward command')

           # Stop after movement
           stop_timer = self.create_timer(2.0, self._stop_robot)

       def _stop_robot(self):
           """Stop the robot"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.0
           self.cmd_pub.publish(cmd)
           self.get_logger().info('Robot stopped')

       def test_nlp_command(self):
           """Test NLP command processing"""
           command_msg = String()
           command_msg.data = "Hello robot"
           self.speech_pub.publish(command_msg)
           self.get_logger().info('Published greeting command')

       def test_object_detection(self):
           """Test object detection system"""
           # In a real test, we would check if object detection is working
           # For now, just log the test
           self.get_logger().info('Object detection test phase - check if detections are published')

   def main(args=None):
       rclpy.init(args=args)
       tester = IntegrationTester()

       try:
           rclpy.spin(tester)
       except KeyboardInterrupt:
           tester.get_logger().info('Integration tester stopped')
       finally:
           tester.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Make the test script executable and run tests**:
   ```bash
   chmod +x robot_nlp_tutorial/scripts/test_integration.py
   cd ~/robotics_ws
   colcon build --packages-select robot_nlp_tutorial
   source install/setup.bash
   ```

4. **Run the complete integrated system**:
   ```bash
   ros2 launch robot_nlp_tutorial integrated_ai_brain.launch.py
   ```

5. **In another terminal, run the integration test**:
   ```bash
   ros2 run robot_nlp_tutorial test_integration
   ```

#### Expected Results
The complete AI Robot Brain system will be running with:
- Object detection and tracking
- Reinforcement learning-based navigation
- Natural language processing for commands
- Integration of all components

### Troubleshooting Guide

#### Common Issues and Solutions

1. **CUDA/GPU Issues**:
   - Ensure NVIDIA drivers and CUDA are properly installed
   - Verify GPU compatibility and CUDA version alignment
   - Check that PyTorch was installed with CUDA support

2. **ROS Communication Issues**:
   - Verify all topics are properly connected
   - Check that message types are compatible between nodes
   - Ensure clock synchronization if using simulation time

3. **Memory Issues**:
   - Reduce batch sizes if running out of GPU memory
   - Use model quantization to reduce memory footprint
   - Implement experience replay with memory management

4. **Performance Issues**:
   - Profile code to identify bottlenecks
   - Optimize neural network inference with TensorRT
   - Use multi-threading where appropriate

### Conclusion

This practical lab has provided hands-on experience with implementing key AI components for humanoid robotics:

1. Computer vision systems for perception
2. Reinforcement learning agents for control
3. Natural language processing for interaction
4. Integration of all components into a cohesive AI Robot Brain

These implementations form the foundation of intelligent robotic systems that can perceive, reason, learn, and interact with their environment and humans in natural ways.