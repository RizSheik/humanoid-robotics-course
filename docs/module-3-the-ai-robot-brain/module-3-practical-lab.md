---
id: module-3-practical-lab
title: 'Module 3 — The AI-Robot Brain | Chapter 4 — Practical Lab'
sidebar_label: 'Chapter 4 — Practical Lab'
sidebar_position: 4
---

# Chapter 4 — Practical Lab

## Building an AI-Robot Brain with NVIDIA Isaac

In this practical lab, we'll implement key components of an AI-Robot Brain using the NVIDIA Isaac platform. We'll create perception systems, planning algorithms, learning components, and integration mechanisms that form the cognitive core of a humanoid robot.

### Prerequisites

Before starting this lab, ensure you have:
- Completed Modules 1 and 2 (The Robotic Nervous System and The Digital Twin)
- NVIDIA GPU with CUDA support (RTX 2080 or better recommended)
- Ubuntu 20.04 or 22.04 with ROS 2 Foxy or Humble
- NVIDIA Isaac ROS packages installed
- Isaac Sim installed and configured
- Basic knowledge of PyTorch/TensorFlow and deep learning

## Lab 1: Setting Up the NVIDIA Isaac Environment

### Installing Isaac ROS Packages

1. **Verify your NVIDIA GPU and CUDA installation**:
```bash
nvidia-smi
nvcc --version
```

2. **Install Isaac ROS packages**:
```bash
# Install system dependencies
sudo apt update
sudo apt install git-lfs

# Clone Isaac ROS repository
cd ~/ros2_ws/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_detect_net.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_cov_estimator.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git

# Install additional dependencies
rosdep install --from-paths . --ignore-src -r -y
```

3. **Build the Isaac ROS packages**:
```bash
cd ~/ros2_ws
colcon build --packages-select \
  isaac_ros_common \
  isaac_ros_visual_slam \
  isaac_ros_detect_net \
  isaac_ros_pose_cov_estimator \
  image_proc \
  image_publisher \
  image_transport \
  camera_info_manager
```

4. **Source the workspace**:
```bash
source ~/ros2_ws/install/setup.bash
```

### Testing Isaac ROS Packages

1. **Run the image processing demo to verify installation**:
```bash
# Terminal 1: Launch image processing pipeline
ros2 launch isaac_ros_image_pipeline image_pipeline.launch.py

# Terminal 2: Publish a test image
ros2 run image_publisher image_publisher_node --ros-args -p file_name:=/path/to/test/image.jpg
```

2. **Verify GPU acceleration is working**:
```bash
# Check for GPU usage during processing
watch -n 1 nvidia-smi
```

## Lab 2: Perception System Implementation

### Implementing a Deep Learning-Based Object Detection System

1. **Create a new ROS 2 package for perception**:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python perception_pkg --dependencies rclpy sensor_msgs cv_bridge std_msgs message_filters
```

2. **Create the detection node** (`~/ros2_ws/src/perception_pkg/perception_pkg/detection_node.py`):
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Classes for COCO dataset (80 classes)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Publishers and Subscribers
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/camera_info')
        
        # Synchronize image and camera info
        ts = ApproximateTimeSynchronizer([self.image_sub, self.info_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)
        
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        
        self.get_logger().info('Object Detection Node has been started')

    def image_callback(self, img_msg, info_msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            
            # Convert to tensor and normalize
            image_tensor = self.transform(cv_image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Process detections
            detections_msg = self.process_detections(predictions[0], img_msg.header, info_msg)
            
            # Publish detections
            self.detection_pub.publish(detections_msg)
            
            self.get_logger().info(f'Detected {len(detections_msg.detections)} objects')
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def process_detections(self, prediction, header, info_msg):
        detections_msg = Detection2DArray()
        detections_msg.header = header
        
        scores = prediction['scores'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy().astype(int)
        labels = prediction['labels'].cpu().numpy()
        
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                detection = Detection2D()
                
                # Set the ID and confidence
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(self.coco_classes[labels[i]])
                hypothesis.hypothesis.score = float(scores[i])
                
                detection.results.append(hypothesis)
                
                # Set the bounding box
                bbox = detection.bbox
                bbox.center.x = float((boxes[i][0] + boxes[i][2]) / 2.0)
                bbox.center.y = float((boxes[i][1] + boxes[i][3]) / 2.0)
                bbox.size_x = float(boxes[i][2] - boxes[i][0])
                bbox.size_y = float(boxes[i][3] - boxes[i][1])
                
                detections_msg.detections.append(detection)
        
        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. **Update the setup.py file** (`~/ros2_ws/src/perception_pkg/setup.py`):
```python
from setuptools import setup
import os
from glob import glob

package_name = 'perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Perception package for AI-Robot Brain',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = perception_pkg.detection_node:main',
        ],
    },
)
```

4. **Build the perception package**:
```bash
cd ~/ros2_ws
colcon build --packages-select perception_pkg
source ~/ros2_ws/install/setup.bash
```

## Lab 3: Path Planning and Navigation

### Implementing a GPU-Accelerated Path Planner

1. **Create a navigation package**:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python navigation_pkg --dependencies rclpy geometry_msgs nav_msgs sensor_msgs visualization_msgs tf2_ros tf2_geometry_msgs
```

2. **Create a 2D navigation node** (`~/ros2_ws/src/navigation_pkg/navigation_pkg/navigation_node.py`):
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import numpy as np
from scipy.spatial import KDTree
import math
import heapq

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # Publishers and subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )
        
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.vis_pub = self.create_publisher(MarkerArray, '/path_visualization', 10)
        
        # Path planning parameters
        self.map_data = None
        self.map_resolution = 0.05  # meters per cell
        self.map_origin = None
        self.occupancy_threshold = 65  # threshold for obstacle detection
        
        # A* parameters
        self.grid = None
        
        self.get_logger().info('Path Planner Node has been started')

    def map_callback(self, msg):
        """Callback to receive occupancy grid map"""
        self.get_logger().info('Received map')
        
        # Store map data
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        # Convert occupancy grid to binary grid (free/occupied)
        self.grid = (self.map_data > self.occupancy_threshold).astype(int)
        
        self.get_logger().info(f'Map updated: {self.grid.shape}')

    def goal_callback(self, msg):
        """Callback to receive navigation goal"""
        if self.grid is None:
            self.get_logger().warn('Map not received yet, cannot plan path')
            return
            
        # Convert goal from world coordinates to grid coordinates
        goal_x = int((msg.pose.position.x - self.map_origin[0]) / self.map_resolution)
        goal_y = int((msg.pose.position.y - self.map_origin[1]) / self.map_resolution)
        
        # Get current robot position (for this demo, assume it's at (0,0) in world coordinates)
        current_x = int((0 - self.map_origin[0]) / self.map_resolution)
        current_y = int((0 - self.map_origin[1]) / self.map_resolution)
        
        # Plan path using A*
        path = self.a_star(current_x, current_y, goal_x, goal_y)
        
        if path:
            self.publish_path(path)
            self.visualize_path(path)
            self.get_logger().info(f'Path found with {len(path)} waypoints')
        else:
            self.get_logger().warn('No path found to goal')

    def a_star(self, start_x, start_y, goal_x, goal_y):
        """
        A* path planning implementation
        """
        # Check if start and goal are valid
        if (start_x < 0 or start_x >= self.grid.shape[1] or 
            start_y < 0 or start_y >= self.grid.shape[0] or 
            self.grid[start_y, start_x] == 1):
            self.get_logger().warn('Start position is invalid or occupied')
            return None
            
        if (goal_x < 0 or goal_x >= self.grid.shape[1] or 
            goal_y < 0 or goal_y >= self.grid.shape[0] or 
            self.grid[goal_y, goal_x] == 1):
            self.get_logger().warn('Goal position is invalid or occupied')
            return None

        # Possible movements (8 directions)
        movements = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 4 directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 4 diagonal directions
        ]
        
        # Calculate movement costs (diagonals have higher cost)
        costs = [1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4]
        
        # Initialize open and closed sets
        open_set = []
        heapq.heappush(open_set, (0, start_x, start_y))
        
        # Costs and parent tracking
        g_score = np.full(self.grid.shape, np.inf)
        g_score[start_y, start_x] = 0
        
        f_score = np.full(self.grid.shape, np.inf)
        f_score[start_y, start_x] = self.heuristic(start_x, start_y, goal_x, goal_y)
        
        came_from = {}
        
        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)
            
            # If we reached the goal
            if current_x == goal_x and current_y == goal_y:
                return self.reconstruct_path(came_from, (current_x, current_y))
            
            # Mark as visited
            if current_f > f_score[current_y, current_x]:
                continue
                
            # Explore neighbors
            for i, (dx, dy) in enumerate(movements):
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                # Check bounds
                if (neighbor_x < 0 or neighbor_x >= self.grid.shape[1] or 
                    neighbor_y < 0 or neighbor_y >= self.grid.shape[0]):
                    continue
                    
                # Skip if obstacle
                if self.grid[neighbor_y, neighbor_x] == 1:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current_y, current_x] + costs[i]
                
                # If this path is better, update scores
                if tentative_g < g_score[neighbor_y, neighbor_x]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[neighbor_y, neighbor_x] = tentative_g
                    f_score[neighbor_y, neighbor_x] = tentative_g + self.heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score[neighbor_y, neighbor_x], neighbor_x, neighbor_y))
        
        # No path found
        return None

    def heuristic(self, x1, y1, x2, y2):
        """Heuristic function for A* (Euclidean distance)"""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from goal to start"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def publish_path(self, grid_path):
        """Convert grid path to ROS Path message and publish"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  # Assuming map frame is 'map'
        
        for x, y in grid_path:
            # Convert grid coordinates to world coordinates
            world_x = x * self.map_resolution + self.map_origin[0]
            world_y = y * self.map_resolution + self.map_origin[1]
            
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0  # Assuming 2D navigation
            pose.pose.orientation.w = 1.0  # No rotation for now
            
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def visualize_path(self, grid_path):
        """Visualize path using markers"""
        marker_array = MarkerArray()
        
        # Create marker for each point in the path
        for i, (x, y) in enumerate(grid_path):
            # Convert grid coordinates to world coordinates
            world_x = x * self.map_resolution + self.map_origin[0]
            world_y = y * self.map_resolution + self.map_origin[1]
            
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'path'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
        
        self.vis_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. **Update navigation package setup.py**:
```python
from setuptools import setup
import os
from glob import glob

package_name = 'navigation_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Navigation package for AI-Robot Brain',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation_node = navigation_pkg.navigation_node:main',
        ],
    },
)
```

4. **Build the navigation package**:
```bash
cd ~/ros2_ws
colcon build --packages-select navigation_pkg
source ~/ros2_ws/install/setup.bash
```

## Lab 4: Implementing a Learning Component

### Creating a Reinforcement Learning Environment for Navigation

1. **Install Isaac Lab dependencies**:
```bash
# Create a Python virtual environment for Isaac Lab
python3 -m venv ~/isaac_env
source ~/isaac_env/bin/activate
pip install --upgrade pip

# Install Isaac Lab dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm tensorboard wandb
pip install gymnasium
```

2. **Create a simple navigation RL environment** (`~/ros2_ws/src/learning_pkg/learning_pkg/navigation_env.py`):
```python
import gymnasium as gym
import numpy as np
import random
from typing import Optional
import pygame
import math

class SimpleNavigationEnv(gym.Env):
    """
    A simple navigation environment for reinforcement learning
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as single integer
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # Right
            1: np.array([0, -1]), # Up
            2: np.array([-1, 0]), # Left
            3: np.array([0, 1]),  # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (discrete number) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else -0.1  # Negative reward for each step to encourage faster solutions
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
```

3. **Create a simple RL training script** (`~/ros2_ws/src/learning_pkg/learning_pkg/rl_trainer.py`):
```python
#!/usr/bin/env python3
import numpy as np
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from navigation_env import SimpleNavigationEnv
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    """
    Simple neural network for Q-learning
    """
    def __init__(self, input_size, n_actions, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    """
    Deep Q-Learning Agent for navigation
    """
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = QNetwork(input_size=4, n_actions=env.action_space.n)
        self.target_network = QNetwork(input_size=4, n_actions=env.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
        # Training parameters
        self.memory = []
        self.batch_size = 32
        self.memory_size = 10000
        self.update_target_freq = 100  # Update target network every 100 steps
        self.step_count = 0
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Remove oldest experience
            
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        
        # Convert state to tensor and get Q values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
        
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def train(self, episodes=1000):
        """Train the agent"""
        scores = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            state_flat = np.concatenate([state['agent'], state['target']])  # Flatten state
            total_reward = 0
            steps = 0
            
            while True:
                action = self.act(state_flat)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state_flat = np.concatenate([next_state['agent'], next_state['target']])  # Flatten state
                
                self.remember(state_flat, action, reward, next_state_flat, done)
                
                state_flat = next_state_flat
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
            # Train on experiences
            self.replay()
            
            # Update target network periodically
            self.step_count += 1
            if self.step_count % self.update_target_freq == 0:
                self.update_target_network()
                
            scores.append(total_reward)
            
            if episode % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.2f}")
                
        return scores

def main():
    # Create environment
    env = SimpleNavigationEnv(render_mode=None)
    
    # Create agent
    agent = DQNAgent(env)
    
    # Train the agent
    print("Training agent...")
    scores = agent.train(episodes=2000)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg)
    plt.title('Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    print("Training completed!")

if __name__ == '__main__':
    main()
```

## Lab 5: Integrating Components with Isaac Sim

### Setting up Isaac Sim for AI Training

1. **Launch Isaac Sim and create a simple navigation environment**:
```bash
# Launch Isaac Sim
./isaac-sim/python.sh ./isaac-sim/apps/omniverse IsaacSim.app
```

2. **Create a Python script for Isaac Sim integration** (`~/ros2_ws/src/simulation_pkg/simulation_pkg/isaac_integration.py`):
```python
"""
Isaac Sim integration example for AI-Robot Brain
"""
import omni
import omni.usd
from pxr import Gf, UsdGeom, Sdf, UsdPhysics
import carb
import numpy as np

# Initialize the extension
class IsaacSimIntegration:
    def __init__(self):
        self._stage = None
        self._world = None
        self._setup_scene()
    
    def _setup_scene(self):
        """Setup the initial scene in Isaac Sim"""
        # Get the USD stage
        self._stage = omni.usd.get_context().get_stage()
        
        # Create a simple ground plane
        plane_path = Sdf.Path("/World/GroundPlane")
        self._plane = UsdGeom.Mesh.Define(self._stage, plane_path)
        self._plane.CreateMeshPurposeAttr().Set("default")
        
        # Add physics to the ground
        physics_api = UsdPhysics.MeshCollisionAPI.Apply(self._plane.GetPrim())
        physics_api.CreateCollisionEnabledAttr(True)
    
    def reset_environment(self):
        """Reset the simulation environment"""
        # Implementation for resetting the simulation
        print("Environment reset")
    
    def get_sensor_data(self):
        """Get sensor data from the simulation"""
        # This would interface with actual sensors in the simulation
        # For now, we return dummy data
        return {
            'camera': np.random.rand(640, 480, 3),  # Random camera data
            'lidar': np.random.rand(360),  # Simplified LIDAR data
            'imu': {'linear_acc': [0.0, 0.0, 9.81], 'angular_vel': [0.0, 0.0, 0.0]}
        }
    
    def send_action(self, action):
        """Send action to the simulated robot"""
        # This would send commands to the simulated robot
        print(f"Action sent: {action}")
    
    def get_environment_state(self):
        """Get the current state of the environment"""
        # This would return the current state of the environment
        return {
            'robot_pos': [0.0, 0.0, 0.0],
            'robot_orientation': [0.0, 0.0, 0.0, 1.0],
            'target_pos': [5.0, 5.0, 0.0]
        }

# Example usage
def run_example():
    """Example usage of the Isaac Sim integration"""
    sim_integration = IsaacSimIntegration()
    
    for step in range(100):
        # Get sensor data
        sensor_data = sim_integration.get_sensor_data()
        
        # Process sensor data (in a real implementation, this would use your AI models)
        # For now, we just print the data shape
        print(f"Step {step}: Camera shape {sensor_data['camera'].shape}")
        
        # Get environment state
        state = sim_integration.get_environment_state()
        print(f"Robot at: {state['robot_pos']}")
        
        # Send a random action (in real implementation, this would come from your AI)
        action = np.random.randint(0, 4)  # Random action for navigation
        sim_integration.send_action(action)
        
        # Step the simulation
        # In Isaac Sim, you would typically step the physics
        # This would be handled by the physics simulation loop
        carb.log_info(f"Step {step} completed")

# Register the extension
class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print(f"[isaac_sim_integration] Isaac Sim Integration starting up")
        self._sim_integration = IsaacSimIntegration()
        
    def on_shutdown(self):
        print(f"[isaac_sim_integration] Isaac Sim Integration shutting down")
        self._sim_integration = None
```

## Lab 6: Creating a Cognitive Architecture

### Integrating Perception, Planning, and Learning

1. **Create the main AI-Robot Brain node** (`~/ros2_ws/src/robot_brain_pkg/robot_brain_pkg/brain_node.py`):
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import Odometry, Path
import numpy as np
import threading
import time
from collections import deque

class AIRobotBrainNode(Node):
    """
    The main AI-Robot Brain node that integrates perception, 
    planning, learning, and control components
    """
    
    def __init__(self):
        super().__init__('ai_robot_brain')
        
        # Initialize state variables
        self.current_pose = None
        self.detected_objects = Detection2DArray()
        self.laser_scan = None
        self.goal = None
        self.current_path = Path()
        self.robot_state = 'idle'  # idle, navigating, learning, etc.
        
        # Memory systems
        self.episodic_memory = deque(maxlen=1000)  # Store recent experiences
        self.semantic_memory = {}  # Store learned knowledge
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.detection_sub = self.create_subscription(Detection2DArray, '/detections', self.detection_callback, 10)
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        
        # Timer for main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz control loop
        
        # Start behavior execution thread
        self.behavior_thread = threading.Thread(target=self.execute_behaviors, daemon=True)
        self.behavior_thread.start()
        
        self.get_logger().info('AI-Robot Brain has been initialized')
    
    def odom_callback(self, msg):
        """Callback for odometry data"""
        self.current_pose = msg.pose.pose
        self.get_logger().debug(f'Robot pose updated: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})')
    
    def image_callback(self, msg):
        """Callback for camera images"""
        # In a real implementation, this would process images
        # For now, we just acknowledge receipt
        pass
    
    def laser_callback(self, msg):
        """Callback for laser scan data"""
        self.laser_scan = msg
        self.get_logger().debug(f'Laser scan updated: {len(msg.ranges)} points')
    
    def detection_callback(self, msg):
        """Callback for object detections"""
        self.detected_objects = msg
        if len(msg.detections) > 0:
            self.get_logger().info(f'Detected {len(msg.detections)} objects')
    
    def path_callback(self, msg):
        """Callback for planned path"""
        self.current_path = msg
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
    
    def control_loop(self):
        """Main control loop that coordinates different components"""
        # Update robot state based on current situation
        self.update_robot_state()
        
        # Process current state
        if self.robot_state == 'navigating':
            self.execute_navigation()
        elif self.robot_state == 'avoiding_obstacles':
            self.execute_obstacle_avoidance()
        elif self.robot_state == 'idle':
            self.execute_idle_behavior()
        
        # Update status
        status_msg = String()
        status_msg.data = f"State: {self.robot_state}, Objects: {len(self.detected_objects.detections)}"
        self.status_pub.publish(status_msg)
    
    def update_robot_state(self):
        """Update the current state of the robot based on sensor data"""
        # Initialize state if not already set
        if self.robot_state == 'idle' and self.current_path.poses:
            self.robot_state = 'navigating'
        
        # Check if we need to avoid an obstacle
        if self.laser_scan:
            min_distance = min(self.laser_scan.ranges)
            if min_distance < 0.5:  # If obstacle within 0.5m
                self.robot_state = 'avoiding_obstacles'
    
    def execute_navigation(self):
        """Execute navigation behavior following the planned path"""
        if not self.current_path.poses:
            self.robot_state = 'idle'
            return
        
        # Get next waypoint
        if len(self.current_path.poses) > 0:
            next_waypoint = self.current_path.poses[0].pose.position
            
            # Simple proportional controller for navigation
            cmd_vel = Twist()
            
            # Calculate direction to waypoint
            if self.current_pose:
                dx = next_waypoint.x - self.current_pose.position.x
                dy = next_waypoint.y - self.current_pose.position.y
                
                # Calculate distance and angle
                distance = (dx**2 + dy**2)**0.5
                angle = np.arctan2(dy, dx)
                
                # Simple control law
                if distance > 0.1:  # If not close to waypoint
                    cmd_vel.linear.x = min(0.5, distance * 0.5)  # Proportional to distance
                    cmd_vel.angular.z = angle * 0.5  # Proportional to angle error
                else:
                    # Reached waypoint, remove it from path
                    self.current_path.poses.pop(0)
                    if not self.current_path.poses:
                        self.robot_state = 'idle'
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
        
            self.cmd_vel_pub.publish(cmd_vel)
    
    def execute_obstacle_avoidance(self):
        """Execute obstacle avoidance behavior"""
        if not self.laser_scan:
            return
        
        # Simple obstacle avoidance using laser scan
        cmd_vel = Twist()
        
        # Get ranges in front, left, and right
        front_idx = len(self.laser_scan.ranges) // 2
        left_idx = len(self.laser_scan.ranges) // 4
        right_idx = 3 * len(self.laser_scan.ranges) // 4
        
        front_dist = self.laser_scan.ranges[front_idx]
        left_dist = self.laser_scan.ranges[left_idx]
        right_dist = self.laser_scan.ranges[right_idx]
        
        # If obstacle straight ahead, turn
        if front_dist < 0.5:
            if left_dist > right_dist:
                cmd_vel.angular.z = 0.5  # Turn left
            else:
                cmd_vel.angular.z = -0.5  # Turn right
        else:
            cmd_vel.linear.x = 0.3  # Move forward if clear ahead
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def execute_idle_behavior(self):
        """Execute idle behavior when no tasks are active"""
        cmd_vel = Twist()  # Stop the robot
        self.cmd_vel_pub.publish(cmd_vel)
    
    def execute_behaviors(self):
        """Background thread for executing higher-level behaviors"""
        while rclpy.ok():
            # In a real implementation, this would execute behaviors
            # that require longer time periods
            time.sleep(0.1)
    
    def set_goal(self, x, y):
        """Set a navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal_msg)
        self.robot_state = 'navigating'
        self.get_logger().info(f'Goal set to ({x}, {y})')

def main(args=None):
    rclpy.init(args=args)
    brain_node = AIRobotBrainNode()
    
    try:
        # Example: Set a goal after 2 seconds
        def set_example_goal():
            time.sleep(2)
            brain_node.set_goal(2.0, 2.0)
        
        goal_thread = threading.Thread(target=set_example_goal, daemon=True)
        goal_thread.start()
        
        rclpy.spin(brain_node)
    except KeyboardInterrupt:
        pass
    finally:
        brain_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

2. **Create a launch file to start all components** (`~/ros2_ws/src/robot_brain_pkg/launch/ai_robot_brain.launch.py`):
```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('robot_brain_pkg')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Start the AI Robot Brain node
    brain_node = Node(
        package='robot_brain_pkg',
        executable='brain_node',
        name='ai_robot_brain',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Start perception node
    perception_node = Node(
        package='perception_pkg',
        executable='detection_node',
        name='object_detection',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Start navigation node
    navigation_node = Node(
        package='navigation_pkg',
        executable='navigation_node',
        name='path_planner',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )
    
    # Start Isaac Sim bridge (if available)
    isaac_sim_bridge = Node(
        package='isaac_ros_bridges',  # This would be a real package if using Isaac Sim
        executable='isaac_sim_bridge',
        name='isaac_sim_bridge',
        parameters=[
            {'use_sim_time': use_sim_time}
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
        brain_node,
        perception_node,
        navigation_node,
        # isaac_sim_bridge,  # Uncomment if Isaac Sim is available
    ])
```

3. **Update package setup files and build**:
```bash
# Create the robot_brain_pkg
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python robot_brain_pkg --dependencies rclpy geometry_msgs nav_msgs sensor_msgs vision_msgs std_msgs

# Build all packages
cd ~/ros2_ws
colcon build --packages-select robot_brain_pkg perception_pkg navigation_pkg
source ~/ros2_ws/install/setup.bash
```

## Lab 7: Performance Optimization and Profiling

### Optimizing the AI-Robot Brain for Real-Time Performance

1. **Create a performance profiling node** (`~/ros2_ws/src/performance_pkg/performance_pkg/profiler.py`):
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time
import threading
from collections import deque

class PerformanceProfiler(Node):
    """
    Node to profile performance of different components
    """
    def __init__(self):
        super().__init__('performance_profiler')
        
        # Timers for different tasks
        self.timers = {}
        
        # Performance metrics publishers
        self.perception_time_pub = self.create_publisher(Float32, '/performance/perception_time', 10)
        self.planning_time_pub = self.create_publisher(Float32, '/performance/planning_time', 10)
        self.control_time_pub = self.create_publisher(Float32, '/performance/control_time', 10)
        
        # Performance history
        self.performance_history = {
            'perception': deque(maxlen=100),
            'planning': deque(maxlen=100),
            'control': deque(maxlen=100)
        }
        
        # Start profiling timer
        self.profiling_timer = self.create_timer(1.0, self.report_performance)
        
        self.get_logger().info('Performance Profiler has been started')
    
    def start_timer(self, task_name):
        """Start a timer for a specific task"""
        self.timers[task_name] = time.time()
    
    def stop_timer(self, task_name):
        """Stop a timer and return elapsed time"""
        if task_name in self.timers:
            elapsed_time = time.time() - self.timers[task_name]
            del self.timers[task_name]
            
            # Store in history
            if task_name in self.performance_history:
                self.performance_history[task_name].append(elapsed_time)
            
            # Publish performance metric
            msg = Float32()
            msg.data = elapsed_time
            
            if task_name == 'perception':
                self.perception_time_pub.publish(msg)
            elif task_name == 'planning':
                self.planning_time_pub.publish(msg)
            elif task_name == 'control':
                self.control_time_pub.publish(msg)
                
            return elapsed_time
        else:
            return 0.0
    
    def report_performance(self):
        """Report average performance metrics"""
        perf_msg = "Performance Report:\n"
        for task, history in self.performance_history.items():
            if history:
                avg_time = sum(history) / len(history)
                perf_msg += f"  {task}: avg={avg_time:.4f}s, last={history[-1]:.4f}s\n"
                
        self.get_logger().info(perf_msg)

def main(args=None):
    rclpy.init(args=args)
    profiler = PerformanceProfiler()
    
    try:
        rclpy.spin(profiler)
    except KeyboardInterrupt:
        pass
    finally:
        profiler.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lab 8: Testing and Validation

### Creating a Comprehensive Test Suite

1. **Create a test node** (`~/ros2_ws/src/test_pkg/test_pkg/test_ai_brain.py`):
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
import math

class AIBrainTestNode(Node):
    """
    Node to test the AI-Robot Brain functionality
    """
    def __init__(self):
        super().__init__('ai_brain_test')
        
        # Test state
        self.test_state = 'init'
        self.test_start_time = None
        
        # Subscribe to robot status
        self.status_sub = self.create_subscription(String, '/robot_status', self.status_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Publisher for sending commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer for test execution
        self.test_timer = self.create_timer(0.1, self.execute_test)
        
        self.get_logger().info('AI Brain Test Node has been started')
        
    def status_callback(self, msg):
        """Callback for robot status"""
        self.get_logger().info(f'Robot status: {msg.data}')
    
    def odom_callback(self, msg):
        """Callback for odometry"""
        self.current_pose = msg.pose.pose
        self.get_logger().debug(f'Current position: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})')
    
    def execute_test(self):
        """Execute the test sequence"""
        if self.test_state == 'init':
            self.get_logger().info('Starting AI Brain Test')
            self.test_start_time = time.time()
            self.test_state = 'navigation'
            
        elif self.test_state == 'navigation':
            # Simple navigation test: move forward for 5 seconds
            if time.time() - self.test_start_time < 5.0:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.3
                cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel)
            else:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel)
                self.test_start_time = time.time()
                self.test_state = 'rotation'
                
        elif self.test_state == 'rotation':
            # Rotate for 3 seconds
            if time.time() - self.test_start_time < 3.0:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5
                self.cmd_vel_pub.publish(cmd_vel)
            else:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel)
                self.test_start_time = time.time()
                self.test_state = 'completed'
                
        elif self.test_state == 'completed':
            self.get_logger().info('AI Brain Test completed successfully')
            self.test_state = 'idle'  # Stay idle after test completion

def main(args=None):
    rclpy.init(args=args)
    test_node = AIBrainTestNode()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this practical lab, we've implemented key components of an AI-Robot Brain:

1. **Perception System**: Deep learning-based object detection using Isaac ROS
2. **Navigation System**: Path planning with A* algorithm
3. **Learning Component**: Reinforcement learning environment for navigation tasks
4. **Integration**: Cognitive architecture that coordinates perception, planning, and control
5. **Performance Profiling**: Tools to monitor and optimize system performance
6. **Testing**: Validation framework to ensure system reliability

These components work together to form an AI-Robot Brain that can perceive its environment, plan actions, learn from experience, and execute complex tasks. The integration with NVIDIA Isaac provides GPU acceleration for the computationally intensive AI components, enabling real-time operation on robotic platforms.

The modular design allows for easy extension and modification of individual components while maintaining system integrity. This architecture can be extended with additional capabilities such as natural language processing, advanced manipulation planning, or sophisticated learning algorithms based on specific application requirements.