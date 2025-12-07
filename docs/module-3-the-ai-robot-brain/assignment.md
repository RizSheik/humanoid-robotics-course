---
id: module-3-assignment
title: Module 3 — The AI Robot Brain | Chapter 6 — Assignment
sidebar_label: Chapter 6 — Assignment
sidebar_position: 6
---

# Module 3 — The AI Robot Brain

## Chapter 6 — Assignment

### Assignment Overview

This assignment focuses on implementing a complete AI Robot Brain system for a humanoid robot. The objective is to create an integrated system that includes perception, decision-making, learning, and action execution capabilities. You will develop deep learning models for perception, reinforcement learning agents for control, and natural language processing for human-robot interaction, all integrated within the NVIDIA Isaac ecosystem.

### Learning Objectives

Upon completion, you will be able to:

1. Design and implement AI perception systems for humanoid robots
2. Create reinforcement learning agents for robot control
3. Develop natural language processing components for human-robot interaction
4. Integrate AI components with robot control systems
5. Train and deploy AI models in simulation environments
6. Evaluate and validate AI Robot Brain performance

### Assignment Requirements

#### Core Components

You must implement the following core components:

1. **Computer Vision System**: Object detection, recognition, and tracking
2. **Reinforcement Learning Controller**: Navigation and manipulation control
3. **Natural Language Processing**: Command interpretation and response generation
4. **Memory System**: Experience storage and retrieval for continual learning
5. **Cognitive Architecture**: Integration of perception, planning, and action
6. **Simulation Integration**: Full integration with Isaac Sim environment

#### Technical Requirements

1. **NVIDIA Isaac Compatibility**: All components must run efficiently on Isaac platform
2. **ROS/ROS2 Integration**: Proper message passing and communication patterns
3. **Real-time Performance**: Systems must operate at appropriate frequencies
4. **Safety Considerations**: Implement safety checks and fail-safe mechanisms
5. **Scalability**: Architecture should support addition of new capabilities

#### Documentation and Testing

1. **Technical Documentation**: API documentation, architecture diagrams, and user guides
2. **Unit Tests**: Comprehensive tests for each component
3. **Integration Tests**: Tests for component interaction
4. **Performance Benchmarks**: Execution time and accuracy metrics
5. **Validation Results**: Demonstration of system capabilities

### Detailed Implementation Steps

#### Step 1: Project Structure and Setup

Create a comprehensive ROS 2 workspace for the AI Robot Brain system:

```
ai_robot_brain/
├── ai_perception/
│   ├── ai_perception/
│   │   ├── __init__.py
│   │   ├── object_detection.py
│   │   ├── object_tracking.py
│   │   ├── scene_understanding.py
│   │   └── sensor_fusion.py
│   ├── config/
│   │   ├── perception_params.yaml
│   │   └── model_configs.yaml
│   ├── models/
│   │   ├── yolov8.pt
│   │   └── segmentation_model.pt
│   ├── launch/
│   │   └── perception_pipeline.launch.py
│   ├── test/
│   │   └── test_perception.py
│   ├── setup.py
│   └── package.xml
├── rl_controller/
│   ├── rl_controller/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py
│   │   ├── environment.py
│   │   ├── trainer.py
│   │   └── action_selector.py
│   ├── config/
│   │   └── rl_params.yaml
│   ├── models/
│   │   └── trained_agent.pth
│   ├── launch/
│   │   └── rl_agent.launch.py
│   ├── test/
│   │   └── test_rl.py
│   ├── setup.py
│   └── package.xml
├── nlp_system/
│   ├── nlp_system/
│   │   ├── __init__.py
│   │   ├── command_parser.py
│   │   ├── intent_classifier.py
│   │   ├── response_generator.py
│   │   └── dialogue_manager.py
│   ├── config/
│   │   └── nlp_params.yaml
│   ├── models/
│   │   └── language_model.pt
│   ├── launch/
│   │   └── nlp_pipeline.launch.py
│   ├── test/
│   │   └── test_nlp.py
│   ├── setup.py
│   └── package.xml
├── memory_system/
│   ├── memory_system/
│   │   ├── __init__.py
│   │   ├── episodic_memory.py
│   │   ├── semantic_memory.py
│   │   ├── experience_buffer.py
│   │   └── continual_learning.py
│   ├── config/
│   │   └── memory_params.yaml
│   ├── launch/
│   │   └── memory_system.launch.py
│   ├── test/
│   │   └── test_memory.py
│   ├── setup.py
│   └── package.xml
├── cognitive_arch/
│   ├── cognitive_arch/
│   │   ├── __init__.py
│   │   ├── brain_node.py
│   │   ├── decision_maker.py
│   │   ├── state_manager.py
│   │   └── behavior_controller.py
│   ├── config/
│   │   └── arch_params.yaml
│   ├── launch/
│   │   └── ai_brain.launch.py
│   ├── test/
│   │   └── test_arch.py
│   ├── setup.py
│   └── package.xml
├── isaac_integration/
│   ├── isaac_integration/
│   │   ├── __init__.py
│   │   ├── sim_bridge.py
│   │   ├── sensor_processor.py
│   │   └── action_executor.py
│   ├── config/
│   │   └── isaac_params.yaml
│   ├── launch/
│   │   └── isaac_bridge.launch.py
│   ├── setup.py
│   └── package.xml
└── ai_robot_brain/
    ├── launch/
    │   └── complete_ai_brain.launch.py
    ├── config/
    │   └── ai_brain_params.yaml
    ├── docs/
    │   ├── architecture.md
    │   └── usage_guide.md
    ├── test/
    │   └── integration_tests.py
    └── package.xml
```

#### Step 2: Implement Computer Vision System

Create the `ai_perception/ai_perception/object_detection.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict
import time

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load YOLO model
        self.get_logger().info('Loading YOLO model...')
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        self.get_logger().info('Model loaded successfully')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        self.declare_parameter('publish_rate', 10)
        self.publish_rate = self.get_parameter('publish_rate').value

        self.declare_parameter('class_filter', ['person', 'bicycle', 'car', 'motorcycle',
                                               'airplane', 'bus', 'train', 'truck', 'boat',
                                               'traffic light', 'fire hydrant', 'stop sign',
                                               'parking meter', 'bench', 'bird', 'cat',
                                               'dog', 'horse', 'sheep', 'cow', 'elephant',
                                               'bear', 'zebra', 'giraffe', 'backpack',
                                               'umbrella', 'handbag', 'tie', 'suitcase',
                                               'frisbee', 'skis', 'snowboard', 'sports ball',
                                               'kite', 'baseball bat', 'baseball glove',
                                               'skateboard', 'surfboard', 'tennis racket',
                                               'bottle', 'wine glass', 'cup', 'fork', 'knife',
                                               'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                                               'orange', 'broccoli', 'carrot', 'hot dog',
                                               'pizza', 'donut', 'cake', 'chair', 'couch',
                                               'potted plant', 'bed', 'dining table', 'toilet',
                                               'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                               'cell phone', 'microwave', 'oven', 'toaster',
                                               'sink', 'refrigerator', 'book', 'clock', 'vase',
                                               'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
        self.class_filter = self.get_parameter('class_filter').value

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/ai_robot_brain/detections',
            10
        )

        self.debug_img_pub = self.create_publisher(
            Image,
            '/ai_robot_brain/debug_detections',
            10
        )

        # Camera info for 3D reconstruction (if available)
        self.camera_info = None
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Process timer
        self.process_timer = self.create_timer(1.0/self.publish_rate, self.process_callback)
        self.last_image = None
        self.process_times = []

        self.get_logger().info('Object Detection Node initialized')

    def camera_info_callback(self, msg):
        """Store camera intrinsics for 3D reconstruction"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Store latest image for processing"""
        self.last_image = msg

    def process_callback(self):
        """Process latest image with object detection"""
        if self.last_image is None:
            return

        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding='bgr8')

            # Run object detection
            results = self.model(cv_image)

            # Create detection message
            detections_msg = Detection2DArray()
            detections_msg.header = self.last_image.header

            # Process results
            if len(results.pred[0]) > 0:
                for *box, conf, cls in results.pred[0].tolist():
                    # Filter by confidence
                    if conf < self.confidence_threshold:
                        continue

                    # Filter by class
                    class_name = self.model.names[int(cls)]
                    if class_name not in self.class_filter:
                        continue

                    # Create detection
                    detection = Detection2D()
                    detection.header = self.last_image.header

                    # Set bounding box
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    detection.bbox.center.x = float(center_x)
                    detection.bbox.center.y = float(center_y)
                    detection.bbox.size_x = float(width)
                    detection.bbox.size_y = float(height)

                    # Set results (classification)
                    result = detection.results.add()
                    result.id = str(int(cls))
                    result.score = conf

                    # Add detection to array
                    detections_msg.detections.append(detection)

            # Publish detections
            self.detection_pub.publish(detections_msg)

            # Create debug image with bounding boxes
            debug_image = self.draw_detections(cv_image, results)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = self.last_image.header
            self.debug_img_pub.publish(debug_msg)

            # Measure processing time
            process_time = time.time() - start_time
            self.process_times.append(process_time)

            if len(self.process_times) > 10:
                self.process_times.pop(0)

            avg_time = sum(self.process_times) / len(self.process_times) if self.process_times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0

            self.get_logger().info(
                f'Processed {len(detections_msg.detections)} detections. '
                f'Avg time: {avg_time:.3f}s ({fps:.1f} FPS)'
            )

        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')

    def draw_detections(self, image, results):
        """Draw bounding boxes on image for debugging"""
        result_img = image.copy()

        if len(results.pred[0]) > 0:
            for *box, conf, cls in results.pred[0].tolist():
                if conf < self.confidence_threshold:
                    continue

                class_name = self.model.names[int(cls)]
                if class_name not in self.class_filter:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Draw rectangle
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f'{class_name}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return result_img

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Object Detection Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Implement Reinforcement Learning Controller

Create the `rl_controller/rl_controller/ppo_agent.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)

        # Critic (value network)
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        features = self.shared_layers(state)

        # Actor
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.sigmoid(self.actor_std(features)) + 1e-5

        # Critic
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(state)

        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

class PPOAgent(Node):
    def __init__(self):
        super().__init__('ppo_rl_agent')

        # Neural network parameters
        self.state_dim = 25  # 20 LiDAR readings + 2 robot pose + 3 velocities
        self.action_dim = 2  # linear vel, angular vel
        self.hidden_dim = 256
        self.lr = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.entropy_coef = 0.01

        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize networks
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        # Experience buffer
        self.buffer = deque(maxlen=10000)

        # Environment setup
        self.laser_scan = None
        self.odometry = None
        self.target_pos = [5.0, 0.0]  # Example target position

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float32, '/rl/reward', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Training timer
        self.train_timer = self.create_timer(0.1, self.train_step)
        self.step_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        self.get_logger().info('PPO RL Agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_scan = msg

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odometry = msg

    def get_state(self):
        """Extract state from sensor data"""
        if self.laser_scan is None or self.odometry is None:
            # Return default state if no sensor data
            return torch.zeros(self.state_dim).to(self.device)

        # 20 LiDAR readings (sampled evenly)
        laser_readings = self.laser_scan.ranges
        state = []

        # Sample 20 readings evenly from the scan
        step = len(laser_readings) // 20
        for i in range(0, len(laser_readings), step):
            if i < len(laser_readings):
                dist = laser_readings[i]
                if math.isnan(dist) or dist > 10.0:
                    dist = 10.0  # Max distance
                state.append(dist)

        # Pad if not enough readings
        while len(state) < 20:
            state.append(10.0)

        # Robot pose (x, y)
        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y
        state.extend([x, y])

        # Robot velocities (linear x, angular z)
        lin_vel = self.odometry.twist.twist.linear.x
        ang_vel = self.odometry.twist.twist.angular.z
        state.extend([lin_vel, ang_vel])

        # Target direction
        target_dir = math.atan2(self.target_pos[1] - y, self.target_pos[0] - x)

        # Robot orientation
        from quaternion import euler_from_quaternion
        orient = self.odometry.pose.pose.orientation
        _, _, theta = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])

        # Relative target direction
        rel_target_dir = target_dir - theta
        state.append(math.cos(rel_target_dir))
        state.append(math.sin(rel_target_dir))

        # Normalize state
        state = np.array(state, dtype=np.float32)

        # Normalize different parts differently
        for i in range(20):  # LiDAR readings: normalize 0-10m to [-1, 1]
            state[i] = np.clip(state[i] / 5.0 - 1.0, -1.0, 1.0)

        # Position: normalize -10 to 10 to [-1, 1]
        state[20] = np.clip(state[20] / 10.0, -1.0, 1.0)  # x
        state[21] = np.clip(state[21] / 10.0, -1.0, 1.0)  # y

        # Velocities: normalize -2 to 2 to [-1, 1]
        state[22] = np.clip(state[22] / 2.0, -1.0, 1.0)  # linear vel
        state[23] = np.clip(state[23] / 2.0, -1.0, 1.0)  # angular vel

        # Direction: already normalized
        state[24] = state[24]  # cos(rel_target_dir)
        state[25] = state[25]  # sin(rel_target_dir)

        return torch.FloatTensor(state).to(self.device)

    def compute_reward(self, old_state, action, new_state):
        """Compute reward based on state transition"""
        if self.odometry is None:
            return 0.0

        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y

        # Distance to target
        dist_to_target = math.sqrt((x - self.target_pos[0])**2 + (y - self.target_pos[1])**2)

        # Reward based on distance to target (closer is better)
        target_reward = max(0, 10 - dist_to_target)  # Decreases as distance increases

        # Penalty for getting too close to obstacles
        if self.laser_scan:
            min_dist = min(self.laser_scan.ranges) if self.laser_scan.ranges else float('inf')
            if min_dist < 0.5:  # Closer than 0.5m to obstacle
                obstacle_penalty = -5.0
            elif min_dist < 1.0:  # Between 0.5m and 1.0m
                obstacle_penalty = -2.0
            else:
                obstacle_penalty = 0.0
        else:
            obstacle_penalty = 0.0

        # Small penalty for large control actions
        action_penalty = -0.1 * (abs(action[0]) + abs(action[1]))  # Discourage aggressive movements

        # Bonus for making progress toward target
        progress_bonus = 0.0
        if hasattr(self, 'prev_dist_to_target'):
            if dist_to_target < self.prev_dist_to_target:
                progress_bonus = 1.0

        self.prev_dist_to_target = dist_to_target

        total_reward = target_reward + obstacle_penalty + action_penalty + progress_bonus
        return total_reward

    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            action_mean, action_std, state_value = self.actor_critic.forward(state.unsqueeze(0))

            # Create distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], state_value.cpu().numpy()[0]

    def train_step(self):
        """Execute one step of training"""
        state = self.get_state()
        action, log_prob, state_value = self.select_action(state)

        # Execute action (in sim, publish to cmd_vel)
        cmd_msg = Twist()
        cmd_msg.linear.x = float(action[0]) * 0.5  # Scale linear velocity
        cmd_msg.angular.z = float(action[1]) * 1.0  # Scale angular velocity
        self.cmd_pub.publish(cmd_msg)

        # Calculate reward for action
        reward = self.compute_reward(None, action, state)
        self.current_episode_reward += reward

        # Check for episode end conditions
        done = self.check_done_condition()

        # Store experience in buffer
        experience = (state.cpu().numpy(), action, reward, log_prob, state_value, done)
        self.buffer.append(experience)

        # Publish reward
        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)

        # Update episode tracking
        self.step_count += 1
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

            # Log episode info
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                self.get_logger().info(f'Episode {len(self.episode_rewards)}, Average Reward: {avg_reward:.2f}')

    def check_done_condition(self):
        """Check if episode should end"""
        if self.odometry is None:
            return False

        x = self.odometry.pose.pose.position.x
        y = self.odometry.pose.pose.position.y

        # Distance to target
        dist_to_target = math.sqrt((x - self.target_pos[0])**2 + (y - self.target_pos[1])**2)

        # Check for collision
        collision = False
        if self.laser_scan:
            min_dist = min(self.laser_scan.ranges) if self.laser_scan.ranges else float('inf')
            collision = min_dist < 0.2  # Collision if closer than 0.2m to obstacle

        # Episode ends if close to target or collided
        done = dist_to_target < 0.5 or collision or self.step_count > 1000

        if done:
            self.step_count = 0  # Reset step count for next episode

        return done

    def update_policy(self):
        """Update policy using PPO algorithm"""
        if len(self.buffer) < 32:  # Not enough samples to update
            return

        # Sample batch of experiences
        batch_size = min(64, len(self.buffer))
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        # Extract batch components
        states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
        state_values = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp[5] for exp in batch]).to(self.device)

        # Compute discounted rewards
        discounted_rewards = []
        running_add = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_add = 0
            running_add = reward + (self.gamma * running_add)
            discounted_rewards.insert(0, running_add)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Optimize policy K times
        for _ in range(self.K_epochs):
            # Get current policy values
            action_means, action_stds, state_vals = self.actor_critic(states)

            # Calculate ratio
            dist = torch.distributions.Normal(action_means, action_stds)
            cur_log_probs = dist.log_prob(actions)

            ratio = torch.exp(cur_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))

            # Calculate surrogates
            advantages = discounted_rewards - state_vals.squeeze()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_vals.squeeze(), discounted_rewards)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

def main(args=None):
    rclpy.init(args=args)
    agent = PPOAgent()

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Shutting down PPO RL Agent')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Implement Natural Language Processing System

Create the `nlp_system/nlp_system/command_parser.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nlp_system.intent_classifier import IntentClassifier
from nlp_system.response_generator import ResponseGenerator
import re
import json
from typing import Dict, List, Optional

class CommandParserNode(Node):
    def __init__(self):
        super().__init__('command_parser_node')

        # Initialize NLP components
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()

        # Publishers and subscribers
        self.speech_sub = self.create_subscription(
            String,
            '/ai_robot_brain/speech_recognition/text',
            self.speech_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tts_pub = self.create_publisher(String, '/ai_robot_brain/tts/text', 10)

        # Robot state
        self.robot_state = {
            'location': [0.0, 0.0],
            'battery_level': 100.0,
            'current_task': None,
            'objects_detected': []
        }

        # Command handlers
        self.command_handlers = {
            'move': self.handle_move_command,
            'stop': self.handle_stop_command,
            'greet': self.handle_greet_command,
            'inform': self.handle_inform_command,
            'question': self.handle_question_command,
            'action': self.handle_action_command
        }

        self.get_logger().info('Command Parser Node initialized')

    def speech_callback(self, msg):
        """Process incoming speech command"""
        command_text = msg.data
        self.get_logger().info(f'Received command: "{command_text}"')

        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(command_text)

        if confidence < 0.3:  # Low confidence
            response = self.response_generator.generate_uncertain_response(command_text)
            self.speak_response(response)
            return

        # Extract entities
        entities = self.extract_entities(command_text, intent)

        # Execute command based on intent
        if intent in self.command_handlers:
            try:
                handler = self.command_handlers[intent]
                response = handler(command_text, entities)
                self.speak_response(response)
            except Exception as e:
                error_resp = f"Sorry, I encountered an error processing that command: {str(e)}"
                self.speak_response(error_resp)
                self.get_logger().error(f'Error in command handler: {str(e)}')
        else:
            # Unknown intent
            response = self.response_generator.generate_unknown_command_response(command_text)
            self.speak_response(response)

    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract named entities from text based on intent"""
        entities = {}
        text_lower = text.lower()

        # Location entities
        location_pattern = r'(to the )?(kitchen|bedroom|living room|office|bathroom|garage|garden|hallway|entrance|dining room|study|storage|workshop|patio|museum|library|cafeteria)'
        location_match = re.search(location_pattern, text_lower)
        if location_match:
            entities['location'] = location_match.group(2)

        # Object entities
        object_pattern = r'(the )?(ball|cup|bottle|book|toy|phone|keys|wallet|pen|notebook|laptop|tablet|glasses|hat|jacket|shoes|backpack|umbrella|camera|watch|ring|earrings|belt|tie|scarf|gloves|purse|briefcase)'
        object_match = re.search(object_pattern, text_lower)
        if object_match:
            entities['object'] = object_match.group(2)

        # Person entities
        person_pattern = r'(to )?(me|you|him|her|them|john|jane|bob|alice|charlie|diana|eve|frank|grace|heidi|ivan|judy|mallory|oscar|peggy|sybil|trudy|wendy|alex|sam|morgan|casey|jamie)'
        person_match = re.search(person_pattern, text_lower)
        if person_match:
            entities['person'] = person_match.group(2)

        # Direction entities
        direction_pattern = r'(go )?(forward|backward|back|ahead|left|right|upstairs|downstairs|inside|outside|around|over|under|through)'
        direction_match = re.search(direction_pattern, text_lower)
        if direction_match:
            entities['direction'] = direction_match.group(2)

        # Numbers/entities
        number_pattern = r'(\d+(?:\.\d+)?)'
        numbers = re.findall(number_pattern, text_lower)
        if numbers:
            entities['numbers'] = numbers

        return entities

    def handle_move_command(self, command: str, entities: Dict) -> str:
        """Handle movement commands"""
        cmd_msg = Twist()

        # Determine movement based on command
        if 'forward' in command.lower() or 'ahead' in command.lower():
            cmd_msg.linear.x = 0.3
            cmd_msg.angular.z = 0.0
            response = "Moving forward."
        elif 'back' in command.lower() or 'backward' in command.lower():
            cmd_msg.linear.x = -0.2
            cmd_msg.angular.z = 0.0
            response = "Moving backward."
        elif 'left' in command.lower():
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = 0.5
            response = "Turning left."
        elif 'right' in command.lower():
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = -0.5
            response = "Turning right."
        elif 'location' in entities:
            # Navigate to specific location
            location = entities['location']
            response = f"Going to the {location}."
            # In a real implementation, this would trigger navigation to a specific location
            cmd_msg.linear.x = 0.2  # Placeholder movement
            cmd_msg.angular.z = 0.0
        else:
            # Generic movement command
            response = "Moving in the requested direction."
            cmd_msg.linear.x = 0.2
            cmd_msg.angular.z = 0.0

        # Publish movement command
        self.cmd_pub.publish(cmd_msg)

        return response

    def handle_stop_command(self, command: str, entities: Dict) -> str:
        """Handle stop commands"""
        # Stop the robot
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_pub.publish(cmd_msg)

        return "Stopping now."

    def handle_greet_command(self, command: str, entities: Dict) -> str:
        """Handle greeting commands"""
        greetings = ["Hello!", "Hi there!", "Greetings!", "Nice to meet you!", "How can I help you?"]

        # Check for personalization
        if 'person' in entities:
            person_name = entities['person']
            return f"Hello {person_name}! How can I assist you today?"
        else:
            import random
            return random.choice(greetings)

    def handle_inform_command(self, command: str, entities: Dict) -> str:
        """Handle informational commands"""
        if 'battery' in command.lower() or 'power' in command.lower():
            return f"My battery level is {self.robot_state['battery_level']:.1f} percent."
        elif 'location' in command.lower() or 'where' in command.lower():
            x, y = self.robot_state['location']
            return f"I am currently at position ({x:.2f}, {y:.2f})."
        elif 'task' in command.lower() or 'doing' in command.lower():
            task = self.robot_state['current_task'] or "nothing specific"
            return f"I am currently {task}."
        else:
            return "I can provide information about my battery level, location, and current task."

    def handle_question_command(self, command: str, entities: Dict) -> str:
        """Handle question commands"""
        # Process various types of questions
        if 'who' in command.lower() or 'what' in command.lower():
            if 'made' in command.lower() or 'created' in command.lower():
                return "I am an AI Robot Brain created for humanoid robotics applications."
            elif 'can' in command.lower() or 'able' in command.lower():
                abilities = [
                    "I can navigate environments, recognize objects, understand speech, and interact with humans.",
                    "I can move around, recognize objects, and respond to voice commands."
                ]
                import random
                return random.choice(abilities)

        elif 'how' in command.lower():
            if 'work' in command.lower():
                return "I use AI and machine learning to perceive, reason, and act in my environment."
            elif 'are' in command.lower():
                return "I am functioning well, thank you for asking!"

        elif 'when' in command.lower():
            if 'created' in command.lower():
                return "I was created as part of the AI Robot Brain system for humanoid robotics."

        # Default response for questions
        return self.response_generator.generate_general_response(command)

    def handle_action_command(self, command: str, entities: Dict) -> str:
        """Handle action commands like pick up, put down, etc."""
        # For now, just acknowledge
        if 'pick' in command.lower() or 'grab' in command.lower():
            obj = entities.get('object', 'something')
            return f"I would pick up the {obj} if I had a manipulator arm."
        elif 'drop' in command.lower() or 'put' in command.lower():
            obj = entities.get('object', 'it')
            return f"I would put down {obj} if I had items in my grippers."
        elif 'find' in command.lower() or 'look' in command.lower():
            obj = entities.get('object', 'something')
            return f"Looking for {obj}. I would scan the environment visually."
        else:
            return "I can perform various actions like navigation and simple manipulations."

    def speak_response(self, text: str):
        """Publish response to text-to-speech system"""
        response_msg = String()
        response_msg.data = text
        self.tts_pub.publish(response_msg)
        self.get_logger().info(f"Responding: '{text}'")

class IntentClassifier:
    def __init__(self):
        # Simple intent classification using keyword matching
        # In a real system, this would use machine learning models
        self.intent_patterns = {
            'move': [
                r'\bg(o|ing)\b', r'\bm(ove|ovement)\b', r'\bw(alk|ander)\b',
                r'\br(otate|evolve|ight|ound)\b', r'\bt(urn|oward)\b'
            ],
            'stop': [
                r'\bst(op|ation)\b', r'\bhalt\b', r'\bpause\b', r'\bfreez(e|ing)\b'
            ],
            'greet': [
                r'\b(hi|hello|hey|greetings)\b', r'\bhowdy\b', r'\bgood (morning|afternoon|evening)\b'
            ],
            'inform': [
                r'\bwh(at|ere|ich)\b', r'\btell\b', r'\bsh(ow|are)\b',
                r'\bmy\b.*\bbattery\b', r'\bcurren(t|cy)\b'
            ],
            'question': [
                r'\b(how|what|why|when|where|who|which)\b.*\b(are|is|can|will|would|should|could)\b',
                r'\bcould you\b', r'\bcan you\b', r'\bwill you\b', r'\bcould\b', r'\bcan\b'
            ],
            'action': [
                r'\bpick\b|\bgrab\b|\btake\b', r'\bdrop\b|\bput\b|\brelease\b',
                r'\bfind\b|\blook for\b|\bsearch\b', r'\bfollow\b|\bcome to\b|\bc(ome|oming)'
            ]
        }

    def classify_intent(self, text: str) -> tuple:
        """Classify intent of the given text"""
        text_lower = text.lower()
        scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[intent] = score

        # Find the intent with the highest score
        if scores:
            best_intent = max(scores, key=scores.get)
            best_score = scores[best_intent]

            # Calculate confidence (normalize by number of patterns)
            max_possible_score = len(self.intent_patterns[best_intent])
            if max_possible_score > 0:
                confidence = best_score / max_possible_score
            else:
                confidence = 0.0

            return best_intent, confidence
        else:
            return 'unknown', 0.0

class ResponseGenerator:
    def __init__(self):
        # Predefined responses for various situations
        self.responses = {
            'uncertain': [
                "I'm not sure I understood that correctly.",
                "Could you repeat that?",
                "I didn't catch that. Could you say it again?",
                "I'm sorry, I didn't understand that command."
            ],
            'unknown_command': [
                "I don't know how to do that yet.",
                "That's beyond my current capabilities.",
                "I'm not programmed to perform that action.",
                "I'm still learning and don't know how to handle that."
            ],
            'confirmation': [
                "OK, I'll do that.",
                "I understand. Proceeding with the command.",
                "Acknowledged. Processing your request.",
                "Got it. Executing the requested action."
            ]
        }

    def generate_uncertain_response(self, command: str) -> str:
        """Generate response when confidence is low"""
        import random
        return random.choice(self.responses['uncertain'])

    def generate_unknown_command_response(self, command: str) -> str:
        """Generate response for unknown commands"""
        import random
        return random.choice(self.responses['unknown_command'])

    def generate_general_response(self, command: str) -> str:
        """Generate general response for common queries"""
        return "I understand you're asking me something, but I need to learn more to provide a specific answer."

def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Command Parser Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 5: Create Memory System

Create the `memory_system/memory_system/episodic_memory.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import pickle
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import threading
import time
from collections import OrderedDict

class EpisodicMemory(Node):
    def __init__(self):
        super().__init__('episodic_memory_node')

        # Memory storage
        self.memory = OrderedDict()
        self.capacity = 1000  # Maximum number of experiences to store
        self.storage_path = '/tmp/robot_memory.pkl'

        # Memory management
        self.access_count = {}  # Track how often each experience is accessed
        self.importance_scores = {}  # How important each experience is
        self.lock = threading.Lock()

        # Publishers and subscribers for memory-related topics
        self.memory_query_sub = self.create_subscription(
            String,
            '/ai_robot_brain/memory/query',
            self.query_callback,
            10
        )

        self.memory_store_sub = self.create_subscription(
            String,
            '/ai_robot_brain/memory/store',
            self.store_callback,
            10
        )

        self.memory_response_pub = self.create_publisher(
            String,
            '/ai_robot_brain/memory/response',
            10
        )

        # Memory persistence timer
        self.save_timer = self.create_timer(30.0, self.save_memory)  # Save every 30 seconds

        # Load existing memory
        self.load_memory()

        self.get_logger().info(f'Episodic Memory initialized with capacity {self.capacity}')

    def store_experience(self, experience: Dict[str, Any], importance: float = 0.5) -> str:
        """Store a new experience in memory"""
        with self.lock:
            # Create a unique ID for this experience
            timestamp = time.time()
            experience_id = f"exp_{int(timestamp * 1000000)}"  # Use microseconds for uniqueness

            # Store the experience with metadata
            experience_entry = {
                'id': experience_id,
                'timestamp': timestamp,
                'date': datetime.now().isoformat(),
                'data': experience,
                'importance': float(importance),
                'access_count': 0
            }

            # Add to memory
            self.memory[experience_id] = experience_entry
            self.importance_scores[experience_id] = importance
            self.access_count[experience_id] = 0

            # Remove oldest experiences if capacity exceeded
            if len(self.memory) > self.capacity:
                oldest_key = next(iter(self.memory))
                del self.memory[oldest_key]
                del self.importance_scores[oldest_key]
                del self.access_count[oldest_key]

            self.get_logger().info(f'Stored experience: {experience_id}')
            return experience_id

    def retrieve_experience(self, experience_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific experience from memory"""
        with self.lock:
            if experience_id in self.memory:
                entry = self.memory[experience_id]
                entry['access_count'] += 1  # Update access count
                self.access_count[experience_id] += 1
                return entry['data']
            return None

    def query_by_content(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Find experiences similar to the query"""
        with self.lock:
            matches = []

            for exp_id, entry in self.memory.items():
                # Simple similarity calculation based on keywords in the query
                similarity = self._calculate_similarity(entry['data'], query)

                if similarity > 0.1:  # Threshold for relevance
                    matches.append({
                        'id': exp_id,
                        'data': entry['data'],
                        'similarity': similarity,
                        'timestamp': entry['timestamp'],
                        'importance': entry['importance']
                    })

            # Sort by similarity and importance
            matches.sort(key=lambda x: x['similarity'] * x['importance'], reverse=True)

            return matches[:limit]

    def _calculate_similarity(self, experience: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Calculate similarity between experience and query"""
        similarity = 0.0

        # Compare based on common keys
        for key, query_value in query.items():
            if key in experience:
                exp_value = experience[key]

                # For numeric values, calculate inverse distance
                if isinstance(query_value, (int, float)) and isinstance(exp_value, (int, float)):
                    # Normalize the difference
                    diff = abs(query_value - exp_value)
                    # Use exponential decay function: exp(-diff)
                    similarity += np.exp(-diff * 0.1)

                # For string values, use fuzzy matching
                elif isinstance(query_value, str) and isinstance(exp_value, str):
                    if query_value.lower() in exp_value.lower() or exp_value.lower() in query_value.lower():
                        similarity += 1.0
                    elif query_value.lower() == exp_value.lower():
                        similarity += 2.0

        # Normalize similarity (0 to 1)
        max_possible = len(query) * 2  # Max if all strings match perfectly
        return similarity / max_possible if max_possible > 0 else 0.0

    def query_callback(self, msg: String):
        """Handle memory query requests"""
        try:
            # Parse the query from the message
            try:
                query_data = json.loads(msg.data)
            except json.JSONDecodeError:
                # If not JSON, treat as simple text query
                query_data = {'text': msg.data}

            # Extract query parameters
            limit = query_data.get('limit', 5)
            experience_id = query_data.get('id')

            # Perform the query
            if experience_id:
                # Retrieve specific experience
                result = self.retrieve_experience(experience_id)
                if result is not None:
                    response_data = {
                        'type': 'retrieval',
                        'success': True,
                        'data': result,
                        'id': experience_id
                    }
                else:
                    response_data = {
                        'type': 'retrieval',
                        'success': False,
                        'error': f'Experience with ID {experience_id} not found'
                    }
            else:
                # Query by content
                results = self.query_by_content(query_data, limit)
                response_data = {
                    'type': 'query',
                    'success': True,
                    'results': results,
                    'count': len(results)
                }

            # Publish response
            response_msg = String()
            response_msg.data = json.dumps(response_data)
            self.memory_response_pub.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing query: {str(e)}')
            error_msg = String()
            error_msg.data = json.dumps({
                'type': 'error',
                'success': False,
                'error': str(e)
            })
            self.memory_response_pub.publish(error_msg)

    def store_callback(self, msg: String):
        """Handle memory store requests"""
        try:
            # Parse the experience data from the message
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                # If not JSON, treat as simple experience data
                data = {'event': msg.data, 'timestamp': time.time()}

            # Extract importance if provided (default to 0.5)
            importance = data.get('importance', 0.5)

            # Remove importance from the actual experience data
            if 'importance' in data:
                data.pop('importance')

            # Store the experience
            exp_id = self.store_experience(data, importance)

            # Publish confirmation
            response_msg = String()
            response_msg.data = json.dumps({
                'type': 'store',
                'success': True,
                'id': exp_id
            })
            self.memory_response_pub.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'Error storing experience: {str(e)}')
            error_msg = String()
            error_msg.data = json.dumps({
                'type': 'error',
                'success': False,
                'error': str(e)
            })
            self.memory_response_pub.publish(error_msg)

    def save_memory(self):
        """Save memory to persistent storage"""
        with self.lock:
            try:
                with open(self.storage_path, 'wb') as f:
                    pickle.dump({
                        'memory': self.memory,
                        'access_count': self.access_count,
                        'importance_scores': self.importance_scores
                    }, f)
                self.get_logger().info(f'Memory saved to {self.storage_path}')
            except Exception as e:
                self.get_logger().error(f'Error saving memory: {str(e)}')

    def load_memory(self):
        """Load memory from persistent storage"""
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                self.memory = data.get('memory', OrderedDict())
                self.access_count = data.get('access_count', {})
                self.importance_scores = data.get('importance_scores', {})
            self.get_logger().info(f'Memory loaded from {self.storage_path}')
        except FileNotFoundError:
            self.get_logger().info('No existing memory file found, starting fresh')
        except Exception as e:
            self.get_logger().error(f'Error loading memory: {str(e)}')

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory"""
        with self.lock:
            total_experiences = len(self.memory)
            total_accesses = sum(self.access_count.values())
            avg_importance = np.mean(list(self.importance_scores.values())) if self.importance_scores else 0.0

            return {
                'total_experiences': total_experiences,
                'total_accesses': total_accesses,
                'average_importance': avg_importance,
                'capacity': self.capacity,
                'utilization': total_experiences / self.capacity if self.capacity > 0 else 0
            }

def main(args=None):
    rclpy.init(args=args)
    node = EpisodicMemory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Episodic Memory Node')
        node.save_memory()  # Save on shutdown
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 6: Create Cognitive Architecture

Create the `cognitive_arch/cognitive_arch/brain_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import time
import threading
from typing import Dict, Any, Optional
import json

class CognitiveArchitectureNode(Node):
    def __init__(self):
        super().__init__('cognitive_architecture_node')

        # System state and context
        self.current_state = {
            'location': [0.0, 0.0, 0.0],  # x, y, theta
            'battery_level': 100.0,
            'stuck_counter': 0,
            'current_task': 'idle',
            'task_progress': 0.0,
            'last_decision_time': time.time(),
            'attention_focus': None,
            'motivational_state': 'exploration'  # exploration, task_completion, safety
        }

        # Task queue
        self.task_queue = []
        self.active_task = None

        # Goal hierarchy
        self.goals = {
            'primary': {'id': 'survival', 'priority': 10, 'active': True},
            'secondary': {'id': 'exploration', 'priority': 5, 'active': True},
            'tertiary': {'id': 'interaction', 'priority': 3, 'active': True}
        }

        # Attention system
        self.attention_buffer = []
        self.attention_threshold = 0.7

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_robot_brain/status', 10)

        # Perception inputs
        self.perception_sub = self.create_subscription(
            String,
            '/ai_robot_brain/perception/detections',
            self.perception_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Command inputs
        self.command_sub = self.create_subscription(
            String,
            '/ai_robot_brain/commands',
            self.command_callback,
            10
        )

        # Memory interface
        self.memory_query_pub = self.create_publisher(
            String,
            '/ai_robot_brain/memory/query',
            10
        )

        self.memory_store_pub = self.create_publisher(
            String,
            '/ai_robot_brain/memory/store',
            10
        )

        self.memory_response_sub = self.create_subscription(
            String,
            '/ai_robot_brain/memory/response',
            self.memory_response_callback,
            10
        )

        # Main cognitive cycle timer
        self.cognitive_timer = self.create_timer(0.1, self.cognitive_cycle)

        self.perception_data = {}
        self.laser_data = None
        self.odom_data = None
        self.pending_memory_requests = {}

        self.get_logger().info('Cognitive Architecture Node initialized')

    def perception_callback(self, msg):
        """Process incoming perception data"""
        try:
            data = json.loads(msg.data)
            self.perception_data = data
            self.update_attention(data)
        except Exception as e:
            self.get_logger().error(f'Error processing perception data: {str(e)}')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg
        # Update robot location in state
        self.current_state['location'] = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self.quaternion_to_yaw(msg.pose.pose.orientation)
        ]

    def command_callback(self, msg):
        """Process incoming commands"""
        try:
            command_data = json.loads(msg.data) if msg.data.startswith('{') else {'command': msg.data}

            # Add command to task queue
            self.task_queue.append(command_data)

            # Update motivational state based on command
            if 'emergency' in msg.data or 'stop' in msg.data:
                self.current_state['motivational_state'] = 'safety'
            elif 'go to' in msg.data or 'navigate' in msg.data:
                self.current_state['motivational_state'] = 'task_completion'
            elif 'interact' in msg.data or 'talk' in msg.data:
                self.current_state['motivational_state'] = 'interaction'

        except Exception as e:
            self.get_logger().error(f'Error processing command: {str(e)}')

    def update_attention(self, perception_data):
        """Update attention based on salient stimuli"""
        salient_objects = []

        # Check for objects that exceed attention threshold
        if 'detections' in perception_data:
            for detection in perception_data['detections']:
                if detection.get('confidence', 0) > self.attention_threshold:
                    salient_objects.append(detection)

        if 'closest_obstacle_distance' in perception_data:
            if perception_data['closest_obstacle_distance'] < 0.5:  # Dangerous proximity
                salient_objects.append({'type': 'obstacle', 'distance': perception_data['closest_obstacle_distance']})

        # Update attention focus
        if salient_objects:
            # Select most salient object (for now, just the first)
            self.current_state['attention_focus'] = salient_objects[0]

            # Trigger memory query for similar past experiences
            self.query_memory_for_pattern(salient_objects[0])

    def query_memory_for_pattern(self, stimulus):
        """Query memory for similar past experiences"""
        query = {
            'stimulus_type': stimulus.get('type', 'unknown'),
            'stimulus_attributes': stimulus,
            'context': {
                'location': self.current_state['location'],
                'motivational_state': self.current_state['motivational_state']
            }
        }

        query_msg = String()
        query_msg.data = json.dumps(query)
        self.memory_query_pub.publish(query_msg)

    def memory_response_callback(self, msg):
        """Handle memory responses"""
        try:
            response = json.loads(msg.data)

            if response.get('type') == 'query' and response.get('success'):
                results = response.get('results', [])

                # Process retrieved experiences for decision making
                self.process_memory_results(results)

        except Exception as e:
            self.get_logger().error(f'Error processing memory response: {str(e)}')

    def process_memory_results(self, results):
        """Process retrieved memory results for decision making"""
        if not results:
            return

        # Find the most relevant experience
        best_experience = max(results, key=lambda r: r['similarity'] * r['importance'])

        # Extract recommended action from experience
        recommended_action = best_experience['data'].get('recommended_action')

        if recommended_action:
            # Apply the recommended action with some probability
            # based on similarity and importance
            confidence = best_experience['similarity'] * best_experience['importance']

            if confidence > 0.7:  # High confidence in recommendation
                self.execute_action(recommended_action)

    def cognitive_cycle(self):
        """Main cognitive processing cycle"""
        current_time = time.time()

        # Update state
        self.update_robot_state()

        # Process goals and motivations
        active_goal = self.select_active_goal()

        # Plan actions based on active goal
        action = self.decision_making_cycle(active_goal)

        # Execute action if available
        if action is not None:
            self.execute_action(action)

        # Update decision time
        self.current_state['last_decision_time'] = current_time

        # Publish status
        status_msg = String()
        status_msg.data = json.dumps({
            'state': self.current_state.copy(),
            'active_task': self.active_task,
            'goal': active_goal
        })
        self.status_pub.publish(status_msg)

        # Log current state periodically
        if int(current_time) % 10 == 0:  # Log every 10 seconds
            self.get_logger().info(
                f"State: {self.current_state['motivational_state']}, "
                f"Battery: {self.current_state['battery_level']:.1f}%, "
                f"Location: ({self.current_state['location'][0]:.2f}, {self.current_state['location'][1]:.2f})"
            )

    def update_robot_state(self):
        """Update internal state based on sensor data"""
        # Update battery level (simulated)
        if self.current_state['last_decision_time']:
            time_since_last = time.time() - self.current_state['last_decision_time']
            self.current_state['battery_level'] -= time_since_last * 0.001  # Small drain over time
            self.current_state['battery_level'] = max(0, self.current_state['battery_level'])

        # Update stuck counter based on laser data
        if self.laser_data:
            min_distance = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')
            if min_distance < 0.3:  # Probably stuck
                self.current_state['stuck_counter'] += 1
            else:
                self.current_state['stuck_counter'] = max(0, self.current_state['stuck_counter'] - 1)

    def select_active_goal(self) -> str:
        """Select the most appropriate active goal based on current state"""
        # Check motivational state
        if self.current_state['motivational_state'] == 'safety':
            return 'avoid_collision'
        elif self.current_state['battery_level'] < 20:
            return 'return_to_base'
        elif self.current_state['motivational_state'] == 'task_completion':
            return 'complete_current_task'
        elif self.current_state['motivational_state'] == 'interaction':
            return 'respond_to_human'
        else:  # exploration
            return 'explore_environment'

    def decision_making_cycle(self, goal: str):
        """Main decision making based on current goal"""
        if goal == 'avoid_collision':
            return self.decision_avoid_collision()
        elif goal == 'return_to_base':
            return self.decision_return_to_base()
        elif goal == 'complete_current_task':
            return self.decision_complete_task()
        elif goal == 'respond_to_human':
            return self.decision_respond_to_human()
        elif goal == 'explore_environment':
            return self.decision_explore()
        else:
            return self.decision_idle()

    def decision_avoid_collision(self):
        """Decision for collision avoidance"""
        if self.laser_data:
            ranges = self.laser_data.ranges
            if ranges:
                min_dist = min(ranges)
                if min_dist < 0.4:  # Need to avoid
                    # Simple avoidance: turn away from closest obstacle
                    front_ranges = ranges[len(ranges)//2-10:len(ranges)//2+10]
                    closest_idx = min(range(len(front_ranges)), key=lambda i: front_ranges[i])
                    turn_direction = 'left' if closest_idx < len(front_ranges)//2 else 'right'

                    if turn_direction == 'left':
                        return {'linear_x': 0.0, 'angular_z': 0.5}
                    else:
                        return {'linear_x': 0.0, 'angular_z': -0.5}

        return {'linear_x': 0.1, 'angular_z': 0.0}  # Continue forward slowly

    def decision_return_to_base(self):
        """Decision for returning to charging base"""
        # Simplified: just move forward to return to base
        # In a real system, this would involve navigation to a known location
        return {'linear_x': 0.2, 'angular_z': 0.0}

    def decision_complete_task(self):
        """Decision for completing current task"""
        # If we have an active task, work on it
        if self.task_queue:
            current_task = self.task_queue[0]  # Take first task

            # Example task processing
            if 'navigate_to' in current_task.get('command', ''):
                # Navigate to specific location (simplified)
                return {'linear_x': 0.2, 'angular_z': 0.0}
            elif 'pick_up' in current_task.get('command', ''):
                # Attempt to pick up object (simplified)
                return {'linear_x': 0.0, 'angular_z': 0.0}  # Stay in place

        # If no specific task, explore
        return self.decision_explore()

    def decision_respond_to_human(self):
        """Decision for responding to human"""
        # Check if we have attention-focused stimulus from human
        if self.current_state['attention_focus']:
            stim = self.current_state['attention_focus']
            if stim.get('type') == 'person':
                # Move toward the person
                return {'linear_x': 0.1, 'angular_z': 0.0}

        return {'linear_x': 0.0, 'angular_z': 0.0}  # Stay put

    def decision_explore(self):
        """Decision for environment exploration"""
        # Simple exploration behavior: move forward until obstacle, then turn
        if self.laser_data:
            ranges = self.laser_data.ranges
            if ranges:
                front_dist = ranges[len(ranges)//2]  # Distance directly ahead
                if front_dist < 1.0:  # Obstacle ahead
                    # Turn randomly to explore
                    import random
                    turn_direction = random.choice(['left', 'right'])
                    if turn_direction == 'left':
                        return {'linear_x': 0.0, 'angular_z': 0.5}
                    else:
                        return {'linear_x': 0.0, 'angular_z': -0.5}

        # Default: move forward
        return {'linear_x': 0.2, 'angular_z': 0.0}

    def decision_idle(self):
        """Default idle decision"""
        return {'linear_x': 0.0, 'angular_z': 0.0}

    def execute_action(self, action: Dict[str, Any]):
        """Execute a robotic action"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(action.get('linear_x', 0.0))
        cmd_msg.angular.z = float(action.get('angular_z', 0.0))

        # Apply action limits
        cmd_msg.linear.x = max(-0.5, min(0.5, cmd_msg.linear.x))
        cmd_msg.angular.z = max(-1.0, min(1.0, cmd_msg.angular.z))

        self.cmd_pub.publish(cmd_msg)

    def quaternion_to_yaw(self, orientation):
        """Convert quaternion to yaw angle"""
        import math
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = CognitiveArchitectureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cognitive Architecture Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 7: Create Integration Launch File

Create the complete integration launch file `ai_robot_brain/launch/complete_ai_brain.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Package names
    pkg_ai_perception = 'ai_perception'
    pkg_rl_controller = 'rl_controller'
    pkg_nlp_system = 'nlp_system'
    pkg_memory_system = 'memory_system'
    pkg_cognitive_arch = 'cognitive_arch'
    pkg_isaac_integration = 'isaac_integration'

    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    log_level = LaunchConfiguration('log_level', default='info')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level for the nodes'
    )

    # AI Perception Node
    ai_perception_node = Node(
        package=pkg_ai_perception,
        executable='object_detection_node',
        name='object_detection_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_ai_perception),
                'config',
                'perception_params.yaml'
            ])
        ],
        remappings=[
            ('/camera/color/image_raw', '/camera/color/image_raw'),
            ('/camera/color/camera_info', '/camera/color/camera_info'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # RL Controller Node
    rl_agent_node = Node(
        package=pkg_rl_controller,
        executable='ppo_rl_agent',
        name='ppo_rl_agent',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_rl_controller),
                'config',
                'rl_params.yaml'
            ])
        ],
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/odom'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # NLP System Node
    nlp_node = Node(
        package=pkg_nlp_system,
        executable='command_parser_node',
        name='command_parser_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_nlp_system),
                'config',
                'nlp_params.yaml'
            ])
        ],
        remappings=[
            ('/ai_robot_brain/speech_recognition/text', '/speech_to_text/output'),
            ('/ai_robot_brain/tts/text', '/text_to_speech/input'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Memory System Node
    memory_node = Node(
        package=pkg_memory_system,
        executable='episodic_memory_node',
        name='episodic_memory_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_memory_system),
                'config',
                'memory_params.yaml'
            ])
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Cognitive Architecture Node
    cognitive_node = Node(
        package=pkg_cognitive_arch,
        executable='cognitive_architecture_node',
        name='cognitive_architecture_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_cognitive_arch),
                'config',
                'arch_params.yaml'
            ])
        ],
        remappings=[
            ('/scan', '/scan'),
            ('/odom', '/odom'),
            ('/cmd_vel', '/cmd_vel'),
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Isaac Integration Node
    isaac_bridge_node = Node(
        package=pkg_isaac_integration,
        executable='isaac_bridge_node',
        name='isaac_bridge_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(pkg_isaac_integration),
                'config',
                'isaac_params.yaml'
            ])
        ],
        arguments=['--ros-args', '--log-level', log_level],
        output='screen'
    )

    # Create launch description with delayed startup
    ld = LaunchDescription([
        declare_use_sim_time,
        declare_log_level,
    ])

    # Add nodes with startup delays to manage dependencies
    startup_order = [
        (ai_perception_node, 0.0),
        (memory_node, 1.0),
        (rl_agent_node, 2.0),
        (nlp_node, 3.0),
        (cognitive_node, 4.0),
        (isaac_bridge_node, 5.0),
    ]

    current_time = 0.0
    for node, delay in startup_order:
        ld.add_action(TimerAction(
            period=current_time + delay,
            actions=[node]
        ))

    return ld
```

### Testing and Validation

Create a comprehensive testing script to validate the AI Robot Brain system:

```bash
#!/bin/bash

echo "Starting AI Robot Brain validation..."

# Build all packages
cd ~/robotics_ws
colcon build --packages-select ai_perception rl_controller nlp_system memory_system cognitive_arch

# Source setup
source install/setup.bash

# Launch the complete system
echo "Launching complete AI Robot Brain system..."
ros2 launch ai_robot_brain complete_ai_brain.launch.py &

# Wait for system to initialize
sleep 10

# Run validation tests
echo "Running perception validation..."
ros2 topic echo /ai_robot_brain/detections --field detections --timeout 5

echo "Running memory validation..."
echo '{"text": "test query", "limit": 1}' | ros2 topic pub /ai_robot_brain/memory/query std_msgs/String --once

echo "Running NLP validation..."
echo '{"text": "hello robot"}' | ros2 topic pub /ai_robot_brain/commands std_msgs/String --once

# Wait a bit more
sleep 5

echo "Validation complete. Check system status:"
ros2 component list

echo "Stopping system..."
kill %%

echo "AI Robot Brain assignment implementation complete!"
```

### Documentation Requirements

Create the following documentation files:

1. `ai_robot_brain/docs/architecture.md` - System architecture diagram and explanation
2. `ai_robot_brain/docs/usage_guide.md` - How to run and interact with the AI Robot Brain
3. `ai_robot_brain/docs/api_reference.md` - API documentation for each component
4. `ai_robot_brain/docs/performance_metrics.md` - Evaluation results and metrics

### Performance Evaluation

Each component should be evaluated based on:

1. **Computational Efficiency**: Processing time, memory usage, CPU/GPU utilization
2. **Accuracy**: Correctness of perception, navigation, and decision-making
3. **Robustness**: Performance under various environmental conditions
4. **Integration**: How well components work together
5. **Scalability**: How well the system scales with additional capabilities

### Submission Requirements

Your assignment submission should include:

1. **Complete Source Code**: All ROS 2 packages with proper structure
2. **Configuration Files**: All YAML configuration files
3. **Documentation**: Architecture, API documentation, and usage guides
4. **Test Results**: Output from unit and integration tests
5. **Performance Analysis**: Results from performance benchmarking
6. **Video Demonstration**: Short video showing the system in action
7. **Report**: Detailed report explaining your implementation and lessons learned

### Grading Criteria

- **Functionality (40%)**: All components work as specified
- **Code Quality (25%)**: Clean, well-documented, and maintainable code
- **System Integration (20%)**: Components work together smoothly
- **Testing and Validation (10%)**: Comprehensive testing and validation
- **Documentation (5%)**: Clear and complete documentation

This assignment provides a comprehensive implementation of an AI Robot Brain for humanoid robotics, integrating perception, decision-making, learning, and action execution in a cohesive system. The implementation follows ROS 2 best practices and leverages NVIDIA Isaac technologies for simulation and deployment.