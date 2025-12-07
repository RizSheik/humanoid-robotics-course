---
id: module-4-practical-lab
title: Module 4 — Vision-Language-Action Systems | Chapter 4 — Practical Lab
sidebar_label: Chapter 4 — Practical Lab
sidebar_position: 4
---

# Module 4 — Vision-Language-Action Systems

## Chapter 4 — Practical Lab

### Lab Setup and Prerequisites

This practical lab focuses on implementing Vision-Language-Action (VLA) systems for humanoid robots. Before beginning, ensure you have:

#### Hardware Requirements
- Computer with NVIDIA GPU (RTX 3080 or equivalent recommended)
- Access to a humanoid robot simulation (Isaac Sim) or real robot
- RGB-D camera for perception
- Network access for downloading models

#### Software Requirements
- Ubuntu 22.04 LTS with ROS 2 Humble Hawksbill
- NVIDIA GPU drivers (520+)
- CUDA 11.8+ and cuDNN
- Isaac Sim 2023.1+
- Isaac ROS packages
- Python 3.10+ with PyTorch and related libraries
- Git for version control

#### Installation Commands
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install python3-dev python3-pip build-essential

# Install ROS 2 Humble dependencies
sudo apt install ros-humble-ros-base ros-humble-cv-bridge ros-humble-tf2-ros
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-moveit ros-humble-vision-opencv

# Install Isaac ROS dependencies
sudo apt install ros-humble-isaac-ros-* ros-humble-nvblox-*

# Install Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate
pip3 install opencv-python pillow numpy scipy
pip3 install pyquaternion transforms3d
```

### Lab Exercise 1: Implementing a Basic VLA System

#### Objective
Create a simple Vision-Language-Action system that can understand basic commands and execute corresponding actions.

#### Step-by-Step Instructions

1. **Create the project structure**:
   ```bash
   mkdir -p ~/robotics_ws/src/vla_robot_system
   cd ~/robotics_ws/src/vla_robot_system
   ros2 pkg create --build-type ament_python vla_robot_controller
   ```

2. **Create the VLA controller node** - `vla_robot_controller/vla_robot_controller/vla_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from sensor_msgs.msg import Image, JointState
   from geometry_msgs.msg import Twist
   from cv_bridge import CvBridge
   import torch
   import torch.nn as nn
   from transformers import GPT2Tokenizer, GPT2LMHeadModel
   import numpy as np
   import cv2
   from PIL import Image as PILImage

   class VLASystemNode(Node):
       def __init__(self):
           super().__init__('vla_system_node')

           # Initialize CV bridge
           self.bridge = CvBridge()

           # Initialize tokenizer
           self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
           self.tokenizer.pad_token = self.tokenizer.eos_token

           # Load language model (in practice, you'd use a specialized model)
           self.language_model = GPT2LMHeadModel.from_pretrained('gpt2')
           self.language_model.eval()

           # Initialize vision processing model
           self.setup_vision_model()

           # Initialize action generation
           self.setup_action_generator()

           # Publishers and subscribers
           self.image_sub = self.create_subscription(
               Image,
               '/camera/rgb/image_rect_color',
               self.image_callback,
               10
           )

           self.command_sub = self.create_subscription(
               String,
               '/vla_commands',
               self.command_callback,
               10
           )

           self.joint_state_sub = self.create_subscription(
               JointState,
               '/joint_states',
               self.joint_state_callback,
               10
           )

           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

           # Internal state
           self.current_image = None
           self.current_joint_states = None
           self.command_queue = []
           self.perception_cache = {}

           self.get_logger().info('VLA System Node initialized')

       def setup_vision_model(self):
           """Setup neural network for vision processing"""
           # For this lab, using a simple CNN for object detection
           # In practice, use pre-trained models like YOLO or DETR
           self.vision_model = nn.Sequential(
               nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
               nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d((4, 4)),
               nn.Flatten(),
               nn.Linear(128 * 4 * 4, 512),
               nn.ReLU(),
               nn.Linear(512, 256)
           )
           self.vision_model.eval()

       def setup_action_generator(self):
           """Setup network to generate robot actions from VLA inputs"""
           self.action_generator = nn.Sequential(
               nn.Linear(512 + 768 + 100, 1024),  # Vision (256) + Language (768) + State (100)
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(1024, 512),
               nn.ReLU(),
               nn.Linear(512, 64),  # Discrete actions
               nn.Softmax(dim=-1)
           )

           # Continuous action head
           self.continuous_action_head = nn.Sequential(
               nn.Linear(512, 128),
               nn.ReLU(),
               nn.Linear(128, 20)  # Joint velocities
           )

       def image_callback(self, msg):
           """Process incoming camera images"""
           try:
               # Convert ROS Image to OpenCV
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

               # Convert to tensor
               pil_image = PILImage.fromarray(cv_image)
               transform = T.Compose([
                   T.Resize((224, 224)),
                   T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
               ])
               tensor_image = transform(pil_image).unsqueeze(0)

               # Get vision features
               with torch.no_grad():
                   vision_features = self.vision_model(tensor_image).squeeze(0)

               self.current_image = tensor_image
               self.current_vision_features = vision_features

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def command_callback(self, msg):
           """Process incoming language commands"""
           command_text = msg.data
           self.get_logger().info(f'Received command: {command_text}')

           # Add to processing queue
           self.command_queue.append(command_text)

       def joint_state_callback(self, msg):
           """Process joint states"""
           self.current_joint_states = msg

           # Create a simplified robot state vector
           if msg.position:
               # Limit to first 100 joints (pad or truncate as needed)
               state_vec = np.zeros(100)
               for i, pos in enumerate(msg.position[:100]):
                   state_vec[i] = pos
               self.robot_state_vector = torch.tensor(state_vec).float()
           else:
               self.robot_state_vector = torch.zeros(100)

       def process_command(self, command_text):
           """Process a language command through the VLA system"""
           if self.current_image is None or self.current_vision_features is None:
               self.get_logger().warn('No vision data available')
               return

           try:
               # Process language command
               inputs = self.tokenizer(
                   command_text,
                   return_tensors='pt',
                   padding=True,
                   truncation=True,
                   max_length=50
               )

               with torch.no_grad():
                   language_outputs = self.language_model(
                       input_ids=inputs['input_ids'],
                       attention_mask=inputs['attention_mask']
                   )
                   # Get last token's hidden state as language representation
                   language_features = language_outputs.last_hidden_state[:, -1, :].squeeze(0)

               # Combine vision, language, and state
               combined_features = torch.cat([
                   self.current_vision_features,
                   language_features,
                   self.robot_state_vector
               ], dim=0).unsqueeze(0)  # Add batch dimension

               # Generate actions
               with torch.no_grad():
                   discrete_actions = self.action_generator(combined_features)
                   continuous_actions = self.continuous_action_head(combined_features)

               # Convert to robot commands
               self.execute_commands(discrete_actions, continuous_actions)

           except Exception as e:
               self.get_logger().error(f'Error processing command: {e}')

       def execute_commands(self, discrete_actions, continuous_actions):
           """Execute the generated robot commands"""
           # Select most probable discrete action
           action_idx = torch.argmax(discrete_actions, dim=1).item()

           # Convert to robot commands based on action index
           cmd_msg = Twist()

           if action_idx == 0:  # Move forward
               cmd_msg.linear.x = 0.2
               cmd_msg.angular.z = 0.0
           elif action_idx == 1:  # Turn left
               cmd_msg.linear.x = 0.0
               cmd_msg.angular.z = 0.5
           elif action_idx == 2:  # Turn right
               cmd_msg.linear.x = 0.0
               cmd_msg.angular.z = -0.5
           elif action_idx == 3:  # Move backward
               cmd_msg.linear.x = -0.1
               cmd_msg.angular.z = 0.0
           else:  # Stop
               cmd_msg.linear.x = 0.0
               cmd_msg.angular.z = 0.0

           # Publish command
           self.cmd_vel_pub.publish(cmd_msg)
           self.get_logger().info(f'Executed action {action_idx}: linear={cmd_msg.linear.x}, angular={cmd_msg.angular.z}')

       def process_command_queue(self):
           """Process all pending commands"""
           while self.command_queue:
               command = self.command_queue.pop(0)
               self.process_command(command)

   def main(args=None):
       rclpy.init(args=args)
       node = VLASystemNode()

       # Timer to process command queue
       timer = node.create_timer(0.1, node.process_command_queue)

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('VLA System shutting down')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create setup.py** for the package:
   ```python
   from setuptools import setup, find_packages
   import os
   from glob import glob

   package_name = 'vla_robot_controller'

   setup(
       name=package_name,
       version='0.0.0',
       packages=find_packages(),
       data_files=[
           ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=[
           'setuptools',
           'torch',
           'transformers',
           'opencv-python',
           'numpy',
           'torchvision'
       ],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='VLA System for Humanoid Robotics',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'vla_node = vla_robot_controller.vla_node:main',
           ],
       },
   )
   ```

4. **Update package.xml**:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>vla_robot_controller</name>
     <version>0.0.0</version>
     <description>Vision-Language-Action System for Humanoid Robotics</description>
     <maintainer email="your.email@example.com">Your Name</maintainer>
     <license>Apache License 2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>sensor_msgs</depend>
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

5. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select vla_robot_controller
   source install/setup.bash
   ```

6. **Run the VLA system**:
   ```bash
   # In one terminal, start the VLA node
   ros2 run vla_robot_controller vla_node

   # In another terminal, send commands
   echo '{"text": "move forward"}' | ros2 topic pub /vla_commands std_msgs/String --once
   ```

#### Expected Results
The VLA system should:
- Receive camera images and process them
- Accept natural language commands
- Generate appropriate robot actions (movement commands in this case)
- Publish command messages to control the robot

### Lab Exercise 2: Advanced VLA System with Foundation Models

#### Objective
Implement a more advanced VLA system using pre-trained foundation models for better performance.

#### Step-by-Step Instructions

1. **Download Foundation Models**:
   ```bash
   # Download pre-trained models (this may take time)
   python3 -c "
   from transformers import AutoTokenizer, AutoModel
   import torch

   # Download tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
   model = AutoModel.from_pretrained('bert-base-uncased')

   print('Models downloaded successfully')
   "
   ```

2. **Create Advanced VLA Node** - `vla_robot_controller/vla_robot_controller/advanced_vla_node.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from sensor_msgs.msg import Image, JointState, PointCloud2
   from geometry_msgs.msg import Twist, PointStamped
   from cv_bridge import CvBridge
   from vision_msgs.msg import Detection2DArray, Detection2D
   import torch
   import torch.nn as nn
   import numpy as np
   import cv2
   from PIL import Image as PILImage
   from transformers import AutoTokenizer, AutoModel
   import open3d as o3d
   from scipy.spatial.transform import Rotation as R
   import time

   class AdvancedVLANode(Node):
       def __init__(self):
           super().__init__('advanced_vla_node')

           # Initialize components
           self.bridge = CvBridge()

           # Initialize foundation models
           self.setup_foundation_models()

           # ROS 2 interfaces
           self.setup_ros_interfaces()

           # Internal state
           self.setup_internal_state()

           # Processing timers
           self.processing_timer = self.create_timer(0.1, self.vla_processing_cycle)

           self.get_logger().info('Advanced VLA Node initialized successfully')

       def setup_foundation_models(self):
           """Initialize foundation models for VLA system"""
           # Vision model - using CLIP for multi-modal understanding
           from transformers import CLIPProcessor, CLIPModel
           self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
           self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
           self.clip_model.eval()

           # Language model - using a more advanced model for understanding
           self.lang_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
           self.lang_model = AutoModel.from_pretrained("bert-base-uncased")
           self.lang_model.eval()

           # Action prediction head - custom head for robot actions
           self.action_head = nn.Sequential(
               nn.Linear(512 + 768, 256),  # CLIP + BERT features
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Linear(128, 64)  # Action space
           )

           # Continuous action head for precise control
           self.continuous_head = nn.Sequential(
               nn.Linear(128, 256),
               nn.ReLU(),
               nn.Linear(256, 50)  # Continuous control space
           )

           # Object detection model for scene understanding
           self.obj_det_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
           self.obj_det_model.eval()

       def setup_ros_interfaces(self):
           """Set up all ROS 2 publishers and subscribers"""
           # Image subscription
           self.image_sub = self.create_subscription(
               Image,
               '/camera/rgb/image_raw',
               self.image_callback,
               10
           )

           # Point cloud for 3D understanding
           self.pc_sub = self.create_subscription(
               PointCloud2,
               '/camera/depth/points',
               self.point_cloud_callback,
               10
           )

           # Joint state subscription
           self.joint_sub = self.create_subscription(
               JointState,
               '/joint_states',
               self.joint_state_callback,
               10
           )

           # Command subscription
           self.command_sub = self.create_subscription(
               String,
               '/natural_language_command',
               self.command_callback,
               10
           )

           # Publishers
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.detection_pub = self.create_publisher(Detection2DArray, '/perception/detections', 10)
           self.debug_img_pub = self.create_publisher(Image, '/debug/vla_output', 10)
           self.status_pub = self.create_publisher(String, '/vla/status', 10)

       def setup_internal_state(self):
           """Initialize internal state variables"""
           self.current_image = None
           self.current_pc = None
           self.current_joints = None
           self.command_queue = []
           self.object_detections = []
           self.scene_description = ""
           self.last_process_time = time.time()

       def image_callback(self, msg):
           """Process image data"""
           try:
               # Convert ROS image to OpenCV
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
               self.current_image = cv_image

               # Perform object detection
               results = self.obj_det_model(cv_image)
               self.object_detections = self.process_detections(results)

               # Update scene description
               self.update_scene_description(cv_image)

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def point_cloud_callback(self, msg):
           """Process point cloud data"""
           # Convert point cloud to Open3D format for 3D scene understanding
           # This is a simplified version - full implementation would be more complex
           pass

       def joint_state_callback(self, msg):
           """Process joint states"""
           self.current_joints = msg

       def command_callback(self, msg):
           """Process natural language commands"""
           self.command_queue.append(msg.data)
           self.get_logger().info(f'Added command to queue: {msg.data}')

       def update_scene_description(self, image_cv):
           """Generate a description of the current scene"""
           # Detect objects in the scene
           results = self.obj_det_model(image_cv)
           detections = results.pandas().xyxy[0]  # Get detections as DataFrame

           # Compile a description of the scene
           objects = []
           for _, det in detections.iterrows():
               if det['confidence'] > 0.5:  # Confidence threshold
                   objects.append({
                       'name': det['name'],
                       'confidence': det['confidence'],
                       'bbox': [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
                   })

           # Create scene description text
           obj_names = [obj['name'] for obj in objects]
           unique_objs = list(set(obj_names))

           if len(unique_objs) > 0:
               self.scene_description = f"The scene contains: {', '.join(unique_objs)}"
           else:
               self.scene_description = "The scene appears to be empty (no detectable objects)"

       def process_detections(self, results):
           """Process YOLOv5 detection results for ROS messages"""
           detections = []

           for *xyxy, conf, cls in results.xyxy[0].tolist():  # Results in xyxy format
               if conf > 0.5:  # Only include confident detections
                   detection = Detection2D()

                   # Set bounding box center and size
                   center_x = (xyxy[0] + xyxy[2]) / 2
                   center_y = (xyxy[1] + xyxy[3]) / 2
                   width = xyxy[2] - xyxy[0]
                   height = xyxy[3] - xyxy[1]

                   detection.bbox.center.x = float(center_x)
                   detection.bbox.center.y = float(center_y)
                   detection.bbox.size_x = float(width)
                   detection.bbox.size_y = float(height)

                   # Add hypothesis
                   hypothesis = ObjectHypothesisWithPose()
                   hypothesis.hypothesis.class_id = str(int(cls))
                   hypothesis.hypothesis.score = float(conf)
                   detection.results.append(hypothesis)

                   detections.append(detection)

           return detections

       def process_language_command(self, command_text):
           """Process natural language command using foundation models"""
           # Encode text using CLIP
           inputs = self.clip_processor(
               text=[command_text],
               images=[PILImage.fromarray((self.current_image*255).astype(np.uint8))] if self.current_image is not None else None,
               return_tensors="pt",
               padding=True,
               truncation=True
           )

           with torch.no_grad():
               outputs = self.clip_model(**inputs)
               text_features = outputs.text_embeds  # Shape: [1, 512]
               image_features = outputs.image_embeds if self.current_image is not None else torch.zeros(1, 512)

           # Combine vision and language features
           combined_features = torch.cat([image_features, text_features], dim=1)

           # Generate action prediction
           with torch.no_grad():
               action_logits = self.action_head(combined_features)
               continuous_actions = self.continuous_head(combined_features)

           return action_logits, continuous_actions

       def execute_vla_cycle(self):
           """Main VLA processing cycle"""
           if not self.command_queue:
               return

           start_time = time.time()

           if self.current_image is None:
               self.get_logger().warn('No image data available for VLA processing')
               return

           # Process the next command in queue
           command_text = self.command_queue.pop(0)

           # Process the command
           try:
               action_logits, continuous_actions = self.process_language_command(command_text)

               # Select action
               action_idx = torch.argmax(action_logits, dim=1).item()
               action_probs = torch.softmax(action_logits, dim=1)
               confidence = action_probs[0][action_idx].item()

               # Execute action if confidence is high enough
               if confidence > 0.7:
                   self.execute_action(action_idx, continuous_actions)

                   # Publish status
                   status_msg = String()
                   status_msg.data = f"Executed action {action_idx} with confidence {confidence:.2f}: {command_text}"
                   self.status_pub.publish(status_msg)
               else:
                   status_msg = String()
                   status_msg.data = f"Low confidence ({confidence:.2f}) for command: {command_text}"
                   self.status_pub.publish(status_msg)
                   self.get_logger().warn(f'Low confidence action: {confidence:.2f}')

           except Exception as e:
               self.get_logger().error(f'Error in VLA cycle: {e}')

           # Calculate processing time
           elapsed = time.time() - start_time
           self.get_logger().info(f'VLA cycle completed in {elapsed:.3f}s')

       def execute_action(self, action_idx, continuous_actions):
           """Execute robot action based on prediction"""
           # Convert action index to robot command
           cmd_msg = Twist()

           # Define action space (these would be learned mappings in real system)
           action_space = {
               0: {'name': 'move_forward', 'linear_x': 0.3, 'angular_z': 0.0},
               1: {'name': 'turn_left', 'linear_x': 0.0, 'angular_z': 0.5},
               2: {'name': 'turn_right', 'linear_x': 0.0, 'angular_z': -0.5},
               3: {'name': 'move_backward', 'linear_x': -0.2, 'angular_z': 0.0},
               4: {'name': 'stop', 'linear_x': 0.0, 'angular_z': 0.0},
               5: {'name': 'approach_object', 'linear_x': 0.1, 'angular_z': 0.0}
           }

           if action_idx < len(action_space):
               action_def = action_space[action_idx]
               cmd_msg.linear.x = action_def['linear_x']
               cmd_msg.angular.z = action_def['angular_z']
           else:
               # Default stop action if unknown
               cmd_msg.linear.x = 0.0
               cmd_msg.angular.z = 0.0

           # Publish command
           self.cmd_vel_pub.publish(cmd_msg)
           self.get_logger().info(f'Executed action: {action_idx}, linear={cmd_msg.linear.x}, angular={cmd_msg.angular.z}')

   def main(args=None):
       rclpy.init(args=args)
       node = AdvancedVLASystemNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Advanced VLA System shutting down')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create a launch file** - `vla_robot_controller/launch/vla_system.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, RegisterEventHandler
   from launch.event_handlers import OnProcessStart
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Package name
       pkg_name = 'vla_robot_controller'

       # Launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')
       robot_namespace = LaunchConfiguration('robot_namespace', default='')

       # VLA node
       vla_node = Node(
           package='vla_robot_controller',
           executable='advanced_vla_node',
           name='advanced_vla_node',
           parameters=[
               {'use_sim_time': use_sim_time}
           ],
           remappings=[
               ('/camera/rgb/image_raw', '/head_camera/rgb/image_raw'),
               ('/camera/depth/points', '/head_camera/depth/points'),
           ],
           output='screen'
       )

       # Robot state publisher for TF
       robot_state_publisher = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           parameters=[
               {'use_sim_time': use_sim_time}
           ]
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation clock if true'
           ),
           DeclareLaunchArgument(
               'robot_namespace',
               default_value='',
               description='Robot namespace'
           ),
           robot_state_publisher,
           vla_node
       ])
   ```

4. **Test the Advanced VLA System**:
   ```bash
   # Build the package
   cd ~/robotics_ws
   colcon build --packages-select vla_robot_controller
   source install/setup.bash

   # Launch the system
   ros2 launch vla_robot_controller vla_system.launch.py
   ```

5. **Send commands to test the system**:
   ```bash
   # In another terminal
   ros2 topic pub /natural_language_command std_msgs/String "data: 'move forward'"
   ros2 topic pub /natural_language_command std_msgs/String "data: 'turn left'"
   ros2 topic pub /natural_language_command std_msgs/String "data: 'stop'"
   ```

#### Expected Results
The advanced VLA system should:
- Use foundation models for better understanding
- Integrate vision-language processing
- Generate more accurate robot commands
- Provide status feedback
- Handle complex natural language commands

### Lab Exercise 3: Integration with Isaac Sim

#### Objective
Integrate the VLA system with NVIDIA Isaac Sim for testing in a simulated environment.

#### Step-by-Step Instructions

1. **Set up Isaac Sim environment**:
   ```bash
   # Make sure Isaac Sim is installed
   # Create a simple humanoid robot world
   mkdir -p ~/robotics_ws/src/vla_robot_controller/worlds

   # Create a simple world file for testing
   cat > ~/robotics_ws/src/vla_robot_controller/worlds/simple_vla_test.sdf << 'EOF'
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_vla_test">
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1.0</real_time_factor>
         <real_time_update_rate>1000.0</real_time_update_rate>
       </physics>

       <include>
         <uri>model://ground_plane</uri>
       </include>

       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Simple objects for perception -->
       <model name="table">
         <pose>2 0 0 0 0 0</pose>
         <link name="link">
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

       <model name="red_box">
         <pose>2.5 0.5 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.2 0.2 0.2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.2 0.2 0.2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.1 0.1 1</ambient>
               <diffuse>0.8 0.1 0.1 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <model name="blue_ball">
         <pose>-1 1 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <sphere>
                 <radius>0.15</radius>
               </sphere>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <sphere>
                 <radius>0.15</radius>
               </sphere>
             </geometry>
             <material>
               <ambient>0.1 0.1 0.8 1</ambient>
               <diffuse>0.1 0.1 0.8 1</diffuse>
             </material>
           </visual>
         </link>
       </model>
     </world>
   </sdf>
   EOF
   ```

2. **Create Isaac bridge integration** - `vla_robot_controller/vla_robot_controller/isaac_bridge.py`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from sensor_msgs.msg import Image, JointState
   from geometry_msgs.msg import Twist, PoseStamped
   from geometry_msgs.msg import TransformStamped
   from tf2_ros import TransformBroadcaster
   import numpy as np
   import time
   from cv_bridge import CvBridge

   class IsaacBridgeNode(Node):
       def __init__(self):
           super().__init__('isaac_bridge_node')

           # Initialize CV bridge
           self.cv_bridge = CvBridge()

           # Publishers for Isaac Sim
           self.cmd_pub = self.create_publisher(Twist, '/isaac/cmd_vel', 10)
           self.joint_cmd_pub = self.create_publisher(JointState, '/isaac/joint_commands', 10)

           # Subscribers from Isaac Sim
           self.camera_sub = self.create_subscription(
               Image,
               '/isaac/head_camera/rgb/image_raw',
               self.camera_callback,
               10
           )

           self.joint_state_sub = self.create_subscription(
               JointState,
               '/isaac/joint_states',
               self.joint_state_callback,
               10
           )

           self.odom_sub = self.create_subscription(
               Odometry,
               '/isaac/odom',
               self.odom_callback,
               10
           )

           # VLA system interface
           self.vla_command_sub = self.create_subscription(
               String,
               '/vla/commands',
               self.vla_command_callback,
               10
           )

           self.vla_feedback_pub = self.create_publisher(
               String,
               '/vla/feedback',
               10
           )

           # TF broadcaster
           self.tf_broadcaster = TransformBroadcaster(self)

           # Robot state
           self.robot_pose = [0.0, 0.0, 0.0]  # x, y, theta
           self.robot_velocity = [0.0, 0.0]    # linear, angular

           # Bridge state
           self.last_command_time = time.time()
           self.command_timeout = 1.0  # seconds

           self.get_logger().info('Isaac Bridge Node initialized')

       def camera_callback(self, msg):
           """Forward camera image to VLA system"""
           # This would normally go to perception processing
           # For now, just republish to VLA system
           # In practice, you would process and send to VLA components
           pass

       def joint_state_callback(self, msg):
           """Process joint states from Isaac Sim"""
           # Update internal robot state
           self.current_joint_states = msg

           # Forward to VLA system if needed
           # This would trigger state updates in the VLA system

       def odom_callback(self, msg):
           """Process odometry from Isaac Sim"""
           # Update robot pose
           self.robot_pose[0] = msg.pose.pose.position.x
           self.robot_pose[1] = msg.pose.pose.position.y

           # Extract orientation (assuming 2D planar movement)
           from quaternion import euler_from_quaternion
           orientation = msg.pose.pose.orientation
           _, _, self.robot_pose[2] = euler_from_quaternion([
               orientation.x, orientation.y, orientation.z, orientation.w
           ])

           # Update velocity
           self.robot_velocity[0] = msg.twist.twist.linear.x  # Linear velocity
           self.robot_velocity[1] = msg.twist.twist.angular.z  # Angular velocity

       def vla_command_callback(self, msg):
           """Process commands from VLA system for Isaac Sim"""
           try:
               # Parse command from VLA system
               command_data = msg.data

               # For this demo, assume command is in format: "TYPE ARGS"
               parts = command_data.split(' ', 1)
               command_type = parts[0]
               command_args = parts[1] if len(parts) > 1 else ""

               if command_type == "MOVE":
                   # Parse movement command
                   args = command_args.split()
                   if len(args) >= 2:
                       linear_vel = float(args[0])
                       angular_vel = float(args[1]) if len(args) > 1 else 0.0

                       # Send to Isaac Sim
                       cmd_msg = Twist()
                       cmd_msg.linear.x = linear_vel
                       cmd_msg.angular.z = angular_vel
                       self.cmd_pub.publish(cmd_msg)

                       self.get_logger().info(f'Sent movement command: linear={linear_vel}, angular={angular_vel}')

               elif command_type.startswith("JOINT"):
                   # Process joint command
                   self.execute_joint_command(command_type, command_args)

               # Send feedback to VLA system
               feedback_msg = String()
               feedback_msg.data = f"Command executed: {command_data}"
               self.vla_feedback_pub.publish(feedback_msg)

               self.last_command_time = time.time()

           except Exception as e:
               self.get_logger().error(f'Error processing VLA command: {e}')
               # Send error feedback
               feedback_msg = String()
               feedback_msg.data = f"Error executing command: {e}"
               self.vla_feedback_pub.publish(feedback_msg)

       def execute_joint_command(self, command_type, command_args):
           """Execute joint-related commands"""
           if command_type == "JOINT_POSITION":
               # Command format: "JOINT_POSITION joint_name_1:position_1,joint_name_2:position_2,..."
               try:
                   joint_pairs = command_args.split(',')
                   joint_names = []
                   joint_positions = []

                   for pair in joint_pairs:
                       name, pos = pair.split(':')
                       joint_names.append(name.strip())
                       joint_positions.append(float(pos.strip()))

                   # Create joint command
                   joint_cmd = JointState()
                   joint_cmd.name = joint_names
                   joint_cmd.position = joint_positions
                   joint_cmd.header.stamp = self.get_clock().now().to_msg()

                   # Send to Isaac Sim
                   self.joint_cmd_pub.publish(joint_cmd)
                   self.get_logger().info(f'Sent joint positions: {dict(zip(joint_names, joint_positions))}')

               except Exception as e:
                   self.get_logger().error(f'Error parsing joint command: {e}')

       def publish_robot_state(self):
           """Publish robot state for monitoring"""
           # This could publish TF transforms, robot status, etc.
           # Publish transform from map to robot
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'map'
           t.child_frame_id = 'base_link'

           t.transform.translation.x = self.robot_pose[0]
           t.transform.translation.y = self.robot_pose[1]
           t.transform.translation.z = 0.0

           from quaternion import quaternion_from_euler
           quat = quaternion_from_euler(0, 0, self.robot_pose[2])
           t.transform.rotation.x = quat.x
           t.transform.rotation.y = quat.y
           t.transform.rotation.z = quat.z
           t.transform.rotation.w = quat.w

           self.tf_broadcaster.sendTransform(t)

   def main(args=None):
       rclpy.init(args=args)
       node = IsaacBridgeNode()

       # Timer for publishing robot state
       state_timer = node.create_timer(0.1, node.publish_robot_state)

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Isaac Bridge shutting down')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create Isaac-specific launch file** - `vla_robot_controller/launch/isaac_vla_integration.launch.py`:
   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory

   def generate_launch_description():
       # Package directories
       pkg_isaac_ros = get_package_share_directory('isaac_ros_launch')
       pkg_vla = get_package_share_directory('vla_robot_controller')

       # Launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       world_file = LaunchConfiguration('world', default='simple_vla_test.sdf')

       declare_use_sim_time = DeclareLaunchArgument(
           'use_sim_time',
           default_value='true',
           description='Use simulation clock if true'
       )

       declare_world_arg = DeclareLaunchArgument(
           'world',
           default_value='simple_vla_test.sdf',
           description='World file for Isaac Sim'
       )

       # Include Isaac Sim launch
       isaac_sim = IncludeLaunchDescription(
           PythonLaunchDescriptionSource(
               os.path.join(pkg_isaac_ros, 'isaac_sim.launch.py')
           ),
           launch_arguments={
               'world': PathJoinSubstitution([pkg_vla, 'worlds', world_file]),
               'use_sim_time': use_sim_time
           }.items()
       )

       # VLA system node
       vla_system = Node(
           package='vla_robot_controller',
           executable='advanced_vla_node',
           name='advanced_vla_system',
           parameters=[
               {'use_sim_time': use_sim_time}
           ],
           output='screen'
       )

       # Isaac bridge node
       isaac_bridge = Node(
           package='vla_robot_controller',
           executable='isaac_bridge',
           name='isaac_bridge_node',
           parameters=[
               {'use_sim_time': use_sim_time}
           ],
           output='screen'
       )

       # RViz for visualization
       rviz_config = PathJoinSubstitution([
           get_package_share_directory('vla_robot_controller'),
           'rviz',
           'vla_robot.rviz'
       ])

       rviz = Node(
           package='rviz2',
           executable='rviz2',
           arguments=['-d', rviz_config],
           condition=LaunchConfigurationNotEquals('headless', 'true')
       )

       return LaunchDescription([
           declare_use_sim_time,
           declare_world_arg,
           isaac_sim,
           vla_system,
           isaac_bridge,
           rviz
       ])
   ```

4. **Run the complete integrated system**:
   ```bash
   # Terminal 1: Start Isaac Sim with VLA integration
   ros2 launch vla_robot_controller isaac_vla_integration.launch.py world:=simple_vla_test.sdf

   # Terminal 2: Test with commands
   ros2 topic pub /natural_language_command std_msgs/String "data: 'move to the red box'"

   # Terminal 3: Monitor the system
   ros2 topic echo /vla/status
   ```

#### Expected Results
After completing the integration:
- Isaac Sim should run with the test world
- VLA system should process natural language commands
- Robot should execute actions based on commands in simulation
- All perception, language understanding, and action execution should work together

### Performance Testing and Validation

#### Test Scenarios

Create a comprehensive test script `vla_robot_controller/test/vla_performance_test.py`:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
import time
import threading
import numpy as np

class VLAPerformanceTester(Node):
    def __init__(self):
        super().__init__('vla_performance_tester')

        # Test parameters
        self.test_duration = 60  # seconds
        self.test_scenarios = [
            "move forward",
            "turn left",
            "go to the red object",
            "find the blue ball",
            "stop"
        ]

        # Publishers
        self.command_pub = self.create_publisher(String, '/natural_language_command', 10)
        self.performance_pub = self.create_publisher(Float32, '/vla/performance_score', 10)

        # Test results tracking
        self.test_start_time = None
        self.commands_sent = 0
        self.responses_received = 0
        self.response_times = []
        self.test_thread = None

        self.get_logger().info('VLA Performance Tester initialized')

    def run_performance_test(self):
        """Run comprehensive performance test"""
        self.get_logger().info(f'Starting performance test for {self.test_duration} seconds')

        self.test_start_time = time.time()

        # Send commands at regular intervals
        while time.time() - self.test_start_time < self.test_duration:
            start_time = time.time()

            # Send test command
            cmd_idx = self.commands_sent % len(self.test_scenarios)
            command = self.test_scenarios[cmd_idx]

            cmd_msg = String()
            cmd_msg.data = command
            self.command_pub.publish(cmd_msg)

            self.commands_sent += 1

            # Wait before sending next command
            processing_time = time.time() - start_time
            sleep_time = max(0.5 - processing_time, 0.1)  # Maintain ~2Hz command rate
            time.sleep(sleep_time)

        # Calculate and report results
        self.report_test_results()

    def report_test_results(self):
        """Report performance test results"""
        total_time = time.time() - self.test_start_time
        command_rate = self.commands_sent / total_time if total_time > 0 else 0

        avg_response_time = np.mean(self.response_times) if self.response_times else float('inf')
        std_response_time = np.std(self.response_times) if self.response_times else 0

        self.get_logger().info(f'=== VLA Performance Test Results ===')
        self.get_logger().info(f'Total test duration: {total_time:.2f}s')
        self.get_logger().info(f'Commands sent: {self.commands_sent}')
        self.get_logger().info(f'Command rate: {command_rate:.2f} Hz')
        self.get_logger().info(f'Avg response time: {avg_response_time:.3f}s')
        self.get_logger().info(f'Std response time: {std_response_time:.3f}s')

        # Calculate performance score (higher is better)
        performance_score = Float32()
        if avg_response_time != float('inf'):
            # Score based on response time (inversely proportional)
            response_score = max(0, 1.0 - (avg_response_time / 2.0))  # Max 2s response time
            command_rate_score = min(1.0, command_rate / 5.0)  # Max 5 Hz command rate
            performance_score.data = (response_score + command_rate_score) / 2.0
        else:
            performance_score.data = 0.0

        self.performance_pub.publish(performance_score)
        self.get_logger().info(f'Overall performance score: {performance_score.data:.3f}')


def main(args=None):
    rclpy.init(args=args)
    tester = VLAPerformanceTester()

    # Run test in separate thread to allow ROS spinning
    test_thread = threading.Thread(target=tester.run_performance_test)
    test_thread.start()

    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Performance test interrupted')
    finally:
        test_thread.join(timeout=2.0)  # Wait for test to finish
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Deployment and Optimization

#### Optimizing for Real-time Performance

```python
import torch
import torch_tensorrt
from torch.utils.mobile_optimizer import optimize_for_mobile

class OptimizedVLASystem:
    def __init__(self, vla_model):
        self.original_model = vla_model
        self.optimized_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_quantized = False

    def optimize_model(self):
        """Apply various optimization techniques"""

        # 1. Quantization for faster inference
        self.quantize_model()

        # 2. Pruning to reduce model size
        self.prune_model()

        # 3. TensorRT optimization
        self.tensorrt_optimize()

        return self.optimized_model

    def quantize_model(self):
        """Apply post-training quantization"""
        if not self.is_quantized:
            # Using PyTorch's dynamic quantization for CPU
            # For GPU, use torch_tensorrt
            self.optimized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            self.is_quantized = True

    def tensorrt_optimize(self):
        """Optimize model with TensorRT for GPU inference"""
        try:
            # Convert model to TensorRT optimized format
            example_inputs = self.get_example_inputs()

            trt_model = torch_tensorrt.compile(
                self.optimized_model.eval(),
                inputs=example_inputs,
                enabled_precisions={torch.float, torch.half},
                workspace_size=1 << 25,  # 32MB workspace
                truncate_long_and_double=True
            )

            self.optimized_model = trt_model
            print("Model optimized with TensorRT")
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            # Use original model as fallback

    def get_example_inputs(self):
        """Get example inputs for optimization trace"""
        vision_input = torch.randn(1, 3, 224, 224).to(self.device)
        language_input = torch.randint(0, 1000, (1, 20)).to(self.device)  # Token IDs
        state_input = torch.randn(1, 100).to(self.device)  # Robot state

        return [
            torch_tensorrt.Input(
                min_shape=[1, 3, 224, 224],
                opt_shape=[4, 3, 224, 224],
                max_shape=[8, 3, 224, 224]
            ),
            torch_tensorrt.Input(
                min_shape=[1, 20],
                opt_shape=[4, 20],
                max_shape=[8, 50]
            ),
            torch_tensorrt.Input(
                min_shape=[1, 100],
                opt_shape=[4, 100],
                max_shape=[8, 100]
            )
        ]

    def optimize_inference(self):
        """Additional inference optimizations"""
        # Enable cuDNN benchmarking for optimized kernels
        torch.backends.cudnn.benchmark = True

        # Enable TensorFloat-32 for A100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set memory fraction to prevent GPU memory issues
        if self.device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

    def profile_performance(self, num_iterations=100):
        """Profile model performance"""
        if self.optimized_model is None:
            return None

        # Warm up
        example_inputs = self.get_example_inputs()
        for _ in range(10):
            with torch.no_grad():
                _ = self.optimized_model(*example_inputs)

        # Profile
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = self.optimized_model(*example_inputs)
            torch.cuda.synchronize()  # For accurate timing on GPU
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'fps': fps,
            'num_iterations': num_iterations
        }

def deploy_optimized_vla():
    """Deploy optimized VLA system for production use"""
    # Initialize model
    vla_model = AdvancedVLAModel()  # Your trained model

    # Optimize for deployment
    optimizer = OptimizedVLASystem(vla_model)
    optimized_model = optimizer.optimize_model()

    # Apply additional optimizations
    optimizer.optimize_inference()

    # Profile performance
    perf_stats = optimizer.profile_performance()

    print(f"Optimized VLA deployed - Performance: {perf_stats['fps']:.2f} FPS")

    return optimized_model

if __name__ == "__main__":
    optimized_model = deploy_optimized_vla()
```

### Conclusion and Next Steps

This practical lab has provided hands-on experience with:

1. **Basic VLA System**: Implementation of a foundational Vision-Language-Action system
2. **Advanced VLA**: Integration of foundation models for enhanced performance
3. **Isaac Integration**: Connection with NVIDIA Isaac Sim for simulation testing
4. **Performance Optimization**: Techniques for real-time deployment
5. **Validation**: Comprehensive testing and evaluation procedures

The implemented system demonstrates key concepts of AI Robot Brains including perception, reasoning, and action, all integrated within the NVIDIA Isaac ecosystem. This foundation prepares you for more advanced topics in the course and real-world humanoid robotics applications.

To extend this work, consider implementing:
- More sophisticated perception systems (3D object detection, semantic segmentation)
- Advanced navigation and manipulation capabilities
- Multi-modal learning approaches
- Human-robot interaction improvements
- Safety and ethics considerations