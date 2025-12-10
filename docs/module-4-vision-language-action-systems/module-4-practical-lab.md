---
id: module-4-practical-lab
title: 'Module 4 — Vision-Language-Action Systems | Chapter 5 — Practical Lab'
sidebar_label: 'Chapter 5 — Practical Lab'
sidebar_position: 5
---

# Chapter 5 — Practical Lab

## Vision-Language-Action Systems: Hands-On Implementation

In this practical lab, you will implement a complete Vision-Language-Action (VLA) system for humanoid robotics. You'll work with actual robotics frameworks and develop a system capable of interpreting natural language commands and executing corresponding physical actions.

### Lab Overview

This lab involves implementing a simplified VLA system using the NVIDIA Isaac ecosystem and ROS 2. The system will integrate:

- Computer vision for object detection and scene understanding
- Natural language processing for instruction interpretation
- Robotics control for action execution

### Prerequisites

Before starting this lab, ensure you have:

- ROS 2 Humble Hawksbill installed
- NVIDIA Isaac ROS packages (perception, manipulation)
- Isaac Sim for testing (if available)
- Python 3.8+ with PyTorch and Transformers libraries
- Basic knowledge of deep learning frameworks

### Lab Setup

#### 1. Environment Configuration

First, create the ROS package structure for your VLA system:

```bash
mkdir -p ~/vla_project/src
cd ~/vla_project
colcon build
source install/setup.bash
```

Now create the VLA package:

```bash
cd ~/vla_project/src
ros2 pkg create --build-type ament_python vla_system
```

#### 2. Dependencies Installation

In your workspace, install required Python dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install openai-clip  # For CLIP-based grounding
pip install opencv-python
pip install numpy
```

### Part 1: Vision Processing Module

Create the vision processing component in `~/vla_project/src/vla_system/vla_system/vision_processor.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for detection results
        self.detection_publisher = self.create_publisher(
            String,  # Define a custom message type later
            '/vision/detections',
            10
        )
        
        # Load pre-trained object detection model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        # Define COCO class names for visualization
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        self.get_logger().info('Vision processor initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process the image with the detection model
            detections = self.process_image(cv_image)
            
            # Publish the results
            self.publish_detections(detections)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, image):
        # Preprocess image for model
        transform = T.Compose([
            T.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Extract relevant information
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter detections by confidence
        detections = []
        for i in range(len(boxes)):
            if scores[i] > 0.5:  # Confidence threshold
                detections.append({
                    'bbox': boxes[i],
                    'label': self.coco_names[labels[i]],
                    'score': scores[i]
                })

        return detections

    def publish_detections(self, detections):
        # Convert detections to message and publish
        # (Implementation depends on your custom message definition)
        pass

def main(args=None):
    rclpy.init(args=args)
    vision_processor = VisionProcessor()
    
    try:
        rclpy.spin(vision_processor)
    except KeyboardInterrupt:
        pass
    finally:
        vision_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 2: Language Processing Module

Create the language processing component in `~/vla_project/src/vla_system/vla_system/language_processor.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import re

class LanguageProcessor(Node):
    def __init__(self):
        super().__init__('language_processor')
        
        # Subscribe to language commands
        self.subscription = self.create_subscription(
            String,
            '/robot/commands',
            self.command_callback,
            10
        )
        
        # Publisher for parsed commands
        self.parsed_publisher = self.create_publisher(
            String,  # Define a custom message type
            '/language/parsed',
            10
        )
        
        # Define action verbs
        self.action_verbs = {
            'move': ['go', 'move', 'navigate', 'walk', 'drive'],
            'pick': ['pick', 'grasp', 'grab', 'take', 'lift'],
            'place': ['place', 'put', 'set', 'drop', 'release'],
            'look': ['look', 'see', 'find', 'search', 'locate'],
            'grasp': ['grasp', 'grip', 'hold', 'catch'],
            'release': ['release', 'let go', 'drop', 'release']
        }
        
        # Define object categories
        self.object_categories = [
            'cup', 'bottle', 'book', 'box', 'chair', 'table',
            'person', 'ball', 'bowl', 'remote', 'laptop', 'phone',
            'key', 'toy', 'plant', 'monitor'
        ]
        
        # Define spatial relations
        self.spatial_relations = [
            'on', 'at', 'to', 'near', 'by', 'next to', 'in front of',
            'behind', 'left of', 'right of', 'above', 'below', 'under'
        ]
        
        self.get_logger().info('Language processor initialized')

    def command_callback(self, msg):
        try:
            command_text = msg.data
            parsed_command = self.parse_command(command_text)
            
            # Publish the parsed command
            parsed_msg = String()
            parsed_msg.data = str(parsed_command)
            self.parsed_publisher.publish(parsed_msg)
            
            self.get_logger().info(f'Parsed command: {parsed_command}')
            
        except Exception as e:
            self.get_logger().error(f'Error parsing command: {str(e)}')

    def parse_command(self, command_text):
        # Convert to lowercase for processing
        command_lower = command_text.lower()
        
        # Identify the main action
        action = self.identify_action(command_lower)
        
        # Extract objects
        objects = self.extract_objects(command_lower)
        
        # Identify spatial relations and locations
        locations = self.extract_spatial_info(command_lower)
        
        # Create parsed command structure
        parsed = {
            'action': action,
            'objects': objects,
            'locations': locations,
            'raw_command': command_text
        }
        
        return parsed

    def identify_action(self, command):
        for action_type, verbs in self.action_verbs.items():
            for verb in verbs:
                if verb in command:
                    return action_type
        return 'unknown'

    def extract_objects(self, command):
        found_objects = []
        for obj in self.object_categories:
            if obj in command:
                found_objects.append(obj)
        return found_objects

    def extract_spatial_info(self, command):
        found_relations = []
        for relation in self.spatial_relations:
            if relation in command:
                # Extract the part after the spatial relation
                parts = command.split(relation)
                if len(parts) > 1:
                    found_relations.append({
                        'relation': relation,
                        'location': parts[1].strip()
                    })
        return found_relations

def main(args=None):
    rclpy.init(args=args)
    language_processor = LanguageProcessor()
    
    try:
        rclpy.spin(language_processor)
    except KeyboardInterrupt:
        pass
    finally:
        language_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 3: Action Planning Module

Create the action planning component in `~/vla_project/src/vla_system/vla_system/action_planner.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class ActionPlanner(Node):
    def __init__(self):
        super().__init__('action_planner')
        
        # Subscribe to parsed language commands
        self.lang_sub = self.create_subscription(
            String,
            '/language/parsed',
            self.language_callback,
            10
        )
        
        # Subscribe to vision detections
        self.vision_sub = self.create_subscription(
            String,
            '/vision/detections',
            self.vision_callback,
            10
        )
        
        # Publisher for planned actions
        self.action_publisher = self.create_publisher(
            String,  # Custom action message
            '/robot/actions',
            10
        )
        
        # Store latest detections
        self.latest_detections = []
        self.latest_command = None
        
        self.get_logger().info('Action planner initialized')

    def language_callback(self, msg):
        try:
            self.latest_command = json.loads(msg.data)
            # If we have both vision and language data, plan action
            if self.latest_detections:
                self.plan_action()
        except Exception as e:
            self.get_logger().error(f'Error processing language input: {str(e)}')

    def vision_callback(self, msg):
        try:
            # Process vision message (in a real system this would be a custom message)
            self.latest_detections = self.parse_detection_msg(msg.data)
            # If we have both vision and language data, plan action
            if self.latest_command:
                self.plan_action()
        except Exception as e:
            self.get_logger().error(f'Error processing vision input: {str(e)}')

    def parse_detection_msg(self, msg_data):
        # Parse the detection message into usable format
        # This would depend on your detection message format
        return []

    def plan_action(self):
        if not self.latest_command or not self.latest_detections:
            return

        command = self.latest_command
        detections = self.latest_detections
        
        # Plan action based on command and detections
        planned_action = self.create_action_from_command(command, detections)
        
        # Publish planned action
        action_msg = String()
        action_msg.data = json.dumps(planned_action)
        self.action_publisher.publish(action_msg)
        
        self.get_logger().info(f'Planned action: {planned_action}')

    def create_action_from_command(self, command, detections):
        action_type = command.get('action', 'unknown')
        objects = command.get('objects', [])
        locations = command.get('locations', [])

        # Create appropriate action based on command type
        if action_type == 'move':
            return self.plan_navigation_action(objects, locations, detections)
        elif action_type == 'pick':
            return self.plan_manipulation_action('pick', objects, detections)
        elif action_type == 'place':
            return self.plan_manipulation_action('place', objects, detections)
        elif action_type == 'look':
            return self.plan_perception_action(objects, detections)
        else:
            return {'type': 'unknown', 'description': 'Unknown action'}

    def plan_navigation_action(self, objects, locations, detections):
        # Plan navigation to a location
        if locations:
            target_location = locations[0].get('location', '')
            # This would involve path planning to the location
            return {
                'type': 'navigate',
                'target': target_location,
                'description': f'Move to {target_location}'
            }
        elif objects:
            # Try to navigate to an object
            target_object = objects[0]
            # Find the object in detections
            for detection in detections:
                if detection['label'] == target_object:
                    # Plan navigation to the object
                    return {
                        'type': 'navigate',
                        'target': target_object,
                        'bbox': detection['bbox'],
                        'description': f'Move to {target_object}'
                    }
        
        return {'type': 'unknown', 'description': 'No navigation target'}

    def plan_manipulation_action(self, action_type, objects, detections):
        # Plan manipulation action for objects
        if objects:
            target_object = objects[0]
            # Find the object in detections
            for detection in detections:
                if detection['label'] == target_object:
                    return {
                        'type': action_type,
                        'target': target_object,
                        'bbox': detection['bbox'],
                        'description': f'{action_type.capitalize()} {target_object}'
                    }
        
        return {'type': 'unknown', 'description': f'No {action_type} target'}

    def plan_perception_action(self, objects, detections):
        # Plan perception action (e.g., look for objects)
        if objects:
            target_object = objects[0]
            # Check if object is already detected
            detected = any(det['label'] == target_object for det in detections)
            return {
                'type': 'perceive',
                'target': target_object,
                'detected': detected,
                'description': f'Look for {target_object}, found: {detected}'
            }
        
        return {'type': 'perceive', 'description': 'Look around'}

def main(args=None):
    rclpy.init(args=args)
    action_planner = ActionPlanner()
    
    try:
        rclpy.spin(action_planner)
    except KeyboardInterrupt:
        pass
    finally:
        action_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 4: Control Module

Create the control component in `~/vla_project/src/vla_system/vla_system/controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json

class VLAController(Node):
    def __init__(self):
        super().__init__('vla_controller')
        
        # Subscribe to planned actions
        self.action_sub = self.create_subscription(
            String,
            '/robot/actions',
            self.action_callback,
            10
        )
        
        # Publisher for robot commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher for gripper commands (if available)
        # self.gripper_pub = self.create_publisher(GripperCommand, '/gripper/command', 10)
        
        self.get_logger().info('VLA controller initialized')

    def action_callback(self, msg):
        try:
            action = json.loads(msg.data)
            self.execute_action(action)
        except Exception as e:
            self.get_logger().error(f'Error executing action: {str(e)}')

    def execute_action(self, action):
        action_type = action.get('type', 'unknown')
        
        if action_type == 'navigate':
            self.execute_navigation(action)
        elif action_type == 'pick':
            self.execute_manipulation('pick', action)
        elif action_type == 'place':
            self.execute_manipulation('place', action)
        elif action_type == 'perceive':
            self.execute_perception(action)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')

    def execute_navigation(self, action):
        target = action.get('target', 'unknown')
        self.get_logger().info(f'Navigating to {target}')
        
        # Create simple navigation command (in a real system, this would use navigation stack)
        twist = Twist()
        twist.linear.x = 0.2  # Move forward slowly
        twist.angular.z = 0.0
        
        # In a real system, you'd use the navigation2 stack
        # Here we'll just send a simple movement command
        self.cmd_vel_pub.publish(twist)
        
        # Stop after a short time
        self.create_timer(2.0, self.stop_robot)

    def execute_manipulation(self, manip_type, action):
        target = action.get('target', 'unknown')
        self.get_logger().info(f'Attempting to {manip_type} {target}')
        
        # In a real system, this would control the manipulator
        # For now, just log the action
        if manip_type == 'pick':
            self.get_logger().info(f'Picking up {target} - Gripper close command')
        elif manip_type == 'place':
            self.get_logger().info(f'Placing {target} - Gripper open command')

    def execute_perception(self, action):
        target = action.get('target', 'environment')
        detected = action.get('detected', False)
        self.get_logger().info(f'Perceiving {target}, found: {detected}')
        
        # In a real system, this might trigger more detailed perception
        if not detected:
            self.get_logger().info('Target not detected, continuing search...')

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = VLAController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Part 5: System Integration and Testing

Create a launch file to run the complete VLA system in `~/vla_project/src/vla_system/launch/vla_system.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vla_system',
            executable='vision_processor',
            name='vision_processor',
            output='screen'
        ),
        Node(
            package='vla_system',
            executable='language_processor',
            name='language_processor',
            output='screen'
        ),
        Node(
            package='vla_system',
            executable='action_planner',
            name='action_planner',
            output='screen'
        ),
        Node(
            package='vla_system',
            executable='controller',
            name='controller',
            output='screen'
        )
    ])
```

### Launch the System

First, make sure your Python scripts are executable and have proper entry points. Add this to your `setup.py` in the package:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'vla_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A package for Vision-Language-Action systems',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_processor = vla_system.vision_processor:main',
            'language_processor = vla_system.language_processor:main',
            'action_planner = vla_system.action_planner:main',
            'controller = vla_system.controller:main',
        ],
    },
)
```

Rebuild your package:

```bash
cd ~/vla_project
colcon build --packages-select vla_system
source install/setup.bash
```

Now you can launch the complete system:

```bash
ros2 launch vla_system vla_system.launch.py
```

### Testing the System

To test the system, publish commands to the appropriate topics:

```bash
# Send a navigation command
ros2 topic pub /robot/commands std_msgs/String "data: 'Move to the table'"

# Send a manipulation command
ros2 topic pub /robot/commands std_msgs/String "data: 'Pick up the red cup'"

# Simulate camera feed (if you don't have a real camera)
# Use a tool like image_publisher or simulate with a test image
```

### Lab Deliverables

For this lab, you should:

1. Implement the complete VLA pipeline as shown above
2. Test each component individually
3. Integrate the components and test the complete system
4. Document any modifications or improvements you made
5. Report on the challenges you faced and how you addressed them

### Troubleshooting Tips

1. **Ensure all nodes are in the same ROS domain**: Check that nodes can communicate by verifying ROS_DOMAIN_ID is the same.

2. **Check message compatibility**: Ensure all nodes publish and subscribe to compatible message types.

3. **Verify dependencies**: Make sure all required packages are installed and properly configured.

4. **Monitor logs**: Use `ros2 run rqt_console rqt_console` or check individual node logs to debug issues.

5. **Test incrementally**: Test each node separately before connecting them together.

### Extensions for Advanced Students

For additional challenge, consider implementing:

1. **Learning-based components**: Train a simple neural network for action prediction
2. **Safety validation**: Add collision checking and safety validation
3. **Memory/History**: Implement a system to remember past actions and states
4. **Multi-step planning**: Plan sequences of actions for complex tasks
5. **Error recovery**: Implement recovery behaviors when actions fail

This lab provides a foundation for developing more sophisticated Vision-Language-Action systems for robotics applications.