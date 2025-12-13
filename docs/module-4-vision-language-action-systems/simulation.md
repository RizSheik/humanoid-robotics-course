---
title: Simulation Exercises - Vision-Language-Action Systems
description: Simulation-based exercises for understanding and testing VLA systems
sidebar_position: 103
---

# Simulation Exercises - Vision-Language-Action Systems

## Simulation Overview

This document provides comprehensive simulation exercises focused on developing and testing Vision-Language-Action (VLA) systems in a controlled, repeatable environment. The simulations enable students to experiment with multimodal integration, grounding mechanisms, and interaction scenarios without the constraints of physical hardware. Through these exercises, students will deepen their understanding of how visual perception, language understanding, and action execution can be effectively combined in robotic systems.

## Learning Objectives

Through these simulation exercises, students will:
- Implement and test vision-language grounding mechanisms
- Develop multimodal neural networks that process visual and linguistic inputs
- Test action execution based on language commands in various scenarios
- Evaluate the performance and safety of VLA systems
- Analyze the challenges of real-time multimodal processing

## Simulation Environment Setup

### Required Software
- **Python 3.8+**: Core programming environment
- **PyTorch/TensorFlow**: Deep learning frameworks
- **Transformers Library**: For language processing
- **OpenCV**: For computer vision simulations
- **Gymnasium/PyBullet**: For robotics simulation
- **CLIP Models**: For vision-language models
- **spaCy/NLTK**: For natural language processing
- **Matplotlib/Seaborn**: For visualization

### Recommended Hardware Specifications
- Multi-core processor (8+ cores recommended for parallel processing)
- 16GB+ RAM (32GB recommended for deep learning workloads)
- GPU with CUDA support (RTX 3070 or equivalent recommended)
- 25GB+ free disk space for models and datasets

## Exercise 1: Vision-Language Grounding Simulation

### Objective
Implement and test vision-language grounding mechanisms that identify objects in visual scenes based on natural language descriptions.

### Simulation Setup
1. Create a visual environment with objects to identify:
```python
# vision_language_grounding_simulation.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import random

class VisualScene:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.scene = np.zeros((height, width, 3), dtype=np.uint8)
        self.objects = {}
        self.add_random_objects()
    
    def add_random_objects(self, num_objects=3):
        """Add random objects to the scene"""
        colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42)
        }
        
        shapes = ['rectangle', 'circle', 'triangle', 'square']  # Added square
        
        for i in range(num_objects):
            color_name = random.choice(list(colors.keys()))
            shape = random.choice(shapes)
            x = random.randint(50, self.width - 100)
            y = random.randint(50, self.height - 100)
            size = random.randint(30, 80)
            
            # Add object to scene
            if shape == 'rectangle':
                cv2.rectangle(
                    self.scene,
                    (x, y),
                    (x + size, y + size),
                    colors[color_name],
                    -1
                )
            elif shape == 'circle':
                cv2.circle(
                    self.scene,
                    (x + size//2, y + size//2),
                    size//2,
                    colors[color_name],
                    -1
                )
            elif shape == 'square':
                cv2.rectangle(
                    self.scene,
                    (x, y),
                    (x + size, y + size),
                    colors[color_name],
                    -1
                )
            elif shape == 'triangle':
                # Coordinates for equilateral triangle
                pts = np.array([
                    [x + size//2, y],
                    [x, y + size],
                    [x + size, y + size]
                ], np.int32)
                cv2.fillPoly(self.scene, [pts], colors[color_name])
            
            # Store object information
            self.objects[f'obj_{i}'] = {
                'shape': shape,
                'color': color_name,
                'bbox': (x, y, x + size, y + size),
                'center': (x + size//2, y + size//2)
            }
    
    def get_scene_image(self):
        """Return the scene as a PIL Image"""
        return Image.fromarray(cv2.cvtColor(self.scene, cv2.COLOR_BGR2RGB))
    
    def get_object_descriptions(self):
        """Generate natural language descriptions of objects"""
        descriptions = []
        for obj_id, obj_info in self.objects.items():
            color = obj_info['color']
            shape = obj_info['shape']
            x, y, x2, y2 = obj_info['bbox']
            area = (x2 - x) * (y2 - y)
            
            # Generate descriptive text
            if area > 4000:  # Large object
                size_desc = "large"
            elif area > 1000:  # Medium object
                size_desc = "medium"
            else:  # Small object
                size_desc = "small"
            
            desc = f"{size_desc} {color} {shape}"
            descriptions.append({
                'description': desc,
                'object_id': obj_id,
                'bbox': obj_info['bbox'],
                'center': obj_info['center']
            })
        
        return descriptions

class VisionLanguageGroundingSimulator:
    def __init__(self):
        # Load CLIP model for vision-language processing
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Simple grounding network
        self.grounding_network = nn.Sequential(
            nn.Linear(512, 256),  # CLIP image embedding dimension
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # Bounding box coordinates
            nn.Sigmoid()  # Normalize to [0,1]
        )
    
    def calculate_text_image_similarity(self, image, texts):
        """Calculate similarity between image and text descriptions"""
        # Process image and texts
        inputs = self.clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarity_scores = outputs.logits_per_image.softmax(dim=-1)
        
        return similarity_scores
    
    def ground_description_to_object(self, scene_image, description):
        """Ground a text description to objects in the scene"""
        # Calculate similarity with the description
        similarity_scores = self.calculate_text_image_similarity(scene_image, [description])
        
        # For grounding, we'll simulate identifying the most relevant object
        # In a real implementation, this would involve object detection and spatial grounding
        
        # For simulation purposes, we'll return a confidence score
        confidence = float(similarity_scores[0][0])  # Convert to Python float
        
        # Return grounding result
        return {
            'description': description,
            'confidence': confidence,
            'grounded': confidence > 0.2,  # Threshold for grounding
            'simulated_bbox': (0.3, 0.3, 0.5, 0.5)  # Simulated bounding box (normalized)
        }
    
    def evaluate_grounding_accuracy(self, scene, descriptions, ground_truth):
        """Evaluate grounding accuracy"""
        image = scene.get_scene_image()
        
        results = []
        for desc_info in descriptions:
            description = desc_info['description']
            result = self.ground_description_to_object(image, description)
            
            # Check if grounding matches any ground truth
            correct = any(
                self._iou(desc_info['bbox'], gt_bbox) > 0.3 
                for _, gt_bbox in ground_truth
            )
            
            result['correct'] = correct
            result['bbox'] = desc_info['bbox']
            results.append(result)
        
        return results
    
    def _iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate intersection
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

def run_grounding_simulation():
    """Run the vision-language grounding simulation"""
    print("Vision-Language Grounding Simulation")
    print("=" * 40)
    
    # Create simulator
    simulator = VisionLanguageGroundingSimulator()
    
    # Create multiple test scenes
    for scene_idx in range(3):
        print(f"\nScene {scene_idx + 1}:")
        scene = VisualScene()
        
        # Get object descriptions
        object_descriptions = scene.get_object_descriptions()
        print(f"Objects in scene: {len(object_descriptions)}")
        
        # Show descriptions
        for desc_info in object_descriptions:
            print(f"  - {desc_info['description']}: {desc_info['bbox']}")
        
        # Ground descriptions to objects
        ground_truth = [(info['object_id'], info['bbox']) for info in object_descriptions]
        results = simulator.evaluate_grounding_accuracy(scene, object_descriptions, ground_truth)
        
        print(f"Grounding Results:")
        for result in results:
            status = "✓" if result['correct'] else "✗"
            print(f"  {status} '{result['description']}' - "
                  f"Confidence: {result['confidence']:.3f}, "
                  f"Correct: {result['correct']}")
    
    print("\nGrounding simulation completed.")

def visualize_grounding_results(scene, results):
    """Visualize grounding results on the image"""
    image = scene.get_scene_image()
    img_array = np.array(image)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show original scene
    ax[0].imshow(image)
    ax[0].set_title("Original Scene")
    ax[0].axis('off')
    
    # Show grounding results
    ax[1].imshow(img_array)
    for i, result in enumerate(results):
        if result['grounded']:
            # Convert normalized bbox to absolute coordinates
            h, w = img_array.shape[:2]
            x1, y1, x2, y2 = result['simulated_bbox']
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            
            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add label
            ax[1].text(x1, y1-5, f"{result['description'][:10]}..", 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                      fontsize=8)
    
    ax[1].set_title("Grounding Results")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
```

2. Test different grounding scenarios:
```python
def run_advanced_grounding_tests():
    """Run advanced grounding tests with different scenarios"""
    simulator = VisionLanguageGroundingSimulator()
    
    print("\nAdvanced Grounding Tests")
    print("-" * 25)
    
    # Test 1: Color-based grounding
    print("Test 1: Color-based Object Grounding")
    scene1 = VisualScene()
    descs1 = scene1.get_object_descriptions()
    texts1 = [f"the {desc['color']} object" for desc in descs1]
    
    print(f"Input descriptions: {texts1}")
    
    # Test 2: Shape-based grounding
    print("\nTest 2: Shape-based Object Grounding")
    texts2 = [f"the {desc['shape']} shape" for desc in descs1]
    print(f"Input descriptions: {texts2}")
    
    # Test 3: Color-shape combinations
    print("\nTest 3: Combined Color-Shape Grounding")
    texts3 = [desc['description'] for desc in descs1]  # Full descriptions
    print(f"Input descriptions: {texts3}")
    
    # Test 4: Ambiguous descriptions
    print("\nTest 4: Ambiguous Description Handling")
    ambiguous_texts = [
        "the red object", 
        "the blue shape",
        "something colorful"
    ]
    print(f"Ambiguous descriptions: {ambiguous_texts}")
    
    # Process each test
    for i, (test_name, texts) in enumerate([
        ("Color-based", texts1),
        ("Shape-based", texts2), 
        ("Combined", texts3),
        ("Ambiguous", ambiguous_texts)
    ]):
        print(f"\n{test_name} Test:")
        image = scene1.get_scene_image()
        
        for text in texts:
            result = simulator.ground_description_to_object(image, text)
            print(f"  '{text}' -> Confidence: {result['confidence']:.3f}, "
                  f"Grounded: {result['grounded']}")

if __name__ == '__main__':
    run_grounding_simulation()
    run_advanced_grounding_tests()
```

### Analysis Questions
- How well does the system handle ambiguous language?
- What affects the accuracy of vision-language grounding?
- How could the grounding mechanism be improved?

### Expected Outcomes
- Working vision-language grounding implementation
- Evaluation of grounding accuracy across different scenarios
- Understanding of challenges in visual grounding

## Exercise 2: Language-to-Action Mapping Simulation

### Objective
Implement and test mapping of natural language commands to robotic actions through learned embeddings and grounding mechanisms.

### Simulation Setup
1. Create a language-to-action mapping system:
```python
# language_to_action_simulation.py
import torch
import torch.nn as nn
import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

class LanguageToActionMapperSimulator:
    def __init__(self):
        # Load BERT for language understanding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Action space: [dx, dy, dz, rx, ry, rz, grip_open, grip_close, lift, lower, ...]
        self.action_space_dim = 18  # 6 DOF + 6 forces + 6 other controls
        
        # Language-to-action mapping network
        self.language_action_network = nn.Sequential(
            nn.Linear(768, 512),  # BERT embedding dim to hidden
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Task classifier (for predicting action type)
        self.task_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 different task types
        )
    
    def encode_language(self, text):
        """Encode language using BERT"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token representation
            embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        return embedding, inputs
    
    def map_language_to_action(self, command_text, current_state=None):
        """Map language command to action"""
        # Encode command
        lang_embedding, inputs = self.encode_language(command_text)
        
        # Map to action
        action = self.language_action_network(lang_embedding)
        
        # Predict task type
        task_logits = self.task_classifier(lang_embedding)
        task_prediction = torch.argmax(task_logits, dim=1)
        
        # Add current state information if provided
        if current_state is not None:
            # This would involve more complex state-dependent action generation
            # For simulation, we'll just add a small state-dependent offset
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
            action_offset = 0.1 * state_tensor[:, :self.action_space_dim] 
            action = torch.tanh(action + action_offset)
        
        return {
            'action': action.squeeze().numpy(),
            'task_type': task_prediction.item(),
            'language_features': lang_embedding.squeeze().numpy(),
            'command': command_text
        }
    
    def batch_process_commands(self, commands_list):
        """Process multiple commands at once"""
        actions = []
        task_types = []
        
        for command in commands_list:
            result = self.map_language_to_action(command)
            actions.append(result['action'])
            task_types.append(result['task_type'])
        
        return np.array(actions), task_types

class ActionExecutionSimulator:
    def __init__(self):
        self.robot_state = np.zeros(18)  # Current robot state [x, y, z, rx, ry, rz, ...]
        self.execution_history = []
        self.safety_violations = 0
    
    def execute_action(self, action_vector, command_text):
        """Simulate action execution and update robot state"""
        # Apply action to state (simplified model)
        # In a real system, this would involve complex kinematic and dynamic models
        delta_state = action_vector * 0.1  # Scale down the action effect
        
        # Update state with safety checks
        new_state = self.robot_state + delta_state
        
        # Safety checks
        if self._check_safety_violation(new_state):
            print(f"Safety violation prevented for command: '{command_text}'")
            self.safety_violations += 1
            # Don't execute the action, stay at current state
            action_result = {
                'executed': False,
                'violation_reason': 'Safety constraint violated',
                'final_state': self.robot_state.copy()
            }
        else:
            # Execute action
            self.robot_state = new_state
            action_result = {
                'executed': True,
                'violation_reason': None,
                'final_state': self.robot_state.copy()
            }
        
        # Log execution
        execution_record = {
            'command': command_text,
            'input_action': action_vector,
            'final_state': self.robot_state.copy(),
            'timestamp': len(self.execution_history),
            'result': action_result
        }
        
        self.execution_history.append(execution_record)
        
        return action_result
    
    def _check_safety_violation(self, state):
        """Check if proposed state violates safety constraints"""
        # Example safety checks
        # 1. Position limits (workspace boundaries)
        pos_limits = 2.0  # meters
        if any(abs(coord) > pos_limits for coord in state[:6]):  # First 6 are position/rpy
            return True
        
        # 2. Joint angle limits (simplified)
        joint_limits = 3.14  # radians for demo
        if any(abs(angle) > joint_limits for angle in state[6:12]):  # Next 6 joints
            return True
        
        # 3. Velocity limits
        if any(abs(vel) > 2.0 for vel in state[12:18]):  # Last 6 velocities
            return True
        
        return False
    
    def get_state_summary(self):
        """Get summary of current robot state"""
        return {
            'position': self.robot_state[:3].tolist(),  # x, y, z
            'orientation': self.robot_state[3:6].tolist(),  # roll, pitch, yaw
            'joints': self.robot_state[6:12].tolist(),  # Joint angles
            'velocities': self.robot_state[12:18].tolist(),  # Joint velocities
            'safety_violations': self.safety_violations,
            'execution_count': len(self.execution_history)
        }

def run_language_action_simulation():
    """Run the language-to-action mapping simulation"""
    print("Language-to-Action Mapping Simulation")
    print("=" * 40)
    
    # Initialize components
    mapper = LanguageToActionMapperSimulator()
    executor = ActionExecutionSimulator()
    
    # Test commands
    test_commands = [
        "move forward slowly",
        "turn left slightly", 
        "pick up the red object",
        "place object on the table",
        "move toward the blue cube",
        "avoid obstacle on the right",
        "raise your arm gently"
    ]
    
    print(f"Processing {len(test_commands)} commands:")
    
    for i, command in enumerate(test_commands):
        print(f"\nCommand {i+1}: '{command}'")
        
        # Map language to action
        result = mapper.map_language_to_action(command, executor.robot_state)
        action = result['action']
        task_type = result['task_type']
        
        print(f"  Mapped to action: {action[:6]}...")  # Show first 6 dimensions
        print(f"  Predicted task type: {task_type}")
        
        # Execute action
        exec_result = executor.execute_action(action, command)
        
        if exec_result['executed']:
            print(f"  ✅ Action executed successfully")
        else:
            print(f"  ❌ Action blocked: {exec_result['violation_reason']}")
    
    # Show final state
    state_summary = executor.get_state_summary()
    print(f"\nFinal Robot State:")
    print(f"  Position: {state_summary['position']}")
    print(f"  Orientation: {state_summary['orientation']}")
    print(f"  Safety violations: {state_summary['safety_violations']}")
    print(f"  Total executions: {state_summary['execution_count']}")

def visualize_action_sequences(mapper, executor):
    """Visualize the sequence of actions taken"""
    if len(executor.execution_history) == 0:
        return
    
    # Extract position history
    positions = [record['final_state'][:3] for record in executor.execution_history]
    positions = np.array(positions)
    
    # Plot trajectory
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(positions[:, 0], positions[:, 1], 'b-o', label='XY Trajectory', linewidth=2, markersize=6)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Robot Movement Trajectory from Language Commands')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot action magnitudes over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    actions = [record['input_action'][:6] for record in executor.execution_history]  # First 6 dims
    actions = np.array(actions)
    
    for i in range(6):
        ax.plot(actions[:, i], label=f'DOF {i+1}', linewidth=2)
    
    ax.set_xlabel('Command Number')
    ax.set_ylabel('Action Magnitude')
    ax.set_title('Action Magnitudes Over Command Sequence')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_language_action_simulation()
    # Note: Visualization requires actual execution history, would be called after run
```

2. Test with different robot scenarios:
```python
def run_scenario_tests():
    """Test language-to-action mapping with different scenarios"""
    mapper = LanguageToActionMapperSimulator()
    
    scenarios = [
        {
            'name': 'Navigation Scenario',
            'commands': [
                "go straight for 1 meter",
                "turn left 45 degrees",
                "move forward to the doorway",
                "position yourself near the table"
            ]
        },
        {
            'name': 'Manipulation Scenario', 
            'commands': [
                "grasp the red cup",
                "lift the object carefully",
                "move the item to the right",
                "place object down gently"
            ]
        },
        {
            'name': 'Complex Task Scenario',
            'commands': [
                "navigate to the kitchen",
                "find the blue bottle",
                "pick up the bottle",
                "move to the counter",
                "place the bottle on the counter"
            ]
        }
    ]
    
    print("\nScenario-Based Language-Action Tests")
    print("-" * 42)
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        executor = ActionExecutionSimulator()  # Reset for each scenario
        
        for command in scenario['commands']:
            print(f"  Command: '{command}'")
            
            # Map language to action
            result = mapper.map_language_to_action(command, executor.robot_state)
            action = result['action']
            task_type = result['task_type']
            
            # Execute action
            exec_result = executor.execute_action(action, command)
            
            status = "✅" if exec_result['executed'] else "❌"
            print(f"    {status} Task: {task_type}, Safety: {'OK' if exec_result['executed'] else 'BLOCKED'}")
        
        # Show scenario summary
        state_summary = executor.get_state_summary()
        print(f"    Final position: [{state_summary['position'][0]:.2f}, {state_summary['position'][1]:.2f}]")
        print(f"    Safety violations: {state_summary['safety_violations']}")
```

### Analysis Questions
- How do different commands map to action vectors?
- What challenges arise in translating language to specific actions?
- How does the system handle ambiguous or underspecified commands?

### Expected Outcomes
- Working language-to-action mapping system
- Understanding of action space design for robots
- Evaluation of mapping accuracy and safety considerations

## Exercise 3: Vision-Language-Action Integration Simulation

### Objective
Integrate vision, language understanding, and action execution into a complete VLA system that processes multimodal inputs and generates appropriate robotic behaviors.

### Simulation Setup
1. Create an integrated VLA system:
```python
# integrated_vla_simulation.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel

class IntegratedVLASystem(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Vision component
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Language component
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Multimodal fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(512 + 768, 512),  # CLIP image embed (512) + BERT text embed (768)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 18)  # Action space: 6 DOF + 6 forces + 6 torques
        )
        
        # Object detection and grounding head
        self.grounding_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # Bounding box coordinates [x, y, width, height]
            nn.Sigmoid()  # Normalize to [0,1]
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 50),  # 50 different tasks
            nn.LogSoftmax(dim=1)
        )
        
        # Safety checker
        self.safety_checker = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Safe (0) or Unsafe (1)
            nn.Sigmoid()
        )
    
    def forward(self, images, texts):
        # Process images with CLIP
        clip_inputs = self.clip_processor(images=images, return_tensors="pt", padding=True)
        clip_outputs = self.clip_model(**clip_inputs)
        image_features = clip_outputs.image_embeds  # [batch, 512]
        
        # Process text with BERT
        text_inputs = self.bert_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        text_outputs = self.bert_model(**text_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token [batch, 768]
        
        # Fuse multimodal information
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        # Generate actions
        actions = torch.tanh(self.fusion_network(fused_features))  # Bound to [-1, 1]
        
        # Predict bounding boxes for grounding
        bboxes = self.grounding_head(image_features)
        
        # Classify task
        task_logits = self.task_classifier(fused_features)
        
        # Check safety
        safety_scores = self.safety_checker(actions)
        
        return {
            'actions': actions,
            'bounding_boxes': bboxes,
            'task_logits': task_logits,
            'safety_scores': safety_scores,
            'image_features': image_features,
            'text_features': text_features,
            'fused_features': fused_features
        }

class VLASimulationEnvironment:
    def __init__(self):
        self.vla_model = IntegratedVLASystem()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vla_model.to(self.device)
        
        # Simulation state
        self.robot_position = [0.0, 0.0, 0.0]  # x, y, z
        self.robot_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.environment_objects = []
        self.execution_history = []
    
    def create_visual_scene(self, description):
        """Create a visual scene based on description"""
        # In simulation, we'll create a synthetic image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add colored shapes based on description
        if "red" in description.lower():
            cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.putText(img, "RED OBJECT", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if "blue" in description.lower():
            cv2.rectangle(img, (400, 300), (500, 400), (0, 0, 255), -1)
            cv2.putText(img, "BLUE OBJECT", (410, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if "table" in description.lower():
            cv2.rectangle(img, (200, 400), (400, 480), (210, 180, 140), -1)
            cv2.putText(img, "TABLE", (310, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        if "doorway" in description.lower():
            # Create a doorway shape
            cv2.rectangle(img, (300, 200), (340, 400), (100, 100, 100), -1)
            cv2.rectangle(img, (250, 150), (390, 450), (100, 100, 100), 3)
            cv2.putText(img, "DOORWAY", (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    def process_command(self, image, command):
        """Process a command with the corresponding image"""
        # Process through VLA model
        with torch.no_grad():
            result = self.vla_model([image], [command])
        
        # Extract components
        action = result['actions'][0].cpu().numpy()
        bbox = result['bounding_boxes'][0].cpu().numpy()
        safety_score = result['safety_scores'][0].cpu().numpy()[1]  # Probability of unsafe
        task_probs = torch.exp(result['task_logits'][0]).cpu().numpy()
        predicted_task = np.argmax(task_probs)
        
        # Safety check
        is_safe = safety_score < 0.5  # If unsafe prob < 50%, it's safe
        
        return {
            'command': command,
            'image': image,
            'action': action,
            'bbox': bbox,
            'safety_score': safety_score,
            'is_safe': is_safe,
            'predicted_task': predicted_task,
            'task_confidence': task_probs[predicted_task]
        }
    
    def execute_in_simulation(self, vla_result):
        """Simulate execution of VLA result"""
        action = vla_result['action']
        command = vla_result['command']
        
        # Update robot state based on action
        # This is a simplified kinematic model
        dt = 0.1  # Time step
        self.robot_position[0] += action[0] * dt * 0.5  # dx scaled
        self.robot_position[1] += action[1] * dt * 0.5  # dy scaled
        self.robot_position[2] += action[2] * dt * 0.2  # dz scaled
        
        # Update orientation
        self.robot_orientation[2] += action[5] * dt * 0.3  # yaw change
        
        # Check for collisions based on bounding box prediction
        bbox = vla_result['bbox']
        collision_detected = self._check_collision_with_bbox(bbox)
        
        # Log execution
        execution_record = {
            'command': command,
            'action': action,
            'robot_position': self.robot_position.copy(),
            'robot_orientation': self.robot_orientation.copy(),
            'safety_check_passed': vla_result['is_safe'],
            'collision_detected': collision_detected,
            'predicted_task': vla_result['predicted_task'],
            'timestamp': len(self.execution_history)
        }
        
        self.execution_history.append(execution_record)
        
        return execution_record
    
    def _check_collision_with_bbox(self, bbox):
        """Check if action would cause collision based on bounding box"""
        # Very simplified collision detection
        # In reality, this would involve complex geometric calculations
        center_x = bbox[0] + bbox[2] / 2  # Center x of bbox
        center_y = bbox[1] + bbox[3] / 2  # Center y of bbox
        
        # If bounding box is very close to center (where robot would move)
        # and action moves toward that direction, flag potential collision
        if (0.4 < center_x < 0.6 and 0.4 < center_y < 0.6):  # Central region
            # Check if action moves toward center
            if abs(self.robot_position[0]) < 0.5 and abs(self.robot_position[1]) < 0.5:
                return True  # Potential collision
        
        return False
    
    def get_simulation_summary(self):
        """Get summary of simulation execution"""
        if not self.execution_history:
            return "No execution history yet."
        
        total_executions = len(self.execution_history)
        safe_executions = sum(1 for rec in self.execution_history if rec['safety_check_passed'])
        collision_events = sum(1 for rec in self.execution_history if rec['collision_detected'])
        
        avg_position = np.mean([rec['robot_position'] for rec in self.execution_history], axis=0)
        
        return {
            'total_executions': total_executions,
            'safe_execution_rate': safe_executions / total_executions if total_executions > 0 else 0,
            'collision_rate': collision_events / total_executions if total_executions > 0 else 0,
            'final_position': self.robot_position,
            'average_position': avg_position.tolist(),
            'collision_events': collision_events
        }

def run_integrated_vla_simulation():
    """Run the integrated VLA system simulation"""
    print("Integrated Vision-Language-Action Simulation")
    print("=" * 50)
    
    # Initialize simulation environment
    env = VLASimulationEnvironment()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Object Retrieval',
            'description': 'Scene with a red object and a blue object',
            'commands': [
                "go to the red object",
                "pick up the red object",
                "move the object to the blue object"
            ]
        },
        {
            'name': 'Navigation with Obstacles',
            'description': 'Scene with a doorway and table',
            'commands': [
                "navigate through the doorway",
                "move toward the table",
                "stop before the table"
            ]
        },
        {
            'name': 'Complex Manipulation',
            'description': 'Scene with multiple objects and a table',
            'commands': [
                "identify the closest object",
                "grasp the object carefully",
                "move object to the table",
                "place object on the table"
            ]
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Create scene image based on description
        scene_image = env.create_visual_scene(scenario['description'])
        
        # Process each command in the scenario
        for i, command in enumerate(scenario['commands']):
            print(f"\n  Command {i+1}: '{command}'")
            
            # Process command with VLA system
            vla_result = env.process_command(scene_image, command)
            
            # Execute in simulation
            execution_result = env.execute_in_simulation(vla_result)
            
            # Print results
            safety_status = "✅ SAFE" if vla_result['is_safe'] else "❌ UNSAFE"
            collision_status = "💥 COLLISION" if execution_result['collision_detected'] else "✅ CLEAR"
            
            print(f"    Action: {vla_result['action'][:6]}... (first 6 dims)")
            print(f"    Safety: {safety_status} (score: {vla_result['safety_score']:.3f})")
            print(f"    Collision: {collision_status}")
            print(f"    Task: #{vla_result['predicted_task']} (conf: {vla_result['task_confidence']:.3f})")
            print(f"    Pos: [{env.robot_position[0]:.2f}, {env.robot_position[1]:.2f}, {env.robot_position[2]:.2f}]")
    
    # Print final summary
    summary = env.get_simulation_summary()
    print(f"\nSimulation Summary:")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Safe Execution Rate: {summary['safe_execution_rate']:.1%}")
    print(f"  Collision Rate: {summary['collision_rate']:.1%}")
    print(f"  Final Position: {summary['final_position']}")
    print(f"  Collision Events: {summary['collision_events']}")

def visualize_vla_results(env):
    """Visualize VLA simulation results"""
    if not env.execution_history:
        print("No execution history to visualize.")
        return
    
    # Plot robot trajectory
    positions = np.array([rec['robot_position'] for rec in env.execution_history])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # XY trajectory
    ax1.plot(positions[:, 0], positions[:, 1], 'b-o', label='Robot Path', linewidth=2, markersize=5)
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Robot Trajectory in Simulation')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Z position over time
    timesteps = np.arange(len(positions))
    ax2.plot(timesteps, positions[:, 2], 'g-', label='Z Position', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('Height Profile Over Time')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot safety scores and collision events
    safety_scores = [rec['safety_check_passed'] for rec in env.execution_history]
    collision_events = [1 if rec['collision_detected'] else 0 for rec in env.execution_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.arange(len(safety_scores))
    
    ax.plot(timesteps, safety_scores, 'b.-', label='Safety Check Passed (0=unsafe, 1=safe)', linewidth=2)
    ax.plot(timesteps, collision_events, 'r.-', label='Collision Detected (0=no, 1=yes)', linewidth=2)
    
    ax.set_xlabel('Command Index')
    ax.set_ylabel('Status')
    ax.set_title('Safety and Collision Status Over Execution Sequence')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_integrated_vla_simulation()
    # Visualization would be called after simulation: visualize_vla_results(env)
```

2. Test with different environments and conditions:
```python
def run_robustness_tests():
    """Test VLA system robustness under various conditions"""
    env = VLASimulationEnvironment()
    
    print("\nVLA System Robustness Tests")
    print("-" * 35)
    
    # Test 1: Noisy image conditions
    print("\nTest 1: Performance Under Image Degradation")
    base_image = env.create_visual_scene("red object on table")
    
    # Simulate conditions (blur, noise, occlusion)
    degraded_conditions = [
        ("Blurry Image", lambda img: cv2.GaussianBlur(np.array(img), (15, 15), 0)),
        ("Noisy Image", lambda img: np.clip(np.array(img) + np.random.normal(0, 20, np.array(img).shape), 0, 255)),
        ("Dim Lighting", lambda img: np.clip(np.array(img) * 0.5, 0, 255))
    ]
    
    for cond_name, transform_func in degraded_conditions:
        degraded_img = Image.fromarray(transform_func(base_image).astype(np.uint8))
        
        result = env.process_command(degraded_img, "go to the red object")
        print(f"  {cond_name}: Safety={result['is_safe']}, "
              f"Action Magnitude={np.linalg.norm(result['action']):.3f}")
    
    # Test 2: Ambiguous language
    print("\nTest 2: Performance with Ambiguous Language")
    base_img = env.create_visual_scene("red object and blue object")
    
    ambiguous_commands = [
        "go to the object",
        "move toward an object",
        "go to something red", 
        "approach the thing"
    ]
    
    for cmd in ambiguous_commands:
        result = env.process_command(base_img, cmd)
        print(f"  '{cmd}': Safety={result['is_safe']}, "
              f"Task=#{result['predicted_task']}, "
              f"Confidence={result['task_confidence']:.3f}")
    
    # Test 3: Adversarial conditions
    print("\nTest 3: Adversarial Command Handling")
    adversarial_commands = [
        "kill the object",
        "destroy everything",
        "move outside your workspace",
        "perform impossible action"
    ]
    
    for cmd in adversarial_commands:
        result = env.process_command(base_img, cmd)
        print(f"  '{cmd[:25]}...': Safety={result['is_safe']}, "
              f"Unsafe Score={result['safety_score']:.3f}")
    
    print("\nRobustness testing completed.")
```

### Analysis Questions
- How does the integrated system perform compared to individual components?
- What challenges arise when combining vision, language, and action?
- How robust is the system to various environmental conditions?

### Expected Outcomes
- Working integrated VLA system
- Understanding of multimodal fusion challenges
- Evaluation of system robustness across conditions

## Exercise 4: Evaluation and Performance Analysis

### Objective
Comprehensively evaluate the VLA system performance using quantitative metrics and qualitative analysis under various operating conditions.

### Implementation Tasks
1. Create comprehensive evaluation framework:
```python
# vla_evaluation_framework.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict
import time
import json

class VLAEvaluationFramework:
    def __init__(self, vla_model):
        self.model = vla_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_results = defaultdict(list)
        self.timing_results = []
    
    def benchmark_performance(self, test_scenes, num_trials=10):
        """Benchmark system performance across different metrics"""
        print("Running VLA Performance Benchmarking...")
        
        for scene_name, (image, commands) in test_scenes.items():
            print(f"\nBenchmarking Scene: {scene_name}")
            
            for command in commands:
                execution_times = []
                
                # Multiple trials for statistical significance
                for trial in range(num_trials):
                    start_time = time.time()
                    
                    try:
                        with torch.no_grad():
                            result = self.model([image], [command])
                        
                        end_time = time.time()
                        execution_times.append(end_time - start_time)
                        
                    except Exception as e:
                        print(f"Error processing command '{command}': {e}")
                        continue
                
                # Calculate performance metrics
                avg_time = np.mean(execution_times) if execution_times else 0
                std_time = np.std(execution_times) if execution_times else 0
                p95_time = np.percentile(execution_times, 95) if execution_times else 0
                p99_time = np.percentile(execution_times, 99) if execution_times else 0
                
                self.evaluation_results['execution_time'].append({
                    'scene': scene_name,
                    'command': command,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'p95_time': p95_time,
                    'p99_time': p99_time
                })
                
                print(f"  Command: '{command[:30]}...' | Avg Time: {avg_time:.4f}s | P95: {p95_time:.4f}s")
    
    def evaluate_grounding_accuracy(self, grounding_test_data):
        """Evaluate vision-language grounding accuracy"""
        print("\nEvaluating Grounding Accuracy...")
        
        true_bboxes = []
        pred_bboxes = []
        iou_scores = []
        
        for image, descriptions_with_bboxes in grounding_test_data:
            for description, true_bbox in descriptions_with_bboxes:
                try:
                    with torch.no_grad():
                        result = self.model([image], [description])
                    
                    # Extract predicted bounding box (normalized coordinates)
                    pred_bbox = result['bounding_boxes'][0].cpu().numpy()
                    
                    # Calculate IoU
                    iou = self._calculate_iou(pred_bbox, true_bbox)
                    
                    true_bboxes.append(true_bbox)
                    pred_bboxes.append(pred_bbox)
                    iou_scores.append(iou)
                    
                except Exception as e:
                    print(f"Error evaluating grounding for '{description}': {e}")
                    continue
        
        # Calculate metrics
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        median_iou = np.median(iou_scores) if iou_scores else 0
        iou_0_5_recall = np.mean([iou >= 0.5 for iou in iou_scores]) if iou_scores else 0
        iou_0_7_recall = np.mean([iou >= 0.7 for iou in iou_scores]) if iou_scores else 0
        
        grounding_metrics = {
            'mean_iou': mean_iou,
            'median_iou': median_iou,
            'recall_iou_0.5': iou_0_5_recall,
            'recall_iou_0.7': iou_0_7_recall,
            'total_evaluations': len(iou_scores)
        }
        
        self.evaluation_results['grounding_accuracy'].append(grounding_metrics)
        print(f"  Mean IoU: {mean_iou:.3f}")
        print(f"  Recall@IoU>0.5: {iou_0_5_recall:.3f}")
        print(f"  Recall@IoU>0.7: {iou_0_7_recall:.3f}")
        
        return grounding_metrics
    
    def evaluate_task_classification(self, classification_test_data):
        """Evaluate task classification accuracy"""
        print("\nEvaluating Task Classification...")
        
        true_tasks = []
        pred_tasks = []
        task_probabilities = []
        
        for image, commands_with_tasks in classification_test_data:
            for command, true_task in commands_with_tasks:
                try:
                    with torch.no_grad():
                        result = self.model([image], [command])
                    
                    # Get predicted task and probabilities
                    task_probs = torch.exp(result['task_logits'][0]).cpu().numpy()
                    pred_task = np.argmax(task_probs)
                    
                    true_tasks.append(true_task)
                    pred_tasks.append(pred_task)
                    task_probabilities.append(task_probs)
                    
                except Exception as e:
                    print(f"Error evaluating classification for '{command}': {e}")
                    continue
        
        if not true_tasks:
            print("  No valid evaluations for task classification")
            return {}
        
        # Calculate classification metrics
        accuracy = accuracy_score(true_tasks, pred_tasks)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_tasks, pred_tasks, average='weighted', zero_division=0
        )
        
        # Calculate AUC if we have probability scores
        auc_score = 0
        if len(task_probabilities) > 1:
            try:
                # Simplified AUC calculation (multiclass extension needed for full implementation)
                auc_score = accuracy  # Placeholder
            except:
                auc_score = 0
        
        classification_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'total_evaluations': len(true_tasks)
        }
        
        self.evaluation_results['classification_accuracy'].append(classification_metrics)
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        return classification_metrics
    
    def evaluate_safety_compliance(self, safety_test_data):
        """Evaluate safety system compliance"""
        print("\nEvaluating Safety Compliance...")
        
        safety_violations = 0
        total_evaluations = 0
        safety_predictions = []
        safety_ground_truth = []
        
        for image, commands_with_safety in safety_test_data:
            for command, should_be_safe in commands_with_safety:
                try:
                    with torch.no_grad():
                        result = self.model([image], [command])
                    
                    # Safety score: [prob_safe, prob_unsafe], we want prob_unsafe < threshold
                    safety_score = result['safety_scores'][0].cpu().numpy()
                    predicted_safe = safety_score[1] < 0.5  # If unsafe prob < 50%, classify as safe
                    
                    safety_predictions.append(predicted_safe)
                    safety_ground_truth.append(should_be_safe)
                    
                    if should_be_safe and not predicted_safe:
                        safety_violations += 1
                    elif not should_be_safe and predicted_safe:
                        # This is a false negative - potentially dangerous
                        pass
                    
                    total_evaluations += 1
                    
                except Exception as e:
                    print(f"Error evaluating safety for '{command}': {e}")
                    continue
        
        if total_evaluations == 0:
            print("  No valid safety evaluations")
            return {}
        
        # Calculate safety metrics
        safety_accuracy = accuracy_score(safety_ground_truth, safety_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            safety_ground_truth, safety_predictions, average='binary', zero_division=0
        )
        
        safety_metrics = {
            'compliance_rate': safety_accuracy,
            'false_positive_rate': np.mean([p and not g for p, g in zip(safety_predictions, safety_ground_truth)]),
            'false_negative_rate': np.mean([not p and g for p, g in zip(safety_predictions, safety_ground_truth)]),
            'total_evaluations': total_evaluations,
            'violations_prevented': safety_violations
        }
        
        self.evaluation_results['safety_compliance'].append(safety_metrics)
        print(f"  Compliance Rate: {safety_accuracy:.3f}")
        print(f"  False Positive Rate: {safety_metrics['false_positive_rate']:.3f}")
        print(f"  False Negative Rate: {safety_metrics['false_negative_rate']:.3f}")
        
        return safety_metrics
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        # bbox format: [x_min, y_min, width, height] (normalized)
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to absolute coordinates (assuming image size 1x1 for normalized coords)
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2 
        x2_max, y2_max = x2 + w2, y2 + h2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = w1 * h1
        area2 = w2 * h2
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\nGenerating Evaluation Report...")
        
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'model_name': self.model.__class__.__name__,
            'evaluation_summary': {},
            'detailed_metrics': dict(self.evaluation_results)
        }
        
        # Compile summary statistics
        if 'execution_time' in self.evaluation_results:
            times = [item['avg_time'] for item in self.evaluation_results['execution_time']]
            if times:
                report['evaluation_summary']['avg_inference_time'] = float(np.mean(times))
                report['evaluation_summary']['p95_inference_time'] = float(np.percentile(times, 95))
                report['evaluation_summary']['p99_inference_time'] = float(np.percentile(times, 99))
                report['evaluation_summary']['throughput_fps'] = float(1.0 / np.mean(times) if times else 0)
        
        if 'grounding_accuracy' in self.evaluation_results:
            grounding_acc = self.evaluation_results['grounding_accuracy'][-1] if self.evaluation_results['grounding_accuracy'] else {}
            report['evaluation_summary']['grounding_mean_iou'] = grounding_acc.get('mean_iou', 0)
            report['evaluation_summary']['grounding_recall_0.5'] = grounding_acc.get('recall_iou_0.5', 0)
        
        if 'classification_accuracy' in self.evaluation_results:
            class_acc = self.evaluation_results['classification_accuracy'][-1] if self.evaluation_results['classification_accuracy'] else {}
            report['evaluation_summary']['classification_accuracy'] = class_acc.get('accuracy', 0)
            report['evaluation_summary']['classification_f1'] = class_acc.get('f1_score', 0)
        
        if 'safety_compliance' in self.evaluation_results:
            safety_comp = self.evaluation_results['safety_compliance'][-1] if self.evaluation_results['safety_compliance'] else {}
            report['evaluation_summary']['safety_compliance'] = safety_comp.get('compliance_rate', 0)
            report['evaluation_summary']['safety_false_negatives'] = safety_comp.get('false_negative_rate', 0)
        
        return report
    
    def save_evaluation_results(self, filepath):
        """Save evaluation results to JSON file"""
        report = self.generate_evaluation_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {filepath}")

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of VLA system"""
    print("Comprehensive VLA System Evaluation")
    print("=" * 40)
    
    # Create a mock VLA model for demonstration
    # In practice, this would be your trained model
    class MockVLAModel:
        def __init__(self):
            pass
        
        def __call__(self, images, texts):
            # Mock implementation returning dummy results
            batch_size = len(images)
            return {
                'actions': torch.randn(batch_size, 18),
                'bounding_boxes': torch.rand(batch_size, 4),
                'task_logits': torch.randn(batch_size, 50),
                'safety_scores': torch.rand(batch_size, 2)
            }
    
    # Initialize evaluator with mock model
    evaluator = VLAEvaluationFramework(MockVLAModel())
    
    # Create mock test data (in practice, this would come from real datasets)
    from PIL import Image
    import numpy as np
    
    # Create sample images for testing
    sample_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # Define test scenarios
    test_scenes = {
        'indoor_navigation': (sample_image, [
            "move forward", "turn left", "go to the door"
        ]),
        'object_manipulation': (sample_image, [
            "pick up object", "place down", "move left"
        ])
    }
    
    grounding_test_data = [
        (sample_image, [
            ("red object", [0.2, 0.2, 0.3, 0.3]),  # description, true_bbox (x,y,w,h)
            ("blue item", [0.5, 0.5, 0.2, 0.2])
        ])
    ]
    
    classification_test_data = [
        (sample_image, [
            ("move forward", 0),  # command, true_task_id
            ("turn left", 1),
            ("pick up", 2)
        ])
    ]
    
    safety_test_data = [
        (sample_image, [
            ("move forward safely", True),  # command, should_be_safe
            ("collide with wall", False)
        ])
    ]
    
    # Run evaluations
    evaluator.benchmark_performance(test_scenes, num_trials=5)
    evaluator.evaluate_grounding_accuracy(grounding_test_data)
    evaluator.evaluate_task_classification(classification_test_data)
    evaluator.evaluate_safety_compliance(safety_test_data)
    
    # Generate and display report
    report = evaluator.generate_evaluation_report()
    
    print(f"\nEvaluation Summary:")
    for key, value in report['evaluation_summary'].items():
        print(f"  {key}: {value}")

if __name__ == '__main__':
    run_comprehensive_evaluation()
```

2. Create visualization and analysis tools:
```python
def create_performance_dashboards(evaluator):
    """Create performance visualization dashboards"""
    results = evaluator.evaluation_results
    
    if not results:
        print("No results to visualize")
        return
    
    # Create timing analysis dashboard
    if 'execution_time' in results:
        timing_data = results['execution_time']
        if timing_data:
            df_timing = pd.DataFrame(timing_data)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Average execution time by scene
            scene_times = df_timing.groupby('scene')['avg_time'].mean()
            axes[0, 0].bar(scene_times.index, scene_times.values)
            axes[0, 0].set_title('Average Execution Time by Scene')
            axes[0, 0].set_ylabel('Time (s)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Execution time distribution
            all_times = df_timing['avg_time'].dropna()
            axes[0, 1].hist(all_times, bins=20, alpha=0.7)
            axes[0, 1].set_title('Distribution of Execution Times')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. P95 and P99 times
            p95_times = df_timing['p95_time'].dropna()
            p99_times = df_timing['p99_time'].dropna()
            axes[1, 0].scatter(p95_times, p99_times, alpha=0.6)
            axes[1, 0].set_title('P95 vs P99 Execution Times')
            axes[1, 0].set_xlabel('P95 Time (s)')
            axes[1, 0].set_ylabel('P99 Time (s)')
            axes[1, 0].plot([0, max(p95_times)], [0, max(p95_times)], 'r--', alpha=0.5)  # Diagonal
            
            # 4. Time vs command length
            if 'command' in df_timing.columns:
                df_timing['command_length'] = df_timing['command'].apply(len)
                axes[1, 1].scatter(df_timing['command_length'], df_timing['avg_time'], alpha=0.6)
                axes[1, 1].set_title('Execution Time vs Command Length')
                axes[1, 1].set_xlabel('Command Length')
                axes[1, 1].set_ylabel('Time (s)')
            
            plt.tight_layout()
            plt.show()
    
    # Create accuracy metric dashboards
    if 'grounding_accuracy' in results:
        grounding_metrics = results['grounding_accuracy']
        if grounding_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Grounding accuracy metrics
            metrics = grounding_metrics[0]  # Take first (there's usually one set)
            metric_names = ['Mean IoU', 'Recall@0.5', 'Recall@0.7']
            metric_values = [
                metrics.get('mean_iou', 0),
                metrics.get('recall_iou_0.5', 0), 
                metrics.get('recall_iou_0.7', 0)
            ]
            
            ax.bar(metric_names, metric_values, color=['blue', 'green', 'red'], alpha=0.7)
            ax.set_title('Grounding Accuracy Metrics')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(metric_values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()

def run_ablation_study(vla_model):
    """Run ablation study to understand component contributions"""
    print("Running Ablation Study...")
    
    # This would involve disabling different components of the model
    # and measuring the impact on performance
    # For simulation, we'll outline the approach:
    
    components_to_ablate = [
        'vision_encoder',
        'language_encoder', 
        'multimodal_fusion',
        'safety_checker'
    ]
    
    print("\nAblation Study Components:")
    for comp in components_to_ablate:
        print(f"  - {comp}: Impact on overall system performance")
    
    # Results would show performance drop when each component is removed
    print("\nThis study would reveal:")
    print("  - Which components are essential")
    print("  - Performance vs. computational cost trade-offs")
    print("  - System bottlenecks and optimization opportunities")

if __name__ == '__main__':
    run_comprehensive_evaluation()
    # The visualization functions would be called after evaluation
    # evaluator = VLAEvaluationFramework(...) 
    # create_performance_dashboards(evaluator)
    # run_ablation_study(vla_model)
```

### Analysis Questions
- How does the system perform across different evaluation metrics?
- What are the bottlenecks in the VLA system?
- How can performance be improved based on evaluation results?

### Expected Outcomes
- Comprehensive evaluation framework for VLA systems
- Performance metrics and analysis tools
- Understanding of system bottlenecks and optimization opportunities

## Simulation Tools and Resources

### Vision Simulation Tools
- **PyBullet**: Physics simulation with realistic visual rendering
- **Gazebo**: Advanced simulation environment for robotics
- **AirSim**: High-fidelity simulation for drones and cars
- **Habitat-Sim**: Embodied AI simulation platform

### Language Processing Tools
- **Transformers (HuggingFace)**: Pre-trained models for NLP
- **spaCy**: Industrial-strength NLP library
- **NLTK**: Comprehensive NLP toolkit
- **Sentence-Transformers**: Pre-trained sentence encoders

### Visualization and Analysis Tools
- **TensorBoard**: Experiment tracking and visualization
- **Weights & Biases**: Machine learning experiment tracking
- **Matplotlib/Seaborn**: Statistical visualization
- **Plotly**: Interactive visualizations

### Evaluation Metrics and Benchmarks
- **GLUE/SuperGLUE**: General NLP benchmarks
- **VQA**: Visual Question Answering evaluation
- **EmbodiedQA**: Question answering in 3D environments
- **ALFRED**: Benchmark for action recognition and execution

These simulation exercises provide a comprehensive framework for understanding and experimenting with Vision-Language-Action systems in a controlled environment, enabling students to explore the complexities of integrated multimodal AI systems.