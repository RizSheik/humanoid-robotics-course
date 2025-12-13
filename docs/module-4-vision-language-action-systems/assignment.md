---
title: Assignment - Vision-Language-Action System Design and Implementation
description: Comprehensive assignment on designing and implementing integrated VLA systems
sidebar_position: 104
---

# Assignment - Vision-Language-Action System Design and Implementation

## Assignment Overview

This assignment requires students to design, implement, and evaluate a complete Vision-Language-Action (VLA) system for a specific robotic application. Students will demonstrate understanding of multimodal integration, implementation of core VLA components, and evaluation of system performance in realistic scenarios. The assignment emphasizes both theoretical understanding and practical implementation skills necessary for creating intelligent robotic systems that can perceive, understand, and act based on natural language commands.

## Assignment Objectives

Students will demonstrate the ability to:
- Design integrated VLA system architectures for specific robotic applications
- Implement core components of vision-language-action integration systems
- Evaluate and analyze VLA system performance across multiple dimensions
- Apply safety and security considerations to VLA system design
- Document design decisions and justify technical choices
- Assess the ethical implications of VLA systems

## Assignment Structure

The assignment consists of four major components:
1. **System Design and Architecture** (25 points)
2. **Implementation and Core Components** (35 points)
3. **Evaluation and Analysis** (25 points)
4. **Documentation and Reflection** (15 points)

**Total Points: 100**

## Part 1: System Design and Architecture

### Task Description
Design a complete VLA system architecture for a specific robotic application, including specification of components, data flows, and integration approaches.

### Design Requirements

#### Application Options
Choose one of the following applications for your design:
A) **Household Assistant Robot** - Multi-room navigation and object manipulation in homes
B) **Industrial Inspection Robot** - Factory floor inspection with natural language reporting
C) **Healthcare Assistant Robot** - Hospital navigation and patient interaction
D) **Educational Robot** - Classroom interaction and demonstration tasks

#### Technical Requirements
1. **Vision System Requirements**:
   - Object detection and recognition capabilities
   - 3D scene understanding
   - Real-time processing (30+ FPS)
   - Robustness to lighting conditions
   - Integration with spatial reasoning

2. **Language System Requirements**:
   - Natural language understanding for commands
   - Dialogue management capabilities
   - Grounding to visual and spatial context
   - Support for task-specific language
   - Conversation for clarification

3. **Action System Requirements**:
   - Navigation and manipulation capabilities
   - Real-time action execution
   - Safety and collision avoidance
   - Task planning and execution
   - Multi-modal feedback control

4. **Integration Requirements**:
   - Seamless multimodal fusion
   - Real-time performance
   - Safety and robustness
   - Error handling and recovery
   - Scalability considerations

### Deliverable Requirements
1. **System Architecture Diagram** showing all major components and their interconnections
2. **Component Specification** including technologies, algorithms, and interfaces
3. **Data Flow Analysis** describing how information flows between modalities
4. **Performance Requirements** with specific metrics and targets
5. **Safety and Security Analysis** identifying potential risks and mitigation strategies
6. **Technology Stack Justification** explaining choice of frameworks and tools

### Submission Format
- Technical design document (8-12 pages including diagrams)
- Architecture diagrams using proper notation (UML, block diagrams, etc.)
- Technology selection rationale with comparison alternatives
- Safety and security risk analysis
- References to relevant literature supporting design choices

## Part 2: Implementation and Core Components

### Task Description
Implement key components of your designed VLA system, focusing on one core functionality with supporting components.

### Implementation Requirements

#### Minimum Implementation
Students must implement the following components:

**Core Functionality Implementation** (Students choose one focus area):
1. **Vision-Language Grounding System**: Map visual objects to language descriptions with spatial understanding
2. **Language-to-Action Mapping**: Convert natural language commands to executable robot actions
3. **Multimodal Fusion Network**: Integrate vision and language inputs for decision making
4. **Dialogue Management System**: Handle multi-turn conversations with action execution

**Supporting Components**:
- Data preprocessing pipeline
- Model training and evaluation framework
- Basic safety and validation checks
- Integration with simulation environment
- Performance monitoring and logging

#### Technical Implementation Tasks
1. **Data Processing Pipeline**:
   ```python
   # vision_language_grounding.py skeleton
   import torch
   import torch.nn as nn
   import numpy as np
   from transformers import CLIPProcessor, CLIPModel
   from PIL import Image
   
   class VisionLanguageGrounding(nn.Module):
       def __init__(self):
           super().__init__()
           
           # Load pre-trained vision and language models
           self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
           self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
           
           # Vision-language fusion module
           self.fusion_module = nn.Sequential(
               nn.Linear(512 + 768, 512),  # CLIP vision + BERT text features
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(512, 256),
               nn.ReLU(),
               nn.Linear(256, 4)  # Bounding box coordinates
           )
           
           # Task classifier (optional additional component)
           self.task_classifier = nn.Sequential(
               nn.Linear(512 + 768, 256),
               nn.ReLU(),
               nn.Linear(256, 50)  # 50 different tasks
           )
       
       def forward(self, images, texts):
           # Process images and texts through CLIP
           inputs = self.clip_processor(
               text=texts, 
               images=images, 
               return_tensors="pt", 
               padding=True, 
               truncation=True,
               max_length=77
           )
           
           outputs = self.clip_model(**inputs)
           
           # Extract features
           image_embeds = outputs.image_embeds  # [batch, 512]
           text_embeds = outputs.text_embeds    # [batch, 512]
           
           # Fuse vision and language features
           fused_features = torch.cat([image_embeds, text_embeds], dim=1)
           
           # Predict bounding box
           bbox_predictions = torch.sigmoid(self.fusion_module(fused_features))
           
           # Classify task
           task_logits = self.task_classifier(fused_features)
           
           return {
               'bounding_boxes': bbox_predictions,
               'task_logits': task_logits,
               'image_features': image_embeds,
               'text_features': text_embeds
           }
       
       def ground_command_to_object(self, image_path, command, object_classes=None):
           """
           Ground a language command to specific objects in the image
           
           Args:
               image_path: Path to input image
               command: Natural language command
               object_classes: Optional list of expected object classes
               
           Returns:
               Dictionary with grounding results
           """
           # Load and preprocess image
           image = Image.open(image_path)
           
           # Process with model
           with torch.no_grad():
               result = self([image], [command])
           
           # Post-process results
           bbox = result['bounding_boxes'][0].cpu().numpy()
           task_prediction = torch.argmax(result['task_logits'][0]).item()
           
           return {
               'command': command,
               'image_path': image_path,
               'grounded_bbox': bbox,
               'task_prediction': task_prediction,
               'confidence': float(torch.max(torch.softmax(result['task_logits'][0], dim=0)))
           }
   ```

2. **Training Framework**:
   Implement a training pipeline that prepares data and trains the model:
   ```python
   # training_framework.py
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader, Dataset
   import numpy as np
   
   class VLADataset(Dataset):
       def __init__(self, data_path, transforms=None):
           """
           Dataset class for VLA training
           Expected format: List of (image_path, text_description, bounding_box, task_label)
           """
           self.data = self.load_data(data_path)
           self.transforms = transforms
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           image_path, text, bbox, task = self.data[idx]
           
           # Load and process image
           image = Image.open(image_path)
           if self.transforms:
               image = self.transforms(image)
           
           return {
               'image': image,
               'text': text,
               'bbox': torch.tensor(bbox, dtype=torch.float32),
               'task': task
           }
   
   def train_vla_model(model, train_loader, val_loader, num_epochs=50):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = model.to(device)
       
       # Define loss functions
       bbox_criterion = nn.SmoothL1Loss()  # For bounding box regression
       task_criterion = nn.CrossEntropyLoss()  # For task classification
       
       optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
       scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
       
       best_val_loss = float('inf')
       training_history = {
           'train_losses': [],
           'val_losses': [],
           'train_bounding_box_losses': [],
           'train_task_losses': [],
           'val_bounding_box_losses': [],
           'val_task_losses': []
       }
       
       for epoch in range(num_epochs):
           # Training phase
           model.train()
           epoch_train_loss = 0
           epoch_train_bbox_loss = 0
           epoch_train_task_loss = 0
           
           for batch_idx, batch in enumerate(train_loader):
               images = [img.to(device) for img in batch['image']]
               texts = batch['text']
               bboxes = batch['bbox'].to(device)
               tasks = batch['task'].to(device)
               
               optimizer.zero_grad()
               
               # Forward pass
               outputs = model(images, texts)
               
               # Compute losses
               bbox_loss = bbox_criterion(outputs['bounding_boxes'], bboxes)
               task_loss = task_criterion(outputs['task_logits'], tasks)
               
               total_loss = bbox_loss + 0.5 * task_loss  # Weight task loss less heavily
               
               # Backward pass
               total_loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
               optimizer.step()
               
               epoch_train_loss += total_loss.item()
               epoch_train_bbox_loss += bbox_loss.item()
               epoch_train_task_loss += task_loss.item()
               
               if batch_idx % 10 == 0:  # Print progress
                   print(f'Epoch {epoch}, Batch {batch_idx}: Total Loss = {total_loss.item():.4f}, '
                         f'BBox Loss = {bbox_loss.item():.4f}, Task Loss = {task_loss.item():.4f}')
           
           # Validation phase
           model.eval()
           epoch_val_loss = 0
           epoch_val_bbox_loss = 0
           epoch_val_task_loss = 0
           
           with torch.no_grad():
               for batch in val_loader:
                   images = [img.to(device) for img in batch['image']]
                   texts = batch['text']
                   bboxes = batch['bbox'].to(device)
                   tasks = batch['task'].to(device)
                   
                   outputs = model(images, texts)
                   
                   bbox_loss = bbox_criterion(outputs['bounding_boxes'], bboxes)
                   task_loss = task_criterion(outputs['task_logits'], tasks)
                   total_loss = bbox_loss + 0.5 * task_loss
                   
                   epoch_val_loss += total_loss.item()
                   epoch_val_bbox_loss += bbox_loss.item()
                   epoch_val_task_loss += task_loss.item()
           
           # Calculate averages
           avg_train_loss = epoch_train_loss / len(train_loader)
           avg_val_loss = epoch_val_loss / len(val_loader)
           
           avg_train_bbox_loss = epoch_train_bbox_loss / len(train_loader)
           avg_val_bbox_loss = epoch_val_bbox_loss / len(val_loader)
           
           avg_train_task_loss = epoch_train_task_loss / len(train_loader)
           avg_val_task_loss = epoch_val_task_loss / len(val_loader)
           
           # Update scheduler
           scheduler.step(avg_val_loss)
           
           # Store history
           training_history['train_losses'].append(avg_train_loss)
           training_history['val_losses'].append(avg_val_loss)
           training_history['train_bounding_box_losses'].append(avg_train_bbox_loss)
           training_history['train_task_losses'].append(avg_train_task_loss)
           training_history['val_bounding_box_losses'].append(avg_val_bbox_loss)
           training_history['val_task_losses'].append(avg_val_task_loss)
           
           print(f'Epoch {epoch+1}/{num_epochs}:')
           print(f'  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
           print(f'  BBox - Train: {avg_train_bbox_loss:.4f}, Val: {avg_val_bbox_loss:.4f}')
           print(f'  Task - Train: {avg_train_task_loss:.4f}, Val: {avg_val_task_loss:.4f}')
           
           # Save best model
           if avg_val_loss < best_val_loss:
               best_val_loss = avg_val_loss
               torch.save(model.state_dict(), 'best_vla_model.pth')
               print(f'  New best model saved with validation loss: {best_val_loss:.4f}')
       
       return model, training_history
   ```

3. **Evaluation Framework**:
   Implement evaluation metrics and validation:
   ```python
   # evaluation_framework.py
   import torch
   import numpy as np
   from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
   import matplotlib.pyplot as plt
   
   class VLAEvaluator:
       def __init__(self, model, device='cpu'):
           self.model = model.to(device)
           self.device = device
       
       def evaluate_grounding_accuracy(self, test_dataset):
           """Evaluate vision-language grounding accuracy"""
           self.model.eval()
           ious = []
           task_accuracies = []
           
           with torch.no_grad():
               for sample in test_dataset:
                   image = sample['image'].unsqueeze(0).to(self.device)
                   text = [sample['text']]
                   true_bbox = sample['bbox'].numpy()
                   
                   result = self.model([image], text)
                   pred_bbox = result['bounding_boxes'][0].cpu().numpy()
                   
                   # Calculate IoU (Intersection over Union)
                   iou = self.calculate_iou(true_bbox, pred_bbox)
                   ious.append(iou)
                   
                   # Evaluate task prediction
                   true_task = sample['task'].item()
                   pred_task = torch.argmax(result['task_logits'][0]).item()
                   task_accuracies.append(int(true_task == pred_task))
           
           mean_iou = np.mean(ious)
           task_accuracy = np.mean(task_accuracies)
           
           return {
               'mean_iou': mean_iou,
               'task_accuracy': task_accuracy,
               'ious': ious,
               'task_correct': task_accuracies
           }
       
       def calculate_iou(self, bbox1, bbox2):
           """Calculate Intersection over Union between two bounding boxes (x, y, width, height)"""
           # Convert from [x, y, w, h] to [x1, y1, x2, y2]
           x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
           x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
           
           # Calculate intersection area
           inter_x1 = max(x1_1, x1_2)
           inter_y1 = max(y1_1, y1_2)
           inter_x2 = min(x2_1, x2_2)
           inter_y2 = min(y2_1, y2_2)
           
           if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
               return 0.0
           
           inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
           
           # Calculate union area
           area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
           area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
           union_area = area1 + area2 - inter_area
           
           return inter_area / union_area if union_area > 0 else 0.0
       
       def evaluate_efficiency(self, test_dataset, num_samples=100):
           """Evaluate computational efficiency"""
           import time
           
           self.model.eval()
           execution_times = []
           
           with torch.no_grad():
               for i, sample in enumerate(test_dataset):
                   if i >= num_samples:
                       break
                   
                   image = sample['image'].unsqueeze(0).to(self.device)
                   text = [sample['text']]
                   
                   start_time = time.time()
                   result = self.model([image], text)
                   end_time = time.time()
                   
                   execution_times.append(end_time - start_time)
           
           return {
               'mean_execution_time': np.mean(execution_times),
               'p95_execution_time': np.percentile(execution_times, 95),
               'p99_execution_time': np.percentile(execution_times, 99),
               'throughput': 1.0 / np.mean(execution_times)  # FPS
           }
       
       def generate_performance_report(self, test_dataset):
           """Generate comprehensive performance report"""
           grounding_results = self.evaluate_grounding_accuracy(test_dataset)
           efficiency_results = self.evaluate_efficiency(test_dataset)
           
           report = {
               'grounding_performance': grounding_results,
               'efficiency_metrics': efficiency_results,
               'overall_assessment': self._calculate_overall_score(
                   grounding_results, efficiency_results
               )
           }
           
           return report
       
       def _calculate_overall_score(self, grounding_results, efficiency_results):
           """Calculate overall performance score"""
           # Weighted score based on grounding accuracy and efficiency
           iou_score = grounding_results['mean_iou']
           task_score = grounding_results['task_accuracy']
           time_score = 1.0 / efficiency_results['mean_execution_time']  # Higher is better for speed
           
           # Normalize time score to [0, 1] (assuming 0.1s is acceptable)
           time_score = min(time_score / 10.0, 1.0)  # Cap at 1.0
           
           overall_score = 0.4 * iou_score + 0.3 * task_score + 0.3 * time_score
           return overall_score
   
   def visualize_results(model, test_samples, num_samples=5):
       """Visualize grounding results"""
       model.eval()
       
       fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 8))
       axes = axes.flatten()
       
       with torch.no_grad():
           for i in range(min(num_samples, len(test_samples))):
               sample = test_samples[i]
               image = sample['image']
               text = sample['text']
               true_bbox = sample['bbox']
               
               # Get predictions
               result = model([image], [text])
               pred_bbox = result['bounding_boxes'][0].cpu().numpy()
               
               # Convert tensor to PIL image for visualization
               img_np = np.transpose(image.numpy(), (1, 2, 0))
               img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0,1]
               
               axes[i].imshow(img_np)
               
               # Draw true bounding box
               true_rect = plt.Rectangle(
                   (true_bbox[0], true_bbox[1]), 
                   true_bbox[2], true_bbox[3],
                   linewidth=2, edgecolor='green', facecolor='none'
               )
               axes[i].add_patch(true_rect)
               
               # Draw predicted bounding box
               pred_rect = plt.Rectangle(
                   (pred_bbox[0], pred_bbox[1]), 
                   pred_bbox[2], pred_bbox[3],
                   linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
               )
               axes[i].add_patch(pred_rect)
               
               axes[i].set_title(f"'{text}'\nIoU: {calculate_iou(true_bbox.numpy(), pred_bbox):.3f}")
               axes[i].axis('off')
       
       plt.tight_layout()
       plt.show()
   ```

### Advanced Implementation Options (for extra credit)

Students may choose to implement these advanced features for additional points:

#### Option A: Multimodal Attention Mechanisms (+5 points)
Implement sophisticated attention mechanisms for better vision-language integration.

#### Option B: Reinforcement Learning Integration (+5 points)
Include RL components for learning from interaction and improving performance over time.

#### Option C: Safety and Ethical Considerations Implementation (+5 points)
Implement comprehensive safety checks and ethical decision-making components.

### Submission Requirements
- Complete source code with documentation
- Training and evaluation scripts
- Configuration files and model checkpoints
- Performance evaluation results
- Video demonstration of key functionality
- Installation and setup instructions

## Part 3: Evaluation and Performance Analysis

### Task Description
Comprehensively evaluate your VLA system using appropriate metrics and analyze its performance across multiple dimensions.

### Analysis Tasks

#### Quantitative Evaluation
1. **Accuracy Metrics**:
   - Vision-language grounding accuracy (IoU scores)
   - Task classification accuracy
   - Command execution success rate
   - Safety compliance rate

2. **Performance Metrics**:
   - Real-time execution capabilities
   - Memory and computational resource usage
   - Throughput and latency measurements
   - Scalability analysis

3. **Robustness Testing**:
   - Performance under various environmental conditions
   - Robustness to noisy inputs
   - Error recovery capabilities
   - Failure mode analysis

#### Qualitative Analysis
1. **System Behavior Assessment**:
   - Naturalness of interaction
   - Context awareness and grounding
   - Error handling and recovery
   - User experience considerations

2. **Design Trade-off Evaluation**:
   - Accuracy vs. speed trade-offs
   - Complexity vs. maintainability
   - Resource usage vs. capability
   - Safety vs. functionality balance

### Testing Scenarios
Create and test your system under various scenarios:
1. **Nominal Operation**: Standard operation conditions
2. **Disturbed Conditions**: Environmental disturbances
3. **Edge Cases**: Unusual or rare situations
4. **Stress Testing**: High-load or rapid-fire commands

### Analysis Methodology
1. **Statistical Analysis**: Use appropriate statistical methods to analyze results
2. **Comparative Analysis**: Compare with baseline approaches
3. **Sensitivity Analysis**: Evaluate system response to parameter changes
4. **Risk Assessment**: Document potential failure modes and mitigation

### Submission Requirements
- Detailed evaluation report with methodology and results
- Statistical analysis with appropriate tests
- Visualizations of key performance metrics
- Comparison with design requirements
- Identification of system limitations and improvement opportunities

## Part 4: Documentation and Reflection

### Task Description
Create comprehensive documentation and reflect on the design and implementation process.

### Report Sections

#### Executive Summary (0.5 pages)
- Brief overview of implemented VLA system
- Key design decisions and their rationale
- Main performance results and validation outcomes
- Overall assessment of system success

#### Technical Implementation (2-3 pages)
- Detailed explanation of implementation approach
- Justification for technical choices made
- Challenges encountered and solutions developed
- Code organization and architecture overview

#### Performance Evaluation (1-2 pages)
- Summary of validation results
- Analysis of strengths and weaknesses
- Comparison with design requirements
- Recommendations for improvements

#### Safety and Ethical Considerations (0.5-1 pages)
- Safety mechanisms implemented
- Ethical implications of the VLA system
- Privacy considerations
- Bias and fairness considerations

#### Lessons Learned (1 page)
- Key insights from the implementation process
- What worked well and what didn't
- Technical and process learnings
- Skills developed during the project

#### Future Work (0.5 pages)
- Suggested enhancements and extensions
- Research directions for continued work
- Technology trends and potential improvements
- Reflection on learning experience

### Documentation Requirements
- Complete API documentation for implemented components
- System installation and usage guide
- Configuration examples and best practices
- Troubleshooting guide for common issues

### Writing Quality Requirements
- Professional, technical writing style
- Clear, concise explanations with appropriate terminology
- Consistent formatting and structure
- Proper citations and references
- Logical flow and organization

## Application-Specific Requirements

### For Option A - Household Assistant Robot
- Navigation in cluttered home environments
- Object recognition and manipulation in domestic settings
- Natural language interaction with family members
- Safety considerations for household environments

### For Option B - Industrial Inspection Robot
- Robust operation in industrial environments
- Technical language understanding for inspection tasks
- Precise action execution for measurements and reporting
- Integration with industrial systems

### For Option C - Healthcare Assistant Robot
- Safe interaction with patients and medical staff
- Medical terminology understanding
- Privacy and HIPAA compliance considerations
- Gentle and appropriate robot behavior

### For Option D - Educational Robot
- Engaging educational content delivery
- Age-appropriate language understanding
- Interactive learning engagement
- Safety for educational environments with children

## Submission Requirements

### All Parts Combined
1. System Design Document (PDF format)
2. Complete Source Code Package with documentation
3. Performance Analysis Report (PDF format)
4. Technical Documentation and Reflection Report (PDF format)
5. Video Demonstration (5-10 minutes showing key functionality)
6. Project Summary Sheet (executive summary and key deliverables)

### Technical Requirements
- Code must be properly documented with comments
- All components must be functional and tested
- Analysis must be reproducible using provided code
- All deliverables must be consistent with each other
- Use appropriate version control (Git) with clear commit messages

## Grading Criteria

### Part 1: System Design and Architecture (25 points)
- Technical correctness and completeness (10 points)
- Appropriateness for the chosen application (5 points)
- Adequacy of design for specified requirements (5 points)
- Quality of documentation and presentation (5 points)

### Part 2: Implementation and Core Components (35 points)
- Correctness of implementation (15 points)
- Innovation and sophistication of approach (10 points)
- Code quality, documentation and maintainability (5 points)
- Demonstration of functionality (5 points)

### Part 3: Evaluation and Performance Analysis (25 points)
- Thoroughness of evaluation approach (10 points)
- Appropriateness of analysis methods (5 points)
- Quality of results and interpretation (5 points)
- Identification of limitations and improvement opportunities (5 points)

### Part 4: Documentation and Reflection (15 points)
- Quality of overall documentation (5 points)
- Depth of reflection and learning insights (5 points)
- Professional presentation and organization (3 points)
- Technical writing quality (2 points)

## Additional Resources

### Recommended Reading
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision
- Brown, T., et al. (2020). Language Models are Few-Shot Learners
- Recent papers on vision-language-action integration from top AI/robotics conferences
- Industry standards for robot safety and human-robot interaction

### Technical Resources
- CLIP model documentation and examples
- Hugging Face Transformers library
- PyTorch and TensorFlow documentation
- Robotics simulation environments (PyBullet, Gazebo, Habitat)
- Computer vision libraries (OpenCV, PIL)

## Deadline and Submission

- **Assignment Deadline**: [Insert specific date]
- **Late Submission Policy**: 5% reduction per day late
- **Submission Method**: Upload to course management system
- **File Size Limit**: 100MB total (use compression if necessary)

This assignment represents a significant project that integrates the vision, language, and action concepts learned in Module 4, requiring students to demonstrate both theoretical understanding and practical implementation skills for creating integrated AI robotic systems.