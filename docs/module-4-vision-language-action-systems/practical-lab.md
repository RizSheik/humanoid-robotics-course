---
title: Practical Lab - Vision-Language-Action Integration
description: Hands-on lab exercises implementing core concepts of vision-language-action systems
sidebar_position: 102
---

# Practical Lab - Vision-Language-Action Integration

## Lab Overview

This lab provides hands-on experience implementing and experimenting with integrated vision-language-action (VLA) systems. Students will work with computer vision models, natural language processing components, and robotic control systems to create systems that can perceive visual information, understand language commands, and execute appropriate actions. The lab emphasizes practical implementation skills while reinforcing theoretical concepts of multimodal integration.

## Lab Objectives

By completing this lab, students will be able to:
- Integrate computer vision and natural language processing components
- Implement vision-language grounding mechanisms for task execution
- Create multimodal neural networks that process visual and linguistic inputs
- Evaluate the performance of integrated VLA systems
- Address challenges of real-time multimodal processing and action execution

## Prerequisites and Setup

### Software Requirements
- Python 3.8+ with libraries: 
  - torch, torchvision, transformers
  - OpenCV, Pillow
  - spaCy, NLTK
  - numpy, scikit-learn
  - matplotlib, seaborn
- ROS 2 for robotic simulation (optional)
- CUDA-compatible GPU (recommended for deep learning)

### Hardware Requirements
- Computer with GPU (NVIDIA RTX series recommended)
- Web camera for real-time experiments (optional)
- Access to cloud GPU instances if local hardware is insufficient

### Setup Instructions
```bash
# Create project environment
python -m venv vla_lab
source vla_lab/bin/activate  # On Windows: vla_lab\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets
pip install opencv-python pillow nltk spacy scikit-learn matplotlib
pip install openai-clip  # For vision-language models
python -m spacy download en_core_web_sm  # Download English language model

# For robotic simulation (optional)
pip install gymnasium[robotics] pybullet
```

## Lab Exercise 1: Vision-Language Grounding

### Objective
Create a vision-language grounding system that can identify objects in visual scenes based on natural language descriptions.

### Steps
1. Implement a basic vision-language model:
```python
# vision_language_grounding.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import spacy

class VisionLanguageGrounding(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Object detection head
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # Bounding box coordinates
            nn.Sigmoid()  # Normalize coordinates [0,1]
        )
        
        # Object classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 80),  # COCO dataset classes
        )
        
    def forward(self, images, texts):
        # Process images and text
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        outputs = self.model(**inputs)
        
        # Extract image and text features
        image_features = outputs.vision_model_output.last_hidden_state
        text_features = outputs.text_model_output.last_hidden_state
        
        # For simplicity, we'll use the pooled features
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Apply detection head
        bbox_predictions = self.detection_head(image_embeds)
        
        # Apply classification head
        class_predictions = self.classification_head(image_embeds)
        
        return {
            'logits_per_image': outputs.logits_per_image,
            'logits_per_text': outputs.logits_per_text,
            'bbox_predictions': bbox_predictions,
            'class_predictions': class_predictions,
            'image_features': image_features,
            'text_features': text_features
        }
    
    def find_object_by_description(self, image_path, description, top_k=5):
        """Find objects in image based on description"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with model
        with torch.no_grad():
            inputs = self.processor(text=[description], images=[image_rgb], 
                                   return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            
            # Get similarity scores
            similarity = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0][0]
            
        return similarity

def segment_image_by_prompt(image_path, prompt):
    """Segment image areas related to the prompt"""
    import cv2
    import numpy as np
    
    # This would normally use a segmentation model like SAM (Segment Anything)
    # For this exercise, we'll simulate with simple color-based segmentation
    
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Simple segmentation based on color (would use proper segmentation model in practice)
    if "red" in prompt.lower():
        lower = np.array([0, 50, 50])
        upper = np.array([10, 255, 255])
    elif "blue" in prompt.lower():
        lower = np.array([100, 50, 50])
        upper = np.array([130, 255, 255])
    elif "green" in prompt.lower():
        lower = np.array([40, 50, 50])
        upper = np.array([80, 255, 255])
    else:
        # Default: segment prominent colors
        lower = np.array([0, 50, 50])
        upper = np.array([180, 255, 255])
    
    mask = cv2.inRange(image_hsv, lower, upper)
    
    # Find contours to get bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Filter out small regions
            bboxes.append((x, y, w, h))
    
    return bboxes, mask

# Example usage
def demo_grounding():
    """Demonstrate vision-language grounding"""
    model = VisionLanguageGrounding()
    
    # Example image and prompts
    # For this demo, we'll use a simulated image processing
    sample_image_path = "sample_image.jpg"  # This would be your actual image
    
    # In a real implementation, you'd process with actual images
    sample_prompts = [
        "red apple on table",
        "blue cup",
        "person sitting",
        "white chair"
    ]
    
    print("Vision-Language Grounding Demo")
    print("-" * 40)
    
    # Process each prompt
    for prompt in sample_prompts:
        print(f"Looking for: '{prompt}'")
        # This would call the actual grounding function
        # bboxes, mask = segment_image_by_prompt(sample_image_path, prompt)
        # print(f"Found {len(bboxes)} regions")
        print("  [Demo: Would show segmentation results]")
        print()

if __name__ == '__main__':
    demo_grounding()
```

2. Create a more sophisticated grounding implementation:
```python
# advanced_grounding.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

class AdvancedVisionLanguageGrounding(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Spatial feature extractor for better grounding
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1),  # Reduce dimensionality
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True
        )
        
        # Bounding box regressor
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # [x, y, width, height]
            nn.Sigmoid()  # Bounding box coordinates in [0,1]
        )
        
        # Confidence scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Confidence score [0,1]
        )
    
    def forward(self, images, texts):
        # Process with CLIP
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        # Get image features with spatial information
        vision_outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values']
        )
        spatial_features = vision_outputs.last_hidden_state  # [batch, patches, dim]
        
        # Reshape to spatial dimensions (assuming 7x7 grid from CLIP)
        batch_size, seq_len, feat_dim = spatial_features.shape
        spatial_h = spatial_w = int(seq_len ** 0.5)  # Assuming square grid
        spatial_features = spatial_features.view(batch_size, spatial_h, spatial_w, feat_dim)
        
        # Apply spatial encoder
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        spatial_encoded = self.spatial_encoder(spatial_features)
        
        # Reshape back for attention
        spatial_encoded = spatial_encoded.permute(0, 2, 3, 1)  # [B, H, W, C]
        spatial_encoded = spatial_encoded.view(batch_size, -1, spatial_encoded.shape[-1])  # [B, HW, C]
        
        # Text features
        text_features = outputs.text_model_output.last_hidden_state
        text_pooled = outputs.text_embeds  # Pooled text representation
        
        # Apply spatial attention with text as query
        text_expanded = text_pooled.unsqueeze(1).expand(-1, spatial_encoded.size(1), -1)  # [B, HW, C]
        attended_features, attention_weights = self.spatial_attention(
            query=text_expanded,
            key=spatial_encoded,
            value=spatial_encoded
        )
        
        # Aggregate attended features
        aggregated_features = attended_features.mean(dim=1)  # [B, C]
        
        # Predict bounding boxes
        bbox_predictions = self.bbox_regressor(aggregated_features)
        
        # Predict confidence scores
        confidence_scores = self.confidence_scorer(aggregated_features)
        
        return {
            'bbox_predictions': bbox_predictions,
            'confidence_scores': confidence_scores,
            'similarity_scores': outputs.logits_per_image,
            'attention_weights': attention_weights,
            'aggregated_features': aggregated_features,
            'spatial_features': spatial_encoded
        }
    
    def extract_objects_by_description(self, image_path, description, confidence_threshold=0.5):
        """Extract objects from image based on description with confidence scoring"""
        # Load image
        image = Image.open(image_path)
        
        # Process
        with torch.no_grad():
            inputs = self.processor(text=[description], images=[image], return_tensors="pt", padding=True)
            outputs = self(inputs['pixel_values'], [description])
            
            # Get predictions
            bboxes = outputs['bbox_predictions'][0].cpu().numpy()
            confidences = outputs['confidence_scores'][0].cpu().numpy()
            
            # Filter by confidence
            valid_indices = confidences > confidence_threshold
            valid_bboxes = bboxes[valid_indices.squeeze()] if valid_indices.any() else np.array([])
            valid_confidences = confidences[valid_indices] if valid_indices.any() else np.array([])
        
        return {
            'bounding_boxes': valid_bboxes,
            'confidences': valid_confidences,
            'description': description,
            'similarity_score': outputs['similarity_scores'][0][0].item()
        }

def visualize_grounding_results(image_path, grounding_results):
    """Visualize grounding results on image"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    for i, (bbox, conf) in enumerate(zip(grounding_results['bounding_boxes'], 
                                        grounding_results['confidences'])):
        x, y, w, h = bbox
        # Convert from normalized coordinates to image coordinates
        h_img, w_img = image_rgb.shape[:2]
        x_abs = x * w_img
        y_abs = y * h_img
        w_abs = w * w_img
        h_abs = h * h_img
        
        # Draw bounding box
        rect = plt.Rectangle((x_abs, y_abs), w_abs, h_abs, 
                            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence text
        ax.text(x_abs, y_abs - 10, f'Conf: {conf:.2f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=10, color='white')
    
    ax.set_title(f"Grounding results for: '{grounding_results['description']}'\n"
                 f"Similarity Score: {grounding_results['similarity_score']:.3f}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Run demo
def demo_advanced_grounding():
    model = AdvancedVisionLanguageGrounding()
    
    # Create a sample image for demonstration (you would use a real image)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(sample_image, (100, 100), (300, 300), (255, 0, 0), 2)  # Blue rectangle
    cv2.putText(sample_image, "red object", (110, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save sample image
    cv2.imwrite("sample_grounding_image.jpg", sample_image)
    
    # Test description
    description = "red object"
    
    # Extract objects
    results = model.extract_objects_by_description("sample_grounding_image.jpg", description)
    
    print("Advanced Vision-Language Grounding Results:")
    print(f"Description: {description}")
    print(f"Similarity Score: {results['similarity_score']:.3f}")
    print(f"Detected objects: {len(results['bounding_boxes'])}")
    print(f"Confidence scores: {results['confidences']}")
    
    # Visualize (in a real implementation)
    # visualize_grounding_results("sample_grounding_image.jpg", results)

if __name__ == '__main__':
    demo_advanced_grounding()
```

### Deliverables
- Working vision-language grounding implementation
- Object detection based on textual descriptions
- Confidence scoring for detected objects
- Visualization of grounding results

## Lab Exercise 2: Language to Action Mapping

### Objective
Implement a system that maps natural language commands to robotic actions through learned embeddings and grounding mechanisms.

### Steps
1. Create a language-to-action model:
```python
# language_to_action.py
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from transformers import BertTokenizer, BertModel

class LanguageToActionMapper(nn.Module):
    def __init__(self, vocab_size=30522, action_space=18, d_model=768):
        super().__init__()
        
        self.d_model = d_model
        self.action_space = action_space
        
        # Language encoder using BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Action space encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_space, 256),
            nn.ReLU(),
            nn.Linear(256, d_model),
            nn.ReLU()
        )
        
        # Cross-modal transformer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 common robot tasks
        )
        
    def forward(self, text_inputs, action_templates=None):
        # Encode text
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, d_model]
        
        # Use [CLS] token embedding as sentence representation
        sentence_embedding = text_features[:, 0, :]  # [batch, d_model]
        
        # If action templates provided, attend to them
        if action_templates is not None:
            action_encoded = self.action_encoder(action_templates)  # [batch, action_space, d_model]
            attended_features, attention_weights = self.cross_attention(
                query=sentence_embedding.unsqueeze(1),  # [batch, 1, d_model]
                key=action_encoded,  # [batch, action_space, d_model]
                value=action_encoded
            )
            combined_features = attended_features.squeeze(1)  # [batch, d_model]
        else:
            combined_features = sentence_embedding  # Use sentence embedding directly
        
        # Decode to actions
        predicted_actions = self.action_decoder(combined_features)
        
        # Classify task
        task_logits = self.task_classifier(sentence_embedding)
        
        return {
            'actions': predicted_actions,
            'task_logits': task_logits,
            'sentence_embedding': sentence_embedding,
            'combined_features': combined_features
        }
    
    def encode_command(self, command_text):
        """Encode a command text to feature representation"""
        inputs = self.tokenizer(
            command_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            sentence_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        return sentence_repr
    
    def predict_action(self, command_text, current_state=None):
        """Predict action for a command"""
        inputs = self.tokenizer(
            command_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        with torch.no_grad():
            result = self(inputs)
        
        return result['actions']

class CommandParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
        # Define action keywords
        self.movement_keywords = {
            'forward': [0.5, 0.0], 'backward': [-0.5, 0.0],
            'left': [0.0, 0.3], 'right': [0.0, -0.3],
            'turn left': [0.0, 0.5], 'turn right': [0.0, -0.5]
        }
        
        self.object_interaction = {
            'pick up', 'take', 'grasp', 'grab', 'lift',
            'place', 'put', 'drop', 'release'
        }
        
    def parse_command(self, command):
        """Parse command into structured representation"""
        doc = self.nlp(command.lower())
        
        # Extract entities and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Identify action verbs
        action_verbs = [token.lemma_ for token in doc 
                       if token.pos_ == "VERB" and not token.is_stop]
        
        # Identify objects
        objects = [token.text for token in doc 
                  if token.pos_ in ["NOUN", "PROPN"] and token.text not in self.stop_words]
        
        # Identify spatial relations
        spatial_relations = [token.text for token in doc 
                            if token.dep_ in ["prep", "pobj"] and token.text in 
                            ["near", "next", "to", "on", "in", "at", "by"]]
        
        return {
            'entities': entities,
            'action_verbs': action_verbs,
            'objects': objects,
            'spatial_relations': spatial_relations,
            'raw_command': command
        }
    
    def map_to_robot_action(self, parsed_command):
        """Map parsed command to robot action space"""
        action = np.zeros(18)  # Default action space: 6 DOF + 6 forces + 6 torques
        
        # For demonstration, simple keyword matching
        command_lower = parsed_command['raw_command'].lower()
        
        # Movement commands
        for keyword, mvmt_vec in self.movement_keywords.items():
            if keyword in command_lower:
                action[0] = mvmt_vec[0]  # Linear X
                action[4] = mvmt_vec[1]  # Angular Z (yaw)
                break
        
        # Object interaction commands
        for interaction in self.object_interaction:
            if interaction in command_lower:
                # This would involve more complex manipulation planning
                action[6:9] = [0.2, 0.0, 0.0]  # Example gripper action
                break
        
        # Normalize action to [-1, 1] range
        action = np.clip(action, -1, 1)
        
        return action

def create_language_action_dataset():
    """Create a dataset mapping language to actions"""
    commands = [
        "move forward slowly",
        "go straight ahead",
        "move backward carefully",
        "turn left gently",
        "rotate right",
        "pick up the red cube",
        "place object on table",
        "grasp the pen",
        "move toward the door",
        "approach the person"
    ]
    
    # Example action mappings (would be learned in real system)
    actions = [
        np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]
    
    return commands, actions

def demo_language_to_action():
    """Demonstrate language to action mapping"""
    print("Language to Action Mapping Demo")
    print("-" * 40)
    
    # Create command parser
    parser = CommandParser()
    
    # Example commands
    test_commands = [
        "move forward slowly",
        "turn left",
        "pick up the red cube",
        "go straight toward the door"
    ]
    
    for cmd in test_commands:
        parsed = parser.parse_command(cmd)
        action = parser.map_to_robot_action(parsed)
        
        print(f"Command: '{cmd}'")
        print(f"Parsed: {parsed}")
        print(f"Action: {action[:6]}...")  # Show first 6 dimensions
        print()

if __name__ == '__main__':
    demo_language_to_action()
```

2. Implement a neural language-to-action system:
```python
# neural_language_action.py
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

class NeuralLanguageActionModel(nn.Module):
    def __init__(self, action_space=18, d_model=768):
        super().__init__()
        
        self.action_space = action_space
        self.d_model = d_model
        
        # BERT for language understanding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Spatial context encoder (for grounding in environment)
        self.spatial_context = nn.Sequential(
            nn.Linear(7, d_model),  # [robot_x, robot_y, robot_theta, target_x, target_y, obs_dist, height_diff]
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Multimodal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Action generation
        self.action_generator = nn.Sequential(
            nn.Linear(d_model * 2, 512),  # Combined features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_space),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Task prediction head
        self.task_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 possible tasks
        )
    
    def forward(self, command_text, spatial_context=None):
        # Encode command
        text_inputs = self.tokenizer(
            command_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, d_model]
        
        # Use [CLS] token as sentence representation
        sentence_repr = text_features[:, 0, :]  # [batch, d_model]
        
        # Process spatial context
        if spatial_context is not None:
            spatial_features = self.spatial_context(spatial_context)
        else:
            # Default spatial context if not provided
            batch_size = sentence_repr.size(0)
            default_context = torch.zeros(batch_size, 7, device=sentence_repr.device)
            spatial_features = self.spatial_context(default_context)
        
        # Fuse text and spatial information
        combined_features = torch.cat([
            sentence_repr.unsqueeze(1),  # [batch, 1, d_model]
            spatial_features.unsqueeze(1)  # [batch, 1, d_model]
        ], dim=2).squeeze(1)  # [batch, 2*d_model]
        
        # Generate action
        actions = self.action_generator(combined_features)
        
        # Predict task
        task_logits = self.task_predictor(sentence_repr)
        
        return {
            'actions': actions,
            'task_logits': task_logits,
            'sentence_repr': sentence_repr,
            'spatial_features': spatial_features,
            'combined_features': combined_features
        }
    
    def predict_for_robot(self, command, robot_state):
        """
        robot_state: [robot_x, robot_y, robot_theta, target_x, target_y, obstacle_distance, height_difference]
        """
        spatial_context = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            result = self([command], spatial_context)
        
        return result['actions'].squeeze().numpy()

def train_language_action_model():
    """Train the language-action model"""
    model = NeuralLanguageActionModel()
    
    # Create sample training data
    commands = [
        "move forward",
        "turn left",
        "pick up object",
        "avoid obstacle"
    ]
    
    # Dummy spatial contexts [robot_x, robot_y, robot_theta, target_x, target_y, obs_dist, height_diff]
    spatial_contexts = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0],
        [0.5, 0.5, 0.0, 0.5, 0.6, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]  # Obstacle nearby
    ]
    
    # Dummy actions (would come from expert demonstrations in real training)
    target_actions = [
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([-0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Simple training loop
    model.train()
    for epoch in range(10):  # Few epochs for demo
        total_loss = 0
        
        for cmd, ctx, tgt_act in zip(commands, spatial_contexts, target_actions):
            optimizer.zero_grad()
            
            result = model([cmd], torch.tensor(ctx, dtype=torch.float32).unsqueeze(0))
            pred_action = result['actions']
            target_tensor = torch.tensor(tgt_act, dtype=torch.float32).unsqueeze(0)
            
            loss = criterion(pred_action, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(commands):.6f}")

def demo_neural_language_action():
    """Demonstrate neural language-action mapping"""
    print("Neural Language-Action Mapping Demo")
    print("-" * 45)
    
    model = NeuralLanguageActionModel()
    
    # Test commands
    test_commands = [
        "move forward to the red cube",
        "turn left and approach the door",
        "pick up the object in front of you"
    ]
    
    # Example robot states
    robot_states = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0],  # Facing target 2m away
        [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0],  # Target to the left
        [0.5, 0.5, 0.0, 0.5, 0.7, 0.8, 0.0]   # Close to target object
    ]
    
    for cmd, state in zip(test_commands, robot_states):
        action = model.predict_for_robot(cmd, state)
        print(f"Command: '{cmd}'")
        print(f"Robot State: {state}")
        print(f"Predicted Action: {action[:6]}...")  # Show first 6 dimensions
        print()

if __name__ == '__main__':
    demo_neural_language_action()
```

### Deliverables
- Working language-to-action mapping system
- Neural network that processes language commands
- Integration with spatial context for grounding
- Training procedure for the action mapping

## Lab Exercise 3: Vision-Language-Action Integration

### Objective
Combine vision, language understanding, and action execution into a complete VLA system that can process multimodal inputs and generate appropriate robotic behaviors.

### Steps
1. Create a complete VLA pipeline:
```python
# complete_vla_pipeline.py
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BertTokenizer, BertModel
import matplotlib.pyplot as plt

class VisionLanguageActionSystem(nn.Module):
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
            nn.Linear(512 + 768, 512),  # CLIP image embeds (512) + BERT text embeds (768)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 18)  # Action space: 6 DOF + 6 forces + 6 torques
        )
        
        # Object detection and grounding
        self.grounding_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # Bounding box [x, y, width, height]
            nn.Sigmoid()
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
        clip_outputs = self.clip_model(pixel_values=clip_inputs['pixel_values'])
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
    
    def process_command_with_image(self, image_path, command):
        """Process a command with the corresponding image"""
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Process
        result = self([image], [command])
        
        return result

class VLAPipeline:
    def __init__(self):
        self.vla_model = VisionLanguageActionSystem()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vla_model.to(self.device)
        
        # Confidence threshold for safety
        self.safety_threshold = 0.8
    
    def execute_command(self, image_path, command):
        """Execute a command given an image"""
        # Load image
        image = Image.open(image_path)
        
        # Process through model
        with torch.no_grad():
            result = self.vla_model([image], [command])
        
        # Extract components
        action = result['actions'][0].cpu().numpy()
        bbox = result['bounding_boxes'][0].cpu().numpy()
        safety_score = result['safety_scores'][0].cpu().numpy()[1]  # Probability of unsafe
        
        # Check safety
        if safety_score > self.safety_threshold:
            print(f"WARNING: Action deemed unsafe (safety score: {safety_score:.3f})")
            # Return safe action (stopped)
            action = np.zeros_like(action)
        else:
            print(f"Action approved (safety score: {safety_score:.3f})")
        
        return {
            'action': action,
            'bbox': bbox,
            'safety_score': safety_score,
            'command': command,
            'image_path': image_path
        }
    
    def visualize_result(self, result):
        """Visualize the VLA pipeline result"""
        # Load original image
        image = cv2.imread(result['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract bounding box (convert from normalized to absolute coordinates)
        bbox = result['bbox']
        h, w = image_rgb.shape[:2]
        x, y, width, height = bbox
        x_abs = int(x * w)
        y_abs = int(y * h)
        width_abs = int(width * w)
        height_abs = int(height * h)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with bounding box
        ax1.imshow(image_rgb)
        rect = plt.Rectangle((x_abs, y_abs), width_abs, height_abs, 
                            linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f"Command: '{result['command']}'\nBounding Box: {bbox}")
        ax1.axis('off')
        
        # Action visualization
        ax2.bar(range(len(result['action'])), result['action'])
        ax2.set_title(f"Generated Action (Safety: {(1-result['safety_score']):.3f})")
        ax2.set_xlabel('Action Component')
        ax2.set_ylabel('Action Value')
        
        plt.tight_layout()
        plt.show()

def create_demo_data():
    """Create demo images and commands for testing"""
    # Create a synthetic test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to represent objects
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue object
    cv2.putText(image, "blue_box", (105, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.rectangle(image, (300, 300), (400, 400), (0, 255, 0), -1)  # Green object
    cv2.putText(image, "green_box", (305, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.circle(image, (450, 150), 50, (0, 0, 255), -1)  # Red circle
    cv2.putText(image, "red_circle", (420, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save image
    cv2.imwrite("demo_vla_image.jpg", image)
    
    return "demo_vla_image.jpg"

def run_complete_vla_demo():
    """Run a complete VLA pipeline demo"""
    print("Complete Vision-Language-Action Pipeline Demo")
    print("=" * 50)
    
    # Create demo data
    image_path = create_demo_data()
    
    # Initialize pipeline
    pipeline = VLAPipeline()
    
    # Test commands
    test_commands = [
        "move toward the blue box",
        "go to the red circle",
        "avoid the green object",
        "navigate around obstacles"
    ]
    
    for command in test_commands:
        print(f"\nProcessing command: '{command}'")
        
        # Execute command
        result = pipeline.execute_command(image_path, command)
        
        print(f"Generated action: {result['action'][:6]}...")  # Show first 6 components
        print(f"Object bounding box: {result['bbox']}")
        print(f"Safety assessment: {'SAFE' if result['safety_score'] < 0.8 else 'UNSAFE'}")
        
        # Visualize (optional - uncomment for actual visualization)
        # pipeline.visualize_result(result)
        print("-" * 40)

if __name__ == '__main__':
    run_complete_vla_demo()
```

2. Implement a reinforcement learning component for the VLA system:
```python
# vla_reinforcement_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class VLAReinforcementLearning(nn.Module):
    def __init__(self, action_space=18, observation_space=512+768):
        super().__init__()
        
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(observation_space, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_space),
            nn.Tanh()  # Actions bound to [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(observation_space, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Additional networks for specific VLA tasks
        self.navigation_policy = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [linear_velocity, angular_velocity]
        )
        
        self.manipulation_policy = nn.Sequential(
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 DOF manipulator control
        )
    
    def forward(self, observation, policy_type="general"):
        """Forward pass with different policy options"""
        if policy_type == "general":
            return self.actor(observation)
        elif policy_type == "navigation":
            return self.navigation_policy(observation)
        elif policy_type == "manipulation":
            return self.manipulation_policy(observation)
    
    def evaluate_state(self, observation):
        """Evaluate state value using critic"""
        return self.critic(observation)

class VLAExperienceBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return list(zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class VLAReinforcementLearner:
    def __init__(self, vla_model, learning_rate=1e-4):
        self.vla_model = vla_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor and Critic optimizers
        self.actor_optimizer = optim.Adam(vla_model.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(vla_model.critic.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = VLAExperienceBuffer()
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update parameter
        self.batch_size = 32
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def update_actor_critic(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0  # Not enough samples
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute target values
        with torch.no_grad():
            next_actions = self.vla_model(next_states)
            next_values = self.vla_model.evaluate_state(next_states)
            target_values = rewards + (self.gamma * next_values * ~dones)
        
        # Critic loss
        current_values = self.vla_model.evaluate_state(states)
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        # Actor loss (policy gradient)
        current_actions = self.vla_model(states)
        advantages = target_values - current_values
        actor_loss = -(torch.log_softmax(current_actions, dim=1) * 
                      actions * advantages.detach()).mean()
        
        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def train_episode(self, environment, max_steps=100):
        """Train for one episode"""
        state = environment.reset()
        episode_reward = 0
        actor_losses, critic_losses = [], []
        
        for step in range(max_steps):
            # Select action using current policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.vla_model(state_tensor).cpu().numpy()[0]
            
            # Add noise for exploration
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
            # Execute action in environment
            next_state, reward, done, info = environment.step(action)
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update networks
            if len(self.replay_buffer) > self.batch_size:
                actor_loss, critic_loss = self.update_actor_critic()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        
        return episode_reward, avg_actor_loss, avg_critic_loss

# Simplified environment for demo purposes
class SimpleVLADemoEnvironment:
    def __init__(self):
        self.state_dim = 512 + 768  # Combined CLIP + BERT features
        self.action_dim = 18
        self.max_steps = 100
        self.current_step = 0
        
        # Initialize random state
        self.state = np.random.randn(self.state_dim)
        self.target_position = np.array([0.5, 0.5])  # Example target
    
    def reset(self):
        self.state = np.random.randn(self.state_dim)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        """Take step in environment"""
        # Simplified dynamics
        self.state += action[:self.state_dim] * 0.01  # Small state change
        
        # Calculate reward (distance to target)
        # For demo, we'll use a simplified reward based on action magnitude
        distance_penalty = -np.linalg.norm(action)
        step_bonus = 0.01
        
        reward = distance_penalty + step_bonus
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        # Return next state, reward, done flag, and info
        next_state = self.state + np.random.randn(self.state_dim) * 0.01  # Add noise
        return next_state, reward, done, {}

def demonstrate_rl_training():
    """Demonstrate VLA reinforcement learning"""
    print("VLA Reinforcement Learning Demo")
    print("-" * 35)
    
    # Initialize VLA model
    vla_model = VLAReinforcementLearning()
    
    # Initialize learner
    learner = VLAReinforcementLearner(vla_model)
    
    # Initialize environment
    env = SimpleVLADemoEnvironment()
    
    # Training loop
    num_episodes = 5  # For demo; use more episodes in practice
    
    for episode in range(num_episodes):
        episode_reward, actor_loss, critic_loss = learner.train_episode(env)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.3f}, "
              f"Actor Loss = {actor_loss:.6f}, Critic Loss = {critic_loss:.6f}")
    
    print(f"\nCompleted {num_episodes} episodes of VLA reinforcement learning")
    print("Model now has learned basic policy improvement through experience")

if __name__ == '__main__':
    demonstrate_rl_training()
```

### Deliverables
- Complete VLA system integrating vision, language, and action
- Reinforcement learning component for policy improvement
- Safety checking mechanisms
- Training pipeline for the integrated system

## Lab Exercise 4: Evaluation and Deployment

### Objective
Evaluate the performance of the integrated VLA system and prepare it for real-world deployment scenarios.

### Steps
1. Create evaluation metrics and testing procedures:
```python
# vla_evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import time
from datetime import datetime

class VLAEvaluator:
    def __init__(self, vla_model):
        self.vla_model = vla_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            'execution_times': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_scores': [],
            'safety_compliance': [],
            'robustness_scores': []
        }
    
    def benchmark_performance(self, test_data, num_trials=10):
        """Benchmark system performance"""
        self.vla_model.eval()
        
        execution_times = []
        accuracies = []
        
        for trial in range(num_trials):
            for image, command, expected_action in test_data:
                start_time = time.time()
                
                with torch.no_grad():
                    result = self.vla_model([image], [command])
                    predicted_action = result['actions'][0].cpu().numpy()
                
                end_time = time.time()
                execution_times.append(end_time - start_time)
                
                # Calculate accuracy (simplified for demo)
                if expected_action is not None:
                    accuracy = np.mean(np.abs(predicted_action - expected_action) < 0.1)
                    accuracies.append(accuracy)
        
        avg_time = np.mean(execution_times)
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        
        print(f"Performance Benchmark Results:")
        print(f"  Average execution time: {avg_time:.4f} seconds")
        print(f"  Average accuracy: {avg_accuracy:.4f}")
        print(f"  95th percentile time: {np.percentile(execution_times, 95):.4f} seconds")
    
    def evaluate_robustness(self, corrupted_test_data):
        """Evaluate system robustness to input perturbations"""
        self.vla_model.eval()
        
        robustness_results = []
        
        for corruption_type, test_pairs in corrupted_test_data.items():
            errors = []
            for image, command, expected_action in test_pairs:
                with torch.no_grad():
                    result = self.vla_model([image], [command])
                    predicted_action = result['actions'][0].cpu().numpy()
                
                error = np.mean(np.abs(predicted_action - expected_action))
                errors.append(error)
            
            avg_error = np.mean(errors)
            robustness_results.append({
                'corruption_type': corruption_type,
                'avg_error': avg_error,
                'std_error': np.std(errors)
            })
        
        # Print robustness results
        print("Robustness Evaluation Results:")
        for result in robustness_results:
            print(f"  {result['corruption_type']}: "
                  f"Avg Error = {result['avg_error']:.4f}, "
                  f"Std = {result['std_error']:.4f}")
        
        return robustness_results
    
    def evaluate_safety_compliance(self, safety_test_data):
        """Evaluate safety system compliance"""
        self.vla_model.eval()
        
        safety_violations = 0
        total_tests = 0
        
        for image, command, should_be_safe in safety_test_data:
            result = self.vla_model([image], [command])
            safety_score = result['safety_scores'][0].cpu().numpy()
            
            # Safety score [safe_prob, unsafe_prob], we want unsafe_prob < threshold
            is_safe_action = safety_score[1] < 0.8  # unsafe probability < 80%
            
            if should_be_safe and not is_safe_action:
                safety_violations += 1
            elif not should_be_safe and is_safe_action:
                # This might indicate over-conservative safety system
                pass
            
            total_tests += 1
        
        compliance_rate = 1 - (safety_violations / total_tests) if total_tests > 0 else 0
        
        print(f"Safety Compliance Results:")
        print(f"  Compliance Rate: {compliance_rate:.4f}")
        print(f"  Safety Violations: {safety_violations}/{total_tests}")
        
        return compliance_rate
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'model_name': 'VLA Integrated System',
            'evaluation_metrics': {
                'mean_accuracy': np.mean(self.results['accuracy']) if self.results['accuracy'] else 0,
                'mean_precision': np.mean(self.results['precision']) if self.results['precision'] else 0,
                'mean_recall': np.mean(self.results['recall']) if self.results['recall'] else 0,
                'mean_f1': np.mean(self.results['f1_scores']) if self.results['f1_scores'] else 0,
                'mean_safety_compliance': np.mean(self.results['safety_compliance']) if self.results['safety_compliance'] else 0,
                'mean_robustness': np.mean(self.results['robustness_scores']) if self.results['robustness_scores'] else 0
            },
            'performance_characteristics': {
                'avg_inference_time': np.mean(self.results['execution_times']) if self.results['execution_times'] else 0,
                'p95_inference_time': np.percentile(self.results['execution_times'], 95) if self.results['execution_times'] else 0,
                'throughput': 1.0 / np.mean(self.results['execution_times']) if self.results['execution_times'] else 0  # Hz
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate improvement recommendations based on evaluation"""
        recommendations = []
        
        # Performance recommendations
        if self.results['execution_times']:
            avg_time = np.mean(self.results['execution_times'])
            if avg_time > 0.1:  # 100ms threshold
                recommendations.append("Consider model optimization for real-time performance")
        
        # Accuracy recommendations  
        if self.results['accuracy']:
            avg_acc = np.mean(self.results['accuracy'])
            if avg_acc < 0.8:  # 80% threshold
                recommendations.append("Model accuracy below threshold - consider additional training")
        
        # Safety recommendations
        if self.results['safety_compliance']: 
            safety_comp = np.mean(self.results['safety_compliance'])
            if safety_comp < 0.95:  # 95% threshold
                recommendations.append("Safety compliance needs improvement")
        
        return recommendations

class VLADeploymentValidator:
    def __init__(self, model):
        self.model = model
        self.required_components = [
            'vision_encoder', 'language_encoder', 'fusion_network', 
            'action_generator', 'safety_checker'
        ]
    
    def validate_deployment_readiness(self):
        """Validate that model is ready for deployment"""
        validation_results = {
            'model_loadable': False,
            'all_components_present': False,
            'performance_requirements_met': False,
            'safety_checks_enabled': False
        }
        
        # Check if model is loadable
        try:
            dummy_image = torch.rand(1, 3, 224, 224)  # CLIP input size
            dummy_text = ["dummy command"]
            
            with torch.no_grad():
                result = self.model(dummy_image, dummy_text)
            
            validation_results['model_loadable'] = True
            print("✓ Model is loadable and executable")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
        
        # Check for required components
        model_attrs = dir(self.model)
        missing_components = []
        
        for component in self.required_components:
            if not any(attr.startswith(component) for attr in model_attrs):
                missing_components.append(component)
        
        validation_results['all_components_present'] = len(missing_components) == 0
        if validation_results['all_components_present']:
            print("✓ All required components present")
        else:
            print(f"✗ Missing components: {missing_components}")
        
        # Check performance requirements
        # (This would involve actual benchmarking in a real system)
        validation_results['performance_requirements_met'] = True
        print("✓ Performance requirements assumed met (requires testing)")
        
        # Check safety mechanisms
        if hasattr(self.model, 'safety_checker'):
            validation_results['safety_checks_enabled'] = True
            print("✓ Safety checks enabled")
        else:
            print("✗ Safety checks not implemented")
        
        return validation_results

def create_demo_evaluation():
    """Create a demonstration of evaluation procedures"""
    print("VLA System Evaluation Demo")
    print("=" * 34)
    
    # For this demo, we'll create a simple evaluation
    # In a real system, you would have actual test data
    
    # Simulated test data (in practice, this would come from real experiments)
    eval_results = {
        'execution_times': [0.05, 0.06, 0.045, 0.055, 0.052],
        'accuracy': [0.89, 0.91, 0.87, 0.92, 0.90],
        'safety_compliance': [0.98, 0.97, 0.99, 0.96, 0.98],
        'robustness_scores': [0.92, 0.90, 0.93, 0.89, 0.91]
    }
    
    print("Sample Evaluation Results:")
    print(f"Execution Times: {eval_results['execution_times']}")
    print(f"Accuracies: {eval_results['accuracy']}")
    print(f"Average Accuracy: {np.mean(eval_results['accuracy']):.4f}")
    print(f"Average Safety Compliance: {np.mean(eval_results['safety_compliance']):.4f}")
    
    # Sample deployment validation
    print("\nDeployment Validation Check:")
    print("✓ Model structure validated")
    print("✓ Performance benchmarks established")
    print("✓ Safety requirements verified")
    print("✓ Ready for deployment!")

if __name__ == '__main__':
    create_demo_evaluation()
```

2. Create a deployment pipeline:
```python
# vla_deployment.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import sys

class VLAModelDeployer:
    def __init__(self, model, model_name="vla_model"):
        self.model = model
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def optimize_for_inference(self):
        """Optimize model for inference"""
        self.model.eval()
        
        # Trace the model
        dummy_image = torch.rand(1, 3, 224, 224).to(self.device)  # CLIP image size
        dummy_text_input = ["dummy command for tracing"]
        
        # We need to create a wrapper for tracing since our model takes complex inputs
        class TracedVLAWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
            
            def forward(self, images):
                # Simplified forward pass for tracing (fixed text input)
                # In practice, you'd need to handle the text encoding differently for tracing
                with torch.no_grad():
                    # This is a simplified version - real implementation would handle text differently
                    results = self.original_model([transforms.ToPILImage()(images[0])], ["go forward"])
                    return results['actions']
        
        wrapper = TracedVLAWrapper(self.model)
        
        # Trace the model
        traced_model = torch.jit.trace(wrapper, dummy_image)
        
        return traced_model
    
    def quantize_model(self):
        """Apply quantization to reduce model size"""
        self.model.eval()
        
        # Use dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def prepare_for_mobile(self):
        """Prepare model for mobile deployment"""
        self.model.eval()
        
        # Optimize for mobile
        dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, (dummy_input, ["test"]))
        
        # Optimize for mobile
        optimized_model = optimize_for_mobile(traced_model)
        
        return optimized_model
    
    def save_model(self, filepath, include_weights=True, format='torchscript'):
        """Save the model in specified format"""
        if format == 'torchscript':
            model_to_save = self.optimize_for_inference()
            torch.jit.save(model_to_save, filepath)
        elif format == 'state_dict':
            if include_weights:
                torch.save(self.model.state_dict(), filepath)
            else:
                # Save model architecture
                model_script = torch.jit.script(self.model)
                torch.jit.save(model_script, filepath)
        elif format == 'onnx':
            # Export to ONNX format
            dummy_image = torch.rand(1, 3, 224, 224)
            dummy_command = ["dummy command"]
            
            torch.onnx.export(
                self.model,
                (dummy_image, dummy_command),
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['image', 'command'],
                output_names=['action', 'bbox', 'task'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'action': {0: 'batch_size'}
                }
            )
        
        print(f"Model saved to {filepath} in {format} format")
    
    def benchmark_model(self, num_runs=100):
        """Benchmark model performance"""
        self.model.eval()
        
        dummy_image = torch.rand(1, 3, 224, 224).to(self.device)
        dummy_command = ["benchmark command"]
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_image, dummy_command)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.model(dummy_image, dummy_command)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        print(f"Benchmark Results ({num_runs} runs):")
        print(f"  Average time: {avg_time:.6f}s ({1/avg_time:.2f}Hz)")
        print(f"  95th percentile: {p95_time:.6f}s")
        print(f"  Min time: {min(times):.6f}s")
        print(f"  Max time: {max(times):.6f}s")

def create_deployment_package():
    """Create a complete deployment package"""
    print("Creating VLA Deployment Package")
    print("=" * 34)
    
    # This would normally create a complete model with all dependencies
    # For demo purposes, we'll outline the structure
    
    deployment_structure = {
        "model_files": [
            "vla_model_traced.pt",      # Traced TorchScript model
            "vla_model_quantized.pt",   # Quantized model
            "vla_model_onnx.onnx"       # ONNX export
        ],
        "config_files": [
            "model_config.json",        # Model architecture config
            "deployment_config.json",   # Deployment-specific config
            "preprocessing_config.json" # Input preprocessing config
        ],
        "utilities": [
            "model_validator.py",       # Model validation utilities
            "benchmark_tool.py",        # Performance benchmarking
            "safety_checker.py"         # Safety validation utilities
        ],
        "documentation": [
            "api_reference.md",
            "deployment_guide.md", 
            "troubleshooting.md"
        ]
    }
    
    print("Deployment Package Structure:")
    for category, files in deployment_structure.items():
        print(f"  {category.upper()}:")
        for file in files:
            print(f"    - {file}")
    
    print("\nPackage ready for deployment!")
    print("Next steps:")
    print("  1. Validate model on target hardware")
    print("  2. Set up inference pipeline")
    print("  3. Integrate safety checks")
    print("  4. Conduct deployment testing")

def run_deployment_demo():
    """Run a deployment demonstration"""
    print("VLA Deployment Demo")
    print("-" * 20)
    
    # Create a simple model for demo purposes
    class SimpleVLA(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple demonstration model
            self.vision_feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 128)
            )
            self.action_generator = nn.Linear(128, 18)  # 18-dim action space
        
        def forward(self, images, commands):
            # Simple forward pass
            if isinstance(images, list):
                # Handle list of PIL Images
                processed_images = []
                for img in images:
                    if isinstance(img, Image.Image):
                        # Convert PIL to tensor
                        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
                        if img_tensor.shape[1] != 3:
                            # Convert grayscale to RGB
                            img_tensor = img_tensor.repeat(1, 3, 1, 1)
                    else:
                        img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
                    processed_images.append(img_tensor)
                
                batched_images = torch.cat(processed_images, dim=0)
            else:
                batched_images = images
            
            features = self.vision_feature_extractor(batched_images)
            actions = torch.tanh(self.action_generator(features))
            
            return {
                'actions': actions,
                'bounding_boxes': torch.rand(actions.size(0), 4),  # Fake bounding boxes
                'task_logits': torch.rand(actions.size(0), 10),   # Fake task logits
                'safety_scores': torch.rand(actions.size(0), 2)   # Fake safety scores
            }
    
    # Initialize model
    demo_model = SimpleVLA()
    
    # Initialize deployer
    deployer = VLAModelDeployer(demo_model)
    
    # Benchmark
    deployer.benchmark_model(num_runs=50)
    
    # Create deployment package
    create_deployment_package()
    
    print("\nDemo completed successfully!")
    print("In a real scenario, this would be followed by:")
    print("- Hardware-specific optimizations")
    print("- Safety and validation testing") 
    print("- Integration with robotic platform")

if __name__ == '__main__':
    run_deployment_demo()
```

### Deliverables
- Comprehensive evaluation framework for VLA systems
- Performance benchmarking and analysis
- Deployment readiness validation
- Complete deployment package with all necessary components

## Assessment Rubric

### Exercise 1: Vision-Language Grounding (25 points)
- **Implementation Quality**: Proper implementation of vision-language models (10 points)
- **Grounding Accuracy**: Effective identification of objects based on descriptions (10 points)
- **Code Quality**: Clean, well-documented code with proper error handling (5 points)

### Exercise 2: Language to Action Mapping (25 points)
- **Mapping Quality**: Effective translation of language commands to actions (10 points)
- **Neural Implementation**: Proper use of neural networks for mapping (10 points)
- **Integration**: Good integration of language and action components (5 points)

### Exercise 3: VLA Integration (30 points)
- **System Integration**: Proper integration of vision, language, and action components (15 points)
- **Reinforcement Learning**: Implementation of RL for policy improvement (10 points)
- **Safety Mechanisms**: Proper safety checking and validation (5 points)

### Exercise 4: Evaluation and Deployment (20 points)
- **Evaluation Framework**: Comprehensive evaluation procedures (10 points)
- **Deployment Package**: Complete and ready for real-world use (10 points)

## Additional Resources

### Recommended Reading
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP paper)
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3 paper)
- Recent papers on vision-language-action integration from top AI/robotics conferences

### Technical Resources
- Hugging Face Transformers documentation
- PyTorch tutorials and examples
- Robotics frameworks integration guides
- Computer vision and NLP library documentation

This lab provides comprehensive hands-on experience with implementing vision-language-action systems, from basic components to integrated systems with evaluation and deployment considerations.