---
id: module-4-chapter-4-practical-exercises
title: 'Module 4 — Vision-Language-Action Systems | Chapter 4 — Practical Exercises'
sidebar_label: 'Chapter 4 — Practical Exercises'
sidebar_position: 5
---

# Chapter 4 — Practical Exercises

## Vision-Language-Action Integration in Humanoid Robotics

### Exercise 1: Multimodal Fusion Architecture Implementation

#### Objective
Implement a multimodal fusion architecture that effectively combines vision, language, and action modalities for humanoid robotics.

#### Background
Multimodal fusion is critical in VLA systems to create unified representations that capture the relationships between visual input, language commands, and action execution. This exercise focuses on implementing different fusion strategies.

#### Steps
1. Implement early fusion architecture
2. Implement late fusion architecture
3. Implement intermediate fusion with cross-attention
4. Compare performance across fusion strategies

```python
#!/usr/bin/env python3
"""Multimodal Fusion for Vision-Language-Action Systems"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math

class VisionEncoder(nn.Module):
    """Vision feature encoder"""
    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(64 * 7 * 7, output_dim)  # Adjust based on input size
        
    def forward(self, x):
        conv_out = self.conv_layers(x)
        flat_out = torch.flatten(conv_out, 1)
        return self.fc(flat_out)

class LanguageEncoder(nn.Module):
    """Language feature encoder"""
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)  # [B, seq_len, embed_dim]
        lstm_out, (hidden, _) = self.lstm(embedded)  # [B, seq_len, hidden_dim]
        # Use final hidden state as representation
        final_hidden = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=1)  # [B, hidden_dim]
        return self.projection(final_hidden)

class ActionDecoder(nn.Module):
    """Action decoder from multimodal features"""
    def __init__(self, input_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class EarlyFusionVLA(nn.Module):
    """Early fusion VLA architecture - combines modalities early"""
    def __init__(self, vision_dim=512, language_dim=512, action_dim=19):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.language_encoder = LanguageEncoder(hidden_dim=language_dim)
        
        # Early fusion - concatenate and process together
        fusion_input_dim = vision_dim + language_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.action_decoder = ActionDecoder(256, action_dim)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, vision_input, language_input):
        # Encode modalities separately
        vision_features = self.vision_encoder(vision_input)
        language_features = self.language_encoder(language_input)
        
        # Early fusion - concatenate features
        concatenated = torch.cat([vision_features, language_features], dim=-1)
        
        # Process fused representation
        fused_features = self.fusion_network(concatenated)
        
        # Generate action and value
        action = self.action_decoder(fused_features)
        value = self.value_head(fused_features)
        
        return action, value, fused_features

class LateFusionVLA(nn.Module):
    """Late fusion VLA architecture - combines modality outputs"""
    def __init__(self, vision_dim=512, language_dim=512, action_dim=19):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.language_encoder = LanguageEncoder(hidden_dim=language_dim)
        
        # Separate processing branches
        self.vision_branch = nn.Sequential(
            nn.Linear(vision_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.language_branch = nn.Sequential(
            nn.Linear(language_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Late fusion network
        fusion_output_dim = 256 + 256  # vision + language features
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.action_decoder = ActionDecoder(256, action_dim)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, vision_input, language_input):
        # Encode modalities separately
        vision_features = self.vision_encoder(vision_input)
        language_features = self.language_encoder(language_input)
        
        # Process in separate branches
        vision_processed = self.vision_branch(vision_features)
        language_processed = self.language_branch(language_features)
        
        # Late fusion - concatenate processed features
        fused_input = torch.cat([vision_processed, language_processed], dim=-1)
        fused_features = self.fusion_network(fused_input)
        
        # Generate action and value
        action = self.action_decoder(fused_features)
        value = self.value_head(fused_features)
        
        return action, value, fused_features

class CrossAttentionFusionVLA(nn.Module):
    """Cross-attention based fusion VLA architecture"""
    def __init__(self, vision_dim=512, language_dim=512, action_dim=19, num_heads=8):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.language_encoder = LanguageEncoder(hidden_dim=language_dim)
        
        # Cross-attention modules
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=min(vision_dim, language_dim),
            num_heads=num_heads,
            batch_first=True
        )
        
        self.language_vision_attention = nn.MultiheadAttention(
            embed_dim=min(vision_dim, language_dim), 
            num_heads=num_heads,
            batch_first=True
        )
        
        # For this implementation, we'll use a common dimension
        self.common_dim = min(vision_dim, language_dim)
        self.vision_projection = nn.Linear(vision_dim, self.common_dim)
        self.language_projection = nn.Linear(language_dim, self.common_dim)
        
        # Fusion network after attention
        fusion_input_dim = self.common_dim * 2  # attended vision + attended language
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.action_decoder = ActionDecoder(256, action_dim)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, vision_input, language_input):
        # Encode modalities
        vision_features = self.vision_encoder(vision_input)  # [B, vision_dim]
        language_features = self.language_encoder(language_input)  # [B, language_dim]
        
        # Project to common dimension
        vision_proj = self.vision_projection(vision_features).unsqueeze(1)  # [B, 1, common_dim]
        language_proj = self.language_projection(language_features).unsqueeze(1)  # [B, 1, common_dim]
        
        # Cross-attention: vision attends to language and vice versa
        vision_attended, _ = self.vision_language_attention(
            query=vision_proj,      # [B, 1, common_dim]
            key=language_proj,      # [B, 1, common_dim] 
            value=language_proj     # [B, 1, common_dim]
        )
        
        language_attended, _ = self.language_vision_attention(
            query=language_proj,    # [B, 1, common_dim]
            key=vision_proj,        # [B, 1, common_dim]
            value=vision_proj       # [B, 1, common_dim]
        )
        
        # Flatten and concatenate attended features
        vision_attended_flat = vision_attended.squeeze(1)      # [B, common_dim]
        language_attended_flat = language_attended.squeeze(1)  # [B, common_dim]
        
        concatenated = torch.cat([vision_attended_flat, language_attended_flat], dim=-1)
        
        # Process through fusion network
        fused_features = self.fusion_network(concatenated)
        
        # Generate outputs
        action = self.action_decoder(fused_features)
        value = self.value_head(fused_features)
        
        return action, value, fused_features

class TransformerBasedFusion(nn.Module):
    """Transformer-based VLA architecture with self-attention"""
    
    def __init__(self, vision_dim=512, language_dim=512, action_dim=19, 
                 d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.d_model = d_model
        
        # Modality-specific encoders
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.language_encoder = LanguageEncoder(hidden_dim=language_dim)
        
        # Projection to common dimension
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.language_proj = nn.Linear(language_dim, d_model)
        
        # Add trainable modality tokens
        self.vision_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.language_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Action and value heads
        self.action_head = nn.Linear(d_model, action_dim)
        self.value_head = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, vision_input, language_input):
        batch_size = vision_input.size(0)
        
        # Encode modalities
        vision_features = self.vision_encoder(vision_input)
        language_features = self.language_encoder(language_input)
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [B, 1, d_model]
        language_proj = self.language_proj(language_features).unsqueeze(1)  # [B, 1, d_model]
        
        # Add modality tokens
        vision_with_token = vision_proj + self.vision_token
        language_with_token = language_proj + self.language_token
        
        # Concatenate modalities
        concat_features = torch.cat([vision_with_token, language_with_token], dim=1)  # [B, 2, d_model]
        
        # Apply transformer
        attended_features = self.transformer(concat_features)  # [B, 2, d_model]
        
        # Use pooled representation (concatenate both modalities' outputs)
        pooled_features = attended_features.view(batch_size, -1)  # [B, 2*d_model]
        
        # Apply final normalization
        normalized_features = self.norm(pooled_features)
        
        # Generate action and value
        action = self.action_head(normalized_features)
        value = self.value_head(normalized_features[:, :self.d_model])  # Use just vision part for value
        
        return action, value, normalized_features

class FusionComparisonEvaluator:
    """Evaluator to compare different fusion strategies"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_fusion_strategies(self, models_dict, test_loader):
        """Evaluate multiple fusion strategies on test data"""
        results = {}
        
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            model.eval()
            total_loss = 0
            correct_actions = 0
            total_actions = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    vision_input = batch['vision']
                    language_input = batch['language']
                    targets = batch['actions']
                    
                    actions, values, _ = model(vision_input, language_input)
                    
                    # Calculate loss
                    loss = F.mse_loss(actions, targets)
                    total_loss += loss.item()
                    
                    # Calculate accuracy (for discrete action tasks)
                    if targets.dtype in [torch.long, torch.int]:
                        preds = torch.argmax(actions, dim=-1)
                        correct = (preds == targets).sum().item()
                        correct_actions += correct
                        total_actions += targets.numel()
            
            avg_loss = total_loss / len(test_loader)
            accuracy = correct_actions / total_actions if total_actions > 0 else 0
            
            results[name] = {
                'avg_loss': avg_loss,
                'accuracy': accuracy,
                'total_samples': len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * test_loader.batch_size
            }
        
        return results

def create_mock_data(batch_size=32, seq_len=20):
    """Create mock data for testing fusion architectures"""
    # Vision: random images (normalized to [0,1])
    vision_data = torch.rand(batch_size, 3, 84, 84)  # Example image dimensions
    
    # Language: random token IDs
    language_data = torch.randint(0, 10000, (batch_size, seq_len))  # Vocab size 10k
    
    # Actions: random continuous actions
    actions = torch.randn(batch_size, 19)  # Example action dimension
    
    return {
        'vision': vision_data,
        'language': language_data,
        'actions': actions
    }

def main():
    """Main function to compare fusion strategies"""
    print("Comparing VLA Fusion Architectures")
    print("=" * 40)
    
    # Initialize models
    models = {
        'Early Fusion': EarlyFusionVLA(),
        'Late Fusion': LateFusionVLA(),
        'Cross-Attention Fusion': CrossAttentionFusionVLA(),
        'Transformer Fusion': TransformerBasedFusion()
    }
    
    # Create mock test data
    print("Creating mock test data...")
    mock_batch = create_mock_data(batch_size=16)  # Small batch for demonstration
    
    # Test forward pass for each model
    for name, model in models.items():
        try:
            vision_in = mock_batch['vision']
            language_in = mock_batch['language']
            
            action_out, value_out, features_out = model(vision_in, language_in)
            
            print(f"{name}:")
            print(f"  Action output shape: {action_out.shape}")
            print(f"  Value output shape: {value_out.shape}")
            print(f"  Feature output shape: {features_out.shape}")
            print(f"  Action range: [{action_out.min():.3f}, {action_out.max():.3f}]")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}\n")
    
    # In a real scenario, we would have a proper dataset and evaluation
    evaluator = FusionComparisonEvaluator()
    
    print("Fusion architecture comparison completed!")

if __name__ == "__main__":
    main()
```

### Exercise 2: Vision-Language Grounding Implementation

#### Objective
Implement vision-language grounding mechanisms to connect natural language instructions to visual entities.

#### Steps
1. Create vision-language grounding network
2. Implement attention mechanisms for grounding
3. Test grounding accuracy
4. Evaluate on manipulation tasks

```python
#!/usr/bin/env python3
"""Vision-Language Grounding for Robotics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet50
import torchvision.transforms as T
from typing import List, Dict, Tuple, Optional

class VisionLanguageGrounding(nn.Module):
    """Vision-Language grounding for robotics"""
    
    def __init__(self, vision_dim=2048, language_dim=768, hidden_dim=512):
        super().__init__()
        
        # Vision encoder (using ResNet feature extractor)
        self.vision_encoder = resnet50(pretrained=True)
        # Replace final classifier layer
        self.vision_encoder.fc = nn.Identity()
        self.vision_feature_dim = vision_dim
        
        # Language encoder (BERT-based)
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.language_feature_dim = language_dim
        
        # Feature projection to common space
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross-modal attention for grounding
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Grounding score prediction
        self.grounding_predictor = nn.Linear(hidden_dim, 1)
        
        # Object detection and classification head
        self.detection_head = nn.Linear(hidden_dim, 21)  # 20 object classes + 1 background
        self.localization_head = nn.Linear(hidden_dim, 4)  # Bounding box [x, y, w, h]
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, images, text_inputs, attention_mask=None):
        """
        Forward pass for vision-language grounding
        
        Args:
            images: [B, C, H, W] input images
            text_inputs: [B, seq_len] tokenized text
            attention_mask: [B, seq_len] attention mask for padding
        """
        batch_size = images.size(0)
        
        # Process vision
        vision_features = self.vision_encoder(images)  # [B, vision_dim]
        
        # Reshape for projection: [B, 1, vision_dim] for attention compatibility
        vision_features = vision_features.unsqueeze(1)  # [B, 1, vision_dim]
        vision_proj = self.vision_proj(vision_features)  # [B, 1, hidden_dim]
        
        # Process language
        if attention_mask is None:
            attention_mask = (text_inputs != 0).float()  # Assume 0 is padding token
        
        language_outputs = self.language_encoder(
            input_ids=text_inputs,
            attention_mask=attention_mask
        )
        language_features = language_outputs.last_hidden_state  # [B, seq_len, language_dim]
        language_proj = self.language_proj(language_features)  # [B, seq_len, hidden_dim]
        
        # Cross-modal attention: vision queries language
        vision_attended, attention_weights = self.cross_attention(
            query=vision_proj,           # [B, 1, hidden_dim]
            key=language_proj,           # [B, seq_len, hidden_dim]
            value=language_proj,         # [B, seq_len, hidden_dim]
            key_padding_mask=~attention_mask.bool()  # Invert mask for PyTorch attention
        )
        
        # Similarly, language queries vision
        language_attended, lang_att_weights = self.cross_attention(
            query=language_proj,         # [B, seq_len, hidden_dim]
            key=vision_proj,             # [B, 1, hidden_dim]
            value=vision_proj            # [B, 1, hidden_dim]
        )
        
        # Combine attended features
        combined_features = vision_attended + language_attended.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        combined_features = self.dropout(combined_features)
        
        # Grounding prediction
        grounding_scores = torch.sigmoid(self.grounding_predictor(combined_features))  # [B, 1, 1]
        
        # Detection and localization (in real system, would be more complex)
        detection_logits = self.detection_head(combined_features.squeeze(1))  # [B, 21]
        localization_preds = torch.sigmoid(self.localization_head(combined_features.squeeze(1)))  # [B, 4]
        
        return {
            'grounding_scores': grounding_scores.squeeze(-1),  # [B, 1]
            'detection_logits': detection_logits,              # [B, 21]
            'localization_predictions': localization_preds,    # [B, 4]
            'vision_language_attention': attention_weights,    # [B, 1, seq_len]
            'language_vision_attention': lang_att_weights,     # [B, seq_len, 1]
            'combined_features': combined_features.squeeze(1)  # [B, hidden_dim]
        }

class SpatiallyAwareGrounding(nn.Module):
    """Spatially-aware vision-language grounding"""
    
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        
        # Spatial position encoding
        self.spatial_encoding = nn.Linear(4, feature_dim // 4)  # x, y, w, h -> spatial features
        
        # Vision-language transformer with spatial awareness
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Grounding prediction with spatial context
        self.grounding_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1)
        )
        
        # Spatial relation prediction
        self.spatial_relation_predictor = nn.Linear(feature_dim, 6)  # left, right, above, below, near, far
    
    def forward(self, vision_features, language_features, spatial_coords):
        """
        Forward pass with spatial awareness
        
        Args:
            vision_features: [B, num_regions, feat_dim] visual features per region
            language_features: [B, seq_len, feat_dim] language features
            spatial_coords: [B, num_regions, 4] spatial coordinates [x, y, w, h]
        """
        batch_size, num_regions, feat_dim = vision_features.shape
        
        # Add spatial encoding to vision features
        spatial_encodings = self.spatial_encoding(spatial_coords)  # [B, num_regions, spatial_dim]
        
        # Combine spatial and visual features
        spatially_aware_features = vision_features + spatial_encodings  # [B, num_regions, feat_dim]
        
        # Create multimodal sequence: [language, spatially_aware_vision]
        # Repeat language features for each vision region to enable cross-modal interaction
        repeated_lang = language_features.unsqueeze(1).expand(-1, num_regions, -1, -1).reshape(
            batch_size * num_regions, -1, feat_dim
        )
        
        repeated_vision = spatially_aware_features.unsqueeze(2).expand(
            -1, -1, language_features.size(1), -1
        ).reshape(batch_size * num_regions, language_features.size(1), feat_dim)
        
        # Combine modalities
        combined_features = torch.cat([repeated_lang, repeated_vision], dim=1)
        
        # Apply spatial transformer
        attended_features = self.spatial_transformer(combined_features)
        
        # Extract vision-grounded language representations
        lang_part = attended_features[:, :language_features.size(1), :]  # [B*num_regions, lang_seq, feat_dim]
        vision_part = attended_features[:, language_features.size(1):, :]  # [B*num_regions, vis_seq, feat_dim]
        
        # Pool vision part to get region representations
        region_representations = vision_part.mean(dim=1)  # [B*num_regions, feat_dim]
        region_representations = region_representations.reshape(batch_size, num_regions, -1)  # [B, num_regions, feat_dim]
        
        # Grounding prediction for each region
        grounding_scores = torch.sigmoid(self.grounding_predictor(region_representations))  # [B, num_regions, 1]
        
        # Spatial relation prediction
        spatial_relations = self.spatial_relation_predictor(region_representations)  # [B, num_regions, 6]
        
        return {
            'region_grounding_scores': grounding_scores.squeeze(-1),  # [B, num_regions]
            'spatial_relations': spatial_relations,                   # [B, num_regions, 6]
            'region_representations': region_representations,         # [B, num_regions, feat_dim]
            'combined_attentions': attended_features                  # [B, seq_len, feat_dim]
        }

class GroundingEvaluator:
    """Evaluator for vision-language grounding"""
    
    def __init__(self):
        self.metrics = {
            'grounding_accuracy': [],
            'spatial_precision': [],
            'object_localization_iou': [],
            'language_alignment_score': []
        }
    
    def evaluate_grounding(self, model, test_dataset):
        """Evaluate grounding performance"""
        model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_dataset:
                images = batch['images']
                texts = batch['texts']
                text_masks = batch['text_masks']
                ground_truth_boxes = batch['ground_truth_boxes']
                target_objects = batch['target_objects']
                
                # Get model predictions
                outputs = model(images, texts, text_masks)
                
                # Calculate grounding accuracy
                batch_accuracy = self.calculate_grounding_accuracy(
                    outputs['grounding_scores'],
                    target_objects,
                    ground_truth_boxes
                )
                
                # Calculate localization IoU
                batch_iou = self.calculate_localization_iou(
                    outputs['localization_predictions'],
                    ground_truth_boxes
                )
                
                # Update metrics
                self.metrics['grounding_accuracy'].extend(batch_accuracy)
                self.metrics['object_localization_iou'].extend(batch_iou)
                
                total_samples += len(batch_accuracy)
        
        # Calculate overall metrics
        overall_metrics = {
            'avg_grounding_accuracy': np.mean(self.metrics['grounding_accuracy']),
            'avg_localization_iou': np.mean(self.metrics['object_localization_iou']),
            'total_evaluated': total_samples
        }
        
        return overall_metrics
    
    def calculate_grounding_accuracy(self, pred_scores, target_objects, gt_boxes):
        """Calculate grounding accuracy"""
        accuracies = []
        
        for i in range(len(pred_scores)):
            # Find the region with highest grounding score
            if pred_scores[i].numel() > 0:
                best_region_idx = torch.argmax(pred_scores[i])
                # In a real implementation, you would compare to ground truth
                # For this example, we'll use a proxy metric
                accuracy = float(pred_scores[i].max() > 0.5)  # Simple threshold-based accuracy
                accuracies.append(accuracy)
        
        return accuracies
    
    def calculate_localization_iou(self, pred_boxes, gt_boxes):
        """Calculate Intersection over Union for localization"""
        ious = []
        
        for pred, gt in zip(pred_boxes, gt_boxes):
            # Convert normalized coords to absolute if needed
            # pred and gt are [x, y, w, h] in normalized coordinates (0-1)
            
            # Calculate IoU
            # Convert to [x1, y1, x2, y2] format
            pred_x1 = pred[0] - pred[2] / 2
            pred_y1 = pred[1] - pred[3] / 2
            pred_x2 = pred[0] + pred[2] / 2
            pred_y2 = pred[1] + pred[3] / 2
            
            gt_x1 = gt[0] - gt[2] / 2
            gt_y1 = gt[1] - gt[3] / 2
            gt_x2 = gt[0] + gt[2] / 2
            gt_y2 = gt[1] + gt[3] / 2
            
            # Calculate intersection
            inter_x1 = max(pred_x1, gt_x1)
            inter_y1 = max(pred_y1, gt_y1)
            inter_x2 = min(pred_x2, gt_x2)
            inter_y2 = min(pred_y2, gt_y2)
            
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                
                # Calculate union
                pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                union_area = pred_area + gt_area - inter_area
                
                iou = inter_area / union_area if union_area > 0 else 0
            else:
                iou = 0
            
            ious.append(iou)
        
        return ious

def main():
    """Main function to test vision-language grounding"""
    print("Testing Vision-Language Grounding Implementation")
    print("=" * 50)
    
    # Initialize grounding model
    grounding_model = VisionLanguageGrounding()
    
    # Create example inputs
    images = torch.randn(4, 3, 224, 224)  # Example batch of images
    text_inputs = torch.randint(100, 2000, (4, 16))  # Random token IDs
    attention_masks = torch.ones(4, 16)  # All tokens valid
    
    # Forward pass
    outputs = grounding_model(images, text_inputs, attention_masks)
    
    print(f"Grounding scores shape: {outputs['grounding_scores'].shape}")
    print(f"Detection logits shape: {outputs['detection_logits'].shape}")
    print(f"Localization predictions shape: {outputs['localization_predictions'].shape}")
    print(f"Grounding scores range: [{outputs['grounding_scores'].min():.3f}, {outputs['grounding_scores'].max():.3f}]")
    
    # Test spatially-aware grounding
    print("\nTesting Spatially-Aware Grounding:")
    spatial_model = SpatiallyAwareGrounding()
    
    # Example inputs for spatial model
    vision_feats = torch.randn(4, 10, 512)  # 4 images, 10 regions, 512-dim features
    language_feats = torch.randn(4, 20, 512)  # 4 samples, 20 tokens, 512-dim features
    spatial_coords = torch.rand(4, 10, 4)  # 4 samples, 10 regions, 4 spatial coords [x, y, w, h]
    
    spatial_outputs = spatial_model(vision_feats, language_feats, spatial_coords)
    
    print(f"Spatial grounding scores shape: {spatial_outputs['region_grounding_scores'].shape}")
    print(f"Spatial relations shape: {spatial_outputs['spatial_relations'].shape}")
    print(f"Region representations shape: {spatial_outputs['region_representations'].shape}")
    
    print("\nVision-Language Grounding implementation test completed successfully!")

if __name__ == "__main__":
    main()
```

### Exercise 3: End-to-End VLA System Integration

#### Objective
Integrate all components into a complete VLA system and test on humanoid robot simulation.

#### Steps
1. Integrate vision, language, and action networks
2. Implement system for humanoid control
3. Test in simulation environment
4. Validate system performance

```python
#!/usr/bin/env python3
"""End-to-End Vision-Language-Action System for Humanoid Robotics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
import time
import random
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces

class HumanoidVLASystem(nn.Module):
    """Complete VLA system for humanoid robotics"""
    
    def __init__(self, vision_dim=512, language_dim=768, action_dim=22, hidden_dim=512):
        super().__init__()
        
        # Vision processing module
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        
        # Language processing module
        self.language_encoder = LanguageEncoder(vocab_size=30522, embedding_dim=256, hidden_dim=language_dim)
        
        # Multimodal fusion module
        self.fusion_module = CrossAttentionFusionModule(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim
        )
        
        # Action generation module
        self.action_generator = ActionGenerator(
            input_dim=hidden_dim,
            action_dim=action_dim
        )
        
        # Value estimation for RL
        self.value_estimator = ValueEstimator(hidden_dim)
        
        # Language grounding module
        self.grounding_module = VisionLanguageGrounding(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim
        )
        
        # Task planner (for complex instructions)
        self.task_planner = TaskPlanner(hidden_dim)
        
        # For humanoid control stability
        self.register_buffer('action_scale', torch.ones(action_dim))
        self.register_buffer('action_bias', torch.zeros(action_dim))
    
    def forward(self, images, language_tokens, attention_mask=None):
        """
        Forward pass for complete VLA system
        
        Args:
            images: [B, C, H, W] visual input
            language_tokens: [B, seq_len] tokenized language input
            attention_mask: [B, seq_len] attention mask for language
        """
        batch_size = images.size(0)
        
        # Extract vision features
        vision_features = self.vision_encoder(images)  # [B, vision_dim]
        
        # Extract language features
        language_features = self.language_encoder(language_tokens, attention_mask)  # [B, language_dim]
        
        # Multimodal fusion
        fused_features = self.fusion_module(vision_features, language_features)  # [B, hidden_dim]
        
        # Generate action
        raw_action = self.action_generator(fused_features)  # [B, action_dim]
        
        # Apply action scaling and biasing for humanoid control
        scaled_action = torch.tanh(raw_action) * self.action_scale + self.action_bias
        
        # Estimate value for RL training
        value = self.value_estimator(fused_features)  # [B, 1]
        
        # Language grounding (for complex instructions)
        grounding_output = self.grounding_module(
            images, language_tokens, attention_mask
        )
        
        # Task planning for complex instructions
        task_plan = self.task_planner(fused_features, grounding_output['grounding_scores'])
        
        return {
            'action': scaled_action,
            'raw_action': raw_action,
            'value': value,
            'fused_features': fused_features,
            'grounding_output': grounding_output,
            'task_plan': task_plan
        }
    
    def get_action(self, images, language_tokens, attention_mask=None, deterministic=False):
        """Get action for control (with optional sampling for exploration)"""
        output = self(images, language_tokens, attention_mask)
        
        if deterministic:
            # Return deterministic action (for deployment)
            return output['action'], None, output['value']
        else:
            # Add small amount of exploration noise for training
            action = output['action']
            noise = torch.randn_like(action) * 0.1  # 10% exploration noise
            noisy_action = torch.clamp(action + noise, -1.0, 1.0)
            
            # Calculate log probability (simplified for this example)
            log_prob = -0.5 * ((noisy_action - action) ** 2).sum(dim=1, keepdim=True)
            
            return noisy_action, log_prob, output['value']

class VisionLanguageActionEnvironment(gym.Env):
    """Gym environment for VLA system training"""
    
    def __init__(self, vla_system):
        super().__init__()
        self.vla_system = vla_system
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(22,), dtype=np.float32  # Humanoid joint actions
        )
        
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float32
            ),
            'language_tokens': spaces.Box(
                low=0, high=30522, shape=(50,), dtype=np.int32  # Max sequence length
            ),
            'language_attention_mask': spaces.Box(
                low=0, high=1, shape=(50,), dtype=np.float32
            )
        })
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 500
        self.episode_reward = 0.0
        
        # Simulate humanoid robot state
        self.robot_state = np.zeros(22)  # Joint positions
        self.target_position = np.random.uniform(-2, 2, size=(3,))  # Random target
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset robot state
        self.robot_state = np.random.randn(22) * 0.1
        
        # Generate new target
        self.target_position = np.random.uniform(-2, 2, size=(3,))
        
        # Generate random observation (in practice, would come from simulation)
        obs = {
            'vision': np.random.randn(3, 224, 224).astype(np.float32),
            'language_tokens': np.random.randint(100, 2000, size=(50,)).astype(np.int32),
            'language_attention_mask': np.ones(50, dtype=np.float32)
        }
        
        return obs, {}
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Simulate robot dynamics (simplified)
        self.robot_state += action * 0.01  # Small step per action
        
        # Calculate reward (example: distance to target)
        current_position = self.robot_state[:3]  # Use first 3 joints as position proxy
        distance_to_target = np.linalg.norm(current_position - self.target_position)
        
        # Reward is negative distance (encourage getting closer)
        step_reward = -distance_to_target
        
        # Add small bonus for staying within joint limits
        joint_limit_penalty = -np.sum(np.abs(self.robot_state) > 1.5) * 0.1
        step_reward += joint_limit_penalty
        
        self.episode_reward += step_reward
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # Time limit above
        
        # Generate new observation (in practice, would come from simulation)
        new_obs = {
            'vision': np.random.randn(3, 224, 224).astype(np.float32),
            'language_tokens': np.random.randint(100, 2000, size=(50,)).astype(np.int32),
            'language_attention_mask': np.ones(50, dtype=np.float32)
        }
        
        info = {
            'step': self.current_step,
            'current_position': current_position,
            'target_position': self.target_position,
            'distance_to_target': distance_to_target,
            'episode_reward': self.episode_reward
        }
        
        return new_obs, step_reward, terminated, truncated, info
    
    def set_language_instruction(self, instruction):
        """Set language instruction for the current episode"""
        # In a real implementation, this would tokenize the instruction
        # For this example, we'll just store it
        self.current_instruction = instruction

class VLAController:
    """Controller interface for VLA system"""
    
    def __init__(self, vla_model):
        self.model = vla_model
        self.model.eval()
        
        # Tokenizer for language processing
        try:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        except ImportError:
            print("Warning: Transformers not available, using mock tokenizer")
            self.tokenizer = None
    
    def process_command(self, visual_observation, language_command):
        """Process visual and language inputs to generate action"""
        # Prepare visual input
        if isinstance(visual_observation, np.ndarray):
            vision_tensor = torch.FloatTensor(visual_observation).unsqueeze(0)
        else:
            vision_tensor = visual_observation.unsqueeze(0)
        
        # Prepare language input
        if self.tokenizer:
            tokenized = self.tokenizer(
                language_command,
                return_tensors='pt',
                padding='max_length',
                max_length=50,
                truncation=True
            )
            language_input = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            # Mock tokenization (in practice you'd have a proper tokenizer)
            tokens = language_command.split()[:50]  # Limit length
            language_input = torch.zeros(1, 50, dtype=torch.long)
            attention_mask = torch.zeros(1, 50)
            
            for i, token in enumerate(tokens):
                # Convert to mock token ID (in practice, use real tokenizer)
                language_input[0, i] = hash(token) % 10000  # Mock token ID
                attention_mask[0, i] = 1
        
        # Get action from VLA system
        with torch.no_grad():
            output = self.model(vision_tensor, language_input, attention_mask)
            action = output['action']
        
        return action.squeeze(0), output  # Return action and full output for debugging

def main():
    """Main function to test complete VLA system"""
    print("Initializing Complete Vision-Language-Action System")
    print("=" * 55)
    
    # Initialize VLA system
    vla_system = HumanoidVLASystem(vision_dim=512, language_dim=768, action_dim=22)
    
    # Create environment
    env = VisionLanguageActionEnvironment(vla_system)
    
    # Create controller
    controller = VLAController(vla_system)
    
    print("Testing VLA system in environment...")
    
    # Run a few steps to verify system works
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(10):
        # Example language command
        command = f"Move toward target position"
        
        # Process with VLA system
        action, output = controller.process_command(obs['vision'], command)
        
        # Execute in environment
        new_obs, reward, terminated, truncated, info = env.step(action.numpy())
        
        total_reward += reward
        
        print(f"Step {step}: Action={action[:3].numpy()}, Reward={reward:.3f}, "
              f"Distance to target={info['distance_to_target']:.3f}")
        
        if terminated or truncated:
            break
        
        obs = new_obs
    
    print(f"\nTest completed. Total reward: {total_reward:.3f}")
    print("Complete VLA system is functioning properly!")

if __name__ == "__main__":
    main()
```

### Exercise 4: Performance Optimization and Evaluation

#### Objective
Optimize the VLA system for real-time performance and evaluate its effectiveness.

#### Steps
1. Implement performance optimization techniques
2. Create evaluation metrics
3. Test system under different conditions
4. Analyze performance bottlenecks

```python
#!/usr/bin/env python3
"""Performance optimization and evaluation for VLA systems"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.benchmark import Timer
from typing import Dict, List, Tuple
import psutil
import GPUtil
import threading
import queue
from collections import deque

class VLAPerformanceOptimizer:
    """Performance optimization for VLA systems"""
    
    def __init__(self, vla_model):
        self.model = vla_model
        self.original_device = next(vla_model.parameters()).device
        
        # Performance tracking
        self.inference_times = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.gpu_utilization = deque(maxlen=1000)
        
        # Optimization status
        self.optimized = False
        self.optimization_config = {}
    
    def optimize_inference(self):
        """Apply inference optimizations"""
        print("Applying inference optimizations...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # Optimize model with torchscript if possible
        try:
            self.model = torch.jit.optimize_for_inference(
                torch.jit.script(self.model)
            )
            print("Applied JIT optimization")
        except Exception as e:
            print(f"JIT optimization failed: {e}")
        
        # Apply model-specific optimizations
        self._optimize_model_components()
        
        # Set appropriate floating point precision
        if self.use_mixed_precision:
            self.use_half_precision = True
            self.model.half()
        
        self.optimized = True
        print("Inference optimizations applied")
    
    def _optimize_model_components(self):
        """Optimize specific components of the VLA model"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm with FrozenBatchNorm for inference
                new_module = torch.nn.BatchNorm2d(
                    module.num_features,
                    module.eps,
                    module.momentum,
                    module.affine
                )
                new_module.load_state_dict(module.state_dict())
                new_module.eval()
                
                # Replace in the model (this is conceptual, actual replacement would need more care)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
    def optimize_with_tensorrt(self, input_specs):
        """Optimize model with TensorRT (if available)"""
        try:
            import torch_tensorrt
            
            # Compile model with TensorRT
            trt_model = torch_tensorrt.compile(
                self.model,
                inputs=input_specs,
                enabled_precisions={torch.float, torch.half}
            )
            
            self.model = trt_model
            self.optimization_config['tensorrt'] = True
            print("Applied TensorRT optimization")
            return True
        except ImportError:
            print("TensorRT not available, skipping optimization")
            return False
        except Exception as e:
            print(f"TensorRT optimization failed: {e}")
            return False
    
    def quantize_model(self):
        """Apply quantization to reduce model size and improve speed"""
        print("Applying model quantization...")
        
        # Use PyTorch's dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=torch.qint8
        )
        
        self.model = quantized_model
        self.optimization_config['quantized'] = True
        print("Model quantization applied")
        
        return quantized_model
    
    def benchmark_model(self, num_runs=100):
        """Benchmark model performance"""
        print(f"Benchmarking model performance over {num_runs} runs...")
        
        # Prepare dummy inputs
        vision_input = torch.randn(1, 3, 224, 224, device=next(self.model.parameters()).device)
        language_input = torch.randint(0, 1000, (1, 20), device=next(self.model.parameters()).device)
        attention_mask = torch.ones(1, 20, device=next(self.model.parameters()).device)
        
        # Warm up
        for _ in range(5):
            _ = self.model(vision_input, language_input, attention_mask)
        
        # Benchmark
        start_time = time.time()
        for i in range(num_runs):
            with torch.no_grad():
                output = self.model(vision_input, language_input, attention_mask)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        print(f"Average inference time: {avg_time:.4f}s ({fps:.2f} FPS)")
        print(f"Total time for {num_runs} runs: {total_time:.4f}s")
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_time': total_time,
            'runs': num_runs
        }
    
    def profile_memory_usage(self):
        """Profile memory usage of the model"""
        print("Profiling memory usage...")
        
        # Get initial memory
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated()
            initial_cached = torch.cuda.memory_reserved()
        
        initial_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run inference and measure memory
        vision_input = torch.randn(1, 3, 224, 224, requires_grad=False)
        language_input = torch.randint(0, 1000, (1, 20), requires_grad=False)
        attention_mask = torch.ones(1, 20, requires_grad=False)
        
        # Measure memory during inference
        for i in range(10):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                output = self.model(vision_input, language_input, attention_mask)
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_gpu_memory = torch.cuda.max_memory_allocated()
                cached_gpu_memory = torch.cuda.memory_reserved()
                self.gpu_memory_usage.append({
                    'peak_mb': peak_gpu_memory / 1024 / 1024,
                    'cached_mb': cached_gpu_memory / 1024 / 1024
                })
            
            cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.cpu_memory_usage.append(cpu_memory)
        
        # Calculate averages
        avg_gpu_peak = np.mean([m['peak_mb'] for m in self.gpu_memory_usage]) if self.gpu_memory_usage else 0
        avg_cpu_usage = np.mean(self.cpu_memory_usage) if self.cpu_memory_usage else 0
        
        print(f"GPU Memory - Peak: {avg_gpu_peak:.2f}MB, Cached: {np.mean([m['cached_mb'] for m in self.gpu_memory_usage]):.2f}MB")
        print(f"CPU Memory Usage: {avg_cpu_usage:.2f}MB")
        
        return {
            'gpu_peak_avg_mb': avg_gpu_peak,
            'gpu_cached_avg_mb': np.mean([m['cached_mb'] for m in self.gpu_memory_usage]) if self.gpu_memory_usage else 0,
            'cpu_avg_mb': avg_cpu_usage
        }

class VLAEvaluator:
    """Evaluation framework for VLA systems"""
    
    def __init__(self, vla_model, test_dataset=None):
        self.model = vla_model
        self.test_dataset = test_dataset
        
        # Evaluation metrics
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'grounding_accuracy': [],
            'task_completion_rate': [],
            'response_time': [],
            'safety_violations': []
        }
        
        # Performance trackers
        self.performance_monitor = VLAPerformanceOptimizer(vla_model)
    
    def evaluate_on_dataset(self, dataset_loader):
        """Evaluate model on a dataset"""
        print("Starting dataset evaluation...")
        self.model.eval()
        
        correct_actions = 0
        total_actions = 0
        grounding_correct = 0
        grounding_total = 0
        task_success = 0
        total_tasks = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataset_loader):
                vision_input = batch['vision']
                language_input = batch['language']
                language_attention = batch.get('attention_mask')
                targets = batch['actions']
                ground_truth_grounding = batch.get('grounding_targets')
                task_success_targets = batch.get('task_success')
                
                # Model inference
                start_time = time.time()
                outputs = self.model(vision_input, language_input, language_attention)
                inference_time = time.time() - start_time
                
                # Evaluate action accuracy
                predicted_actions = outputs['action']
                batch_accuracy = self.calculate_action_accuracy(predicted_actions, targets)
                
                correct_actions += batch_accuracy['correct']
                total_actions += batch_accuracy['total']
                
                # Evaluate grounding if available
                if ground_truth_grounding is not None:
                    grounding_acc = self.evaluate_grounding(
                        outputs['grounding_output'], 
                        ground_truth_grounding
                    )
                    grounding_correct += grounding_acc['correct']
                    grounding_total += grounding_acc['total']
                
                # Evaluate task completion if available
                if task_success_targets is not None:
                    task_success += self.evaluate_task_completion(
                        outputs, task_success_targets
                    )
                    total_tasks += len(task_success_targets)
                
                # Record response time
                self.metrics['response_time'].append(inference_time)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx} batches")
        
        # Calculate overall metrics
        overall_metrics = {
            'action_accuracy': correct_actions / total_actions if total_actions > 0 else 0,
            'grounding_accuracy': grounding_correct / grounding_total if grounding_total > 0 else 0,
            'task_completion_rate': task_success / total_tasks if total_tasks > 0 else 0,
            'avg_response_time': np.mean(self.metrics['response_time']),
            'total_evaluated': total_actions
        }
        
        return overall_metrics
    
    def calculate_action_accuracy(self, predictions, targets, threshold=0.1):
        """Calculate action accuracy for continuous control"""
        # For continuous actions, calculate how many actions are within threshold
        with torch.no_grad():
            diff = torch.abs(predictions - targets)
            within_threshold = diff < threshold
            correct_actions = within_threshold.sum().item()
            total_elements = predictions.numel()
            
            return {
                'correct': correct_actions,
                'total': total_elements,
                'accuracy_per_dim': (within_threshold.sum(dim=0) / len(predictions)).tolist()
            }
    
    def evaluate_grounding(self, grounding_output, ground_truth):
        """Evaluate vision-language grounding accuracy"""
        pred_scores = grounding_output['region_grounding_scores']  # [B, num_regions]
        gt_labels = ground_truth  # [B] with ground truth region indices
        
        with torch.no_grad():
            pred_regions = torch.argmax(pred_scores, dim=1)  # [B]
            correct = (pred_regions == gt_labels).sum().item()
            total = len(gt_labels)
            
            return {
                'correct': correct,
                'total': total
            }
    
    def evaluate_task_completion(self, model_output, target_success):
        """Evaluate if tasks were completed successfully"""
        # In a real system, this would be more complex
        # For this example, we'll use a simple proxy
        success_predictions = model_output['task_plan']['success_probability'] > 0.8
        actual_success = torch.tensor(target_success, dtype=torch.bool)
        
        correct_completions = (success_predictions == actual_success).sum().item()
        return correct_completions
    
    def generate_evaluation_report(self, metrics):
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {
                'action_accuracy': metrics.get('action_accuracy', 0),
                'grounding_accuracy': metrics.get('grounding_accuracy', 0),
                'task_completion_rate': metrics.get('task_completion_rate', 0),
                'avg_response_time_ms': metrics.get('avg_response_time', 0) * 1000,
                'total_evaluated': metrics.get('total_evaluated', 0)
            },
            'detailed_metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if metrics.get('action_accuracy', 0) < 0.7:
            recommendations.append(
                "Action accuracy is low (<70%), consider retraining with more diverse data"
            )
        
        if metrics.get('grounding_accuracy', 0) < 0.6:
            recommendations.append(
                "Vision-language grounding accuracy is low, consider improving attention mechanisms"
            )
        
        if metrics.get('avg_response_time', float('inf')) > 0.1:  # >100ms
            recommendations.append(
                "Response time is high, consider model optimization or hardware upgrade"
            )
        
        if not recommendations:
            recommendations.append("Performance looks good across all metrics")
        
        return recommendations

def main():
    """Main function to run optimization and evaluation"""
    print("Starting VLA System Optimization and Evaluation")
    print("=" * 50)
    
    # Initialize a simple VLA model for testing
    vla_model = HumanoidVLASystem(vision_dim=512, language_dim=768, action_dim=22)
    
    # Initialize optimizer
    optimizer = VLAPerformanceOptimizer(vla_model)
    
    # Apply optimizations
    optimizer.optimize_inference()
    
    # Benchmark performance
    benchmark_results = optimizer.benchmark_model(num_runs=50)
    
    print("\nPerformance Benchmark Results:")
    for key, value in benchmark_results.items():
        print(f"  {key}: {value}")
    
    # Memory profiling
    memory_profile = optimizer.profile_memory_usage()
    
    print("\nMemory Profile:")
    for key, value in memory_profile.items():
        print(f"  {key}: {value}")
    
    # Initialize evaluator
    evaluator = VLAEvaluator(vla_model)
    
    # For this demo, we'll create mock evaluation data
    # In a real implementation, you would have an actual dataset
    print("\nPerforming mock evaluation...")
    
    # Simulate evaluation process
    mock_metrics = {
        'action_accuracy': 0.85,
        'grounding_accuracy': 0.78,
        'task_completion_rate': 0.82,
        'avg_response_time': 0.045,  # 45ms
        'total_evaluated': 1000
    }
    
    report = evaluator.generate_evaluation_report(mock_metrics)
    
    print("\nEvaluation Report Summary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print("\nVLA system optimization and evaluation completed!")

if __name__ == "__main__":
    main()
```

### Assessment Criteria

Your implementation will be evaluated based on:

1. **Technical Implementation** (30%)
   - Correct implementation of multimodal fusion techniques
   - Proper integration of vision-language grounding
   - Efficient system architecture
   - Performance optimization techniques

2. **System Integration** (25%)
   - Seamless integration between components
   - Proper data flow between modalities
   - Effective error handling
   - Real-time performance considerations

3. **Domain Randomization** (20%)
   - Effective implementation of randomization techniques
   - Diversity in generated training data
   - Sim-to-real transfer potential
   - Coverage of parameter spaces

4. **Validation & Evaluation** (15%)
   - Appropriate evaluation metrics
   - Proper validation of implementations
   - Performance analysis
   - Quality of results

5. **Documentation & Presentation** (10%)
   - Clear code documentation
   - Understanding of design choices
   - Quality of implementation notes
   - Professional presentation

### Troubleshooting Tips

1. **Memory Issues**: Use model quantization and mixed precision to reduce memory consumption
2. **Performance Issues**: Profile code to identify bottlenecks, consider model pruning
3. **Grounding Issues**: Verify attention mechanisms and spatial encodings are properly implemented
4. **Training Instability**: Use gradient clipping and appropriate learning rates
5. **Integration Issues**: Ensure proper tensor format conversions between modalities

### Extensions for Advanced Students

- Implement neural architecture search for optimal fusion architectures
- Create adaptive domain randomization based on learning progress
- Develop sim-to-real transfer techniques with minimal real-world data
- Add uncertainty quantification to VLA predictions
- Implement continual learning capabilities for VLA systems

This practical exercise provides comprehensive experience with implementing Vision-Language-Action systems for humanoid robotics, covering the complete pipeline from low-level architecture to system integration and evaluation.