---
id: module-4-chapter-2-practical-exercises
title: 'Module 4 — Vision-Language-Action Systems | Chapter 2 — Practical Exercises'
sidebar_label: 'Chapter 2 — Practical Exercises'
sidebar_position: 5
---

# Chapter 2 — Practical Exercises

## Advanced Transformer Architectures for Vision-Language-Action Systems

### Exercise 1: Implementing Cross-Modal Attention

#### Objective
Implement and validate cross-modal attention mechanisms for vision-language integration in humanoid robotics.

#### Background
Cross-modal attention is fundamental to VLA systems, allowing vision and language modalities to influence each other. This exercise implements various attention mechanisms and validates their effectiveness.

#### Steps
1. Implement standard cross-attention
2. Implement vision-language attention
3. Test on robotic perception tasks
4. Validate attention visualization

```python
#!/usr/bin/env python3
"""Cross-modal attention implementation for VLA systems"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import math

class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision-language integration"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(CrossModalAttention, self).__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention
        
        Args:
            query: Query tensor from one modality [batch_size, seq_len_q, d_model]
            key: Key tensor from another modality [batch_size, seq_len_k, d_model]
            value: Value tensor from another modality [batch_size, seq_len_k, d_model]
            mask: Optional attention mask [batch_size, 1, 1, seq_len_k]
        
        Returns:
            Output tensor [batch_size, seq_len_q, d_model] and attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        residual = query  # Save for residual connection
        
        batch_size, seq_len_q, _ = query.shape
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_h, seq_q, d_k]
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # [B, n_h, seq_k, d_k]
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_h, seq_k, d_k]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_h, seq_q, seq_k]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, n_h, seq_q, seq_k]
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [B, n_h, seq_q, d_k]
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )  # [B, seq_q, d_model]
        
        # Final linear layer
        output = self.W_o(output)
        output = self.dropout_layer(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class VisionLanguageAttention(nn.Module):
    """Specialized attention for vision-language integration in robotics"""
    
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int, n_heads: int = 8):
        super(VisionLanguageAttention, self).__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Projection layers to common space
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross attention modules
        self.v2l_attention = CrossModalAttention(hidden_dim, n_heads)  # Vision to Language
        self.l2v_attention = CrossModalAttention(hidden_dim, n_heads)  # Language to Vision
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projections
        self.vision_output = nn.Linear(hidden_dim, vision_dim)
        self.language_output = nn.Linear(hidden_dim, language_dim)
    
    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor, 
                language_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass for vision-language attention
        
        Args:
            vision_features: Vision feature tensor [batch_size, num_patches, vision_dim]
            language_features: Language feature tensor [batch_size, seq_len, language_dim]
            language_mask: Mask for language tokens [batch_size, 1, 1, seq_len]
        
        Returns:
            Dictionary with attended features and attention weights
        """
        batch_size = vision_features.size(0)
        
        # Project features to common dimension
        vision_common = self.vision_proj(vision_features)    # [B, num_patches, hidden_dim]
        language_common = self.language_proj(language_features)  # [B, seq_len, hidden_dim]
        
        # Vision-to-Language attention: make language attend to relevant visual features
        language_attended, v2l_weights = self.v2l_attention(
            query=language_common,  # Language as query
            key=vision_common,      # Vision as key
            value=vision_common,    # Vision as value
            mask=None  # Vision usually doesn't need masking
        )
        
        # Language-to-Vision attention: make vision attend to relevant language concepts
        vision_attended, l2v_weights = self.l2v_attention(
            query=vision_common,    # Vision as query
            key=language_common,    # Language as key
            value=language_common,  # Language as value
            mask=language_mask      # Mask for language tokens
        )
        
        # Multi-modal fusion
        # Concatenate attended features
        fused_features = torch.cat([
            vision_attended,
            language_attended
        ], dim=-1)
        
        # Apply fusion network
        fused_output = self.fusion(fused_features)
        
        # Separate outputs back to original dimensions
        vision_output = self.vision_output(vision_attended)
        language_output = self.language_output(language_attended)
        
        return {
            'vision_output': vision_output,
            'language_output': language_output,
            'fused_features': fused_output,
            'vision_language_attention': l2v_weights,  # Vision attending to language
            'language_vision_attention': v2l_weights,  # Language attending to vision
            'vision_features': vision_features,
            'language_features': language_features
        }

class MultimodalTransformerBlock(nn.Module):
    """Complete multimodal transformer block with vision-language fusion"""
    
    def __init__(self, vision_dim: int, language_dim: int, hidden_dim: int, 
                 n_heads: int, ff_multiplier: int = 4, dropout: float = 0.1):
        super(MultimodalTransformerBlock, self).__init__()
        
        # Cross-modal attention
        self.cross_attention = VisionLanguageAttention(
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        )
        
        # Feed-forward networks for each modality
        self.vision_ffn = nn.Sequential(
            nn.Linear(vision_dim, vision_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim * ff_multiplier, vision_dim),
            nn.Dropout(dropout)
        )
        
        self.language_ffn = nn.Sequential(
            nn.Linear(language_dim, language_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_dim * ff_multiplier, language_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.vision_norm1 = nn.LayerNorm(vision_dim)
        self.language_norm1 = nn.LayerNorm(language_dim)
        self.vision_norm2 = nn.LayerNorm(vision_dim)
        self.language_norm2 = nn.LayerNorm(language_dim)
    
    def forward(self, vision_input: torch.Tensor, language_input: torch.Tensor,
                language_mask: Optional[torch.Tensor] = None) -> dict:
        """Forward pass with residual connections"""
        # Cross-modal attention
        cross_modal_out = self.cross_attention(
            vision_features=vision_input,
            language_features=language_input,
            language_mask=language_mask
        )
        
        # Residual connection + normalization for vision
        vision_residual = cross_modal_out['vision_output']
        vision_norm = self.vision_norm1(vision_input + vision_residual)
        
        # Vision FFN
        vision_ffn_out = self.vision_ffn(vision_norm)
        vision_output = self.vision_norm2(vision_norm + vision_ffn_out)
        
        # Residual connection + normalization for language
        language_residual = cross_modal_out['language_output']
        language_norm = self.language_norm1(language_input + language_residual)
        
        # Language FFN
        language_ffn_out = self.language_ffn(language_norm)
        language_output = self.language_norm2(language_norm + language_ffn_out)
        
        # Preserve attention weights for analysis
        cross_modal_out.update({
            'vision_output': vision_output,
            'language_output': language_output
        })
        
        return cross_modal_out

# Test function
def test_cross_modal_attention():
    """Test the cross-modal attention implementation"""
    print("Testing Cross-Modal Attention Implementation")
    print("=" * 50)
    
    # Define dimensions
    batch_size = 4
    num_patches = 196  # 14x14 patches from 224x224 image with 16x16 patches
    vision_dim = 768    # Typical for ViT
    seq_len = 32        # Language sequence length
    language_dim = 512  # Typical for BERT
    hidden_dim = 512    # Hidden dimension for attention
    n_heads = 8
    
    # Create random inputs
    vision_input = torch.randn(batch_size, num_patches, vision_dim)
    language_input = torch.randn(batch_size, seq_len, language_dim)
    
    # Create attention mask (for language - pad tokens are masked out)
    language_mask = torch.ones(batch_size, 1, 1, seq_len)  # All tokens visible initially
    # Simulate some padding by masking out the last 10 tokens
    language_mask[:, :, :, -10:] = 0
    
    print(f"Input shapes:")
    print(f"  Vision: {vision_input.shape}")
    print(f"  Language: {language_input.shape}")
    print(f"  Mask: {language_mask.shape}")
    
    # Initialize the attention module
    vl_attention = VisionLanguageAttention(
        vision_dim=vision_dim,
        language_dim=language_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads
    )
    
    # Forward pass
    output = vl_attention(
        vision_features=vision_input,
        language_features=language_input,
        language_mask=language_mask
    )
    
    print(f"\nOutput shapes:")
    print(f"  Vision output: {output['vision_output'].shape}")
    print(f"  Language output: {output['language_output'].shape}")
    print(f"  Fused features: {output['fused_features'].shape}")
    print(f"  Attention weights: {output['language_vision_attention'].shape}")
    
    # Test multimodal transformer block
    print(f"\nTesting Multimodal Transformer Block...")
    transformer_block = MultimodalTransformerBlock(
        vision_dim=vision_dim,
        language_dim=language_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads
    )
    
    block_output = transformer_block(
        vision_input=vision_input,
        language_input=language_input,
        language_mask=language_mask
    )
    
    print(f"Transformer block output shapes:")
    print(f"  Vision: {block_output['vision_output'].shape}")
    print(f"  Language: {block_output['language_output'].shape}")
    
    return vl_attention, block_output

def visualize_attention_weights(vision_features, language_features, attention_weights, 
                               save_path="attention_visualization.png"):
    """Visualize attention weights between vision and language"""
    # Convert tensors to numpy for plotting
    attn_weights = attention_weights[0, 0].detach().cpu().numpy()  # Take first head of first batch
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Full attention heatmap
    im1 = axes[0].imshow(attn_weights, cmap='viridis', aspect='auto')
    axes[0].set_title('Vision-Language Attention Weights')
    axes[0].set_xlabel('Language Tokens')
    axes[0].set_ylabel('Vision Patches')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Attention over vision patches (averaged across language tokens)
    vision_attention = np.mean(attn_weights, axis=1)  # Average across language tokens
    axes[1].plot(vision_attention)
    axes[1].set_title('Attention to Vision Patches')
    axes[1].set_xlabel('Vision Patch Index')
    axes[1].set_ylabel('Attention Weight')
    axes[1].grid(True)
    
    # 3. Attention over language tokens (averaged across vision patches)
    language_attention = np.mean(attn_weights, axis=0)  # Average across vision patches
    axes[2].plot(language_attention)
    axes[2].set_title('Attention to Language Tokens')
    axes[2].set_xlabel('Language Token Index')
    axes[2].set_ylabel('Attention Weight')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Attention visualization saved to {save_path}")

if __name__ == "__main__":
    model, output = test_cross_modal_attention()
    
    # Visualize attention weights if needed
    # visualize_attention_weights(
    #     output['vision_features'], 
    #     output['language_features'], 
    #     output['language_vision_attention']
    # )
    
    print("\nCross-modal attention implementation test completed successfully!")
```

### Exercise 2: Vision-Transformer for Robotics Perception

#### Objective
Implement a Vision Transformer architecture tailored for robotic perception tasks and integrate it with language understanding.

#### Steps
1. Implement Vision Transformer encoder
2. Add robotic-specific attention mechanisms
3. Integrate with language processing
4. Test on robotic vision tasks

```python
#!/usr/bin/env python3
"""Vision Transformer for Robotics Perception"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings for Vision Transformer"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer for patch extraction
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, embed_dim)  # +1 for CLS token
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding
        
        Args:
            x: Input image [batch_size, channels, height, width]
        
        Returns:
            Patch embeddings with positional encoding [batch_size, n_patches + 1, embed_dim]
        """
        B, C, H, W = x.shape
        
        # Validate input dimensions
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})."
        
        # Extract patches using convolution
        x = self.proj(x)  # [B, embed_dim, n_patches_h, n_patches_w]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        # Replicate class token for the entire batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenate class token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # [B, n_patches + 1, embed_dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply dropout
        x = self.dropout(x)
        
        return x

class RoboticVisionTransformerBlock(nn.Module):
    """Transformer block specialized for robotic vision tasks"""
    
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, attention_dropout: float = 0.1, 
                 stochastic_depth: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoboticSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attention_dropout=attention_dropout
        )
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # Pre-norm MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class RoboticSelfAttention(nn.Module):
    """Self-attention mechanism with robotic-specific enhancements"""
    
    def __init__(self, embed_dim: int, n_heads: int, attention_dropout: float = 0.0):
        super().__init__()
        
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} not divisible by n_heads {n_heads}"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(attention_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # B: batch, N: sequence length, C: channels/embedding dim
        
        # Project to query, key, value
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Make torchscript happy (cannot use tensor indexing)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.proj_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class RoboticVisionTransformer(nn.Module):
    """Vision Transformer specialized for robotics applications"""
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 representation_size: Optional[int] = None,
                 num_classes: int = 1000,
                 distilled: bool = False,
                 stochastic_depth: float = 0.0,
                 **kwargs):
        super().__init__()
        
        # Store configuration
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.distilled = distilled
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.patch_grid = (img_size // patch_size, img_size // patch_size)
        
        # Class/token embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, depth)]
        
        # Transformer layers
        self.blocks = nn.Sequential(*[
            RoboticVisionTransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth=dpr[i]
            ) for i in range(depth)
        ])
        
        # Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Representation layer (optional)
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        if self.distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.init_weights()
        
        # Robotic perception heads
        self.perception_heads = self._create_perception_heads()
        
    def _create_perception_heads(self):
        """Create additional perception heads for robotics tasks"""
        return nn.ModuleDict({
            # Object detection head
            'object_detection': nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 4),  # bbox coordinates (x, y, w, h)
            ),
            # Object classification head
            'object_classification': nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 100),  # 100 object categories
            ),
            # Depth estimation head (for 64x64 depth map)
            'depth_estimation': nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 64*64),  # Flattened 64x64 depth map
            ),
            # Surface normal estimation head
            'surface_normals': nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 64*64*3),  # Flattened 64x64x3 normals
            )
        })
    
    def init_weights(self):
        """Initialize weights"""
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=.02)
        
        # Initialize all linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for module"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through the transformer"""
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add class tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through transformer blocks
        x = self.blocks(x)
        
        # Apply final norm
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning features and perception outputs"""
        # Extract features
        features = self.forward_features(x)
        
        # Apply perception heads
        perception_outputs = {}
        
        for head_name, head in self.perception_heads.items():
            if head_name in ['depth_estimation', 'surface_normals']:
                # For spatial outputs, reshape to 2D maps
                batch_size = features.size(0)
                if head_name == 'depth_estimation':
                    output_size = (64, 64)
                else:  # surface_normals
                    output_size = (64, 64, 3)
                
                # Extract relevant features (maybe just the CLS token or spatial features)
                spatial_features = features[:, 1:, :]  # Exclude CLS token
                
                # Process with head
                output_flat = head(spatial_features.mean(dim=1))  # Average pool spatial dims
                
                # Reshape to spatial map
                if head_name == 'depth_estimation':
                    output_map = output_flat.reshape(batch_size, *output_size)
                else:  # surface normals
                    output_map = output_flat.reshape(batch_size, 64, 64, 3).permute(0, 3, 1, 2)  # BCHW format
                
                perception_outputs[head_name] = output_map
            else:
                # For non-spatial outputs, use CLS token
                cls_features = features[:, 0, :]  # Extract CLS token features
                perception_outputs[head_name] = head(cls_features)
        
        # Classification output (if needed)
        if hasattr(self, 'head') and self.head is not None:
            x_cls = self.pre_logits(features[:, 0])  # CLS token for classification
            classification_output = self.head(x_cls)
            perception_outputs['classification'] = classification_output
        
        return {
            'features': features,
            'perception_outputs': perception_outputs,
            'cls_features': features[:, 0, :]  # CLS token features for downstream tasks
        }

class SpatialAttention(nn.Module):
    """Spatial attention for robotics-specific visual attention"""
    
    def __init__(self, embed_dim: int, spatial_dims: Tuple[int, int]):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.embed_dim = embed_dim
        
        # Spatial attention convolutions
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.channel_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        # Softmax for attention weights
        self.spatial_softmax = nn.Softmax(dim=-1)  # Apply softmax over spatial dimensions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to Vision Transformer features
        
        Args:
            x: Input features [batch, sequence_length, embed_dim]
               where sequence_length = height * width + 1 (for CLS token)
        
        Returns:
            Attended features [batch, sequence_length, embed_dim]
        """
        B, N, C = x.shape
        H, W = self.spatial_dims
        P = int(math.sqrt(N - 1))  # Exclude CLS token when inferring patch grid
        
        # Reshape to [B, C, H, W] excluding CLS token
        # Assume N = P*P + 1 (patches + CLS token)
        patches = x[:, 1:, :]  # Exclude CLS token
        patches = patches.transpose(1, 2).reshape(B, C, P, P)  # [B, C, P, P]
        
        # Apply spatial attention
        spatial_weights = self.spatial_conv(patches)  # [B, C, P, P]
        channel_weights = self.channel_conv(F.relu(spatial_weights))  # [B, C, P, P]
        
        # Apply attention weights to input patches
        attended_patches = patches * channel_weights
        attended_patches = attended_patches.reshape(B, C, P*P).transpose(1, 2)  # [B, P*P, C]
        
        # Combine with CLS token
        cls_token = x[:, 0:1, :]  # [B, 1, C]
        attended_features = torch.cat([cls_token, attended_patches], dim=1)  # [B, P*P+1, C]
        
        return attended_features

def test_vit_for_robotics():
    """Test the Vision Transformer implementation"""
    print("Testing Vision Transformer for Robotics")
    print("=" * 50)
    
    # Define parameters
    batch_size = 2
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    depth = 12
    n_heads = 12
    
    # Create random input
    input_tensor = torch.randn(batch_size, in_channels, img_size, img_size)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Initialize the Vision Transformer
    vit = RoboticVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        num_classes=1000  # For ImageNet classification
    )
    
    # Forward pass
    output = vit(input_tensor)
    
    print(f"\nOutput structure:")
    print(f"  Features shape: {output['features'].shape}")
    print(f"  CLS features shape: {output['cls_features'].shape}")
    
    print(f"\nPerception outputs:")
    for head_name, head_output in output['perception_outputs'].items():
        print(f"  {head_name}: {head_output.shape}")
    
    # Test spatial attention
    print(f"\nTesting spatial attention...")
    spatial_attn = SpatialAttention(
        embed_dim=embed_dim,
        spatial_dims=(14, 14)  # 14x14 patches from 224/16
    )
    
    attended_features = spatial_attn(output['features'])
    print(f"Attended features shape: {attended_features.shape}")
    
    return vit, output

if __name__ == "__main__":
    model, output = test_vit_for_robotics()
    print("\nVision Transformer for Robotics test completed successfully!")
```

### Exercise 3: Transformer Integration with Language Models

#### Objective
Integrate vision transformers with large language models for complete VLA systems.

#### Steps
1. Implement vision-language fusion
2. Create multimodal embeddings
3. Implement cross-modal transformer
4. Test on robotics tasks

```python
#!/usr/bin/env python3
"""Transformer integration with language models for VLA systems"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple

class VisionLanguageFusion(nn.Module):
    """Fusion module for integrating vision and language features"""
    
    def __init__(self, vision_dim: int, language_dim: int, fusion_dim: int):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.fusion_dim = fusion_dim
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.language_proj = nn.Linear(language_dim, fusion_dim)
        
        # Cross-attention for vision-to-language and language-to-vision
        self.vision_language_attention = CrossModalAttention(fusion_dim, 8)
        self.language_vision_attention = CrossModalAttention(fusion_dim, 8)
        
        # Fusion mechanism
        self.fusion_mechanism = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Layer normalization
        self.vision_norm = nn.LayerNorm(fusion_dim)
        self.language_norm = nn.LayerNorm(fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
    
    def forward(self, vision_features: torch.Tensor, 
                language_features: torch.Tensor,
                language_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Fuse vision and language features
        
        Args:
            vision_features: [batch, vision_seq_len, vision_dim]
            language_features: [batch, lang_seq_len, language_dim]
            language_mask: [batch, 1, 1, lang_seq_len] for attention masking
        
        Returns:
            Dictionary with fused features and attention weights
        """
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)    # [B, V_seq, fusion_dim]
        lang_proj = self.language_proj(language_features)  # [B, L_seq, fusion_dim]
        
        # Normalize
        vision_norm = self.vision_norm(vision_proj)
        lang_norm = self.language_norm(lang_proj)
        
        # Cross-attention: language attends to vision
        lang_attended, v2l_attention = self.vision_language_attention(
            query=lang_norm,      # Language queries vision
            key=vision_norm,      # Vision is key
            value=vision_norm,    # Vision is value
            mask=None             # No mask for vision (usually dense)
        )
        
        # Cross-attention: vision attends to language
        vision_attended, l2v_attention = self.language_vision_attention(
            query=vision_norm,    # Vision queries language
            key=lang_norm,        # Language is key
            value=lang_norm,      # Language is value
            mask=language_mask    # Mask for language padding
        )
        
        # Concatenate and fuse
        concat_features = torch.cat([vision_attended, lang_attended], dim=-1)
        fused_features = self.fusion_mechanism(concat_features)
        fused_features = self.fusion_norm(fused_features)
        
        return {
            'fused_features': fused_features,
            'vision_attended': vision_attended,
            'language_attended': lang_attended,
            'v2l_attention': v2l_attention,  # Language attending to vision
            'l2v_attention': l2v_attention   # Vision attending to language
        }

class VLATransformer(nn.Module):
    """Complete Vision-Language-Action Transformer for robotics"""
    
    def __init__(self, 
                 vision_model: nn.Module,
                 language_model: nn.Module,
                 action_space_dim: int,
                 fusion_dim: int = 1024,
                 transformer_layers: int = 6,
                 n_heads: int = 8):
        super().__init__()
        
        self.vision_model = vision_model
        self.language_model = language_model
        self.action_space_dim = action_space_dim
        self.fusion_dim = fusion_dim
        
        # Vision-language fusion
        self.vision_language_fusion = VisionLanguageFusion(
            vision_dim=vision_model.embed_dim,
            language_dim=language_model.config.hidden_size,
            fusion_dim=fusion_dim
        )
        
        # Multimodal transformer layers
        self.multimodal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=n_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        
        # Action generation head
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, action_space_dim)
        )
        
        # Action parameterization (for stochastic policies)
        self.action_mean = nn.Linear(fusion_dim, action_space_dim)
        self.action_std = nn.Linear(fusion_dim, action_space_dim)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            'grasping': nn.Linear(fusion_dim, 1),  # Grasp success prediction
            'navigation': nn.Linear(fusion_dim, 4),  # Navigation (x, y, theta, confidence)
            'manipulation': nn.Linear(fusion_dim, 7),  # Manipulation (position + orientation)
        })
    
    def forward(self, 
                images: torch.Tensor, 
                language_input_ids: torch.Tensor,
                language_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the complete VLA system
        
        Args:
            images: [batch, channels, height, width]
            language_input_ids: [batch, seq_len] - tokenized text
            language_attention_mask: [batch, seq_len] - attention mask for text
        
        Returns:
            Dictionary with action predictions and task-specific outputs
        """
        batch_size = images.size(0)
        
        # Process vision
        vision_features = self.vision_model(images)  # [B, V_seq, V_dim]
        if isinstance(vision_features, dict):
            vision_features = vision_features['features']  # Extract features if dict returned
        
        # Process language
        language_outputs = self.language_model(
            input_ids=language_input_ids,
            attention_mask=language_attention_mask
        )
        language_features = language_outputs.last_hidden_state  # [B, L_seq, L_dim]
        
        # Fuse vision and language
        fusion_result = self.vision_language_fusion(
            vision_features=vision_features,
            language_features=language_features,
            language_mask=language_attention_mask.unsqueeze(1).unsqueeze(2) if language_attention_mask is not None else None
        )
        
        # Apply multimodal transformer
        multimodal_features = fusion_result['fused_features']
        multimodal_encoded = self.multimodal_transformer(multimodal_features)
        
        # Use pooled representation for action generation (e.g., CLS token style)
        # Take the first token as the pooled representation
        pooled_features = multimodal_encoded[:, 0, :]  # [B, fusion_dim]
        
        # Generate action predictions
        action_mean = torch.tanh(self.action_mean(pooled_features))  # Tanh to bound actions
        action_log_std = self.action_std(pooled_features)
        action_std = torch.exp(action_log_std)
        
        # Sample action from distribution (for stochastic policy)
        noise = torch.randn_like(action_std)
        action_sample = action_mean + action_std * noise
        
        # Task-specific outputs
        task_outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_outputs[task_name] = task_head(pooled_features)
        
        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'action_sample': action_sample,
            'task_outputs': task_outputs,
            'multimodal_features': multimodal_encoded,
            'attention_weights': {
                'vision_language_attention': fusion_result['v2l_attention'],
                'language_vision_attention': fusion_result['l2v_attention']
            }
        }

class ActionSequenceGenerator(nn.Module):
    """Generate action sequences from vision-language inputs"""
    
    def __init__(self, vla_model: VLATransformer, max_sequence_length: int = 100):
        super().__init__()
        
        self.vla_model = vla_model
        self.max_sequence_length = max_sequence_length
        
        # Sequence modeling components
        self.action_lstm = nn.LSTM(
            input_size=vla_model.action_space_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Sequence termination predictor
        self.termination_head = nn.Linear(512, 1)
        
    def forward(self, images: torch.Tensor, language_input_ids: torch.Tensor,
                language_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Generate action sequence"""
        batch_size = images.size(0)
        
        action_sequence = []
        termination_probs = []
        
        # Get initial action and features from VLA model
        initial_output = self.vla_model(images, language_input_ids, language_attention_mask)
        
        current_action = initial_output['action_sample'].unsqueeze(1)  # [B, 1, action_dim]
        action_sequence.append(current_action)
        
        # Use multimodal features as initial hidden state
        multimodal_features = initial_output['multimodal_features'][:, 0, :]  # [B, fusion_dim]
        
        # Initialize LSTM hidden state
        h0 = torch.tanh(multimodal_features.unsqueeze(0).repeat(2, 1, 1))  # [num_layers, B, hidden_dim]
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        
        # Generate action sequence
        for step in range(1, self.max_sequence_length):
            # Process current action through LSTM
            lstm_out, hidden = self.action_lstm(current_action, hidden)
            
            # Predict termination
            termination_prob = torch.sigmoid(self.termination_head(lstm_out.squeeze(1)))
            termination_probs.append(termination_prob)
            
            # Stop if probability of continuation is low
            if termination_prob.mean() < 0.1:
                break
            
            # In a real implementation, we'd feed back the action and updated state
            # For this example, we'll continue with dummy actions
            # In practice, you'd need to maintain robot state and scene context
            next_action = torch.randn(batch_size, 1, self.vla_model.action_space_dim) * 0.1
            action_sequence.append(next_action)
            current_action = next_action
        
        # Concatenate action sequence
        full_sequence = torch.cat(action_sequence, dim=1)  # [B, seq_len, action_dim]
        
        return {
            'action_sequence': full_sequence,
            'termination_probs': torch.cat(termination_probs, dim=1),
            'sequence_length': len(action_sequence)
        }

def create_example_vla_system():
    """Create an example VLA system with mock models"""
    
    # Create mock vision model (in practice, use a pre-trained one like ViT)
    class MockVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            # Simple CNN to simulate vision features
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # 14x14 patches
            self.patch_embed = nn.Linear(64, self.embed_dim)
            self.position_embed = nn.Parameter(torch.randn(1, 14*14 + 1, self.embed_dim))
        
        def forward(self, x):
            x = torch.relu(self.conv(x))  # [B, 64, H, W]
            x = self.adaptive_pool(x)     # [B, 64, 14, 14]
            x = x.permute(0, 2, 3, 1)   # [B, 14, 14, 64]
            x = self.patch_embed(x)      # [B, 14, 14, 768]
            x = x.reshape(x.size(0), -1, x.size(-1))  # [B, 196, 768]
            
            # Add CLS token
            cls_token = torch.zeros(x.size(0), 1, x.size(2))
            x = torch.cat([cls_token, x], dim=1)  # [B, 197, 768]
            
            # Add position embeddings
            x = x + self.position_embed[:, :x.size(1), :]
            
            return x
    
    # Use a pre-trained language model from HuggingFace
    language_model_name = "bert-base-uncased"
    language_model = AutoModel.from_pretrained(language_model_name)
    
    # Create VLA system
    vla_system = VLATransformer(
        vision_model=MockVisionModel(),
        language_model=language_model,
        action_space_dim=7,  # 7-DOF robot arm actions
        fusion_dim=512,
        transformer_layers=4
    )
    
    return vla_system

def test_vla_integration():
    """Test the complete VLA system"""
    print("Testing Vision-Language-Action System Integration")
    print("=" * 60)
    
    # Create the VLA system
    vla_system = create_example_vla_system()
    
    # Create example inputs
    batch_size = 1
    image_size = 224
    seq_length = 32
    
    images = torch.randn(batch_size, 3, image_size, image_size)
    
    # Create mock language input (token IDs)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "Pick up the red cup from the table"
    language_tokens = tokenizer(
        text, 
        return_tensors='pt', 
        padding='max_length', 
        max_length=seq_length,
        truncation=True
    )
    
    print(f"Image input shape: {images.shape}")
    print(f"Language input IDs shape: {language_tokens['input_ids'].shape}")
    print(f"Language attention mask shape: {language_tokens['attention_mask'].shape}")
    
    # Forward pass
    with torch.no_grad():
        output = vla_system(
            images=images,
            language_input_ids=language_tokens['input_ids'],
            language_attention_mask=language_tokens['attention_mask']
        )
    
    print(f"\nVLA System Output:")
    print(f"  Action mean: {output['action_mean'].shape}")
    print(f"  Action std: {output['action_std'].shape}")
    print(f"  Action sample: {output['action_sample'].shape}")
    print(f"  Multimodal features: {output['multimodal_features'].shape}")
    
    print(f"\nTask-specific outputs:")
    for task_name, task_output in output['task_outputs'].items():
        print(f"  {task_name}: {task_output.shape}")
    
    # Test action sequence generation
    print(f"\nTesting action sequence generation...")
    sequence_generator = ActionSequenceGenerator(vla_system, max_sequence_length=10)
    seq_output = sequence_generator(
        images=images,
        language_input_ids=language_tokens['input_ids'],
        language_attention_mask=language_tokens['attention_mask']
    )
    
    print(f"  Action sequence: {seq_output['action_sequence'].shape}")
    print(f"  Termination probs: {seq_output['termination_probs'].shape}")
    print(f"  Sequence length: {seq_output['sequence_length']}")
    
    return vla_system, output

if __name__ == "__main__":
    system, output = test_vla_integration()
    print("\nVLA system integration test completed successfully!")
```

### Exercise 4: Advanced Training Techniques

#### Objective
Implement advanced training techniques for VLA systems including curriculum learning and multi-task learning.

#### Steps
1. Implement curriculum learning scheduler
2. Create multi-task learning framework
3. Implement dynamic difficulty adjustment
4. Validate learning effectiveness

```python
#!/usr/bin/env python3
"""Advanced training techniques for VLA systems"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Any, Callable
import random
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for VLA training"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    curriculum_schedule: List[str] = None  # Curriculum stages
    
    # Curriculum learning parameters
    difficulty_levels: int = 5
    curriculum_strategy: str = "sequential"  # sequential, adaptive, random
    
    # Multi-task parameters
    task_weights: Dict[str, float] = None
    
    # Domain randomization
    domain_randomization: bool = True
    dr_frequency: int = 10  # How often to randomize domains

class CurriculumScheduler:
    """Manages curriculum learning progression"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_difficulty = 0  # Start with easiest
        self.steps_at_current_level = 0
        self.progress_threshold = 0.7  # Threshold to advance difficulty
        
    def update_difficulty(self, validation_score: float, 
                         validation_losses: Dict[str, float]) -> bool:
        """Update difficulty based on performance"""
        improved = self._evaluate_improvement(validation_losses)
        
        if improved and validation_score > self.progress_threshold:
            if self.current_difficulty < self.config.difficulty_levels - 1:
                self.current_difficulty += 1
                print(f"Curriculum advancement: Difficulty increased to level {self.current_difficulty}")
                return True
        
        return False
    
    def _evaluate_improvement(self, losses: Dict[str, float]) -> bool:
        """Evaluate if model has sufficiently improved to advance"""
        # In practice, this would look at various metrics
        # For this example, we'll use a simple approach based on loss improvement
        return True  # Simplified for demonstration
    
    def get_current_task_sampler(self, task_dataset: Dataset) -> DataLoader:
        """Get data loader with current difficulty level"""
        # This would sample data based on current difficulty
        # In practice, difficulty could be implemented through:
        # - Object complexity (simple vs. cluttered scenes)
        # - Language complexity (simple vs. complex instructions) 
        # - Action complexity (single vs. multi-step tasks)
        # - Environmental complexity (clean vs. cluttered)
        
        # For this example, we'll return a standard data loader
        return DataLoader(task_dataset, batch_size=self.config.batch_size)

class MultiTaskTrainer:
    """Handles multi-task training for VLA systems"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Optimizer with different learning rates for different components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Task-specific loss functions
        self.loss_functions = {
            'action_prediction': nn.MSELoss(),
            'object_detection': nn.CrossEntropyLoss(),
            'grasp_prediction': nn.BCEWithLogitsLoss(),
            'navigation': nn.MSELoss(),  # For navigation targets
            'language_understanding': nn.CrossEntropyLoss()
        }
        
        # Task weights (can be dynamic)
        self.task_weights = config.task_weights or {
            'action_prediction': 1.0,
            'object_detection': 0.8,
            'grasp_prediction': 0.9,
            'navigation': 0.7,
            'language_understanding': 0.6
        }
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for components"""
        # Different learning rates for different components
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "vision_model" in n],
                "lr": self.config.learning_rate * 0.1,  # Lower LR for pre-trained vision
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "language_model" in n],
                "lr": self.config.learning_rate * 0.1,  # Lower LR for pre-trained language
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "fusion" in n or "action_head" in n],
                "lr": self.config.learning_rate,  # Full LR for new components
            },
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        # Cosine annealing with warmup
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_epochs * 1000  # Estimate
        )
    
    def compute_multitask_loss(self, predictions: Dict[str, torch.Tensor], 
                             targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted sum of multiple task losses"""
        total_loss = 0.0
        task_losses = {}
        
        for task_name in self.task_weights.keys():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                loss_fn = self.loss_functions[task_name]
                
                if task_name == 'object_detection':
                    # For object detection, targets might be [batch, num_objects, 5] (x, y, w, h, class)
                    loss = loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))
                elif task_name == 'grasp_prediction':
                    # Binary classification loss
                    loss = loss_fn(pred, target.float())
                elif task_name == 'action_prediction':
                    # MSE for continuous action space
                    loss = loss_fn(pred, target)
                else:
                    # Default: MSE loss
                    loss = loss_fn(pred, target)
                
                weighted_loss = self.task_weights[task_name] * loss
                total_loss += weighted_loss
                task_losses[task_name] = weighted_loss.item()
        
        return total_loss, task_losses
    
    def dynamic_task_weighting(self, task_losses: Dict[str, float], 
                             performance_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Dynamically adjust task weights based on performance"""
        # Increase weight for tasks with deteriorating performance
        new_weights = self.task_weights.copy()
        
        for task_name in self.task_weights.keys():
            if task_name in performance_history and len(performance_history[task_name]) > 1:
                recent_performance = performance_history[task_name][-3:]  # Last 3 epochs
                trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                
                if trend > 0:  # Performance improving, decrease weight slightly
                    new_weights[task_name] *= 0.95
                elif trend < 0:  # Performance declining, increase weight
                    new_weights[task_name] *= 1.05
                
                # Keep weights within reasonable bounds
                new_weights[task_name] = max(0.1, min(2.0, new_weights[task_name]))
        
        return new_weights
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        task_losses_accum = {task: 0.0 for task in self.task_weights}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and targets
            images = batch['images']
            language_input_ids = batch['language_input_ids']
            language_attention_mask = batch['language_attention_mask']
            targets = batch['targets']
            
            # Forward pass
            outputs = self.model(
                images=images,
                language_input_ids=language_input_ids,
                language_attention_mask=language_attention_mask
            )
            
            # Compute multitask loss
            batch_loss, batch_task_losses = self.compute_multitask_loss(
                outputs['task_outputs'], targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Accumulate losses
            total_loss += batch_loss.item()
            for task, task_loss in batch_task_losses.items():
                task_losses_accum[task] += task_loss
            
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {batch_loss.item():.4f}")
        
        # Average losses
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches 
                          for task, loss in task_losses_accum.items()}
        
        return {
            'total_loss': avg_total_loss,
            **avg_task_losses
        }
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        task_losses_accum = {task: 0.0 for task in self.task_weights}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['images']
                language_input_ids = batch['language_input_ids']
                language_attention_mask = batch['language_attention_mask']
                targets = batch['targets']
                
                outputs = self.model(
                    images=images,
                    language_input_ids=language_input_ids,
                    language_attention_mask=language_attention_mask
                )
                
                batch_loss, batch_task_losses = self.compute_multitask_loss(
                    outputs['task_outputs'], targets
                )
                
                total_loss += batch_loss.item()
                for task, task_loss in batch_task_losses.items():
                    task_losses_accum[task] += task_loss
                
                num_batches += 1
        
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches 
                          for task, loss in task_losses_accum.items()}
        
        return {
            'total_loss': avg_total_loss,
            **avg_task_losses
        }

class DomainRandomizationTrainer:
    """Trainer with domain randomization capabilities"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.domain_configs = self._generate_domain_configs()
        self.current_domain = 0
    
    def _generate_domain_configs(self) -> List[Dict[str, Any]]:
        """Generate different domain configurations for randomization"""
        configs = []
        
        # Lighting conditions
        lighting_conditions = [
            {'intensity': 100, 'temperature': 3000, 'direction': [0, -1, -1]},  # Dim, warm, overhead
            {'intensity': 500, 'temperature': 5000, 'direction': [0, -1, -0.5]}, # Normal, neutral, overhead
            {'intensity': 1000, 'temperature': 8000, 'direction': [-1, -1, -0.3]}, # Bright, cool, side
            {'intensity': 200, 'temperature': 4000, 'direction': [1, 1, -0.8]},  # Low, neutral, backlit
        ]
        
        # Material properties
        material_properties = [
            {'roughness': 0.1, 'metallic': 0.0, 'specular': 0.5},  # Plastic-like
            {'roughness': 0.5, 'metallic': 0.2, 'specular': 0.8},  # Matte metal
            {'roughness': 0.05, 'metallic': 0.9, 'specular': 0.9}, # Shiny metal
            {'roughness': 0.8, 'metallic': 0.0, 'specular': 0.2}, # Rough fabric
        ]
        
        # Object arrangements
        object_arrangements = [
            {'density': 'sparse', 'distribution': 'uniform', 'types': ['simple']},
            {'density': 'moderate', 'distribution': 'cluttered', 'types': ['varied']},
            {'density': 'dense', 'distribution': 'clustered', 'types': ['complex']},
        ]
        
        # Generate combinations
        for light in lighting_conditions:
            for material in material_properties:
                for arrangement in object_arrangements:
                    config = {
                        'lighting': light,
                        'materials': material,
                        'arrangement': arrangement,
                        'sensor_noise': random.uniform(0.001, 0.01),
                        'dynamic_objects': random.choice([True, False])
                    }
                    configs.append(config)
        
        return configs
    
    def randomize_domain(self) -> Dict[str, Any]:
        """Randomize current domain configuration"""
        if self.config.domain_randomization:
            self.current_domain = random.randint(0, len(self.domain_configs) - 1)
            current_config = self.domain_configs[self.current_domain]
            
            print(f"Domain randomized - Configuration {self.current_domain}:")
            print(f"  Lighting: {current_config['lighting']['intensity']} intensity, {current_config['lighting']['temperature']}K")
            print(f"  Materials: Roughness {current_config['materials']['roughness']:.2f}, Metallic {current_config['materials']['metallic']:.2f}")
            print(f"  Arrangement: {current_config['arrangement']['density']} {current_config['arrangement']['distribution']}")
        
        return current_config if self.config.domain_randomization else {}
    
    def train_with_domains(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train with domain randomization"""
        trainer = MultiTaskTrainer(self.model, self.config)
        
        performance_history = {task: [] for task in trainer.task_weights.keys()}
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Apply domain randomization periodically
            if epoch % self.config.dr_frequency == 0:
                self.randomize_domain()
            
            # Train for one epoch
            train_metrics = trainer.train_epoch(train_loader)
            print(f"Train - Total Loss: {train_metrics['total_loss']:.4f}")
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            print(f"Val - Total Loss: {val_metrics['total_loss']:.4f}")
            
            # Record performance for dynamic weighting
            for task in trainer.task_weights.keys():
                if task in val_metrics:
                    performance_history[task].append(val_metrics[task])
            
            # Dynamically adjust task weights based on performance
            new_weights = trainer.dynamic_task_weighting(val_metrics, performance_history)
            trainer.task_weights = new_weights
            print(f"Updated task weights: {new_weights}")

def main():
    """Main training function demonstrating advanced techniques"""
    print("Initializing Advanced VLA Training System")
    print("=" * 50)
    
    # Create mock VLA system (in practice, this would be a real VLA model)
    vla_system = create_example_vla_system()
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=5e-5,
        batch_size=8,
        num_epochs=3,  # Reduced for demo
        difficulty_levels=3,
        task_weights={
            'action_prediction': 1.0,
            'object_detection': 0.8,
            'navigation': 0.7
        }
    )
    
    # Create mock datasets (in practice, these would be real VLA datasets)
    print("Creating mock training data...")
    
    class MockVLADataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Create mock data
            images = torch.randn(3, 224, 224)
            language_input_ids = torch.randint(0, 1000, (32,))
            language_attention_mask = torch.ones(32)
            
            # Mock targets for different tasks
            targets = {
                'action_prediction': torch.randn(7),  # 7-DOF actions
                'object_detection': torch.randint(0, 10, (50, 5)),  # 50 detections: [x, y, w, h, class]
                'navigation': torch.randn(4),  # [delta_x, delta_y, theta, confidence]
            }
            
            return {
                'images': images,
                'language_input_ids': language_input_ids,
                'language_attention_mask': language_attention_mask,
                'targets': targets
            }
    
    train_dataset = MockVLADataset(size=50)
    val_dataset = MockVLADataset(size=20)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize curriculum scheduler
    curriculum_scheduler = CurriculumScheduler(config)
    
    # Initialize domain randomization trainer
    dr_trainer = DomainRandomizationTrainer(vla_system, config)
    
    # Run training
    print("Starting training with advanced techniques...")
    dr_trainer.train_with_domains(train_loader, val_loader)
    
    print("\nAdvanced training with curriculum learning and domain randomization completed!")

if __name__ == "__main__":
    main()
```

### Exercise 5: Performance Optimization and Deployment

#### Objective
Optimize VLA transformer models for deployment on robotic platforms and validate performance.

#### Steps
1. Implement model optimization techniques
2. Create deployment-ready model
3. Test performance on constrained hardware
4. Validate accuracy preservation

```python
#!/usr/bin/env python3
"""Performance optimization for VLA systems"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import get_default_qconfig, prepare, convert
import numpy as np
from typing import Dict, Tuple, Optional
import time

class VLAOptimizer:
    """Optimization techniques for VLA models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_model = model  # Keep reference to original
    
    def quantize_model(self, calib_loader: DataLoader, num_batches: int = 32) -> nn.Module:
        """Apply quantization to reduce model size and improve inference speed"""
        print("Applying quantization to VLA model...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Configure quantization
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare_qat(self.model)
        
        # Calibrate with sample data
        print("Calibrating quantized model...")
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= num_batches:
                    break
                
                # Forward pass for calibration
                _ = model_prepared(
                    images=batch['images'],
                    language_input_ids=batch['language_input_ids'],
                    language_attention_mask=batch['language_attention_mask']
                )
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        print("Quantization completed!")
        return quantized_model
    
    def prune_model(self, pruning_ratio: float = 0.3) -> nn.Module:
        """Apply pruning to reduce model parameters"""
        import torch.nn.utils.prune as prune
        
        print(f"Applying pruning with ratio {pruning_ratio}...")
        
        # Identify layers to prune (convolutional and linear layers)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    # Apply unstructured magnitude pruning
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    print(f"Pruned layer {name}")
                except Exception as e:
                    print(f"Could not prune layer {name}: {e}")
        
        print(f"Pruning completed with {pruning_ratio*100}% sparsity")
        return self.model
    
    def fuse_operations(self) -> nn.Module:
        """Fuse operations for optimization (Conv-BatchNorm-ReLU patterns)"""
        print("Fusing operations for optimization...")
        
        # Import functional fusion tools
        from torch.quantization import fuse_modules
        
        # Create a copy of the model for fusion
        fused_model = self.model
        
        # Apply fusing to common patterns
        for name, module in fused_model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for Conv-BN-ReLU patterns and fuse them
                # This is a simplified example - in practice, you'd identify specific patterns
                children = list(module.named_children())
                child_modules = [child[1] for child in children]
                
                # Fuse common patterns like Conv-BN-ReLU
                fused_patterns = [
                    ['conv', 'bn', 'relu'],
                    ['conv', 'relu'],
                    ['linear', 'relu'],
                    ['conv', 'bn']
                ]
                
                # For this example, we'll manually specify common fusions
                # In real implementation, you'd have a more systematic approach
                pass
        
        print("Operation fusion completed!")
        return fused_model
    
    def optimize_transformer_blocks(self) -> nn.Module:
        """Apply specific optimizations to transformer blocks"""
        print("Optimizing transformer blocks...")
        
        for name, module in self.model.named_modules():
            if 'transformer' in name.lower():
                if hasattr(module, 'apply'):
                    # Apply optimizations specific to transformer layers
                    self._optimize_attention_for_inference(module)
        
        return self.model
    
    def _optimize_attention_for_inference(self, module):
        """Optimize attention mechanisms for inference"""
        # In practice, this could involve:
        # - Flash attention implementation
        # - Sparse attention patterns
        # - Kernel fusion optimizations
        pass
    
    def generate_optimized_model(self, calibration_loader: DataLoader) -> nn.Module:
        """Generate fully optimized model with multiple techniques"""
        print("Generating optimized VLA model...")
        
        # Start with clean model
        opt_model = copy.deepcopy(self.original_model)
        
        # Apply optimizations in sequence
        # 1. Fuse operations first
        opt_model = self.fuse_operations(opt_model)
        
        # 2. Apply quantization
        opt_model = self.quantize_model(opt_model, calibration_loader)
        
        # 3. Apply pruning
        # opt_model = self.prune_model(opt_model, pruning_ratio=0.2)  # May conflict with quantization
        
        # 4. Optimize transformer blocks
        opt_model = self.optimize_transformer_blocks(opt_model)
        
        print("Model optimization sequence completed!")
        return opt_model

class PerformanceBenchmark:
    """Benchmark performance of VLA models"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def measure_inference_time(self, input_data: Dict, num_runs: int = 100) -> Dict[str, float]:
        """Measure inference time and other performance metrics"""
        print(f"Measuring inference performance over {num_runs} runs...")
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(**input_data)
        
        # Measure inference time
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                with torch.autograd.profiler.record_function("model_inference"):
                    output = self.model(**input_data)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / avg_time
        
        # Memory usage (approximate)
        if self.device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(self.device)
            memory_mb = max_memory / 1024 / 1024
        else:
            memory_mb = 0  # Hard to measure on CPU
        
        results = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'frames_per_second': fps,
            'estimated_memory_usage_mb': memory_mb,
            'num_runs': num_runs
        }
        
        return results
    
    def benchmark_model_sizes(self) -> Dict[str, int]:
        """Compare model sizes before and after optimization"""
        original_count = sum(p.numel() for p in self.original_model.parameters() if p.requires_grad)
        
        if hasattr(self, 'optimized_model'):
            optimized_count = sum(p.numel() for p in self.optimized_model.parameters() if p.requires_grad)
            reduction_percentage = (original_count - optimized_count) / original_count * 100
        else:
            optimized_count = original_count
            reduction_percentage = 0
        
        return {
            'original_parameters': original_count,
            'optimized_parameters': optimized_count,
            'size_reduction_percentage': reduction_percentage
        }
    
    def validate_accuracy_preservation(self, val_loader: DataLoader, 
                                     original_model: nn.Module, 
                                     optimized_model: nn.Module) -> Dict[str, float]:
        """Validate that accuracy is preserved after optimization"""
        print("Validating accuracy preservation after optimization...")
        
        original_model.eval()
        optimized_model.eval()
        
        differences = []
        mse_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get outputs from both models
                orig_output = original_model(
                    images=batch['images'].to(self.device),
                    language_input_ids=batch['language_input_ids'].to(self.device),
                    language_attention_mask=batch['language_attention_mask'].to(self.device)
                )
                
                opt_output = optimized_model(
                    images=batch['images'].to(self.device),
                    language_input_ids=batch['language_input_ids'].to(self.device),
                    language_attention_mask=batch['language_attention_mask'].to(self.device)
                )
                
                # Compare key outputs (e.g., action predictions)
                orig_actions = orig_output.get('action_mean', orig_output.get('action_sample'))
                opt_actions = opt_output.get('action_mean', opt_output.get('action_sample'))
                
                if orig_actions is not None and opt_actions is not None:
                    # Calculate difference
                    diff = torch.mean(torch.abs(orig_actions - opt_actions)).item()
                    mse = torch.mean((orig_actions - opt_actions) ** 2).item()
                    
                    differences.append(diff)
                    mse_losses.append(mse)
        
        avg_difference = np.mean(differences) if differences else 0
        avg_mse = np.mean(mse_losses) if mse_losses else 0
        
        return {
            'avg_output_difference': avg_difference,
            'avg_mse': avg_mse,
            'preservation_threshold_met': avg_difference < 0.05  # Less than 5% difference
        }

def create_deployable_model(model: nn.Module, 
                          example_inputs: Dict,
                          output_path: str):
    """Create a deployable model using TorchScript"""
    print(f"Creating deployable model at {output_path}...")
    
    model.eval()
    
    # Trace the model with example inputs
    traced_model = torch.jit.trace(model, (
        example_inputs['images'],
        example_inputs['language_input_ids'],
        example_inputs['language_attention_mask']
    ))
    
    # Optimize for inference
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save the optimized model
    torch.jit.save(optimized_model, output_path)
    
    print(f"Deployable model saved to {output_path}")
    return optimized_model

def main():
    """Main optimization and benchmarking function"""
    print("VLA Model Optimization and Benchmarking")
    print("=" * 50)
    
    # Create example VLA system
    vla_system = create_example_vla_system()
    
    # Create example inputs for testing
    example_inputs = {
        'images': torch.randn(1, 3, 224, 224),
        'language_input_ids': torch.randint(0, 1000, (1, 32)),
        'language_attention_mask': torch.ones(1, 32)
    }
    
    # Create optimizer
    optimizer = VLAOptimizer(vla_system)
    
    # Create mock calibration data loader
    class MockCalibrationLoader:
        def __init__(self, size=10):
            self.size = size
        
        def __iter__(self):
            for _ in range(self.size):
                yield {
                    'images': torch.randn(1, 3, 224, 224),
                    'language_input_ids': torch.randint(0, 1000, (1, 32)),
                    'language_attention_mask': torch.ones(1, 32)
                }
        
        def __len__(self):
            return self.size
    
    calib_loader = MockCalibrationLoader(size=5)  # Small for demo
    
    # Optimize the model
    print("Optimizing model...")
    optimized_model = optimizer.generate_optimized_model(calib_loader)
    
    # Create benchmarking system
    benchmark = PerformanceBenchmark(optimized_model)
    
    # Benchmark performance
    print("\nBenchmarking original model...")
    original_perf = benchmark.measure_inference_time(example_inputs)
    
    print("\nBenchmarking optimized model...")
    opt_benchmark = PerformanceBenchmark(optimized_model)
    optimized_perf = opt_benchmark.measure_inference_time(example_inputs)
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"Original: {original_perf['frames_per_second']:.2f} FPS")
    print(f"Optimized: {optimized_perf['frames_per_second']:.2f} FPS")
    print(f"Speedup: {optimized_perf['frames_per_second'] / original_perf['frames_per_second']:.2f}x")
    
    # Validate accuracy preservation
    print("\nValidating accuracy preservation...")
    acc_validation = benchmark.validate_accuracy_preservation(
        val_loader=calib_loader,
        original_model=vla_system,
        optimized_model=optimized_model
    )
    
    print(f"Accuracy validation result: {'PASS' if acc_validation['preservation_threshold_met'] else 'FAIL'}")
    print(f"Average output difference: {acc_validation['avg_output_difference']:.6f}")
    
    # Create deployable model
    deploy_path = "./deployable_vla_model.pt"
    deployable_model = create_deployable_model(optimized_model, example_inputs, deploy_path)
    
    print(f"\nDeployable model created at: {deploy_path}")
    print("\nVLA model optimization and benchmarking completed!")

if __name__ == "__main__":
    main()
```

## Assessment Criteria

Your implementation will be evaluated based on:

1. **Technical Correctness** (30%):
   - Proper implementation of transformer architectures
   - Correct vision-language fusion mechanisms
   - Appropriate attention mechanisms
   - Valid domain randomization techniques

2. **System Integration** (25%):
   - Seamless integration between components
   - Proper data flow and modality handling
   - Efficient memory and computation management
   - Appropriate error handling

3. **Advanced Techniques** (20%):
   - Effective curriculum learning implementation
   - Proper multi-task learning approach
   - Model optimization techniques
   - Performance validation

4. **Validation and Testing** (15%):
   - Thorough testing of components
   - Performance benchmarking
   - Accuracy preservation validation
   - Domain transfer assessment

5. **Documentation and Presentation** (10%):
   - Clear code documentation
   - Understanding of design choices
   - Quality of performance analysis
   - Professional presentation

## Troubleshooting Tips

1. **Memory Issues**: Use gradient checkpointing for large models
2. **NaN Values**: Check learning rates and gradient clipping
3. **Poor Convergence**: Verify data preprocessing and normalization
4. **Slow Training**: Optimize data loading and preprocessing pipelines
5. **Overfitting**: Implement proper regularization and validation

## Extensions for Advanced Students

- Implement neural architecture search for optimal VLA architectures
- Create meta-learning systems for rapid adaptation
- Develop unsupervised domain adaptation techniques
- Integrate reinforcement learning with VLA systems
- Design multi-robot collaborative VLA systems

This practical exercise provides comprehensive experience with implementing advanced transformer architectures for vision-language-action systems in humanoid robotics, from basic components to deployment optimization.