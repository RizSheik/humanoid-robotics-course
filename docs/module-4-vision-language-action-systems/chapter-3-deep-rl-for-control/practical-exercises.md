---
id: module-4-chapter-3-practical-exercises
title: 'Module 4 — Vision-Language-Action Systems | Chapter 3 — Practical Exercises'
sidebar_label: 'Chapter 3 — Practical Exercises'
sidebar_position: 5
---

# Chapter 3 — Practical Exercises

## Deep Reinforcement Learning for Humanoid Control: Implementation Guide

### Exercise 1: PPO Implementation for Humanoid Locomotion

#### Objective
Implement Proximal Policy Optimization (PPO) for humanoid robot locomotion control.

#### Background
PPO is a popular policy-gradient algorithm that balances sample efficiency with stability. In humanoid robotics, it can be used for learning robust locomotion policies.

#### Steps
1. Set up humanoid robot environment
2. Implement PPO algorithm
3. Configure for locomotion task
4. Train and evaluate policy

```python
#!/usr/bin/env python3
"""PPO Implementation for Humanoid Locomotion Control"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import copy
from collections import deque
import random

# Humanoid environment with vision, language, and action modalities
class HumanoidVLAEnv(gym.Env):
    def __init__(self):
        super(HumanoidVLAEnv, self).__init__()
        
        # Observation space: [vision_features, proprioception, language_features]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(512 + 20 + 768,),  # Example: vision(512), proprio(20), language(768)
            dtype=np.float32
        )
        
        # Action space: Joint velocities for humanoid
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(19,),  # Example: 19 DOF humanoid
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.max_steps = 1000
        self.vision_features = np.random.randn(512)
        self.proprioception_state = np.random.randn(20)
        self.language_features = np.random.randn(768)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Simulate initial state
        self.vision_features = np.random.randn(512) * 0.1
        self.proprioception_state = np.random.randn(20) * 0.01
        self.language_features = np.random.randn(768) * 0.1
        
        observation = np.concatenate([
            self.vision_features,
            self.proprioception_state,
            self.language_features
        ])
        
        return observation, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action (simulate physics)
        action = np.clip(action, -1.0, 1.0)
        self.proprioception_state += action * 0.01  # Simplified physics
        
        # Update vision features based on action effects
        self.vision_features += np.random.randn(512) * 0.01
        
        # Calculate reward (simulate forward progress, stability, energy efficiency)
        forward_progress = 0.1  # Simplified
        stability = 1.0 - min(0.5, np.abs(self.proprioception_state[:3]).mean())  # Head stability
        energy_penalty = -0.01 * np.abs(action).sum()  # Penalty for excessive energy use
        
        reward = forward_progress + stability + energy_penalty
        
        # Determine done condition
        terminated = self.current_step >= self.max_steps
        truncated = False  # In gymnasium, this is separate from terminated
        
        # Update next state
        self.vision_features += np.random.randn(512) * 0.02
        self.proprioception_state += np.random.randn(20) * 0.005
        self.language_features *= 0.99  # Language persists slightly
        
        observation = np.concatenate([
            self.vision_features,
            self.proprioception_state,
            self.language_features
        ])
        
        info = {
            'step': self.current_step,
            'reward_components': {
                'forward_progress': forward_progress,
                'stability': stability,
                'energy_penalty': energy_penalty
            }
        }
        
        return observation, reward, terminated, truncated, info

# PPO Policy Network
class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(PPONetwork, self).__init__()
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        features = self.shared_layers(x)
        
        # Actor: policy mean and log std
        mean = torch.tanh(self.actor_mean(features))
        log_std = self.actor_logstd.expand_as(mean)
        
        # Critic: value estimation
        value = self.critic(features)
        
        return mean, log_std, value
    
    def get_action(self, x):
        mean, log_std, value = self.forward(x)
        std = torch.exp(log_std)
        
        # Sample action from normal distribution
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        
        # Calculate log probability
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        # Apply tanh squashing for bounded actions
        action = torch.tanh(action)
        
        # Adjust log probability for tanh squashing
        action_log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        
        return action, action_log_prob, value

# PPO Agent Implementation
class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 epochs=10, mini_batch_size=64, lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.network = PPONetwork(obs_dim, action_dim).to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lam = lam
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """Select action with current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute generalized advantage estimates"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO objective"""
        if len(self.states) == 0:
            return np.nan, np.nan
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(self.rewards, self.values, self.dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset for mini-batch updates
        dataset_size = len(states)
        num_batches = max(1, dataset_size // self.mini_batch_size)
        
        policy_losses = []
        value_losses = []
        
        # Update network multiple epochs
        for epoch in range(self.epochs):
            # Shuffle indices for mini-batch
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, self.mini_batch_size):
                batch_indices = indices[i:i+self.mini_batch_size]
                
                # Extract mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                means, log_stds, new_values = self.network(batch_states)
                
                # Create new action distribution
                normals = torch.distributions.Normal(means, torch.exp(log_stds))
                new_log_probs = normals.log_prob(batch_actions).sum(-1, keepdim=True)
                
                # Ratio for PPO
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (Clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_losses_unclipped = (new_values - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses_unclipped, value_losses_clipped).mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        # Clear the experience buffer
        self.states, self.actions, self.rewards, self.log_probs, self.values, self.dones = [], [], [], [], [], []
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def save_model(self, path):
        """Save model to specified path"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model from specified path"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_ppo_humanoid(env, agent, num_episodes=1000, update_interval=2048):
    """Train humanoid locomotion policy using PPO"""
    print("Starting PPO training for humanoid locomotion...")
    
    episode_rewards = deque(maxlen=100)
    total_steps = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policy periodically
        if len(agent.states) >= update_interval:
            policy_loss, value_loss = agent.update()
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {np.mean(episode_rewards):.2f}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    print("PPO training completed!")
    return agent

def main():
    """Run PPO training for humanoid locomotion"""
    print("Initializing PPO for Humanoid Locomotion Environment")
    
    # Create environment and agent
    env = HumanoidVLAEnv()
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=64,
        lam=0.95
    )
    
    # Train the agent
    trained_agent = train_ppo_humanoid(env, agent, num_episodes=500)
    
    # Save the trained model
    trained_agent.save_model("./ppo_humanoid_checkpoint.pth")
    print("Trained model saved to ./ppo_humanoid_checkpoint.pth")

if __name__ == "__main__":
    main()
```

### Exercise 2: Vision-Language-Action Integration

#### Objective
Implement VLA integration with DRL for complex humanoid tasks.

#### Steps
1. Create multimodal network architecture
2. Implement cross-modal attention
3. Integrate with policy learning
4. Test on manipulation tasks

```python
#!/usr/bin/env python3
"""Vision-Language-Action Integration for Humanoid DRL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math

class VisionLanguageFusion(nn.Module):
    """Fusion network for vision and language modalities"""
    
    def __init__(self, vision_dim=512, language_dim=768, fusion_dim=512, num_heads=8):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.fusion_dim = fusion_dim
        
        # Linear projections to common space
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.language_proj = nn.Linear(language_dim, fusion_dim)
        
        # Cross-attention layers
        self.vision_language_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.language_vision_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(0.1)
        )
        self.language_ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.vision_norm1 = nn.LayerNorm(fusion_dim)
        self.vision_norm2 = nn.LayerNorm(fusion_dim)
        self.language_norm1 = nn.LayerNorm(fusion_dim)
        self.language_norm2 = nn.LayerNorm(fusion_dim)
        
        # Final fusion layer
        self.final_fusion = nn.Linear(fusion_dim * 2, fusion_dim)
    
    def forward(self, vision_features, language_features, language_mask=None):
        """
        Fuse vision and language features
        
        Args:
            vision_features: [batch_size, num_patches, vision_dim]
            language_features: [batch_size, seq_len, language_dim]
            language_mask: [batch_size, seq_len] - mask for padded tokens (1 for valid, 0 for padding)
        
        Returns:
            Fused features [batch_size, fused_dim]
        """
        # Project features to common dimension
        vision_proj = self.vision_proj(vision_features)      # [B, V_seq, F]
        language_proj = self.language_proj(language_features)  # [B, L_seq, F]
        
        # Apply cross-attention: language attends to vision
        # Key and Value from vision, Query from language
        if language_mask is not None:
            # Convert mask to attention mask format
            lang_attn_mask = (language_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L_seq]
        else:
            lang_attn_mask = None
        
        lang_vision_output, lang_vision_attn_weights = self.language_vision_attn(
            query=language_proj,      # [B, L_seq, F] Query
            key=vision_proj,          # [B, V_seq, F] Key
            value=vision_proj,        # [B, V_seq, F] Value
            attn_mask=lang_attn_mask  # Attention mask
        )
        
        # Apply cross-attention: vision attends to language
        # Key and Value from language, Query from vision
        vision_lang_output, vision_lang_attn_weights = self.vision_language_attn(
            query=vision_proj,        # [B, V_seq, F] Query
            key=language_proj,        # [B, L_seq, F] Key
            value=language_proj,      # [B, L_seq, F] Value
            attn_mask=lang_attn_mask  # Attention mask
        )
        
        # Apply layer norms and feed-forward networks
        # Language branch
        lang_residual = lang_vision_output
        lang_norm1 = self.language_norm1(lang_residual)
        lang_ffn_output = self.language_ffn(lang_norm1)
        lang_output = lang_residual + lang_ffn_output
        
        # Vision branch
        vision_residual = vision_lang_output
        vision_norm1 = self.vision_norm1(vision_residual)
        vision_ffn_output = self.vision_ffn(vision_norm1)
        vision_output = vision_residual + vision_ffn_output
        
        # Global pooling for fusion
        lang_pooled = lang_output.mean(dim=1)    # [B, fusion_dim]
        vision_pooled = vision_output.mean(dim=1)  # [B, fusion_dim]
        
        # Concatenate and fuse
        concatenated = torch.cat([vision_pooled, lang_pooled], dim=-1)  # [B, 2*fusion_dim]
        fused_features = self.final_fusion(concatenated)              # [B, fusion_dim]
        
        return {
            'fused_features': fused_features,
            'vision_output': vision_output,
            'language_output': lang_output,
            'vision_language_attention': vision_lang_attn_weights,
            'language_vision_attention': lang_vision_attn_weights
        }

class VLAPolicyNetwork(nn.Module):
    """Complete VLA policy network incorporating proprioception"""
    
    def __init__(self, vision_dim=512, language_dim=768, proprio_dim=20, 
                 action_dim=19, hidden_dim=512, num_heads=8):
        super().__init__()
        
        self.fusion_module = VisionLanguageFusion(
            vision_dim=vision_dim,
            language_dim=language_dim,
            fusion_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Integrate proprioception with fused vision-language features
        self.sensor_integration = nn.Sequential(
            nn.Linear(hidden_dim + proprio_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (for action generation)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network (for value estimation)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, vision_features, language_features, proprioception, 
                language_mask=None):
        """
        Forward pass for VLA policy
        
        Args:
            vision_features: [batch_size, num_patches, vision_dim]
            language_features: [batch_size, seq_len, language_dim] 
            proprioception: [batch_size, proprio_dim]
            language_mask: [batch_size, seq_len] - mask for padded tokens
        
        Returns:
            (action_mean, action_logstd, value)
        """
        # Fuse vision and language
        fusion_result = self.fusion_module(
            vision_features, 
            language_features, 
            language_mask
        )
        fused_vl = fusion_result['fused_features']
        
        # Integrate with proprioception
        integrated_features = torch.cat([fused_vl, proprioception], dim=-1)
        processed_features = self.sensor_integration(integrated_features)
        
        # Actor: generate action mean
        action_mean = torch.tanh(self.actor_mean(processed_features))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        # Critic: estimate value
        value = self.critic(processed_features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, vision_features, language_features, proprioception, 
                   language_mask=None):
        """Get action with log probability for PPO training"""
        action_mean, action_logstd, value = self.forward(
            vision_features, language_features, proprioception, language_mask
        )
        
        std = torch.exp(action_logstd)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.rsample()
        
        # Calculate log probability
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        # Apply tanh squashing for bounded actions
        action = torch.tanh(action)
        
        # Adjust log probability for tanh squashing
        log_prob_adjusted = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        
        return action, log_prob_adjusted, value

# VLA Environment with multimodal inputs
class HumanoidVLAEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Environment dimensions
        self.vision_dim = 512
        self.language_dim = 768
        self.proprio_dim = 20
        self.action_dim = 19
        
        # Observation spaces (represented as dictionaries for clarity)
        self.observation_space = gym.spaces.Dict({
            'vision': gym.spaces.Box(-np.inf, np.inf, shape=(20, self.vision_dim)),  # 20 patches example
            'language': gym.spaces.Box(-np.inf, np.inf, shape=(32, self.language_dim)),  # 32 tokens
            'language_mask': gym.spaces.Box(0, 1, shape=(32,), dtype=bool),  # Mask for language tokens
            'proprioception': gym.spaces.Box(-np.inf, np.inf, shape=(self.proprio_dim,))
        })
        
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_dim,))
        
        # Initialize state variables
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Generate random initial state
        observation = {
            'vision': np.random.randn(20, self.vision_dim) * 0.1,
            'language': np.random.randn(32, self.language_dim) * 0.01,
            'language_mask': np.ones(32, dtype=bool),  # All tokens valid initially
            'proprioception': np.random.randn(self.proprio_dim) * 0.01
        }
        
        return observation, {}
    
    def step(self, action):
        self.current_step += 1
        
        # Validate and clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Simulate environment dynamics
        # (In a real implementation, this would connect to a physics simulator)
        new_proprio = self.current_obs['proprioception'] + action * 0.01
        
        # Update vision features (simulating environment changes from actions)
        new_vision = self.current_obs['vision'] + np.random.randn(*self.current_obs['vision'].shape) * 0.01
        
        # Calculate reward based on task (example: forward progress + stability)
        reward = self.calculate_reward(action, new_proprio)
        
        # Determine done condition
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Prepare next observation
        next_obs = {
            'vision': new_vision,
            'language': self.current_obs['language'],  # Language typically persists
            'language_mask': self.current_obs['language_mask'],
            'proprioception': new_proprio
        }
        
        self.current_obs = next_obs
        
        info = {
            'step': self.current_step,
            'action_magnitude': np.linalg.norm(action)
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def calculate_reward(self, action, new_proprio):
        """Calculate reward based on action and new state"""
        # Example reward components:
        forward_progress = 0.1  # Simplified - would come from forward movement
        stability = 1.0 - min(0.5, np.abs(new_proprio[:3]).mean())  # Head stability
        energy_penalty = -0.01 * np.abs(action).sum()  # Energy efficiency
        smoothness = -0.001 * np.abs(action).var()  # Smooth movement penalty
        
        return forward_progress + stability + energy_penalty + smoothness

# VLA PPO Agent
class VLAPPOAgent:
    def __init__(self, vision_dim=512, language_dim=768, proprio_dim=20, 
                 action_dim=19, lr=3e-4, gamma=0.99, clip_epsilon=0.2, 
                 epochs=10, mini_batch_size=64, lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # VLA Policy network
        self.network = VLAPolicyNetwork(
            vision_dim=vision_dim,
            language_dim=language_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim
        ).to(self.device)
        
        self.target_network = copy.deepcopy(self.network).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lam = lam
        
        # Experience storage
        self.vision_features = []
        self.language_features = []
        self.language_masks = []
        self.proprioception = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, vision_features, language_features, proprioception, 
                      language_mask=None):
        """Select action based on current observation"""
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                torch.FloatTensor(vision_features).unsqueeze(0).to(self.device),
                torch.FloatTensor(language_features).unsqueeze(0).to(self.device),
                torch.FloatTensor(proprioception).unsqueeze(0).to(self.device),
                torch.BoolTensor(language_mask).unsqueeze(0).to(self.device) if language_mask is not None else None
            )
        
        return (
            action.cpu().numpy()[0], 
            log_prob.cpu().numpy()[0], 
            value.cpu().numpy()[0]
        )
    
    def store_transition(self, vision_feat, language_feat, lang_mask, proprio, 
                         action, reward, log_prob, value, done):
        """Store transition in experience buffer"""
        self.vision_features.append(vision_feat)
        self.language_features.append(language_feat)
        self.language_masks.append(lang_mask if lang_mask is not None else np.ones(language_feat.shape[0]))
        self.proprioception.append(proprio)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update VLA policy using collected experiences"""
        if len(self.vision_features) == 0:
            return np.nan, np.nan
        
        # Convert to tensors
        vision_tensor = torch.FloatTensor(self.vision_features).to(self.device)
        language_tensor = torch.FloatTensor(self.language_features).to(self.device)
        language_mask_tensor = torch.BoolTensor(self.language_masks).to(self.device)
        proprio_tensor = torch.FloatTensor(self.proprioception).to(self.device)
        actions_tensor = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        old_values_tensor = torch.FloatTensor(self.values).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(self.rewards, self.values, self.dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        
        # Mini-batch training
        dataset_size = len(vision_tensor)
        indices = torch.randperm(dataset_size)
        
        for epoch in range(self.epochs):
            for i in range(0, dataset_size, self.mini_batch_size):
                batch_indices = indices[i:i+self.mini_batch_size]
                
                # Extract batch
                batch_vision = vision_tensor[batch_indices]
                batch_language = language_tensor[batch_indices]
                batch_masks = language_mask_tensor[batch_indices]
                batch_proprio = proprio_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                means, logstds, new_values = self.network(
                    batch_vision, batch_language, batch_proprio, batch_masks
                )
                
                # Calculate new log probabilities
                normals = torch.distributions.Normal(means, torch.exp(logstds))
                new_log_probs = normals.log_prob(batch_actions).sum(-1, keepdim=True)
                
                # PPO ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Clipped value loss
                value_pred_clipped = old_values_tensor[batch_indices] + torch.clamp(
                    new_values - old_values_tensor[batch_indices],
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_losses_unclipped = (new_values - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses_unclipped, value_losses_clipped).mean()
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward and optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        # Clear experience buffer
        self.vision_features = []
        self.language_features = []
        self.language_masks = []
        self.proprioception = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute generalized advantage estimates"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns

def train_vla_humanoid(env, agent, num_episodes=500, update_interval=2048):
    """Train VLA-enabled humanoid policy"""
    print("Starting VLA-enhanced PPO training for humanoid...")
    
    episode_rewards = deque(maxlen=100)
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Extract observation components
            vision = obs['vision']
            language = obs['language']
            language_mask = obs['language_mask']
            proprio = obs['proprioception']
            
            # Select action
            action, log_prob, value = agent.select_action(vision, language, proprio, language_mask)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(vision, language, language_mask, proprio, 
                                 action, reward, log_prob, value, done)
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policy when buffer is full
        if len(agent.vision_features) >= update_interval:
            policy_loss, value_loss = agent.update()
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {np.mean(episode_rewards):.2f}")
        
        # Progress update
        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent

def main():
    print("Initializing VLA-Enhanced PPO Training for Humanoid Robots")
    
    # Create VLA environment and agent
    env = HumanoidVLAEnv()
    agent = VLAPPOAgent(
        vision_dim=512,
        language_dim=768,
        proprio_dim=20,
        action_dim=19,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=32,  # Smaller due to multimodal complexity
        lam=0.95
    )
    
    # Train the agent
    trained_agent = train_vla_humanoid(env, agent, num_episodes=300)
    
    # Save model
    torch.save({
        'model_state_dict': agent.network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, './vla_humanoid_ppo_model.pth')
    
    print("VLA-Enhanced humanoid training completed!")

if __name__ == "__main__":
    main()
```

### Exercise 3: Hierarchical RL for Long-Horizon Tasks

#### Objective
Implement hierarchical reinforcement learning for complex humanoid tasks.

#### Steps
1. Create hierarchical policy architecture
2. Implement manager-worker coordination
3. Train on long-horizon manipulation tasks
4. Evaluate hierarchy effectiveness

```python
#!/usr/bin/env python3
"""Hierarchical Reinforcement Learning for Humanoid Robotics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random

class ManagerPolicy(nn.Module):
    """Manager policy that generates high-level goals"""
    
    def __init__(self, state_dim, goal_dim, hidden_dim=512, num_layers=3):
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, goal_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Goal generation network
        self.goal_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, goal_dim),
            nn.Tanh()  # Bound goals to [-1, 1]
        )
        
        # Value function for manager
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.network[:-1](state)  # All layers except last linear layer
        goal = self.goal_decoder(features) 
        value = self.value_head(features)
        
        return goal, value

class WorkerPolicy(nn.Module):
    """Worker policy that achieves manager-provided goals"""
    
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=512, num_layers=3):
        super().__init__()
        
        # Combine state and goal
        input_dim = state_dim + goal_dim
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        # Action generation
        self.network = nn.Sequential(*layers)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Worker value function
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state, goal):
        # Concatenate state and goal
        combined_input = torch.cat([state, goal], dim=-1)
        
        features = self.network(combined_input)
        
        # Generate action
        action_mean = torch.tanh(self.action_mean(features))
        action_logstd = self.action_logstd.expand_as(action_mean)
        
        # Value estimation
        value = self.value_head(features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, state, goal):
        action_mean, action_logstd, value = self.forward(state, goal)
        
        # Sample action from distribution
        std = torch.exp(action_logstd)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.rsample()
        
        # Calculate log probability
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        # Apply tanh squashing (if needed)
        action = torch.tanh(action)
        log_prob_adjusted = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        
        return action, log_prob_adjusted, value

class HierarchicalPolicy(nn.Module):
    """Combined hierarchical policy with manager and worker"""
    
    def __init__(self, state_dim, goal_dim, action_dim, manager_horizon=10):
        super().__init__()
        
        self.manager = ManagerPolicy(state_dim, goal_dim)
        self.worker = WorkerPolicy(state_dim, goal_dim, action_dim)
        
        # Manager horizon (how long each goal should last)
        self.manager_horizon = manager_horizon
        
        # Current goal and step counter
        self.current_goal = None
        self.goal_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, state, update_manager=False):
        """Generate action based on current goal or get new goal if needed"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Update manager if needed
        if update_manager or self.current_goal is None or self.goal_steps >= self.manager_horizon:
            goal, manager_value = self.manager(state_tensor)
            self.current_goal = goal.detach()
            self.goal_steps = 0
        else:
            manager_value = None
        
        # Get action from worker based on current goal
        action, log_prob, worker_value = self.worker(state_tensor, self.current_goal)
        
        self.goal_steps += 1
        
        return {
            'action': action.squeeze(0),
            'log_prob': log_prob.squeeze(0),
            'manager_value': manager_value.squeeze(0) if manager_value is not None else None,
            'worker_value': worker_value.squeeze(0),
            'current_goal': self.current_goal.squeeze(0)
        }

# Hierarchical VLA Environment
class HumanoidHierarchicalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # State includes vision, proprioception, and language
        self.state_dim = 512 + 20 + 768  # vision(512) + proprio(20) + language(768)
        self.goal_dim = 10  # High-level goal dimension
        self.action_dim = 19  # Joint actions
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.action_dim,), dtype=np.float32
        )
        
        self.max_steps = 1000
        self.current_step = 0
        self.manager_horizon = 10  # How often manager updates goal
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_since_manager_update = 0
        
        # Generate random initial state
        self.current_state = np.random.randn(self.state_dim) * 0.1
        
        return self.current_state, {}
    
    def step(self, action):
        self.current_step += 1
        self.steps_since_manager_update += 1
        
        # Apply action (simplified physics simulation)
        action = np.clip(action, -1.0, 1.0)
        self.current_state += action[:self.current_state.shape[0]] * 0.01
        self.current_state += np.random.randn(self.state_dim) * 0.02  # Add noise
        
        # Calculate reward
        reward = self.calculate_reward(action)
        
        # Determine done condition
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Update manager goal periodically
        manager_update = (self.steps_since_manager_update >= self.manager_horizon)
        if manager_update:
            self.steps_since_manager_update = 0
        
        info = {
            'step': self.current_step,
            'manager_update': manager_update,
            'goal_progress': 0.0  # In a real system, this would track progress toward goal
        }
        
        return self.current_state, reward, terminated, truncated, info
    
    def calculate_reward(self, action):
        """Calculate reward based on action and state"""
        # Example: combination of progress, energy efficiency, stability
        progress = 0.05  # Simplified
        energy_cost = -0.01 * np.abs(action).sum()  # Energy penalty
        stability = 0.1 - min(0.1, np.abs(self.current_state[:3]).mean())  # Stability bonus
        
        return progress + energy_cost + stability

class HierarchicalPPOAgent:
    """PPO agent for hierarchical policy"""
    
    def __init__(self, state_dim, goal_dim, action_dim, manager_horizon=10, 
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10, 
                 mini_batch_size=64, lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hierarchical policy
        self.policy = HierarchicalPolicy(state_dim, goal_dim, action_dim, manager_horizon).to(self.device)
        
        # Separate optimizers for manager and worker
        self.manager_optimizer = optim.Adam(
            list(self.policy.manager.parameters()), lr=lr
        )
        self.worker_optimizer = optim.Adam(
            list(self.policy.worker.parameters()), lr=lr
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lam = lam
        self.manager_horizon = manager_horizon
        
        # Storage for experiences
        self.states = []
        self.goals = []
        self.actions = []
        self.rewards = []
        self.manager_log_probs = []
        self.worker_log_probs = []
        self.manager_values = []
        self.worker_values = []
        self.dones = []
        self.manager_updates = []
    
    def select_action(self, state):
        """Select action using hierarchical policy"""
        with torch.no_grad():
            output = self.policy(state, update_manager=True)  # Always update for fresh goal
        
        return (
            output['action'].cpu().numpy(),
            output['log_prob'].cpu().numpy(),
            output['manager_value'].cpu().numpy() if output['manager_value'] is not None else None,
            output['worker_value'].cpu().numpy(),
            output['current_goal'].cpu().numpy()
        )
    
    def store_transition(self, state, goal, action, reward, manager_log_prob, 
                         worker_log_prob, manager_value, worker_value, done, 
                         manager_update):
        """Store transition in experience buffer"""
        self.states.append(state)
        self.goals.append(goal)
        self.actions.append(action)
        self.rewards.append(reward)
        self.manager_log_probs.append(manager_log_prob if manager_log_prob is not None else 0.0)
        self.worker_log_probs.append(worker_log_prob)
        self.manager_values.append(manager_value if manager_value is not None else 0.0)
        self.worker_values.append(worker_value)
        self.dones.append(done)
        self.manager_updates.append(manager_update)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantage estimates"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self):
        """Update both manager and worker policies"""
        if len(self.states) == 0:
            return np.nan, np.nan, np.nan
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        goals = torch.FloatTensor(self.goals).to(self.device)
        manager_log_probs = torch.FloatTensor(self.manager_log_probs).to(self.device)
        worker_log_probs = torch.FloatTensor(self.worker_log_probs).to(self.device)
        manager_values = torch.FloatTensor(self.manager_values).to(self.device)
        worker_values = torch.FloatTensor(self.worker_values).to(self.device)
        
        # Compute advantages for both levels
        manager_advantages, manager_returns = self.compute_advantages(
            self.rewards, self.manager_values, self.dones
        )
        worker_advantages, worker_returns = self.compute_advantages(
            self.rewards, self.worker_values, self.dones
        )
        
        # Normalize advantages
        manager_advantages = (manager_advantages - manager_advantages.mean()) / (manager_advantages.std() + 1e-8)
        worker_advantages = (worker_advantages - worker_advantages.mean()) / (worker_advantages.std() + 1e-8)
        
        # Training metrics
        manager_losses = []
        worker_losses = []
        total_losses = []
        
        # Update both policies
        dataset_size = len(states)
        indices = torch.randperm(dataset_size)
        
        for epoch in range(self.epochs):
            for i in range(0, dataset_size, self.mini_batch_size):
                batch_indices = indices[i:i+self.mini_batch_size]
                
                # Extract batch
                batch_states = states[batch_indices]
                batch_goals = goals[batch_indices]
                batch_actions = actions[batch_indices]
                batch_manager_log_probs = manager_log_probs[batch_indices]
                batch_worker_log_probs = worker_log_probs[batch_indices]
                batch_manager_advantages = manager_advantages[batch_indices]
                batch_worker_advantages = worker_advantages[batch_indices]
                batch_manager_returns = manager_returns[batch_indices]
                batch_worker_returns = worker_returns[batch_indices]
                
                # Update worker (more frequent updates)
                worker_means, worker_logstds, new_worker_values = self.policy.worker(
                    batch_states, batch_goals
                )
                
                # Worker log probabilities
                worker_normals = torch.distributions.Normal(worker_means, torch.exp(worker_logstds))
                new_worker_log_probs = worker_normals.log_prob(batch_actions).sum(-1, keepdim=True)
                
                # Worker PPO
                worker_ratios = torch.exp(new_worker_log_probs - batch_worker_log_probs)
                worker_surr1 = worker_ratios * batch_worker_advantages
                worker_surr2 = torch.clamp(worker_ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_worker_advantages
                worker_loss = -torch.min(worker_surr1, worker_surr2).mean()
                
                # Worker value loss
                worker_value_loss = F.mse_loss(new_worker_values, batch_worker_returns)
                
                total_worker_loss = worker_loss + 0.5 * worker_value_loss
                
                # Backward and optimize worker
                self.worker_optimizer.zero_grad()
                total_worker_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.worker.parameters(), 0.5)
                self.worker_optimizer.step()
                
                # Update manager (less frequent)
                # For manager, we'll update based on goal achievement
                new_manager_values = self.policy.manager(batch_states)[1]  # Get only value
                
                manager_value_loss = F.mse_loss(new_manager_values, batch_manager_returns)
                manager_loss = 0.0  # Manager doesn't have policy gradient loss in this simplified version
                
                total_manager_loss = manager_value_loss
                
                # Backward and optimize manager
                self.manager_optimizer.zero_grad()
                total_manager_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.manager.parameters(), 0.5)
                self.manager_optimizer.step()
                
                manager_losses.append(total_manager_loss.item())
                worker_losses.append(total_worker_loss.item())
                total_losses.append(total_manager_loss.item() + total_worker_loss.item())
        
        # Clear experience buffer
        self.states = []
        self.goals = []
        self.actions = []
        self.rewards = []
        self.manager_log_probs = []
        self.worker_log_probs = []
        self.manager_values = []
        self.worker_values = []
        self.dones = []
        self.manager_updates = []
        
        return np.mean(manager_losses), np.mean(worker_losses), np.mean(total_losses)

def train_hierarchical_humanoid(env, agent, num_episodes=500, update_interval=2048):
    """Train hierarchical policy for humanoid robot"""
    print("Starting Hierarchical RL training for humanoid robot...")
    
    episode_rewards = deque(maxlen=100)
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Initialize hierarchical state
        agent.policy.current_goal = None
        agent.policy.goal_steps = 0
        
        while True:
            # Select action using hierarchical policy
            action, log_prob, manager_value, worker_value, goal = agent.select_action(obs)
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(
                obs, goal, action, reward,
                manager_value, log_prob, manager_value, worker_value,
                done, info.get('manager_update', False)
            )
            
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policy when buffer is full
        if len(agent.states) >= update_interval:
            manager_loss, worker_loss, total_loss = agent.update()
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {np.mean(episode_rewards):.2f}")
        
        # Progress update
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return agent

def main():
    print("Initializing Hierarchical RL for Humanoid Robotics")
    
    # Create environment and hierarchical agent
    env = HumanoidHierarchicalEnv()
    agent = HierarchicalPPOAgent(
        state_dim=env.state_dim,
        goal_dim=10,
        action_dim=env.action_dim,
        manager_horizon=10,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=32,
        lam=0.95
    )
    
    # Train the hierarchical agent
    trained_agent = train_hierarchical_humanoid(env, agent, num_episodes=300)
    
    # Save model
    torch.save({
        'manager_state_dict': agent.policy.manager.state_dict(),
        'worker_state_dict': agent.policy.worker.state_dict(),
        'manager_optimizer_state_dict': agent.manager_optimizer.state_dict(),
        'worker_optimizer_state_dict': agent.worker_optimizer.state_dict()
    }, './hierarchical_humanoid_model.pth')
    
    print("Hierarchical humanoid training completed!")

if __name__ == "__main__":
    main()
```

### Exercise 4: Multi-Agent Coordination for Humanoids

#### Objective
Implement multi-agent reinforcement learning for coordinated humanoid teams.

#### Steps
1. Create multi-agent environment
2. Implement communication protocols
3. Train coordinated policies
4. Evaluate team performance

```python
#!/usr/bin/env python3
"""Multi-Agent Reinforcement Learning for Humanoid Teams"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from collections import deque

class HumanoidMultiAgentEnv(gym.Env):
    """Multi-agent humanoid environment for coordination"""
    
    def __init__(self, num_agents=2):
        super().__init__()
        
        self.num_agents = num_agents
        self.agent_obs_dim = 512 + 20 + 768  # vision + proprio + language
        self.agent_action_dim = 19  # joint actions
        self.max_steps = 1000
        self.current_step = 0
        
        # Observation space: Dictionary with observations for each agent
        obs_spaces = {}
        for i in range(num_agents):
            obs_spaces[f'agent_{i}'] = Dict({
                'self_obs': Box(low=-np.inf, high=np.inf, shape=(self.agent_obs_dim,), dtype=np.float32),
                'other_agents': Box(low=-np.inf, high=np.inf, shape=(num_agents-1, self.agent_obs_dim), dtype=np.float32),
                'language_instruction': Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)  # Shared instruction
            })
        self.observation_space = Dict(obs_spaces)
        
        # Action space: Dictionary with actions for each agent
        action_spaces = {}
        for i in range(num_agents):
            action_spaces[f'agent_{i}'] = Box(low=-1.0, high=1.0, shape=(self.agent_action_dim,), dtype=np.float32)
        self.action_space = Dict(action_spaces)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize agent states
        self.agent_states = []
        for i in range(self.num_agents):
            # Each agent has its own observation based on position
            agent_obs = np.random.randn(self.agent_obs_dim) * 0.1
            self.agent_states.append(agent_obs)
        
        # Shared language instruction
        self.shared_instruction = np.random.randn(768) * 0.01
        
        # Create observations for all agents
        observations = {}
        for i in range(self.num_agents):
            # Other agents' observations (excluding self)
            others_obs = []
            for j in range(self.num_agents):
                if j != i:
                    others_obs.append(self.agent_states[j])
            
            if len(others_obs) > 0:
                others_obs = np.stack(others_obs)
            else:
                others_obs = np.zeros((0, self.agent_obs_dim))  # Empty if only 1 agent
            
            observations[f'agent_{i}'] = {
                'self_obs': self.agent_states[i],
                'other_agents': others_obs,
                'language_instruction': self.shared_instruction
            }
        
        return observations, {}
    
    def step(self, actions):
        self.current_step += 1
        
        # Apply actions to each agent
        next_states = []
        rewards = {}
        terminations = {}
        
        # Process actions for each agent
        for i in range(self.num_agents):
            agent_action = actions[f'agent_{i}']
            agent_action = np.clip(agent_action, -1.0, 1.0)
            
            # Update agent state based on action
            new_state = self.agent_states[i] + agent_action[:self.agent_states[i].shape[0]] * 0.01
            new_state += np.random.randn(self.agent_obs_dim) * 0.02  # Add noise
            next_states.append(new_state)
        
        # Calculate team reward (example: collective task completion)
        team_position = np.mean([state[:3] for state in next_states], axis=0)  # Average position
        proximity_penalty = 0  # Encourage agents to stay together
        if self.num_agents > 1:
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    dist = np.linalg.norm(next_states[i][:3] - next_states[j][:3])
                    proximity_penalty -= min(0.1, dist)  # Negative penalty (positive reward)
        
        # Individual and team rewards
        for i in range(self.num_agents):
            individual_reward = 0.05  # Base reward
            energy_penalty = -0.01 * np.abs(actions[f'agent_{i}']).sum()
            
            rewards[f'agent_{i}'] = individual_reward + energy_penalty + proximity_penalty * 0.1  # Distribute team reward
            terminations[f'agent_{i}'] = False  # All agents terminate together
        
        # Update states
        self.agent_states = next_states
        
        # Create next observations
        next_observations = {}
        for i in range(self.num_agents):
            # Other agents' observations (excluding self)
            others_obs = []
            for j in range(self.num_agents):
                if j != i:
                    others_obs.append(self.agent_states[j])
            
            if len(others_obs) > 0:
                others_obs = np.stack(others_obs)
            else:
                others_obs = np.zeros((0, self.agent_obs_dim))
            
            next_observations[f'agent_{i}'] = {
                'self_obs': self.agent_states[i],
                'other_agents': others_obs,
                'language_instruction': self.shared_instruction
            }
        
        # Truncations (all agents truncated together)
        truncations = {f'agent_{i}': self.current_step >= self.max_steps for i in range(self.num_agents)}
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'team_position': team_position.tolist(),
            'team_proximity': proximity_penalty
        }
        
        # Terminations: all agents terminate together
        all_terminated = all(terminations.values())
        
        return next_observations, rewards, terminations, truncations, info

class CommunicationModule(nn.Module):
    """Communication module for multi-agent coordination"""
    
    def __init__(self, agent_obs_dim, msg_dim=128, num_agents=2):
        super().__init__()
        
        self.agent_obs_dim = agent_obs_dim
        self.msg_dim = msg_dim
        self.num_agents = num_agents
        
        # Encode local observations for communication
        self.obs_encoder = nn.Sequential(
            nn.Linear(agent_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, msg_dim)
        )
        
        # Message aggregation
        self.msg_aggregator = nn.MultiheadAttention(
            embed_dim=msg_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Decode aggregated message
        self.msg_decoder = nn.Sequential(
            nn.Linear(msg_dim, 256),
            nn.ReLU(),
            nn.Linear(256, agent_obs_dim)
        )
    
    def forward(self, agent_obs_list):
        """
        Exchange messages between agents
        
        Args:
            agent_obs_list: List of [num_agents, batch_size, obs_dim]
        
        Returns:
            List of enhanced observations [num_agents, batch_size, obs_dim]
        """
        batch_size = agent_obs_list[0].size(0)
        
        # Encode each agent's observation into a message
        encoded_messages = []
        for obs in agent_obs_list:
            msg = self.obs_encoder(obs)  # [batch, msg_dim]
            encoded_messages.append(msg.unsqueeze(1))  # [batch, 1, msg_dim]
        
        # Stack messages
        all_messages = torch.cat(encoded_messages, dim=1)  # [batch, num_agents, msg_dim]
        
        # Self-attention across agents to aggregate information
        attended_messages, attn_weights = self.msg_aggregator(
            query=all_messages,
            key=all_messages,
            value=all_messages
        )  # [batch, num_agents, msg_dim]
        
        # Decode enhanced messages back to observation space
        enhanced_obs_list = []
        for i in range(self.num_agents):
            decoded = self.msg_decoder(attended_messages[:, i, :])  # [batch, obs_dim]
            
            # Combine original observation with communicated information
            enhanced_obs = agent_obs_list[i] + 0.1 * decoded
            enhanced_obs_list.append(enhanced_obs)
        
        return enhanced_obs_list, attn_weights

class MultiAgentPolicy(nn.Module):
    """Policy for a single agent in multi-agent environment"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Actor network
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs):
        features = self.obs_encoder(obs)
        
        # Action generation
        action_mean = torch.tanh(self.action_mean(features))
        action_logstd = self.action_logstd.expand_as(action_mean)
        
        # Value estimation
        value = self.critic(features)
        
        return action_mean, action_logstd, value
    
    def get_action(self, obs):
        action_mean, action_logstd, value = self(obs)
        
        std = torch.exp(action_logstd)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.rsample()
        
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        action = torch.tanh(action)
        log_prob_adjusted = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        
        return action, log_prob_adjusted, value

class MultiAgentMACA(nn.Module):
    """Multi-Agent Centralized Training with Decentralized Execution (MACA)"""
    
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.num_agents = num_agents
        
        # Individual agent policies
        self.agent_policies = nn.ModuleList([
            MultiAgentPolicy(obs_dim, action_dim, hidden_dim)
            for _ in range(num_agents)
        ])
        
        # Centralized critic that takes all agents' observations and actions
        self.centralized_critic = nn.Sequential(
            nn.Linear(num_agents * (obs_dim + action_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs_list, action_list=None):
        """Forward pass for centralized critic"""
        if action_list is None:
            action_list = [None for _ in range(self.num_agents)]
        
        # Get actions from policies if not provided
        action_outputs = []
        value_outputs = []
        
        for i in range(self.num_agents):
            obs = obs_list[i]
            if action_list[i] is None:
                action_mean, action_logstd, value = self.agent_policies[i](obs)
                std = torch.exp(action_logstd)
                normal = torch.distributions.Normal(action_mean, std)
                action = normal.rsample()
                
                log_prob = normal.log_prob(action).sum(-1, keepdim=True)
                action = torch.tanh(action)
                log_prob_adjusted = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
                
                action_outputs.append((action, log_prob_adjusted))
            else:
                action_outputs.append((action_list[i], None))
            value_outputs.append(value)
        
        # For centralized critic, concatenate all obs and actions
        all_obs = torch.cat(obs_list, dim=-1)
        all_actions = torch.cat([action[0] for action in action_outputs], dim=-1)
        
        combined_input = torch.cat([all_obs, all_actions], dim=-1)
        central_value = self.centralized_critic(combined_input)
        
        return action_outputs, value_outputs, central_value

class MultiAgentPPO:
    """PPO for multi-agent environments"""
    
    def __init__(self, num_agents, obs_dim, action_dim, lr=3e-4, 
                 gamma=0.99, clip_epsilon=0.2, epochs=10, mini_batch_size=32, 
                 lam=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        
        # Multi-agent policy network
        self.policy = MultiAgentMACA(num_agents, obs_dim, action_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizers = [
            optim.Adam(list(self.policy.agent_policies[i].parameters()), lr=lr)
            for i in range(num_agents)
        ]
        self.critic_optimizer = optim.Adam(
            list(self.policy.centralized_critic.parameters()), lr=lr
        )
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lam = lam
        
        # Experience storage
        self.reset_experience_buffer()
    
    def reset_experience_buffer(self):
        """Reset experience buffer"""
        self.experience = {
            f'agent_{i}': {
                'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 
                'values': [], 'dones': []
            } for i in range(self.num_agents)
        }
        self.central_values = []
        self.team_rewards = []
    
    def select_actions(self, obs_dict):
        """Select actions for all agents"""
        obs_list = [torch.FloatTensor(obs_dict[f'agent_{i}']).unsqueeze(0).to(self.device) 
                   for i in range(self.num_agents)]
        
        with torch.no_grad():
            action_outputs, value_outputs, central_value = self.policy(obs_list)
        
        actions = {}
        log_probs = {}
        
        for i in range(self.num_agents):
            action, log_prob = action_outputs[i]
            actions[f'agent_{i}'] = action.squeeze(0).cpu().numpy()
            log_probs[f'agent_{i}'] = log_prob.squeeze(0).cpu().numpy()
        
        return actions, log_probs, central_value.cpu().numpy()
    
    def store_experience(self, obs_dict, actions_dict, rewards_dict, 
                         log_probs_dict, values_dict, dones_dict, team_reward):
        """Store experience in buffer"""
        for i in range(self.num_agents):
            agent_obs = obs_dict[f'agent_{i}']
            agent_action = actions_dict[f'agent_{i}']
            agent_reward = rewards_dict[f'agent_{i}']
            agent_log_prob = log_probs_dict[f'agent_{i}']
            agent_value = values_dict[f'agent_{i}']
            agent_done = dones_dict[f'agent_{i}']
            
            self.experience[f'agent_{i}']['obs'].append(agent_obs)
            self.experience[f'agent_{i}']['actions'].append(agent_action)
            self.experience[f'agent_{i}']['log_probs'].append(agent_log_prob)
            self.experience[f'agent_{i}']['rewards'].append(agent_reward)
            self.experience[f'agent_{i}']['values'].append(agent_value)
            self.experience[f'agent_{i}']['dones'].append(agent_done)
        
        # Store team reward for centralized training
        self.team_rewards.append(team_reward)
    
    def update(self):
        """Update multi-agent policies"""
        if len(self.experience['agent_0']['obs']) == 0:
            return np.nan
        
        # Convert to tensors
        obs_tensors = {}
        action_tensors = {} 
        log_prob_tensors = {}
        reward_tensors = {}
        value_tensors = {}
        done_tensors = {}
        
        for i in range(self.num_agents):
            exp = self.experience[f'agent_{i}']
            obs_tensors[i] = torch.FloatTensor(exp['obs']).to(self.device)
            action_tensors[i] = torch.FloatTensor(exp['actions']).to(self.device)
            log_prob_tensors[i] = torch.FloatTensor(exp['log_probs']).to(self.device)
            reward_tensors[i] = torch.FloatTensor(exp['rewards']).to(self.device)
            value_tensors[i] = torch.FloatTensor(exp['values']).to(self.device)
            done_tensors[i] = torch.FloatTensor(exp['dones']).to(self.device)
        
        # Compute advantages for each agent
        agent_advantages = {}
        agent_returns = {}
        
        for i in range(self.num_agents):
            advantages, returns = self.compute_advantages(
                reward_tensors[i].cpu().numpy(),
                value_tensors[i].cpu().numpy(),
                done_tensors[i].cpu().numpy()
            )
            agent_advantages[i] = advantages
            agent_returns[i] = returns
        
        # Training metrics
        policy_losses = []
        value_losses = []
        
        dataset_size = len(obs_tensors[0])
        indices = torch.randperm(dataset_size)
        
        for epoch in range(self.epochs):
            for i in range(0, dataset_size, self.mini_batch_size):
                batch_indices = indices[i:i+self.mini_batch_size]
                
                # Individual agent updates
                for agent_id in range(self.num_agents):
                    batch_obs = obs_tensors[agent_id][batch_indices]
                    batch_actions = action_tensors[agent_id][batch_indices]
                    batch_old_log_probs = log_prob_tensors[agent_id][batch_indices]
                    batch_advantages = agent_advantages[agent_id][batch_indices]
                    batch_returns = agent_returns[agent_id][batch_indices]
                    
                    # Get policy output
                    means, logstds, new_values = self.policy.agent_policies[agent_id](batch_obs)
                    
                    # Calculate log probabilities
                    normals = torch.distributions.Normal(means, torch.exp(logstds))
                    new_log_probs = normals.log_prob(batch_actions).sum(-1, keepdim=True)
                    
                    # PPO ratios
                    ratios = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # PPO loss
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(new_values, batch_returns)
                    
                    # Backward and optimize
                    self.policy_optimizers[agent_id].zero_grad()
                    (policy_loss + 0.5 * value_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.agent_policies[agent_id].parameters(), 0.5
                    )
                    self.policy_optimizers[agent_id].step()
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
        
        # Clear experience buffer
        self.reset_experience_buffer()
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantage estimates"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

def train_multi_agent_humanoid(env, maca_agent, num_episodes=500, update_interval=2048):
    """Train multi-agent humanoid coordination"""
    print(f"Starting Multi-Agent training for {env.num_agents} humanoid agents...")
    
    episode_rewards = deque(maxlen=100)
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Select actions for all agents
            actions, log_probs, central_value = maca_agent.select_actions(obs)
            
            # Take actions in environment
            next_obs, rewards, terminations, truncations, info = env.step(actions)
            
            # Calculate team reward (use average of individual rewards for simplicity)
            team_reward = sum(rewards.values()) / len(rewards)
            
            # Store experience
            maca_agent.store_experience(
                obs, actions, rewards, log_probs, 
                {f'agent_{i}': central_value for i in range(env.num_agents)},
                terminations, team_reward
            )
            
            obs = next_obs
            episode_reward += team_reward
            total_steps += 1
            
            # Check if any agent is done
            if any(terminations.values()) or any(truncations.values()):
                break
        
        episode_rewards.append(episode_reward)
        
        # Update policy when buffer is full
        if len(maca_agent.experience['agent_0']['obs']) >= update_interval:
            policy_loss, value_loss = maca_agent.update()
            print(f"Episode {episode}, Team Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {np.mean(episode_rewards):.2f}")
        
        # Progress update
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Episode {episode}, Average Team Reward: {avg_reward:.2f}")
    
    return maca_agent

def main():
    print("Initializing Multi-Agent Humanoid Training")
    
    # Create multi-agent environment
    num_agents = 2
    env = HumanoidMultiAgentEnv(num_agents=num_agents)
    
    # Create MACA (Multi-Agent Centralized-Actor Critic) agent
    maca_agent = MultiAgentPPO(
        num_agents=num_agents,
        obs_dim=env.agent_obs_dim,
        action_dim=env.agent_action_dim,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        mini_batch_size=32,
        lam=0.95
    )
    
    # Train the multi-agent system
    trained_agent = train_multi_agent_humanoid(env, maca_agent, num_episodes=300)
    
    # Save the trained model
    torch.save({
        'policy_state_dict': maca_agent.policy.state_dict(),
        'agent_optimizers': [opt.state_dict() for opt in maca_agent.policy_optimizers],
        'critic_optimizer': maca_agent.critic_optimizer.state_dict()
    }, './multi_agent_humanoid_model.pth')
    
    print("Multi-agent humanoid training completed!")

if __name__ == "__main__":
    main()
```

### Exercise 5: Advanced Validation and Deployment

#### Objective
Validate and deploy the trained VLA system.

#### Steps
1. Create validation datasets
2. Implement validation metrics
3. Test on real hardware in simulation
4. Prepare deployment configurations

```python
#!/usr/bin/env python3
"""Validation and Deployment of VLA Systems"""

import numpy as np
import torch
import torch.nn as nn
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os

class VLAValidator:
    """Validation system for Vision-Language-Action models"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model = self.load_model(model_path)
        self.config = self.load_config(config_path) if config_path else {}
        
        # Validation metrics
        self.metrics = {
            'success_rate': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'mse_error': [],
            'dtw_distance': []  # Dynamic Time Warping for trajectory validation
        }
        
        # Performance tracking
        self.performance_stats = {
            'inference_time': [],
            'memory_usage': [],
            'fps': []
        }
    
    def load_model(self, model_path: str):
        """Load trained VLA model"""
        # This would load the specific model architecture based on training
        # For this example, we'll create a placeholder
        print(f"Loading model from {model_path}")
        
        # In a real implementation, this would load the actual saved model
        # with the correct architecture
        return None
    
    def load_config(self, config_path: str) -> Dict:
        """Load validation configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def validate_on_dataset(self, dataset_loader) -> Dict:
        """Validate model performance on a dataset"""
        print("Starting validation on dataset...")
        
        model_outputs = []
        ground_truth = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataset_loader):
                # Extract batch components (this would match your dataset format)
                vision_inputs = batch['vision']  # [B, C, H, W] or [B, patches, features]
                language_inputs = batch['language']  # [B, seq_len, features]
                language_masks = batch.get('language_mask', None)  # [B, seq_len] if available
                proprio_inputs = batch['proprioception']  # [B, proprio_dim]
                action_targets = batch['actions']  # [B, action_dim]
                
                # Measure inference time
                start_time = time.time()
                
                # Get model predictions
                if hasattr(self.model, 'get_action'):
                    predicted_actions, _, _ = self.model.get_action(
                        vision_inputs, language_inputs, proprio_inputs, language_masks
                    )
                else:
                    # Fallback for different model architectures
                    predicted_actions = self.model(
                        vision_inputs, language_inputs, proprio_inputs, language_masks
                    )
                
                inference_time = time.time() - start_time
                self.performance_stats['inference_time'].append(inference_time)
                
                # Store results
                model_outputs.extend(predicted_actions.cpu().numpy())
                ground_truth.extend(action_targets.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Validated {batch_idx} batches")
        
        # Calculate validation metrics
        results = self.calculate_validation_metrics(
            np.array(model_outputs), 
            np.array(ground_truth)
        )
        
        return results
    
    def calculate_validation_metrics(self, predictions: np.ndarray, 
                                   targets: np.ndarray) -> Dict:
        """Calculate comprehensive validation metrics"""
        results = {}
        
        # Basic accuracy (for discrete action spaces) - convert to classification task
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # For continuous actions, calculate error metrics
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(predictions - targets))
            
            # Calculate success rate (actions within acceptable tolerance)
            tolerance = 0.1  # Define acceptable action tolerance
            success_count = np.sum(np.all(np.abs(predictions - targets) < tolerance, axis=1))
            success_rate = success_count / len(predictions)
            
            results.update({
                'mse_error': mse,
                'rmse_error': rmse,
                'mae_error': mae,
                'success_rate': success_rate,
                'tolerance': tolerance
            })
        else:
            # For discrete actions
            predicted_classes = np.round(predictions).astype(int)
            target_classes = targets.astype(int)
            
            accuracy = accuracy_score(target_classes, predicted_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_classes, predicted_classes, average='weighted'
            )
            
            results.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Additional metrics
        if predictions.shape[1] > 1:  # Multi-dimensional actions
            # Calculate trajectory similarity using DTW if sequences are available
            if len(predictions) == len(targets) and len(predictions) > 1:
                # Calculate DTW distance for action sequences
                dtw_distances = []
                for i in range(len(predictions)):
                    if i > 0:
                        pred_seq = predictions[i-1:i+1]  # Current and previous
                        target_seq = targets[i-1:i+1]
                        
                        # Calculate DTW distance between sequences
                        # Simple Euclidean distance for this example
                        dist = np.linalg.norm(pred_seq - target_seq)
                        dtw_distances.append(dist)
                
                if dtw_distances:
                    results['avg_dtw_distance'] = np.mean(dtw_distances)
        
        # Performance metrics
        results['avg_inference_time'] = np.mean(self.performance_stats['inference_time']) if self.performance_stats['inference_time'] else 0
        results['std_inference_time'] = np.std(self.performance_stats['inference_time']) if self.performance_stats['inference_time'] else 0
        
        if len(self.performance_stats['inference_time']) > 0:
            results['fps'] = 1.0 / results['avg_inference_time']
        
        return results
    
    def validate_robustness(self, test_scenarios: List[Dict]) -> Dict:
        """Validate model robustness across different scenarios"""
        robustness_results = {}
        
        for scenario in test_scenarios:
            print(f"Validating scenario: {scenario['name']}")
            
            # Apply scenario-specific transformations
            scenario_predictions = []
            scenario_targets = []
            
            # This would run validation in the specific scenario
            # In practice, you'd have simulation environments for each scenario
            for batch in scenario['data_loader']:
                vision_inputs = batch['vision']
                language_inputs = batch['language']
                language_masks = batch.get('language_mask', None)
                proprio_inputs = batch['proprioception']
                action_targets = batch['actions']
                
                # Generate predictions
                predicted_actions, _, _ = self.model.get_action(
                    vision_inputs, language_inputs, proprio_inputs, language_masks
                )
                
                scenario_predictions.extend(predicted_actions.cpu().numpy())
                scenario_targets.extend(action_targets.cpu().numpy())
            
            # Calculate scenario-specific metrics
            scenario_metrics = self.calculate_validation_metrics(
                np.array(scenario_predictions),
                np.array(scenario_targets)
            )
            
            robustness_results[scenario['name']] = scenario_metrics
        
        return robustness_results

class VLAValidator:
    """Comprehensive validation system for VLA models"""
    
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        # In a real system, this would load the actual trained model
        self.model_path = model_path
        self.results_dir = "./validation_results/"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_comprehensive_validation(self, test_data_loader, validation_metrics=['accuracy', 'precision', 'recall', 'f1']):
        """Run comprehensive validation on test dataset"""
        print("Starting comprehensive validation...")
        
        all_predictions = []
        all_targets = []
        all_inputs = []  # Store inputs for analysis
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_data_loader):
                # Get model predictions
                predictions = self.model(inputs)  # In real system, this would be more complex
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_inputs.extend(inputs.cpu().numpy())  # Store for analysis
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx} batches")
        
        # Calculate metrics
        results = self.calculate_validation_metrics(
            np.array(all_predictions), 
            np.array(all_targets)
        )
        
        # Save results
        self.save_validation_results(results)
        
        # Generate plots
        self.generate_validation_plots(
            np.array(all_predictions),
            np.array(all_targets),
            np.array(all_inputs)
        )
        
        return results
    
    def calculate_validation_metrics(self, predictions, targets):
        """Calculate comprehensive validation metrics"""
        results = {}
        
        # Accuracy metrics for classification
        if len(np.unique(targets)) < 20:  # Assume classification if fewer than 20 unique values
            results['overall_accuracy'] = accuracy_score(targets, predictions.argmax(axis=1))
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions.argmax(axis=1), average='weighted'
            )
            
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1
        else:
            # For regression tasks
            mse = np.mean((predictions - targets) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - targets))
            
            results['mse'] = mse
            results['rmse'] = rmse
            results['mae'] = mae
            
            # Success rate (within certain tolerance)
            tolerance = 0.1
            success_rate = np.mean(np.abs(predictions - targets) < tolerance)
            results['success_rate'] = success_rate
        
        # Additional metrics
        results['mean_error'] = np.mean(np.abs(predictions - targets))
        results['std_error'] = np.std(np.abs(predictions - targets))
        
        # Correlation metrics
        correlations = []
        for i in range(min(predictions.shape[1], targets.shape[1])):
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        
        results['mean_correlation'] = np.mean(correlations)
        results['correlations'] = correlations
        
        return results
    
    def generate_validation_plots(self, predictions, targets, inputs):
        """Generate validation plots and analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Prediction vs Target scatter
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Use first dimension for scatter plot
            axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5)
            axes[0, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                           [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Target Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Prediction vs Target (Dimension 1)')
        else:
            axes[0, 0].scatter(targets, predictions, alpha=0.5)
            axes[0, 0].plot([targets.min(), targets.max()], 
                           [targets.min(), targets.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Target Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Prediction vs Target')
        
        # Plot 2: Error distribution
        errors = np.abs(predictions - targets)
        if errors.ndim > 1:
            errors = errors.flatten()
        axes[0, 1].hist(errors, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Absolute Errors')
        
        # Plot 3: Correlation heatmap for multi-dimensional outputs
        if predictions.shape[1] > 1 and targets.shape[1] > 1:
            min_dim = min(predictions.shape[1], targets.shape[1])
            correlation_matrix = np.corrcoef(
                predictions[:, :min_dim].T,
                targets[:, :min_dim].T
            )
            
            sns.heatmap(correlation_matrix[:min_dim, min_dim:], 
                       annot=True, fmt='.2f', ax=axes[1, 0], cmap='coolwarm')
            axes[1, 0].set_title('Prediction-Target Correlation Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'Single Dimension\nNo Heatmap Needed', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Correlation Analysis')
        
        # Plot 4: Success rate over different error tolerances
        tolerances = np.linspace(0.01, 0.5, 50)
        success_rates = []
        for tol in tolerances:
            success_rates.append(np.mean(np.all(np.abs(predictions - targets) < tol, axis=1)))
        
        axes[1, 1].plot(tolerances, success_rates)
        axes[1, 1].set_xlabel('Error Tolerance')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate vs Error Tolerance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/validation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_validation_results(self, results):
        """Save validation results to file"""
        results_file = f"{self.results_dir}/validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Validation results saved to {results_file}")
    
    def run_robustness_tests(self):
        """Run tests to evaluate model robustness"""
        print("Running robustness tests...")
        
        # Test 1: Noise sensitivity
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_results = {}
        
        for noise_level in noise_levels:
            # Add noise to inputs and measure performance drop
            # This would be implemented in a real system
            degradation = 0.05 * noise_level * 100  # Simulated result
            noise_results[noise_level] = 1.0 - degradation
        
        # Test 2: Domain shift evaluation
        # This would test on data from different domains/environments
        
        # Test 3: Adversarial robustness
        # This would test resilience to adversarial examples
        
        robustness_results = {
            'noise_robustness': noise_results,
            'domain_generalization': {},  # Would contain domain-specific results
            'adversarial_robustness': {}  # Would contain adversarial test results
        }
        
        robustness_file = f"{self.results_dir}/robustness_results.json"
        with open(robustness_file, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        print(f"Robustness results saved to {robustness_file}")
        return robustness_results

def main():
    print("Starting VLA System Validation")
    
    # Create validator instance
    validator = VLAValidator(model_path="./trained_vla_model.pth")
    
    # Run comprehensive validation
    # Note: In a real implementation, you would load a test dataset
    # For this example, we'll simulate the process
    
    # Simulate validation results
    print("Simulated validation completed successfully")
    
    # Run robustness tests
    robustness_results = validator.run_robustness_tests()
    
    print("Validation and robustness testing completed!")
    print("Results saved in ./validation_results/ directory")

if __name__ == "__main__":
    main()
```

## Assessment Criteria

Your implementation will be evaluated based on:

1. **Technical Implementation** (30%)
   - Correct implementation of hierarchical and multi-agent RL
   - Proper integration of VLA components
   - Efficient use of Isaac Sim features
   - Clean, well-documented code

2. **System Integration** (25%)
   - Seamless coordination between components
   - Proper data flow and communication
   - Robust error handling and validation
   - Real-time performance maintenance

3. **Domain Randomization Effects** (20%)
   - Appropriate randomization parameters
   - Effective diversity generation
   - Transfer learning improvement
   - Validity of synthetic data

4. **Validation and Testing** (15%)
   - Comprehensive testing procedures
   - Proper evaluation metrics
   - Performance validation
   - Quality assurance measures

5. **Documentation and Presentation** (10%)
   - Clear technical documentation
   - Understanding of design choices
   - Quality of implementation notes
   - Professional code formatting

## Troubleshooting Tips

1. **Isaac Sim Crashes**: Verify GPU memory and driver compatibility
2. **Slow Performance**: Reduce scene complexity or physics parameters
3. **NaN Values in Training**: Check learning rates and gradient clipping
4. **Convergence Issues**: Adjust hyperparameters or increase training time
5. **Domain Transfer**: Ensure sufficient domain randomization range

## Extensions for Advanced Students

- Implement meta-learning for rapid adaptation
- Create curriculum learning approaches
- Add uncertainty quantification to VLA systems
- Implement real robot deployment of learned policies
- Add human-in-the-loop learning capabilities

This practical exercise provides hands-on experience with advanced Isaac Sim techniques for humanoid robotics, focusing on synthetic data generation, domain randomization, and multi-modal learning integration.