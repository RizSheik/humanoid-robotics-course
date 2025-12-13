---
title: Simulation Exercises - AI Robot Brain Systems
description: Simulation-based exercises for understanding and testing AI systems in robotics
sidebar_position: 103
---

# Simulation Exercises - AI Robot Brain Systems

## Simulation Overview

This document provides comprehensive simulation exercises designed to help students understand and experiment with AI systems for robotics in a controlled, repeatable environment. Through these simulations, students will explore neural network architectures, machine learning algorithms, reinforcement learning, and integration challenges without the constraints of physical hardware.

## Learning Objectives

Through these simulation exercises, students will:
- Implement and experiment with various AI architectures for robotic applications
- Evaluate different machine learning approaches for robot control and perception
- Test reinforcement learning algorithms in simulated robotic environments
- Understand the challenges of training AI systems for robotics
- Analyze performance and safety considerations of AI robot brains

## Simulation Environment Setup

### Required Software
- **Python 3.8+**: Core programming environment with scientific libraries
- **PyTorch/TensorFlow**: Deep learning frameworks
- **Gym/Gymnasium**: Reinforcement learning environments
- **Stable-Baselines3**: Reinforcement learning algorithms
- **OpenCV**: Computer vision simulations
- **PyBullet**: Physics simulation for complex robotic systems
- **CUDA-compatible GPU (recommended)**: For accelerated deep learning

### Recommended Hardware Specifications
- Multi-core processor (8+ cores recommended for parallel training)
- 16GB+ RAM (32GB recommended for deep learning workloads)
- Dedicated GPU (NVIDIA RTX 3070 or equivalent for accelerated training)
- 25GB+ free disk space for models and datasets

## Exercise 1: Neural Network Architectures for Robotics

### Objective
Explore different neural network architectures and their suitability for robotic tasks.

### Simulation Setup
1. Install required packages:
```bash
pip install torch torchvision torchaudio
pip install gymnasium[box2d] pybullet
pip install matplotlib seaborn
```

2. Create a simulation environment for testing different architectures:
```python
# nn_architecture_comparison.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

class SimpleRobotSensors(nn.Module):
    def __init__(self):
        super(SimpleRobotSensors, self).__init__()
        # Simulate robot with 8 range sensors and 3 IMU values
        self.input_size = 11  # 8 rangefinders + 3 IMU values
    
    def generate_sample_data(self, num_samples=1000):
        """Generate simulated sensor data"""
        # 8 rangefinder sensors (0.1 to 10m range)
        range_sensors = np.random.uniform(0.1, 10.0, (num_samples, 8))
        
        # 3 IMU values (orientation, acceleration, angular velocity)
        imu_values = np.random.uniform(-1.0, 1.0, (num_samples, 3))
        
        # Combine sensor inputs
        sensor_data = np.concatenate([range_sensors, imu_values], axis=1)
        
        # Generate target actions based on sensors
        actions = np.zeros((num_samples, 2))  # [linear_vel, angular_vel]
        
        for i in range(num_samples):
            # Simple policy: avoid obstacles, move forward
            front_dist = range_sensors[i, 0]  # Front sensor
            right_dist = range_sensors[i, 1]  # Right sensor
            left_dist = range_sensors[i, 2]   # Left sensor
            
            # Calculate desired action
            linear_vel = min(1.0, front_dist / 2.0)  # Scale linear velocity with front clearance
            angular_vel = (right_dist - left_dist) * 0.5  # Turn toward better side
            
            actions[i] = [linear_vel, angular_vel]
        
        return torch.FloatTensor(sensor_data), torch.FloatTensor(actions)

class FCLayer(nn.Module):
    """Fully Connected Layer for comparison"""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FCLayer, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class CNNRobotController(nn.Module):
    """CNN-based controller treating sensors as spatial data"""
    def __init__(self, input_channels=11):  # 11 sensors treated as channels
        super(CNNRobotController, self).__init__()
        
        # Reshape input to treat sensors as spatial data (11x1x1)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),  # Adaptive pooling to fixed size
        )
        
        # Fully connected layers for output
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: linear_vel, angular_vel
        )
    
    def forward(self, x):
        # Reshape for CNN: (batch, channels, sequence)
        x = x.unsqueeze(2)  # Add sequence dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class LSTMRobotController(nn.Module):
    """LSTM-based controller for temporal sequence processing"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
        super(LSTMRobotController, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        # For this exercise, treat single sensor reading as sequence of length 1
        x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, _ = self.lstm(x)
        # Take the last output
        output = self.fc(lstm_out[:, -1, :])
        return output

def train_and_evaluate_model(model, train_loader, val_loader, epochs=50):
    """Train and evaluate a model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        for sensors, actions in train_loader:
            sensors, actions = sensors.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(sensors)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for sensors, actions in val_loader:
                sensors, actions = sensors.to(device), actions.to(device)
                outputs = model(sensors)
                loss = criterion(outputs, actions)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def compare_architectures():
    """Compare different neural architectures for robot control"""
    # Generate dataset
    sensor_simulator = SimpleRobotSensors()
    sensor_data, target_actions = sensor_simulator.generate_sample_data(num_samples=5000)
    
    # Split into train/validation
    split_idx = int(0.8 * len(sensor_data))
    train_sensors = sensor_data[:split_idx]
    train_actions = target_actions[:split_idx]
    val_sensors = sensor_data[split_idx:]
    val_actions = target_actions[split_idx:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_sensors, train_actions)
    val_dataset = TensorDataset(val_sensors, val_actions)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Define architectures to compare
    architectures = {
        'FC_Network': FCLayer(input_size=11, hidden_sizes=[64, 64, 32], output_size=2),
        'CNN_Controller': CNNRobotController(input_channels=11),
        'LSTM_Controller': LSTMRobotController(input_size=11, hidden_size=64, output_size=2)
    }
    
    results = {}
    
    for name, model in architectures.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        train_losses, val_losses = train_and_evaluate_model(model, train_loader, val_loader)
        
        training_time = time.time() - start_time
        
        results[name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_time': training_time,
            'final_val_loss': val_losses[-1] if val_losses else float('inf')
        }
        
        print(f"{name} - Final validation loss: {results[name]['final_val_loss']:.6f}")
        print(f"Training time: {training_time:.2f} seconds")
    
    return results
```

### Implementation Tasks
1. **Architecture Comparison**: Run the architecture comparison to see which performs best for robot control
2. **Custom Architecture**: Design and implement your own architecture that combines elements from different approaches
3. **Hyperparameter Tuning**: Experiment with different hyperparameters to optimize performance

```python
def plot_architecture_comparison(results):
    """Plot results of architecture comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        ax1.plot(result['train_losses'], label=name)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation curves
    ax2 = axes[0, 1]
    for name, result in results.items():
        ax2.plot(result['val_losses'], label=name)
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Bar chart of final losses
    ax3 = axes[1, 0]
    names = list(results.keys())
    final_losses = [results[name]['final_val_loss'] for name in names]
    bars = ax3.bar(names, final_losses)
    ax3.set_title('Final Validation Loss')
    ax3.set_ylabel('Loss')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{loss:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Bar chart of training times
    ax4 = axes[1, 1]
    training_times = [results[name]['training_time'] for name in names]
    bars_time = ax4.bar(names, training_times)
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add time labels on bars
    for bar, time_val in zip(bars_time, training_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# Run the comparison
if __name__ == '__main__':
    results = compare_architectures()
    plot_architecture_comparison(results)
```

### Analysis Questions
- Which architecture performed best for the robot control task? Why?
- How did the training times differ between architectures?
- What are the advantages and disadvantages of each approach for robotics?

## Exercise 2: Reinforcement Learning for Robot Navigation

### Objective
Implement and test reinforcement learning algorithms for robot navigation in simulation.

### Simulation Setup
1. Create navigation environments:
```python
# rl_navigation_exercise.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class NavigationEnvironment(gym.Env):
    """Simple navigation environment for robot"""
    def __init__(self):
        super(NavigationEnvironment, self).__init__()
        
        # Define action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Define observation space: [x, y, theta, goal_x, goal_y, obstacle_info]
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi, -10, -10, 0, 0, 0]),  # x, y, theta, goal_x, goal_y, dist_front, left, right
            high=np.array([10, 10, np.pi, 10, 10, 10, 10, 10]),
            dtype=np.float32
        )
        
        self.max_episode_steps = 200
        self.step_count = 0
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random start and goal positions
        self.robot_pos = np.random.uniform(-8, 8, size=2)
        self.robot_theta = np.random.uniform(-np.pi, np.pi)
        self.goal_pos = np.random.uniform(-9, 9, size=2)
        
        # Create obstacles
        self.obstacles = np.random.uniform(-9, 9, size=(3, 2))
        
        self.step_count = 0
        return self._get_observation(), {}
    
    def step(self, action):
        linear_vel, angular_vel = action
        
        # Update robot position (simple kinematic model)
        dt = 0.1
        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * dt
        self.robot_theta += angular_vel * dt
        
        # Keep angle in [-pi, pi]
        self.robot_theta = np.arctan2(np.sin(self.robot_theta), np.cos(self.robot_theta))
        
        # Calculate distances to obstacles
        robot_point = self.robot_pos
        front_vec = np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
        left_vec = np.array([np.cos(self.robot_theta + np.pi/4), np.sin(self.robot_theta + np.pi/4)])
        right_vec = np.array([np.cos(self.robot_theta - np.pi/4), np.sin(self.robot_theta - np.pi/4)])
        
        front_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * front_vec - obs)) 
                         for obs in self.obstacles])
        left_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * left_vec - obs)) 
                        for obs in self.obstacles])
        right_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * right_vec - obs)) 
                         for obs in self.obstacles])
        
        # Calculate distance to goal
        goal_dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # Calculate reward
        reward = 0
        
        # Distance-based reward
        reward -= 0.01 * goal_dist  # Encourage moving toward goal
        
        # Collision penalty
        min_obstacle_dist = min(front_dist, left_dist, right_dist)
        if min_obstacle_dist < 0.3:
            reward -= 5.0
            done = True
        elif goal_dist < 0.5:  # Reached goal
            reward += 10.0
            done = True
        elif self.step_count >= self.max_episode_steps:  # Episode timeout
            reward -= 1.0
            done = True
        elif abs(self.robot_pos[0]) > 10 or abs(self.robot_pos[1]) > 10:  # Out of bounds
            reward -= 2.0
            done = True
        else:
            done = False
        
        self.step_count += 1
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        goal_direction = self.goal_pos - self.robot_pos
        goal_distance = np.linalg.norm(goal_direction)
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0]) - self.robot_theta
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))  # Normalize angle
        
        # Calculate obstacle distances
        robot_point = self.robot_pos
        front_vec = np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
        left_vec = np.array([np.cos(self.robot_theta + np.pi/4), np.sin(self.robot_theta + np.pi/4)])
        right_vec = np.array([np.cos(self.robot_theta - np.pi/4), np.sin(self.robot_theta - np.pi/4)])
        
        front_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * front_vec - obs)) 
                         for obs in self.obstacles])
        left_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * left_vec - obs)) 
                        for obs in self.obstacles])
        right_dist = min([max(0.1, np.linalg.norm(robot_point + 0.5 * right_vec - obs)) 
                         for obs in self.obstacles])
        
        return np.array([
            self.robot_pos[0], self.robot_pos[1], self.robot_theta,
            self.goal_pos[0], self.goal_pos[1], 
            front_dist, left_dist, right_dist
        ], dtype=np.float32)
    
    def render(self):
        """Render the environment (optional visualization)"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw robot
        ax.plot(self.robot_pos[0], self.robot_pos[1], 'ro', markersize=10, label='Robot')
        
        # Draw goal
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'go', markersize=10, label='Goal')
        
        # Draw obstacles
        for obs in self.obstacles:
            ax.plot(obs[0], obs[1], 'ks', markersize=8, label='Obstacle' if obs[0] == self.obstacles[0][0] else "")
        
        # Draw robot orientation
        arrow_length = 0.5
        ax.arrow(self.robot_pos[0], self.robot_pos[1], 
                 arrow_length * np.cos(self.robot_theta), 
                 arrow_length * np.sin(self.robot_theta),
                 head_width=0.1, head_length=0.1, fc='red', ec='red',
                 label='Direction' if self.robot_theta == self.robot_theta else "")
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title('Navigation Environment')
        
        plt.show()

class DQNNetwork(nn.Module):
    """Deep Q-Network for navigation"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.fc_layers(x)

class DQNAgent:
    """DQN Agent for navigation"""
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks (discretize continuous action space for DQN)
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay()

def discretize_action_space(action_idx, n_linear=5, n_angular=5):
    """Map discrete action index to continuous action"""
    linear_vels = np.linspace(-1.0, 1.0, n_linear)
    angular_vels = np.linspace(-1.0, 1.0, n_angular)
    
    linear_idx = action_idx // n_angular
    angular_idx = action_idx % n_angular
    
    return np.array([linear_vels[linear_idx], angular_vels[angular_idx]])

def train_dqn_navigation(episodes=1000):
    """Train DQN agent for navigation"""
    env = NavigationEnvironment()
    state_size = env.observation_space.shape[0]
    action_size = 5 * 5  # 5 linear velocities × 5 angular velocities
    
    agent = DQNAgent(state_size, action_size)
    
    scores = deque(maxlen=100)
    training_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for t in range(200):  # Max steps per episode
            # Choose action
            action_idx = agent.act(state)
            action = discretize_action_space(action_idx)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience
            agent.learn(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        training_scores.append(total_reward)
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            avg_score = np.mean(scores) if scores else 0
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}')
    
    return agent, training_scores

def plot_training_results(scores):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg)
    plt.title('Moving Average of Scores (100-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Implementation Tasks
1. **DQN Implementation**: Train a DQN agent for navigation
2. **Performance Analysis**: Analyze training progress and agent performance
3. **Algorithm Comparison**: Implement and compare with other RL algorithms (PPO, SAC)

```python
# Continue with PPO implementation
class PPOPolicy(nn.Module):
    """PPO Policy Network for continuous action space"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PPOPolicy, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor network (outputs mean and log std for Gaussian policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        mean = torch.tanh(self.actor_mean(features))  # Bound actions to [-1, 1]
        value = self.critic(state)
        return mean, self.actor_logstd.expand_as(mean), value
    
    def get_action(self, state):
        mean, logstd, value = self.forward(state)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, k_epochs=4):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        
        self.policy = PPOPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def update(self, states, actions, rewards, logprobs, values, masks):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        old_logprobs = torch.FloatTensor(logprobs).unsqueeze(1)
        values = torch.FloatTensor(values).unsqueeze(1)
        masks = torch.FloatTensor(masks).unsqueeze(1)
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_return = 0
        
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_return = reward[0] + self.gamma * running_return * mask[0]
            discounted_rewards.insert(0, running_return)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).unsqueeze(1)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        for _ in range(self.k_epochs):
            new_means, new_logstds, new_values = self.policy(states)
            dist = torch.distributions.Normal(new_means, torch.exp(new_logstds))
            new_logprobs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            entropy = dist.entropy().sum(dim=1, keepdim=True)
            
            # Calculate ratios
            ratios = torch.exp(new_logprobs - old_logprobs)
            
            # Calculate advantages
            advantages = discounted_rewards - values
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = (discounted_rewards - new_values).pow(2).mean()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

def train_ppo_navigation(episodes=1000):
    """Train PPO agent for navigation"""
    env = NavigationEnvironment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    scores = deque(maxlen=100)
    training_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_buffer = []
        total_reward = 0
        
        for t in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action from policy
            action, log_prob, value = agent.policy.get_action(state_tensor)
            action_np = action.cpu().data.numpy().flatten()
            
            # Take action
            next_state, reward, done, _, _ = env.step(action_np)
            
            # Store transition
            episode_buffer.append([state, action_np, reward, log_prob.item(), 
                                 value.item(), 1 - done])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update policy
        if len(episode_buffer) > 0:
            states = [transition[0] for transition in episode_buffer]
            actions = [transition[1] for transition in episode_buffer]
            rewards = [transition[2] for transition in episode_buffer]
            logprobs = [transition[3] for transition in episode_buffer]
            values = [transition[4] for transition in episode_buffer]
            masks = [transition[5] for transition in episode_buffer]
            
            agent.update(states, actions, rewards, logprobs, values, masks)
        
        scores.append(total_reward)
        training_scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores) if scores else 0
            print(f'Episode {episode}, Average Score: {avg_score:.2f}')
    
    return agent, training_scores
```

### Advanced Tasks
1. **Multi-Objective RL**: Implement reward shaping for multiple objectives
2. **Curriculum Learning**: Start with simple tasks and gradually increase difficulty
3. **Transfer Learning**: Apply knowledge from one environment to another

### Analysis Questions
- How did the DQN and PPO agents perform differently?
- What challenges arose when applying RL to continuous control?
- How could the reward function be improved?

## Exercise 3: Deep Learning for Robot Perception

### Objective
Implement deep learning models for robot perception tasks like object detection, classification, and segmentation.

### Simulation Setup
1. Create perception simulation environment:
```python
# perception_simulation.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

class PerceptionEnvironment:
    def __init__(self):
        # Define object classes for detection
        self.object_classes = ['cube', 'sphere', 'cylinder', 'cone', 'pyramid']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    def generate_synthetic_image(self, width=480, height=360, num_objects=3):
        """Generate synthetic images with 3D objects"""
        # Create background
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add noise for realism
        noise = np.random.normal(0, 5, (height, width, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        bboxes = []
        classes = []
        
        for _ in range(num_objects):
            # Random object parameters
            obj_class = random.choice(range(len(self.object_classes)))
            color = self.colors[obj_class]
            x = random.randint(30, width - 60)
            y = random.randint(30, height - 60)
            size = random.randint(20, 50)
            
            # Draw object based on class
            if self.object_classes[obj_class] == 'cube':
                # Draw square
                cv2.rectangle(image, (x - size//2, y - size//2), 
                             (x + size//2, y + size//2), color, thickness=-1)
            elif self.object_classes[obj_class] == 'sphere':
                # Draw circle
                cv2.circle(image, (x, y), size//2, color, thickness=-1)
            elif self.object_classes[obj_class] == 'cylinder':
                # Draw ellipse
                axes = (size//2, size//3)
                cv2.ellipse(image, (x, y), axes, 0, 0, 360, color, thickness=-1)
            elif self.object_classes[obj_class] == 'cone':
                # Draw triangle
                pts = np.array([(x, y - size//2), (x - size//2, y + size//2), (x + size//2, y + size//2)], np.int32)
                cv2.fillPoly(image, [pts], color)
            elif self.object_classes[obj_class] == 'pyramid':
                # Draw pyramid (square base with apex)
                pts = np.array([(x, y - size//2), (x - size//2, y + size//2), 
                               (x, y + size//3), (x + size//2, y + size//2)], np.int32)
                cv2.fillPoly(image, [pts], color)
            
            # Store bounding box
            bbox = [x - size//2, y - size//2, x + size//2, y + size//2]
            bboxes.append(bbox)
            classes.append(obj_class)
        
        return image, bboxes, classes
    
    def generate_dataset(self, num_samples=1000):
        """Generate a dataset of images with annotations"""
        images = []
        annotations = []  # List of (bbox, class) tuples for each image
        
        for _ in range(num_samples):
            image, bboxes, classes = self.generate_synthetic_image()
            images.append(image)
            annotations.append(list(zip(bboxes, classes)))
        
        return images, annotations

class SimpleObjectDetector(nn.Module):
    """Simple object detection model"""
    def __init__(self, num_classes=5, input_channels=3):
        super(SimpleObjectDetector, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # x, y, width, height (normalized)
        )
    
    def forward(self, x):
        features = self.features(x)
        
        class_logits = self.classifier(features)
        bbox_coords = self.bbox_regressor(features)
        
        return class_logits, bbox_coords

def train_object_detector(model, dataloader, num_epochs=10):
    """Train the object detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()  # Huber loss for bounding box regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for images, class_targets, bbox_targets in dataloader:
            images = images.to(device)
            class_targets = class_targets.to(device)
            bbox_targets = bbox_targets.to(device)
            
            optimizer.zero_grad()
            
            class_logits, bbox_preds = model(images)
            
            # Calculate losses
            class_loss = criterion_cls(class_logits, class_targets)
            bbox_loss = criterion_bbox(bbox_preds, bbox_targets)
            
            # Total loss (with balancing weights)
            total_batch_loss = class_loss + 0.5 * bbox_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def create_detection_dataloader(images, annotations, batch_size=8):
    """Create a data loader for detection training"""
    from torch.utils.data import DataLoader, Dataset
    
    class DetectionDataset(Dataset):
        def __init__(self, images, annotations):
            self.images = images
            self.annotations = annotations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            annotations = self.annotations[idx]
            
            # Convert to tensor
            image_tensor = self.transform(Image.fromarray(image.astype('uint8')))
            
            # For this simple example, use the first annotation
            # In a real implementation, you'd handle multiple objects
            if annotations:
                bbox, class_idx = annotations[0]
                # Normalize bounding box coordinates
                h, w = image.shape[:2]
                norm_bbox = torch.FloatTensor([
                    bbox[0] / w,  # x_min
                    bbox[1] / h,  # y_min
                    bbox[2] / w,  # x_max
                    bbox[3] / h   # y_max
                ])
            else:
                # Default: no object detected
                class_idx = 0
                norm_bbox = torch.FloatTensor([0, 0, 0, 0])
            
            return image_tensor, torch.LongTensor([class_idx]), norm_bbox
    
    dataset = DetectionDataset(images, annotations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def visualize_detections(image, bboxes, classes, predicted_classes=None):
    """Visualize detection results"""
    img_copy = image.copy()
    
    for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
        color = PerceptionEnvironment().colors[cls]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = PerceptionEnvironment().object_classes[cls]
        if predicted_classes is not None:
            label += f' (pred: {PerceptionEnvironment().object_classes[predicted_classes[i]]})'
        
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    return img_copy

def run_perception_exercise():
    """Run the perception simulation exercise"""
    # Create environment and generate data
    env = PerceptionEnvironment()
    images, annotations = env.generate_dataset(num_samples=500)
    
    # Create data loader
    dataloader = create_detection_dataloader(images, annotations, batch_size=4)
    
    # Create and train model
    model = SimpleObjectDetector(num_classes=5)
    print("Training object detection model...")
    train_object_detector(model, dataloader, num_epochs=3)  # Few epochs for demonstration
    
    # Test detection on sample images
    print("Testing detection on sample images...")
    test_images = []
    for i in range(3):
        img, bboxes, classes = env.generate_synthetic_image()
        test_images.append((img, bboxes, classes))
    
    # Visualize results
    for i, (test_img, true_bboxes, true_classes) in enumerate(test_images):
        result_img = visualize_detections(test_img, true_bboxes, true_classes)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detection Results - Sample {i+1}')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    run_perception_exercise()
```

### Implementation Tasks
1. **Object Detection Model**: Implement and train an object detection model
2. **Performance Evaluation**: Test the model on various scenarios
3. **Real-World Adaptation**: Modify the model for real-world images

### Advanced Tasks
1. **Multi-Task Learning**: Train a single model for detection, classification, and segmentation
2. **Adversarial Training**: Improve robustness with adversarial examples
3. **Few-Shot Learning**: Train models that can generalize to new objects with limited examples

### Analysis Questions
- How well did the detection model perform?
- What were the main challenges in training a detection model?
- How could the synthetic data generation be improved?

## Exercise 4: Integration and Performance Analysis

### Objective
Integrate perception, learning, and control systems, and analyze overall performance and safety considerations.

### Implementation Tasks
1. **Integrated System Architecture**: Combine perception, learning, and control
2. **Performance Metrics**: Define and measure system performance
3. **Safety Analysis**: Identify and mitigate safety risks

```python
# integrated_system_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

class PerceptionController:
    """Integrates perception and control"""
    def __init__(self, perception_model, control_model):
        self.perception_model = perception_model
        self.control_model = control_model
        self.action_history = deque(maxlen=10)
        
    def process_perception(self, sensor_input):
        """Process sensor input through perception model"""
        # In a real implementation, this would run the perception model
        # For simulation, return mock detections
        detections = {
            'objects': [('cube', (100, 100, 150, 150), 0.9)],
            'obstacles': [((50, 50, 80, 80), 0.8)],
            'target': (200, 200, 250, 250),  # Target bounding box
            'confidence': 0.9  # Overall confidence
        }
        return detections
    
    def generate_control_action(self, detections, robot_state):
        """Generate control action based on detections and state"""
        if not detections['objects'] and not detections['obstacles']:
            # No detections - move forward cautiously
            return np.array([0.2, 0.0])  # Move forward slowly
        
        # Simple navigation based on detections
        if detections['objects']:
            # Move towards first detected object
            obj_bbox = detections['objects'][0][1]
            obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
            image_center_x = 320  # Assuming 640x480 image
            
            angular_correction = (image_center_x - obj_center_x) / 100  # Proportional control
            linear_speed = 0.3 if detections['confidence'] > 0.7 else 0.1
            
            action = np.array([linear_speed, angular_correction])
        else:
            # No objects detected, but obstacles present
            # Avoid obstacles
            closest_obstacle = min(detections['obstacles'], key=lambda x: x[1])
            obstacle_bbox = closest_obstacle[0]
            obstacle_center_x = (obstacle_bbox[0] + obstacle_bbox[2]) / 2
            image_center_x = 320
            
            # Turn away from obstacle
            turn_direction = 1.0 if obstacle_center_x < image_center_x else -1.0
            action = np.array([0.1, turn_direction * 0.5])  # Move forward and turn away
        
        # Limit actions to safe ranges
        action[0] = np.clip(action[0], -1.0, 1.0)  # Linear velocity
        action[1] = np.clip(action[1], -1.0, 1.0)  # Angular velocity
        
        self.action_history.append(action)
        return action

class SafetyMonitor:
    """Monitors system safety in integrated system"""
    def __init__(self):
        self.safety_thresholds = {
            'proximity': 0.3,  # Minimum distance to obstacles (meters)
            'acceleration': 2.0,  # Maximum linear acceleration (m/s²)
            'angular_acceleration': 3.0,  # Maximum angular acceleration (rad/s²)
            'temperature': 80.0,  # Maximum component temperature (°C)
            'battery_low': 15.0   # Minimum battery level (%)
        }
        self.safety_violations = 0
        self.last_action = np.array([0.0, 0.0])
        self.last_time = time.time()
    
    def update_safety_state(self, robot_state, sensor_data):
        """Update safety monitoring with new state and sensor data"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Check various safety conditions
        violations = []
        
        # Check proximity to obstacles
        if 'obstacle_distances' in sensor_data:
            min_distance = min(sensor_data['obstacle_distances'].values())
            if min_distance < self.safety_thresholds['proximity']:
                violations.append(f"Obstacle too close: {min_distance:.2f}m < {self.safety_thresholds['proximity']}m")
        
        # Check acceleration limits
        if dt > 0 and len(self.last_action) > 0:
            linear_accel = abs(robot_state.get('linear_vel', 0) - self.last_action[0]) / dt
            angular_accel = abs(robot_state.get('angular_vel', 0) - self.last_action[1]) / dt
            
            if linear_accel > self.safety_thresholds['acceleration']:
                violations.append(f"Linear acceleration too high: {linear_accel:.2f} > {self.safety_thresholds['acceleration']}")
            
            if angular_accel > self.safety_thresholds['angular_acceleration']:
                violations.append(f"Angular acceleration too high: {angular_accel:.2f} > {self.safety_thresholds['angular_acceleration']}")
        
        self.last_action = np.array([robot_state.get('linear_vel', 0), robot_state.get('angular_vel', 0)])
        
        # Check component temperatures if available
        if 'temperatures' in sensor_data:
            for component, temp in sensor_data['temperatures'].items():
                if temp > self.safety_thresholds['temperature']:
                    violations.append(f"{component} temperature too high: {temp:.1f}°C > {self.safety_thresholds['temperature']}°C")
        
        # Check battery level
        if robot_state.get('battery_level', 100) < self.safety_thresholds['battery_low']:
            violations.append(f"Battery too low: {robot_state.get('battery_level', 100):.1f}% < {self.safety_thresholds['battery_low']}%")
        
        return len(violations) == 0, violations

class IntegratedRobotSystem:
    """Main integrated system combining perception, learning, and control"""
    def __init__(self):
        self.perception_controller = PerceptionController(None, None)
        self.safety_monitor = SafetyMonitor()
        self.performance_metrics = {
            'execution_time': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'safety_violations': deque(maxlen=100)
        }
        self.active = True
    
    def execute_cycle(self, sensor_input, robot_state):
        """Execute one cycle of the integrated system"""
        start_time = time.time()
        
        # 1. Process perception
        detections = self.perception_controller.process_perception(sensor_input)
        
        # 2. Generate control action
        action = self.perception_controller.generate_control_action(detections, robot_state)
        
        # 3. Safety check
        is_safe, violations = self.safety_monitor.update_safety_state(robot_state, sensor_input)
        
        if not is_safe:
            print(f"Safety violations detected: {violations}")
            self.safety_monitor.safety_violations += 1
            
            # Override control action for safety
            action = np.array([0.0, 0.0])  # Emergency stop
        
        # 4. Calculate performance metrics
        execution_time = time.time() - start_time
        self.performance_metrics['execution_time'].append(execution_time)
        
        # Simulate success/failure based on various factors
        success = self.evaluate_success(action, detections, robot_state)
        self.performance_metrics['success_rate'].append(1 if success else 0)
        self.performance_metrics['safety_violations'].append(1 if not is_safe else 0)
        
        return action, is_safe, execution_time
    
    def evaluate_success(self, action, detections, robot_state):
        """Evaluate success of the action"""
        # Simple success criteria:
        # - Action is within bounds
        # - No safety violations
        # - Makes progress toward target if one exists
        if not detections.get('target'):
            return True  # No target, just don't crash
        
        # If there's a target, check if we're moving toward it
        target_in_view = detections.get('target') is not None
        action_appropriate = np.all(np.abs(action) <= 1.0)  # Action bounds
        
        return target_in_view and action_appropriate
    
    def get_performance_summary(self):
        """Get performance metrics"""
        if not self.performance_metrics['execution_time']:
            return "No data collected yet"
        
        summary = {
            'avg_execution_time': np.mean(self.performance_metrics['execution_time']),
            'success_rate': np.mean(self.performance_metrics['success_rate']),
            'safety_violation_rate': np.mean(self.performance_metrics['safety_violations']),
            'total_cycles': len(self.performance_metrics['execution_time'])
        }
        return summary

def run_integration_analysis():
    """Run complete integration analysis"""
    system = IntegratedRobotSystem()
    
    # Simulate sensor data
    sensor_inputs = []
    robot_states = []
    
    for i in range(100):  # Run 100 cycles
        # Simulate sensor input
        sensor_data = {
            'obstacle_distances': {
                'front': np.random.uniform(0.3, 5.0),
                'left': np.random.uniform(0.3, 5.0),
                'right': np.random.uniform(0.3, 5.0)
            },
            'temperatures': {
                'motor_1': np.random.uniform(20, 40),
                'motor_2': np.random.uniform(20, 40),
                'cpu': np.random.uniform(30, 50)
            },
            'battery_level': 100 - i * 0.1  # Decreasing battery
        }
        
        # Simulate robot state
        robot_state = {
            'x': np.random.uniform(-5, 5),
            'y': np.random.uniform(-5, 5),
            'theta': np.random.uniform(-np.pi, np.pi),
            'linear_vel': np.random.uniform(-1, 1),
            'angular_vel': np.random.uniform(-1, 1),
            'battery_level': sensor_data['battery_level']
        }
        
        # Execute one cycle
        action, is_safe, exec_time = system.execute_cycle(sensor_data, robot_state)
        
        if i % 20 == 0:  # Print every 20 cycles
            print(f"Cycle {i}: Action={action}, Safe={is_safe}, Time={exec_time:.4f}s")
    
    # Print performance summary
    summary = system.get_performance_summary()
    print("\nPerformance Summary:")
    print(f"Average execution time: {summary['avg_execution_time']:.4f}s")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Safety violation rate: {summary['safety_violation_rate']:.2%}")
    print(f"Total cycles executed: {summary['total_cycles']}")
    
    return system

def plot_integration_performance(system):
    """Plot integration performance metrics"""
    metrics = system.performance_metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Execution time
    if metrics['execution_time']:
        axes[0, 0].plot(metrics['execution_time'])
        axes[0, 0].set_title('Execution Time per Cycle')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].grid(True)
    
    # Success rate (moving average)
    if metrics['success_rate']:
        success_ma = np.convolve(metrics['success_rate'], 
                               np.ones(10)/10, mode='valid')
        axes[0, 1].plot(success_ma)
        axes[0, 1].set_title('Success Rate (Moving Average, 10-cycle window)')
        axes[0, 1].set_xlabel('Cycle')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True)
    
    # Safety violations
    if metrics['safety_violations']:
        axes[1, 0].plot(metrics['safety_violations'])
        axes[1, 0].set_title('Safety Violations per Cycle')
        axes[1, 0].set_xlabel('Cycle')
        axes[1, 0].set_ylabel('Violation (0/1)')
        axes[1, 0].grid(True)
    
    # Cumulative safety violations
    if metrics['safety_violations']:
        cumulative_violations = np.cumsum(metrics['safety_violations'])
        axes[1, 1].plot(cumulative_violations)
        axes[1, 1].set_title('Cumulative Safety Violations')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Total Violations')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Run the complete integration analysis
if __name__ == '__main__':
    system = run_integration_analysis()
    plot_integration_performance(system)
```

### Advanced Tasks
1. **Real-Time Performance**: Optimize system for real-time constraints
2. **Robustness Testing**: Test system under various failure conditions
3. **Learning-Enabled Components**: Integrate machine learning for perception and control

### Analysis Questions
- How did the integrated system perform compared to individual components?
- What were the main bottlenecks in the system?
- How could safety be further improved?

## Simulation Tools and Resources

### Deep Learning Frameworks
- **PyTorch**: For building and training neural networks
- **TensorFlow**: Alternative deep learning framework
- **Stable-Baselines3**: For reinforcement learning algorithms

### Visualization Tools
- **Matplotlib**: For plotting and analysis
- **OpenCV**: For computer vision tasks
- **TensorBoard**: For training visualization

### Robotics Simulation
- **PyBullet**: Physics simulation environment
- **Gazebo**: Robotic simulation platform
- **AirSim**: High-fidelity simulation for drones and cars

## Troubleshooting Common Issues

### Performance Problems
- **Slow Training**: Use GPU acceleration and optimize batch sizes
- **Overfitting**: Add regularization and use more diverse training data
- **Memory Issues**: Use mini-batches and clear unused variables

### Integration Challenges
- **Architecture Mismatch**: Ensure consistent input/output dimensions
- **Timing Issues**: Implement proper buffering and synchronization
- **Data Format Inconsistencies**: Standardize data formats across modules

### Safety and Stability
- **Unstable Control**: Implement safety limits and monitoring
- **Unexpected Behaviors**: Add comprehensive testing and validation
- **Failure Modes**: Implement graceful degradation strategies

These simulation exercises provide a comprehensive framework for understanding AI systems in robotics, from individual components to integrated systems with safety considerations.