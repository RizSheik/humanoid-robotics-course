---
title: Practical Lab - AI Robot Brain Implementation
description: Hands-on lab exercises implementing core concepts of AI systems for robotics
sidebar_position: 102
---

# Practical Lab - AI Robot Brain Implementation

## Lab Overview

This lab provides hands-on experience implementing and experimenting with AI systems for robotics. Students will work with real datasets, implement various machine learning and deep learning algorithms, and create functional AI systems for robotic tasks. The lab emphasizes practical implementation skills while reinforcing theoretical concepts of AI in robotics applications.

## Lab Objectives

By completing this lab, students will be able to:
- Implement machine learning algorithms for robotic control and perception
- Create and train neural networks for robotic applications
- Apply reinforcement learning techniques to robotic tasks
- Evaluate AI system performance in robotic contexts
- Implement safety mechanisms for AI-based robotic systems

## Prerequisites and Setup

### Software Requirements
- Python 3.8+ with libraries: torch, torchvision, stable-baselines3, gym, numpy, matplotlib
- ROS 2 Humble Hawksbill with robotics packages
- CUDA-compatible GPU (optional but recommended for deep learning)
- Git for repository management

### Hardware Requirements
- Computer with GPU (NVIDIA recommended) or access to cloud GPU instances
- Internet connection for downloading models and datasets
- Simulation environment (Gazebo or PyBullet)

### Setup Commands
```bash
# Create project environment
python -m venv ai_robot_lab
source ai_robot_lab/bin/activate  # On Windows: ai_robot_lab\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3[extra] gymnasium[box2d] opencv-python matplotlib tensorboard
pip install pybullet numpy pandas scikit-learn

# Clone robotics simulation environment
git clone https://github.com/robotics-ai-course/robot-sim-env.git
cd robot-sim-env && pip install -e .
```

## Lab Exercise 1: Supervised Learning for Robot Control

### Objective
Implement supervised learning algorithms for robot control tasks, specifically for mapping sensor inputs to motor commands.

### Steps
1. Create a dataset of sensor readings and corresponding motor commands:
```python
# sensor_motor_dataset.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RobotSensorMotorDataset:
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.data, self.targets = self.generate_data()
    
    def generate_data(self):
        """
        Generate synthetic data simulating robot sensor-actuator mapping
        Sensors: 10 values (e.g., distances from 10 range sensors)
        Actions: 2 values (e.g., linear and angular velocity)
        """
        # Generate sensor inputs
        sensors = np.random.uniform(0.1, 10.0, (self.num_samples, 10))  # Distance readings
        
        # Generate target actions based on sensor readings
        # Simple policy: move away from obstacles, follow wall on right
        actions = np.zeros((self.num_samples, 2))
        
        for i in range(self.num_samples):
            # Extract relevant sensor information
            front_dist = sensors[i, 0]  # Front sensor
            right_dist = sensors[i, 1]  # Right sensor
            left_dist = sensors[i, 2]   # Left sensor
            
            # Simple control policy
            linear_vel = min(1.0, front_dist / 2.0)  # Slow down when close to front obstacle
            angular_vel = (right_dist - left_dist) * 0.2  # Turn toward closer wall
            
            actions[i] = [linear_vel, angular_vel]
        
        return torch.FloatTensor(sensors), torch.FloatTensor(actions)

class SensorMotorNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super(SensorMotorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_supervised_model():
    # Create dataset
    dataset = RobotSensorMotorDataset(num_samples=10000)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset.data))
    val_size = len(dataset.data) - train_size
    
    train_data, val_data = torch.utils.data.random_split(
        TensorDataset(dataset.data, dataset.targets), 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    # Initialize model
    model = SensorMotorNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for sensors, actions in train_loader:
            optimizer.zero_grad()
            outputs = model(sensors)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sensors, actions in val_loader:
                outputs = model(sensors)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}')
    
    return model

if __name__ == '__main__':
    trained_model = train_supervised_model()
    
    # Test with new data
    test_sensors = torch.randn(5, 10)  # 5 test samples
    with torch.no_grad():
        predictions = trained_model(test_sensors)
    
    print("Test predictions (linear_vel, angular_vel):")
    for i in range(5):
        print(f"Sample {i}: {predictions[i].numpy()}")
```

2. Implement data preprocessing and normalization:
```python
# preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        """Fit scaler and transform data"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        data_scaled = self.scaler.fit_transform(data)
        return torch.FloatTensor(data_scaled)
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        data_scaled = self.scaler.transform(data)
        return torch.FloatTensor(data_scaled)
    
    def inverse_transform(self, data):
        """Inverse transform to original scale"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        original_data = self.scaler.inverse_transform(data)
        return torch.FloatTensor(original_data)

def normalize_robot_data(sensors, actions):
    """Normalize sensor and action data for robot learning"""
    sensor_preprocessor = DataPreprocessor()
    action_preprocessor = DataPreprocessor()
    
    # Normalize sensors and actions
    normalized_sensors = sensor_preprocessor.fit_transform(sensors)
    normalized_actions = action_preprocessor.fit_transform(actions)
    
    return normalized_sensors, normalized_actions, sensor_preprocessor, action_preprocessor
```

### Deliverables
- Working supervised learning model for sensor-motor mapping
- Training and validation loss curves
- Model performance evaluation
- Proper data preprocessing pipeline

## Lab Exercise 2: Deep Reinforcement Learning for Robot Navigation

### Objective
Implement deep reinforcement learning algorithms for robot navigation tasks, specifically using Deep Q-Networks (DQN) or Proximal Policy Optimization (PPO).

### Steps
1. Set up a navigation environment simulation:
```python
# navigation_env.py
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleNavigationEnv(gym.Env):
    def __init__(self):
        super(SimpleNavigationEnv, self).__init__()
        
        # Define action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Define observation space: position, goal direction, obstacle distances
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -np.pi, 0, 0, 0]),  # x, y, theta, front_dist, left_dist, right_dist
            high=np.array([10, 10, np.pi, 10, 10, 10]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self):
        # Random start position
        self.robot_pos = np.random.uniform(-8, 8, size=2)
        
        # Random goal position
        self.goal_pos = np.random.uniform(-9, 9, size=2)
        
        # Random obstacles
        self.obstacles = np.random.uniform(-9, 9, size=(3, 2))
        
        # Initial robot orientation
        self.robot_theta = np.random.uniform(-np.pi, np.pi)
        
        return self._get_observation()
    
    def step(self, action):
        # Extract action values
        linear_vel, angular_vel = action
        
        # Update robot position (simple kinematic model)
        dt = 0.1
        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * dt
        self.robot_theta += angular_vel * dt
        
        # Calculate distances to obstacles
        front_dist = np.min([np.linalg.norm(self.robot_pos - obs) for obs in self.obstacles])
        left_dist = np.min([np.linalg.norm(self.robot_pos + np.array([np.cos(self.robot_theta + np.pi/4), 
                                                                       np.sin(self.robot_theta + np.pi/4)])*0.5 - obs) 
                            for obs in self.obstacles])
        right_dist = np.min([np.linalg.norm(self.robot_pos + np.array([np.cos(self.robot_theta - np.pi/4), 
                                                                        np.sin(self.robot_theta - np.pi/4)])*0.5 - obs) 
                             for obs in self.obstacles])
        
        # Calculate goal direction
        goal_dir = self.goal_pos - self.robot_pos
        goal_distance = np.linalg.norm(goal_dir)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0]) - self.robot_theta
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))  # Normalize angle
        
        # Calculate reward
        reward = -0.01  # Small time penalty
        reward -= goal_distance * 0.1  # Encourage moving toward goal
        
        if goal_distance < 0.5:  # Reached goal
            reward += 10.0
            done = True
        elif front_dist < 0.3 or left_dist < 0.3 or right_dist < 0.3:  # Collision
            reward -= 5.0
            done = True
        elif np.abs(self.robot_pos[0]) > 10 or np.abs(self.robot_pos[1]) > 10:  # Out of bounds
            reward -= 2.0
            done = True
        else:
            done = False
        
        # Episode truncation for safety
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        else:
            self.step_count += 1
            
        if self.step_count > 200:  # Truncate after 200 steps
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        # Calculate goal direction and distances
        goal_dir = self.goal_pos - self.robot_pos
        goal_distance = np.linalg.norm(goal_dir)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0]) - self.robot_theta
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        
        # Calculate distances to obstacles
        front_dist = max(0.1, np.min([np.linalg.norm(self.robot_pos - obs) for obs in self.obstacles]))
        left_dist = max(0.1, np.min([np.linalg.norm(self.robot_pos + 
                                                     np.array([np.cos(self.robot_theta + np.pi/4), 
                                                              np.sin(self.robot_theta + np.pi/4)])*0.5 - obs) 
                                     for obs in self.obstacles]))
        right_dist = max(0.1, np.min([np.linalg.norm(self.robot_pos + 
                                                      np.array([np.cos(self.robot_theta - np.pi/4), 
                                                               np.sin(self.robot_theta - np.pi/4)])*0.5 - obs) 
                                      for obs in self.obstacles]))
        
        return np.array([self.robot_pos[0], self.robot_pos[1], self.robot_theta, 
                         goal_angle, goal_distance, front_dist, left_dist, right_dist])

def visualize_environment(env):
    """Visualize the navigation environment"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot robot position
    ax.plot(env.robot_pos[0], env.robot_pos[1], 'ro', markersize=10, label='Robot')
    
    # Plot goal position
    ax.plot(env.goal_pos[0], env.goal_pos[1], 'go', markersize=10, label='Goal')
    
    # Plot obstacles
    for obs in env.obstacles:
        ax.plot(obs[0], obs[1], 'ks', markersize=8, label='Obstacle' if obs[0] == env.obstacles[0][0] else "")
    
    # Plot robot orientation
    ax.arrow(env.robot_pos[0], env.robot_pos[1], 
             0.5 * np.cos(env.robot_theta), 0.5 * np.sin(env.robot_theta),
             head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Navigation Environment')
    
    plt.show()
```

2. Implement DQN for navigation:
```python
# dqn_navigation.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_navigation(episodes=1000):
    env = SimpleNavigationEnv()
    
    state_size = env.observation_space.shape[0]
    # For this example, discretize the continuous action space
    action_size = 9  # Discretize into 3 linear vel * 3 angular vel options
    
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for t in range(200):  # Max 200 steps per episode
            action_idx = agent.act(state)
            
            # Convert discrete action to continuous
            linear_vels = [-0.5, 0.0, 0.5]
            angular_vels = [-0.3, 0.0, 0.3]
            
            linear_vel = linear_vels[action_idx // 3]
            angular_vel = angular_vels[action_idx % 3]
            
            action = np.array([linear_vel, angular_vel])
            
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action_idx, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores

def plot_training_progress(scores):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg)
    plt.title('Moving Average of Scores (100-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    agent, scores = train_dqn_navigation(episodes=500)
    plot_training_progress(scores)
```

3. Implement PPO as an alternative approach:
```python
# ppo_navigation.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (outputs mean and std for continuous actions)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic network (outputs value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        
        # Actor
        mean = torch.tanh(self.actor_mean(features)) * 2.0  # Scale to action space
        std = F.softplus(self.actor_std(features)) + 1e-5  # Add small value to std
        
        # Critic
        value = self.critic(state)
        
        return mean, std, value
    
    def get_action(self, state):
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        # Log probability of the action
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, states, actions, rewards, logprobs, values, masks):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        logprobs = torch.FloatTensor(logprobs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_reward = reward + self.gamma * running_reward * mask
            discounted_rewards.insert(0, running_reward)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        # Update policy K epochs
        for _ in range(self.k_epochs):
            # Get current policy's action and value
            new_means, new_stds, new_values = self.policy(states)
            dist = torch.distributions.Normal(new_means, new_stds)
            new_logprobs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate ratios
            ratios = torch.exp(new_logprobs - logprobs.detach())
            
            # Calculate surrogates
            advantages = discounted_rewards - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(new_values.squeeze(), discounted_rewards)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train_ppo_navigation(episodes=1000):
    env = SimpleNavigationEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    # Storage for episode data
    states = []
    actions = []
    rewards = []
    logprobs = []
    values = []
    masks = []
    
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_steps = 0
        
        while True:
            # Get action from policy
            action, logprob, value = agent.policy.get_action(torch.FloatTensor(state).unsqueeze(0))
            action = action.cpu().data.numpy().flatten()
            
            next_state, reward, done, _ = env.step(action)
            
            # Store the transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            logprobs.append(logprob.item())
            values.append(value.item())
            masks.append(1 - done)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if done or episode_steps >= 200:  # Max steps
                scores.append(total_reward)
                break
        
        # Update the policy
        if episode % 10 == 0 and len(states) > 0:
            agent.update(states, actions, rewards, logprobs, values, masks)
            
            # Clear storage for next batch
            states = []
            actions = []
            rewards = []
            logprobs = []
            values = []
            masks = []
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    return agent, scores

if __name__ == '__main__':
    agent, scores = train_ppo_navigation(episodes=500)
    plot_training_progress(scores)  # Using the same plotting function from DQN
```

### Deliverables
- Working DQN implementation for robot navigation
- Working PPO implementation as alternative
- Comparison of both approaches
- Training curves and performance metrics

## Lab Exercise 3: Vision-Based Robot Control

### Objective
Implement computer vision and deep learning techniques for robot perception and control, specifically for object detection and manipulation.

### Steps
1. Create a vision-based control system:
```python
# vision_control.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleObjectDetector, self).__init__()
        
        # Simple CNN for object detection
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)  # x, y, width, height
        )
    
    def forward(self, x):
        features = self.features(x)
        
        class_logits = self.classifier(features)
        bbox_coords = self.bbox_regressor(features)
        
        return class_logits, bbox_coords

class VisionBasedController:
    def __init__(self, detector_model_path=None):
        # Initialize object detector
        self.detector = SimpleObjectDetector(num_classes=5)  # 5 object classes
        
        if detector_model_path:
            self.detector.load_state_dict(torch.load(detector_model_path))
        
        # Define object classes
        self.classes = ["red_cube", "blue_sphere", "green_cylinder", "yellow_block", "purple_pyramid"]
        
        # Transformation for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_objects(self, image):
        """Detect objects in an image"""
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Run detection
        with torch.no_grad():
            class_logits, bbox_coords = self.detector(input_tensor)
            
            # Get class predictions
            class_probs = torch.softmax(class_logits, dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            confidence = class_probs[0, predicted_class].item()
            
            # Get bounding box coordinates (these would be actual bbox coords in a real detector)
            bbox = bbox_coords[0].numpy()
        
        return self.classes[predicted_class], confidence, bbox
    
    def calculate_control_action(self, detected_objects, target_object="red_cube"):
        """Calculate robot control action based on detected objects"""
        if not detected_objects:
            return np.array([0.0, 0.0])  # No movement if no objects detected
        
        # Find target object
        target_found = False
        for obj_class, confidence, bbox in detected_objects:
            if obj_class == target_object and confidence > 0.5:
                target_found = True
                # Calculate center of bounding box
                center_x = bbox[0] + bbox[2] / 2  # x + width/2
                center_y = bbox[1] + bbox[3] / 2  # y + height/2
                
                # Calculate control action based on object position
                # Simple proportional controller
                image_center_x = 320  # Assuming 640x480 image
                image_center_y = 240  # Assuming 640x480 image
                
                # Calculate error from center
                x_error = center_x - image_center_x
                y_error = center_y - image_center_y
                
                # Convert to control commands
                linear_vel = max(0.0, 1.0 - np.abs(y_error) / 300)  # Move forward if object is close to center vertically
                angular_vel = -x_error / 300  # Turn to center object horizontally
                
                # Limit velocities
                linear_vel = np.clip(linear_vel, 0.0, 1.0)
                angular_vel = np.clip(angular_vel, -1.0, 1.0)
                
                break
        
        if not target_found:
            # No target found, maybe search for it
            return np.array([0.0, 0.3])  # Turn slowly to search
        
        return np.array([linear_vel, angular_vel])

def simulate_vision_control():
    """Simulate vision-based control in a simple environment"""
    controller = VisionBasedController()
    
    # Simulate a series of "images" with objects
    actions = []
    positions = []
    
    # Initial robot position (x, y, theta)
    robot_pos = np.array([0.0, 0.0, 0.0])
    
    for step in range(100):
        # Simulate "capturing" an image and detecting objects
        # In a real scenario, this would come from a camera
        if step < 20:
            # Object not in view initially
            detected_objects = []
        elif step < 50:
            # Object comes into view off-center
            bbox = np.array([100, 150, 100, 100])  # x, y, width, height
            detected_objects = [("red_cube", 0.8, bbox)]
        else:
            # Object is centered
            bbox = np.array([270, 200, 100, 100])  # x, y, width, height
            detected_objects = [("red_cube", 0.9, bbox)]
        
        # Calculate control action
        action = controller.calculate_control_action(detected_objects)
        
        # Simulate robot movement (simple kinematic model)
        dt = 0.1
        linear_vel, angular_vel = action
        robot_pos[0] += linear_vel * np.cos(robot_pos[2]) * dt
        robot_pos[1] += linear_vel * np.sin(robot_pos[2]) * dt
        robot_pos[2] += angular_vel * dt
        
        actions.append(action)
        positions.append(robot_pos.copy())
    
    return np.array(actions), np.array(positions)

def plot_vision_control_results(actions, positions):
    """Plot results of vision-based control"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot linear velocity over time
    axes[0, 0].plot(actions[:, 0])
    axes[0, 0].set_title('Linear Velocity Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Linear Velocity')
    axes[0, 0].grid(True)
    
    # Plot angular velocity over time
    axes[0, 1].plot(actions[:, 1])
    axes[0, 1].set_title('Angular Velocity Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Angular Velocity')
    axes[0, 1].grid(True)
    
    # Plot robot trajectory
    axes[1, 0].plot(positions[:, 0], positions[:, 1])
    axes[1, 0].set_title('Robot Trajectory')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    axes[1, 0].grid(True)
    axes[1, 0].axis('equal')
    
    # Plot robot orientation over time
    axes[1, 1].plot(positions[:, 2])
    axes[1, 1].set_title('Robot Orientation Over Time')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Orientation (radians)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    actions, positions = simulate_vision_control()
    plot_vision_control_results(actions, positions)
```

2. Implement a more sophisticated vision system with real image processing:
```python
# advanced_vision.py
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self, model_name='resnet18', pretrained=True):
        # Load a pre-trained ResNet model
        self.model = getattr(models, model_name)(pretrained=pretrained)
        
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Transformation for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract features from an image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        input_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features.view(features.size(0), -1)
        
        return features

class ObjectLocalizer(nn.Module):
    def __init__(self, feature_dim=512):
        super(ObjectLocalizer, self).__init__()
        
        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # bbox: x, y, width, height
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 object classes
        )
    
    def forward(self, features):
        bbox = torch.sigmoid(self.localization_head(features))  # Bounding box (0-1 normalized)
        class_logits = self.classification_head(features)
        return bbox, class_logits

class AdvancedVisionController:
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_extractor = FeatureExtractor().to(device)
        self.object_detector = ObjectLocalizer().to(device)
        
        # Object classes
        self.classes = ["red_cube", "blue_sphere", "green_cylinder", "yellow_block", "purple_pyramid"]
    
    def process_frame(self, frame):
        """Process a single video frame"""
        # Extract features
        features = self.feature_extractor.extract_features(frame)
        features = features.to(self.device)
        
        # Detect objects
        bbox, class_logits = self.object_detector(features)
        
        # Convert to probabilities
        class_probs = torch.softmax(class_logits, dim=1)
        predicted_class_idx = torch.argmax(class_probs, dim=1).item()
        confidence = class_probs[0, predicted_class_idx].item()
        
        # Convert bbox from normalized coordinates to image coordinates
        # Assuming original image size is 640x480
        img_h, img_w = frame.shape[:2]
        bbox_coords = bbox[0].cpu().numpy()
        x = int(bbox_coords[0] * img_w)
        y = int(bbox_coords[1] * img_h)
        width = int(bbox_coords[2] * img_w)
        height = int(bbox_coords[3] * img_h)
        
        return {
            'class': self.classes[predicted_class_idx],
            'confidence': confidence,
            'bbox': (x, y, width, height)
        }
    
    def control_from_vision(self, frame, target_class="red_cube"):
        """Generate control commands based on vision input"""
        detection = self.process_frame(frame)
        
        if detection['class'] == target_class and detection['confidence'] > 0.7:
            x, y, w, h = detection['bbox']
            
            # Calculate object center
            obj_center_x = x + w / 2
            obj_center_y = y + h / 2
            
            # Image center
            img_center_x = frame.shape[1] / 2
            img_center_y = frame.shape[0] / 2
            
            # Calculate error
            x_error = obj_center_x - img_center_x
            y_error = obj_center_y - img_center_y
            
            # Generate control commands
            linear_vel = max(0.0, 1.0 - abs(y_error) / 200)
            angular_vel = -x_error / 400
            
            return np.clip(linear_vel, 0.0, 1.0), np.clip(angular_vel, -1.0, 1.0)
        else:
            # Turn slowly to find target
            return 0.0, 0.2  # Move forward slowly and turn
```

### Deliverables
- Working vision-based object detection system
- Vision-to-control pipeline implementation
- Performance evaluation of the vision system
- Comparison with traditional control methods

## Lab Exercise 4: Integration and Safety

### Objective
Integrate multiple AI components into a complete robotic system and implement safety mechanisms.

### Steps
1. Create an integrated system with safety checks:
```python
# integrated_system.py
import threading
import time
import numpy as np

class SafetyMonitor:
    def __init__(self):
        self.safety_violations = 0
        self.emergency_stop = False
        self.safety_thresholds = {
            'proximity': 0.3,  # Stop if obstacle within 0.3m
            'velocity': 1.0,    # Max linear velocity
            'angular_velocity': 2.0,  # Max angular velocity
            'temperature': 80.0  # Max component temperature (degrees C)
        }
        self.component_temperatures = {'motor_1': 25.0, 'motor_2': 25.0, 'cpu': 35.0}
        self.obstacle_distances = {'front': 10.0, 'left': 10.0, 'right': 10.0}
    
    def update_sensor_data(self, sensor_data):
        """Update sensor readings"""
        if 'obstacle_distances' in sensor_data:
            self.obstacle_distances.update(sensor_data['obstacle_distances'])
        if 'temperatures' in sensor_data:
            self.component_temperatures.update(sensor_data['temperatures'])
    
    def check_safety(self, action):
        """Check if the proposed action is safe"""
        linear_vel, angular_vel = action
        
        # Check velocity limits
        if abs(linear_vel) > self.safety_thresholds['velocity']:
            return False, f"Linear velocity {linear_vel} exceeds limit {self.safety_thresholds['velocity']}"
        
        if abs(angular_vel) > self.safety_thresholds['angular_velocity']:
            return False, f"Angular velocity {angular_vel} exceeds limit {self.safety_thresholds['angular_velocity']}"
        
        # Check for proximity violations
        if self.obstacle_distances.get('front', 10.0) < self.safety_thresholds['proximity']:
            return False, f"Obstacle too close: {self.obstacle_distances['front']} < {self.safety_thresholds['proximity']}"
        
        # Check component temperatures
        for comp, temp in self.component_temperatures.items():
            if temp > self.safety_thresholds['temperature']:
                return False, f"Component {comp} too hot: {temp} > {self.safety_thresholds['temperature']}"
        
        return True, "Action is safe"

class IntegratedRobotSystem:
    def __init__(self):
        self.supervised_controller = None  # From Exercise 1
        self.rl_controller = None          # From Exercise 2
        self.vision_controller = None      # From Exercise 3
        self.safety_monitor = SafetyMonitor()
        
        self.current_state = np.zeros(10)  # Current state vector
        self.current_action = np.zeros(2)  # Current action (linear, angular)
        self.system_active = True
        
        # Task priorities
        self.task_priorities = {
            'safety': 1,
            'navigation': 2,
            'vision': 3,
            'control': 4
        }
    
    def set_controllers(self, supervised_ctrl, rl_ctrl, vision_ctrl):
        """Set the AI controllers"""
        self.supervised_controller = supervised_ctrl
        self.rl_controller = rl_ctrl
        self.vision_controller = vision_ctrl
    
    def get_ai_action(self, state, sensor_data, task_priority='navigation'):
        """Get action from appropriate AI controller based on task priority"""
        if task_priority == 'vision':
            # Use vision controller if available
            if self.vision_controller:
                return self.vision_controller.get_action_from_vision(sensor_data.get('image', None))
            else:
                return np.array([0.0, 0.0])
        
        elif task_priority == 'navigation':
            # Use RL controller for navigation
            if self.rl_controller:
                return self.rl_controller.get_action(state)
            else:
                return np.array([0.0, 0.0])
        
        else:  # Default to supervised controller
            if self.supervised_controller:
                return self.supervised_controller.get_action(state)
            else:
                return np.array([0.0, 0.0])
    
    def execute_control_cycle(self, sensor_data):
        """Execute one control cycle with safety monitoring"""
        if not self.system_active:
            return np.array([0.0, 0.0])  # Emergency stop
        
        # Update safety monitor with sensor data
        self.safety_monitor.update_sensor_data(sensor_data)
        
        # Get desired action from AI controller
        # This would depend on current task and state
        if sensor_data.get('vision_available', False):
            desired_action = self.get_ai_action(
                state=self.current_state,
                sensor_data=sensor_data,
                task_priority='vision'
            )
        else:
            desired_action = self.get_ai_action(
                state=self.current_state,
                sensor_data=sensor_data,
                task_priority='navigation'
            )
        
        # Check if action is safe
        is_safe, reason = self.safety_monitor.check_safety(desired_action)
        
        if is_safe:
            self.current_action = desired_action
        else:
            # Safety violation - implement safe response
            print(f"Safety violation: {reason}")
            self.safety_monitor.safety_violations += 1
            
            # Emergency stop if too many violations
            if self.safety_monitor.safety_violations > 5:
                self.emergency_stop()
                return np.array([0.0, 0.0])
            
            # Otherwise, try to find a safe action
            self.current_action = self.get_safe_action(desired_action)
        
        return self.current_action
    
    def get_safe_action(self, unsafe_action):
        """Generate a safe action when the desired action is unsafe"""
        # Implement a safe action selection algorithm
        # For now, just reduce velocities
        safe_action = unsafe_action.copy()
        
        # Reduce linear velocity if too close to obstacles
        if self.safety_monitor.obstacle_distances['front'] < 1.0:
            safe_action[0] = min(safe_action[0], 0.2)  # Limit forward speed
        
        if self.safety_monitor.obstacle_distances['front'] < 0.5:
            safe_action[0] = 0.0  # Stop if very close
        
        # Reduce angular velocity if needed
        if abs(self.safety_monitor.obstacle_distances['left'] - 
               self.safety_monitor.obstacle_distances['right']) < 0.2:
            safe_action[1] = 0.0  # Don't turn if equally close to both sides
        
        return safe_action
    
    def emergency_stop(self):
        """Execute emergency stop"""
        print("EMERGENCY STOP ACTIVATED")
        self.system_active = False
        self.current_action = np.array([0.0, 0.0])
    
    def resume_system(self):
        """Resume system after emergency stop"""
        self.system_active = True
        self.safety_monitor.safety_violations = 0
        print("System resumed")

def simulate_integrated_system():
    """Simulate the integrated system"""
    system = IntegratedRobotSystem()
    
    # Simulate sensor data stream
    for step in range(100):
        # Simulate sensor readings
        sensor_data = {
            'obstacle_distances': {
                'front': np.random.uniform(0.5, 5.0),
                'left': np.random.uniform(0.5, 5.0),
                'right': np.random.uniform(0.5, 5.0)
            },
            'temperatures': {
                'motor_1': 25.0 + np.random.uniform(0, 5),
                'motor_2': 25.0 + np.random.uniform(0, 5),
                'cpu': 35.0 + np.random.uniform(0, 10)
            },
            'vision_available': step > 20,  # Vision available after step 20
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Simulated image
        }
        
        # Execute control cycle
        action = system.execute_control_cycle(sensor_data)
        
        # Update state (simulated)
        system.current_state = np.random.randn(10)  # Random state for simulation
        
        print(f"Step {step}: Action = {action}, Obstacles = {sensor_data['obstacle_distances']}")
        
        time.sleep(0.1)  # Simulate real-time constraints

if __name__ == '__main__':
    simulate_integrated_system()
```

2. Implement failure detection and recovery mechanisms:
```python
# failure_detection.py
import threading
import time
from collections import deque

class FailureDetector:
    def __init__(self):
        self.anomaly_threshold = 0.7
        self.performance_threshold = 0.1  # Minimum improvement per time window
        
        # Performance tracking
        self.performance_history = deque(maxlen=50)
        self.action_history = deque(maxlen=20)
        
        # Anomaly detection
        self.anomaly_history = deque(maxlen=100)
        self.failure_modes = {
            'stuck': False,
            'oscillation': False,
            'no_progress': False,
            'sensor_failure': False
        }
    
    def detect_stuck(self, position_history):
        """Detect if robot is stuck in one location"""
        if len(position_history) < 10:
            return False
        
        recent_positions = list(position_history)[-10:]
        distances = [np.linalg.norm(recent_positions[i] - recent_positions[i-1]) 
                    for i in range(1, len(recent_positions))]
        
        avg_distance = np.mean(distances)
        return avg_distance < 0.01  # Less than 1cm movement
    
    def detect_oscillation(self, actions):
        """Detect oscillatory behavior in actions"""
        if len(actions) < 6:
            return False
        
        # Check if actions are oscillating
        recent_actions = list(actions)[-6:]
        action_changes = [np.linalg.norm(recent_actions[i] - recent_actions[i-1]) 
                         for i in range(1, len(recent_actions))]
        
        # High frequency of direction changes indicates oscillation
        direction_changes = sum(1 for change in action_changes if change > 0.5)
        return direction_changes > 4  # More than 4/6 changes
    
    def detect_no_progress(self, goal_distances):
        """Detect if robot is not making progress toward goal"""
        if len(goal_distances) < 5:
            return False
        
        recent_distances = list(goal_distances)[-5:]
        initial_distance = recent_distances[0]
        final_distance = recent_distances[-1]
        
        # Check if distance isn't decreasing
        return (final_distance - initial_distance) > 0.1
    
    def evaluate_integrated_system(self, system, sensor_data):
        """Evaluate the integrated system and detect failures"""
        # Simulate data collection
        position = np.array([sensor_data.get('x', 0), sensor_data.get('y', 0)])
        goal_distance = sensor_data.get('goal_distance', 10.0)
        
        # Update histories
        self.action_history.append(system.current_action)
        
        # Detect various failure modes
        self.failure_modes['stuck'] = self.detect_stuck([position])  # Simplified
        self.failure_modes['oscillation'] = self.detect_oscillation(list(self.action_history))
        self.failure_modes['no_progress'] = self.detect_no_progress([goal_distance])
        self.failure_modes['sensor_failure'] = sensor_data.get('sensor_error', False)
        
        # Return failure status
        any_failure = any(self.failure_modes.values())
        return any_failure, self.failure_modes

class RecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'stuck': self.recover_from_stuck,
            'oscillation': self.recover_from_oscillation,
            'no_progress': self.recover_from_no_progress,
            'sensor_failure': self.recover_from_sensor_failure
        }
    
    def recover_from_stuck(self, system, sensor_data):
        """Recovery strategy for stuck robot"""
        print("Recovering from stuck condition...")
        # Try reverse and turn
        return np.array([-0.3, 0.5])  # Move back and turn
    
    def recover_from_oscillation(self, system, sensor_data):
        """Recovery strategy for oscillation"""
        print("Recovering from oscillation...")
        # Reduce controller gains temporarily
        return np.array([0.0, 0.0])  # Stop and restart with new parameters
    
    def recover_from_no_progress(self, system, sensor_data):
        """Recovery strategy for no progress"""
        print("Recovering from no progress...")
        # Try alternative path
        return np.array([0.0, 0.3])  # Turn to look for alternative route
    
    def recover_from_sensor_failure(self, system, sensor_data):
        """Recovery strategy for sensor failure"""
        print("Recovering from sensor failure...")
        # Use backup sensors or stop safely
        return np.array([0.0, 0.0])  # Stop safely
    
    def execute_recovery(self, failure_modes, system, sensor_data):
        """Execute appropriate recovery strategy"""
        for failure_type, occurred in failure_modes.items():
            if occurred:
                recovery_action = self.recovery_strategies[failure_type](system, sensor_data)
                return recovery_action
        
        # If no specific failure, return safe action
        return np.array([0.0, 0.0])
```

### Deliverables
- Complete integrated robot system with multiple AI components
- Safety monitoring and emergency stop mechanisms
- Failure detection and recovery systems
- Performance evaluation of the integrated system

## Assessment Rubric

### Exercise 1: Supervised Learning (20 points)
- **Implementation Quality**: Correct implementation of supervised learning model (8 points)
- **Training Process**: Proper training and validation procedures (6 points)
- **Performance Evaluation**: Appropriate metrics and analysis (6 points)

### Exercise 2: Reinforcement Learning (25 points)
- **Algorithm Implementation**: Correct DQN/PPO implementation (10 points)
- **Training Performance**: Successful learning with improvement over time (10 points)
- **Comparison Analysis**: Valid comparison between approaches (5 points)

### Exercise 3: Vision-Based Control (25 points)
- **Vision System**: Working object detection and localization (10 points)
- **Vision-to-Control**: Proper integration of vision and control (10 points)
- **Performance Evaluation**: Analysis of vision system performance (5 points)

### Exercise 4: Integration and Safety (30 points)
- **System Integration**: Successful integration of all components (10 points)
- **Safety Mechanisms**: Proper safety monitoring and enforcement (10 points)
- **Failure Handling**: Effective failure detection and recovery (10 points)

## Additional Resources

### Recommended Reading
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics

### Troubleshooting Guide
- Check GPU/CUDA availability for deep learning components
- Verify simulation environments are properly configured
- Validate sensor data formats and ranges
- Monitor memory usage during training

This lab provides comprehensive hands-on experience with implementing AI systems for robotics, from basic learning algorithms to complex integrated systems with safety considerations.