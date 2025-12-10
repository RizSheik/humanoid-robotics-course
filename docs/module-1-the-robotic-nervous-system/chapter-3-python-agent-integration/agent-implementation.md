---
id: module-1-chapter-3-agent-implementation
title: 'Module 1 — The Robotic Nervous System | Chapter 3 — Agent Implementation'
sidebar_label: 'Chapter 3 — Agent Implementation'
---

# Chapter 3 — Agent Implementation

## Creating Intelligent ROS 2 Agents

An agent in ROS 2 is a node that performs autonomous tasks by perceiving its environment through sensors, making decisions based on its programmed logic, and acting through actuators. This chapter covers the implementation of intelligent agents for humanoid robotics.

### Agent Architecture

A typical ROS 2 agent contains the following components:

1. **Perception Module**: Processes sensor data to understand the environment
2. **Cognition Module**: Processes information and makes decisions
3. **Action Module**: Executes actions based on decisions
4. **Communication Module**: Interacts with other nodes and systems

### Basic Agent Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RobotAgent(Node):
    def __init__(self):
        super().__init__('robot_agent')
        
        # Perception module
        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )
        
        # Action module
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Agent state
        self.closest_obstacle_distance = float('inf')
        self.agent_state = 'exploring'  # exploring, avoiding, stopped
        
        # Decision-making timer
        self.timer = self.create_timer(0.1, self.decision_callback)
        
        self.get_logger().info('Robot agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        # Find closest obstacle
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if r > msg.range_min and r < msg.range_max]
            if valid_ranges:
                self.closest_obstacle_distance = min(valid_ranges)

    def decision_callback(self):
        """Make decisions based on perceived data"""
        msg = Twist()
        
        if self.closest_obstacle_distance < 1.0:  # Obstacle within 1 meter
            self.agent_state = 'avoiding'
            # Turn to avoid obstacle
            msg.angular.z = 0.5
            msg.linear.x = 0.1
        else:
            self.agent_state = 'exploring'
            # Move forward
            msg.linear.x = 0.3
            msg.angular.z = 0.0
            
        self.cmd_vel_publisher.publish(msg)
        
        self.get_logger().info(f'State: {self.agent_state}, Distance: {self.closest_obstacle_distance:.2f}')

def main(args=None):
    rclpy.init(args=args)
    agent = RobotAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()
```

### State Machine Agent

More complex agents can be implemented using state machines:

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    EXPLORING = 2
    AVOIDING_OBSTACLE = 3
    NAVIGATING = 4
    PICKING_UP = 5
    PLACING = 6

class StateMachineAgent(Node):
    def __init__(self):
        super().__init__('state_machine_agent')
        
        # Perception
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        
        # Action
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize state
        self.current_state = RobotState.IDLE
        self.closest_obstacle = float('inf')
        
        # State timer
        self.timer = self.create_timer(0.1, self.state_machine_callback)
        
        self.get_logger().info('State machine agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)

    def state_machine_callback(self):
        """State machine logic"""
        if self.current_state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.current_state == RobotState.EXPLORING:
            self.handle_exploring_state()
        elif self.current_state == RobotState.AVOIDING_OBSTACLE:
            self.handle_avoiding_state()
        # Add more states as needed

    def handle_idle_state(self):
        """Handle idle state"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)
        
        # Transition to exploring if safe
        if self.closest_obstacle > 2.0:
            self.current_state = RobotState.EXPLORING

    def handle_exploring_state(self):
        """Handle exploring state"""
        msg = Twist()
        msg.linear.x = 0.3
        msg.angular.z = 0.0
        
        if self.closest_obstacle < 0.8:
            self.current_state = RobotState.AVOIDING_OBSTACLE
            
        self.cmd_vel_publisher.publish(msg)

    def handle_avoiding_state(self):
        """Handle obstacle avoidance state"""
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.5  # Turn right
        
        # Return to exploring when obstacle is far enough
        if self.closest_obstacle > 1.2:
            self.current_state = RobotState.EXPLORING
            
        self.cmd_vel_publisher.publish(msg)
```

### Behavior-Based Agent

An alternative approach is to implement behavior-based agents:

```python
from abc import ABC, abstractmethod

class Behavior(ABC):
    """Abstract base class for behaviors"""
    
    def __init__(self, name):
        self.name = name
        self.active = False
    
    @abstractmethod
    def execute(self, agent):
        """Execute the behavior"""
        pass
    
    @abstractmethod
    def check_activation(self, agent):
        """Check if this behavior should be activated"""
        pass

class AvoidObstacleBehavior(Behavior):
    def __init__(self):
        super().__init__("avoid_obstacle")
        self.min_distance = 0.8

    def check_activation(self, agent):
        return agent.closest_obstacle < self.min_distance

    def execute(self, agent):
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.5  # Turn to avoid
        agent.cmd_vel_publisher.publish(msg)
        return True  # Behavior completed

class ExploreBehavior(Behavior):
    def __init__(self):
        super().__init__("explore")

    def check_activation(self, agent):
        return agent.closest_obstacle >= 0.8

    def execute(self, agent):
        msg = Twist()
        msg.linear.x = 0.3
        msg.angular.z = 0.0
        agent.cmd_vel_publisher.publish(msg)
        return True

class BehaviorBasedAgent(Node):
    def __init__(self):
        super().__init__('behavior_based_agent')
        
        # Perception
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        
        # Action
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Behaviors
        self.behaviors = [
            AvoidObstacleBehavior(),
            ExploreBehavior()
        ]
        
        # Agent state
        self.closest_obstacle = float('inf')
        
        # Execution timer
        self.timer = self.create_timer(0.1, self.execute_behavior)
        
        self.get_logger().info('Behavior-based agent initialized')

    def laser_callback(self, msg):
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)

    def execute_behavior(self):
        """Select and execute the most appropriate behavior"""
        for behavior in self.behaviors:
            if behavior.check_activation(self):
                behavior.execute(self)
                self.get_logger().info(f'Executing behavior: {behavior.name}')
                return
        
        # If no behavior is active, publish stop command
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
```

### Agent Communication

Agents often need to communicate with other ROS 2 nodes:

```python
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Pose

class CommunicatingAgent(Node):
    def __init__(self):
        super().__init__('communicating_agent')
        
        # Publishers
        self.status_publisher = self.create_publisher(Bool, 'agent_status', 10)
        self.task_publisher = self.create_publisher(String, 'agent_task', 10)
        self.pose_publisher = self.create_publisher(Pose, 'agent_pose', 10)
        
        # Subscribers
        self.command_subscription = self.create_subscription(
            String, 'agent_commands', self.command_callback, 10)
        
        # Agent variables
        self.operational = True
        self.current_task = "idle"
        self.current_pose = Pose()
        
        # Communication timer
        self.timer = self.create_timer(1.0, self.communicate_status)
        
        self.get_logger().info('Communicating agent initialized')

    def command_callback(self, msg):
        """Handle incoming commands"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')
        
        if command == 'start':
            self.current_task = "active"
        elif command == 'stop':
            self.current_task = "idle"
        elif command == 'shutdown':
            self.operational = False

    def communicate_status(self):
        """Publish agent status"""
        status_msg = Bool()
        status_msg.data = self.operational
        self.status_publisher.publish(status_msg)
        
        task_msg = String()
        task_msg.data = self.current_task
        self.task_publisher.publish(task_msg)
        
        pose_msg = Pose()
        # Update with actual pose data
        self.pose_publisher.publish(pose_msg)
```

### Advanced Agent Concepts

#### Learning Agents

Agents can incorporate learning capabilities:

```python
import numpy as np

class LearningAgent(Node):
    def __init__(self):
        super().__init__('learning_agent')
        
        # Perception
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        
        # Action
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Learning components
        self.q_table = np.zeros((10, 3))  # Simplified Q-table
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
        
        self.previous_state = None
        self.previous_action = None
        self.cumulative_reward = 0
        
        # Learning timer
        self.timer = self.create_timer(0.1, self.learning_callback)

    def laser_callback(self, msg):
        """Process laser scan and determine state"""
        # Simplified: discretize laser readings to state
        if len(msg.ranges) > 0:
            avg_distance = np.mean([r for r in msg.ranges 
                                   if msg.range_min < r < msg.range_max])
            # Discretize into 10 states based on distance
            self.state = min(int(avg_distance * 5), 9)  # 0-9

    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice([0, 1, 2])  # straight, left, right
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])

    def calculate_reward(self):
        """Calculate reward based on environment"""
        # Positive reward for moving forward safely
        # Negative reward for being too close to obstacles
        if self.closest_obstacle < 0.5:
            return -10  # Crash penalty
        elif self.closest_obstacle < 1.0:
            return -1   # Too close penalty
        else:
            return 1    # Safe movement reward

    def learning_callback(self):
        """Learning algorithm"""
        current_state = self.state
        
        # Select action
        action = self.get_action(current_state)
        
        # Execute action
        msg = Twist()
        if action == 0:  # Move straight
            msg.linear.x = 0.3
        elif action == 1:  # Turn left
            msg.angular.z = 0.5
        else:  # Turn right
            msg.angular.z = -0.5
            
        self.cmd_vel_publisher.publish(msg)
        
        # Learn from experience if there was a previous state
        if self.previous_state is not None:
            reward = self.calculate_reward()
            old_value = self.q_table[self.previous_state, self.previous_action]
            next_max = np.max(self.q_table[current_state])
            
            # Q-learning update
            new_value = old_value + self.learning_rate * (
                reward + self.discount_factor * next_max - old_value
            )
            self.q_table[self.previous_state, self.previous_action] = new_value
        
        # Update state-action pair for next iteration
        self.previous_state = current_state
        self.previous_action = action
```

### Agent Design Patterns

When designing agents, consider these patterns:

1. **Reactive Agents**: Respond immediately to environmental changes
2. **Deliberative Agents**: Plan ahead before acting
3. **Hybrid Agents**: Combine reactive and deliberative approaches
4. **Learning Agents**: Improve behavior through experience

This chapter provides the foundation for implementing intelligent agents that can operate autonomously in robotic systems.