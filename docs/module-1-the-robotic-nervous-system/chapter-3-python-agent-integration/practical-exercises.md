---
id: module-1-chapter-3-practical-exercises
title: 'Module 1 — The Robotic Nervous System | Chapter 3 — Practical Exercises'
sidebar_label: 'Chapter 3 — Practical Exercises'
---

# Chapter 3 — Practical Exercises

## Python Agent Integration: Hands-On Implementation

This practical lab focuses on implementing Python-based agents for robotic systems using the rclpy library. You will build increasingly complex agents that demonstrate various concepts in robot autonomy.

### Exercise 1: Basic Robot Agent

#### Objective
Create a simple reactive robot agent that responds to sensor inputs.

#### Steps
1. Create a new ROS 2 package for the agent
2. Implement a basic agent node that subscribes to laser scan data
3. Make the agent react to obstacles by changing its motion

```python
# basic_agent.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class BasicAgent(Node):
    def __init__(self):
        super().__init__('basic_agent')
        
        # Create subscriber for laser scan
        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )
        
        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize agent variables
        self.closest_obstacle = float('inf')
        
        # Create timer for agent behavior
        self.timer = self.create_timer(0.1, self.agent_behavior)
        
        self.get_logger().info('Basic agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data to find closest obstacle"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)
        
        self.get_logger().info(f'Closest obstacle: {self.closest_obstacle:.2f}m')

    def agent_behavior(self):
        """Basic obstacle avoidance behavior"""
        msg = Twist()
        
        if self.closest_obstacle < 1.0:  # Obstacle within 1 meter
            # Rotate to avoid obstacle
            msg.angular.z = 0.5
            msg.linear.x = 0.0
        else:
            # Move forward
            msg.linear.x = 0.3
            msg.angular.z = 0.0
        
        self.cmd_vel_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    agent = BasicAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 2: State Machine Agent

#### Objective
Implement an agent with a finite state machine to demonstrate different behaviors.

#### Steps
1. Define different robot states (IDLE, EXPLORING, AVOIDING, etc.)
2. Implement state transitions based on sensor input
3. Create different behaviors for each state

```python
# state_machine_agent.py
from enum import Enum
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RobotState(Enum):
    IDLE = 1
    EXPLORING = 2
    AVOIDING_OBSTACLE = 3
    SEARCHING = 4

class StateMachineAgent(Node):
    def __init__(self):
        super().__init__('state_machine_agent')
        
        # Create subscriber for laser scan
        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )
        
        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize agent state
        self.current_state = RobotState.IDLE
        self.closest_obstacle = float('inf')
        self.state_timer = 0
        
        # Create timer for state machine
        self.timer = self.create_timer(0.1, self.state_machine)
        
        self.get_logger().info('State machine agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)

    def state_machine(self):
        """State machine logic"""
        if self.current_state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.current_state == RobotState.EXPLORING:
            self.handle_exploring_state()
        elif self.current_state == RobotState.AVOIDING_OBSTACLE:
            self.handle_avoiding_state()
        elif self.current_state == RobotState.SEARCHING:
            self.handle_searching_state()
        
        self.state_timer += 1

    def handle_idle_state(self):
        """Handle idle state behavior"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)
        
        # Transition to exploring if no obstacles nearby
        if self.closest_obstacle > 2.0:
            self.current_state = RobotState.EXPLORING
            self.state_timer = 0
            self.get_logger().info('Transitioned to EXPLORING')

    def handle_exploring_state(self):
        """Handle exploration behavior"""
        msg = Twist()
        msg.linear.x = 0.3
        msg.angular.z = 0.0
        
        # Check for obstacles
        if self.closest_obstacle < 0.8:
            self.current_state = RobotState.AVOIDING_OBSTACLE
            self.state_timer = 0
            self.get_logger().info('Transitioned to AVOIDING_OBSTACLE')
        
        # Add some random turning to explore
        elif self.state_timer > 100:  # Every 10 seconds
            msg.angular.z = 0.3  # Turn slightly
            if self.state_timer > 120:
                self.state_timer = 0  # Reset after 2 seconds of turning
                
        self.cmd_vel_publisher.publish(msg)

    def handle_avoiding_state(self):
        """Handle obstacle avoidance"""
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.5  # Turn away from obstacle
        
        # Return to exploring when clear
        if self.closest_obstacle > 1.2:
            self.current_state = RobotState.EXPLORING
            self.state_timer = 0
            self.get_logger().info('Transitioned to EXPLORING')
        
        self.cmd_vel_publisher.publish(msg)

    def handle_searching_state(self):
        """Handle searching behavior"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.5  # Turn in place
        
        # If found clear path, start exploring
        if self.closest_obstacle > 1.5:
            self.current_state = RobotState.EXPLORING
            self.state_timer = 0
            self.get_logger().info('Transitioned to EXPLORING')
        
        self.cmd_vel_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    agent = StateMachineAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 3: Behavior-Based Agent

#### Objective
Implement an agent using a behavior-based architecture where different behaviors compete for control.

#### Steps
1. Create behavior classes that inherit from a base behavior class
2. Implement priority-based behavior selection
3. Create a coordinator that manages behavior execution

```python
# behavior_based_agent.py
from abc import ABC, abstractmethod
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class Behavior(ABC):
    """Abstract base class for all behaviors"""
    
    def __init__(self, priority=0):
        self.priority = priority
    
    @abstractmethod
    def condition(self, agent):
        """Check if behavior should be active"""
        pass
    
    @abstractmethod
    def action(self, agent):
        """Execute the behavior"""
        pass

class AvoidBehavior(Behavior):
    def __init__(self):
        super().__init__(priority=2)  # High priority
    
    def condition(self, agent):
        return agent.closest_obstacle < 0.8
    
    def action(self, agent):
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.6  # Turn to avoid
        agent.cmd_vel_publisher.publish(msg)
        return True

class ExploreBehavior(Behavior):
    def __init__(self):
        super().__init__(priority=1)  # Medium priority
    
    def condition(self, agent):
        return agent.closest_obstacle >= 0.8 and agent.closest_obstacle < 3.0
    
    def action(self, agent):
        msg = Twist()
        msg.linear.x = 0.3
        msg.angular.z = 0.0
        agent.cmd_vel_publisher.publish(msg)
        return True

class IdleBehavior(Behavior):
    def __init__(self):
        super().__init__(priority=0)  # Low priority
    
    def condition(self, agent):
        return agent.closest_obstacle >= 3.0
    
    def action(self, agent):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        agent.cmd_vel_publisher.publish(msg)
        return True

class BehaviorBasedAgent(Node):
    def __init__(self):
        super().__init__('behavior_based_agent')
        
        # Create subscriber for laser scan
        self.laser_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )
        
        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Initialize behaviors
        self.behaviors = [
            AvoidBehavior(),
            ExploreBehavior(),
            IdleBehavior()
        ]
        
        # Agent state
        self.closest_obstacle = float('inf')
        
        # Behavior execution timer
        self.timer = self.create_timer(0.1, self.execute_highest_priority_behavior)
        
        self.get_logger().info('Behavior-based agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)

    def execute_highest_priority_behavior(self):
        """Find and execute the highest priority active behavior"""
        active_behaviors = []
        
        for behavior in self.behaviors:
            if behavior.condition(self):
                active_behaviors.append(behavior)
        
        if active_behaviors:
            # Find behavior with highest priority
            highest_priority_behavior = max(active_behaviors, key=lambda b: b.priority)
            highest_priority_behavior.action(self)
            
            self.get_logger().info(f'Executing behavior: {highest_priority_behavior.__class__.__name__}')
        else:
            # If no behavior is active, stop the robot
            stop_msg = Twist()
            self.cmd_vel_publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    agent = BehaviorBasedAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 4: Agent Communication and Coordination

#### Objective
Implement communication between multiple agents to coordinate their behaviors.

#### Steps
1. Create an agent that publishes its status
2. Create another agent that subscribes to status messages
3. Implement coordination logic based on other agents' status

```python
# coordinator_agent.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class CoordinatorAgent(Node):
    def __init__(self):
        super().__init__('coordinator_agent')
        
        # Communication
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)
        self.status_subscription = self.create_subscription(
            String, 'other_robot_status', self.status_callback, 10)
        
        # Motion control
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Agent state
        self.closest_obstacle = float('inf')
        self.other_robot_status = "unknown"
        self.my_role = "explorer"  # explorer, helper, avoider
        
        # Communication timer
        self.timer = self.create_timer(0.5, self.communicate_and_act)
        
        self.get_logger().info('Coordinator agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            self.closest_obstacle = min(valid_ranges)

    def status_callback(self, msg):
        """Handle status from other robot"""
        self.other_robot_status = msg.data
        self.get_logger().info(f'Other robot status: {msg.data}')

    def communicate_and_act(self):
        """Communicate status and decide actions based on coordination"""
        # Publish my current status
        status_msg = String()
        status_msg.data = f"role:{self.my_role},obstacle:{self.closest_obstacle:.2f}"
        self.status_publisher.publish(status_msg)
        
        # Decide action based on my status and other robot's status
        msg = Twist()
        
        if self.other_robot_status == "role:explorer,obstacle:0.0" or "obstacle:0.0" in self.other_robot_status:
            # Other robot has critical issues, change my role to helper
            self.my_role = "helper"
            msg.linear.x = 0.2  # Move toward other robot
            msg.angular.z = 0.0
        elif self.closest_obstacle < 0.8:
            # I have an obstacle, request help if other robot is available
            self.my_role = "avoider"
            msg.linear.x = 0.0
            msg.angular.z = 0.5  # Turn to avoid
        else:
            # No immediate issues, explore
            self.my_role = "explorer"
            msg.linear.x = 0.3
            msg.angular.z = 0.1  # Slight exploration turn
        
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f'My role: {self.my_role}, Command: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    agent = CoordinatorAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 5: Advanced Agent with Learning Component

#### Objective
Implement a learning agent that improves its behavior over time using a simple reinforcement learning algorithm.

#### Steps
1. Implement a Q-learning algorithm
2. Create states based on sensor inputs
3. Define actions the agent can take
4. Calculate rewards based on outcomes

```python
import numpy as np
import random

class LearningAgent(Node):
    def __init__(self):
        super().__init__('learning_agent')
        
        # Communication and control
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Learning parameters
        self.num_states = 10  # Discretized distance states
        self.num_actions = 3  # 0: forward, 1: turn left, 2: turn right
        
        # Q-table initialization
        self.q_table = np.zeros((self.num_states, self.num_actions))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.exploration_decay = 0.995  # Decay exploration over time
        
        # Agent state tracking
        self.current_state = 0
        self.previous_state = None
        self.previous_action = None
        self.episode_reward = 0
        
        # Learning timer
        self.timer = self.create_timer(0.1, self.learning_step)
        
        self.get_logger().info('Learning agent initialized')

    def laser_callback(self, msg):
        """Process laser scan and discretize to state"""
        valid_ranges = [r for r in msg.ranges 
                       if msg.range_min < r < msg.range_max]
        if valid_ranges:
            avg_distance = np.mean(valid_ranges)
            # Discretize distance into states (0-9)
            self.current_state = min(int(avg_distance * 3), self.num_states - 1)

    def choose_action(self):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: best known action
            return int(np.argmax(self.q_table[self.current_state]))

    def calculate_reward(self):
        """Calculate reward based on current state"""
        if self.current_state == 0:  # Very close to obstacle
            return -10  # Large negative reward for potential collision
        elif self.current_state < 3:  # Close to obstacle
            return -5
        elif self.current_state >= 7:  # Far from obstacles
            return 1  # Small positive reward for safe navigation
        else:
            return 0  # Neutral reward

    def learning_step(self):
        """Main learning algorithm step"""
        # Choose action
        action = self.choose_action()
        
        # Execute action by publishing velocity command
        msg = Twist()
        if action == 0:  # Move forward
            msg.linear.x = 0.3
            msg.angular.z = 0.0
        elif action == 1:  # Turn left
            msg.linear.x = 0.1
            msg.angular.z = 0.5
        else:  # Turn right
            msg.linear.x = 0.1
            msg.angular.z = -0.5
            
        self.cmd_vel_publisher.publish(msg)
        
        # Learn from experience if we have a previous state-action pair
        if self.previous_state is not None:
            reward = self.calculate_reward()
            self.episode_reward += reward
            
            # Q-learning update
            current_q = self.q_table[self.previous_state, self.previous_action]
            next_max_q = np.max(self.q_table[self.current_state])
            
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * next_max_q - current_q
            )
            
            self.q_table[self.previous_state, self.previous_action] = new_q
        
        # Update state-action pair for next iteration
        self.previous_state = self.current_state
        self.previous_action = action
        
        # Decay exploration rate
        self.epsilon *= self.exploration_decay
        if self.epsilon < 0.01:  # Minimum exploration rate
            self.epsilon = 0.01
        
        self.get_logger().info(f'State: {self.current_state}, Action: {action}, Epsilon: {self.epsilon:.3f}, Reward: {self.episode_reward:.2f}')

def main(args=None):
    rclpy.init(args=args)
    agent = LearningAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        # Print final Q-table
        agent.get_logger().info(f'Final Q-table:\n{agent.q_table}')
        agent.get_logger().info(f'Final epsilon: {agent.epsilon:.3f}')
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Assessment Criteria

Your implementation will be evaluated based on:

1. **Correctness**: All agents behave as specified
2. **Code Quality**: Clean, well-documented, and modular code
3. **Understanding**: Ability to explain agent architectures and decision-making
4. **Creativity**: Innovative solutions and extensions to the basic requirements

### Troubleshooting Tips

1. **Node Communication Issues**: Check that your robot simulation is publishing the expected topics
2. **Timing Issues**: Adjust timer periods to ensure proper synchronization
3. **Behavior Conflicts**: Ensure clear priority rules when multiple behaviors might be active
4. **Learning Convergence**: Monitor Q-values to ensure learning is occurring properly

### Extensions for Advanced Students

- Implement more complex multi-agent coordination scenarios
- Add neural networks for more advanced learning
- Implement hierarchical task networks for complex behaviors
- Add perception processing for more sophisticated environment understanding
- Create a simulator to test agents without physical hardware

This practical exercise provides hands-on experience with implementing intelligent agents in ROS 2, from basic reactive systems to advanced learning-based agents.