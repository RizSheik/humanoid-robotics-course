---
id: module-1-chapter-1-nodes-architecture
title: 'Module 1 — The Robotic Nervous System | Chapter 1 — Nodes Architecture'
sidebar_label: 'Chapter 1 — Nodes Architecture'
---

# Chapter 1 — Nodes Architecture

## Understanding ROS 2 Node Architecture

Nodes are the fundamental building blocks of any ROS 2 system. Understanding their architecture is crucial for developing efficient and robust robotic applications.

### Node Architecture Components

#### Node Interface
The node interface is the core component that provides the necessary functionality for communication in ROS 2. Each node contains:

- **Node Handle**: Provides access to ROS-specific functionality like creating publishers, subscribers, services, etc.
- **Callback Group**: Groups callbacks that should be executed together to manage concurrency
- **Executor**: Manages the execution of callbacks from various sources

#### Communication Interfaces
Every node can contain multiple communication interfaces:

- Publishers: Send messages on topics
- Subscribers: Receive messages from topics
- Service Servers: Provide services
- Service Clients: Request services
- Action Servers: Provide actions
- Action Clients: Request actions

### Node Lifecycle

ROS 2 introduces a lifecycle concept that allows nodes to transition through different states:

1. **Unconfigured**: The node is initialized but not configured
2. **Inactive**: The node is configured but not active
3. **Active**: The node is fully operational
4. **Finalized**: The node is shut down

### Creating Robust Nodes

#### Best Practices for Node Design

```cpp
// C++ Example: Well-structured ROS 2 node
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("robot_controller")
    {
        // Initialize publishers, subscribers, services
        publisher_ = this->create_publisher<std_msgs::msg::String>("robot_status", 10);
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "robot_commands", 10,
            std::bind(&RobotController::commandCallback, this, std::placeholders::_1));
        
        // Create timers
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&RobotController::timerCallback, this));
    }

private:
    void commandCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received command: '%s'", msg->data.c_str());
    }
    
    void timerCallback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Robot is running";
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

```python
# Python Example: Well-structured ROS 2 node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Initialize publishers, subscribers, services
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)
        self.subscription = self.create_subscription(
            String,
            'robot_commands',
            self.command_callback,
            10)
        
        # Create timers
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.i = 0

    def command_callback(self, msg):
        self.get_logger().info(f'Received command: "{msg.data}"')

    def timer_callback(self):
        msg = String()
        msg.data = f'Robot is running: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Node Communication Patterns

#### Publisher-Subscriber Pattern
The most common pattern in ROS 2, enabling asynchronous, decoupled communication between nodes.

#### Client-Server Pattern
Used for synchronous request-response communication.

#### Action Pattern
Used for long-running tasks with feedback and goal management.

### Advanced Node Concepts

#### Callback Groups
Callback groups allow you to control the threading model of your node. You can group callbacks to be executed in the same thread or execute them in different threads.

#### Executors
Executors manage the execution of callbacks in your node. Different executors provide different concurrency models:

- **Single-threaded executor**: All callbacks run in the same thread
- **Multi-threaded executor**: Callbacks can run in multiple threads
- **Static single-threaded executor**: Optimized version of single-threaded

This architecture enables the development of complex, reliable robotic systems with proper separation of concerns and robust communication patterns.