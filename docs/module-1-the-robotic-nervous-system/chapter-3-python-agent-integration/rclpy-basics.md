---
id: module-1-chapter-3-rclpy-basics
title: 'Module 1 — The Robotic Nervous System | Chapter 3 — RCLPY Basics'
sidebar_label: 'Chapter 3 — RCLPY Basics'
---

# Chapter 3 — RCLPY Basics

## Introduction to ROS 2 Client Library for Python

The ROS 2 Client Library for Python (rclpy) provides Python bindings for ROS 2 concepts, allowing developers to create ROS 2 nodes, publishers, subscribers, services, and actions in Python.

### Setting Up rclpy

Before creating any ROS 2 node in Python, you must initialize the rclpy library:

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)
    
    # Create a node instance
    my_node = MyNode()
    
    # Spin the node to process callbacks
    rclpy.spin(my_node)
    
    # Destroy the node explicitly (optional)
    my_node.destroy_node()
    
    # Shutdown the ROS 2 Python client library
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Basic Node

A node is the fundamental building block of a ROS 2 system:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')  # Initialize with node name
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    
    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Understanding Node Lifecycle

Every rclpy node goes through several lifecycle stages:

1. **Initialization**: The node is created and registered with the ROS 2 graph
2. **Execution**: The node runs and processes messages/serves requests
3. **Shutdown**: The node is destroyed and unregistered from the graph

### Creating Publishers

Publishers allow nodes to send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        
        # Create a publisher with topic name and queue size
        self.publisher_ = self.create_publisher(String, 'my_topic', 10)
        
        # Create a timer to periodically publish messages
        self.timer = self.create_timer(0.5, self.publish_message)
        self.counter = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Message #{self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1
```

### Creating Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        
        # Create a subscriber with topic name, message type, and callback
        self.subscription = self.create_subscription(
            String,
            'my_topic',
            self.listener_callback,
            10)  # Queue size
        
        # Prevent unused variable warning
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

### Working with Timers

Timers allow periodic execution of functions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')
        
        self.publisher_ = self.create_publisher(String, 'timer_topic', 10)
        
        # Create timer with period in seconds
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Timer message #{self.counter}'
        self.publisher_.publish(msg)
        self.counter += 1
```

### Parameters in rclpy

Parameters allow runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with default values
        self.declare_parameter('publish_frequency', 1.0)
        self.declare_parameter('message_content', 'Hello')
        
        # Get parameter values
        frequency = self.get_parameter('publish_frequency').value
        content = self.get_parameter('message_content').value
        
        self.message_content = content
        self.publisher_ = self.create_publisher(String, 'parameter_topic', 10)
        
        # Use parameters to configure behavior
        period = 1.0 / frequency
        self.timer = self.create_timer(period, self.publish_message)

    def publish_message(self):
        msg = String()
        msg.data = f'{self.message_content} #{int(1.0 / self.get_timer_period(self.timer))}'
        self.publisher_.publish(msg)
```

### Quality of Service (QoS) in rclpy

QoS settings can be configured for publishers and subscribers:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

class QoSNode(Node):
    def __init__(self):
        super().__init__('qos_node')
        
        # Create a QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        # Use QoS profile for publisher
        self.publisher_ = self.create_publisher(String, 'qos_topic', qos_profile)
        
        # Use QoS profile for subscriber
        self.subscription = self.create_subscription(
            String,
            'qos_topic',
            self.listener_callback,
            qos_profile
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'QoS message: {msg.data}')
```

### Error Handling and Logging

rclpy provides extensive logging capabilities:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class LogNode(Node):
    def __init__(self):
        super().__init__('log_node')
        
        self.publisher_ = self.create_publisher(String, 'log_topic', 10)
        self.timer = self.create_timer(1.0, self.work_with_logging)

    def work_with_logging(self):
        try:
            # Do some work
            result = self.perform_calculation()
            
            # Log the result
            msg = String()
            msg.data = f'Result: {result}'
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published result: {result}')
            
        except Exception as e:
            # Log errors
            self.get_logger().error(f'Error performing calculation: {str(e)}')
            # Handle the error appropriately

    def perform_calculation(self):
        # Some calculation that might fail
        return 42
```

### Testing with rclpy

rclpy provides tools for testing nodes:

```python
import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.subscriber = self.create_subscription(
            String,
            'test_topic',
            self.callback,
            10
        )
        self.received_messages = []

    def callback(self, msg):
        self.received_messages.append(msg.data)

class TestNodeMethods(unittest.TestCase):
    def test_node_creation(self):
        rclpy.init()
        node = TestNode()
        self.assertIsNotNone(node)
        rclpy.shutdown()

if __name__ == '__main__':
    unittest.main()
```

This chapter provides the foundational knowledge of rclpy, enabling you to create robust and efficient Python-based ROS 2 nodes.