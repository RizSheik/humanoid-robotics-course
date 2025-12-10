---
id: module-1-chapter-2-topics-communication
title: 'Module 1 — The Robotic Nervous System | Chapter 2 — Topics Communication'
sidebar_label: 'Chapter 2 — Topics Communication'
---

# Chapter 2 — Topics Communication

## Understanding Topic-Based Communication in ROS 2

Topics enable asynchronous, many-to-many communication between ROS 2 nodes through a publish-subscribe pattern. This communication method is fundamental to building distributed robotic systems.

### Topic Communication Model

In the publish-subscribe pattern:
- Publishers send messages to a topic without knowledge of subscribers
- Subscribers receive messages from a topic without knowledge of publishers
- Multiple publishers can send to the same topic
- Multiple subscribers can receive from the same topic

### Quality of Service (QoS) Settings

QoS settings allow fine-tuning of message delivery characteristics:

#### Reliability Policy
- **Reliable**: Ensure all messages are delivered (with retries)
- **Best Effort**: Don't ensure delivery, but try to deliver

#### Durability Policy
- **Transient Local**: Late-joining subscribers receive the last message published
- **Volatile**: Only receive messages published after subscription

#### History Policy
- **Keep Last**: Maintain a specific number of most recent messages
- **Keep All**: Maintain all messages (limited by resource constraints)

```cpp
// Example of configuring QoS for a publisher
rclcpp::QoS qos_profile(10);  // history depth of 10
qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

auto publisher = node->create_publisher<std_msgs::msg::String>("topic_name", qos_profile);
```

### Creating Publishers

```cpp
// C++ Example: Creating a publisher
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class TopicPublisher : public rclcpp::Node
{
public:
    TopicPublisher() : Node("topic_publisher")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("robot_sensors", 10);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&TopicPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Sensor reading: " + std::to_string(counter_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t counter_ = 0;
};
```

```python
# Python Example: Creating a publisher
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TopicPublisher(Node):
    def __init__(self):
        super().__init__('topic_publisher')
        self.publisher_ = self.create_publisher(String, 'robot_sensors', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Creating Subscribers

```cpp
// C++ Example: Creating a subscriber
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class TopicSubscriber : public rclcpp::Node
{
public:
    TopicSubscriber() : Node("topic_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "robot_sensors", 10,
            std::bind(&TopicSubscriber::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received sensor data: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

```python
# Python Example: Creating a subscriber
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TopicSubscriber(Node):
    def __init__(self):
        super().__init__('topic_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_sensors',
            self.topic_callback,
            10)
        self.subscription  # prevent unused variable warning

    def topic_callback(self, msg):
        self.get_logger().info(f'Received sensor data: "{msg.data}"')
```

### Message Types

ROS 2 comes with standard message types but also allows custom message definitions:

Common message packages:
- **std_msgs**: Basic data types (Int, Float, String, etc.)
- **sensor_msgs**: Sensor data (LaserScan, Image, etc.)
- **geometry_msgs**: Geometric primitives (Pose, Point, Vector3, etc.)
- **nav_msgs**: Navigation-specific messages

### Performance Considerations

- Use appropriate QoS settings for your application
- Consider message size and frequency
- Use appropriate history depth based on application needs
- Be aware of message serialization overhead

Topic-based communication forms the backbone of most ROS 2 systems, enabling flexible and decoupled robotic applications.