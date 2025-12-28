# Chapter 2: Topics and Asynchronous Communication


<div className="robotDiagram">
  <img src="/static/img/book-image/Flowchart_showing_ROS_2_nodes_communicat_1.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


## Learning Objectives

After completing this chapter, students will be able to:
- Implement publisher and subscriber nodes in ROS 2
- Design appropriate message types for different robot subsystems
- Configure Quality of Service (QoS) settings for various communication needs
- Handle message serialization and deserialization
- Debug common issues in topic-based communication
- Design efficient topic architectures for humanoid robot coordination

## 2.1 Understanding Topics in ROS 2

Topics in ROS 2 implement a publish-subscribe messaging pattern, which enables asynchronous communication between nodes. In this model, publisher nodes send messages to topics, while subscriber nodes receive messages from topics. This decoupling allows for flexible system design where publishers and subscribers can be developed and deployed independently.

### 2.1.1 Characteristics of Topic Communication

- **Asynchronous**: Publishers don't wait for responses from subscribers
- **Many-to-many**: Multiple publishers can send to a single topic; multiple subscribers can receive from a topic
- **Loose coupling**: Publishers don't need to know about subscribers, and vice versa
- **Reliable delivery**: With appropriate QoS settings, messages can be delivered reliably

### 2.1.2 Message Passing Flow

The message flow in topic communication:
1. Publisher node creates a message and publishes it to a topic
2. DDS (middleware) manages the distribution of the message to all subscribers
3. Subscribers receive the message in their callback functions
4. Each subscriber processes the message independently

## 2.2 Message Types and Serialization

### 2.2.1 Standard Message Types

ROS 2 provides standard message types defined in the `std_msgs` package:
- `std_msgs/msg/String`: Simple string messages
- `std_msgs/msg/Int32`, `std_msgs/msg/Float64`: Numeric values
- `std_msgs/msg/Bool`: Boolean values
- `std_msgs/msg/Header`: Header with timestamp and frame ID

### 2.2.2 Geometry Message Types

The `geometry_msgs` package provides common robotic message types:
- `geometry_msgs/msg/Twist`: Linear and angular velocity commands
- `geometry_msgs/msg/Pose`: Position and orientation
- `geometry_msgs/msg/Point`: 3D point
- `geometry_msgs/msg/Quaternion`: Rotation representation
- `geometry_msgs/msg/TransformStamped`: Coordinate transformations

### 2.2.3 Sensor Message Types

The `sensor_msgs` package provides messages for sensor data:
- `sensor_msgs/msg/JointState`: Joint positions, velocities, and efforts
- `sensor_msgs/msg/LaserScan`: LIDAR data
- `sensor_msgs/msg/Image`: Camera images
- `sensor_msgs/msg/Imu`: Inertial measurement unit data
- `sensor_msgs/msg/BatteryState`: Battery information

### 2.2.4 Custom Message Types

For specialized applications, custom messages can be defined using the `.msg` format:

```txt
# Custom message: RobotStatus.msg
bool is_active
float32 battery_voltage
string current_task
int32 error_code
```

Custom messages are compiled during the build process and can be used like standard message types.

## 2.3 Implementing Publishers and Subscribers

### 2.3.1 Basic Publisher (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
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
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.3.2 Basic Subscriber (Python)

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2.3.3 Publisher and Subscriber (C++)

```cpp
// Publisher
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher() : Node("minimal_publisher"), count_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello World: " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};
```

```cpp
// Subscriber
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
    MinimalSubscriber() : Node("minimal_subscriber")
    {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
            });
    }

private:
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

## 2.4 Quality of Service (QoS) Configuration

### 2.4.1 Reliability Policy

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Reliable - ensure all messages are received
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)

# Best effort - maximize performance over reliability
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)
```

### 2.4.2 Durability Policy

```python
from rclpy.qos import QoSProfile, DurabilityPolicy

# Volatile - don't store messages for late-joining subscribers
qos_profile = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)

# Transient local - store messages for late-joining subscribers
qos_profile = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

### 2.4.3 History Policy

```python
from rclpy.qos import QoSProfile, HistoryPolicy

# Keep last N messages
qos_profile = QoSProfile(
    depth=5,
    history=HistoryPolicy.KEEP_LAST
)

# Keep all messages (up to limits)
qos_profile = QoSProfile(
    depth=5,
    history=HistoryPolicy.KEEP_ALL
)
```

## 2.5 Advanced Topic Patterns

### 2.5.1 Latched Topics

Latched topics store the last published message and send it immediately to new subscribers. This is useful for configuration data that needs to be available immediately upon connection.

```python
from rclpy.qos import QoSProfile, LivelinessPolicy

# Create a latch-like behavior using TRANSIENT_LOCAL durability
qos_profile = QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
)
```

### 2.5.2 Topic Remapping

Topic remapping allows changing topic names without modifying source code:

```bash
# Remap /topic to /new_topic
ros2 run package_name node_name --ros-args --remap __ns:=/robot1 --remap topic:=new_topic
```

### 2.5.3 Topic Namespacing

Namespacing helps organize topics for multi-robot systems:

```python
# Create namespace for robot1
self.publisher_ = self.create_publisher(String, 'robot1/sensor_data', 10)
```

## 2.6 Topic Design for Humanoid Robots

### 2.6.1 Sensor Data Topics

For humanoid robots, common sensor topics include:
- Joint states: `joint_states` (sensor_msgs/JointState)
- IMU data: `imu/data` (sensor_msgs/Imu)
- Camera images: `camera/image_raw` (sensor_msgs/Image)
- LIDAR scans: `scan` (sensor_msgs/LaserScan)

### 2.6.2 Control Command Topics

For commanding humanoid robot subsystems:
- Joint commands: `joint_commands` (trajectory_msgs/JointTrajectory)
- Base velocity: `cmd_vel` (geometry_msgs/Twist)
- Head position: `head_controller/position` (control_msgs/JointTrajectoryControllerState)

### 2.6.3 State Information Topics

For sharing robot state across components:
- Robot state: `robot_state` (custom message with battery, status, etc.)
- Odometry: `odom` (nav_msgs/Odometry)
- TF transforms: `tf` and `tf_static` (tf2_msgs/TFMessage)

## 2.7 Performance Considerations

### 2.7.1 Message Frequency

Consider the appropriate publishing frequency for your application:
- High-frequency data (IMU): 100-1000 Hz
- Control commands: 50-200 Hz
- State information: 1-10 Hz
- Debug/status: 0.1-1 Hz

### 2.7.2 Message Size

Optimize message size for network efficiency:
- Use appropriate data types (e.g., float32 vs float64)
- Minimize string usage in high-frequency messages
- Consider compression for image or point cloud data

### 2.7.3 Resource Management

Manage resources properly in your nodes:
- Use appropriate queue sizes based on expected message rate
- Clean up resources properly when nodes shut down
- Monitor CPU and memory usage during operation

## 2.8 Debugging Topic Communication

### 2.8.1 Common Commands

```bash
# List all topics
ros2 topic list

# Get information about a specific topic
ros2 topic info /topic_name

# Echo messages on a topic
ros2 topic echo /topic_name

# Publish a message from command line
ros2 topic pub /topic_name std_msgs/String "data: 'Hello'"
```

### 2.8.2 Debugging Strategies

1. **Check topic availability**: Use `ros2 topic list` to verify topics exist
2. **Monitor message flow**: Use `ros2 topic echo` to see if messages are being published
3. **Verify node connections**: Use `ros2 node info <node_name>` to check subscriptions/publishers
4. **Check QoS compatibility**: Ensure publishers and subscribers have compatible QoS settings
5. **Monitor system resources**: Check for CPU/memory issues that might affect performance

## 2.9 Best Practices for Topic Design

### 2.9.1 Naming Conventions

Follow consistent naming conventions:
- Use lowercase with underscores: `sensor_data`, `joint_states`
- Group related topics with prefixes: `head_camera/image_raw`, `head_camera/camera_info`
- Use descriptive names that indicate content and purpose

### 2.9.2 Message Design

- Keep messages focused and specific
- Use existing message types when possible
- Include necessary metadata (timestamps, source information)
- Consider extensibility for future requirements

### 2.9.3 Performance Optimization

- Use appropriate QoS settings for your application
- Consider message frequency and system load
- Monitor and optimize for minimal latency
- Implement appropriate buffering strategies

## Chapter Summary

This chapter covered the fundamentals of topic-based communication in ROS 2, including implementation of publishers and subscribers, message types, QoS configuration, and design patterns for humanoid robotics applications. We explored performance considerations and debugging techniques for topic communication, emphasizing best practices for reliable, efficient robotic systems.

## Key Terms
- Publish-Subscribe Pattern
- Message Serialization
- Quality of Service (QoS)
- Latched Topics
- Topic Remapping
- Namespacing
- Message Frequency

## Exercises
1. Implement a publisher-subscriber pair that shares robot sensor data
2. Configure different QoS profiles for various types of sensor data
3. Design a topic architecture for a humanoid robot with 20+ joints
4. Debug a simulated communication problem between two nodes

## References
- ROS 2 Documentation: https://docs.ros.org/
- DDS Specification: https://www.omg.org/spec/DDS/
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.