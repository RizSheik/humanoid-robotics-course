---
id: module-1-chapter-1-practical-exercises
title: "Chapter 1 — Practical Exercises"
slug: /module-1-the-robotic-nervous-system/chapter-1-ros-architecture/practical-exercises
---

# Chapter 1: Practical Exercises - ROS 2 Architecture & Nodes

## Exercise 1: Creating Your First ROS 2 Node

### Objective
Create a simple ROS 2 node in Python that publishes a message to a topic.

### Steps
1. Create a new ROS 2 package named `humanoid_robot_nodes`
2. Implement a publisher node that publishes messages to `/robot_status`
3. Create a subscriber node that listens to `/robot_status` and logs received messages
4. Test your nodes using the `ros2 run` command

### Solution
```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RobotStatusPublisher(Node):

    def __init__(self):
        super().__init__('robot_status_publisher')
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Robot status: operational - {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    robot_status_publisher = RobotStatusPublisher()
    rclpy.spin(robot_status_publisher)
    robot_status_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Exercise 2: Creating a Humanoid Joint Controller Node
Create a node that simulates controlling joint positions for a humanoid robot with 20+ degrees of freedom.

### Exercise 3: Node Communication Patterns
Implement a service server and client for querying humanoid robot joint states.

## Lab Assignment

Develop a ROS 2 node that:
1. Subscribes to sensor data from a humanoid robot
2. Processes the data to determine robot state
3. Publishes commands to control robot behavior
4. Implements proper error handling and logging

## Resources
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [rclpy API Reference](https://docs.ros.org/en/humble/p/rclpy/)