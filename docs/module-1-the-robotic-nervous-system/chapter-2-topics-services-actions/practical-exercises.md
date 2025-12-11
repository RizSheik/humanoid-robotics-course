---
id: module-1-chapter-2-practical-exercises
title: 'Module 1 — The Robotic Nervous System | Chapter 2 — Practical Exercises'
sidebar_label: 'Chapter 2 — Practical Exercises'
---

# Chapter 2 — Practical Exercises

## ROS 2 Communication Patterns: Hands-On Implementation

This practical lab will guide you through implementing different communication patterns in ROS 2, including topics, services, and actions.

### Exercise 1: Publisher-Subscriber Pattern Implementation

#### Objective
Implement a simple publisher-subscriber system to understand asynchronous communication.

#### Steps
1. Create a publisher node that publishes messages containing sensor data
2. Create a subscriber node that receives and processes these messages
3. Configure Quality of Service (QoS) settings to experiment with message delivery guarantees

```python
# publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher_ = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading {self.counter}: {42.5 + self.counter * 0.1}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()
    
    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# subscriber_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()
    
    try:
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 2: Service Implementation

#### Objective
Create a request-response system to implement a robot command interface.

#### Steps
1. Define a custom service message for robot commands
2. Implement a service server that processes commands
3. Create a service client that sends commands

```python
# robot_command.srv
# Request
string command
float64 value
---
# Response
bool success
string message
```

```python
# service_server.py
import rclpy
from rclpy.node import Node
from your_package.srv import RobotCommand  # You'll need to define this service

class RobotCommandServer(Node):
    def __init__(self):
        super().__init__('robot_command_server')
        self.srv = self.create_service(
            RobotCommand, 
            'robot_command', 
            self.command_callback
        )
        self.get_logger().info('Robot command server started')

    def command_callback(self, request, response):
        self.get_logger().info(f'Received command: {request.command} with value: {request.value}')
        
        # Process the command
        if request.command == 'move_forward':
            # Simulate movement
            self.get_logger().info(f'Moving forward {request.value} meters')
            response.success = True
            response.message = f'Moved forward {request.value} meters'
        elif request.command == 'turn':
            # Simulate turn
            self.get_logger().info(f'Turning {request.value} degrees')
            response.success = True
            response.message = f'Turned {request.value} degrees'
        else:
            response.success = False
            response.message = f'Unknown command: {request.command}'
        
        return response

def main(args=None):
    rclpy.init(args=args)
    robot_command_server = RobotCommandServer()
    
    try:
        rclpy.spin(robot_command_server)
    except KeyboardInterrupt:
        pass
    finally:
        robot_command_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# service_client.py
import rclpy
from rclpy.node import Node
from your_package.srv import RobotCommand  # You'll need to define this service

class RobotCommandClient(Node):
    def __init__(self):
        super().__init__('robot_command_client')
        self.cli = self.create_client(RobotCommand, 'robot_command')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = RobotCommand.Request()

    def send_request(self, command, value):
        self.req.command = command
        self.req.value = float(value)
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        
        return self.future.result()

def main():
    rclpy.init()
    
    robot_command_client = RobotCommandClient()
    
    # Send a move forward command
    response = robot_command_client.send_request('move_forward', 2.5)
    robot_command_client.get_logger().info(
        f'Result: {response.success}, Message: {response.message}'
    )
    
    # Send a turn command
    response = robot_command_client.send_request('turn', 90.0)
    robot_command_client.get_logger().info(
        f'Result: {response.success}, Message: {response.message}'
    )
    
    robot_command_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 3: Action Implementation

#### Objective
Implement a long-running task with feedback using the action pattern.

#### Steps
1. Define an action interface for a navigation task
2. Create an action server that simulates navigation
3. Implement an action client that sends navigation goals

```python
# navigate_to_pose.action
# Goal
geometry_msgs/Pose target_pose
---
# Result
bool success
string message
---
# Feedback
float32 distance_remaining
geometry_msgs/Pose current_pose
```

```python
# navigation_action_server.py
import rclpy
import rclpy.action
from rclpy.node import Node
from geometry_msgs.msg import Pose
from your_package.action import NavigateToPose  # You'll need to define this action

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = rclpy.action.ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        # Accept all goals
        self.get_logger().info('Received navigation goal')
        return rclpy.action.ActionServer.GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept all cancel requests
        self.get_logger().info('Received request to cancel navigation')
        return rclpy.action.CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing navigation goal...')
        
        # Simulate navigation with feedback
        feedback_msg = NavigateToPose.Feedback()
        feedback_msg.distance_remaining = 10.0  # Starting distance
        feedback_msg.current_pose = Pose()  # Placeholder
        
        # Simulate navigation progress
        for i in range(0, 100, 5):  # Simulate 100% progress in 5% increments
            # Check if there was a cancel request
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = NavigateToPose.Result()
                result.success = False
                result.message = 'Navigation canceled'
                self.get_logger().info('Navigation canceled')
                return result
            
            # Update feedback
            feedback_msg.distance_remaining = 10.0 - (i / 10.0)
            goal_handle.publish_feedback(feedback_msg)
            
            self.get_logger().info(f'Navigation progress: {i}%, Distance remaining: {feedback_msg.distance_remaining}')
            
            # Simulate processing time
            await asyncio.sleep(0.5)
        
        # Complete the goal
        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.success = True
        result.message = 'Navigation completed successfully'
        
        self.get_logger().info('Navigation completed successfully')
        return result

def main(args=None):
    rclpy.init(args=args)
    navigation_action_server = NavigationActionServer()
    
    try:
        rclpy.spin(navigation_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 4: Integration Challenge

#### Objective
Combine all communication patterns in a single robotic system.

#### Challenge Description
Create a robot controller system that:
1. Uses topics to broadcast sensor data
2. Uses services for immediate commands (like emergency stop)
3. Uses actions for long-running tasks (like navigation)

#### Implementation Hints
- Design nodes that communicate with each other using the appropriate patterns
- Consider how to coordinate between different communication patterns
- Implement error handling and fault tolerance

### Assessment Criteria

Your implementation will be evaluated on:

1. **Correctness**: All communication patterns work as expected
2. **Code Quality**: Clean, well-commented, and properly structured code
3. **Understanding**: Ability to explain the differences between communication patterns
4. **Integration**: How well different patterns work together in a system

### Troubleshooting Tips

1. **Node Discovery Issues**: Ensure all nodes are on the same ROS domain
2. **Message Type Issues**: Verify that message types match between publishers and subscribers
3. **Service/Action Timing**: Handle timeouts and error conditions appropriately
4. **Threading Issues**: Be aware of thread safety when accessing shared resources

### Extensions for Advanced Students

- Implement custom message types for your specific robot application
- Add security features to your communication patterns
- Implement a system for dynamic reconfiguration using parameters
- Create a visualization tool to monitor the communication patterns

This practical exercise provides hands-on experience with the core communication patterns in ROS 2, which form the foundation of any robotic system.