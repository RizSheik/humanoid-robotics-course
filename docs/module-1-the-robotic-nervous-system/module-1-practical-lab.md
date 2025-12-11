---
id: module-1-practical-lab
title: 'Module 1 — The Robotic Nervous System | Chapter 4 — Practical Lab'
sidebar_label: 'Chapter 4 — Practical Lab'
sidebar_position: 4
---

# Chapter 4 — Practical Lab

## Setting Up Your ROS 2 Development Environment

In this practical lab, we'll set up a complete ROS 2 development environment and create your first ROS 2 package with nodes, topics, and services.

### Prerequisites

Before starting this lab, ensure you have:
- Ubuntu 22.04 LTS installed (or a compatible system)
- Administrative access to install packages
- Internet connection for downloading ROS 2 packages
- Basic familiarity with the terminal

### Lab 1: Installing ROS 2 Humble Hawksbill

1. **Set up the ROS 2 apt repository**:
```bash
# Add the ROS 2 GPG key
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

2. **Install ROS 2 packages**:
```bash
sudo apt update
sudo apt install ros-humble-desktop
```

3. **Install colcon build tools**:
```bash
sudo apt install python3-colcon-common-extensions
```

4. **Install additional dependencies**:
```bash
sudo apt install python3-rosdep python3-vcstool
sudo rosdep init
rosdep update
```

5. **Source ROS 2 environment**:
```bash
source /opt/ros/humble/setup.bash
```

To make this permanent, add the following line to your `~/.bashrc`:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Lab 2: Creating Your First ROS 2 Package

1. **Create a new workspace**:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

2. **Create a new package**:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_pkg --dependencies rclpy std_msgs geometry_msgs
```

3. **Explore the package structure**:
```bash
cd my_robot_pkg
ls -la
```

You should see the following structure:
```
├── package.xml
├── setup.cfg
├── setup.py
└── my_robot_pkg/
    ├── __init__.py
    └── my_robot_pkg/
        └── __init__.py
```

### Lab 3: Creating a Publisher Node

1. **Create a directory for Python scripts**:
```bash
mkdir ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts
```

2. **Create a publisher node**:
```bash
touch ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/publisher_member_function.py
chmod +x ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/publisher_member_function.py
```

3. **Add the following code to the file**:
```python
#!/usr/bin/env python3
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

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

4. **Update setup.py to make the script executable**:
Edit `~/ros2_ws/src/my_robot_pkg/setup.py` and add to the `entry_points` section:
```python
'console_scripts': [
    'publisher_member_function = my_robot_pkg.scripts.publisher_member_function:main',
],
```

### Lab 4: Creating a Subscriber Node

1. **Create a subscriber script**:
```bash
touch ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/subscriber_member_function.py
chmod +x ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/subscriber_member_function.py
```

2. **Add the following code**:
```python
#!/usr/bin/env python3
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

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

3. **Update setup.py with the subscriber**:
Add another entry to the `console_scripts` list:
```python
'subscriber_member_function = my_robot_pkg.scripts.subscriber_member_function:main',
```

### Lab 5: Building and Running Your Nodes

1. **Build the package**:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
```

2. **Source the workspace**:
```bash
source ~/ros2_ws/install/setup.bash
```

3. **Run the publisher node**:
Open a new terminal, source the workspace, and run:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg publisher_member_function
```

4. **Run the subscriber node**:
Open another terminal, source the workspace, and run:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg subscriber_member_function
```

If successful, you should see the publisher sending messages and the subscriber receiving them.

### Lab 6: Using ROS 2 Tools for Introspection

1. **Check available topics**:
```bash
ros2 topic list
```

2. **Check topic information**:
```bash
ros2 topic info /topic
```

3. **Echo messages from a topic**:
```bash
ros2 topic echo /topic std_msgs/msg/String
```

4. **Monitor node information**:
```bash
ros2 node list
ros2 node info /minimal_publisher
ros2 node info /minimal_subscriber
```

### Lab 7: Creating a Service

1. **Create a service server script**:
```bash
touch ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/service_member_function.py
chmod +x ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/service_member_function.py
```

2. **Add the following code**:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\n')
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

3. **Update setup.py with the service**:
Add to `console_scripts`:
```python
'service_member_function = my_robot_pkg.scripts.service_member_function:main',
```

4. **Create a service client**:
```bash
touch ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/client_member_function.py
chmod +x ~/ros2_ws/src/my_robot_pkg/my_robot_pkg/scripts/client_member_function.py
```

5. **Add the following code**:
```python
#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    
    minimal_client.get_logger().info(
        f'Result of add_two_ints: {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

6. **Update setup.py with the client**:
Add to `console_scripts`:
```python
'client_member_function = my_robot_pkg.scripts.client_member_function:main',
```

### Lab 8: Testing Your Service

1. **Rebuild the package**:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
```

2. **Run the service server**:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg service_member_function
```

3. **In another terminal, call the service**:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run my_robot_pkg client_member_function 2 3
```

You should see the response showing the sum of the two integers (5 in this case).

### Lab 9: Using Launch Files

1. **Create a launch directory**:
```bash
mkdir -p ~/ros2_ws/src/my_robot_pkg/launch
```

2. **Create a launch file**:
```bash
touch ~/ros2_ws/src/my_robot_pkg/launch/two_nodes_launch.py
```

3. **Add the following launch file code**:
```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_pkg',
            executable='publisher_member_function',
            name='publisher_node'
        ),
        Node(
            package='my_robot_pkg',
            executable='subscriber_member_function',
            name='subscriber_node'
        )
    ])
```

4. **Update setup.py to include launch files**:
Add to the `data_files` section:
```python
(os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
```

5. **Rebuild and run with the launch file**:
```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_pkg
source ~/ros2_ws/install/setup.bash
ros2 launch my_robot_pkg two_nodes_launch.py
```

### Lab 10: Creating Custom Messages

1. **Create a directory for custom messages**:
```bash
mkdir ~/ros2_ws/src/my_robot_pkg/msg
```

2. **Create a custom message definition**:
```bash
echo "float64 x
float64 y
float64 z
string name" > ~/ros2_ws/src/my_robot_pkg/msg/CustomPose.msg
```

3. **Update package.xml to include message dependencies**:
Add to the `package.xml`:
```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

4. **Update setup.py to generate custom messages**:
Add to the imports at the top:
```python
from glob import glob
```

Add to the `setup()` function:
```python
'rosidl_interface_packages': ['msg/CustomPose.msg'],
```

5. **Rebuild the package**:
```bash
cd ~/ros2_ws
ros2 run my_robot_pkg publisher_member_function
```

## Troubleshooting Common Issues

### Issue: Permission denied when sourcing setup.bash
**Solution**: Ensure correct syntax when adding to .bashrc:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### Issue: Command not found when running ros2 commands
**Solution**: Make sure to source both the ROS 2 installation and your workspace:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Issue: Package not found when building
**Solution**: Ensure you're in the workspace root directory and all dependencies are installed:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

## Lab Assessment

Complete the following tasks to demonstrate your understanding:

1. Successfully create a ROS 2 package with custom nodes
2. Implement a publisher-subscriber pair that communicates properly
3. Create and test a service-server pair
4. Use launch files to start multiple nodes simultaneously
5. Create and use a custom message type

## Summary

This practical lab has walked you through the fundamental steps of setting up a ROS 2 development environment and creating basic ROS 2 applications. You've learned how to:
- Install ROS 2 and set up your environment
- Create and build custom ROS 2 packages
- Implement publishers and subscribers
- Create services and clients
- Use launch files for complex system startup
- Define and use custom message types

These skills form the foundation for developing more complex robotic systems in subsequent modules.