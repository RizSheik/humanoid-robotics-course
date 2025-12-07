---
id: module-1-practical-lab
title: Module 1 — The Robotic Nervous System | Chapter 4 — Practical Lab
sidebar_label: Chapter 4 — Practical Lab
sidebar_position: 4
---

# Module 1 — The Robotic Nervous System

## Chapter 4 — Practical Lab

### Laboratory Setup and Requirements

This practical lab module focuses on implementing the core components of the robotic nervous system using ROS 2. To complete these exercises, you will need:

#### Hardware Requirements
- Computer with minimum 8GB RAM, multi-core processor
- Ubuntu 22.04 LTS or equivalent Linux distribution
- Network connection for package installation
- (Optionally) Access to a humanoid robot or simulation environment

#### Software Requirements
- ROS 2 Humble Hawksbill installation
- Gazebo simulation environment
- Git for version control
- Basic development tools (gcc, g++, cmake)

### Lab Exercise 1: ROS 2 Workspace and Basic Node Creation

#### Objective
Create a ROS 2 workspace and implement a simple publisher-subscriber system.

#### Step-by-Step Instructions

1. **Install ROS 2 Humble** (if not already installed):
   ```bash
   # Add ROS 2 GPG key and repository
   sudo apt update && sudo apt install -y curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   sudo apt update
   sudo apt install ros-humble-desktop
   # Install colcon and other tools
   sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
   # Initialize rosdep
   sudo rosdep init
   rosdep update
   # Source ROS 2
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Create a workspace**:
   ```bash
   mkdir -p ~/robotics_ws/src
   cd ~/robotics_ws
   ```

3. **Create a package**:
   ```bash
   cd ~/robotics_ws/src
   ros2 pkg create --build-type ament_python robot_nervous_system_tutorial
   ```

4. **Implement a publisher node** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/talker.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class Talker(Node):
       def __init__(self):
           super().__init__('talker')
           self.publisher_ = self.create_publisher(String, 'robot_status', 10)
           timer_period = 0.5  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Robot status: operational, counter: {self.i}'
           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       talker = Talker()
       rclpy.spin(talker)
       talker.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

5. **Implement a subscriber node** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/listener.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class Listener(Node):
       def __init__(self):
           super().__init__('listener')
           self.subscription = self.create_subscription(
               String,
               'robot_status',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'I heard: "{msg.data}"')

   def main(args=None):
       rclpy.init(args=args)
       listener = Listener()
       rclpy.spin(listener)
       listener.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

6. **Update package.xml** in `~/robotics_ws/src/robot_nervous_system_tutorial/package.xml`:
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>robot_nervous_system_tutorial</name>
     <version>0.0.0</version>
     <description>ROS 2 tutorial package for humanoid robotics nervous system</description>
     <maintainer email="user@example.com">User</maintainer>
     <license>Apache-2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

7. **Update setup.py** in `~/robotics_ws/src/robot_nervous_system_tutorial/setup.py`:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nervous_system_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='user',
       maintainer_email='user@example.com',
       description='ROS 2 tutorial package for humanoid robotics nervous system',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'talker = robot_nervous_system_tutorial.talker:main',
               'listener = robot_nervous_system_tutorial.listener:main',
           ],
       },
   )
   ```

8. **Build the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nervous_system_tutorial
   source install/setup.bash
   ```

9. **Run the publisher and subscriber in separate terminals**:
   Terminal 1:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial talker
   ```

   Terminal 2:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial listener
   ```

#### Expected Results
You should see the talker publishing messages and the listener receiving them, demonstrating basic topic communication.

### Lab Exercise 2: Implementing Services for Robot Control

#### Objective
Create a service that allows changing robot parameters dynamically.

#### Step-by-Step Instructions

1. **Create a custom service definition** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/srv/UpdateRobotParams.srv`:
   ```
   # Request
   float64 speed # Desired speed parameter
   float64 sensitivity # Sensitivity parameter
   string component # Component to update (e.g., "arm", "leg", "head")

   ---

   # Response
   bool success # True if update was successful
   string message # Status message
   ```

2. **Implement a service server** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/param_server.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from robot_nervous_system_tutorial.srv import UpdateRobotParams

   class ParamServer(Node):
       def __init__(self):
           super().__init__('param_server')
           self.srv = self.create_service(
               UpdateRobotParams,
               'update_robot_params',
               self.update_params_callback
           )
           # Initialize robot parameters
           self.robot_params = {
               'arm_speed': 0.5,
               'leg_sensitivity': 0.8,
               'head_sensitivity': 0.9
           }
           self.get_logger().info('Parameter server started')

       def update_params_callback(self, request, response):
           self.get_logger().info(f'Request to update {request.component} with speed={request.speed}, sensitivity={request.sensitivity}')

           if request.component.lower() == 'arm':
               self.robot_params['arm_speed'] = request.speed
               response.success = True
               response.message = f'Updated arm speed to {request.speed}'
           elif request.component.lower() == 'leg':
               self.robot_params['leg_sensitivity'] = request.sensitivity
               response.success = True
               response.message = f'Updated leg sensitivity to {request.sensitivity}'
           elif request.component.lower() == 'head':
               self.robot_params['head_sensitivity'] = request.sensitivity
               response.success = True
               response.message = f'Updated head sensitivity to {request.sensitivity}'
           else:
               response.success = False
               response.message = f'Unknown component: {request.component}'

           self.get_logger().info(f'Response: {response.message}')
           return response

   def main(args=None):
       rclpy.init(args=args)
       param_server = ParamServer()
       rclpy.spin(param_server)
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Implement a service client** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/param_client.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from robot_nervous_system_tutorial.srv import UpdateRobotParams

   class ParamClient(Node):
       def __init__(self):
           super().__init__('param_client')
           self.cli = self.create_client(UpdateRobotParams, 'update_robot_params')
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')
           self.req = UpdateRobotParams.Request()

       def send_request(self, speed, sensitivity, component):
           self.req.speed = speed
           self.req.sensitivity = sensitivity
           self.req.component = component
           self.future = self.cli.call_async(self.req)
           rclpy.spin_until_future_complete(self, self.future)
           return self.future.result()

   def main(args=None):
       rclpy.init(args=args)

       param_client = ParamClient()

       # Example: Update arm speed
       response = param_client.send_request(0.75, 0.0, 'arm')
       if response:
           param_client.get_logger().info(f'Result: {response.success} - {response.message}')

       # Example: Update leg sensitivity
       response = param_client.send_request(0.0, 0.9, 'leg')
       if response:
           param_client.get_logger().info(f'Result: {response.success} - {response.message}')

       param_client.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py** to include the service:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nervous_system_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           # Add this line to include the service definition
           (os.path.join('share', package_name, 'srv'), glob('robot_nervous_system_tutorial/srv/*.srv')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='user',
       maintainer_email='user@example.com',
       description='ROS 2 tutorial package for humanoid robotics nervous system',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'talker = robot_nervous_system_tutorial.talker:main',
               'listener = robot_nervous_system_tutorial.listener:main',
               'param_server = robot_nervous_system_tutorial.param_server:main',
               'param_client = robot_nervous_system_tutorial.param_client:main',
           ],
       },
   )
   ```

5. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nervous_system_tutorial
   source install/setup.bash
   ```

6. **Run the service server and client**:
   Terminal 1:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial param_server
   ```

   Terminal 2:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial param_client
   ```

#### Expected Results
The client should be able to update robot parameters via the service, and the server should confirm successful updates.

### Lab Exercise 3: TF2 with Robot URDF Model

#### Objective
Create a URDF model of a simple humanoid robot and implement TF2 broadcasters to visualize its state.

#### Step-by-Step Instructions

1. **Create a URDF file** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/urdf/simple_humanoid.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_humanoid">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <box size="0.3 0.2 0.5"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.3 0.2 0.5"/>
         </geometry>
       </collision>
     </link>

     <!-- Head -->
     <link name="head">
       <visual>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
         <material name="white">
           <color rgba="1 1 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
       </collision>
     </link>

     <joint name="head_joint" type="fixed">
       <parent link="base_link"/>
       <child link="head"/>
       <origin xyz="0 0 0.35"/>
     </joint>

     <!-- Left Arm -->
     <link name="left_upper_arm">
       <visual>
         <geometry>
           <cylinder length="0.2" radius="0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.2" radius="0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_shoulder" type="revolute">
       <parent link="base_link"/>
       <child link="left_upper_arm"/>
       <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <link name="left_lower_arm">
       <visual>
         <geometry>
           <cylinder length="0.2" radius="0.04"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.2" radius="0.04"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_elbow" type="revolute">
       <parent link="left_upper_arm"/>
       <child link="left_lower_arm"/>
       <origin xyz="0 0 -0.2" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="0" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <!-- Right Arm -->
     <link name="right_upper_arm">
       <visual>
         <geometry>
           <cylinder length="0.2" radius="0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.2" radius="0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_shoulder" type="revolute">
       <parent link="base_link"/>
       <child link="right_upper_arm"/>
       <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <link name="right_lower_arm">
       <visual>
         <geometry>
           <cylinder length="0.2" radius="0.04"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.2" radius="0.04"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_elbow" type="revolute">
       <parent link="right_upper_arm"/>
       <child link="right_lower_arm"/>
       <origin xyz="0 0 -0.2" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="0" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <!-- Left Leg -->
     <link name="left_upper_leg">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.06"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.06"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_hip" type="revolute">
       <parent link="base_link"/>
       <child link="left_upper_leg"/>
       <origin xyz="0.05 0 -0.25" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.78" upper="0.78" effort="100" velocity="1"/>
     </joint>

     <link name="left_lower_leg">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="left_knee" type="revolute">
       <parent link="left_upper_leg"/>
       <child link="left_lower_leg"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="0" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <!-- Right Leg -->
     <link name="right_upper_leg">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.06"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.06"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_hip" type="revolute">
       <parent link="base_link"/>
       <child link="right_upper_leg"/>
       <origin xyz="-0.05 0 -0.25" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.78" upper="0.78" effort="100" velocity="1"/>
     </joint>

     <link name="right_lower_leg">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.05"/>
         </geometry>
       </collision>
     </link>

     <joint name="right_knee" type="revolute">
       <parent link="right_upper_leg"/>
       <child link="right_lower_leg"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="0" upper="1.57" effort="100" velocity="1"/>
     </joint>
   </robot>
   ```

2. **Create a TF2 broadcaster node** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/robot_state_publisher.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from tf2_ros import TransformBroadcaster
   from geometry_msgs.msg import TransformStamped
   import math

   class RobotStatePublisher(Node):
       def __init__(self):
           super().__init__('robot_state_publisher')
           self.tf_broadcaster = TransformBroadcaster(self)
           self.timer = self.create_timer(0.05, self.broadcast_transforms)  # 20 Hz
           self.time = 0.0

       def broadcast_transforms(self):
           # Update time for animation
           self.time += 0.05

           # Base link (for this example, we'll keep it at the origin)
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'world'
           t.child_frame_id = 'base_link'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = 0.0
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = 0.0
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = 1.0
           self.tf_broadcaster.sendTransform(t)

           # Head (fixed relative to base)
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'head'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = 0.35
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = 0.0
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = 1.0
           self.tf_broadcaster.sendTransform(t)

           # Left arm with animated joints
           # Shoulder movement
           shoulder_angle = math.sin(self.time) * 0.5
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'left_upper_arm'
           t.transform.translation.x = 0.15
           t.transform.translation.y = 0.0
           t.transform.translation.z = 0.1
           # Convert angle to quaternion
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(shoulder_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(shoulder_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           # Elbow movement
           elbow_angle = math.sin(self.time * 1.5) * 0.5
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'left_upper_arm'
           t.child_frame_id = 'left_lower_arm'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.2
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(elbow_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(elbow_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           # Right arm with animated joints
           # Shoulder movement (opposite phase to left arm)
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'right_upper_arm'
           t.transform.translation.x = -0.15
           t.transform.translation.y = 0.0
           t.transform.translation.z = 0.1
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(shoulder_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(shoulder_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'right_upper_arm'
           t.child_frame_id = 'right_lower_arm'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.2
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(elbow_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(elbow_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           # Left leg
           hip_angle = math.sin(self.time * 0.8) * 0.3
           knee_angle = math.sin(self.time * 0.8 + 1.0) * 0.4
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'left_upper_leg'
           t.transform.translation.x = 0.05
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.25
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(hip_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(hip_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'left_upper_leg'
           t.child_frame_id = 'left_lower_leg'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.3
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(knee_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(knee_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           # Right leg (opposite phase to left leg)
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'right_upper_leg'
           t.transform.translation.x = -0.05
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.25
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(hip_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(hip_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'right_upper_leg'
           t.child_frame_id = 'right_lower_leg'
           t.transform.translation.x = 0.0
           t.transform.translation.y = 0.0
           t.transform.translation.z = -0.3
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = math.sin(knee_angle/2.0)
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = math.cos(knee_angle/2.0)
           self.tf_broadcaster.sendTransform(t)

   def main(args=None):
       rclpy.init(args=args)
       node = RobotStatePublisher()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Update setup.py** to include the URDF:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nervous_system_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'srv'), glob('robot_nervous_system_tutorial/srv/*.srv')),
           # Add URDF files
           (os.path.join('share', package_name, 'urdf'), glob('robot_nervous_system_tutorial/urdf/*.urdf')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='user',
       maintainer_email='user@example.com',
       description='ROS 2 tutorial package for humanoid robotics nervous system',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'talker = robot_nervous_system_tutorial.talker:main',
               'listener = robot_nervous_system_tutorial.listener:main',
               'param_server = robot_nervous_system_tutorial.param_server:main',
               'param_client = robot_nervous_system_tutorial.param_client:main',
               'robot_state_publisher = robot_nervous_system_tutorial.robot_state_publisher:main',
           ],
       },
   )
   ```

4. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nervous_system_tutorial
   source install/setup.bash
   ```

5. **Run the robot state publisher and visualize in RViz**:
   Terminal 1:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial robot_state_publisher
   ```

   Terminal 2:
   ```bash
   source /opt/ros/humble/setup.bash
   rviz2
   ```

   In RViz:
   - Add a RobotModel display
   - Set Fixed Frame to 'world'
   - You should see the animated humanoid robot in RViz

#### Expected Results
You should see a simple humanoid robot model in RViz with animated joints moving in a coordinated pattern.

### Lab Exercise 4: Quality of Service (QoS) Optimization

#### Objective
Implement different QoS policies for different types of robot data streams.

#### Step-by-Step Instructions

1. **Create a QoS demonstration node** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/qos_demo.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Float32
   from sensor_msgs.msg import Image, Joy
   import time

   class QoSDemo(Node):
       def __init__(self):
           super().__init__('qos_demo')

           # Publisher with different QoS for different data types

           # Critical control command - Reliable, Transient Local
           self.critical_pub = self.create_publisher(
               String,
               'critical_command',
               rclpy.qos.QoSProfile(
                   depth=10,
                   reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                   durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                   history=rclpy.qos.HistoryPolicy.KEEP_LAST
               )
           )

           # Sensor data - Best Effort, Volatile (for performance)
           self.sensor_pub = self.create_publisher(
               Image,
               'sensor_data',
               rclpy.qos.QoSProfile(
                   depth=1,
                   reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                   durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                   history=rclpy.qos.HistoryPolicy.KEEP_LAST
               )
           )

           # Parameter change - Reliable, Transient Local
           self.param_pub = self.create_publisher(
               Float32,
               'parameter_change',
               rclpy.qos.QoSProfile(
                   depth=5,
                   reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                   durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                   history=rclpy.qos.HistoryPolicy.KEEP_LAST
               )
           )

           # Timer to send messages with different QoS
           self.timer = self.create_timer(1.0, self.publish_messages)
           self.message_count = 0

       def publish_messages(self):
           # Publish critical command
           critical_msg = String()
           critical_msg.data = f'Critical command #{self.message_count}'
           self.critical_pub.publish(critical_msg)
           self.get_logger().info(f'Published critical command: {critical_msg.data}')

           # Publish sensor data (simulated)
           sensor_msg = Image()
           sensor_msg.height = 480
           sensor_msg.width = 640
           sensor_msg.encoding = 'rgb8'
           sensor_msg.data = b'fake_image_data'  # Simulated image data
           sensor_msg.header.stamp = self.get_clock().now().to_msg()
           sensor_msg.header.frame_id = 'camera_frame'
           self.sensor_pub.publish(sensor_msg)
           self.get_logger().info('Published sensor data')

           # Publish parameter change
           param_msg = Float32()
           param_msg.data = 1.0 + (self.message_count * 0.1)
           self.param_pub.publish(param_msg)
           self.get_logger().info(f'Published parameter change: {param_msg.data}')

           self.message_count += 1

   def main(args=None):
       rclpy.init(args=args)
       qos_demo = QoSDemo()
       rclpy.spin(qos_demo)
       qos_demo.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Update setup.py** to include the new script:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nervous_system_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'srv'), glob('robot_nervous_system_tutorial/srv/*.srv')),
           (os.path.join('share', package_name, 'urdf'), glob('robot_nervous_system_tutorial/urdf/*.urdf')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='user',
       maintainer_email='user@example.com',
       description='ROS 2 tutorial package for humanoid robotics nervous system',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'talker = robot_nervous_system_tutorial.talker:main',
               'listener = robot_nervous_system_tutorial.listener:main',
               'param_server = robot_nervous_system_tutorial.param_server:main',
               'param_client = robot_nervous_system_tutorial.param_client:main',
               'robot_state_publisher = robot_nervous_system_tutorial.robot_state_publisher:main',
               'qos_demo = robot_nervous_system_tutorial.qos_demo:main',
           ],
       },
   )
   ```

3. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nervous_system_tutorial
   source install/setup.bash
   ```

4. **Run the QoS demo**:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 run robot_nervous_system_tutorial qos_demo
   ```

#### Expected Results
The node will publish different types of data with appropriate QoS settings, demonstrating how to optimize communication for different use cases in humanoid robotics.

### Lab Exercise 5: Integration and Testing

#### Objective
Create a complete system that integrates all components learned in this module and test it thoroughly.

#### Step-by-Step Instructions

1. **Create a launch file** - Create `~/robotics_ws/src/robot_nervous_system_tutorial/robot_nervous_system_tutorial/launch/integration_demo.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os

   def generate_launch_description():
       return LaunchDescription([
           # Launch the talker node
           Node(
               package='robot_nervous_system_tutorial',
               executable='talker',
               name='robot_status_publisher',
               output='screen'
           ),

           # Launch the parameter server
           Node(
               package='robot_nervous_system_tutorial',
               executable='param_server',
               name='robot_param_server',
               output='screen'
           ),

           # Launch the robot state publisher with TF
           Node(
               package='robot_nervous_system_tutorial',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               output='screen'
           ),

           # Launch the QoS demo
           Node(
               package='robot_nervous_system_tutorial',
               executable='qos_demo',
               name='qos_demo',
               output='screen'
           ),
       ])
   ```

2. **Update setup.py** to include launch files:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'robot_nervous_system_tutorial'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name, 'srv'), glob('robot_nervous_system_tutorial/srv/*.srv')),
           (os.path.join('share', package_name, 'urdf'), glob('robot_nervous_system_tutorial/urdf/*.urdf')),
           (os.path.join('share', package_name, 'launch'), glob('robot_nervous_system_tutorial/launch/*.launch.py')),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='user',
       maintainer_email='user@example.com',
       description='ROS 2 tutorial package for humanoid robotics nervous system',
       license='Apache-2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'talker = robot_nervous_system_tutorial.talker:main',
               'listener = robot_nervous_system_tutorial.listener:main',
               'param_server = robot_nervous_system_tutorial.param_server:main',
               'param_client = robot_nervous_system_tutorial.param_client:main',
               'robot_state_publisher = robot_nervous_system_tutorial.robot_state_publisher:main',
               'qos_demo = robot_nervous_system_tutorial.qos_demo:main',
           ],
       },
   )
   ```

3. **Rebuild the package**:
   ```bash
   cd ~/robotics_ws
   colcon build --packages-select robot_nervous_system_tutorial
   source install/setup.bash
   ```

4. **Run the integrated system**:
   ```bash
   source ~/robotics_ws/install/setup.bash
   ros2 launch robot_nervous_system_tutorial integration_demo.launch.py
   ```

#### Expected Results
All nodes should start successfully and work together as an integrated robotic nervous system, with each component using appropriate communication patterns and QoS settings.

### Troubleshooting Guide

#### Common Issues and Solutions

1. **Package Not Found**:
   - Ensure you've sourced the install setup file: `source install/setup.bash`
   - Verify the package was built correctly: `colcon build`

2. **TF Tree Issues**:
   - Check that all required transforms are being published
   - Use `ros2 run tf2_tools view_frames` to visualize the transform tree
   - Ensure proper timing between transforms

3. **Communication Problems**:
   - Verify that topics/services are matching between publishers and subscribers
   - Check QoS settings compatibility between nodes
   - Use `ros2 topic list` and `ros2 service list` to verify available endpoints

4. **Performance Issues**:
   - Monitor CPU and memory usage
   - Adjust QoS settings appropriately
   - Consider node composition for frequently communicating nodes

### Conclusion

This practical lab has provided hands-on experience with implementing the core components of a robotic nervous system using ROS 2. You have learned to:

- Create and build ROS 2 packages
- Implement publisher-subscriber communication patterns
- Design and use custom services
- Work with TF2 for coordinate frame management
- Apply appropriate QoS policies for different data types
- Integrate multiple components into a cohesive system

These skills form the foundation for building more complex humanoid robotics systems in the modules that follow.