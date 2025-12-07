---
id: module-1-quiz
title: Module 1 — The Robotic Nervous System | Chapter 7 — Quiz
sidebar_label: Chapter 7 — Quiz
sidebar_position: 7
---

# Module 1 — The Robotic Nervous System

## Chapter 7 — Quiz

### Multiple Choice Questions

1. What does ROS stand for?
   a) Robot Operating System
   b) Robotic Operating Software
   c) Robot Operation System
   d) Robotic Operational System

   **Answer: a) Robot Operating System**

2. Which tool is used to visualize the ROS computation graph?
   a) rqt
   b) rviz
   c) roscore
   d) rostopic

   **Answer: a) rqt**

3. What is the primary function of a ROS node?
   a) To store sensor data
   b) To perform computations and/or interact with the hardware
   c) To visualize robot data
   d) To manage robot movements

   **Answer: b) To perform computations and/or interact with the hardware**

4. In ROS 2, what is the default middleware used for communication?
   a) TCPROS
   b) UDPROS
   c) DDS (Data Distribution Service)
   d) MQTT

   **Answer: c) DDS (Data Distribution Service)**

5. Which command is used to launch a ROS package?
   a) rosrun
   b) roslaunch
   c) rosrun launch
   d) roslauncher

   **Answer: b) roslaunch**

6. What is a ROS topic used for?
   a) Storing robot configuration
   b) Synchronous request/reply communication
   c) Asynchronous message passing between nodes
   d) Defining robot geometry

   **Answer: c) Asynchronous message passing between nodes**

7. Which of the following is NOT a common message type in ROS?
   a) std_msgs
   b) sensor_msgs
   c) geometry_msgs
   d) web_msgs

   **Answer: d) web_msgs**

8. What is the main advantage of using DDS in ROS 2 compared to ROS 1's TCPROS?
   a) Better visualization capabilities
   b) Platform independence and better real-time performance
   c) Simpler API
   d) Better debugging tools

   **Answer: b) Platform independence and better real-time performance**

9. Which ROS tool is used for recording and playing back data?
   a) rosbag
   b) rosservice
   c) rosnode
   d) rosparam

   **Answer: a) rosbag**

10. What is a ROS service used for?
    a) Continuous data streaming
    b) Synchronous request/reply communication
    c) Robot visualization
    d) Coordinate transformations

    **Answer: b) Synchronous request/reply communication**

### Short Answer Questions

11. Explain the difference between ROS topics and ROS services.

**Answer:**
ROS topics enable asynchronous, many-to-many communication through publish-subscribe pattern, suitable for continuous data streaming. ROS services enable synchronous, request-reply communication between a client and a server, suitable for tasks requiring immediate responses.

12. Describe the purpose of tf (transform) in ROS and why it's important in robotics.

**Answer:**
tf (transform) is a package that enables tracking of coordinate frames in a distributed system over time. It's important in robotics because it handles the complexity of keeping track of multiple coordinate frames, allowing different parts of a robot to understand where they are in relation to other parts and the environment.

13. What are ROS Actions and how do they differ from services?

**Answer:**
ROS Actions are a more complex communication pattern that allows for long-running tasks, with support for goal requests, feedback during execution, and result reporting. Unlike services, actions can be preempted during execution and provide feedback during the process.

14. Explain the concept of ROS packages and ROS workspaces.

**Answer:**
A ROS package is the smallest unit of organization in ROS, containing nodes, libraries, data, and configuration files. A ROS workspace is a directory containing several packages and their build artifacts, allowing for organized development and compilation of multiple packages together.

15. What is the role of roscore in ROS 1 and how does the architecture differ in ROS 2?

**Answer:**
In ROS 1, roscore is a master process that enables communication between nodes, providing the ROS Master (name resolution) and parameter server. In ROS 2, there is no roscore because the DDS layer handles peer-to-peer communication and discovery automatically, providing better scalability and fault tolerance.

### Practical Exercise Questions

16. You need to create a ROS node that publishes temperature sensor data at 10 Hz. Outline the key steps to implement this.

**Answer:**
1. Create a new ROS package with catkin_create_pkg command
2. Write a publisher node in C++ or Python
3. Initialize the ROS node with rospy.init_node() or rclcpp::init()
4. Create a publisher object for the sensor_msgs/Temperature message type
5. Create a loop running at 10 Hz using rospy.Rate() or equivalent
6. Create Temperature message objects with appropriate data
7. Publish messages using the publisher object
8. Use ros::spin() or rospy.spin() to keep the node running

17. Describe how to set up a launch file to start multiple ROS nodes together.

**Answer:**
1. Create a .launch file in the launch directory of your package
2. Use XML syntax with `<launch>` as the root element
3. Add `<node>` tags for each node you want to launch, specifying name, pkg, type, args
4. Optionally add parameters using `<param>` tags
5. Launch the file using roslaunch package_name filename.launch

18. How would you visualize sensor data from a robot in RViz?

**Answer:**
1. Start roscore and the nodes publishing sensor data
2. Launch RViz with rosrun rviz rviz
3. Add displays for the specific topics (e.g., PointCloud2 for 3D data, LaserScan for laser data)
4. Configure the Fixed Frame to match the robot's coordinate system
5. Adjust visualization parameters like size, color, and filters as needed
6. Save the configuration as a .rviz file for future use

19. In ROS 2, explain how to create a custom message type.

**Answer:**
1. Create a msg directory in your ROS package
2. Define the message format in a .msg file with field types and names
3. Add the message definition to CMakeLists.txt using ament_export_dependencies
4. Update package.xml with build and execution dependencies
5. Build the package with colcon build
6. Use the new message type in your nodes by importing it

20. How do you use the ROS parameter server to configure your nodes?

**Answer:**
Parameters in ROS allow configuration of nodes at runtime. To use the parameter server:
1. Set parameters via command line using rosparam set
2. Load parameters from YAML files using rosparam load
3. Access parameters in nodes using rospy.get_param() or ros::param::get()
4. Set default parameters in launch files using `<param>` tags
5. Use parameter callbacks to react to parameter changes during execution