# Chapter 1: Introduction to ROS 2 Architecture and Middleware


<div className="robotDiagram">
  <img src="/static/img/book-image/Leonardo_Lightning_XL_Full_architecture_overview_workstation_e_0.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


## Learning Objectives

After completing this chapter, students will be able to:
- Explain the architecture of ROS 2 and how it differs from ROS 1
- Identify the key components of the ROS 2 system
- Understand the role of DDS in ROS 2
- Describe the concept of nodes, packages, and workspaces in ROS 2
- Set up a basic ROS 2 development environment

## 1.1 History and Evolution of ROS

The Robot Operating System (ROS) began as a project at Stanford University in 2007 and was later developed by Willow Garage. Originally designed as a flexible framework for writing robot software, ROS became the de facto standard for robotics development due to its rich set of tools, libraries, and conventions.

ROS 1 was designed with a focus on rapid prototyping and research, using a centralized master architecture with TCPROS/UDPROS as the communication layer. However, as robotics applications became more complex and safety-critical, limitations of ROS 1 became apparent, including:
- Single point of failure with the master node
- Lack of security features
- Real-time limitations
- Non-deterministic behavior
- Limited support for multi-robot systems

ROS 2 was developed to address these limitations, with a focus on:
- Production-readiness
- Real-time capabilities
- Security and authentication
- Improved reliability
- Support for commercial applications
- Better multi-robot systems support

## 1.2 ROS 2 Architecture Overview

ROS 2 uses a distributed architecture based on the Data Distribution Service (DDS) standard. Unlike ROS 1's centralized master, ROS 2 has no single point of failure. Each node discovers other nodes directly using DDS discovery mechanisms.

### 1.2.1 Key Components

**Nodes**: The fundamental execution units in ROS 2. A node is an executable that uses ROS 2 client libraries to communicate with other nodes. Nodes can publish/subscribe to topics, provide/call services, and execute actions.

**DDS (Data Distribution Service)**: A vendor-neutral communications middleware that provides a publish-subscribe model. DDS implementations in ROS 2 handle discovery, data serialization, and communication reliability.

**Packages**: Organize ROS 2 code into reusable, manageable units. Each package contains a `package.xml` manifest and a `CMakeLists.txt` or `setup.py` for build configuration.

**Workspaces**: Directories containing multiple packages, typically organized in a source (`src`) directory.

**Launch files**: Allow starting multiple nodes and configuring parameters together.

### 1.2.2 Client Libraries

ROS 2 supports multiple client libraries for different languages:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: C implementation that other languages can wrap
- Additional language support available through ROS 2 ecosystem

## 1.3 DDS and Its Role in ROS 2

DDS (Data Distribution Service) is a middleware standard for real-time, distributed, and embedded systems. In ROS 2, DDS handles:

- **Discovery**: Automatic discovery of publishers and subscribers
- **Data distribution**: Reliable data delivery
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, deadline, liveliness, etc.
- **Security**: Authentication, access control, and encryption

### 1.3.1 Quality of Service (QoS) Settings

QoS settings allow fine-tuning of communication behavior:

- **Reliability**: Either "best effort" or "reliable"
- **Durability**: Either "volatile" or "transient_local"
- **Deadline**: Time bounds for data delivery
- **Liveliness**: How long to wait for participants to respond
- **History**: How many samples to keep
- **Depth**: Used with history policy to set buffer size

## 1.4 Nodes, Topics, Services, and Actions

### 1.4.1 Nodes

Nodes are the fundamental building blocks of ROS 2. Each node represents a single process that performs computation. Nodes communicate with each other through:
- Topics (pub/sub model)
- Services (request/response model)
- Actions (goal-based model)

### 1.4.2 Topics

Topics enable asynchronous communication between nodes using a publish-subscribe model. Publishers send messages to topics, and subscribers receive messages from topics. Multiple publishers and subscribers can use the same topic.

### 1.4.3 Services

Services provide synchronous request-response communication. A client sends a request to a service server, which processes the request and sends back a response. This is useful for one-off computations or queries.

### 1.4.4 Actions

Actions handle long-running tasks with goal, feedback, and result patterns. Actions are useful for tasks that take time to complete and may need to be preempted or monitored during execution.

## 1.5 Setting Up a ROS 2 Environment

### 1.5.1 Installation

ROS 2 can be installed via packages or source. The recommended approach is installing via packages:

1. Set up locale and sources
2. Add the ROS 2 apt repository
3. Install ROS 2 packages
4. Source the ROS 2 setup script

### 1.5.2 Development Workspace

A typical ROS 2 development workspace follows this structure:
```
workspace_name/        # e.g., ~/ros2_ws
  src/                 # Source code
    package_1/
    package_2/
    ...
```

To create a workspace:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## 1.6 Package Structure

ROS 2 packages contain the following structure:
```
package_name/
  CMakeLists.txt     # Build configuration for C++
  package.xml        # Package manifest
  src/               # Source code
  include/           # Header files (C++)
  launch/            # Launch files
  config/            # Configuration files
  test/              # Test files
```

The `package.xml` file contains metadata about the package:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_package</name>
  <version>0.0.0</version>
  <description>Example package</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## 1.7 Basic ROS 2 Commands

Common ROS 2 commands include:
- `ros2 run package_name executable_name` - Run a node
- `ros2 topic list` - List active topics
- `ros2 service list` - List available services
- `ros2 action list` - List active actions
- `ros2 node list` - List active nodes
- `ros2 launch package_name launch_file.py` - Launch nodes from a file

## 1.8 ROS 2 Tools for Development and Debugging

ROS 2 includes many tools for development and debugging:

- **ros2 topic**: List, inspect, and publish to topics
- **ros2 service**: List and call services
- **ros2 action**: List and send goals to actions
- **rqt**: Graphical user interface for ROS 2
- **rviz2**: 3D visualization tool
- **ros2 bag**: Recorder and player for ROS 2 data
- **ros2 doctor**: Diagnostics tool for ROS 2 system

## 1.9 ROS 2 for Humanoid Robotics

For humanoid robots, ROS 2 architecture provides several key advantages:

1. **Distributed Processing**: Different robot subsystems (perception, planning, control) can run on different computers
2. **Modularity**: Components can be developed and tested independently
3. **Flexibility**: New sensors and capabilities can be integrated easily
4. **Real-time Performance**: With appropriate configuration, ROS 2 can meet real-time requirements
5. **Safety**: Proper QoS settings and security features can enhance safety-critical operations

## Chapter Summary

This chapter introduced the fundamental architecture of ROS 2, highlighting its differences from ROS 1 and its use of DDS for distributed communication. We covered the key components including nodes, packages, and workspaces, and explained how QoS settings allow fine-tuning of communication behavior. The chapter concluded with an overview of how ROS 2 is particularly well-suited for the complex communication needs of humanoid robots.

## Key Terms
- Robot Operating System 2 (ROS 2)
- Data Distribution Service (DDS)
- Quality of Service (QoS)
- Nodes
- Topics
- Services
- Actions
- Client Libraries
- Workspace

## Discussion Questions
1. How does the distributed architecture of ROS 2 improve reliability compared to ROS 1?
2. What are the advantages and disadvantages of using DDS as the middleware?
3. How do Quality of Service settings affect communication in robotic systems?
4. Why is modularity important for humanoid robotics applications?

## References
- ROS 2 Documentation: https://docs.ros.org/
- DDS Specification: https://www.omg.org/spec/DDS/
- Faust, A., Tapus, A., Burdick, J. W., & Matarić, M. J. (2010). Aided navigation for elderly people.
- Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS.