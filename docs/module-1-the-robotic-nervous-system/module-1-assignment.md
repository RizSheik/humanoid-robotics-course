---
id: module-1-assignment
title: 'Module 1 — The Robotic Nervous System | Chapter 6 — Assignment'
sidebar_label: 'Chapter 6 — Assignment'
sidebar_position: 6
---

# Chapter 6 — Assignment

## Module 1: Advanced ROS 2 Robot Control System

### Assignment Overview

In this assignment, you will design and implement a complete ROS 2-based control system for a differential drive robot in simulation. The system must demonstrate your understanding of ROS 2 concepts including nodes, topics, services, actions, parameters, and simulation integration.

### Learning Objectives

By completing this assignment, you will demonstrate the ability to:

- Design a modular ROS 2 package structure
- Implement multiple coordinated ROS 2 nodes
- Handle sensor data and robot control commands
- Create custom message types
- Integrate with Gazebo simulation
- Use ROS 2 tools for debugging and visualization
- Apply Quality of Service (QoS) policies appropriately

### Assignment Requirements

#### 1. Robot Model Creation
Create a more sophisticated robot model with:
- Differential drive base
- RGB-D camera
- IMU sensor
- 2D LIDAR
- At least 2 additional sensors of your choice

#### 2. Node Architecture
Design and implement the following ROS 2 nodes:

1. **Robot Controller Node**:
   - Subscribe to velocity commands (`geometry_msgs/Twist`)
   - Publish sensor data (IMU, LIDAR, camera) to appropriate topics
   - Implement a service for getting robot status
   - Handle parameters for robot configuration

2. **Navigation Node**:
   - Implement an action server for navigation goals
   - Use sensor data to avoid obstacles
   - Provide feedback during navigation

3. **Sensor Processing Node**:
   - Subscribe to sensor data from the robot
   - Process LIDAR data to detect obstacles
   - Process camera data to identify colored objects
   - Publish processed information

4. **Path Planning Node**:
   - Subscribe to navigation goals
   - Plan paths avoiding obstacles
   - Send goals to the navigation action server

5. **Visualization Node**:
   - Publish TF transforms for all robot frames
   - Publish markers for visualization in RViz2

#### 3. Custom Message Types
Create at least 2 custom message types for:
- Robot status information (battery level, error states, etc.)
- Processed sensor data (obstacle information, object detection results)

#### 4. Launch Files
Create launch files that:
- Start the Gazebo simulation with your robot
- Launch all the required nodes
- Set appropriate parameters for your system
- Include an option to run with or without the simulation

#### 5. Configuration and Parameters
Implement parameter configuration for:
- Robot physical properties (wheel radius, base width, etc.)
- Sensor properties (update rates, ranges, etc.)
- Algorithm parameters (navigation, path planning, etc.)
- QoS settings for different topics

### Technical Specifications

#### Robot Model Requirements
Your robot model should include:
- Realistic physical properties
- Proper kinematic constraints
- Collision geometries
- Visual properties

#### Communication Requirements
- Use appropriate QoS policies for different types of data
- Implement proper message types for all communications
- Ensure nodes can handle message loss gracefully
- Use services for synchronous operations and actions for long-running tasks

#### Simulation Requirements
- Robot should respond to velocity commands in simulation
- All sensors should publish realistic data
- Robot should be able to navigate to specified goals
- Obstacle avoidance should be functional

### Implementation Guidelines

#### 1. Package Structure
Organize your code in a single ROS 2 package with the following structure:
```
my_robot_system/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── robot_params.yaml
│   └── sensor_params.yaml
├── launch/
│   ├── robot_sim.launch.py
│   └── robot_real.launch.py
├── msg/
│   ├── RobotStatus.msg
│   └── ProcessedSensorData.msg
├── srv/
├── action/
├── urdf/
│   └── robot.urdf.xacro
├── src/
│   ├── controller_node.cpp
│   ├── navigation_node.cpp
│   ├── sensor_processing_node.cpp
│   ├── path_planning_node.cpp
│   └── visualization_node.cpp
├── include/
├── meshes/
└── worlds/
```

#### 2. Node Implementation
Each node should:
- Properly initialize ROS 2
- Implement proper error handling
- Use appropriate logging
- Follow ROS 2 best practices for node lifecycle
- Be well-documented

#### 3. Testing and Validation
Your system should include:
- Unit tests for critical functions
- Integration tests for node communications
- A README file with instructions for building and running
- Documentation for your custom message types and services

### Evaluation Criteria

Your assignment will be evaluated based on:

1. **Functionality** (40%):
   - All nodes working correctly together
   - Proper robot behavior in simulation
   - Successful navigation to goals
   - Accurate sensor data processing

2. **Code Quality** (25%):
   - Well-structured, modular design
   - Proper adherence to ROS 2 conventions
   - Clean, readable code with appropriate comments
   - Proper error handling and logging

3. **Architecture** (20%):
   - Appropriate use of ROS 2 concepts (topics, services, actions)
   - Efficient communication patterns
   - Proper parameter configuration
   - Good separation of concerns between nodes

4. **Documentation** (15%):
   - Clear README with setup and usage instructions
   - Well-documented code
   - Explanation of design choices
   - Troubleshooting guide

### Submission Requirements

Submit the following:

1. **Complete ROS 2 Package**:
   - All source code files
   - URDF model files
   - Launch files
   - Configuration files
   - Package manifest

2. **Documentation**:
   - README file with setup instructions
   - Architecture diagram
   - Explanation of your design choices
   - Instructions for testing the system

3. **Video Demonstration**:
   - 5-10 minute video showing:
     - System architecture overview
     - Robot model in Gazebo
     - Robot following navigation commands
     - Sensor data processing
     - Obstacle avoidance in action

### Technical Constraints and Guidelines

#### Performance Requirements
- The system should maintain at least 30 Hz update rate for critical control topics
- Navigation planning should complete within 5 seconds for typical environments
- Sensor processing should not introduce more than 100ms latency

#### Architecture Constraints
- All nodes must be in a single ROS 2 package for easy evaluation
- Use standard ROS 2 message types where possible
- Implement proper shutdown procedures in all nodes
- Use appropriate QoS settings for different types of data

#### Quality Requirements
- Follow ROS 2 style guidelines (ament_lint_auto)
- Implement proper error handling and graceful degradation
- Use parameter validation to prevent invalid configurations
- Include unit tests for critical components

### Optional Enhancements (Extra Credit)

For students seeking additional challenge, consider implementing:

1. **Machine Learning Component**:
   - Use sensor data to classify objects in the environment
   - Implement a simple neural network for object detection

2. **Multi-Robot Coordination**:
   - Extend the system to work with multiple robots
   - Implement basic multi-robot coordination behaviors

3. **Advanced Navigation**:
   - Implement exploration behavior
   - Create a map of the environment during navigation
   - Use the map for improved navigation

4. **Hardware Integration**:
   - Document how the system would interface with real hardware
   - Create a hardware abstraction layer

### Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Gazebo Tutorials](http://gazebosim.org/tutorials)
- [Robotics Toolbox for ROS 2](https://github.com/ros/robotics-toolbox)
- [Navigation2 Documentation](https://navigation.ros.org/)

### Deadline

This assignment is due at the end of Week 8 of the module. Late submissions will be penalized at a rate of 5% per day.

### Support

For technical support with this assignment:
- Office Hours: Tuesdays and Thursdays, 2-4 PM
- ROS Discourse: Post questions in the course-specific category
- Slack Channel: #module1-assignments in the course workspace

## Assessment Rubric

| Criteria | Excellent (A) | Good (B) | Satisfactory (C) | Needs Improvement (D) |
|----------|---------------|----------|------------------|----------------------|
| Functionality | All features work flawlessly; exceeds requirements | Most features work well; meets requirements | Basic functionality works; minimal issues | Significant functionality missing or broken |
| Code Quality | Exceptional organization; exemplary coding practices | Good organization; follows best practices | Adequate organization; mostly follows practices | Poor organization; does not follow practices |
| Architecture | Innovative and optimal design; advanced concepts used | Good design; proper use of ROS 2 concepts | Basic but correct architecture | Poor architectural decisions |
| Documentation | Comprehensive and clear; exceeds expectations | Clear and complete documentation | Basic but sufficient documentation | Inadequate or unclear documentation |

## Tips for Success

1. **Start Early**: Begin with the robot model and basic node structure before adding complex functionality.

2. **Iterative Development**: Build and test each component individually before integrating them.

3. **Use ROS 2 Tools**: Leverage `ros2 topic`, `ros2 service`, `ros2 action`, and other tools for debugging.

4. **Simulation First**: Test your system thoroughly in simulation before considering real hardware extensions.

5. **Version Control**: Use Git to track your progress and maintain history of your development process.

6. **Documentation as You Go**: Write documentation as you implement features rather than as an afterthought.

## Conclusion

This assignment serves as a comprehensive test of your understanding of ROS 2 and its application in robotic systems. It requires combining multiple concepts learned throughout the module into a cohesive, working system. The assignment not only tests your technical skills but also your ability to architect a complete robotic system.

By successfully completing this assignment, you will have demonstrated proficiency in ROS 2 development and simulation integration, skills that are essential for advanced robotics development.