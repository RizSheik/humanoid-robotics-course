---
id: module-1-weekly-breakdown
title: 'Module 1 — The Robotic Nervous System | Chapter 2 — Weekly Breakdown'
sidebar_label: 'Chapter 2 — Weekly Breakdown'
sidebar_position: 2
---

# Chapter 2 — Weekly Breakdown

## Module 1: The Robotic Nervous System - 8 Week Plan

This module is designed to be completed over 8 weeks, with each week focusing on specific aspects of ROS 2 and the robotic nervous system. Students should plan to spend approximately 8-10 hours per week on this module, including lectures, lab work, and assignments.

### Week 1: Introduction to ROS 2 and Setup

**Topics Covered:**
- History and evolution of ROS to ROS 2
- ROS 2 architecture and concepts
- Installation and environment setup
- Creating your first ROS 2 workspace
- Understanding ROS 2 distributions

**Learning Objectives:**
- Install ROS 2 on your development platform
- Set up the ROS 2 environment
- Understand the differences between ROS and ROS 2
- Create and build a basic ROS 2 workspace

**Practical Lab:**
- Install ROS 2 Humble Hawksbill on Ubuntu 22.04
- Create a workspace and run the basic tutorials
- Verify installation with basic commands

**Reading Assignments:**
- ROS 2 documentation: "Getting Started with ROS 2"
- Chapter 3 of "Programming Robots with ROS" by Morgan Quigley

### Week 2: ROS 2 Nodes and Packages

**Topics Covered:**
- ROS 2 nodes fundamentals
- Creating and organizing packages
- Node communication patterns
- Package build system (colcon)

**Learning Objectives:**
- Create ROS 2 packages using colcon
- Implement simple ROS 2 nodes in Python and C++
- Understand the lifecycle of a ROS 2 node
- Build and run custom ROS 2 packages

**Practical Lab:**
- Create a simple "Hello World" ROS 2 node
- Create a publisher and subscriber node
- Build and run the packages using colcon
- Debug basic node issues

**Reading Assignments:**
- ROS 2 documentation: "Creating Your First ROS 2 Package"
- Research paper: "ROS 2: Why and How We Built It"

### Week 3: Topics and Message Passing

**Topics Covered:**
- Publish-subscribe communication pattern
- Message types and custom messages
- Quality of Service (QoS) settings
- Tools for debugging topics

**Learning Objectives:**
- Implement publishers and subscribers
- Create custom message types
- Configure QoS profiles for different use cases
- Use ROS 2 tools to inspect topic communication

**Practical Lab:**
- Build a publisher-subscriber system for sensor data
- Create a custom message type for robot status
- Experiment with different QoS settings
- Use `ros2 topic` commands for debugging

**Reading Assignments:**
- ROS 2 documentation: "Understanding Topics"
- Research paper: "Quality of Service in ROS 2 Communications"

### Week 4: Services and Actions

**Topics Covered:**
- Service-based communication (request/response)
- Action-based communication (long-running tasks)
- When to use topics vs services vs actions
- Error handling in communication

**Learning Objectives:**
- Implement services for synchronous communication
- Implement actions for long-running tasks
- Choose the appropriate communication pattern
- Handle errors and timeouts in communication

**Practical Lab:**
- Create a service for robot movement commands
- Implement an action server for navigation tasks
- Integrate services and actions with existing nodes
- Test error handling scenarios

**Reading Assignments:**
- ROS 2 documentation: "Writing a Simple Service and Client"
- ROS 2 documentation: "Writing an Action Server and Client"

### Week 5: Parameters and Launch Files

**Topics Covered:**
- Parameter management in ROS 2
- Launch files for complex system startup
- YAML configuration files
- Parameter validation and callbacks

**Learning Objectives:**
- Use parameters to configure node behavior
- Create launch files for multi-node systems
- Manage parameter files in YAML format
- Implement parameter callbacks for dynamic changes

**Practical Lab:**
- Create a parameter server for robot configuration
- Develop launch files for complex system startup
- Use parameter callbacks for dynamic reconfiguration
- Integrate with YAML configuration files

**Reading Assignments:**
- ROS 2 documentation: "Using Parameters in a Class"
- ROS 2 documentation: "Creating a Launch File"

### Week 6: TF Transform System and Navigation

**Topics Covered:**
- TF (Transform) system for coordinate frames
- Robot state publisher
- Navigation concepts in ROS 2
- Integration with sensing and actuation

**Learning Objectives:**
- Work with the TF transform system
- Publish robot state information
- Understand coordinate frame relationships
- Implement basic navigation concepts

**Practical Lab:**
- Create a URDF model for a simple robot
- Publish static and dynamic transforms
- Visualize the robot using RViz2
- Implement a basic navigation stack

**Reading Assignments:**
- ROS 2 documentation: "Using TF2"
- Research paper: "Coordinate Transformations in Robotics"

### Week 7: Real Hardware Integration and Sensor Fusion

**Topics Covered:**
- Integrating real sensors with ROS 2
- Sensor fusion techniques
- Hardware abstraction layers
- Real-time considerations

**Learning Objectives:**
- Interface with real robot hardware
- Implement sensor fusion algorithms
- Handle real-time constraints
- Debug hardware integration issues

**Practical Lab:**
- Connect to real sensors (IMU, LIDAR, cameras)
- Implement sensor fusion for localization
- Deploy nodes to embedded hardware
- Compare simulation vs real hardware behavior

**Reading Assignments:**
- Research paper: "Sensor Fusion in ROS 2 for Localization"
- ROS 2 documentation: "Real-time Programming"

### Week 8: Project Integration and Assessment

**Topics Covered:**
- Integrating all components into a complete system
- Performance optimization
- Testing and debugging complex systems
- Presentation and documentation

**Learning Objectives:**
- Integrate all learned concepts into a cohesive system
- Optimize performance of ROS 2 applications
- Test and debug complex robotic systems
- Document and present the implemented system

**Practical Lab:**
- Complete the module assignment (see Chapter 6)
- Optimize the system for performance
- Test the complete integrated system
- Prepare presentation materials

**Reading Assignments:**
- Best practices for ROS 2 development
- Documentation and testing strategies
- Performance optimization techniques

## Assessment Schedule

- **Week 4**: Mid-module quiz covering Weeks 1-3
- **Week 7**: Project milestone check-in
- **Week 8**: Final project presentation and evaluation

## Additional Resources

- ROS 2 tutorials: https://docs.ros.org/en/humble/Tutorials.html
- ROS Discourse: Community support and discussions
- Robot Ignite Academy: Additional practical exercises
- GitHub repositories: Example code for each week

## Important Notes

- Students are encouraged to start the practical labs early in each week to allow time for debugging
- The hardware lab in Week 7 requires advance booking of robot platforms
- Regular attendance in synchronous sessions is recommended for complex topics
- Office hours are available for additional support with complex integrations