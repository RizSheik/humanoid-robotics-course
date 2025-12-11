---
id: module-1-quiz
title: 'Module 1 — The Robotic Nervous System | Chapter 7 — Quiz'
sidebar_label: 'Chapter 7 — Quiz'
sidebar_position: 7
---

# Chapter 7 — Quiz

## Module 1: The Robotic Nervous System - Assessment

### Quiz Instructions
- This quiz assesses your understanding of ROS 2 concepts from Module 1
- Choose the single best answer for each multiple-choice question
- For short-answer questions, provide concise but complete responses
- Time limit: 90 minutes
- Total points: 100

---

## Section A: Multiple Choice Questions (50 points, 5 points each)

### Question 1
What does DDS stand for in the context of ROS 2 architecture?
A) Dynamic Data System
B) Distributed Data Service
C) Data Distribution Service
D) Device Diagnostic System

### Question 2
Which QoS policy determines whether messages are delivered reliably or on a best-effort basis?
A) History Policy
B) Reliability Policy
C) Durability Policy
D) Lifespan Policy

### Question 3
In ROS 2, what is the purpose of a launch file?
A) To define robot kinematics
B) To store sensor calibration data
C) To start multiple nodes with a single command
D) To visualize robot data in RViz

### Question 4
Which of these is NOT a valid node lifecycle state in ROS 2?
A) Unconfigured
B) Inactive
C) Paused
D) Active

### Question 5
What is the primary difference between ROS 2 services and actions?
A) Services are asynchronous, actions are synchronous
B) Services are for short-term tasks, actions for long-term tasks with feedback
C) Services use topics, actions use parameters
D) No difference, they are interchangeable

### Question 6
In a URDF file, which tag defines the visual properties of a robot link?
A) `<collision>`
B) `<geometry>`
C) `<visual>`
D) `<material>`

### Question 7
Which command is used to build ROS 2 packages?
A) `make build`
B) `catkin_make`
C) `colcon build`
D) `ros2 build`

### Question 8
What does the ROS 2 parameter system allow you to do?
A) Share data between nodes via topics
B) Configure node behavior at runtime
C) Store custom message types
D) Visualize robot frames in RViz

### Question 9
In Gazebo simulation, how are sensors typically implemented?
A) As separate ROS 2 packages
B) As Gazebo plugins
C) As hardware drivers
D) As launch file parameters

### Question 10
Which of the following is true about ROS 2 packages?
A) They can contain only one node
B) They must follow the catkin build system
C) They can contain multiple nodes, libraries, and other resources
D) They cannot depend on other packages

---

## Section B: Short Answer Questions (30 points, 10 points each)

### Question 11
Explain the difference between a publisher-subscriber pattern and a client-service pattern in ROS 2. Provide one example of when you would use each pattern.

### Question 12
Describe the purpose of Quality of Service (QoS) policies in ROS 2. Name and briefly explain two QoS policies that can be configured.

### Question 13
Explain the ROS 2 node lifecycle and describe the primary purpose of each state.

---

## Section C: Application Questions (20 points, 10 points each)

### Question 14
You want to create a robot that responds to sensor data to avoid obstacles while navigating to a goal. Design a ROS 2 node architecture for this system. Specify the nodes you would create, their purposes, and what topics/services they would use to communicate.

### Question 15
A student is trying to simulate a robot in Gazebo but reports that the robot falls through the ground plane. What are three possible causes for this issue and how would you address them?

---

## Answer Key

### Section A: Multiple Choice Answers
1. **C) Data Distribution Service** - DDS (Data Distribution Service) is the middleware standard that ROS 2 is built on top of.

2. **B) Reliability Policy** - The Reliability QoS policy controls whether messages are delivered reliably or on a best-effort basis.

3. **C) To start multiple nodes with a single command** - Launch files allow you to start multiple nodes and configure them with a single command.

4. **C) Paused** - The valid lifecycle states are Unconfigured, Inactive, Active, and Finalized.

5. **B) Services are for short-term tasks, actions for long-term tasks with feedback** - This is the key distinction between services and actions.

6. **C) `<visual>`** - The `<visual>` tag defines how a link appears in simulation and visualization tools.

7. **C) `colcon build`** - Colcon is the build system used in ROS 2.

8. **B) Configure node behavior at runtime** - Parameters allow you to configure node behavior without recompiling.

9. **B) As Gazebo plugins** - Sensors in Gazebo are implemented as plugins that connect to ROS 2.

10. **C) They can contain multiple nodes, libraries, and other resources** - ROS 2 packages are flexible containers for various resources.

### Section B: Short Answer Answers

**Question 11:**
The publisher-subscriber pattern is asynchronous: publishers send messages to topics without knowing who receives them, and subscribers receive messages without knowing who published them. This is ideal for sensor data distribution where multiple nodes might need the same data (e.g., LIDAR scans).
The client-service pattern is synchronous: the client sends a request and waits for a response from the service. This is ideal for operations that return a result in a short time (e.g., setting robot parameters, getting current position).

**Question 12:**
QoS policies allow fine-grained control over message delivery characteristics in ROS 2. They help ensure that different types of data have appropriate delivery guarantees based on their requirements.
Two examples:
- **Reliability Policy**: Controls whether messages are delivered reliably (guaranteed) or on a best-effort basis (may be lost).
- **History Policy**: Controls how many messages are kept in the publisher's queue (KEEP_LAST keeps a fixed number, KEEP_ALL keeps all messages).

**Question 13:**
The ROS 2 node lifecycle consists of four states:
- **Unconfigured**: The node is initialized but not yet configured.
- **Inactive**: The node is configured but not yet activated; resources may be allocated but no processing occurs.
- **Active**: The node is fully operational and processing data.
- **Finalized**: The node has been shut down and resources are cleaned up.

This lifecycle enables complex systems to be managed systematically, especially in safety-critical applications.

### Section C: Application Answers

**Question 14:**
I would design the following nodes:
1. **Navigation Node**: Receives navigation goals as an action, sends velocity commands to the robot controller
2. **Sensor Processing Node**: Reads sensor data (e.g., LIDAR), detects obstacles, publishes obstacle information
3. **Obstacle Avoidance Node**: Subscribes to sensor data, implements obstacle avoidance behavior, sends modified velocity commands
4. **Robot Controller Node**: Subscribes to velocity commands and executes them on the robot

Communication would happen through:
- `/scan` topic for LIDAR data
- `/cmd_vel` topic for velocity commands
- `/navigation/goal` action for navigation requests
- Custom topic for obstacle information

**Question 15:**
Three possible causes:
1. **Missing collision geometries**: The URDF doesn't define collision elements for the robot links. Fix by adding `<collision>` tags to the URDF.
2. **Incorrect mass/inertia values**: The physics properties are not properly defined. Fix by specifying realistic mass and inertia values in the `<inertial>` tags.
3. **Gazebo plugins not properly configured**: The robot isn't properly interfacing with Gazebo's physics engine. Fix by ensuring proper Gazebo plugins are defined in the URDF or SDF file.

---

## Scoring Guidelines

- Section A: Each correct answer = 5 points (50 points total)
- Section B: Each answer graded on completeness and accuracy (10 points max each, 30 points total)
- Section C: Each answer graded on completeness, accuracy, and technical understanding (10 points max each, 20 points total)
- Total: 100 points

### Grade Scale
- A (90-100): Excellent understanding of ROS 2 concepts
- B (80-89): Good understanding with minor gaps
- C (70-79): Adequate understanding with some significant gaps
- D (60-69): Basic understanding with major gaps
- F (Below 60): Insufficient understanding

## Learning Objectives Assessed

This quiz evaluates your understanding of:
1. ROS 2 architecture and DDS foundation
2. Quality of Service policies and their applications
3. Node communication patterns (topics, services, actions)
4. Robot modeling with URDF
5. Simulation concepts with Gazebo
6. Package structure and build systems
7. Parameter management
8. System design and node architecture
9. Troubleshooting common issues

## Review Recommendations

If you scored below your target grade:
- Review ROS 2 architecture and the DDS foundation
- Practice creating and using launch files
- Work with QoS policies in different scenarios
- Strengthen understanding of node lifecycle management
- Gain more experience with robot simulation in Gazebo