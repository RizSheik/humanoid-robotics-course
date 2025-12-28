# Module 2: Quiz - ROS 2 Fundamentals: Topics, Services, and Actions

## Quiz Information

**Duration**: 75 minutes  
**Format**: Multiple choice, short answer, and scenario-based questions  
**Topics Covered**: All material from Module 2  
**Resources Allowed**: Lecture notes and textbooks only; no internet resources during quiz

## Section A: Multiple Choice Questions (30 points, 3 points each)

### Question 1
What is the primary difference between ROS 2 topics and ROS 2 services?
a) Topics are asynchronous while services are synchronous
b) Topics use TCP while services use UDP
c) Topics are for robot control while services are for sensor data
d) There is no difference; they are interchangeable

### Question 2
Which Quality of Service (QoS) policy determines how many samples are kept in the history?
a) Reliability policy
b) Durability policy
c) History policy
d) Deadline policy

### Question 3
In ROS 2, what does DDS stand for?
a) Distributed Data System
b) Data Distribution Service
c) Dynamic Discovery System
d) Distributed Discovery Service

### Question 4
What are the three components of a ROS 2 action?
a) Request, response, feedback
b) Goal, feedback, result
c) Start, progress, end
d) Command, status, outcome

### Question 5
Which QoS durability setting ensures that late-joining subscribers receive the most recent data?
a) Volatile
b) Transient Local
c) System Default
d) Best Effort

### Question 6
What is the purpose of a lifecycle node in ROS 2?
a) To manage robot movement patterns
b) To provide a structured way to manage node state transitions
c) To handle robot navigation tasks
d) To implement robot learning algorithms

### Question 7
What happens when a ROS 2 action goal is canceled?
a) The action stops immediately and returns an error
b) The action completes normally and then reports cancellation
c) The action server may complete the current operation but won't start new ones
d) The action is paused until resumed by the client

### Question 8
Which of the following is NOT a standard message type in ROS 2?
a) std_msgs/msg/String
b) geometry_msgs/msg/Twist
c) robot_msgs/msg/JointState
d) sensor_msgs/msg/Image

### Question 9
In ROS 2, what is the main advantage of node composition?
a) It reduces memory usage
b) It allows multiple nodes to run in a single process, reducing communication overhead
c) It increases the number of available topics
d) It simplifies the build system

### Question 10
Which communication pattern is most appropriate for a long-running task that requires feedback and can be canceled?
a) Topic
b) Service
c) Action
d) Parameter

## Section B: Short Answer Questions (40 points, 10 points each)

### Question 11
Explain the differences between ROS 1 and ROS 2 architectures with specific emphasis on communication. What are the main advantages of the ROS 2 approach?

### Question 12
Describe Quality of Service (QoS) settings in ROS 2. Provide examples of when you would use "reliable" vs "best effort" reliability policies in a humanoid robot system.

### Question 13
What are the key characteristics of ROS 2 actions, and how do they differ from services? Provide a specific example of when you would use an action instead of a service in a humanoid robot application.

### Question 14
Explain the concept of namespaces in ROS 2 and why they are important in multi-robot systems. Provide an example of how you would organize topics for a system with two humanoid robots.

## Section C: Scenario-Based Questions (30 points, 15 points each)

### Question 15
You are designing the communication architecture for a humanoid robot with the following subsystems:
- Joint control (20 joints with encoder feedback)
- Inertial measurement unit (IMU)
- Stereo cameras
- Force/torque sensors in both feet
- High-level motion planner
- Balance controller

Design the communication system using appropriate ROS 2 patterns. Specify:
- Which communication pattern (topic/service/action) you would use for each interaction
- At least 3 specific QoS configurations and justification for each
- How the components would be connected to coordinate walking behavior

### Question 16
A humanoid robot needs to perform a complex manipulation task where it must:
1. Navigate to a specific location
2. Locate and grasp an object
3. Move the object to a new location
4. Release the object

Design an appropriate ROS 2 communication system to coordinate this task. Address:
- Which parts of the task would use topics vs services vs actions
- How error handling and recovery would be implemented
- How the system would handle interruption or cancellation of the task

## Answer Key

### Section A Answers:
1. a) Topics are asynchronous while services are synchronous
2. c) History policy
3. b) Data Distribution Service
4. b) Goal, feedback, result
5. b) Transient Local
6. b) To provide a structured way to manage node state transitions
7. c) The action server may complete the current operation but won't start new ones
8. c) robot_msgs/msg/JointState (sensor_msgs/msg/JointState is the correct type)
9. b) It allows multiple nodes to run in a single process, reducing communication overhead
10. c) Action

### Section B Expected Answers:

**Question 11**: ROS 1 used a centralized master architecture with TCPROS/UDPROS for communication, creating a single point of failure. ROS 2 uses DDS with a distributed architecture and no single point of failure. ROS 2 includes built-in security, real-time capabilities, and improved support for multi-robot systems. The advantages include better reliability, security, real-time performance, and production readiness.

**Question 12**: QoS (Quality of Service) settings allow fine-tuning of communication behavior in ROS 2. For joint control in a humanoid robot, use RELIABLE policy for critical commands to ensure delivery. For camera feeds, use BEST_EFFORT policy to maintain frame rate even with occasional packet loss. For IMU data, RELIABLE with bounded deadline is appropriate for safety-critical balance control.

**Question 13**: Actions are for long-running tasks that require feedback and can be canceled. Unlike services, they support continuous feedback and preemption. An action would be used for humanoid walking, where you need to monitor progress, potentially cancel mid-stride, and get feedback about balance status during the walking sequence.

**Question 14**: Namespaces in ROS 2 provide logical grouping of topics, services, and actions. For two humanoid robots, use namespaces like "/robot1/joint_states" and "/robot2/joint_states" to separate the communication for each robot and avoid naming conflicts in multi-robot systems.

### Section C Expected Answers:

**Question 15**: 
- Joint commands: Topic with reliable QoS for control (100Hz)
- Joint feedback: Topic for encoder readings (100Hz)
- IMU: Topic with reliable/volatile QoS (200Hz)
- Camera: Topic with best-effort QoS (30Hz)
- Motion planner: Action for trajectory generation with feedback
- Balance controller: Service for emergency corrections + topics for continuous data
- QoS examples: 
  1. Joint commands: RELIABLE, DEADLINE 10ms (safety-critical)
  2. Camera: BEST_EFFORT, VOLATILE (performance over reliability)
  3. IMU: RELIABLE, DURABILITY TRANSIENT_LOCAL (for recovery)

**Question 16**:
- Navigation: Action (MoveToLocation) with progress feedback
- Object detection: Service (DetectObject) for on-demand processing
- Grasp planning: Action (PlanGrasp) with feedback
- Object carrying: Topic for continuous state monitoring
- Error handling: Each component reports errors to a central service manager that can initiate recovery behaviors
- Cancellation: Action clients can cancel ongoing tasks, with each component having an emergency stop behavior

## Grading Criteria

### Section A (Multiple Choice):
- 3 points per question
- No partial credit
- Only the best answer receives full credit

### Section B (Short Answer):
- Technical understanding: 5 points
- Accuracy of information: 3 points
- Clarity and completeness: 2 points

### Section C (Scenario-Based):
- Appropriate use of communication patterns: 6 points
- Technical feasibility and understanding: 6 points
- Clarity and completeness: 3 points

## Academic Integrity Notice

This quiz is to be completed individually. You may reference your course materials, but you may not discuss quiz content with other students or receive assistance during the quiz period. All work must be your own.

## Preparation Tips

1. Review all four chapters of Module 2
2. Understand the differences between topics, services, and actions
3. Know how to configure QoS settings appropriately for different scenarios
4. Practice designing communication architectures for robotic systems
5. Understand the practical applications of each communication pattern