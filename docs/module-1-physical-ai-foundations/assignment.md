# Module 1: Assignment - Physical AI and Embodied Intelligence Applications

## Assignment Overview

This assignment challenges students to apply the concepts of Physical AI and embodied intelligence to solve a robotics problem. Students will design, implement, and analyze an embodied robotic system that demonstrates key principles from this module.


<div className="robotDiagram">
  <img src="/static/img/book-image/Ultrarealistic_textbook_cover_design_for_0.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


### Learning Objectives

After completing this assignment, students will be able to:
1. Design a robotic system that leverages physical embodiment for enhanced performance
2. Implement an embodied intelligence approach to solving a robotics problem
3. Analyze the advantages and limitations of embodied approaches
4. Compare embodied solutions to traditional approaches
5. Document and present the design and implementation of an embodied system

### Assignment Components

This assignment consists of three main components:
1. **Design Document** (30%): A detailed document outlining the system design
2. **Implementation** (50%): Implementation of the embodied system in simulation
3. **Analysis Report** (20%): Analysis of the implementation and comparison with alternatives

## Assignment Brief

Design and implement an embodied robotic system that demonstrates one or more of the following principles:
- Morphological computation
- Sensorimotor learning
- Dynamic balance and control
- Passive dynamic behavior
- Adaptive behavior through embodiment

Your system should address a specific problem in robotics, such as manipulation, locomotion, navigation, or object recognition. The solution should demonstrate clear advantages over traditional non-embodied approaches.

### Problem Options

Choose one of the following problems or propose your own (with instructor approval):

#### Option 1: Adaptive Grasping with Morphological Computation
Design a robotic gripper that can adapt to different object shapes using its physical properties rather than complex control algorithms. Implement and simulate the gripper grasping various objects.

#### Option 2: Bio-inspired Locomotion
Design a simple legged robot that uses principles of passive dynamics or central pattern generators to achieve efficient locomotion.

#### Option 3: Sensorimotor Learning for Navigation
Implement a mobile robot that learns to navigate to goals using only local sensorimotor interactions, without creating a map of the environment.

#### Option 4: Dynamic Balance System
Create a balancing system (e.g., inverted pendulum, simple bipedal model) that maintains balance using embodied intelligence principles.

#### Option 5: Student-Proposed Problem
Propose your own problem that can be solved using embodied intelligence principles. The problem must be approved by the instructor before beginning work.

## Design Document Requirements

Your design document should include the following sections:

### 1. Problem Definition (10% of Design Document)
- Clear statement of the problem being addressed
- Why this problem is relevant to robotics
- What makes this problem suitable for an embodied intelligence solution
- Expected challenges and constraints

### 2. Embodied Intelligence Approach (20% of Design Document)
- Detailed explanation of which embodied intelligence principles you're using
- How physical embodiment contributes to the solution
- Comparison with traditional approaches
- Theoretical foundation for your approach

### 3. System Design (30% of Design Document)
- Detailed system architecture
- Component specifications (sensors, actuators, processing units if applicable)
- Physical design of the robot or system components
- Control algorithms and decision-making processes
- Simulation environment design if applicable

### 4. Implementation Plan (20% of Design Document)
- Step-by-step implementation approach
- Required tools and dependencies
- Testing strategy
- Success metrics and evaluation methods

### 5. Risks and Mitigation (20% of Design Document)
- Identify potential risks to successful implementation
- Propose mitigation strategies for each risk
- Alternative approaches if primary approach fails

## Implementation Requirements

### Technical Implementation
- Use ROS 2 for robot control and communication
- Implement in Gazebo, Isaac Sim, or another approved simulation environment
- Write clean, well-documented code following software engineering best practices
- Include unit tests where appropriate

### Code Structure
Your implementation should include:
```
assignment/
├── src/
│   ├── controllers/      # Control algorithms
│   ├── sensors/         # Sensor processing
│   ├── models/          # Robot models and URDF files
│   └── utils/           # Utility functions
├── worlds/              # Simulation environments
├── launch/              # Launch files
├── config/              # Configuration files
├── test/                # Test files
├── CMakeLists.txt       # Build configuration
├── package.xml          # Package manifest
└── README.md            # Usage instructions
```

### Minimum Functionality
- Demonstrate the core embodied intelligence principle
- Show the system working in at least two different scenarios
- Include basic error handling
- Provide visualization of system state

## Analysis Report Requirements

### 1. Implementation Summary (20% of Analysis Report)
- Summary of what was implemented
- Challenges faced and how they were addressed
- Key design decisions and their rationale

### 2. Performance Analysis (30% of Analysis Report)
- Quantitative analysis of system performance
- Comparison with baseline or alternative approaches
- Metrics used and justification for selection
- Visualization of results (charts, graphs, videos)

### 3. Embodied Intelligence Evaluation (30% of Analysis Report)
- Analysis of how embodiment contributed to the solution
- Quantification of advantages gained through embodiment
- Discussion of trade-offs and limitations
- Potential for real-world application

### 4. Reflection and Future Work (20% of Analysis Report)
- What was learned through the implementation
- How the approach might be extended or improved
- Ideas for future work building on this system

## Grading Rubric

### Design Document (30 points total)
- Problem Definition: 3 points
- Embodied Intelligence Approach: 6 points
- System Design: 9 points
- Implementation Plan: 6 points
- Risks and Mitigation: 6 points

### Implementation (50 points total)
- Code Quality: 10 points
- Functionality: 20 points
- Use of Embodied Intelligence Principles: 15 points
- Documentation: 5 points

### Analysis Report (20 points total)
- Implementation Summary: 4 points
- Performance Analysis: 6 points
- Embodied Intelligence Evaluation: 6 points
- Reflection and Future Work: 4 points

## Technical Requirements

### Software
- ROS 2 Humble Hawksbill or later
- Gazebo Harmonic or Isaac Sim
- Python 3.11+ or C++17
- Git for version control

### Hardware (for simulation)
- Computer with at least 8GB RAM
- Graphics card supporting OpenGL 3.3+ for visualization
- Sufficient CPU power for real-time simulation

## Submission Requirements

### Deadline
The assignment is due 3 weeks from the assignment date. Late submissions will be penalized at 5% per day.

### What to Submit
1. Design Document (PDF format)
2. Complete source code in a Git repository
3. Analysis Report (PDF format)
4. Video demonstration of the system (optional but recommended)
5. README with setup and execution instructions

### Code Submission
- Host code in a publicly accessible Git repository
- Include detailed README with setup instructions
- Tag the final submission as "assignment-submission"
- Ensure the repository includes all required files

## Example Project: Compliant Gripper for Adaptive Grasping

To help clarify expectations, here's an outline of a potential project:

### Problem
Traditional robotic grippers require precise control and detailed object models to grasp objects of varying shapes and stiffnesses.

### Embodied Approach
Design a compliant gripper that uses morphological computation - the physical compliance and shape of the gripper fingers adapt to object shapes without requiring complex control algorithms.

### Implementation
1. Design a 3D model of a compliant gripper with soft finger tips
2. Create a simulation environment with various objects
3. Implement a simple position-based controller (no force control needed)
4. Compare with a traditional rigid gripper using force control

### Analysis
Quantify the advantages in terms of:
- Grasp success rate across different object types
- Control complexity (simpler controller for compliant gripper)
- Energy efficiency
- Adaptability to unknown objects

## Resources and References

### Recommended Reading
- Pfeifer, R., & Bongard, J. (2006). How the Body Shapes the Way We Think
- Dudek, G., & Jenkin, M. (2010). Computational Principles of Mobile Robotics
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics

### Tools and Libraries
- ROS 2 Documentation: https://docs.ros.org/
- Gazebo Tutorials: http://gazebosim.org/tutorials
- PyTorch for neural network implementations: https://pytorch.org/

### Evaluation Metrics
Consider using the following metrics in your evaluation:
- Task success rate
- Time to task completion
- Energy consumption
- Control complexity (e.g., number of parameters in controller)
- Adaptation speed to new conditions
- Robustness to environmental changes

## Academic Integrity

This assignment must be completed individually. All code must be your own work, properly documented and cited. You may use existing libraries and frameworks but must clearly indicate what you implemented versus what you used from existing sources.

Plagiarism will result in a zero for the assignment and may lead to additional academic sanctions.

## Questions and Support

If you have questions about the assignment:
1. Check the course discussion forum
2. Attend office hours
3. Contact the instructor via email
4. Form study groups to discuss concepts (but write code individually)

## Instructor Feedback

The instructor will provide feedback on:
- Design document (within 1 week of submission)
- Implementation progress (mid-assignment check-in)
- Final submission (within 2 weeks of deadline)

This assignment is designed to give you hands-on experience with embodied intelligence principles while demonstrating their practical value in robotics. Choose a problem that interests you and allows you to explore the unique advantages of embodied approaches to robotics.