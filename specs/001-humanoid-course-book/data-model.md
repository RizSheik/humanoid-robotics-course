# Data Model: Humanoid Robotics Course Book

## Core Entities

### Course Module
- **Name**: Unique identifier for each module (e.g., "Module 1: Robotic Nervous System")
- **Description**: Brief summary of the module's content and learning objectives
- **Documents**: Collection of 7 document types (overview, weekly breakdown, deep dive, practical lab, simulation, assignment, quiz)
- **Duration**: Estimated time to complete (e.g., 3-4 weeks)
- **Prerequisites**: Knowledge required before starting the module
- **Learning Outcomes**: Specific skills and knowledge students will gain

### Document Type
- **Overview**: High-level summary of the module's content and objectives
- **Weekly Breakdown**: Detailed schedule with topics for each week/section
- **Deep Dive**: In-depth exploration of key concepts with technical details
- **Practical Lab**: Hands-on exercises and implementation tasks
- **Simulation**: Instructions and workflows for simulation environments (Gazebo, Isaac Sim, Webots)
- **Assignment**: Assessment tasks to evaluate student understanding
- **Quiz**: Knowledge-check questions and answers

### Educational Content
- **Title**: Clear descriptive title for the content section
- **Headings Hierarchy**: Deterministic structure for RAG indexing (H1, H2, H3, etc.)
- **Content Body**: Detailed information with technical accuracy
- **Diagrams/Images**: Supporting visual materials with appropriate captions
- **Code Examples**: When needed, real implementation examples (not pseudo code)
- **Citations**: References to authoritative sources in APA format
- **Cross-References**: Links to related content within the textbook

### Simulation Environment
- **ROS2**: Content related to Robot Operating System version 2
- **Gazebo**: Simulation environment for robotics development
- **Isaac Sim**: NVIDIA's robotics simulation platform
- **Webots**: Alternative simulation environment
- **Configuration**: Setup instructions and parameters for each environment
- **Workflows**: Step-by-step procedures for specific tasks

### Assessment Component
- **Question Type**: Multiple choice, short answer, practical implementation, etc.
- **Difficulty Level**: Basic, intermediate, advanced
- **Learning Objective**: Which specific learning outcome is being assessed
- **Answer Key**: Correct responses and explanations
- **Grading Rubric**: Criteria for evaluating student responses

## Relationships

- One **Course Module** contains seven **Document Types**
- Each **Document Type** contains multiple **Educational Content** sections
- **Educational Content** may reference multiple **Simulation Environments**
- **Assessment Components** evaluate understanding of **Educational Content**
- **Course Modules** have prerequisite relationships with each other