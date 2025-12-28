# Quickstart Guide: Physical AI & Humanoid Robotics Course

This guide provides a quick introduction to getting started with the Physical AI & Humanoid Robotics Course project, built with Docusaurus.

## 1. Prerequisites

Before starting with the course content, ensure you have:

- **Node.js**: Version 20.0 or higher
- **npm**: Latest version (comes with Node.js)
- **Git**: Version control system
- **ROS 2**: Humble Hawksbill (for hands-on exercises)
- **Gazebo**: Harmonic (for simulation exercises)
- **NVIDIA Isaac Sim**: Latest version (for AI robotics exercises)
- **Unity**: 2023.2+ (for advanced simulation workflows)

## 2. Project Setup

To set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/RizSheik/humanoid-robotics-course.git
   cd humanoid-robotics-course
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the local development server:
   ```bash
   npm start
   ```

This will open the course website in your browser at `http://localhost:3000`, where you can browse the different modules and content.

## 3. Course Structure

The course is organized into 6 main modules:

1. **Module 1**: Physical AI Foundations - Embodied intelligence and control theory
2. **Module 2**: ROS 2 Fundamentals - Topics, services, actions, and distributed systems
3. **Module 3**: Digital Twin Simulation - Gazebo, Unity, and Isaac Sim environments
4. **Module 4**: AI Robot Brain - NVIDIA Isaac Platform for perception and control
5. **Module 5**: Humanoid Robotics - Specialized development for anthropomorphic robots
6. **Module 6**: Conversational Robotics - Natural human-robot interaction

Each module contains:
- Overview and learning outcomes
- Weekly breakdown of topics
- Deep dive into key concepts
- Practical lab exercises
- Simulation assignments
- Assignments and quizzes

## 4. Working with Course Content

### Browsing Modules
- Navigate through the course using the sidebar
- Each module has its own learning path with progressive difficulty
- Complete the content in sequence for best understanding

### Practical Exercises
- Most chapters include practical exercises
- These exercises often require simulation or hardware environments
- Follow the instructions in the "Practical Lab" sections for detailed setup

### Simulation Environments
- Simulation activities use Gazebo Harmonic, Isaac Sim, or Unity
- Check the appendix for specific setup instructions for each
- Many exercises include example code in Python/ROS 2

## 5. Development Guidelines

### For Contributors
- All content follows formal textbook structure
- Chapters include learning objectives, content with sections/subsections, and exercises
- Code examples use Python 3.11+ and ROS 2 Humble
- Diagrams use Mermaid, draw.io, or include local assets

### For Students
- Complete modules in sequence for optimal learning
- Engage with practical exercises to reinforce concepts
- Use the simulation environments to test concepts
- Complete all assignments and quizzes to verify understanding

## 6. Building the Static Site

To build the production-ready version:

```bash
npm run build
```

The built site will be in the `build/` directory and can be served using:

```bash
npm run serve
```

## 7. Troubleshooting

- If you encounter broken links during development, this is expected as modules are progressively added
- Ensure your ROS 2 environment is sourced before running ROS-based exercises
- Check the appendix for hardware requirements and lab infrastructure setup
- For simulation exercises, ensure your GPU meets the requirements for Isaac Sim or Unity