# Quickstart Guide for Phase 3: Physical AI & Humanoid Robotics Course

**Feature**: `001-phase3-physical-ai-humanoid-robotics`
**Created**: 2025-12-05
**Status**: Draft

## Overview

This quickstart guide provides instructions for setting up the environment to begin working with the Phase 3 Physical AI & Humanoid Robotics Course. It covers both instructor setup for course delivery and student setup for course participation.

## Instructor Quickstart

### Prerequisites

- Administrative access to the machine where the environment will be set up
- Minimum 32GB RAM (64GB recommended)
- Multi-core processor (8+ cores recommended)
- Dedicated GPU with at least 8GB VRAM (NVIDIA RTX 3080 or equivalent recommended)
- 500GB free disk space for full environment (1TB+ recommended for simulation work)
- Stable internet connection for initial setup and package downloads

### Step 1: Environment Setup

1. **Install Ubuntu 22.04 LTS** (or use a compatible distribution with ROS 2 Humble support)
   - For Windows users: Use WSL2 with Ubuntu 22.04 or dual-boot setup
   - For macOS users: Use Ubuntu VM or dual-boot setup (native support recommended)

2. **Install ROS 2 Humble Hawksbill**
   ```bash
   # Add ROS 2 repository
   sudo apt update && sudo apt install -y software-properties-common
   sudo add-apt-repository universe
   sudo apt update
   
   # Install ROS 2 Humble
   sudo apt install -y locales
   sudo locale-gen en_US.UTF-8
   sudo update-locale LANG=en_US.UTF-8
   export LANG=en_US.UTF-8
   
   sudo apt update && sudo apt install -y curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   
   sudo apt update
   sudo apt install -y ros-humble-desktop
   sudo apt install -y python3-rosdep2
   sudo apt install -y python3-colcon-common-extensions
   ```

3. **Initialize rosdep and source ROS environment**
   ```bash
   # Initialize rosdep
   sudo rosdep init
   rosdep update
   
   # Source ROS environment
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Step 2: Install Simulation Environments

1. **Install Gazebo Garden**
   ```bash
   # Add Gazebo repository
   sudo curl -sSL http://get.gazebosim.org | sh
   
   # Install Gazebo Garden
   sudo apt install gz-garden
   ```

2. **Install Unity Hub and Unity Editor** (for visualization)
   - Download Unity Hub from https://unity.com/download
   - Install Unity Hub and use it to install Unity 2023.2 LTS
   - Install the Robotics package and related tools through Unity Hub

### Step 3: Install NVIDIA Isaac Tools

1. **Install NVIDIA Isaac Sim** 
   - Download Isaac Sim from NVIDIA Developer portal
   - Follow installation instructions for your platform
   - Ensure GPU drivers are compatible (NVIDIA RTX series recommended)

2. **Install Isaac ROS packages**
   - Install via ROS packages for integration with robot workflows
   ```bash
   sudo apt install ros-humble-isaac-ros-*  # Installs all Isaac ROS packages
   ```

### Step 4: Install Course Content

1. **Clone the course repository**
   ```bash
   mkdir -p ~/humanoid-robotics-course/src
   cd ~/humanoid-robotics-course
   git clone <repository-url> src/
   ```

2. **Set up the Docusaurus environment**
   ```bash
   cd ~/humanoid-robotics-course
   npm install
   ```

### Step 5: Configure Hardware (if applicable)

1. **Install Jetson Platform Drivers** (if using Jetson hardware)
   - Follow NVIDIA's Jetson platform installation guide
   - Install appropriate ROS 2 packages for Jetson

2. **Install RealSense Drivers** (if using RealSense cameras)
   ```bash
   sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
   sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
   sudo apt-get update
   sudo apt-get install librealsense2-dkms
   sudo apt-get install librealsense2-dev
   ```

## Student Quickstart

### Prerequisites

- Personal computer with 16GB+ RAM (32GB recommended)
- GPU with 4GB+ VRAM (if running simulations locally)
- 200GB free disk space
- Stable internet connection

### Step 1: Access Course Content

1. **Access the course materials** via the Docusaurus-based course platform
   - Navigate to the course website
   - Log in with provided credentials

2. **Set up your development environment**
   - Use the cloud-based development environment provided through the course, OR
   - Set up the local environment following the instructor guide (Steps 1-2)

### Step 2: Get Familiar with Tools

1. **ROS 2 Basics**
   - Complete the ROS 2 tutorials to understand nodes, topics, services
   - Practice creating simple ROS packages

2. **Simulation Fundamentals**
   - Explore the Gazebo tutorials to understand simulation environments
   - Practice launching and controlling robots in simulation

### Step 3: Begin Module 1 - Physical AI Foundations

1. **Follow the course curriculum in order**
   - Start with Module 1: Physical AI & Embodied Intelligence
   - Complete each chapter sequentially
   - Perform all hands-on labs and exercises

2. **Track your progress**
   - Use the built-in progress tracking system
   - Complete assessments to validate your learning

## Troubleshooting Common Issues

### ROS 2 Issues
- **Problem**: ROS 2 commands not found after installation
  - **Solution**: Ensure you've sourced the setup.bash file in your shell: `source /opt/ros/humble/setup.bash`

- **Problem**: Permission errors with ROS 2
  - **Solution**: Check that your user is in the correct groups: `groups $USER`

### Simulation Issues
- **Problem**: Gazebo fails to launch or crashes
  - **Solution**: Ensure GPU drivers are up to date and graphics acceleration is enabled

- **Problem**: Slow simulation performance
  - **Solution**: Adjust simulation settings in Gazebo for better performance on your hardware

### Course Access Issues
- **Problem**: Cannot access course materials
  - **Solution**: Contact your instructor or course administrator for access credentials

- **Problem**: Tutorials not loading correctly
  - **Solution**: Clear browser cache or try a different browser

## Next Steps

After completing this quickstart guide, you should be able to:

1. Navigate to the course materials through the Docusaurus interface
2. Launch ROS 2 and basic simulation environments
3. Begin working through Module 1 of the course content
4. Complete the first hands-on lab exercises

The course content is structured to build upon itself, so we recommend following the modules in sequence to ensure proper knowledge building. Each module has clear learning objectives and hands-on exercises to reinforce concepts.

For support, please refer to the course's dedicated support channels as provided by your instructor.