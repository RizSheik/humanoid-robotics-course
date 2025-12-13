---
title: Lab Architecture
sidebar_position: 2
description: Detailed specifications for setting up robotics laboratories to support the Physical AI & Humanoid Robotics curriculum.
---

# Lab Architecture

## Overview

This document provides comprehensive specifications for establishing robotics laboratories that support the Physical AI & Humanoid Robotics curriculum. The lab architecture accommodates both educational and research activities, with scalable components to support various class sizes and research requirements.

## Laboratory Requirements

### Space Requirements

| Lab Type | Minimum Area | Recommended Area | Capacity |
|----------|--------------|------------------|----------|
| Educational Lab | 100 sqm | 200 sqm | 12-16 students |
| Research Lab | 150 sqm | 300 sqm | 6-10 researchers |
| Combined Lab | 250 sqm | 400 sqm | 18-24 users |

### Environmental Controls

- **Temperature**: Maintain 18-24°C for optimal equipment performance
- **Humidity**: Keep between 30-70% to prevent component damage
- **Power**: Dedicated circuits with surge protection for sensitive equipment
- **Lighting**: Adjustable LED systems suitable for computer vision work
- **Ventilation**: Adequate air circulation for computing equipment

## Infrastructure Setup

### Network Architecture

![Network Setup](/img/Architecture_diagram_cloud_workstation_A_0.jpg)

#### Local Network
- **Primary Network**: Gigabit Ethernet backbone with managed switches
- **Wireless Coverage**: 802.11ac access points with 5GHz support
- **Network Segmentation**: Separate VLANs for robot communication, student workstations, and guest access
- **Bandwidth**: Minimum 100Mbps per student workstation, 1Gbps for compute servers

#### Security Considerations
- **Firewall**: Enterprise-grade appliance for network protection
- **Access Control**: 802.1X authentication for network access
- **Monitoring**: Network traffic analysis for anomaly detection
- **Isolation**: Separate networks for sensitive research data

### Power Infrastructure

| Component | Power Requirements | Notes |
|-----------|-------------------|-------|
| Workstations | 220V, 500W per station | UPS backup recommended |
| Robots | 220V, variable | Dedicated circuits |
| Network Equipment | 220V, 100W | Continuously available |
| Simulation Servers | 220V, 1000W+ | Redundant power |

## Workstation Configurations

### Digital Twin Workstation

| Component | Specification | Purpose |
|-----------|---------------|---------|
| CPU | Intel i9-13900K or AMD Ryzen 9 7950X | Complex physics simulation |
| GPU | NVIDIA RTX 4090 (24GB VRAM) | High-fidelity simulation and training |
| RAM | 64GB DDR5-4800MHz | Large-scale simulation environments |
| Storage | 2TB+ NVMe SSD | Extensive dataset storage |
| OS | Ubuntu 22.04 LTS | ROS2 compatibility |

### Standard Student Workstation

| Component | Specification | Purpose |
|-----------|---------------|---------|
| CPU | Intel i7-12700K or AMD Ryzen 7 5800X | Simulation and development tasks |
| GPU | NVIDIA RTX 3080 (10GB VRAM) | Gazebo/Unity simulation, AI model inference |
| RAM | 32GB DDR4-3200MHz | Running multiple simulation environments |
| Storage | 1TB NVMe SSD | Fast asset loading and model storage |
| OS | Ubuntu 22.04 LTS | ROS2 compatibility |

## Robot Platforms

### Recommended Platforms for Different Activities

| Activity | Robot Platform | Key Features | Module Application |
|----------|----------------|--------------|-------------------|
| Basic ROS Programming | TurtleBot 4 | Fully ROS2-compatible, educational | Module 1 foundation |
| Locomotion & Dynamics | Unitree Go2 | Dynamic locomotion capabilities | Module 2-3 advanced |
| Human Interaction | NAO v6 | Humanoid form, rich interaction | Module 1-4 full spectrum |
| Manipulation | Stretch RE1 | Long-reach manipulator | Module 3-4 tasks |

### Robot Lab Safety Setup

![Safety Setup](/img/A_humanoid_robot_autonomously_navigating_1.jpg)

#### Physical Safety
- **Safety Barriers**: Physical barriers around active robot workspaces
- **Emergency Stops**: Easily accessible emergency stop buttons throughout lab
- **Safety Interlocks**: Automatic system shutdown when safety zones are breached
- **Protective Equipment**: Safety helmets and safety glasses available

#### Operational Safety
- **Supervision Protocols**: Guidelines for human supervision during robot operations
- **Testing Procedures**: Controlled testing environments for new behaviors
- **Emergency Procedures**: Clearly posted emergency response procedures

## Simulation Environment Setup

### Server Infrastructure

#### Primary Simulation Server
- **CPU**: 32+ cores, high frequency for physics computation
- **GPU**: Multiple RTX 4090 GPUs for large-scale simulation
- **RAM**: 128GB+ for complex environments
- **Storage**: Fast NVMe array for environment assets

#### Backup Simulation Server
- **CPU**: 16+ cores for redundancy
- **GPU**: RTX 4080 for backup operations
- **RAM**: 64GB for continued operation
- **Storage**: Mirrored environment assets

### Software Environment

#### Simulation Platforms
- **Gazebo Harmonic**: Physics-based simulation with realistic rendering
- **Unity Robotics Hub**: High-fidelity visual rendering
- **NVIDIA Isaac Sim**: Advanced perception simulation
- **Webots**: Alternative simulation environment

#### Management Systems
- **Containerization**: Docker for consistent simulation environments
- **Orchestration**: Kubernetes for managing large-scale simulations
- **Version Control**: Git-based system for simulation assets
- **Backup Systems**: Automated backup of simulation environments

## Development Workflow Areas

### Collaborative Workspaces
- **Workstations**: Configured for pair programming and team collaboration
- **Display Systems**: Large monitors for code review and system debugging
- **Video Conferencing**: Equipment for remote collaboration and guest lectures

### Individual Work Areas
- **Quiet Zones**: Isolated areas for focused individual work
- **Personal Storage**: Secure storage for student projects and equipment
- **Flexible Setup**: Configurable workstations to accommodate different needs

## Equipment Maintenance

### Preventive Maintenance
- **Robot Platforms**: Monthly inspection and calibration
- **Computing Equipment**: Quarterly maintenance and updates
- **Network Equipment**: Ongoing monitoring and maintenance
- **Safety Equipment**: Regular inspection and testing

### Troubleshooting Resources
- **Maintenance Schedule**: Regular maintenance calendar
- **Spare Parts Inventory**: Critical components and consumables
- **Service Contacts**: Vendor support contacts and SLAs
- **Documentation**: Standard procedures for common issues

## Security and Access Control

### Physical Access
- **Access Cards**: RFID-based access control system
- **Time Restrictions**: Scheduled access based on lab hours
- **Audit Trail**: Logging of all access events
- **Visitor Management**: Procedures for guest access

### Digital Security
- **Authentication**: Multi-factor authentication for all systems
- **Encryption**: Encrypted storage for sensitive research data
- **Backup Security**: Secure backup storage and transmission
- **Network Security**: Firewalls and intrusion detection systems

## Budget Considerations

### Initial Setup Costs
- **Infrastructure**: $100,000 - $300,000 depending on lab size
- **Computing Equipment**: $50,000 - $150,000
- **Robot Platforms**: $30,000 - $100,000
- **Software Licenses**: $20,000 - $50,000

### Ongoing Operational Costs
- **Maintenance**: $10,000 - $25,000 annually
- **Consumables**: $5,000 - $10,000 annually
- **Software Updates**: $8,000 - $15,000 annually
- **Power and Utilities**: $12,000 - $20,000 annually

## Scalability Considerations

The lab architecture is designed for scalability:

- **Modular Design**: Components can be added as needs grow
- **Cloud Integration**: Hybrid cloud-local compute resources
- **Remote Access**: Secure remote access for distributed teams
- **Virtual Labs**: Virtual lab environments for remote students

For specific implementation guidance with the modules covered in this textbook, refer to the practical lab sections of each module.