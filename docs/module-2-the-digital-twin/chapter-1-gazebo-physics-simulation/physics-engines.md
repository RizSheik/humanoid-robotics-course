---
id: module-2-chapter-1-physics-engines
title: 'Module 2 — The Digital Twin | Chapter 1 — Physics Engines'
sidebar_label: 'Chapter 1 — Physics Engines'
---

# Chapter 1 — Physics Engines

## Understanding Physics Simulation for Robotics

Physics engines are critical components in robotics simulation, providing accurate models of physical interactions including collisions, dynamics, and contact forces. For humanoid robotics, these engines enable testing of complex locomotion, manipulation, and interaction behaviors in a safe, repeatable environment.

### What is a Physics Engine?

A physics engine is a software component that simulates the laws of physics in virtual environments. In robotics, physics engines model:

- **Rigid body dynamics**: Motion of solid objects with mass and inertia
- **Collision detection**: Determining when objects intersect
- **Contact response**: Calculating forces when objects touch
- **Constraints**: Joints and other kinematic relationships
- **Friction and damping**: Energy loss during interactions

### Key Physics Engine Concepts

#### Rigid Body Dynamics
In rigid body dynamics, objects maintain constant shape and size. Each body has properties including:

- **Mass**: Resistance to acceleration
- **Inertia tensor**: Resistance to rotational acceleration
- **Position and orientation**: Six degrees of freedom
- **Linear and angular velocity**: Rates of change of position and orientation

#### Collision Detection
Physics engines use two phases for collision detection:

1. **Broad Phase**: Fast, approximate methods to identify potentially colliding pairs
2. **Narrow Phase**: Precise detection for the pairs identified in broad phase

Common algorithms include:
- **Bounding Volume Hierarchies (BVH)**: Using simple shapes to approximate complex geometry
- **Sweep and Prune**: Sorting object extents along axes
- **Spatial Hashing**: Dividing space into discrete bins

#### Contact Response
When objects collide, physics engines calculate contact forces to prevent penetration and simulate realistic interaction. This includes:
- **Normal forces**: Prevent objects from passing through each other
- **Friction forces**: Simulate surface interactions
- **Restitution**: Model energy conservation during collisions

### Popular Physics Engines in Robotics Simulation

#### Bullet Physics
Bullet is widely used in robotics simulation, known for:
- Fast and stable collision detection
- Comprehensive constraint solver
- Support for soft body simulation
- Integration with multiple simulation frameworks

```cpp
// Example Bullet Physics initialization
#include "btBulletDynamicsCommon.h"

// Create collision configuration
btDefaultCollisionConfiguration* collisionConfiguration = 
    new btDefaultCollisionConfiguration();

// Create collision dispatcher
btCollisionDispatcher* dispatcher = 
    new btCollisionDispatcher(collisionConfiguration);

// Create broadphase
btDbvtBroadphase* overlappingPairCache = new btDbvtBroadphase();

// Create solver
btSequentialImpulseConstraintSolver* solver = 
    new btSequentialImpulseConstraintSolver();

// Create dynamics world
btDiscreteDynamicsWorld* dynamicsWorld = 
    new btDiscreteDynamicsWorld(dispatcher, 
                                overlappingPairCache, 
                                solver, 
                                collisionConfiguration);

// Set gravity
dynamicsWorld->setGravity(btVector3(0, -9.81, 0));
```

#### NVIDIA PhysX
NVIDIA PhysX is optimized for GPU acceleration and offers:
- High-performance simulation
- Advanced features for industrial applications
- GPU acceleration capabilities

#### ODE (Open Dynamics Engine)
ODE is a classic physics engine with:
- Stable constraint solving
- Simple API
- Extensive use in robotics research

#### Simbody
Simbody specializes in articulated systems and offers:
- Efficient simulation of complex kinematic trees
- Accurate constraint handling
- Designed specifically for biomechanics and robotics

### Physics Engines in Gazebo

Gazebo supports multiple physics engines, with ODE as the default:

```xml
<!-- Setting physics engine in a Gazebo world -->
<sdf version="1.6">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
    
    <!-- Or use bullet -->
    <physics type="bullet">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Physics Engine Parameters for Humanoid Robotics

When configuring physics engines for humanoid robots, consider:

#### Time Step
- **Small time steps** (0.001s) for accuracy but require more computation
- **Larger time steps** (0.01s) for performance but less accuracy
- For humanoid locomotion, typically use 0.001s for precise control

#### Solver Parameters
- **Constraint solver iterations**: Higher values improve stability
- **Constraint erp (Error Reduction Parameter)**: Controls how quickly errors are corrected
- **Constraint cfm (Constraint Force Mixing)**: Adds compliance to constraints

```xml
<!-- Physics parameters for humanoid simulation -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>200</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Collision Geometry Selection

Different collision geometries have trade-offs:

- **Primitive shapes** (boxes, spheres, cylinders): Fast but less accurate
- **Convex hulls**: Good balance of speed and accuracy for complex shapes
- **Triangle meshes**: Most accurate but computationally expensive

For humanoid robots:
- Use primitive shapes for simple links
- Use convex hull approximation for complex shapes
- Consider compound collision shapes for better accuracy

### Inertial Properties for Humanoid Robots

Accurate inertial properties are crucial for humanoid simulation:

```xml
<!-- Properly defined inertial properties -->
<inertial>
  <mass value="2.38052"/>
  <origin xyz="-0.000295883 -3.74365e-05 0.0293885" rpy="0 0 0"/>
  <inertia ixx="0.0019882" ixy="2.8625e-06" ixz="2.18137e-05"
           iyy="0.00158873" iyz="6.27139e-06"
           izz="0.00328645"/>
</inertial>
```

### Friction Modeling

For humanoid robots, friction is critical for stable walking:

- **Static friction**: Prevents sliding when pushing tangentially
- **Dynamic friction**: Governs sliding motion
- **Coulomb friction**: Models tangential forces at contacts

```xml
<!-- Friction parameters in Gazebo -->
<gazebo reference="foot_link">
  <collision>
    <surface>
      <friction>
        <ode>
          <mu>0.7</mu>      <!-- Static friction -->
          <mu2>0.7</mu2>    <!-- Dynamic friction -->
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

### Stability Considerations for Humanoid Simulation

Humanoid robots require special attention to:
- **Center of mass**: Keep within support polygon for stability
- **Contact stability**: Proper friction and damping parameters
- **Simulation rate**: High enough for control algorithms
- **Mass distribution**: Realistic values for proper dynamics

### Performance vs. Accuracy Trade-offs

When selecting physics engine parameters:

- **Accuracy-focused**: Small time steps, high solver iterations
- **Performance-focused**: Larger time steps, fewer solver iterations
- **Stability-focused**: Conservative parameter selection

For humanoid robotics, prioritize stability and accuracy over performance.

Physics engines are fundamental to robotics simulation, enabling the development and testing of complex behaviors before deployment on physical robots.