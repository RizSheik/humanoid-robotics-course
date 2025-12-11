---
id: module-1-chapter-2-introduction
title: "Chapter 2 — Topics, Services & Actions"
slug: /module-1-the-robotic-nervous-system/chapter-2-topics-services-actions/introduction
---

# Chapter 2: Topics, Services & Actions

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement and use ROS 2 topics for asynchronous communication
- Create and utilize ROS 2 services for synchronous request/response communication
- Design ROS 2 actions for long-running tasks with feedback
- Apply appropriate communication patterns to humanoid robotics scenarios

## Overview

Communication is the backbone of any robotic system. In ROS 2, there are three primary mechanisms for nodes to exchange information: topics, services, and actions. This chapter explores each communication pattern in the context of humanoid robotics, explaining when and how to use each effectively.

## Table of Contents
1. [Topics Communication](./topics-communication)
2. [Services and Actions](./services-actions)
3. [Practical Exercises](./practical-exercises)

## Introduction to Communication Patterns

ROS 2 provides three distinct communication patterns to address different needs in robotic systems:

1. **Topics**: Enable asynchronous message passing between nodes using a publish/subscribe model. Ideal for sensor data streams, robot state updates, and continuous data flow.

2. **Services**: Provide synchronous request/response communication. Suitable for tasks that require a response before proceeding, such as requesting robot state or configuration changes.

3. **Actions**: Designed for long-running tasks that require feedback, goal preemption, and status reporting. Perfect for robot navigation, manipulation tasks, and complex behaviors.

In humanoid robotics, these communication patterns work together to create responsive and capable systems. For example, sensor data streams might use topics, gait planning might utilize services, and full-body motion execution might leverage actions.

## Next Steps

In the following sections, we'll examine each communication pattern in detail, with specific examples relevant to humanoid robotics applications.