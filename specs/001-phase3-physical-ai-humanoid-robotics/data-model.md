# Data Model for Phase 3: Physical AI & Humanoid Robotics Course

**Feature**: `001-phase3-physical-ai-humanoid-robotics`
**Created**: 2025-12-05
**Status**: Draft

## Overview

This document defines the data model for the Phase 3 Physical AI & Humanoid Robotics Course. The model represents the key educational entities and their relationships that will structure the course content, navigation, and learning pathways.

## Entity: Module

**Description**: A major learning unit covering specific aspects of Physical AI & Humanoid Robotics

**Properties**:
- `module_id`: string (unique identifier, e.g., "module-1-physical-ai")
- `title`: string (e.g., "Physical AI & Embodied Intelligence")
- `description`: string (brief overview of the module content)
- `duration`: number (estimated weeks, e.g., 2)
- `week_alignment`: string (e.g., "Weeks 1-2")
- `learning_outcomes`: array of strings (what students will learn)
- `prerequisites`: array of strings (required knowledge from other modules)
- `dependencies`: array of strings (technical dependencies)
- `cover_image`: string (path to module cover image)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Contains multiple `Chapter` entities
- Associated with multiple `Simulation Environment` entities
- Uses multiple `Hardware Component` entities
- Connected to multiple `Workflow` entities

## Entity: Chapter

**Description**: A subsection of a module that covers a specific topic in detail

**Properties**:
- `chapter_id`: string (unique identifier, e.g., "chapter-1-1-intro-physical-ai")
- `title`: string (e.g., "Introduction to Physical AI")
- `description`: string (brief overview of the chapter content)
- `learning_outcomes`: array of strings (specific outcomes for this chapter)
- `estimated_time`: string (e.g., "2-3 hours")
- `content_type`: enum ("theory", "lab", "assessment", "case-study")
- `difficulty_level`: enum ("beginner", "intermediate", "advanced")
- `prerequisites`: array of strings (specific knowledge required)
- `references`: array of objects (containing citation details)
- `associated_files`: array of strings (paths to code, config, etc.)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Belongs to one `Module` entity
- Contains multiple `Section` entities (implicit through content structure)
- Associated with specific `Simulation Environment` entities
- Uses specific `Hardware Component` entities
- Connected to specific `Workflow` entities

## Entity: Section

**Description**: A granular content unit within a chapter

**Properties**:
- `section_id`: string (unique identifier, e.g., "section-1-1-1-embodiment-concept")
- `title`: string (e.g., "Embodiment and Intelligence")
- `content`: string (Markdown formatted content)
- `content_type`: enum ("text", "code", "diagram", "video", "exercise")
- `learning_objective`: string (specific objective for this section)
- `duration`: number (estimated minutes)
- `media_assets`: array of strings (paths to images, diagrams, etc.)
- `interactive_elements`: array of objects (quizzes, exercises)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Belongs to one `Chapter` entity
- May reference specific `Hardware Component` entities
- May describe specific `Workflow` entities

## Entity: Simulation Environment

**Description**: Either Gazebo or Unity based environments for robot development and testing

**Properties**:
- `env_id`: string (unique identifier, e.g., "gazebo-sim", "unity-vis")
- `name`: string (e.g., "Gazebo Simulation", "Unity Visualization")
- `type`: enum ("gazebo", "unity", "isaac-sim")
- `description`: string (overview of the environment capabilities)
- `version`: string (e.g., "Gazebo Harmonic", "Unity 2023.2")
- `requirements`: object (hardware and software requirements)
- `tutorials`: array of strings (paths to environment-specific tutorials)
- `plugins`: array of strings (required plugins or extensions)
- `status`: enum ("available", "deprecated")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Used by multiple `Module` entities
- Associated with multiple `Chapter` entities
- Compatible with specific `Hardware Component` entities
- Required for specific `Workflow` entities

## Entity: Hardware Component

**Description**: Physical parts of the robot system (Jetson, RealSense, etc.) or lab infrastructure

**Properties**:
- `component_id`: string (unique identifier, e.g., "jetson-agx-orin", "realsense-d435i")
- `name`: string (e.g., "NVIDIA Jetson AGX Orin")
- `type`: enum ("computing", "sensor", "actuator", "platform", "lab-item")
- `manufacturer`: string (e.g., "NVIDIA", "Intel")
- `model`: string (specific model number)
- `specifications`: object (technical specifications)
- `documentation`: string (path to setup guide)
- `compatibility`: object (list of compatible systems)
- `safety_notes`: string (important safety considerations)
- `price_range`: string (estimated cost range)
- `status`: enum ("recommended", "optional", "deprecated")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Used in multiple `Module` entities
- Required for specific `Chapter` entities
- Compatible with specific `Simulation Environment` entities
- Part of specific `Workflow` entities

## Entity: Workflow

**Description**: Step-by-step process for completing tasks, either in simulation or with real hardware

**Properties**:
- `workflow_id`: string (unique identifier, e.g., "ros2-node-creation", "sensor-calibration")
- `name`: string (e.g., "Creating a ROS 2 Node")
- `description`: string (brief description of the workflow)
- `environment`: enum ("simulation", "real-hardware", "both")
- `steps`: array of objects (containing step-by-step instructions)
- `expected_outcomes`: array of strings (what the workflow should accomplish)
- `prerequisites`: array of strings (what is needed before starting)
- `estimated_time`: string (time to complete the workflow)
- `complexity`: enum ("basic", "intermediate", "advanced")
- `troubleshooting`: string (common issues and solutions)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Associated with specific `Module` entities
- Used in specific `Chapter` entities
- May involve specific `Simulation Environment` entities
- Requires specific `Hardware Component` entities

## Entity: Lab Exercise

**Description**: Structured hands-on activity designed to reinforce theoretical concepts

**Properties**:
- `lab_id`: string (unique identifier, e.g., "lab-2-1-publisher-subscriber")
- `title`: string (e.g., "ROS 2 Publisher-Subscriber Implementation")
- `description`: string (overview of the lab objectives)
- `module_id`: string (reference to parent module)
- `chapter_id`: string (reference to parent chapter)
- `objectives`: array of strings (what the lab teaches)
- `prerequisites`: array of strings (knowledge/skills needed to start)
- `instructions`: array of objects (step-by-step instructions)
- `deliverables`: array of objects (what students need to submit)
- `assessment_criteria`: array of strings (how the lab will be graded)
- `estimated_time`: string (time to complete the lab)
- `difficulty`: enum ("beginner", "intermediate", "advanced")
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Belongs to specific `Module` and `Chapter` entities
- Uses specific `Simulation Environment` entities
- Requires specific `Hardware Component` entities
- Follows specific `Workflow` entities

## Entity: Assessment

**Description**: Evaluation mechanism to measure student learning outcomes

**Properties**:
- `assessment_id`: string (unique identifier, e.g., "quiz-1-physical-ai", "project-1-ros2")
- `title`: string (e.g., "Physical AI Fundamentals Quiz")
- `type`: enum ("quiz", "project", "exam", "peer-review")
- `module_id`: string (reference to parent module)
- `associated_chapters`: array of strings (chapters covered by assessment)
- `questions`: array of objects (for quizzes) or criteria (for projects)
- `rubric`: object (grading criteria and point distribution)
- `due_date`: date (for time-bound assessments)
- `estimated_time`: string (time to complete the assessment)
- `weight`: number (percentage of module grade)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Connected to specific `Module` entities
- Covers specific `Chapter` entities
- Tests specific learning outcomes from modules and chapters

## Entity: Reference

**Description**: Academic or technical source cited in the course content

**Properties**:
- `reference_id`: string (unique identifier, e.g., "ref-2023-icra-physical-ai")
- `type`: enum ("paper", "book", "documentation", "tutorial", "standard")
- `citation`: string (full APA 7th edition citation)
- `title`: string (title of the source)
- `authors`: array of strings (authors of the source)
- `year`: number (publication year)
- `url`: string (optional link to source)
- `abstract`: string (brief summary of the source content)
- `relevance`: string (why this source is relevant to the course)
- `tags`: array of strings (topics covered in the source)
- `accessed_date`: date (when the source was accessed)
- `status`: enum ("valid", "outdated", "unavailable")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Referenced by multiple `Chapter` and `Section` entities
- Used to support learning outcomes in `Module` entities

## Entity: Learning Outcome

**Description**: Specific, measurable skill or knowledge that students should acquire

**Properties**:
- `outcome_id`: string (unique identifier, e.g., "lo-1-1-understand-physical-ai")
- `module_id`: string (reference to parent module)
- `chapter_id`: string (optional reference to specific chapter)
- `description`: string (what the student will be able to do)
- `measurable_criteria`: string (how this outcome will be measured)
- `difficulty_level`: enum ("understand", "apply", "analyze", "create", "evaluate")
- `alignment`: string (alignment with course objectives)
- `assessment_method`: string (how this outcome will be evaluated)
- `status`: enum ("draft", "review", "published")
- `created_date`: date
- `last_updated`: date

**Relationships**:
- Belongs to specific `Module` entities
- Optionally connected to specific `Chapter` entities
- Tested by specific `Assessment` entities
- Referenced in multiple `Section` entities