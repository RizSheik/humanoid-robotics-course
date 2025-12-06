---
id: 3
title: Create technical plan for textbook
stage: plan
date: 2025-12-05
surface: agent
model: claude-haiku-4-5-20251001
feature: 001-book-spec
branch: 001-book-spec
user: User
command: /sp.plan
labels: ["plan", "textbook-plan"]
links:
  spec: specs/001-book-spec/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-book-spec/plan.md
tests:
null
---

## Prompt

Physical AI & Humanoid Robotics Textbook (Technical Plan)

Task: Create the complete technical plan for building the Physical AI & Humanoid Robotics textbook using Docusaurus + GitHub Pages, guided by our Constitution and Spec.

1. Architecture Sketch

Produce an architecture overview covering:

Book Architecture

4-Module Structure → Chapters → Sections

Separation between high-level theory, hands-on labs, and VLA capstone

Multi-phase writing approach (Outline → Detailed Chapters → Code → Assets)

Technical Tooling Architecture

Docusaurus folder structure

Integration with GitHub Pages

Assets pipeline (diagrams, ROS graphs, Isaac screenshots)

Spec-Kit-Plus workflow for iterative generation

Robotics Stack Architecture (for reference in book)

ROS 2 → Gazebo/Unity → Isaac → VLA pipeline

Digital Twin & Sim-to-Real flow

Hardware (Jetson, RealSense, Unitree, etc.)

2. Section Structure

Define the structure of each module and chapter, at planning level:

Expected chapter sections

Where diagrams, code, or labs will be inserted

Placement of:

Learning outcomes

Hands-on labs

Example pipelines (ROS graphs, SLAM diagrams, Nav2 flows)

Capstone progression checkpoints

Ensure structure aligns with:

13-week schedule

4-module course design

High-level outline from /sp.spec

3. Research Approach

Follow a research-concurrent workflow:

Research while writing chapters, not in a single upfront phase

Prioritize:

ROS 2 Humble/Iron documentation

Gazebo/Ignition documentation

Unity XR Robotics resources

NVIDIA Isaac Sim & Isaac ROS docs

OpenAI Whisper + LLM planning research

Humanoid robotics lectures and datasets

Collect references aligned with Constitution:

APA format

Accuracy & traceability

Prefer primary technical docs and peer-reviewed robotics papers

4. Quality Validation

Define validation steps:

Ensure module coverage matches course business requirements

Match learning outcomes and weekly breakdown

Cross-check technical accuracy with primary documentation

Ensure clarity for AI students with limited robotics background

Verify diagrams/code consistency across modules

Maintain Constitution standards:

Rigor

Clarity

Reproducibility

Academic formatting (APA)

5. Decisions Needing Documentation

List key decisions that require documenting with options + tradeoffs:

A. Simulation Stack

Gazebo vs Unity vs Isaac Sim roles

How much overlap or separation to maintain

B. ROS Versioning

ROS 2 Humble vs Iron

Python (rclpy) vs C++ (ignored or included?)

C. Hardware Strategy

On-prem lab vs cloud-native lab

Jetson model choice (Nano vs NX)

RealSense D435i vs others

D. Book Audience Depth

Beginner-friendly vs high rigor

Amount of math (kinematics/dynamics)

E. VLA Architecture

LLM planning vs Behavior Trees

Whisper vs other speech systems

Each decision → options, tradeoffs, final recommendation.

6. Testing Strategy

Define validation checks based on success criteria:

Structural Validation

All modules and chapters exist

Chapter structure matches outline requirements

Technical Accuracy Validation

Verify ROS 2, Gazebo, Isaac, Unity claims with primary docs

Confirm hardware specs and configurations

Validate VLA pipeline logic

Pedagogical Validation

Check learning flow from beginner → advanced

Confirm labs build toward the capstone

Docusaurus Build Validation

Build without errors

Check page structure and navigation

Ensure assets load and links resolve

Final Acceptance Criteria

Fully meets course requirements

Ready for deep chapter specs (Iteration 2)

Ready for code and diagram generation

## Response snapshot

The technical plan for 'Physical AI & Humanoid Robotics Textbook' has been successfully created at `specs/001-book-spec/plan.md`.

This plan covers:
*   Architecture Sketch (Book, Technical Tooling, Robotics Stack)
*   Section Structure
*   Research Approach
*   Quality Validation
*   Key Decisions Needing Documentation (Simulation Stack, ROS Versioning, Hardware Strategy, Book Audience Depth, VLA Architecture) with options, tradeoffs, and recommendations
*   Testing Strategy (Structural, Technical Accuracy, Pedagogical, Docusaurus Build, Final Acceptance Criteria)

The plan is now complete and ready for review.

## Outcome

- ✅ Impact: Technical plan created and saved.
- 🧪 Tests: N/A
- 📁 Files: specs/001-book-spec/plan.md created.
- 🔁 Next prompts: User can review the plan and proceed with next steps.
- 🧠 Reflection: Successfully generated the technical plan for the textbook, addressing all key areas including architecture, structure, research, quality, decisions, and testing. The plan incorporates recommendations and tradeoffs for critical decisions. The PHR creation script was unavailable, so a manual fallback was used.

## Evaluation notes (flywheel)

- Failure modes observed: create-phr.sh script not found, fallback to agent-native flow.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Ensure `create-phr.sh` script is available or improve fallback mechanism for PHR creation.