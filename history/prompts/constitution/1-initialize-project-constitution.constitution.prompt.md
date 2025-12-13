---
id: 1
title: Initialize project constitution
stage: constitution
date: 2025-12-05
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: 001-physical-ai-textbook
user: User
command: /sp.constitution
labels: ["constitution", "initialization"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
null
---

## Prompt

Project: Textbook for Physical AI & Humanoid Robotics
Platform: Docusaurus + Spec-Kit-Plus + Claude Code
Goal: Create a rigorous, clear, industry-relevant university-level textbook.

Core Principles
Technical Accuracy

Robotics, AI, perception, control, and hardware details must be verified using authoritative sources (peer-reviewed papers, robotics textbooks, SDK documentation, manufacturer manuals).

Mathematical models, kinematics, and control theory must be correct and internally consistent.

Pedagogical Clarity

Written for undergraduate/early-graduate learners.

Complex topics (RL for control, SLAM, dynamics, VLA) must be simplified without losing rigor.

Provide intuitive explanations, diagrams, and step-by-step reasoning.

Hands-On Practicality

Every chapter ties theory to real robotic tasks.

Include labs, example code (Python/ROS 2), simulation workflows (Gazebo, Isaac), and hardware notes (Jetson, RealSense, Unitree).

Interdisciplinary Integration

Combine AI agents, Physical AI, perception, control, LLM-based planning, and human-robot interaction into a unified learning pathway.

Standards
Sources & Citations

All claims traceable to primary/credible sources.

APA 7th edition.

≥50% peer-reviewed robotics/AI sources (ICRA, IROS, RSS, CoRL, NeurIPS).

Non-peer-reviewed allowed: SDK docs, APIs, hardware manuals, university lecture notes.

Content Quality

Clear structure, diagrams, conceptual frameworks, and pseudocode.

Ensure reproducibility (code, configs, lab requirements).

Safety & Ethics

Include robotics safety (ISO 10218, ISO/TS 15066) and ethical AI guidelines.

No unsafe or harmful robot instructions.

Constraints

Length: 40k–80k words, chapter-based.

Format: Docusaurus + Spec-Kit-Plus (Markdown-first, GitHub Pages).

Code: Python, ROS 2, C++, Isaac/Gazebo examples.

Diagrams: Mermaid, draw.io, or local assets.

Zero plagiarism.

Success Criteria

Technical Quality: Accurate, current robotics & AI content with correct equations & algorithms.

Educational Value: Students can learn Physical AI & humanoid robotics from scratch; includes exercises & labs.

Reproducibility: Code and workflows run on common platforms (ROS 2, Jetson, Isaac, Gazebo).

Coherence: Logical flow from fundamentals → control → perception → AI → humanoids → capstone.

Deployment: Book builds cleanly in Docusaurus and is ready for GitHub Pages.

## Response snapshot

The project constitution has been initialized to version 0.1.0. This is a minor version bump due to the creation of the initial content and governance.

The following templates and command files require manual verification and potential updates to align with the new constitution:
*   .specify/templates/plan-template.md
*   .specify/templates/spec-template.md
*   .specify/templates/tasks-template.md
*   All command files under .specify/commands/

Suggested commit message: `docs: initialize constitution to v0.1.0 (initial content + governance)`

## Outcome

- ✅ Impact: Constitution file created and populated.
- 🧪 Tests: N/A
- 📁 Files: .specify/memory/constitution.md updated.
- 🔁 Next prompts: Manual verification of other template files.
- 🧠 Reflection: Successfully initialized the project constitution based on user input, ensuring all principles, standards, constraints, and governance rules are captured. Identified dependent files requiring manual review for consistency.

## Evaluation notes (flywheel)

- Failure modes observed: create-phr.sh script not found, fallback to agent-native flow.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Ensure `create-phr.sh` script is available or improve fallback mechanism.