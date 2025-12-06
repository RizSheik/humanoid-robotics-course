<!--
Sync Impact Report:
Version change: 0.0.0 → 0.1.0
Modified principles: N/A (new constitution)
Added sections: Core Principles, Standards, Constraints, Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
  - .specify/templates/commands/sp.adr.md: ⚠ pending
  - .specify/templates/commands/sp.analyze.md: ⚠ pending
  - .specify/templates/commands/sp.checklist.md: ⚠ pending
  - .specify/templates/commands/sp.clarify.md: ⚠ pending
  - .specify/templates/commands/sp.constitution.md: ⚠ pending
  - .specify/templates/commands/sp.git.commit_pr.md: ⚠ pending
  - .specify/templates/commands/sp.implement.md: ⚠ pending
  - .specify/templates/commands/sp.phr.md: ⚠ pending
  - .specify/templates/commands/sp.plan.md: ⚠ pending
  - .specify/templates/commands/sp.specify.md: ⚠ pending
  - .specify/templates/commands/sp.tasks.md: ⚠ pending
Follow-up TODOs: N/A
-->
# Textbook for Physical AI & Humanoid Robotics Constitution

## Core Principles

### Technical Accuracy
Robotics, AI, perception, control, and hardware details MUST be verified using
authoritative sources (peer-reviewed papers, robotics textbooks, SDK
documentation, manufacturer manuals). Mathematical models, kinematics, and
control theory MUST be correct and internally consistent.

### Pedagogical Clarity
Content MUST be written for undergraduate/early-graduate learners. Complex topics
(RL for control, SLAM, dynamics, VLA) MUST be simplified without losing rigor.
Provide intuitive explanations, diagrams, and step-by-step reasoning.

### Hands-On Practicality
Every chapter MUST tie theory to real robotic tasks. Content MUST include labs,
example code (Python/ROS 2), simulation workflows (Gazebo, Isaac), and hardware
notes (Jetson, RealSense, Unitree).

### Interdisciplinary Integration
Content MUST combine AI agents, Physical AI, perception, control, LLM-based
planning, and human-robot interaction into a unified learning pathway.

## Standards

### Sources & Citations
All claims MUST be traceable to primary/credible sources. APA 7th edition
MUST be used for citations. At least 50% of sources MUST be peer-reviewed
robotics/AI sources (ICRA, IROS, RSS, CoRL, NeurIPS). Non-peer-reviewed sources
(SDK docs, APIs, hardware manuals, university lecture notes) are allowed.

### Content Quality
Content MUST have a clear structure, diagrams, conceptual frameworks, and
pseudocode. Reproducibility (code, configs, lab requirements) MUST be ensured.

### Safety & Ethics
Robotics safety (ISO 10218, ISO/TS 15066) and ethical AI guidelines MUST be
included. No unsafe or harmful robot instructions are permitted.

## Constraints

### Length
The textbook length MUST be between 40k–80k words, chapter-based.

### Format
The format MUST be Docusaurus + Spec-Kit-Plus (Markdown-first, GitHub Pages).

### Code
Code examples MUST be in Python, ROS 2, C++, Isaac/Gazebo.

### Diagrams
Diagrams MUST use Mermaid, draw.io, or local assets.

### Zero Plagiarism
All content MUST be original and free of plagiarism.

## Governance
This constitution supersedes all other practices. Amendments require
documentation, approval, and a migration plan.

### Success Criteria
- **Technical Quality**: Accurate, current robotics & AI content with
  correct equations & algorithms.
- **Educational Value**: Students MUST be able to learn Physical AI & humanoid
  robotics from scratch; includes exercises & labs.
- **Reproducibility**: Code and workflows MUST run on common platforms (ROS 2,
  Jetson, Isaac, Gazebo).
- **Coherence**: Content MUST have a logical flow from fundamentals → control
  → perception → AI → humanoids → capstone.
- **Deployment**: The book MUST build cleanly in Docusaurus and be ready for
  GitHub Pages.

**Version**: 0.1.0 | **Ratified**: 2025-12-05 | **Last Amended**: 2025-12-05