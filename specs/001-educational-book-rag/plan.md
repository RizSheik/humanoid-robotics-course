# Implementation Plan: Educational Book with Integrated RAG Chatbot

**Branch**: `001-educational-book-rag` | **Date**: 12/12/2025 | **Spec**: [link to spec](./spec.md)
**Input**: Feature specification from `/specs/001-educational-book-rag/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

This plan outlines the implementation of an educational book project using Docusaurus with 4 modules (each with 4 chapters) and integrated RAG (Retrieval Augmented Generation) chatbot for enhanced learning experience. The system will serve structured educational content with navigation, images, and AI-powered assistance for students learning humanoid robotics.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend, Markdown for content
**Primary Dependencies**: Docusaurus 3.0+, FastAPI 0.104+, OpenAI SDK, Neon Postgres, pgvector, React 18+
**Storage**: GitHub Pages (static content), Neon Serverless Postgres (vector embeddings and chat history)
**Testing**: pytest for backend, Jest for frontend, Playwright for E2E testing
**Target Platform**: Web-based application accessible via browsers, deployed on GitHub Pages with backend API
**Project Type**: Web application (frontend Docusaurus + backend API services)
**Performance Goals**: Page load <3s, Chatbot response <5s, 99% uptime during peak usage
**Constraints**: <500ms p95 for content delivery, <5s for RAG responses, responsive design for mobile/desktop
**Scale/Scope**: Support 1000+ concurrent users, 16 chapters + modules, vector embeddings for all educational content

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The implementation must align with the project constitution:

**Technical Accuracy Gate**:
- Educational content must be verified using authoritative sources (peer-reviewed papers, robotics textbooks, SDK documentation)
- Mathematical models, kinematics, and control theory in the content must be correct and internally consistent
- RAG chatbot responses must be grounded in verified educational content

**Pedagogical Clarity Gate**:
- Content must be written for undergraduate/early-graduate learners
- Complex topics must be simplified without losing rigor
- Content must include intuitive explanations, diagrams, and step-by-step reasoning

**Hands-On Practicality Gate**:
- Every chapter must tie theory to real robotic tasks
- Content must include labs, example code (Python/ROS 2), simulation workflows (Gazebo, Isaac)
- System must provide practical examples that complement the educational content

**Interdisciplinary Integration Gate**:
- Content must combine AI agents, Physical AI, perception, control, LLM-based planning, and human-robot interaction
- RAG chatbot must be able to answer questions across these interdisciplinary topics

**Standards Compliance**:
- Content must have clear structure, diagrams, conceptual frameworks, and pseudocode
- All claims must be traceable to primary/credible sources (APA 7th edition)
- At least 50% of sources must be peer-reviewed robotics/AI sources (ICRA, IROS, RSS, CoRL, NeurIPS)

**Safety & Ethics**:
- Robotics safety (ISO 10218, ISO/TS 15066) and ethical AI guidelines must be included
- No unsafe or harmful robot instructions are permitted in educational content

**Format Constraints**:
- Format must be Docusaurus + Spec-Kit-Plus (Markdown-first, GitHub Pages)
- Code examples must be in Python, ROS 2, C++, Isaac/Gazebo
- Diagrams must use Mermaid, draw.io, or local assets

**Zero Plagiarism**:
- All content must be original and free of plagiarism
- RAG system must ensure originality in generated responses

**Success Criteria Validation**:
- Educational Value: Students must be able to learn Physical AI & humanoid robotics from scratch
- Reproducibility: Code and workflows must run on common platforms (ROS 2, Jetson, Isaac, Gazebo)
- Coherence: Content must have logical flow from fundamentals → control → perception → AI → humanoids → capstone
- Deployment: The book must build cleanly in Docusaurus and be ready for GitHub Pages

## Project Structure

### Documentation (this feature)

```text
specs/001-educational-book-rag/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application (frontend Docusaurus + backend API services)

backend/
├── src/
│   ├── main.py              # FastAPI application entrypoint
│   ├── models/              # Pydantic models for API contracts
│   │   ├── chat.py          # Chat-related models
│   │   ├── content.py       # Content models
│   │   └── user.py          # User/session models
│   ├── services/            # Business logic
│   │   ├── chat_service.py  # Chatbot logic and RAG processing
│   │   ├── content_service.py # Content management
│   │   └── embedding_service.py # Vector embedding generation
│   ├── routes/              # API route definitions
│   │   ├── chat.py          # Chat endpoints
│   │   └── content.py       # Content API endpoints
│   ├── database/            # Database connection and schema
│   │   ├── connection.py    # Database connection utilities
│   │   └── models.py        # ORM models
│   ├── config.py            # Configuration settings
│   └── utils/               # Utility functions
│       ├── validators.py    # Input validators
│       └── helpers.py       # Helper functions
└── tests/                   # Backend tests
    ├── unit/                # Unit tests
    └── integration/         # Integration tests

frontend/                    # Docusaurus-based frontend
├── blog/                   # Blog posts (if any)
├── docs/                   # Educational content modules and chapters
│   ├── module-1-the-robotic-nervous-system/
│   │   ├── overview.md
│   │   ├── weekly-breakdown.md
│   │   ├── deep-dive.md
│   │   ├── practical-lab.md
│   │   ├── simulation.md
│   │   ├── quiz.md
│   │   └── assignment.md
│   ├── module-2-the-digital-twin/
│   ├── module-3-the-ai-robot-brain/
│   ├── module-4-vision-language-action-systems/
│   └── category.json        # Sidebar navigation structure
├── src/
│   ├── components/          # React components
│   │   ├── Chatbot/         # RAG chatbot UI component
│   │   │   ├── Chatbot.tsx
│   │   │   ├── ChatWindow.tsx
│   │   │   └── Message.tsx
│   │   ├── Navigation/      # Navigation components
│   │   └── UI/              # Common UI components
│   ├── pages/               # Custom pages if needed
│   ├── css/                 # Custom CSS
│   └── theme/               # Docusaurus theme customization
├── static/                  # Static assets
│   └── img/                 # Images for educational content
├── docusaurus.config.js     # Docusaurus configuration
├── sidebars.ts              # Sidebar navigation configuration
└── package.json             # Frontend dependencies

# Existing project structure integration
├── docs/                    # Current educational modules (existing)
│   ├── module-1-the-robotic-nervous-system/
│   ├── module-2-the-digital-twin/
│   ├── module-3-the-ai-robot-brain/
│   └── module-4-vision-language-action-systems/
├── src/
│   ├── components/          # Docusaurus React components
│   └── pages/               # Custom pages
├── static/
│   └── img/                 # Static images
├── docusaurus.config.js     # Main Docusaurus config
└── sidebars.ts              # Sidebar navigation
```

**Structure Decision**: This structure extends the existing Docusaurus project by adding a backend API for the RAG chatbot functionality while maintaining the static educational content delivery through Docusaurus. The chatbot component will be integrated into the Docusaurus pages via React components that communicate with the backend API.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |