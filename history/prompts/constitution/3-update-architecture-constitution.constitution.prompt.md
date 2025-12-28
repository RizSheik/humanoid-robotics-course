---
id: 3
title: Update Architecture Constitution
stage: constitution
date: 2025-12-19
surface: agent
model: Qwen
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, architecture, system-design]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
 - .specify/templates/plan-template.md
 - .specify/templates/spec-template.md
 - .specify/templates/tasks-template.md
tests:
 - none
---

## Prompt

```
/sp.constitution

Principles:
- Single responsibility per file
- Clear separation between frontend and backend
- Backend handles ALL RAG logic (embedding, retrieval, generation)
- Frontend NEVER accesses API keys
- Environment variables used for all secrets
- Code must be runnable locally with ONE command
- No experimental or unused files allowed
- Testing and verification is mandatory before deployment

Quality Rules:
- Minimal folder structure
- Deterministic behavior
- Clear API contracts
- Production-safe defaults
- No prompt history or stored conversations
```

## Response snapshot

I've updated the project constitution to reflect the system architecture principles you specified. The constitution has been updated from version 1.0.0 to 1.1.0, changing from an educational focus to technical architecture principles. The new constitution now includes:

Core Principles:
1. Single Responsibility Per Component
2. Clear Separation Between Frontend and Backend
3. Backend-Centric RAG Logic
4. Frontend Security Isolation
5. Environment-Based Configuration
6. One-Command Local Development
7. Clean Codebase Maintenance
8. Mandatory Verification Before Deployment

Quality Rules:
- Minimal folder structure
- Deterministic behavior
- Clear API contracts
- Production-safe defaults
- No persistent conversation storage

The plan-template.md, spec-template.md, and tasks-template.md have been verified for consistency with the new constitution. The FC-001 validation in plan-template.md already aligned with the new backend-centric RAG logic principle.

## Outcome

- ✅ Impact: Updated constitution to system architecture principles from educational focus
- 🧪 Tests: No automated tests required for constitution update
- 📁 Files: Updated .specify/memory/constitution.md with new architecture principles
- 🔁 Next prompts: None required for this update
- 🧠 Reflection: Successfully migrated constitution from educational to technical focus

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Not applicable