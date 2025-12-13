---
id: "001"
title: "Phase3-Physical-AI-Humanoid-Robotics-Tasks"
stage: "tasks"
date_iso: "2025-12-05"
surface: "agent"
model: "Qwen"
feature: "001-phase3-physical-ai-humanoid-robotics"
branch: "001-phase3-physical-ai-humanoid-robotics"
user: "user"
command: "/sp.tasks"
labels: ["tasks", "education", "robotics", "ai", "humanoid"]
links:
  spec: "specs/001-phase3-physical-ai-humanoid-robotics/spec.md"
  plan: "specs/001-phase3-physical-ai-humanoid-robotics/plan.md"
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "specs/001-phase3-physical-ai-humanoid-robotics/tasks.md"
  - "specs/001-phase3-physical-ai-humanoid-robotics/spec.md"
  - "specs/001-phase3-physical-ai-humanoid-robotics/plan.md"
  - "specs/001-phase3-physical-ai-humanoid-robotics/data-model.md"
  - "specs/001-phase3-physical-ai-humanoid-robotics/research.md"
  - "specs/001-phase3-physical-ai-humanoid-robotics/quickstart.md"
tests_yaml: []
prompt_text: |
  ---
  description: Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Outline

  1. **Setup**: Run `.specify/scripts/powershell/check-prerequisites.ps1 -Json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

  2. **Load design documents**: Read from FEATURE_DIR:
     - **Required**: plan.md (tech stack, libraries, structure), spec.md (user stories with priorities)
     - **Optional**: data-model.md (entities), contracts/ (API endpoints), research.md (decisions), quickstart.md (test scenarios)
     - Note: Not all projects have all documents. Generate tasks based on what's available.

  3. **Execute task generation workflow**:
     - Load plan.md and extract tech stack, libraries, project structure
     - Load spec.md and extract user stories with their priorities (P1, P2, P3, etc.)
     - If data-model.md exists: Extract entities and map to user stories
     - If contracts/ exists: Map endpoints to user stories
     - If research.md exists: Extract decisions for setup tasks
     - Generate tasks organized by user story (see Task Generation Rules below)
     - Generate dependency graph showing user story completion order
     - Create parallel execution examples per user story
     - Validate task completeness (each user story has all needed tasks, independently testable)

  4. **Generate tasks.md**: Use `.specify.specify/templates/tasks-template.md` as structure, fill with:
     - Correct feature name from plan.md
     - Phase 1: Setup tasks (project initialization)
     - Phase 2: Foundational tasks (blocking prerequisites for all user stories)
     - Phase 3+: One phase per user story (in priority order from spec.md)
     - Each phase includes: story goal, independent test criteria, tests (if requested), implementation tasks
     - Final Phase: Polish & cross-cutting concerns
     - All tasks must follow the strict checklist format (see Task Generation Rules below)
     - Clear file paths for each task
     - Dependencies section showing story completion order
     - Parallel execution examples per story
     - Implementation strategy section (MVP first, incremental delivery)

  5. **Report**: Output path to generated tasks.md and summary:
     - Total task count
     - Task count per user story
     - Parallel opportunities identified
     - Independent test criteria for each story
     - Suggested MVP scope (typically just User Story 1)
     - Format validation: Confirm ALL tasks follow the checklist format (checkbox, ID, labels, file paths)

  Context for task generation: 

  The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.

  ## Task Generation Rules

  **CRITICAL**: Tasks MUST be organized by user story to enable independent implementation and testing.

  **Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach.

  ### Checklist Format (REQUIRED)

  Every task MUST strictly follow this format:

  ```text
  - [ ] [TaskID] [P?] [Story?] Description with file path
  ```

  **Format Components**:

  1. **Checkbox**: ALWAYS start with `- [ ]` (markdown checkbox)
  2. **Task ID**: Sequential number (T001, T002, T003...) in execution order
  3. **[P] marker**: Include ONLY if task is parallelizable (different files, no dependencies on incomplete tasks)
  4. **[Story] label**: REQUIRED for user story phase tasks only
     - Format: [US1], [US2], [US3], etc. (maps to user stories from spec.md)
     - Setup phase: NO story label
     - Foundational phase: NO story label  
     - User Story phases: MUST have story label
     - Polish phase: NO story label
  5. **Description**: Clear action with exact file path

  **Examples**:

  - ✅ CORRECT: `- [ ] T001 Create project structure per implementation plan`
  - ✅ CORRECT: `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
  - ✅ CORRECT: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
  - ✅ CORRECT: `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`
  - ❌ WRONG: `- [ ] Create User model` (missing ID and Story label)
  - ❌ WRONG: `T001 [US1] Create model` (missing checkbox)
  - ❌ WRONG: `- [ ] [US1] Create model` (missing Task ID)
  - ❌ WRONG: `- [ ] T001 [US1] Create model` (missing file path)

  ### Task Organization

  1. **From User Stories (spec.md)** - PRIMARY ORGANIZATION:
     - Each user story (P1, P2, P3...) gets its own phase
     - Map all related components to their story:
       - Models needed for that story
       - Services needed for that story
       - Endpoints/UI needed for that story
       - If tests requested: Tests specific to that story
     - Mark story dependencies (most stories should be independent)

  2. **From Contracts**:
     - Map each contract/endpoint → to the user story it serves
     - If tests requested: Each contract → contract test task [P] before implementation in that story's phase

  3. **From Data Model**:
     - Map each entity to the user story(ies) that need it
     - If entity serves multiple stories: Put in earliest story or Setup phase
     - Relationships → service layer tasks in appropriate story phase

  4. **From Setup/Infrastructure**:
     - Shared infrastructure → Setup phase (Phase 1)
     - Foundational/blocking tasks → Foundational phase (Phase 2)
     - Story-specific setup → within that story's phase

  ### Phase Structure

  - **Phase 1**: Setup (project initialization)
  - **Phase 2**: Foundational (blocking prerequisites - MUST complete before user stories)
  - **Phase 3+**: User Stories in priority order (P1, P2, P3...)
    - Within each story: Tests (if requested) → Models → Services → Endpoints → Integration
    - Each phase should be a complete, independently testable increment
  - **Final Phase**: Polish & Cross-Cutting Concerns
response_text: |
  I have successfully generated the tasks.md file for the Phase 3 Physical AI & Humanoid Robotics Course based on the available design artifacts (spec.md, plan.md, data-model.md, research.md, and quickstart.md). 

  The generated tasks.md includes:

  1. Total of 125 tasks organized across 9 phases:
     - Phase 1: Setup (Tasks T001-T008)
     - Phase 2: Foundational (Tasks T009-T019) 
     - Phase 3-9: User Stories 1-7 in priority order (Tasks T020-T104)
     - Final Phase: Polish & Cross-Cutting Concerns (Tasks T105-T125)

  2. Each user story phase includes:
     - Story goal and independent test criteria
     - Learning outcome definitions
     - Chapter content creation (parallelizable tasks marked with [P])
     - Lab exercises specific to the story
     - Assessments for the module
     - Supporting content like diagrams, references, and validation

  3. Task formatting follows the required checklist format:
     - All tasks start with checkbox: - [ ]
     - Sequential task IDs (T001, T002, etc.)
     - Parallelizable tasks marked with [P]
     - User story tasks marked with [US1], [US2], etc.
     - All tasks include specific file paths

  4. Dependencies and parallel execution opportunities identified:
     - Sequential dependencies from US1 to US7 (Physical AI → ROS 2 → Simulation → Isaac → Humanoid → Conversational → Hardware)
     - Parallel execution within chapters of each module
     - Modules 4-6 (Isaac, Humanoid, Conversational) can develop in parallel after foundational modules

  5. Implementation follows MVP-first approach:
     - Suggested MVP: US1 (Physical AI Foundations) as it establishes core concepts
     - Each user story represents independently testable increment
     - Gradual complexity building from fundamentals to capstone project

  The tasks are designed to be immediately executable, with each task specific enough for an LLM to complete without additional context while maintaining alignment with the constitution requirements for technical accuracy, pedagogical clarity, and hands-on practicality.
outcome: "Successfully generated a comprehensive tasks.md file with 125 tasks organized across 9 phases for the Phase 3 Physical AI & Humanoid Robotics Course."
evaluation: "The generated tasks.md file includes all required elements: setup phase, foundational tasks, user story phases in priority order (P1-P3), independent test criteria for each story, dependencies section, parallel execution opportunities, and implementation strategy. All tasks follow the required checklist format with proper ID, story labels, and file paths. The implementation supports the MVP-first approach with User Story 1 as the suggested minimum viable product."
---
