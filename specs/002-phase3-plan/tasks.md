# Tasks: Physical AI & Humanoid Robotics Book

**Input**: Design documents from `/specs/002-physical-ai-book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `docs/`, `static/`, `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in docs/
- [X] T002 [P] Initialize Docusaurus documentation structure with dependencies per plan.md
- [X] T003 Create directory structure for modules and content
- [X] T004 [P] Set up basic docusaurus.config.js with required plugins
- [X] T005 [P] Create basic sidebars.ts structure with placeholder entries

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Create book introduction document with full book overview
- [X] T007 Create category.json files for each module directory
- [X] T008 [P] Set up proper Docusaurus navigation structure for sidebar
- [ ] T009 Define consistent content templates for all chapter types
- [ ] T010 [P] Set up image embedding and referencing system from static/img/
- [ ] T011 Create learning outcomes template for each chapter type

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Student Learns Robotics Concepts (Priority: P1) 🎯 MVP

**Goal**: Implement Module 1: The Robotic Nervous System (ROS 2) with 7 required document types

**Independent Test**: Can be fully tested by navigating through a complete module (Module 1: The Robotic Nervous System) and verifying that all 7 chapter types (overview, weekly-breakdown, deep-dive, practical-lab, simulation, assignment, quiz) are accessible and contain appropriate content.

### Implementation for User Story 1

- [X] T012 [P] [US1] Create Module 1 overview content in docs/module-1-the-robotic-nervous-system/overview.md
- [ ] T013 [P] [US1] Create Module 1 weekly-breakdown content in docs/module-1-the-robotic-nervous-system/weekly-breakdown.md
- [ ] T014 [P] [US1] Create Module 1 deep-dive content in docs/module-1-the-robotic-nervous-system/deep-dive.md
- [X] T015 [US1] Create Module 1 practical-lab content in docs/module-1-the-robotic-nervous-system/practical-lab.md
- [ ] T016 [P] [US1] Create Module 1 simulation content in docs/module-1-the-robotic-nervous-system/simulation.md
- [ ] T017 [P] [US1] Create Module 1 assignment content in docs/module-1-the-robotic-nervous-system/assignment.md
- [X] T018 [US1] Create Module 1 quiz content in docs/module-1-the-robotic-nervous-system/quiz.md
- [X] T019 [US1] Add proper learning outcomes to each Module 1 chapter
- [X] T020 [US1] Embed images from static/img folder in Module 1 content
- [X] T021 [US1] Add proper headings (H1, H2, H3) for RAG-groundable format in Module 1
- [X] T022 [US1] Include step-by-step explanations and real-world examples in Module 1 practical lab
- [ ] T023 [US1] Add cross-references between Module 1 chapters
- [X] T024 [US1] Update sidebars.ts to include Module 1 with all 7 chapters
- [ ] T025 [US1] Validate Module 1 content meets academic standards and accuracy requirements

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Educator Uses Course Materials (Priority: P2)

**Goal**: Implement Module 2: The Digital Twin (Gazebo & Unity) with 7 required document types

**Independent Test**: Can be fully tested by navigating through Module 2 and verifying academic quality, consistency, and educational value.

### Implementation for User Story 2

- [X] T026 [P] [US2] Create Module 2 overview content in docs/module-2-the-digital-twin/overview.md
- [ ] T027 [P] [US2] Create Module 2 weekly-breakdown content in docs/module-2-the-digital-twin/weekly-breakdown.md
- [ ] T028 [P] [US2] Create Module 2 deep-dive content in docs/module-2-the-digital-twin/deep-dive.md
- [ ] T029 [US2] Create Module 2 practical-lab content in docs/module-2-the-digital-twin/practical-lab.md
- [ ] T030 [P] [US2] Create Module 2 simulation content in docs/module-2-the-digital-twin/simulation.md
- [ ] T031 [P] [US2] Create Module 2 assignment content in docs/module-2-the-digital-twin/assignment.md
- [X] T032 [US2] Create Module 2 quiz content in docs/module-2-the-digital-twin/quiz.md
- [ ] T033 [US2] Add proper learning outcomes to each Module 2 chapter
- [X] T034 [US2] Embed images from static/img folder in Module 2 content
- [X] T035 [US2] Add proper headings (H1, H2, H3) for RAG-groundable format in Module 2
- [ ] T036 [US2] Include step-by-step explanations and real-world examples in Module 2 practical lab
- [ ] T037 [US2] Add cross-references between Module 2 chapters
- [ ] T038 [US2] Update sidebars.ts to include Module 2 with all 7 chapters
- [ ] T039 [US2] Validate Module 2 content meets academic standards and accuracy requirements

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - AI System Processes Content for RAG (Priority: P3)

**Goal**: Implement Module 3: The AI-Robot Brain (NVIDIA Isaac™) with 7 required document types, ensuring RAG-groundable format

**Independent Test**: Can be fully tested by verifying that the content follows consistent heading hierarchy and formatting suitable for embedding and retrieval by AI systems.

### Implementation for User Story 3

- [X] T040 [P] [US3] Create Module 3 overview content in docs/module-3-the-ai-robot-brain/overview.md
- [ ] T041 [P] [US3] Create Module 3 weekly-breakdown content in docs/module-3-the-ai-robot-brain/weekly-breakdown.md
- [ ] T042 [P] [US3] Create Module 3 deep-dive content in docs/module-3-the-ai-robot-brain/deep-dive.md
- [ ] T043 [US3] Create Module 3 practical-lab content in docs/module-3-the-ai-robot-brain/practical-lab.md
- [ ] T044 [P] [US3] Create Module 3 simulation content in docs/module-3-the-ai-robot-brain/simulation.md
- [ ] T045 [P] [US3] Create Module 3 assignment content in docs/module-3-the-ai-robot-brain/assignment.md
- [ ] T046 [US3] Create Module 3 quiz content in docs/module-3-the-ai-robot-brain/quiz.md
- [ ] T047 [US3] Add proper learning outcomes to each Module 3 chapter
- [X] T048 [US3] Embed images from static/img folder in Module 3 content
- [X] T049 [US3] Add proper headings (H1, H2, H3) for RAG-groundable format in Module 3
- [ ] T050 [US3] Include step-by-step explanations and real-world examples in Module 3 practical lab
- [ ] T051 [US3] Add cross-references between Module 3 chapters
- [X] T052 [US3] Update sidebars.ts to include Module 3 with all 7 chapters
- [ ] T053 [US3] Validate Module 3 content follows consistent RAG-ready markdown formatting

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Reader Views Supporting Media (Priority: P2)

**Goal**: Implement Module 4: Vision-Language-Action (VLA) Systems with 7 required document types and proper media integration

**Independent Test**: Can be tested by reviewing Module 4 chapters and verifying that images appear correctly positioned with appropriate alt text and captions referencing the static/img folder.

### Implementation for User Story 4

- [X] T054 [P] [US4] Create Module 4 overview content in docs/module-4-vision-language-action-systems/overview.md
- [ ] T055 [P] [US4] Create Module 4 weekly-breakdown content in docs/module-4-vision-language-action-systems/weekly-breakdown.md
- [ ] T056 [P] [US4] Create Module 4 deep-dive content in docs/module-4-vision-language-action-systems/deep-dive.md
- [ ] T057 [US4] Create Module 4 practical-lab content in docs/module-4-vision-language-action-systems/practical-lab.md
- [ ] T058 [P] [US4] Create Module 4 simulation content in docs/module-4-vision-language-action-systems/simulation.md
- [ ] T059 [P] [US4] Create Module 4 assignment content in docs/module-4-vision-language-action-systems/assignment.md
- [ ] T060 [US4] Create Module 4 quiz content in docs/module-4-vision-language-action-systems/quiz.md
- [ ] T061 [US4] Add proper learning outcomes to each Module 4 chapter
- [X] T062 [US4] Embed images from static/img folder in Module 4 content
- [X] T063 [US4] Add proper headings (H1, H2, H3) for RAG-groundable format in Module 4
- [ ] T064 [US4] Include step-by-step explanations and real-world examples in Module 4 practical lab
- [ ] T065 [US4] Add cross-references between Module 4 chapters
- [X] T066 [US4] Update sidebars.ts to include Module 4 with all 7 chapters
- [ ] T067 [US4] Validate Module 4 content with proper image embedding and alt text

---

## Phase 7: Capstone Project Implementation

**Goal**: Implement the capstone project: The Autonomous Humanoid with 7 required document types

- [X] T068 [P] Create capstone overview content in docs/capstone-the-autonomous-humanoid/overview.md
- [ ] T069 [P] Create capstone weekly-breakdown content in docs/capstone-the-autonomous-humanoid/weekly-breakdown.md
- [ ] T070 [P] Create capstone deep-dive content in docs/capstone-the-autonomous-humanoid/deep-dive.md
- [ ] T071 Create capstone practical-lab content in docs/capstone-the-autonomous-humanoid/practical-lab.md
- [ ] T072 [P] Create capstone simulation content in docs/capstone-the-autonomous-humanoid/simulation.md
- [ ] T073 [P] Create capstone assignment content in docs/capstone-the-autonomous-humanoid/assignment.md
- [ ] T074 Create capstone quiz content in docs/capstone-the-autonomous-humanoid/quiz.md
- [ ] T075 Add proper learning outcomes to each capstone chapter
- [ ] T076 Embed images from static/img folder in capstone content
- [ ] T077 Add proper headings (H1, H2, H3) for RAG-groundable format in capstone
- [ ] T078 Include step-by-step explanations integrating concepts from all modules
- [ ] T079 Add cross-references between capstone chapters and other modules
- [ ] T080 Update sidebars.ts to include capstone with all 7 chapters

---

## Phase 8: Appendices Implementation

**Goal**: Implement the appendices with hardware-requirements, lab-architecture, and cloud-vs-onprem content

- [X] T081 [P] Create hardware-requirements appendix in docs/appendices/hardware-requirements.md
- [X] T082 [P] Create lab-architecture appendix in docs/appendices/lab-architecture.md
- [X] T083 [P] Create cloud-vs-onprem appendix in docs/appendices/cloud-vs-onprem.md
- [ ] T084 [P] Update sidebars.ts to include appendices section
- [X] T085 Embed relevant images from static/img folder in appendices
- [X] T086 Add proper headings and tables for hardware requirements
- [X] T087 Include Digital Twin workstation, Edge AI Kit, Robot Lab setup references in appendices
- [ ] T088 Validate all appendices content meets academic standards

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T089 [P] Run Docusaurus build validation: npm run build
- [ ] T090 [P] Run content validation across all modules for consistency
- [ ] T091 [P] Run Markdown linting across all content files
- [ ] T092 Update all links to ensure they work correctly between modules
- [ ] T093 [P] Add proper citations in APA 7th edition format throughout content
- [ ] T094 Verify all 22 specified images are correctly embedded
- [X] T095 [P] Test Docusaurus navigation and sidebar structure
- [ ] T096 Validate RAG-ready content structure across all documents
- [ ] T097 [P] Update docusaurus.config.js with proper site metadata
- [ ] T098 Validate table formatting in practical labs and appendices
- [ ] T099 [P] Run quickstart.md validation checklist
- [ ] T100 Final review of all content for academic quality and accuracy

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Capstone (Phase 7)**: Depends on all modules being complete
- **Appendices (Phase 8)**: Can run parallel to capstone or after
- **Polish (Final Phase)**: Depends on all content being written

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Core implementation: overview → weekly-breakdown → deep-dive → practical-lab → simulation → assignment → quiz
- Add learning outcomes to each chapter
- Embed images and create proper headings
- Add cross-references
- Validate content quality

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All 7 chapters within each module can run in parallel (marked [P] when appropriate)
- Different modules can be worked on in parallel by different content creators

---

## Parallel Example: User Story 1

```bash
# Launch all Module 1 chapters together (they're in different files):
Task: "Create Module 1 overview content in docs/module-1-the-robotic-nervous-system/overview.md"
Task: "Create Module 1 weekly-breakdown content in docs/module-1-the-robotic-nervous-system/weekly-breakdown.md"
Task: "Create Module 1 deep-dive content in docs/module-1-the-robotic-nervous-system/deep-dive.md"
Task: "Create Module 1 simulation content in docs/module-1-the-robotic-nervous-system/simulation.md"
Task: "Create Module 1 assignment content in docs/module-1-the-robotic-nervous-system/assignment.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Add Capstone → Test with all modules → Deploy/Demo
7. Add Appendices → Test with all content → Deploy/Demo
8. Polish everything → Final validation → Full deployment

### Parallel Team Strategy

With multiple developers/content creators:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Once modules complete:
   - Developer E: Capstone project
   - Developer F: Appendices
4. Everyone: Polish & validation phase

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Focus on professional formatting: H1 for file title, H2/H3 for sections, tables for lab setup, bullet points for practical labs