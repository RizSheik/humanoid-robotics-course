---

description: "Task list for Educational Book with Integrated RAG Chatbot implementation"
---

# Tasks: Educational Book with Integrated RAG Chatbot

**Input**: Design documents from `/specs/001-educational-book-rag/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The feature specification does not explicitly request tests, so test tasks are not included in this task list.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `frontend/`, `static/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create backend project structure with FastAPI dependencies in backend/
- [X] T002 [P] Initialize Docusaurus project with required dependencies in package.json
- [X] T003 [P] Configure linting and formatting tools for Python and JavaScript

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Setup database schema and models for Module, Chapter, LearningContent, ChatQuery, ContentEmbedding, UserSession, and Image entities in backend/src/database/models.py
- [X] T005 [P] Configure Neon Postgres connection with pgvector in backend/src/database/connection.py
- [X] T006 [P] Setup API routing and middleware structure in backend/src/main.py
- [X] T007 Create base Pydantic models in backend/src/models/
- [X] T008 Configure environment configuration management in backend/src/config.py
- [X] T009 Setup OpenAI integration with API key management in backend/src/services/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Access Educational Content (Priority: P1) 🎯 MVP

**Goal**: Enable students to access structured educational content organized into modules and chapters with proper navigation

**Independent Test**: Can navigate through the book's modules and chapters, verifying that content loads correctly and is properly structured

### Implementation for User Story 1

- [X] T010 [P] [US1] Create Module model in backend/src/models/content.py
- [X] T011 [P] [US1] Create Chapter model in backend/src/models/content.py
- [X] T012 [US1] Implement ContentService in backend/src/services/content_service.py
- [X] T013 [US1] Implement GET /api/modules endpoint in backend/src/routes/content.py
- [X] T014 [US1] Implement GET /api/modules/{moduleId} endpoint in backend/src/routes/content.py
- [X] T015 [US1] Implement GET /api/chapters/{chapterId} endpoint in backend/src/routes/content.py
- [ ] T016 [US1] Create frontend content display components in src/components/Content/
- [ ] T017 [US1] Add basic navigation elements to Docusaurus theme in src/theme/
- [ ] T018 [US1] Set up basic module and chapter content in docs/ following 4 modules × 4 chapters structure

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Interact with RAG Chatbot (Priority: P1)

**Goal**: Enable students to interact with an AI-powered chatbot that can answer questions about the educational content

**Independent Test**: Can enter various queries related to educational content and verify that the chatbot provides accurate, relevant responses based on the course material

### Implementation for User Story 2

- [X] T019 [P] [US2] Create ChatQuery model in backend/src/models/chat.py
- [X] T020 [P] [US2] Create UserSession model in backend/src/models/user.py
- [X] T021 [US2] Create ContentEmbedding model in backend/src/models/embedding.py
- [X] T022 [US2] Implement ChatService in backend/src/services/chat_service.py
- [X] T023 [US2] Implement EmbeddingService in backend/src/services/embedding_service.py
- [X] T024 [US2] Implement POST /api/chat/query endpoint in backend/src/routes/chat.py
- [X] T025 [US2] Implement GET /api/chat/history endpoint in backend/src/routes/chat.py
- [X] T026 [US2] Create RAG chatbot UI component in src/components/Chatbot/Chatbot.tsx
- [ ] T027 [US2] Implement OpenAI API integration for chat responses in backend/src/services/chat_service.py
- [ ] T028 [US2] Implement content retrieval mechanism for RAG in backend/src/services/embedding_service.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - View Supporting Images and Media (Priority: P2)

**Goal**: Enable students to view supporting images and diagrams within each chapter to better understand complex concepts

**Independent Test**: Can view different chapters and verify that images appear correctly positioned with appropriate alt text and captions

### Implementation for User Story 3

- [ ] T029 [P] [US3] Create Image model in backend/src/models/content.py
- [ ] T030 [US3] Update Chapter model to support image associations in backend/src/models/content.py
- [ ] T031 [US3] Implement ImageService in backend/src/services/content_service.py
- [ ] T032 [US3] Add image handling to GET /api/chapters/{chapterId} endpoint in backend/src/routes/content.py
- [ ] T033 [US3] Create image management utilities in backend/src/utils/
- [ ] T034 [US3] Set up static image storage structure in static/img/
- [ ] T035 [US3] Add image display components to Content components in src/components/Content/
- [ ] T036 [US3] Implement proper image alt text and caption handling in frontend components
- [ ] T037 [US3] Add accessibility features for images (ARIA labels, etc.)

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently

---

## Phase 6: User Story 4 - Navigate Between Modules and Chapters Efficiently (Priority: P2)

**Goal**: Enable students to easily navigate between the different modules and chapters using a structured sidebar

**Independent Test**: Can use the sidebar to navigate between different sections and ensure smooth transitions between content

### Implementation for User Story 4

- [ ] T038 [US4] Update Docusaurus sidebar configuration to follow required structure in sidebars.ts
- [ ] T039 [P] [US4] Create navigation components in src/components/Navigation/
- [ ] T040 [US4] Implement module/chapter navigation logic in frontend
- [ ] T041 [US4] Add user session tracking to remember current position in backend/src/services/user_service.py
- [ ] T042 [US4] Implement PUT /api/session/{sessionId}/progress endpoint in backend/src/routes/user.py
- [ ] T043 [US4] Create POST /api/session endpoint in backend/src/routes/user.py
- [ ] T044 [US4] Integrate navigation state with user sessions
- [ ] T045 [US4] Create custom Docusaurus theme components for enhanced navigation in src/theme/

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: User Story 5 - Engage in Advanced Learning Through Chatbot Features (Priority: P3)

**Goal**: Enable advanced students to use advanced features of the RAG chatbot such as concept comparisons, detailed explanations, and practice questions

**Independent Test**: Can use advanced features of the chatbot and verify they provide value-added functionality based on the educational content

### Implementation for User Story 5

- [ ] T046 [US5] Extend ChatService with advanced features in backend/src/services/chat_service.py
- [ ] T047 [P] [US5] Add advanced UI controls to chatbot component in src/components/Chatbot/
- [ ] T048 [US5] Implement content comparison functionality in backend/src/services/embedding_service.py
- [ ] T049 [US5] Add practice question generation to ChatService in backend/src/services/chat_service.py
- [ ] T050 [US5] Update chat API to support advanced request types in backend/src/routes/chat.py
- [ ] T051 [US5] Enhance frontend chat interface with advanced options in src/components/Chatbot/
- [ ] T052 [US5] Add detailed explanation generation in backend/src/services/chat_service.py

---

## Phase 8: Content Creation & Integration

**Purpose**: Create all required educational content for the 4 modules × 4 chapters structure

- [ ] T053 Create Module 1: The Robotic Nervous System content (overview, weekly-breakdown, deep-dive, practical-lab, simulation, quiz, assignment) in docs/module-1-the-robotic-nervous-system/
- [ ] T054 Create Module 2: The Digital Twin content (overview, weekly-breakdown, deep-dive, practical-lab, simulation, quiz, assignment) in docs/module-2-the-digital-twin/
- [ ] T055 Create Module 3: The AI Robot Brain content (overview, weekly-breakdown, deep-dive, practical-lab, simulation, quiz, assignment) in docs/module-3-the-ai-robot-brain/
- [ ] T056 Create Module 4: Vision Language Action Systems content (overview, weekly-breakdown, deep-dive, practical-lab, simulation, quiz, assignment) in docs/module-4-vision-language-action-systems/
- [ ] T057 Add educational images to static/img/ and integrate with appropriate chapters
- [ ] T058 [P] Ingest all module content into RAG system using backend/src/scripts/ingest_content.py

---

## Phase 9: Deployment & Integration

**Purpose**: Deploy and integrate all components to create a complete, functional system

- [ ] T059 [P] Configure GitHub Pages deployment in docusaurus.config.js
- [ ] T060 [P] Set up backend API deployment configuration
- [ ] T061 Create deployment scripts for frontend and backend
- [ ] T062 Integrate chatbot component with all educational content pages in Docusaurus
- [ ] T063 Test full integration of all user stories end-to-end
- [ ] T064 Optimize performance for content delivery and chat responses
- [ ] T065 Create production environment configuration

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T066 [P] Documentation updates in docs/
- [ ] T067 Code cleanup and refactoring
- [ ] T068 Performance optimization across all stories
- [ ] T069 [P] Accessibility improvements
- [ ] T070 Security hardening
- [ ] T071 Run quickstart.md validation
- [ ] T072 Final testing and validation of all user stories

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Content Creation (Phase 8)**: Can run in parallel with user stories or after foundational
- **Deployment & Integration (Phase 9)**: Depends on multiple user stories being complete
- **Polish (Phase 10)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Builds on content structure from US1
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Builds on content structure from US1
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Uses content from US1
- **User Story 5 (P3)**: Can start after US2 completion - Extends chat capabilities

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create Module model in backend/src/models/content.py"
Task: "Create Chapter model in backend/src/models/content.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 and 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. Complete Phase 4: User Story 2
5. **STOP and VALIDATE**: Test User Stories 1 and 2 independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 + 2 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 3 → Test independently → Deploy/Demo
4. Add User Story 4 → Test independently → Deploy/Demo
5. Add User Story 5 → Test independently → Deploy/Demo
6. Add Content → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence