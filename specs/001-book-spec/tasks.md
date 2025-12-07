---
description: "Tasks for Physical AI & Humanoid Robotics Textbook"
---

# Tasks: 001-book-spec - Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/001-book-spec/`
**Prerequisites**: plan.md (required), adr/001-continuous-project-sync-github-auto-update-for-docusaurus.adr.md (required)

## Format: `[ID] [P?] [Category] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Category]**: High-level category for the task (e.g., Validation, Structure, CI/CD, Content)
- Include exact file paths in descriptions

## Phase 1: Project Structure & Initial Content Scaffolding

**Purpose**: Set up the basic Docusaurus project structure and initial module/chapter files.

- [X] T001 [P] [Structure] Verify Docusaurus project is initialized (package.json, docusaurus.config.js, sidebars.ts) - **TODO: Manual check**
- [X] T002 [P] [Structure] Create `docs/module-1-the-robotic-nervous-system/` directory
- [X] T003 [P] [Structure] Create `docs/module-2-the-digital-twin/` directory
- [X] T004 [P] [Structure] Create `docs/module-3-the-ai-robot-brain/` directory
- [X] T005 [P] [Structure] Create `docs/module-4-vision-language-action-systems/` directory
- [X] T006 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/overview.md` with boilerplate content
- [X] T007 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/weekly-breakdown.md` with boilerplate content
- [X] T008 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/deep-dive.md` with boilerplate content
- [X] T009 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/practical-lab.md` with boilerplate content
- [X] T010 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/simulation.md` with boilerplate content
- [X] T011 [P] [Scaffolding] Create `docs/module-1-the-robotic-nervous-system/assignment.md` with boilerplate content
- [X] T012 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/overview.md` with boilerplate content
- [X] T013 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/weekly-breakdown.md` with boilerplate content
- [X] T014 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/deep-dive.md` with boilerplate content
- [X] T015 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/practical-lab.md` with boilerplate content
- [X] T016 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/simulation.md` with boilerplate content
- [X] T017 [P] [Scaffolding] Create `docs/module-2-the-digital-twin/assignment.md` with boilerplate content
- [X] T018 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/overview.md` with boilerplate content
- [X] T019 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/weekly-breakdown.md` with boilerplate content
- [X] T020 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/deep-dive.md` with boilerplate content
- [X] T021 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/practical-lab.md` with boilerplate content
- [X] T022 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/simulation.md` with boilerplate content
- [X] T023 [P] [Scaffolding] Create `docs/module-3-the-ai-robot-brain/assignment.md` with boilerplate content
- [X] T024 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/overview.md` with boilerplate content
- [X] T025 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/weekly-breakdown.md` with boilerplate content
- [X] T026 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/deep-dive.md` with boilerplate content
- [X] T027 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/practical-lab.md` with boilerplate content
- [X] T028 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/simulation.md` with boilerplate content
- [X] T029 [P] [Scaffolding] Create `docs/module-4-vision-language-action-systems/assignment.md` with boilerplate content

---

## Phase 2: Continuous Project Sync & GitHub Auto-Update Automation

**Purpose**: Implement automated validation, auto-fix, and Git tasks for continuous synchronization and deployment.

### 2.1. Continuous Validation Tasks (Automated Checks)

- [X] T030 [P] [Validation] Implement script to validate `docusaurus.config.js` syntax and essential configurations (e.g., `baseUrl`, `projectName`, `organizationName`)
- [X] T031 [P] [Validation] Implement script to validate `sidebars.ts`: check for existing paths, correctly defined categories, and no broken links
- [X] T032 [P] [Validation] Implement script to validate `package.json`: consistent dependency versions, correct script definitions, metadata integrity
- [X] T033 [P] [Validation] Implement script to validate `tsconfig.json`: proper TypeScript configuration (paths, include directives)
- [X] T034 [P] [Validation] Implement script to validate `docs/` and `category.json` structure: expected module/chapter folders/files, `category.json` structure, content naming conventions
- [X] T035 [P] [Validation] Implement script to validate `static/` and `src/` structure: presence and correct organization of static assets and custom React components/pages

### 2.2. Auto-Fix Tasks (Automated Remediation)

- [X] T036 [P] [Auto-Fix] Implement script to automatically create missing module folders in `docs/`
- [X] T037 [P] [Auto-Fix] Implement script to automatically create missing markdown chapter files in each module with boilerplate content
- [X] T038 [P] [Auto-Fix] Implement script to sync `sidebars.ts` entries with actual file system paths (add new, remove stale)
- [X] T039 [P] [Auto-Fix] Implement script to automatically correct `baseUrl`, `projectName`, `organizationName` in `docusaurus.config.js` based on GitHub repo details
- [X] T040 [P] [CI/CD] Ensure/Add GitHub Actions workflow (`.github/workflows/deploy.yml`) is present and correctly configured for Docusaurus deployment to GitHub Pages - **TODO: Check existing deploy.yml**

### 2.3. Git Tasks (Automated Workflow)

- [X] T041 [P] [Git] Implement script to auto commit meaningful changes resulting from auto-fix tasks or content generation
- [X] T042 [P] [Git] Implement script to auto push committed changes to the connected GitHub repository (e.g., `main` or `gh-pages` branch)

---

## Phase 3: Chapter Writing Pipeline (Phase 2) - Content Creation

**Purpose**: Placeholder tasks for content generation (dependent on Phase 1 & 2 completion).

- [ ] T043 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/overview.md`
- [ ] T044 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/weekly-breakdown.md`
- [ ] T045 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/deep-dive.md`
- [ ] T046 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/practical-lab.md`
- [ ] T047 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/simulation.md`
- [ ] T048 [Content] TODO: Write detailed content for `docs/module-1-the-robotic-nervous-system/assignment.md`
- [ ] T049 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/overview.md`
- [ ] T050 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/weekly-breakdown.md`
- [ ] T051 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/deep-dive.md`
- [ ] T052 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/practical-lab.md`
- [ ] T053 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/simulation.md`
- [ ] T054 [Content] TODO: Write detailed content for `docs/module-2-the-digital-twin/assignment.md`
- [ ] T055 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/overview.md`
- [ ] T056 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/weekly-breakdown.md`
- [ ] T057 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/deep-dive.md`
- [ ] T058 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/practical-lab.md`
- [ ] T059 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/simulation.md`
- [ ] T060 [Content] TODO: Write detailed content for `docs/module-3-the-ai-robot-brain/assignment.md`
- [ ] T061 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/overview.md`
- [ ] T062 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/weekly-breakdown.md`
- [ ] T063 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/deep-dive.md`
- [ ] T064 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/practical-lab.md`
- [ ] T065 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/simulation.md`
- [ ] T066 [Content] TODO: Write detailed content for `docs/module-4-vision-language-action-systems/assignment.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Project Structure & Initial Content Scaffolding**: No dependencies.
- **Phase 2: Continuous Project Sync & GitHub Auto-Update Automation**: Depends on Phase 1 completion.
- **Phase 3: Chapter Writing Pipeline (Phase 2)**: Depends on Phase 1 and Phase 2 completion.

### Within Each Phase

- Tasks can generally be run in parallel where marked [P].
- Specific dependencies are noted in task descriptions.
