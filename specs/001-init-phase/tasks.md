# Tasks: Full Project Mode for Physical AI & Humanoid Robotics Book

**Feature Branch**: `001-init-phase` | **Date**: 2025-12-07 | **Spec**: /specs/001-init-phase/spec.md
**Input**: Full project mode for Physical AI & Humanoid Robotics book.

## Phase 1: Project Setup and Initial Validation

- **Task 1.1**: Analyze Project State
  - **Description**: Run `/sp.analyze` to perform a non-destructive cross-artifact consistency and quality analysis.
  - **Acceptance Criteria**: `/sp.analyze` completes without critical errors, providing insights into the project's current state.

- **Task 1.2**: Generate Missing Project Structure
  - **Description**: Execute `/sp.implement` to create all necessary module folders, `category.json` files, and chapter placeholder `.md` files as defined by the book's structure.
  - **Acceptance Criteria**:
    - All module directories created in `docs/` (`module-1-the-robotic-nervous-system`, `module-2-the-digital-twin`, `module-3-the-ai-robot-brain`, `module-4-vision-language-action-systems`, `capstone-the-autonomous-humanoid`, `appendices`).
    - `category.json` files present in each module and appendices directory.
    - All specified chapter placeholder `.md` files created within their respective module directories.
    - Global appendices markdown files (`hardware-requirements.md`, `lab-architecture.md`, `cloud-vs-onprem.md`) created under `docs/appendices/`.

- **Task 1.3**: Verify Sidebar and Docusaurus Configuration
  - **Description**: Validate `sidebars.ts` and `docusaurus.config.ts` for correct referencing of all modules, chapters, and global settings for GitHub Pages deployment.
  - **Acceptance Criteria**:
    - `sidebars.ts` correctly lists all modules and chapters.
    - `docusaurus.config.ts` has `baseUrl = "/my-book/"` and `url = "https://<org>.github.io"`.
    - `tsconfig.json` is correctly configured (if needed).
    - GitHub Actions workflow (`.github/workflows/deploy.yml`) is valid for GitHub Pages.

## Phase 2: Book Authoring Pipeline

- **Task 2.1**: Implement Module Writing Workflow
  - **Description**: For each module (ROS 2, Digital Twin, AI-Robot Brain, VLA, Capstone), implement a process to generate its associated markdown files (`overview.md`, `weekly-breakdown.md`, `deep-dive.md`, `practical-lab.md`, `simulation.md`, `assignment.md`, `quiz.md`).
  - **Acceptance Criteria (per chapter)**:
    - Chapter content is deterministic, research-backed, citation-checked, formal technical textbook tone, no hallucinations, strictly follows course description.
    - Includes diagrams (Mermaid), tables, equations (LaTeX), and code blocks.
    - Each module's specific markdown files are generated and contain initial content.

- **Task 2.2**: Integrate Post-Chapter Generation Workflow (Continuous GitHub Sync & Deployment)
  - **Description**: After each chapter generation, automate the following steps: update filesystem, update `sidebars.ts`, validate Docusaurus build, auto-commit using `/sp.git.commit_pr`, and auto-push to GitHub.
  - **Acceptance Criteria (per chapter cycle)**:
    - Filesystem updated with new/edited markdown.
    - `sidebars.ts` updated correctly.
    - `npm run build` completes successfully without errors.
    - A new git commit is created with a deterministic summary using `/sp.git.commit_pr`.
    - Changes are successfully pushed to GitHub.
    - `.github/workflows/deploy.yml` remains valid and functional.

## Phase 3: Final Deliverables and Verification

- **Task 3.1**: Generate Global Appendices
  - **Description**: Auto-generate the content for global appendices (`hardware-requirements.md`, `lab-architecture.md`, `cloud-vs-onprem.md`).
  - **Acceptance Criteria**: Appendix files are created under `docs/appendices/` with relevant, high-quality content.

- **Task 3.2**: Generate Capstone Chapter
  - **Description**: Auto-generate the content for the "Capstone: The Autonomous Humanoid" module.
  - **Acceptance Criteria**: Capstone chapter files are created and contain high-quality, comprehensive content.

- **Task 3.3**: Generate Project README
  - **Description**: Auto-generate a comprehensive `README.md` for the project.
  - **Acceptance Criteria**: `README.md` is created/updated at the project root with relevant project information.

- **Task 3.4**: Final Build Validation
  - **Description**: Perform a final Docusaurus build validation to ensure the entire book project builds without errors.
  - **Acceptance Criteria**: `npm run build` completes successfully for the entire project.

- **Task 3.5**: Final Deployment Verification
  - **Description**: Verify the end-to-end GitHub deployment, ensuring the book is accessible via GitHub Pages.
  - **Acceptance Criteria**: The deployed book on GitHub Pages is functional and displays all content correctly.

- **Task 3.6**: Integrate RAG Chatbot Placeholder
  - **Description**: Include a placeholder for RAG chatbot integration.
  - **Acceptance Criteria**: A clear placeholder and instructions for RAG chatbot integration are present in the relevant documentation (e.g., `README.md` or a dedicated `RAG_integration.md`).

- **Task 3.7**: Document FastAPI/Qdrant/Neon Backend Instructions
  - **Description**: Provide instructions for the FastAPI/Qdrant/Neon backend, presumably for the RAG chatbot.
  - **Acceptance Criteria**: Instructions for setting up and using the FastAPI/Qdrant/Neon backend are clearly documented.

- **Task 3.8**: Implement Chapter Interaction Buttons
  - **Description**: Add "Personalize Chapter" and "Translate to Urdu" buttons within the chapter interface.
  - **Acceptance Criteria**: The two specified buttons are implemented and functional within the Docusaurus chapter pages.

- **Task 3.9**: Document Reusable Agent Skills
  - **Description**: Create documentation for reusable agent skills used in this project.
  - **Acceptance Criteria**: A `reusable-agent-skills.md` file (or similar) is created with clear documentation for any custom agent skills.

- **Task 3.10**: Ensure Spec Documents and ADRs are Present
  - **Description**: Verify that all spec documents are in the `specs/` folder and all ADRs are in the `adr/` folder.
  - **Acceptance Criteria**: All feature specifications (`spec.md`, `plan.md`, `tasks.md`) are in `specs/001-init-phase/` and any generated ADRs are in `history/adr/`.
