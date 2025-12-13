---
id: 1
title: educational-book-rag-spec
stage: spec
date: 2025-12-12
surface: agent
model: Qwen
feature: educational-book-rag
branch: 001-educational-book-rag
user: Qwen
command: /sp.specify
labels: ["specification","educational","rag","chatbot","docusaurus"]
links:
  spec: ./specs/001-educational-book-rag/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/001-educational-book-rag/spec.md
 - specs/001-educational-book-rag/checklists/requirements.md
tests:
 - specs/001-educational-book-rag/checklists/requirements.md
---

## Prompt

```
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

The text the user typed after `/sp.specify` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `You are the Specification Phase Engine for a Spec-Driven Development project. Your job is to read the following requirement summary and produce: 1. A complete, unambiguous **functional specification** 2. A clear **non-functional specification** 3. Fully structured **acceptance criteria** 4. Full **system boundaries** 5. Complete **actors & roles** 6. Complete **domain definitions** 7. Complete **user journeys** 8. Complete **module-level specification** 9. Complete **chapter-level specification** 10. A final **SDD-grade specification document** ### REQUIREMENT SUMMARY We are creating a complete educational book project using: - **Docusaurus** - **Spec-Kit Plus** - **Claude Code** - **GitHub Pages deployment** The book contains: - **4 modules** - Each module contains **4 chapters** - Each chapter will include images stored in `src/static/img/` - Sidebar should begin with: - **Book Introduction** - Followed by: Module 1 → Module 2 → Module 3 → Module 4 The project must also include: - **Integrated RAG Chatbot** built using: - OpenAI Agents / ChatKit SDK - FastAPI backend - Neon Serverless Postgres - Vector embeddings + semantic search - Document ingestion pipeline - Secure API endpoints Your responsibility is to create a fully detailed specification describing the entire project, module structure, chapter structure, image placement requirements, and RAG chatbot integration. ### EXPECTED OUTPUT Produce a **full SDD Specification Document**, ready for use in the next phase.` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

1. **Generate a concise short name** (2-4 words) for the branch:
   - Analyze the feature description and extract the most meaningful keywords
   - Create a 2-4 word short name that captures the essence of the feature
   - Use action-noun format when possible (e.g., "add-user-auth", "fix-payment-bug")
   - Preserve technical terms and acronyms (OAuth2, API, JWT, etc.)
   - Keep it concise but descriptive enough to understand the feature at a glance
   - Examples:
     - "I want to add user authentication" → "user-auth"
     - "Implement OAuth2 integration for the API" → "oauth2-api-integration"
     - "Create a dashboard for analytics" → "analytics-dashboard"
     - "Fix payment processing timeout bug" → "fix-payment-timeout"

2. **Check for existing branches before creating new one**:
   
   a. First, fetch all remote branches to ensure we have the latest information:
      ```bash
      git fetch --all --prune
      ```
   
   b. Find the highest feature number across all sources for the short-name:
      - Remote branches: `git ls-remote --heads origin | grep -E 'refs/heads/[0-9]+-<short-name>$'`
      - Local branches: `git branch | grep -E '^[* ]*[0-9]+-<short-name>$'`
      - Specs directories: Check for directories matching `specs/[0-9]+-<short-name>`
   
   c. Determine the next available number:
      - Extract all numbers from all three sources
      - Find the highest number N
      - Use N+1 for the new branch number
   
   d. Run the script `.specify/scripts/powershell/create-new-feature.ps1 -Json "You are the Specification Phase Engine for a Spec-Driven Development project. Your job is to read the following requirement summary and produce: 1. A complete, unambiguous **functional specification** 2. A clear **non-functional specification** 3. Fully structured **acceptance criteria** 4. Full **system boundaries** 5. Complete **actors & roles** 6. Complete **domain definitions** 7. Complete **user journeys** 8. Complete **module-level specification** 9. Complete **chapter-level specification** 10. A final **SDD-grade specification document** ### REQUIREMENT SUMMARY We are creating a complete educational book project using: - **Docusaurus** - **Spec-Kit Plus** - **Claude Code** - **GitHub Pages deployment** The book contains: - **4 modules** - Each module contains **4 chapters** - Each chapter will include images stored in `src/static/img/` - Sidebar should begin with: - **Book Introduction** - Followed by: Module 1 → Module 2 → Module 3 → Module 4 The project must also include: - **Integrated RAG Chatbot** built using: - OpenAI Agents / ChatKit SDK - FastAPI backend - Neon Serverless Postgres - Vector embeddings + semantic search - Document ingestion pipeline - Secure API endpoints Your responsibility is to create a fully detailed specification describing the entire project, module structure, chapter structure, image placement requirements, and RAG chatbot integration. ### EXPECTED OUTPUT Produce a **full SDD Specification Document**, ready for use in the next phase."` with the calculated number and short-name:
      - Pass `--number N+1` and `--short-name "your-short-name"` along with the feature description
      - Bash example: `.specify/scripts/powershell/create-new-feature.ps1 -Json "You are the Specification Phase Engine for a Spec-Driven Development project. Your job is to read the following requirement summary and produce: 1. A complete, unambiguous **functional specification** 2. A clear **non-functional specification** 3. Fully structured **acceptance criteria** 4. Full **system boundaries** 5. Complete **actors & roles** 6. Complete **domain definitions** 7. Complete **user journeys** 8. Complete **module-level specification** 9. Complete **chapter-level specification** 10. A final **SDD-grade specification document** ### REQUIREMENT SUMMARY We are creating a complete educational book project using: - **Docusaurus** - **Spec-Kit Plus** - **Claude Code** - **GitHub Pages deployment** The book contains: - **4 modules** - Each module contains **4 chapters** - Each chapter will include images stored in `src/static/img/` - Sidebar should begin with: - **Book Introduction** - Followed by: Module 1 → Module 2 → Module 3 → Module 4 The project must also include: - **Integrated RAG Chatbot** built using: - OpenAI Agents / ChatKit SDK - FastAPI backend - Neon Serverless Postgres - Vector embeddings + semantic search - Document ingestion pipeline - Secure API endpoints Your responsibility is to create a fully detailed specification describing the entire project, module structure, chapter structure, image placement requirements, and RAG chatbot integration. ### EXPECTED OUTPUT Produce a **full SDD Specification Document**, ready for use in the next phase." --json --number 5 --short-name "user-auth" "Add user authentication"`
      - PowerShell example: `.specify/scripts/powershell/create-new-feature.ps1 -Json "You are the Specification Phase Engine for a Spec-Driven Development project. Your job is to read the following requirement summary and produce: 1. A complete, unambiguous **functional specification** 2. A clear **non-functional specification** 3. Fully structured **acceptance criteria** 4. Full **system boundaries** 5. Complete **actors & roles** 6. Complete **domain definitions** 7. Complete **user journeys** 8. Complete **module-level specification** 9. Complete **chapter-level specification** 10. A final **SDD-grade specification document** ### REQUIREMENT SUMMARY We are creating a complete educational book project using: - **Docusaurus** - **Spec-Kit Plus** - **Claude Code** - **GitHub Pages deployment** The book contains: - **4 modules** - Each module contains **4 chapters** - Each chapter will include images stored in `src/static/img/` - Sidebar should begin with: - **Book Introduction** - Followed by: Module 1 → Module 2 → Module 3 → Module 4 The project must also include: - **Integrated RAG Chatbot** built using: - OpenAI Agents / ChatKit SDK - FastAPI backend - Neon Serverless Postgres - Vector embeddings + semantic search - Document ingestion pipeline - Secure API endpoints Your responsibility is to create a fully detailed specification describing the entire project, module structure, chapter structure, image placement requirements, and RAG chatbot integration. ### EXPECTED OUTPUT Produce a **full SDD Specification Document**, ready for use in the next phase." -Json -Number 5 -ShortName "user-auth" "Add user authentication"`
   
   **IMPORTANT**:
   - Check all three sources (remote branches, local branches, specs directories) to find the highest number
   - Only match branches/directories with the exact short-name pattern
   - If no existing branches/directories found with this short-name, start with number 1
   - You must only ever run this script once per feature
   - The JSON is provided in the terminal as output - always refer to it to get the actual content you're looking for
   - The JSON output will contain BRANCH_NAME and SPEC_FILE paths
   - For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot")

3. Load `.specify/templates/spec-template.md` to understand required sections.

4. Follow this execution flow:

    1. Parse user description from Input
       If empty: ERROR "No feature description provided"
    2. Extract key concepts from description
       Identify: actors, actions, data, constraints
    3. For unclear aspects:
       - Make informed guesses based on context and industry standards
       - Only mark with [NEEDS CLARIFICATION: specific question] if:
         - The choice significantly impacts feature scope or user experience
         - Multiple reasonable interpretations exist with different implications
         - No reasonable default exists
       - **LIMIT: Maximum 3 [NEEDS CLARIFICATION] markers total**
       - Prioritize clarifications by impact: scope > security/privacy > user experience > technical details
    4. Fill User Scenarios & Testing section
       If no clear user flow: ERROR "Cannot determine user scenarios"
    5. Generate Functional Requirements
       Each requirement must be testable
       Use reasonable defaults for unspecified details (document assumptions in Assumptions section)
    6. Define Success Criteria
       Create measurable, technology-agnostic outcomes
       Include both quantitative metrics (time, performance, volume) and qualitative measures (user satisfaction, task completion)
       Each criterion must be verifiable without implementation details
    7. Identify Key Entities (if data involved)
    8. Return: SUCCESS (spec ready for planning)

5. Write the specification to SPEC_FILE using the template structure, replacing placeholders with concrete details derived from the feature description (arguments) while preserving section order and headings.

6. **Specification Quality Validation**: After writing the initial spec, validate it against quality criteria:

   a. **Create Spec Quality Checklist**: Generate a checklist file at `FEATURE_DIR/checklists/requirements.md` using the checklist template structure with these validation items:

      ```markdown
      # Specification Quality Checklist: [FEATURE NAME]
      
      **Purpose**: Validate specification completeness and quality before proceeding to planning
      **Created**: [DATE]
      **Feature**: [Link to spec.md]
      
      ## Content Quality
      
      - [ ] No implementation details (languages, frameworks, APIs)
      - [ ] Focused on user value and business needs
      - [ ] Written for non-technical stakeholders
      - [ ] All mandatory sections completed
      
      ## Requirement Completeness
      
      - [ ] No [NEEDS CLARIFICATION] markers remain
      - [ ] Requirements are testable and unambiguous
      - [ ] Success criteria are measurable
      - [ ] Success criteria are technology-agnostic (no implementation details)
      - [ ] All acceptance scenarios are defined
      - [ ] Edge cases are identified
      - [ ] Scope is clearly bounded
      - [ ] Dependencies and assumptions identified
      
      ## Feature Readiness
      
      - [ ] All functional requirements have clear acceptance criteria
      - [ ] User scenarios cover primary flows
      - [ ] Feature meets measurable outcomes defined in Success Criteria
      - [ ] No implementation details leak into specification
      
      ## Notes
      
      - Items marked incomplete require spec updates before `/sp.clarify` or `/sp.plan`
      ```

   b. **Run Validation Check**: Review the spec against each checklist item:
      - For each item, determine if it passes or fails
      - Document specific issues found (quote relevant spec sections)

   c. **Handle Validation Results**:

      - **If all items pass**: Mark checklist complete and proceed to step 6

      - **If items fail (excluding [NEEDS CLARIFICATION])**:
        1. List the failing items and specific issues
        2. Update the spec to address each issue
        3. Re-run validation until all items pass (max 3 iterations)
        4. If still failing after 3 iterations, document remaining issues in checklist notes and warn user

      - **If [NEEDS CLARIFICATION] markers remain**:
        1. Extract all [NEEDS CLARIFICATION: ...] markers from the spec
        2. **LIMIT CHECK**: If more than 3 markers exist, keep only the 3 most critical (by scope/security/UX impact) and make informed guesses for the rest
        3. For each clarification needed (max 3), present options to user in this format:

           ```markdown
           ## Question [N]: [Topic]
           
           **Context**: [Quote relevant spec section]
           
           **What we need to know**: [Specific question from NEEDS CLARIFICATION marker]
           
           **Suggested Answers**:
           
           | Option | Answer | Implications |
           |--------|--------|--------------|
           | A      | [First suggested answer] | [What this means for the feature] |
           | B      | [Second suggested answer] | [What this means for the feature] |
           | C      | [Third suggested answer] | [What this means for the feature] |
           | Custom | Provide your own answer | [Explain how to provide custom input] |
           
           **Your choice**: _[Wait for user response]_
           ```

        4. **CRITICAL - Table Formatting**: Ensure markdown tables are properly formatted:
           - Use consistent spacing with pipes aligned
           - Each cell should have spaces around content: `| Content |` not `|Content|`
           - Header separator must have at least 3 dashes: `|--------|`
           - Test that the table renders correctly in markdown preview
        5. Number questions sequentially (Q1, Q2, Q3 - max 3 total)
        6. Present all questions together before waiting for responses
        7. Wait for user to respond with their choices for all questions (e.g., "Q1: A, Q2: Custom - [details], Q3: B")
        8. Update the spec by replacing each [NEEDS CLARIFICATION] marker with the user's selected or provided answer
        9. Re-run validation after all clarifications are resolved

   d. **Update Checklist**: After each validation iteration, update the checklist file with current pass/fail status

7. Report completion with branch name, spec file path, checklist results, and readiness for the next phase (`/sp.clarify` or `/sp.plan`).

**NOTE:** The script creates and checks out the new branch and initializes the spec file before writing.

## General Guidelines

## Quick Guidelines

- Focus on **WHAT** users need and **WHY**.
- Avoid HOW to implement (no tech stack, APIs, code structure).
- Written for business stakeholders, not developers.
- DO NOT create any checklists that are embedded in the spec. That will be a separate command.

### Section Requirements

- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

### For AI Generation

When creating this spec from a user prompt:

1. **Make informed guesses**: Use context, industry standards, and common patterns to fill gaps
2. **Document assumptions**: Record reasonable defaults in the Assumptions section
3. **Limit clarifications**: Maximum 3 [NEEDS CLARIFICATION] markers - use only for critical decisions that:
   - Significantly impact feature scope or user experience
   - Have multiple reasonable interpretations with different implications
   - Lack any reasonable default
4. **Prioritize clarifications**: scope > security/privacy > user experience > technical details
5. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
6. **Common areas needing clarification** (only if no reasonable default exists):
   - Feature scope and boundaries (include/exclude specific use cases)
   - User types and permissions (if multiple conflicting interpretations possible)
   - Security/compliance requirements (when legally/financially significant)

**Examples of reasonable defaults** (don't ask about these):

- Data retention: Industry-standard practices for the domain
- Performance targets: Standard web/mobile app expectations unless specified
- Error handling: User-friendly messages with appropriate fallbacks
- Authentication method: Standard session-based or OAuth2 for web apps
- Integration patterns: RESTful APIs unless specified otherwise

### Success Criteria Guidelines

Success criteria must be:

1. **Measurable**: Include specific metrics (time, percentage, count, rate)
2. **Technology-agnostic**: No mention of frameworks, languages, databases, or tools
3. **User-focused**: Describe outcomes from user/business perspective, not system internals
4. **Verifiable**: Can be tested/validated without knowing implementation details

**Good examples**:

- "Users can complete checkout in under 3 minutes"
- "System supports 10,000 concurrent users"
- "95% of searches return results in under 1 second"
- "Task completion rate improves by 40%"

**Bad examples** (implementation-focused):

- "API response time is under 200ms" (too technical, use "Users see results instantly")
- "Database can handle 1000 TPS" (implementation detail, use user-facing metric)
- "React components render efficiently" (framework-specific)
- "Redis cache hit rate above 80%" (technology-specific)

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.
## Response snapshot

```
# Feature Specification: Educational Book with Integrated RAG Chatbot

**Feature Branch**: `001-educational-book-rag`
**Created**: 12/12/2025
**Status**: Draft
**Input**: User description: "Educational book project using Docusaurus with 4 modules (each with 4 chapters) and integrated RAG chatbot for enhanced learning experience"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Educational Content (Priority: P1)

As a student or learner, I want to access structured educational content organized into modules and chapters, so that I can follow a progressive learning path through the humanoid robotics course.

**Why this priority**: This is the foundational functionality - without accessible educational content, the entire learning experience fails.

**Independent Test**: Can be fully tested by navigating through the book's modules and chapters, verifying that content loads correctly and is properly structured.

**Acceptance Scenarios**:

1. **Given** user accesses the educational platform, **When** they navigate to the book introduction, **Then** they can read the introductory content and access the first module
2. **Given** user is reading content in one chapter, **When** they want to move to the next chapter, **Then** they can use navigation elements to go to the next logical section

---

### User Story 2 - Interact with RAG Chatbot (Priority: P1)

As a student or learner, I want to interact with an AI-powered chatbot that can answer questions about the educational content, so that I can get instant clarification and support during my studies.

**Why this priority**: This feature provides intelligent assistance and enhances the learning experience significantly.

**Independent Test**: Can be tested by entering various queries related to the educational content and verifying that the chatbot provides accurate, relevant responses based on the course material.

**Acceptance Scenarios**:

1. **Given** user has read a chapter of the educational content, **When** they ask the chatbot a question about that content, **Then** the chatbot provides a response based on the course material with appropriate citations
2. **Given** user enters an off-topic query, **When** they submit it to the chatbot, **Then** the chatbot responds appropriately acknowledging the query but redirecting to relevant course topics

---

### User Story 3 - View Supporting Images and Media (Priority: P2)

As a student or learner, I want to view supporting images and diagrams within each chapter, so that I can better understand complex concepts related to humanoid robotics.

**Why this priority**: Visual aids significantly enhance comprehension of robotics concepts, making this valuable but secondary to core content.

**Independent Test**: Can be tested by viewing different chapters and verifying that images appear correctly positioned with appropriate alt text and captions.

**Acceptance Scenarios**:

1. **Given** user is reading a chapter that contains images, **When** they scroll to the image location, **Then** the image displays correctly with proper scaling and positioning
2. **Given** user is using assistive technology, **When** they navigate to an image, **Then** they can access appropriate alternative text describing the image content

---

### User Story 4 - Navigate Between Modules and Chapters Efficiently (Priority: P2)

As a student or learner, I want to easily navigate between the different modules and chapters using a structured sidebar, so that I can jump to specific sections or review previous content.

**Why this priority**: Efficient navigation is essential for a good learning experience but is secondary to having the content accessible in the first place.

**Independent Test**: Can be tested by using the sidebar to navigate between different sections and ensuring smooth transitions between content.

**Acceptance Scenarios**:

1. **Given** user is reading a chapter in Module 3, **When** they want to return to the Book Introduction, **Then** they can navigate there using the sidebar
2. **Given** user wants to skip ahead to a later chapter, **When** they select it from the sidebar, **Then** they can access that content seamlessly

---

### User Story 5 - Engage in Advanced Learning Through Chatbot Features (Priority: P3)

As an advanced student, I want to use advanced features of the RAG chatbot such as concept comparisons, detailed explanations, and practice questions, so that I can deepen my understanding of humanoid robotics.

**Why this priority**: These advanced features would enhance the learning experience but are not required for basic functionality.

**Independent Test**: Can be tested by using advanced features of the chatbot and verifying they provide value-added functionality based on the educational content.

**Acceptance Scenarios**:

1. **Given** user is studying complex robotic algorithms, **When** they ask for a comparison between two algorithms, **Then** the chatbot provides a detailed comparison based on the course material
2. **Given** user requests practice questions about the current topic, **When** they engage with the chatbot, **Then** they receive relevant questions based on the educational content

---

### Edge Cases

- What happens when a user tries to access content offline? The core content should still be available if deployed via GitHub Pages.
- How does the system handle extremely large text queries to the chatbot? Input should be validated and limited to reasonable lengths.
- How does the system handle multimedia queries to the chatbot? The system should gracefully handle text-only interactions.
- What happens when multiple users query the chatbot simultaneously? System should handle concurrent requests efficiently.
- How does the system handle content that hasn't been indexed in the RAG system? Appropriate fallback responses should be provided.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide educational content organized into 4 main modules
- **FR-002**: System MUST provide 4 chapters within each module for a total of 16 chapters
- **FR-003**: System MUST allow users to navigate content using a sidebar that begins with Book Introduction followed by Module 1 through Module 4
- **FR-004**: System MUST display images stored in `src/static/img/` alongside relevant educational content
- **FR-005**: System MUST include an integrated RAG (Retrieval Augmented Generation) chatbot
- **FR-006**: System MUST store educational content in a way that enables semantic search for the RAG chatbot
- **FR-007**: System MUST allow users to submit questions to the RAG chatbot and receive contextually relevant responses
- **FR-008**: System MUST serve the educational content and chatbot interface through GitHub Pages
- **FR-009**: System MUST ensure the chatbot's responses are based on the educational content using vector embeddings for semantic search
- **FR-010**: System MUST implement secure API endpoints for the RAG chatbot backend services

### Key Entities

- **Module**: Represents one of the four main sections of the educational book, containing 4 chapters each
- **Chapter**: Individual lesson units that form part of a module, each containing text content, images, and exercises
- **Learning Content**: Educational materials including text, images, diagrams, and other media that comprise the course curriculum
- **Chat Query**: Questions or prompts submitted by users to the RAG chatbot for assistance with the educational content
- **Vector Embeddings**: Numerical representations of the educational content that enable semantic search capabilities
- **User Session**: Temporary data structure that maintains the state of a user's interaction with the educational platform

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Students can access all 16 chapters organized in 4 modules through intuitive navigation within 30 seconds of landing on the homepage
- **SC-002**: The RAG chatbot responds to user queries with relevant information from the educational content with at least 85% accuracy in relevance
- **SC-003**: At least 80% of students using the RAG chatbot report increased understanding of difficult concepts compared to text-only resources
- **SC-004**: Students can successfully view all course images without loading delays exceeding 3 seconds
- **SC-005**: The platform serves content reliably with 99% uptime during peak usage hours
- **SC-006**: Students can engage with the RAG chatbot to receive responses within 5 seconds for typical queries
```

## Outcome

- ✅ Impact: Created complete SDD specification for educational book with integrated RAG chatbot
- 🧪 Tests: Generated quality checklist to validate spec completeness
- 📁 Files: Created spec.md and requirements checklist in specs/001-educational-book-rag/
- 🔁 Next prompts: Ready for next phases like /sp.plan or /sp.clarify
- 🧠 Reflection: Successfully created comprehensive specification with user stories, requirements, and success criteria

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): Standard spec generation
- Next experiment (smallest change to try): Proceed to planning phase