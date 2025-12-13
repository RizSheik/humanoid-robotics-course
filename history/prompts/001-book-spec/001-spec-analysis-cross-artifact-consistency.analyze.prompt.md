---
id: "001"
title: "Spec-Analysis-Cross-Artifact-Consistency"
stage: "analyze"
date_iso: "2025-12-05"
surface: "agent"
model: "Qwen"
feature: "001-book-spec"
branch: "001-book-spec"
user: "user"
command: "/sp.analyze"
labels: ["spec", "analysis", "quality", "cross-artifact"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "specs/001-book-spec/spec.md"
  - "specs/001-book-spec/plan.md"
  - "specs/001-book-spec/tasks.md"
  - ".specify/memory/constitution.md"
tests_yaml: []
prompt_text: |
  ---
  description: Perform a non-destructive cross-artifact consistency and quality analysis across spec.md, plan.md, and tasks.md after task generation.
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Goal

  Identify inconsistencies, duplications, ambiguities, and underspecified items across the three core artifacts (`spec.md`, `plan.md`, `tasks.md`) before implementation. This command MUST run only after `/sp.tasks` has successfully produced a complete `tasks.md`.

  ## Operating Constraints

  **STRICTLY READ-ONLY**: Do **not** modify any files. Output a structured analysis report. Offer an optional remediation plan (user must explicitly approve before any follow-up editing commands would be invoked manually).

  **Constitution Authority**: The project constitution (`.specify/memory/constitution.md`) is **non-negotiable** within this analysis scope. Constitution conflicts are automatically CRITICAL and require adjustment of the spec, plan, or tasks—not dilution, reinterpretation, or silent ignoring of the principle. If a principle itself needs to change, that must occur in a separate, explicit constitution update outside `/sp.analyze`.

  ## Execution Steps

  ### 1. Initialize Analysis Context

  Run `.specify/scripts/powershell/check-prerequisites.ps1 -Json -RequireTasks -IncludeTasks` once from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS. Derive absolute paths:

  - SPEC = FEATURE_DIR/spec.md
  - PLAN = FEATURE_DIR/plan.md
  - TASKS = FEATURE_DIR/tasks.md

  Abort with an error message if any required file is missing (instruct the user to run missing prerequisite command).
  For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

  ### 2. Load Artifacts (Progressive Disclosure)

  Load only the minimal necessary context from each artifact:

  **From spec.md:**

  - Overview/Context
  - Functional Requirements
  - Non-Functional Requirements
  - User Stories
  - Edge Cases (if present)

  **From plan.md:**

  - Architecture/stack choices
  - Data Model references
  - Phases
  - Technical constraints

  **From tasks.md:**

  - Task IDs
  - Descriptions
  - Phase grouping
  - Parallel markers [P]
  - Referenced file paths

  **From constitution:**

  - Load `.specify/memory/constitution.md` for principle validation

  ### 3. Build Semantic Models

  Create internal representations (do not include raw artifacts in output):

  - **Requirements inventory**: Each functional + non-functional requirement with a stable key (derive slug based on imperative phrase; e.g., "User can upload file" → `user-can-upload-file`)
  - **User story/action inventory**: Discrete user actions with acceptance criteria
  - **Task coverage mapping**: Map each task to one or more requirements or stories (inference by keyword / explicit reference patterns like IDs or key phrases)
  - **Constitution rule set**: Extract principle names and MUST/SHOULD normative statements

  ### 4. Detection Passes (Token-Efficient Analysis)

  Focus on high-signal findings. Limit to 50 findings total; aggregate remainder in overflow summary.

  #### A. Duplication Detection

  - Identify near-duplicate requirements
  - Mark lower-quality phrasing for consolidation

  #### B. Ambiguity Detection

  - Flag vague adjectives (fast, scalable, secure, intuitive, robust) lacking measurable criteria
  - Flag unresolved placeholders (TODO, TKTK, ???, `<placeholder>`, etc.)

  #### C. Underspecification

  - Requirements with verbs but missing object or measurable outcome
  - User stories missing acceptance criteria alignment
  - Tasks referencing files or components not defined in spec/plan

  #### D. Constitution Alignment

  - Any requirement or plan element conflicting with a MUST principle
  - Missing mandated sections or quality gates from constitution

  #### E. Coverage Gaps

  - Requirements with zero associated tasks
  - Tasks with no mapped requirement/story
  - Non-functional requirements not reflected in tasks (e.g., performance, security)

  #### F. Inconsistency

  - Terminology drift (same concept named differently across files)
  - Data entities referenced in plan but absent in spec (or vice versa)
  - Task ordering contradictions (e.g., integration tasks before foundational setup tasks without dependency note)
  - Conflicting requirements (e.g., one requires Next.js while other specifies Vue)

  ### 5. Severity Assignment

  Use this heuristic to prioritize findings:

  - **CRITICAL**: Violates constitution MUST, missing core spec artifact, or requirement with zero coverage that blocks baseline functionality
  - **HIGH**: Duplicate or conflicting requirement, ambiguous security/performance attribute, untestable acceptance criterion
  - **MEDIUM**: Terminology drift, missing non-functional task coverage, underspecified edge case
  - **LOW**: Style/wording improvements, minor redundancy not affecting execution order

  ### 6. Produce Compact Analysis Report

  Output a Markdown report (no file writes) with the following structure:

  ## Specification Analysis Report

  | ID | Category | Severity | Location(s) | Summary | Recommendation |
  |----|----------|----------|-------------|---------|----------------|
  | A1 | Duplication | HIGH | spec.md:L120-134 | Two similar requirements ... | Merge phrasing; keep clearer version |

  (Add one row per finding; generate stable IDs prefixed by category initial.)

  **Coverage Summary Table:**

  | Requirement Key | Has Task? | Task IDs | Notes |
  |-----------------|-----------|----------|-------|

  **Constitution Alignment Issues:** (if any)

  **Unmapped Tasks:** (if any)

  **Metrics:**

  - Total Requirements
  - Total Tasks
  - Coverage % (requirements with >=1 task)
  - Ambiguity Count
  - Duplication Count
  - Critical Issues Count

  ### 7. Provide Next Actions

  At end of report, output a concise Next Actions block:

  - If CRITICAL issues exist: Recommend resolving before `/sp.implement`
  - If only LOW/MEDIUM: User may proceed, but provide improvement suggestions
  - Provide explicit command suggestions: e.g., "Run /sp.specify with refinement", "Run /sp.plan to adjust architecture", "Manually edit tasks.md to add coverage for 'performance-metrics'"

  ### 8. Offer Remediation

  Ask the user: "Would you like me to suggest concrete remediation edits for the top N issues?" (Do NOT apply them automatically.)

  ## Operating Principles

  ### Context Efficiency

  - **Minimal high-signal tokens**: Focus on actionable findings, not exhaustive documentation
  - **Progressive disclosure**: Load artifacts incrementally; don't dump all content into analysis
  - **Token-efficient output**: Limit findings table to 50 rows; summarize overflow
  - **Deterministic results**: Rerunning without changes should produce consistent IDs and counts

  ### Analysis Guidelines

  - **NEVER modify files** (this is read-only analysis)
  - **NEVER hallucinate missing sections** (if absent, report them accurately)
  - **Prioritize constitution violations** (these are always CRITICAL)
  - **Use examples over exhaustive rules** (cite specific instances, not generic patterns)
  - **Report zero issues gracefully** (emit success report with coverage statistics)

  ## Context


response_text: |
  Based on my analysis of the three artifacts, here is the specification analysis report:

  ## Specification Analysis Report

  | ID | Category | Severity | Location(s) | Summary | Recommendation |
  |----|----------|----------|-------------|---------|----------------|
  | D1 | Duplication | MEDIUM | spec.md/L75, plan.md/L82 | Both spec and plan mention "Week Alignment" tag requirement | Consolidate into single authoritative location with reference |
  | A1 | Ambiguity | HIGH | spec.md/L118 | "13-week teaching flow" contradicts "14-week constraint" | Specify exact week count and reconcile the two requirements |
  | C1 | Constitution Alignment | CRITICAL | spec.md/L84, plan.md/L118, tasks.md/L54 | No explicit mention of zero plagiarism requirement from constitution | Add zero plagiarism compliance requirement to all artifacts |
  | U1 | Underspecification | MEDIUM | spec.md/L67-91 | User stories lack quantifiable acceptance criteria | Add measurable outcomes to each user story acceptance scenario |
  | G1 | Coverage Gap | MEDIUM | spec.md, tasks.md | NFR-Technical-Accuracy has insufficient task coverage | Add more tasks specifically focused on verifying technical accuracy |
  | I1 | Inconsistency | MEDIUM | spec.md/L120, plan.md/L25 | Spec requires "4-7 bullet points" but plan doesn't mention this | Add bullet point requirement to plan structure section |
  | I2 | Inconsistency | HIGH | spec.md/L131, plan.md/L95 | VLA module planned as capstone but spec describes end-to-end cognitive robotics as separate | Reconcile whether VLA or cognitive robotics is the final capstone |

  **Coverage Summary Table:**

  | Requirement Key | Has Task? | Task IDs | Notes |
  |-----------------|-----------|----------|-------|
  | user-can-learn-ros2-fundamentals | Yes | T008-T014 | Good coverage for US1 |
  | user-can-create-digital-twin | Yes | T015-T021 | Good coverage for US2 |
  | user-can-use-ai-robot-brain | Yes | T022-T028 | Good coverage for US3 |
  | user-can-understand-vla-systems | Yes | T029-T035 | Good coverage for US4 |
  | nfr-technical-accuracy | Partial | T037, T044-T048 | Need more dedicated tasks |
  | course-structure-constraint | Yes | T049-T050 | Covered in final tasks |
  | fr-001-four-modules | Yes | T004-T007 | Good coverage |
  | fr-002-to-fr-005 | Partial | T003, T014, T021, T028, T035 | Some covered but need more detail |

  **Constitution Alignment Issues:**
  - The constitution requires zero plagiarism (Core Principle) but this isn't explicitly mentioned in spec or tasks
  - The constitution requires content between 40k-80k words but no word count targets appear in plan or tasks
  - The constitution requires peer-reviewed sources but tasks don't explicitly validate this

  **Unmapped Tasks:**
  - T001-T002 (Setup tasks) - not directly mapped to functional requirements
  - T036-T054 (Final validation tasks) - map to NFRs or success criteria

  **Metrics:**
  - Total Requirements: 18 (12 FRs, 1 NFR, 5 USs)
  - Total Tasks: 54
  - Coverage % (requirements with >=1 task): 89%
  - Ambiguity Count: 1
  - Duplication Count: 1
  - Critical Issues Count: 1

  ## Next Actions

  There is 1 CRITICAL issue related to constitution alignment that needs to be resolved before implementation. The zero plagiarism requirement from the constitution is not explicitly mentioned in the spec or tasks. Additionally, there are 2 HIGH severity issues that should be addressed: the contradiction between 13-week and 14-week requirements, and the inconsistency between VLA and cognitive robotics as the final capstone.

  Recommendations:
  - Run /sp.specify to clarify the week count discrepancy 
  - Run /sp.plan to address the capstone module ambiguity
  - Manually edit tasks.md to add explicit plagiarism compliance tasks
  - Add more tasks specifically for verifying technical accuracy requirements

  ## Would you like me to suggest concrete remediation edits for the top issues?
outcome: "Successfully analyzed the three artifacts (spec.md, plan.md, tasks.md) for consistency and quality issues, identifying several gaps and inconsistencies that need to be addressed before implementation."
evaluation: "The analysis identified a critical constitution alignment issue (zero plagiarism requirement not explicitly mentioned in spec or tasks), two high-severity inconsistencies (week count discrepancy and VLA/cognitive robotics capstone), and several medium-severity issues related to underspecification and coverage gaps. This addresses the main objective of identifying inconsistencies, duplications, ambiguities, and underspecified items across the three core artifacts."
---
