# ADR-001: Continuous Project Sync and GitHub Auto-Update for Docusaurus

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-06
- **Feature:** 001-book-spec
- **Context:** The Docusaurus project requires a robust mechanism for continuous synchronization of content with the file system, automated validation of configuration and structure, and seamless deployment to GitHub Pages. This decision defines the architectural approach to achieve these goals, including CI/CD pipelines and a standardized module structure for content.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

- **Continuous Validation Tasks**: Automated checks for `docusaurus.config.js`, `sidebars.ts`, `package.json`, `tsconfig.json`, `docs/` and `category.json`, `static/` and `src/` structure.
- **Auto-Fix Tasks**: Automated generation of missing module folders and markdown chapter files, syncing sidebar paths with the file system, correcting `baseUrl`, `projectName`, `organizationName` in Docusaurus config, and integrating a GitHub Actions deploy workflow.
- **Git Tasks**: Automated committing of meaningful changes and pushing to the connected GitHub repository.
- **Module Structure**: Enforcement of a 4-module structure (ROS 2, Digital Twin, AI-Robot Brain, VLA Systems) each containing `overview.md`, `weekly-breakdown.md`, `deep-dive.md`, `practical-lab.md`, `simulation.md`, and `assignment.md`.

## Consequences

### Positive

- **Enhanced Consistency**: Automated validation and auto-fix mechanisms ensure Docusaurus configuration and content structure remain consistent and error-free.
- **Streamlined Workflow**: Git automation reduces manual overhead for committing and deploying changes.
- **Improved Maintainability**: Standardized module structure facilitates content organization, navigation, and future expansion.
- **Faster Feedback Loop**: Continuous validation provides immediate feedback on structural and configuration issues.
- **Clear Content Roadmap**: The defined module structure provides a clear roadmap for content creation and development.

### Negative

- **Initial Setup Complexity**: Implementing comprehensive CI/CD pipelines and auto-fix scripts requires upfront development effort.
- **Dependency on Automation**: Reliance on automated scripts means potential issues in scripts could block workflows if not properly managed.
- **Learning Curve**: New contributors may need to understand the CI/CD and auto-fix mechanisms.
- **Potential for Over-Automation**: Overly aggressive auto-fix features could lead to unintended changes if not carefully designed and tested.

## Alternatives Considered

- **Manual Validation and Deployment**: Relying on manual checks for configuration and structure, and manual deployment processes.
    - **Why rejected**: Prone to human error, slow feedback loop, inconsistent application of standards, increased operational burden.
- **Less Structured Content Organization**: Allowing more flexible or ad-hoc content placement without a strict module and file requirement.
    - **Why rejected**: Leads to disorganization, difficult navigation, inconsistency across modules, and increased cognitive load for contributors.
- **Alternative CI/CD Platforms**: Using other CI/CD solutions (e.g., GitLab CI, Jenkins) instead of GitHub Actions.
    - **Why rejected**: GitHub Actions is tightly integrated with GitHub, providing a more seamless experience for GitHub-hosted projects and leveraging existing platform features.

## References

- Feature Spec: null
- Implementation Plan: specs/001-book-spec/plan.md
- Related ADRs: null
- Evaluator Evidence: history/prompts/001-book-spec/001-continuous-project-sync-github-auto-update-plan.plan.prompt.md
