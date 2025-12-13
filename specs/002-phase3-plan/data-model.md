# Data Model: Physical AI & Humanoid Robotics Book

## Entity: Course Module
**Description**: One of the four core modules of the textbook
**Fields**:
- moduleId: Unique identifier for the module
- title: Name of the module (e.g., "The Robotic Nervous System (ROS 2)")
- description: Brief overview of the module content
- chapters: Collection of 7 chapter types
- prerequisites: Previous modules or knowledge required
- learningOutcomes: List of measurable outcomes for the module

**Relationships**:
- Contains 7 Document Types (one-to-many)
- Connected to Capstone Project (many-to-one, optional)

## Entity: Document Type (Chapter)
**Description**: One of the seven required chapter types in each module
**Fields**:
- chapterId: Unique identifier for the chapter
- title: Title of the chapter (e.g., "overview", "practical-lab")
- content: The markdown content of the chapter
- module: Reference to the parent module
- learningOutcomes: List of specific learning outcomes for this chapter
- prerequisites: Knowledge required before reading this chapter
- relatedChapters: Cross-references to related chapters

**Validation Rules**:
- All chapters must have properly formatted markdown starting with H1
- Content must adhere to RAG-groundable format with consistent headings
- Each chapter must include at least one learning outcome

**State Transitions**:
- Draft → Review → Approved → Published

## Entity: Educational Content
**Description**: The actual text, diagrams, tables, and materials that teach about humanoid robotics
**Fields**:
- contentId: Unique identifier for the content block
- type: Type of content (text, diagram, table, code example, etc.)
- content: The actual content
- topic: The robotics/AI topic being covered
- difficulty: Level of difficulty (beginner, intermediate, advanced)
- citations: References to authoritative sources
- relatedImages: List of images associated with this content

**Validation Rules**:
- Content must reference authoritative sources (peer-reviewed papers, textbooks, etc.)
- Mathematical models must be correct and internally consistent
- All claims must be traceable to primary/credible sources

## Entity: Image Reference
**Description**: Embedded media using relative paths from the static/img/ folder
**Fields**:
- imageId: Unique identifier for the image
- altText: Alternative text for accessibility
- caption: Descriptive caption
- relativePath: Path relative to static/img folder (e.g., /img/module/filename.jpg)
- module: Module where the image is used
- chapter: Chapter where the image is embedded

**Validation Rules**:
- All images must have appropriate alt text for accessibility
- Paths must reference existing files in static/img folder
- Images must be relevant to the educational content

## Entity: Docusaurus Structure
**Description**: The hierarchical organization of content in Markdown files
**Fields**:
- slug: URL-friendly identifier
- sidebarLabel: Name as it appears in the sidebar navigation
- sidebarPosition: Position in the sidebar hierarchy
- parentCategory: The parent category if nested
- filePath: Path to the markdown file

**Validation Rules**:
- All content must follow Docusaurus conventions
- Sidebar order must follow Book Introduction → Modules → Capstone → Appendices
- All files must be properly linked in sidebars.ts

## Entity: Learning Outcome
**Description**: Specific, measurable objectives that define what students should understand
**Fields**:
- outcomeId: Unique identifier for the learning outcome
- description: Clear, measurable statement of what students will learn
- associatedChapter: Reference to the chapter where this outcome is taught
- assessmentMethod: How this outcome will be assessed (quiz, assignment, etc.)
- difficultyLevel: Complexity of the learning outcome

**Validation Rules**:
- All learning outcomes must be specific and measurable
- Outcomes must align with the content of the associated chapter

## Entity: Capstone Project
**Description**: The culminating educational experience focused on The Autonomous Humanoid
**Fields**:
- projectId: Unique identifier for the capstone project
- title: Title of the capstone project
- description: Overview of the capstone requirements
- objectives: Learning objectives specific to the capstone
- deliverables: What students need to submit
- chapters: Collection of 7 chapter types as with modules

**Relationships**:
- Connected to all 4 Course Modules (many-to-many)
- Contains 7 Document Types (one-to-many)

## Entity: Appendix Content
**Description**: Supplementary material covering hardware requirements, lab architecture, etc.
**Fields**:
- appendixId: Unique identifier for the appendix
- title: Title of the appendix
- content: The markdown content of the appendix
- type: Type of appendix (hardware-requirements, lab-architecture, cloud-vs-onprem)
- relatedModules: List of modules that reference this appendix

**Validation Rules**:
- Content must be supplementary to the main modules
- Must provide value to students and instructors
- Should include practical information applicable to course implementation