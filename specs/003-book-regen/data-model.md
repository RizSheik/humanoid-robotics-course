# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document defines the data model for the Physical AI & Humanoid Robotics educational textbook. Since this is a documentation project, the "data model" refers to the organizational structure, content entities, and metadata that define the educational material.

## Key Entities

### Course Module
Represents a major section of the educational content (Module 1-4, Capstone), containing chapters and supplementary materials that cover specific aspects of physical AI and humanoid robotics.

**Fields:**
- moduleId: String - Unique identifier for the module (e.g., "module-1", "capstone")
- title: String - Display title of the module
- description: String - Brief overview of the module content
- chapters: Array<Chapter> - Collection of chapters within the module
- prerequisites: Array<String> - Skills/knowledge required before starting the module
- learningObjectives: Array<String> - Specific learning outcomes
- estimatedDuration: Number - Estimated time to complete the module (hours)
- difficultyLevel: Enum - Beginner, Intermediate, Advanced

### Chapter
Individual learning units within each module, containing explanations, diagrams, tables, and real robotics examples.

**Fields:**
- chapterId: String - Unique identifier within the module (e.g., "ch-1", "ch-2")
- title: String - Chapter title
- content: String - Markdown-formatted chapter content
- learningObjectives: Array<String> - Specific learning outcomes for the chapter
- prerequisites: Array<String> - Skills/knowledge required before starting the chapter
- keyTopics: Array<String> - Main subjects covered in the chapter
- diagrams: Array<DiagramReference> - References to diagrams used in the chapter
- examples: Array<Example> - Code examples and use cases
- durationEstimate: Number - Estimated time to study the chapter (minutes)
- difficultyLevel: Enum - Beginner, Intermediate, Advanced

### Supplementary Document
Supporting educational materials including overview, weekly-breakdown, deep-dive, practical-lab, simulation, assignment, and quiz documents.

**Fields:**
- docType: Enum - "overview", "weekly-breakdown", "deep-dive", "practical-lab", "simulation", "assignment", "quiz"
- title: String - Document title
- content: String - Markdown-formatted document content
- moduleId: String - Associated module
- relatedChapters: Array<String> - Chapters related to this document
- estimatedTime: Number - Time required to engage with the material (minutes)

### Diagram Reference
Visual elements referenced from the static/img directory that support the educational content.

**Fields:**
- diagramId: String - Unique identifier
- altText: String - Alternative text for accessibility
- caption: String - Description of the diagram
- filePath: String - Path to the image file (relative to /static/img/)
- dimensions: Object - Width and height information
- usageNotes: String - How the diagram is used in the content

### Example
Code examples and practical demonstrations included in chapters and lab documents.

**Fields:**
- exampleId: String - Unique identifier
- title: String - Short description of the example
- language: Enum - Programming language (python, c++, ros2, etc.)
- code: String - Actual code content
- description: String - Explanation of what the code does
- relatedConcepts: Array<String> - Concepts illustrated by the example
- prerequisites: Array<String> - Requirements to run the example
- output: String - Expected output or behavior

### Navigation Item
The organized sidebar and menu system that allows users to browse the educational content in a logical sequence.

**Fields:**
- itemId: String - Unique identifier for the navigation item
- title: String - Text displayed in the navigation
- path: String - URL path for the item
- children: Array<NavigationItem> - Sub-items if applicable
- position: Number - Ordering index for proper sequence
- isVisible: Boolean - Whether the item appears in navigation

## Relationships

```
Course Module
    1 -> * Chapter
    1 -> * Supplementary Document
    
Chapter
    * -> * Diagram Reference
    * -> * Example

Course Module
    * -> * Navigation Item (via its chapters and documents)
    
Supplementary Document
    * -> * Navigation Item
```

## Content Validation Rules

### Module-Level Requirements
- Each module must have exactly 4 chapters (chapter-1.md through chapter-4.md)
- Each module must have 7 supplementary documents (overview.md, weekly-breakdown.md, deep-dive.md, practical-lab.md, simulation.md, assignment.md, quiz.md)
- Module titles must match the specified names in the feature requirements
- Learning objectives must align with the module's subject matter
- Prerequisites must be clearly stated and logically coherent

### Chapter-Level Requirements
- Each chapter must contain at least 1,000 words of substantive content
- Chapters must include explanations, diagrams, tables, and real robotics examples
- Difficulty level assessment must be consistent with the material covered
- Duration estimates must reflect realistic reading/understanding time
- Related topics must be accurately represented

### Document-Level Requirements
- All documents must follow the specified file naming convention
- No empty files are permitted
- Image references must point to existing files in the /static/img/ directory
- Content must meet academic standards for university-level coursework
- All content must be original and plagiarism-free

### Diagram Reference Requirements
- File paths must be valid and point to existing image files
- Alt text and captions must be descriptive and accessible
- Dimensions should be appropriate for educational contexts
- Usage notes must clearly explain the diagram's relevance

### Navigation Requirements
- Navigation structure must follow the specified order: Book Introduction, Module 1-4, Capstone, Appendices
- All content must be accessible through the navigation structure
- Links must not result in 404 errors
- Hierarchical relationships must be logical and intuitive

## State Transitions

For this documentation system, the primary "state transition" occurs during content development:

```
Draft → Reviewed → Approved → Published
```

Where:
- Draft: Content is being created with minimal validation
- Reviewed: Content has been checked for technical accuracy and pedagogical clarity
- Approved: Content meets all constitutional requirements
- Published: Content is deployed to the live documentation site