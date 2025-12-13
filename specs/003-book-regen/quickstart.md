# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Overview
This guide provides a quick walkthrough of the Physical AI & Humanoid Robotics educational textbook project. It covers the essential steps needed to set up, develop, and deploy the educational content.

## Prerequisites

Before working with this project, ensure you have:

- **Node.js** version 18 or higher
- **npm** or **yarn** package manager
- **Git** for version control
- Basic knowledge of **Markdown** syntax
- Understanding of **Docusaurus** concepts (optional but helpful)
- Knowledge of robotics and AI concepts (for content review)

## Project Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics-course
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Local Development Server
```bash
npm start
```

This will start the development server and open the documentation site in your default browser at http://localhost:3000.

## Project Structure

The book is organized as follows:

```
docs/
├── introduction.md                  # Book introduction page
├── module-1-the-robotic-nervous-system/
│   ├── chapter-1.md
│   ├── chapter-2.md
│   ├── chapter-3.md
│   ├── chapter-4.md
│   ├── overview.md
│   ├── weekly-breakdown.md
│   ├── deep-dive.md
│   ├── practical-lab.md
│   ├── simulation.md
│   ├── assignment.md
│   └── quiz.md
├── module-2-the-digital-twin/
│   ├── chapter-1.md
│   ├── chapter-2.md
│   ├── chapter-3.md
│   ├── chapter-4.md
│   ├── overview.md
│   ├── weekly-breakdown.md
│   ├── deep-dive.md
│   ├── practical-lab.md
│   ├── simulation.md
│   ├── assignment.md
│   └── quiz.md
├── module-3-the-ai-robot-brain/
│   ├── chapter-1.md
│   ├── chapter-2.md
│   ├── chapter-3.md
│   ├── chapter-4.md
│   ├── overview.md
│   ├── weekly-breakdown.md
│   ├── deep-dive.md
│   ├── practical-lab.md
│   ├── simulation.md
│   ├── assignment.md
│   └── quiz.md
├── module-4-vision-language-action-systems/
│   ├── chapter-1.md
│   ├── chapter-2.md
│   ├── chapter-3.md
│   ├── chapter-4.md
│   ├── overview.md
│   ├── weekly-breakdown.md
│   ├── deep-dive.md
│   ├── practical-lab.md
│   ├── simulation.md
│   ├── assignment.md
│   └── quiz.md
├── capstone-the-autonomous-humanoid/
│   ├── overview.md
│   ├── practical-lab.md
│   ├── simulation.md
│   ├── assignment.md
│   └── quiz.md
└── appendices/
    ├── hardware-requirements.md
    ├── lab-architecture.md
    └── cloud-vs-onprem.md

static/
└── img/                              # Static images referenced in content
```

## Adding Content

### Creating a New Chapter

1. Navigate to the appropriate module directory
2. Create a new Markdown file (e.g., `my-new-chapter.md`)
3. Add the following header to your file:
```markdown
---
title: My New Chapter
description: An overview of what this chapter covers
sidebar_position: 5  # Adjust position as needed
---

# My New Chapter

Your chapter content goes here...
```

4. Include diagrams using the `/static/img/` directory:
```markdown
![Diagram Description](/static/img/diagram-name.png)
```

### Adding Images

1. Place your images in the `static/img/` directory
2. Reference them in your Markdown files using the `/static/img/` prefix:
```markdown
![Alt text](/static/img/filename.png)
```

## Navigation and Sidebar

The sidebar navigation is configured in `sidebars.ts`. The structure follows this hierarchy:
- Book Introduction
- Module 1
- Module 2
- Module 3
- Module 4
- Capstone
- Appendices

Each module contains chapters and supplementary documents as specified.

## Development Workflow

### 1. Content Creation
- Add new content to the appropriate module directory
- Follow the pattern of 4 chapters + 7 supporting documents per module
- Ensure all content meets academic standards and constitutional requirements

### 2. Preview Changes
- Run `npm start` to view changes live
- Verify navigation works correctly
- Check that all links are functioning

### 3. Building the Site
```bash
npm run build
```
This creates a production-ready build in the `build/` directory.

### 4. Testing
```bash
npm run serve
```
This serves the built site locally for testing before deployment.

## Deployment

The site is set up for deployment to GitHub Pages. On pushing to the main branch, it should automatically build and deploy.

## Content Guidelines

### Writing Style
- Maintain technical accuracy while ensuring pedagogical clarity
- Write for undergraduate/graduate level learners
- Use consistent terminology throughout the book
- Include practical examples and hands-on activities

### Technical Requirements
- All content must be original (zero plagiarism)
- Follow APA 7th edition citation format when referencing external sources
- At least 50% of sources should be peer-reviewed robotics/AI publications
- Include safety and ethics considerations in robotics content

### Quality Assurance
- Verify all diagrams, code examples, and links work correctly
- Ensure content meets academic standards
- Test navigation and site structure
- Validate that the build process completes without errors