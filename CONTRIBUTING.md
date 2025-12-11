# Contribution Guidelines

Thank you for your interest in contributing to the Physical AI & Humanoid Robotics Course! This document outlines the guidelines for contributing to this textbook project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Content Guidelines](#content-guidelines)
4. [Style Guide](#style-guide)
5. [Technical Guidelines](#technical-guidelines)
6. [Pull Request Process](#pull-request-process)

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for everyone.

## How to Contribute

### Reporting Issues
- Check existing issues before filing a new one
- Provide detailed descriptions of problems
- Include steps to reproduce when possible
- Use appropriate issue labels

### Improving Content
- Enhance existing chapters with better explanations
- Add examples, diagrams, or exercises
- Fix errors or inaccuracies
- Improve clarity and readability

### Adding New Content
- Discuss significant additions in an issue first
- Follow the existing chapter structure
- Ensure content aligns with learning objectives
- Include appropriate examples and exercises

## Content Guidelines

### Educational Standards
- Content must be suitable for undergraduate/early-graduate learners
- Maintain technical accuracy using authoritative sources
- Provide intuitive explanations for complex concepts
- Include hands-on practical examples

### Structure
- Follow the formal textbook structure with sections and subsections
- Include learning objectives at the beginning of each chapter
- Add summaries at the end of each chapter
- Include exercises to reinforce concepts

### Quality
- Ensure all claims are traceable to credible sources
- Use APA 7th edition citation format
- Maintain zero plagiarism standards
- Verify mathematical models and equations for accuracy

## Style Guide

### Writing Tone
- Use formal technical language appropriate for academic content
- Be concise but comprehensive
- Explain jargon and technical terms when first introduced
- Write in active voice where possible

### Formatting
- Use Markdown for all content
- Follow Docusaurus-specific formatting conventions
- Include proper frontmatter in each chapter file:
  ```
  ---
  sidebar_label: "Label for sidebar"
  title: "Chapter Title"
  ---
  ```
- Use consistent terminology throughout the textbook

## Technical Guidelines

### File Structure
- Place new chapters in the appropriate module directory
- Maintain consistency with existing file naming conventions
- Ensure all links and references are valid
- Test changes locally before submitting

### Code Examples
- Ensure all code examples are working and well-commented
- Use Python for robotics/ROS examples
- Use TypeScript/JavaScript for frontend examples
- Include relevant libraries and versions

### Diagrams and Images
- Use Mermaid for diagrams when possible
- Store images in `static/img/` directory
- Provide alt text for all images
- Ensure diagrams are clear and support the text

## Pull Request Process

1. **Fork the repository** - Click the 'Fork' button in the upper right corner

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the guidelines above

4. **Test your changes**:
   ```bash
   npm start
   ```
   Verify that the site builds and displays your changes correctly

5. **Commit your changes** using conventional commit format:
   ```bash
   git add .
   git commit -m "docs(module-1): add explanation on ROS 2 communication patterns"
   ```

6. **Push to the branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** to the `main` branch

### PR Requirements
- Follow the PR template when available
- Describe the changes and why they were made
- Link to any related issues
- Ensure all tests pass
- Update documentation if needed

## Review Process

All contributions will be reviewed by maintainers. During the review:
- Feedback may be requested for clarifications or changes
- Technical accuracy will be verified
- Consistency with textbook style will be checked
- Learning objectives alignment will be confirmed

## Getting Help

If you need help with your contribution:
- Open an issue describing your question or problem
- Contact maintainers directly if needed
- Check existing documentation and code examples

## Recognition

Contributors will be acknowledged in the project documentation. Significant contributions may be highlighted in release notes.

---

Thank you for helping improve this textbook for the robotics community!