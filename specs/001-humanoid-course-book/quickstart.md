# Quickstart Guide: Humanoid Robotics Course Book Development

## Prerequisites

Before starting development of the humanoid robotics course content, ensure you have:

- Node.js version 18 or higher installed
- npm or yarn package manager
- Git for version control
- A GitHub account for deployment
- Text editor or IDE for Markdown editing
- Understanding of robotics concepts (for content accuracy)

## Setup

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd humanoid-robotics-course
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```
   or
   ```bash
   yarn install
   ```

3. **Start the Docusaurus development server**
   ```bash
   npm run start
   ```
   or
   ```bash
   yarn start
   ```

4. **Open your browser** to `http://localhost:3000` to view the site

## Content Creation Workflow

1. **Navigate to the docs directory**:
   ```bash
   cd docs
   ```

2. **Create content following the required structure**:
   - Each module should be in its own subdirectory
   - Each module must contain 7 document types:
     - `overview.md`
     - `weekly-breakdown.md`
     - `deep-dive.md`
     - `practical-lab.md`
     - `simulation.md`
     - `assignment.md`
     - `quiz.md`

3. **Follow the Docusaurus Markdown format** with proper headings for RAG compatibility:
   ```markdown
   # Module Title
   
   ## Section Heading
   
   Content goes here...
   ```

4. **Add proper frontmatter** to each document:
   ```markdown
   ---
   title: Document Title
   sidebar_position: 1
   description: Brief description of the document content
   ---
   ```

## Creating Module Content

For each module (1-4) and the capstone project:

1. **Create overview document**: High-level summary of the module
2. **Create weekly breakdown**: Detailed schedule with weekly topics
3. **Create deep dive**: Technical details and in-depth exploration
4. **Create practical lab**: Hands-on exercises and implementation tasks
5. **Create simulation document**: Instructions for simulation environments
6. **Create assignment**: Assessment tasks for students
7. **Create quiz**: Knowledge-check questions and answers

## Simulation Environment Setup

When creating simulation content, ensure instructions include:

- Installation requirements for ROS2, Gazebo, Isaac Sim, or Webots
- Configuration steps
- Troubleshooting tips
- Expected outcomes

## Content Quality Standards

- Maintain academic tone and professional technical writing
- Use accurate robotics terminology
- Include proper citations in APA format
- Ensure content aligns with specified focus areas
- Follow deterministic heading hierarchy for RAG indexing
- No pseudo code or video coding references

## Building and Deployment

1. **Build the static site**:
   ```bash
   npm run build
   ```

2. **Preview the build locally**:
   ```bash
   npm run serve
   ```

3. **Deploy to GitHub Pages** (if configured):
   The site will automatically deploy via GitHub Actions when changes are pushed to the main branch

## RAG Readiness Checklist

Before finalizing content, verify:

- [ ] Consistent heading hierarchy (H1, H2, H3, etc.)
- [ ] No large blocks of unstructured text
- [ ] Properly formatted code blocks
- [ ] Descriptive headings that make sense out of context
- [ ] Cross-references to other sections where appropriate