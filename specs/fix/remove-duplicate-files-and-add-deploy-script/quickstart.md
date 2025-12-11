# Quickstart Guide: Humanoid Robotics Course

This guide provides a quick overview to get started with the Humanoid Robotics Course project, built using Docusaurus.

## 1. Project Setup

To set up the project locally, ensure you have Node.js (version 18 or higher) and npm installed. Then, clone the repository and install dependencies:

```bash
git clone https://github.com/RizSheik/humanoid-robotics-course.git
cd humanoid-robotics-course
npm install
```

## 2. Running the Development Server

To start the local development server with hot-reloading:

```bash
npm run start
```

This will open the site in your browser at `http://localhost:3000` (or another available port).

## 3. Building the Static Site

To build the static production-ready version of the site:

```bash
npm run build
```

The built files will be located in the `build/` directory.

## 4. Authoring Content

- **Chapter Files**: Markdown files (`.md`) are located in `docs/` within their respective module folders (e.g., `docs/module-1-foundational/chapter-1-intro.md`).
- **Module Structure**: Each module folder contains a `category.json` file that defines the module's label, title, description, and slug for Docusaurus.
- **Sidebar**: The `sidebars.ts` file defines the navigation structure of the documentation.

## 5. Deployment (GitHub Pages)

The project is configured for continuous deployment to GitHub Pages via GitHub Actions. Pushing to the `main` branch will trigger the deployment workflow.

For more details, refer to `docusaurus.config.js` and the `.github/workflows/` directory.

## 6. Adding New Modules

To add a new module to the textbook:

1. Create a new directory in `docs/` following the naming convention `module-{number}-{topic}`
2. Add a `category.json` file in the new module directory
3. Create chapter files in the module directory with appropriate content
4. Update `sidebars.ts` to include the new module and its chapters
5. Run `npm run build` to verify the changes

## 7. Content Guidelines

- All chapters should follow formal textbook structure with sections, subsections, and exercises
- Use appropriate frontmatter in Markdown files (title, sidebar_label, etc.)
- Include diagrams where appropriate (Mermaid, draw.io, or local assets)
- Ensure all internal links are valid
- Follow the pedagogical clarity principles defined in the project constitution