# Quickstart Guide: Humanoid Robotics Course

This guide provides a quick overview to get started with the Humanoid Robotics Course project, built using Docusaurus.

## 1. Project Setup

To set up the project locally, ensure you have Node.js (version 18 or higher recommended) and npm installed. Then, clone the repository and install dependencies:

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
