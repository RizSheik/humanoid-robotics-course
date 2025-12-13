# Physical AI & Humanoid Robotics Course

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator for the Physical AI & Humanoid Robotics textbook.

## Installation

```bash
npm install
```

## Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

### GitHub Pages
Deployment to GitHub Pages is done automatically via GitHub Actions when changes are pushed to the `main` branch.

### Vercel
The site is also deployed to Vercel. To deploy manually:

```bash
npm install -g vercel
vercel --prod
```

### Automated Deployment Script
For automatic deployment with fixes, run:

```bash
npm run deploy-update
```

This script will:
- Pull latest changes from GitHub
- Fix common Docusaurus issues
- Build the project
- Deploy to Vercel
- Push any fixes back to GitHub

## Contributing

This course is maintained using Spec-Driven Development (SDD) methodology. All contributions should follow the established patterns in the `specs/` directory.

