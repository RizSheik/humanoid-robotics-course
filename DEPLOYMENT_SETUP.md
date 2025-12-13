# Deployment Setup

## Vercel Deployment

This project is configured for deployment on Vercel. Follow these steps to set up automatic deployments:

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

This will open a browser window where you can log in to your Vercel account.

### 3. Link Your Project

```bash
vercel
```

Follow the prompts to link your project to a Vercel project. The CLI will detect this is a Docusaurus project and configure appropriate build settings automatically.

### 4. Configure Project Settings

In your Vercel dashboard, make sure the following settings are configured:

- **Framework**: Docusaurus
- **Build Command**: `npm run build`
- **Output Directory**: `build`
- **Root Directory**: `/` (root of your repository)

### 5. Environment Variables (Optional)

If you want to use automatic deployments with the `deploy-update` script, set the following environment variable:

```bash
export VERCEL_TOKEN=your_vercel_token_here
```

Or add it to your system's environment variables.

### 6. Automatic Deployment Script

This project includes an automated deployment script that handles:

- Pulling latest changes from GitHub
- Fixing common Docusaurus issues
- Building the project
- Deploying to Vercel
- Pushing any fixes back to GitHub

To run the deployment:

```bash
npm run deploy-update
```

The script will:
1. Pull latest changes
2. Check for and fix common issues
3. Install dependencies
4. Build the project
5. Deploy to Vercel
6. Push any fixes back to GitHub

### 7. GitHub Integration (Recommended)

For automatic deployments on every commit, connect your GitHub repository to Vercel:

1. Go to your Vercel dashboard
2. Click "Add New..." and select "Project"
3. Import your GitHub repository
4. Vercel will automatically configure the build settings
5. Pushing to the main branch will trigger a new deployment

### 8. Local Testing

To test the site locally before deployment:

```bash
npm run start
```

This will start a local development server.

## Troubleshooting

- If builds fail due to broken links, the current configuration treats them as warnings rather than errors
- Ensure all dependencies are in `package.json`
- Make sure `docusaurus.config.js` is properly configured for your deployment domain
- If using a custom domain, configure it in the Vercel dashboard