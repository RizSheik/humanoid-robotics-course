#!/usr/bin/env node

// Cross-platform deploy-update script for Docusaurus
// Handles: GitHub → Fix → Deploy → Push Back

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

// Function to run a command and handle errors
function runCommand(cmd, description) {
    console.log(`[INFO] [${new Date().toISOString()}] RUNNING: ${description}`);
    console.log(`Executing: ${cmd}`);
    
    try {
        const result = execSync(cmd, { stdio: 'inherit' });
        return result;
    } catch (error) {
        console.error(`[ERROR] [${new Date().toISOString()}] Command failed: ${cmd}`);
        console.error(error.message);
        process.exit(1);
    }
}

// Function to check if a command exists
function commandExists(command) {
    try {
        execSync(`which ${command}`, { stdio: 'pipe' });
        return true;
    } catch (error) {
        return false;
    }
}

// Main deployment function
function deployUpdate() {
    console.log('🤖 Starting automated DevOps + Deployment assistant...');

    // Step 1: Pull latest changes
    console.log('🔄 Pulling latest changes...');
    runCommand('git pull origin main', 'Pulling latest changes');

    // Step 2: Analyze and fix issues
    console.log('🔍 Analyzing project for issues...');

    // Check if Node.js is installed (we're already running in Node.js, so it's available)

    // Check if npm is installed
    if (!commandExists('npm')) {
        console.error('❌ npm not found. Please install npm before running this script.');
        process.exit(1);
    }

    // Check if Vercel CLI is installed
    if (!commandExists('vercel')) {
        console.log('📦 Installing Vercel CLI...');
        runCommand('npm install -g vercel', 'Installing Vercel CLI');
    }

    // Step 3: Fix common Docusaurus issues
    console.log('🔧 Fixing common Docusaurus issues...');

    // Verify required files exist
    const requiredFiles = [
        'package.json',
        'docusaurus.config.js',
        'sidebars.ts'
    ];

    for (const file of requiredFiles) {
        if (!fs.existsSync(file)) {
            console.error(`❌ Required file missing: ${file}`);
            process.exit(1);
        }
    }

    // Check and fix image paths
    const imagePathsToVerify = [
        'static/img',
        'static/img/hero',
        'static/img/module',
        'static/img/book'
    ];

    for (const path of imagePathsToVerify) {
        if (!fs.existsSync(path)) {
            console.log(`📁 Creating missing directory: ${path}`);
            fs.mkdirSync(path, { recursive: true });
        }
    }

    // Check for broken links in Docusaurus config
    let configContent = fs.readFileSync('docusaurus.config.js', 'utf8');
    if (configContent.includes('your-organization')) {
        console.log('🔧 Fixing Docusaurus config organization name...');
        configContent = configContent.replace(/your-organization/g, 'RizSheik');
        fs.writeFileSync('docusaurus.config.js', configContent);
    }

    // Step 4: Install dependencies
    console.log('📦 Installing project dependencies...');
    runCommand('npm install', 'Installing dependencies');

    // Step 5: Run build to detect issues
    console.log('⚙️ Building project to detect issues...');
    try {
        runCommand('npm run build', 'Building project');
        console.log('✅ Build successful!');
    } catch (error) {
        console.log('⚠️ Build failed, attempting fixes...');
        // Try to clear cache and rebuild
        runCommand('npx docusaurus clear', 'Clearing Docusaurus cache');
        runCommand('npm run build', 'Re-building project after cache clear');
    }

    // Step 6: Check sidebar structure
    console.log('📋 Checking sidebar structure...');
    const sidebarContent = fs.readFileSync('sidebars.ts', 'utf8');
    if (sidebarContent.includes('module-')) {
        console.log('✅ Sidebar structure looks good');
    } else {
        console.log('🔧 Sidebar may need verification');
    }

    // Step 7: Deploy to Vercel
    console.log('🚀 Deploying to Vercel...');
    
    const vercelToken = process.env.VERCEL_TOKEN || process.env.VERCEL_TOKEN_ENV;
    let vercelDeployCmd = 'vercel --prod';
    
    if (vercelToken) {
        vercelDeployCmd = `vercel --token=${vercelToken} --prod`;
    } else {
        console.log('⚠️ Vercel token not found in environment. Please set VERCEL_TOKEN environment variable.');
    }

    // Run vercel deployment
    try {
        runCommand(vercelDeployCmd, 'Deploying to Vercel');
        console.log('✅ Deployment successful!');
    } catch (error) {
        console.error('❌ Deployment failed!');
        process.exit(1);
    }

    // Step 8: Push fixes back to GitHub
    console.log('🔄 Pushing updated code back to GitHub...');

    // Check for changes
    const gitStatus = execSync('git status --porcelain').toString().trim();
    if (gitStatus) {
        console.log('📝 Committing changes...');
        runCommand('git add .', 'Adding all changes');
        runCommand('git commit -m "Auto-fix: Docusaurus deployment fixes"', 'Committing changes');
        runCommand('git push origin main', 'Pushing to GitHub');
        console.log('✅ GitHub push successful!');
    } else {
        console.log('⏭️ No changes to commit');
    }

    console.log('🎉 Deployment workflow completed successfully!');
    console.log('🔗 Your site is now live on Vercel!');
}

// Execute the deployment
deployUpdate();