#!/bin/bash

# Automated DevOps + Deployment Script for Docusaurus Project
# This script handles: GitHub → Fix → Deploy → Push Back

# Function to create log entry
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# Function to run a command and handle errors
run_command() {
    local cmd="$1"
    local description="$2"
    
    log_message "RUNNING: $description"
    echo "Executing: $cmd"
    
    if ! eval "$cmd"; then
        log_error "Command failed: $cmd"
        exit 1
    fi
}

# Step 1: Pull latest changes
echo "🔄 Pulling latest changes..."
run_command "git pull origin main" "Pulling latest changes"

# Step 2: Analyze and fix issues
echo "🔍 Analyzing project for issues..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    log_error "Node.js not found. Please install Node.js before running this script."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    log_error "npm not found. Please install npm before running this script."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    run_command "npm install -g vercel" "Installing Vercel CLI"
fi

# Step 3: Fix common Docusaurus issues
echo "🔧 Fixing common Docusaurus issues..."

# Verify required files exist
for file in package.json docusaurus.config.js sidebars.ts; do
    if [[ ! -f "$file" ]]; then
        log_error "Required file missing: $file"
        exit 1
    fi
done

# Check and fix image paths
for path in static/img static/img/hero static/img/module static/img/book; do
    if [[ ! -d "$path" ]]; then
        echo "📁 Creating missing directory: $path"
        mkdir -p "$path"
    fi
done

# Check for broken links in Docusaurus config
if grep -q "your-organization" docusaurus.config.js; then
    echo "🔧 Fixing Docusaurus config organization name..."
    sed -i.bak 's/your-organization/RizSheik/g' docusaurus.config.js
    rm docusaurus.config.js.bak
fi

# Step 4: Install dependencies
echo "📦 Installing project dependencies..."
run_command "npm install" "Installing dependencies"

# Step 5: Run build to detect issues
echo "⚙️ Building project to detect issues..."
if ! npm run build; then
    echo "⚠️ Build failed, attempting fixes..."
    # Try to clear cache and rebuild
    run_command "npx docusaurus clear" "Clearing Docusaurus cache"
    run_command "npm run build" "Re-building project after cache clear"
fi

# Step 6: Check sidebar structure
echo "📋 Checking sidebar structure..."
if grep -q "module-[0-9]" sidebars.ts; then
    echo "✅ Sidebar structure looks good"
else
    echo "🔧 Sidebar may need verification"
fi

# Step 7: Deploy to Vercel
echo "🚀 Deploying to Vercel..."

VERCEL_TOKEN=${VERCEL_TOKEN:-$VERCEL_TOKEN_ENV}
if [ -z "$VERCEL_TOKEN" ]; then
    echo "⚠️ Vercel token not found in environment. Please set VERCEL_TOKEN environment variable."
    vercel_deploy_cmd="vercel --prod"
else
    vercel_deploy_cmd="vercel --token=$VERCEL_TOKEN --prod"
fi

# Run vercel deployment
if eval "$vercel_deploy_cmd"; then
    echo "✅ Deployment successful!"
else
    log_error "Deployment failed!"
    exit 1
fi

# Step 8: Push fixes back to GitHub
echo "🔄 Pushing updated code back to GitHub..."

# Check for changes
if [[ -n $(git status --porcelain) ]]; then
    echo "📝 Committing changes..."
    run_command "git add ." "Adding all changes"
    run_command "git commit -m 'Auto-fix: Docusaurus deployment fixes'" "Committing changes"
    run_command "git push origin main" "Pushing to GitHub"
    echo "✅ GitHub push successful!"
else
    echo "⏭️ No changes to commit"
fi

echo "🎉 Deployment workflow completed successfully!"
echo "🔗 Your site is now live on Vercel!"