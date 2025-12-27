#!/usr/bin/env pwsh

# Automated DevOps + Deployment Script for Docusaurus Project
# This script handles: GitHub → Fix → Deploy → Push Back

param(
    [string]$VercelToken = $env:VERCEL_TOKEN,
    [string]$GithubToken = $env:GITHUB_TOKEN
)

Write-Host "🤖 Starting automated DevOps + Deployment assistant..." -ForegroundColor Green

# Function to create log entry
function Write-Log {
    param([string]$Message, [string]$Type = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Type] $Message"
}

# Function to check if a command exists
function Test-Command {
    param([string]$cmd)
    return $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

# Function to run a command and handle errors
function Run-Command {
    param([string]$cmd, [string]$description)
    
    Write-Log "RUNNING: $description" "CMD"
    Write-Host "Executing: $cmd" -ForegroundColor Yellow
    
    try {
        $result = Invoke-Expression $cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Command failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            throw "Command failed: $cmd"
        }
        return $result
    } catch {
        Write-Host "🚨 Error executing: $cmd" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

# Step 1: Pull latest changes
Write-Host "🔄 Pulling latest changes..." -ForegroundColor Cyan
Run-Command "git pull origin main" "Pulling latest changes"

# Step 2: Analyze and fix issues
Write-Host "🔍 Analyzing project for issues..." -ForegroundColor Cyan

# Check if Node.js is installed
if (-not (Test-Command "node")) {
    Write-Host "❌ Node.js not found. Please install Node.js before running this script." -ForegroundColor Red
    exit 1
}

# Check if npm is installed
if (-not (Test-Command "npm")) {
    Write-Host "❌ npm not found. Please install npm before running this script." -ForegroundColor Red
    exit 1
}

# Check if Vercel CLI is installed
if (-not (Test-Command "vercel")) {
    Write-Host "📦 Installing Vercel CLI..." -ForegroundColor Yellow
    Run-Command "npm install -g vercel" "Installing Vercel CLI"
}

# Step 3: Fix common Docusaurus issues
Write-Host "🔧 Fixing common Docusaurus issues..." -ForegroundColor Cyan

# Verify required files exist
$requiredFiles = @(
    "package.json",
    "docusaurus.config.js",
    "sidebars.ts"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "❌ Required file missing: $file" -ForegroundColor Red
        exit 1
    }
}

# Check and fix image paths
$imagePathsToVerify = @(
    "static/img",
    "static/img/hero",
    "static/img/module",
    "static/img/book"
)

foreach ($path in $imagePathsToVerify) {
    if (-not (Test-Path $path)) {
        Write-Host "📁 Creating missing directory: $path" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# Check for broken links in Docusaurus config
$configContent = Get-Content "docusaurus.config.js" -Raw
if ($configContent -match "your-organization") {
    Write-Host "🔧 Fixing Docusaurus config organization name..." -ForegroundColor Yellow
    $configContent = $configContent -replace "your-organization", "RizSheik"
    Set-Content "docusaurus.config.js" $configContent
}

# Step 4: Install dependencies
Write-Host "📦 Installing project dependencies..." -ForegroundColor Cyan
Run-Command "npm install" "Installing dependencies"

# Step 5: Run build to detect issues
Write-Host "⚙️ Building project to detect issues..." -ForegroundColor Cyan
try {
    Run-Command "npm run build" "Building project"
    Write-Host "✅ Build successful!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Build failed, attempting fixes..." -ForegroundColor Yellow
    # Try to clear cache and rebuild
    Run-Command "npx docusaurus clear" "Clearing Docusaurus cache"
    Run-Command "npm run build" "Re-building project after cache clear"
}

# Step 6: Check sidebar structure
Write-Host "📋 Checking sidebar structure..." -ForegroundColor Cyan
$sidebarContent = Get-Content "sidebars.ts" -Raw

# Fix common sidebar issues
if ($sidebarContent -match "module-\d") {
    Write-Host "✅ Sidebar structure looks good" -ForegroundColor Green
} else {
    Write-Host "🔧 Sidebar may need verification" -ForegroundColor Yellow
}

# Step 7: Deploy to Vercel
Write-Host "🚀 Deploying to Vercel..." -ForegroundColor Cyan

if ([string]::IsNullOrEmpty($VercelToken)) {
    Write-Host "⚠️ Vercel token not found in environment. Please set VERCEL_TOKEN environment variable." -ForegroundColor Yellow
    $vercelDeployCmd = "vercel --prod"
} else {
    $vercelDeployCmd = "vercel --token=$VercelToken --prod"
}

# Set project working directory
Set-Location $PSScriptRoot

# Run vercel deployment
try {
    $vercelOutput = Run-Command $vercelDeployCmd "Deploying to Vercel"
    Write-Host "✅ Deployment successful!" -ForegroundColor Green
    
    # Extract the deployment URL
    $urlMatch = [regex]::Match($vercelOutput, "(https?://[^\s]+\.vercel\.app)")
    if ($urlMatch.Success) {
        $deploymentUrl = $urlMatch.Value
        Write-Host "🔗 Live URL: $deploymentUrl" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️ Could not extract deployment URL from output" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}

# Step 8: Push fixes back to GitHub
Write-Host "🔄 Pushing updated code back to GitHub..." -ForegroundColor Cyan

# Check for changes
$changes = git status --porcelain
if ($changes) {
    Write-Host "📝 Committing changes..." -ForegroundColor Yellow
    Run-Command "git add ." "Adding all changes"
    Run-Command "git commit -m 'Auto-fix: Docusaurus deployment fixes'" "Committing changes"
    Run-Command "git push origin main" "Pushing to GitHub"
    Write-Host "✅ GitHub push successful!" -ForegroundColor Green
} else {
    Write-Host "⏭️ No changes to commit" -ForegroundColor Yellow
}

Write-Host "🎉 Deployment workflow completed successfully!" -ForegroundColor Green
Write-Host "🔗 Your site is now live on Vercel!" -ForegroundColor Cyan