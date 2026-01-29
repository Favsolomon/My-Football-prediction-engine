
# Deployment Guide - Hosting on Vercel

Vercel is the best free platform for this architecture. It will host your Static Frontend and your Python Backend (as Serverless Functions) seamlessly.

## Prerequisites
1. A [GitHub](https://github.com) account.
2. A [Vercel](https://vercel.com) account (linked to GitHub).

## Step-by-Step Deployment

### 1. Push to GitHub
If your project isn't on GitHub yet, create a new repository and push your code:
```powershell
git init
git add .
git commit -m "Prepare for Vercel deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 2. Import to Vercel
1. Go to the [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **"Add New..."** -> **"Project"**.
3. Select your GitHub repository.
4. Vercel will automatically detect the `vercel.json` configuration.
5. Click **"Deploy"**.

## Project Structure for Vercel
I have reorganized your files to match Vercel's requirements:
- `api/main.py`: The entry point for backend functions.
- `index.html`: The main frontend file at the root.
- `vercel.json`: Handles routing between the UI and the API.

## Important Note on API Limits
> [!WARNING]
> If you are using free API keys (like the Odds API), remember that Vercel might spawn multiple instances. Monitor your usage to ensure you don't hit rate limits prematurely.

## Local Testing
You can still test locally using the new structure:
```powershell
.\.venv\Scripts\python -m uvicorn api.main:app --port 8080
```
