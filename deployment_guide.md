# 🚀 Deployment Guide: How to Publish Your App

Follow these steps to put your app on the internet for free using Streamlit Cloud.

## Step 1: Create a GitHub Account
If you haven't already, go to [GitHub.com](https://github.com/) and sign up for a free account.

## Step 2: Create a New Repository
1. Log in to GitHub.
2. Click the **+** icon in the top right -> **New repository**.
3. Name it `football-predictor` (or anything you like).
4. Make sure it is **Public**.
5. Click **Create repository**.

## Step 3: Upload Your Files
1. In your new repository, click the **uploading an existing file** link (or "Add file" -> "Upload files").
2. Drag and drop the following files from your `Solomon_app` folder:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   *(Do NOT upload the `.venv` folder or `__pycache__`)*
3. Wait for them to finish uploading.
4. In "Commit changes", type "Initial commit" and click **Commit changes**.

## Step 4: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Select your GitHub repository (`your-username/football-predictor`).
4. **Main file path**: `app.py`
5. Click **Deploy!**

## Done! 🎉
Streamlit will install your requirements and launch the app. In 1-2 minutes, you'll have a live URL (e.g., `https://football-predictor.streamlit.app`) to share with everyone!
