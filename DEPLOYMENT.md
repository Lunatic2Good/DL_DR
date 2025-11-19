# Deployment Guide

This guide will help you deploy the Diabetic Retinopathy Classification System to various platforms.

## Option 1: Render (Recommended - Free Tier Available)

### Steps:

1. **Sign up/Login to Render**

   - Go to https://render.com
   - Sign up with GitHub (recommended) or email

2. **Create New Web Service**

   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Or use "Public Git repository" and paste your repo URL

3. **Configure Service**

   - **Name**: `diabetic-retinopathy-classifier` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or paid if you need more resources)

4. **Environment Variables** (Optional)

   - `FLASK_ENV`: `production`
   - `PYTHON_VERSION`: `3.11.0`

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (first build may take 10-15 minutes)
   - Your app will be live at: `https://your-app-name.onrender.com`

### Notes:

- Free tier spins down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- Model files (.pth) need to be uploaded separately or use Git LFS

---

## Option 2: Railway

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy
6. Add environment variable: `PORT` (Railway sets this automatically)

---

## Option 3: PythonAnywhere

1. Sign up at https://www.pythonanywhere.com (free tier available)
2. Go to "Web" tab
3. Click "Add a new web app"
4. Choose Flask and Python 3.10
5. Upload your files via "Files" tab
6. Update WSGI configuration file to point to your app
7. Reload web app

---

## Option 4: Heroku (Paid - No Free Tier)

1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. Open: `heroku open`

---

## Important Notes for Deployment:

### Model Files

- Model files (.pth, .h5) are excluded from git (they're large)
- Options:
  1. Upload models manually after deployment
  2. Use Git LFS for large files
  3. Store models in cloud storage (S3, etc.) and download on startup

### Environment Variables

- Set `FLASK_ENV=production` for production
- Port is automatically set by hosting platform

### Static Files

- Static files (HTML, CSS, JS) are served by Flask
- For better performance, consider using a CDN

### Build Time

- First build may take 10-20 minutes due to PyTorch installation
- Subsequent builds are faster

---

## Troubleshooting

### Build Fails

- Check build logs for errors
- Ensure all dependencies are in requirements.txt
- Verify Python version compatibility

### App Crashes

- Check application logs
- Verify model files are accessible
- Check environment variables

### Slow Performance

- Free tiers have limited resources
- Consider upgrading to paid tier
- Optimize model loading (lazy loading)

---

## Quick Deploy Commands (if using Git)

```bash
# Commit deployment files
git add Procfile runtime.txt requirements.txt render.yaml
git commit -m "Add deployment configuration"
git push origin main
```

Then follow the platform-specific steps above.
