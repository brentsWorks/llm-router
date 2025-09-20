# LLM Router - Railway Deployment Guide

## 🚀 Quick Deploy to Railway

### Prerequisites
- Railway account (free tier available)
- GitHub repository with your code
- API keys configured in your `.env` file

### Step 1: Deploy Backend Service

1. **Go to [Railway.app](https://railway.app)**
2. **Click "New Project" → "Deploy from GitHub repo"**
3. **Select your repository**
4. **Railway will detect `railway.json` and use `Dockerfile.backend`**

5. **Set Environment Variables:**
   ```bash
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=llm-router
   OPENROUTER_API_KEY=your_openrouter_key
   GEMINI_API_KEY=your_gemini_key
   CORS_ORIGINS=*
   LOG_LEVEL=INFO
   TOKENIZERS_PARALLELISM=false
   ```

6. **Deploy** - Railway will build and deploy your backend
7. **Note the backend URL** (e.g., `https://backend-production-abc123.up.railway.app`)

### Step 2: Deploy Frontend Service

1. **In the same Railway project, click "New Service" → "GitHub Repo"**
2. **Select the same repository**
3. **Railway will detect `frontend/railway.json` and use `Dockerfile.frontend`**

4. **Set Environment Variable:**
   ```bash
   VITE_API_URL=https://your-backend-url.up.railway.app
   ```

5. **Deploy** - Railway will build and deploy your frontend
6. **Note the frontend URL** (e.g., `https://frontend-production-def456.up.railway.app`)

### Step 3: Update CORS Settings

1. **Go back to your backend service**
2. **Update the `CORS_ORIGINS` variable:**
   ```bash
   CORS_ORIGINS=https://frontend-production-def456.up.railway.app
   ```
3. **Redeploy the backend** (Railway will auto-redeploy)

### Step 4: Test Your Deployment

1. **Visit your frontend URL**
2. **Try routing a prompt** (e.g., "Write a Python function")
3. **Check that it connects to the backend and executes**

## 🔧 Troubleshooting

### Common Issues:

**CORS Errors:**
- Ensure `CORS_ORIGINS` includes your frontend URL
- Check that both services are deployed and running

**API Connection Errors:**
- Verify `VITE_API_URL` is set correctly in frontend
- Check backend logs for errors

**Build Failures:**
- Check Railway build logs for specific errors
- Ensure all required files are in your repository

### Health Checks:
- **Backend:** `https://your-backend-url.up.railway.app/health`
- **Frontend:** `https://your-frontend-url.up.railway.app/health`

## 📊 Monitoring

- **Railway Dashboard:** View logs, metrics, and deployment status
- **Backend Metrics:** `https://your-backend-url.up.railway.app/metrics`
- **Error Monitoring:** `https://your-backend-url.up.railway.app/monitoring/errors`

## 🎉 You're Live!

Your LLM Router should now be accessible via your Railway URLs with:
- ✅ Intelligent model selection
- ✅ Real LLM execution via OpenRouter
- ✅ Responsive React frontend
- ✅ Automatic HTTPS and scaling

---

**Need Help?** Check Railway's documentation or create an issue in your repository.
