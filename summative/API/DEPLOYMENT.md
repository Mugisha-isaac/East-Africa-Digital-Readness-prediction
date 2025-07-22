# 🚀 Render Docker Deployment Guide

This guide will help you deploy the East Africa Youth Digital Readiness API to Render using Docker.

## 📋 Prerequisites

1. **GitHub Account**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Docker Support**: Using Docker for reliable, consistent deployments

## 🛠️ Deployment Steps

### Step 1: Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy API with Docker to Render"
   git push origin main
   ```

### Step 2: Deploy on Render with Docker

1. **Go to Render Dashboard**:
   - Visit [render.com](https://render.com)
   - Sign in with your GitHub account

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing this API

3. **Configure the Service**:
   - **Name**: `east-africa-digital-readiness-api`
   - **Runtime**: `Docker`
   - **Root Directory**: `summative/API` (if API is in subdirectory)
   - **Docker Command**: `python main.py`
   - **Plan**: Free (for testing) or Starter (for production)

4. **Environment Variables** (automatically set):
   - `PORT`: Will be automatically set by Render

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for Docker build and deployment (10-15 minutes)

### Step 3: Verify Deployment

Once deployed, your API will be available at:
- **API Base URL**: `https://your-service-name.onrender.com`
- **Documentation**: `https://your-service-name.onrender.com/docs`
- **Health Check**: `https://your-service-name.onrender.com/health`

## 🐳 Docker Configuration

This deployment uses Docker with:

- **Base Image**: `python:3.11-slim`
- **Port**: 8000
- **Health Check**: Built-in `/health` endpoint monitoring
- **Security**: Non-root user execution
- **Optimization**: Multi-stage build with dependency caching

## 🔧 Configuration Files

Docker deployment includes:

- `Dockerfile`: Container configuration
- `docker-compose.yml`: Local testing setup
- `.dockerignore`: Build optimization
- `requirements.txt`: Python dependencies
- `render.yaml`: Render service configuration

4. **Set Environment Variables** (if needed):
   - `PORT`: Will be automatically set by Render
   - `PYTHON_VERSION`: 3.12.0

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)

### Step 3: Verify Deployment

Once deployed, your API will be available at:
- **API Base URL**: `https://your-service-name.onrender.com`
- **Documentation**: `https://your-service-name.onrender.com/docs`
- **Health Check**: `https://your-service-name.onrender.com/health`

## 🔧 Configuration Files

This deployment includes:

- `render.yaml`: Render service configuration
- `requirements.txt`: Python dependencies with versions
- `runtime.txt`: Python version specification
- `Procfile`: Alternative process file
- `.gitignore`: Git ignore rules

## 🧪 Testing Your Deployed API

### 1. Health Check
```bash
curl https://your-service-name.onrender.com/health
```

### 2. Model Info
```bash
curl https://your-service-name.onrender.com/model/info
```

### 3. Single Prediction
```bash
curl -X POST "https://your-service-name.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "location_type": 1,
       "household_size": 4,
       "age_of_respondent": 22,
       "gender_of_respondent": 1,
       "relationship_with_head": 2,
       "marital_status": 3,
       "job_type": 3
     }'
```

### 4. Multiple Users Prediction
```bash
curl -X POST "https://your-service-name.onrender.com/predict/users" \
     -H "Content-Type: application/json" \
     -d '{
       "users": [
         {
           "location_type": 1,
           "household_size": 4,
           "age_of_respondent": 22,
           "gender_of_respondent": 1,
           "relationship_with_head": 2,
           "marital_status": 3,
           "job_type": 3
         }
       ]
     }'
```

## 🌐 Update HTML Test Interface

After deployment, update your `test_interface.html`:

```javascript
// Change this line in the HTML file
const API_BASE = 'https://your-service-name.onrender.com';
```

## 📝 Important Notes

1. **Free Plan Limitations**:
   - Service goes to sleep after 15 minutes of inactivity
   - Takes 30-60 seconds to wake up on first request
   - 750 hours/month of runtime

2. **Model Files**:
   - Ensure all `.pkl` files are committed to your repository
   - Total repository size should be under 1GB

3. **Environment Variables**:
   - Render automatically sets `PORT`
   - Your app reads it via `os.getenv("PORT", 8000)`

4. **Logs**:
   - View logs in Render dashboard
   - Use for debugging deployment issues

## 🔄 Continuous Deployment

Render automatically redeploys when you push to your connected branch:

```bash
git add .
git commit -m "Update API"
git push origin main  # Triggers automatic deployment
```

## 🆘 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check `requirements.txt` format
   - Ensure Python version compatibility

2. **Service Won't Start**:
   - Check logs in Render dashboard
   - Verify `main.py` runs locally

3. **Model Loading Errors**:
   - Ensure `.pkl` files are in repository
   - Check file paths in `main.py`

4. **Memory Issues**:
   - Consider upgrading from Free plan
   - Optimize model loading

### Getting Help:

- Check Render documentation: [docs.render.com](https://docs.render.com)
- View service logs in Render dashboard
- Test locally first: `python main.py`

## 🎉 Success!

Your East Africa Youth Digital Readiness API is now live and accessible worldwide!

Access your API documentation at: `https://your-service-name.onrender.com/docs`
