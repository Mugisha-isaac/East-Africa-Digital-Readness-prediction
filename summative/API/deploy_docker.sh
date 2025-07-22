#!/bin/bash

# Clean Docker deployment script for Render
echo "🐳 Deploying East Africa Digital Readiness API with Docker..."

# Check required files
echo "📋 Checking deployment files..."
required_files=("main.py" "requirements.txt" "Dockerfile" "best_model.pkl" "scaler.pkl" "encoders.pkl")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        exit 1
    fi
    echo "✅ $file"
done

# Clean up
echo "🧹 Cleaning..."
rm -rf __pycache__/ *.bak *_old.py

echo ""
echo "🎯 Ready for Render deployment!"
echo ""
echo "� Steps:"
echo "1. git add . && git commit -m 'Clean Docker deployment' && git push"
echo "2. Render: Runtime = Docker, Root = summative/API"
echo "3. API will be live with Swagger docs at /docs"
