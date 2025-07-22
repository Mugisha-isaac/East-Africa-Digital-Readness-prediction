#!/bin/bash

# Clean deployment preparation script for Render
echo "🚀 Preparing East Africa Digital Readiness API for deployment..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the API directory."
    exit 1
fi

# Check required files
echo "📋 Checking required files..."
required_files=("main.py" "requirements.txt" "best_model.pkl" "scaler.pkl" "encoders.pkl")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found."
        exit 1
    else
        echo "✅ $file"
    fi
done

# Clean up unnecessary files
echo "🧹 Cleaning up..."
rm -f *_old.py *.bak
rm -rf __pycache__/

echo "✅ API ready for deployment!"
echo ""
echo "� Deploy to Render:"
echo "1. git add . && git commit -m 'Deploy clean API' && git push"
echo "2. Create Web Service on render.com"
echo "3. Build: pip install -r requirements.txt"
echo "4. Start: python main.py"
