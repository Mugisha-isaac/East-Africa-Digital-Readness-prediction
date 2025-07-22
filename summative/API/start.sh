#!/bin/bash

# East Africa Youth Digital Readiness API Startup Script
# =====================================================

echo "🚀 Starting East Africa Youth Digital Readiness API..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if model files exist
echo "🔍 Checking for model files..."
if [ ! -f "../../best_model.pkl" ]; then
    echo "❌ Model file not found: ../../best_model.pkl"
    echo "Please ensure you have run the training script first."
    exit 1
fi

if [ ! -f "../../scaler.pkl" ]; then
    echo "❌ Scaler file not found: ../../scaler.pkl"
    exit 1
fi

if [ ! -f "../../encoders.pkl" ]; then
    echo "❌ Encoders file not found: ../../encoders.pkl"
    exit 1
fi

if [ ! -f "../../model_metadata.json" ]; then
    echo "❌ Metadata file not found: ../../model_metadata.json"
    exit 1
fi

echo "✅ All model files found!"

# Start the API
echo "🌟 Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "Documentation available at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"
echo ""

python3 main.py
