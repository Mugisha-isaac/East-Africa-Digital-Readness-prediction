#!/usr/bin/env python3
"""
East Africa Youth Digital Readiness Prediction API
==================================================

FastAPI application for predicting digital readiness of youth in East Africa.
Predicts: Phone Access + Bank Account + Education combined
Countries: Rwanda, Tanzania, Kenya, Uganda
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import json
import os
from typing import Dict, List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="East Africa Youth Digital Readiness API",
    description="Predict digital readiness (phone + bank + education) for youth job platform development",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model = None
scaler = None
encoders = None
metadata = None

# Model file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

def load_model_components():
    """Load model, scaler, encoders, and metadata"""
    global model, scaler, encoders, metadata
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✓ Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("✓ Scaler loaded successfully")
        else:
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        # Load encoders
        if os.path.exists(ENCODERS_PATH):
            encoders = joblib.load(ENCODERS_PATH)
            print("✓ Encoders loaded successfully")
        else:
            raise FileNotFoundError(f"Encoders file not found: {ENCODERS_PATH}")
        
        # Load metadata (optional)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            print("✓ Metadata loaded successfully")
            print(f"Model type: {metadata.get('model_type', 'Unknown')}")
            print(f"Model R² score: {metadata.get('r2_score', 'Unknown')}")
        else:
            print("ℹ No metadata file found")
            metadata = {"model_type": "Random Forest", "r2_score": 0.111}
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        raise e

# Pydantic models for request/response
class UserProfile(BaseModel):
    location_type: int = Field(..., ge=0, le=1, description="0=Rural, 1=Urban")
    household_size: int = Field(..., ge=1, le=20, description="Number of people in household")
    age_of_respondent: int = Field(..., ge=16, le=30, description="Age of respondent (youth focus)")
    gender_of_respondent: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    relationship_with_head: int = Field(..., ge=0, le=5, description="Relationship with household head")
    marital_status: int = Field(..., ge=0, le=4, description="Marital status code")
    job_type: int = Field(..., ge=0, le=15, description="Job/occupation type")
    
    @field_validator('age_of_respondent')
    @classmethod
    def validate_age(cls, v):
        if not 16 <= v <= 30:
            raise ValueError('Age must be between 16 and 30 for youth focus')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "location_type": 1,
                "household_size": 4,
                "age_of_respondent": 22,
                "gender_of_respondent": 1,
                "relationship_with_head": 2,
                "marital_status": 3,
                "job_type": 3
            }
        }
    }

class UsersInput(BaseModel):
    users: List[UserProfile] = Field(..., min_length=1, max_length=100)

class PredictionResponse(BaseModel):
    prediction: float
    digital_readiness_level: str
    confidence: str
    user_profile: Dict

class UsersPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict
    total_users: int

# Startup event
@app.on_event("startup")
async def startup_event():
    load_model_components()

# Helper functions
def preprocess_features(user_data: Dict) -> np.ndarray:
    """Preprocess user data for prediction"""
    try:
        # Convert to numpy array in the correct order
        features = np.array([[
            user_data['location_type'],
            user_data['household_size'],
            user_data['age_of_respondent'],
            user_data['gender_of_respondent'],
            user_data['relationship_with_head'],
            user_data['marital_status'],
            user_data['job_type']
        ]])
        
        # Apply scaling
        if scaler:
            features = scaler.transform(features)
        
        return features
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature preprocessing error: {str(e)}")

def interpret_prediction(prediction_value: float, user_data: Dict) -> Dict:
    """Interpret prediction result"""
    # Determine digital readiness level
    if prediction_value >= 2.5:
        level = "High Digital Readiness"
        confidence = "High"
    elif prediction_value >= 1.5:
        level = "Moderate Digital Readiness"
        confidence = "Medium"
    else:
        level = "Low Digital Readiness"
        confidence = "Low"
    
    return {
        "prediction": round(float(prediction_value), 3),
        "digital_readiness_level": level,
        "confidence": confidence,
        "user_profile": user_data
    }

# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "East Africa Youth Digital Readiness Prediction API",
        "description": "Predicts digital readiness (phone + bank + education) for youth job platform development",
        "version": "1.0.0",
        "endpoints": {
            "users_prediction": "/predict/users",
            "single_prediction": "/predict",
            "health": "/health",
            "model_info": "/model/info",
            "documentation": "/docs"
        },
        "countries": ["Rwanda", "Tanzania", "Kenya", "Uganda"],
        "target_audience": "Youth (age 16-30)"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is running and model is loaded",
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata.get("model_type", "Random Forest") if metadata else "Random Forest",
        "r2_score": metadata.get("r2_score", 0.111) if metadata else 0.111,
        "features": [
            "location_type", "household_size", "age_of_respondent",
            "gender_of_respondent", "relationship_with_head", 
            "marital_status", "job_type"
        ],
        "target": "Digital Readiness (Phone + Bank + Education)",
        "countries": ["Rwanda", "Tanzania", "Kenya", "Uganda"],
        "status": "loaded"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_user(user: UserProfile):
    """Predict digital readiness for a single user"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess features
        features = preprocess_features(user.model_dump())
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Interpret result
        result = interpret_prediction(prediction, user.model_dump())
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/users", response_model=UsersPredictionResponse)
async def predict_multiple_users(users_input: UsersInput):
    """Predict digital readiness for multiple users"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        readiness_counts = {"High Digital Readiness": 0, "Moderate Digital Readiness": 0, "Low Digital Readiness": 0}
        
        for user in users_input.users:
            # Preprocess features
            features = preprocess_features(user.model_dump())
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Interpret result
            result = interpret_prediction(prediction, user.model_dump())
            predictions.append(result)
            
            # Count readiness levels
            readiness_counts[result["digital_readiness_level"]] += 1
        
        # Calculate summary statistics
        prediction_values = [p["prediction"] for p in predictions]
        summary = {
            "average_digital_readiness": round(np.mean(prediction_values), 3),
            "min_digital_readiness": round(np.min(prediction_values), 3),
            "max_digital_readiness": round(np.max(prediction_values), 3),
            "readiness_distribution": readiness_counts,
            "total_processed": len(predictions)
        }
        
        return UsersPredictionResponse(
            predictions=predictions,
            summary=summary,
            total_users=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/swagger.yaml", include_in_schema=False)
async def get_swagger_yaml():
    """Serve the swagger.yaml file"""
    swagger_path = os.path.join(os.path.dirname(__file__), "swagger.yaml")
    if os.path.exists(swagger_path):
        return FileResponse(swagger_path)
    else:
        raise HTTPException(status_code=404, detail="Swagger YAML file not found")

# Exception handlers
@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Please check your input data",
            "details": str(exc)
        }
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"🚀 Starting East Africa Digital Readiness API on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False
    )
