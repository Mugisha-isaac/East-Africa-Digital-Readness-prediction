#!/usr/bin/env python3
"""
East Africa Youth Digital Readiness Prediction API
==================================================

FastAPI application for predicting digital readiness of youth in East Africa
for job creation platform development.

Model predicts: Phone Access + Bank Account + Education combined
Countries: Rwanda, Tanzania, Kenya, Uganda
Target audience: Youth (age ≤ 30)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
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

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing components
model = None
scaler = None
encoders = None
metadata = None

# Model paths - adjust these paths according to your model file locations
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

def load_model_components():
    """Load model, scaler, encoders and metadata"""
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
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            print("✓ Metadata loaded successfully")
        else:
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
            
        print(f"Model type: {metadata.get('best_model', 'Unknown')}")
        print(f"Model R² score: {metadata.get('test_r2', 'Unknown')}")
        
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        raise

# Load model components on startup
@app.on_event("startup")
async def startup_event():
    """Load model components when API starts"""
    load_model_components()

# Pydantic models for request/response
class PredictionInput(BaseModel):
    """Input model for digital readiness prediction"""
    
    location_type: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Location type: 0 = Rural, 1 = Urban"
    )
    household_size: int = Field(
        ..., 
        ge=1, 
        le=20, 
        description="Number of people in household (1-20)"
    )
    age_of_respondent: int = Field(
        ..., 
        ge=16, 
        le=30, 
        description="Age of respondent (16-30 years for youth focus)"
    )
    gender_of_respondent: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Gender: 0 = Female, 1 = Male"
    )
    relationship_with_head: int = Field(
        ..., 
        ge=0, 
        le=5, 
        description="Relationship with household head (0-5)"
    )
    marital_status: int = Field(
        ..., 
        ge=0, 
        le=4, 
        description="Marital status (0-4)"
    )
    job_type: int = Field(
        ..., 
        ge=0, 
        le=15, 
        description="Type of job/occupation (0-15)"
    )
    
    @validator('age_of_respondent')
    def validate_age_for_youth_focus(cls, v):
        if v > 30:
            raise ValueError('This model focuses on youth (age ≤ 30). Please provide age ≤ 30.')
        return v
    
    class Config:
        schema_extra = {
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

class PredictionOutput(BaseModel):
    """Output model for digital readiness prediction"""
    
    digital_readiness_score: float = Field(
        ..., 
        description="Digital readiness prediction score (0.0 to 1.0)"
    )
    digital_readiness_percentage: float = Field(
        ..., 
        description="Digital readiness as percentage (0% to 100%)"
    )
    readiness_category: str = Field(
        ..., 
        description="Categorical assessment of digital readiness"
    )
    recommendation: str = Field(
        ..., 
        description="Recommendation based on prediction"
    )
    input_summary: Dict = Field(
        ..., 
        description="Summary of input characteristics"
    )
    model_info: Dict = Field(
        ..., 
        description="Information about the model used"
    )

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    model_loaded: bool
    api_version: str

class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    predictions: List[PredictionInput] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of prediction inputs (max 100)"
    )

class BatchPredictionOutput(BaseModel):
    """Output model for batch predictions"""
    predictions: List[PredictionOutput]
    summary: Dict

class UsersInput(BaseModel):
    """Input model for multiple users - primary endpoint"""
    users: List[PredictionInput] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Array of user profiles for digital readiness prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "users": [
                    {
                        "location_type": 1,
                        "household_size": 4,
                        "age_of_respondent": 22,
                        "gender_of_respondent": 1,
                        "relationship_with_head": 2,
                        "marital_status": 3,
                        "job_type": 3
                    },
                    {
                        "location_type": 0,
                        "household_size": 6,
                        "age_of_respondent": 19,
                        "gender_of_respondent": 0,
                        "relationship_with_head": 3,
                        "marital_status": 3,
                        "job_type": 1
                    },
                    {
                        "location_type": 1,
                        "household_size": 3,
                        "age_of_respondent": 28,
                        "gender_of_respondent": 1,
                        "relationship_with_head": 1,
                        "marital_status": 2,
                        "job_type": 9
                    }
                ]
            }
        }

class UsersOutput(BaseModel):
    """Output model for multiple users predictions"""
    predictions: List[PredictionOutput] = Field(
        ...,
        description="Array of predictions for each user"
    )
    summary: Dict = Field(
        ...,
        description="Summary statistics for the batch"
    )

# Helper functions
def get_readiness_category(score: float) -> str:
    """Categorize digital readiness score"""
    if score >= 0.7:
        return "High Digital Readiness"
    elif score >= 0.4:
        return "Moderate Digital Readiness"
    elif score >= 0.2:
        return "Low Digital Readiness"
    else:
        return "Very Low Digital Readiness"

def get_recommendation(score: float, input_data: Dict) -> str:
    """Generate recommendation based on score and input characteristics"""
    age = input_data.get('age_of_respondent', 0)
    location = input_data.get('location_type', 0)
    gender = input_data.get('gender_of_respondent', 0)
    
    location_str = "urban" if location == 1 else "rural"
    gender_str = "male" if gender == 1 else "female"
    
    if score >= 0.7:
        return f"Excellent candidate for digital job platform! This {age}-year-old {gender_str} from {location_str} area shows high digital readiness."
    elif score >= 0.4:
        return f"Good potential for digital job platform. Consider providing additional digital skills training for this {age}-year-old {gender_str} from {location_str} area."
    elif score >= 0.2:
        return f"Moderate potential. Recommend comprehensive digital literacy program before platform onboarding for this {age}-year-old {gender_str} from {location_str} area."
    else:
        return f"Low digital readiness. Requires extensive digital inclusion support and basic digital skills training for this {age}-year-old {gender_str} from {location_str} area."

def create_input_summary(input_data: Dict) -> Dict:
    """Create a human-readable summary of input data"""
    return {
        "profile": {
            "age": input_data['age_of_respondent'],
            "gender": "Male" if input_data['gender_of_respondent'] == 1 else "Female",
            "location": "Urban" if input_data['location_type'] == 1 else "Rural",
            "household_size": input_data['household_size']
        },
        "social": {
            "relationship_with_head": input_data['relationship_with_head'],
            "marital_status": input_data['marital_status']
        },
        "economic": {
            "job_type": input_data['job_type']
        }
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
            "batch_prediction": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "documentation": "/docs",
            "swagger_file": "/swagger.yaml"
        },
        "countries": ["Rwanda", "Tanzania", "Kenya", "Uganda"],
        "target_audience": "Youth (age ≤ 30)",
        "primary_endpoint": "/predict/users - Send array of user data for predictions"
    }

@app.get("/swagger.yaml")
async def get_swagger_file():
    """Serve the Swagger YAML file"""
    return FileResponse(
        path="swagger.yaml",
        media_type="application/x-yaml",
        filename="swagger.yaml"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and scaler is not None and encoders is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        message="API is running and model is loaded" if model_loaded else "API is running but model is not loaded",
        model_loaded=model_loaded,
        api_version="1.0.0"
    )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model metadata not available")
    
    return {
        "model_type": metadata.get("best_model", "Unknown"),
        "model_performance": {
            "test_r2_score": metadata.get("test_r2", "Unknown"),
            "description": "R² score on test set (higher is better, max 1.0)"
        },
        "target_variable": metadata.get("target", "Unknown"),
        "description": metadata.get("description", "Unknown"),
        "dataset_info": {
            "size": metadata.get("dataset_size", "Unknown"),
            "countries": metadata.get("countries", []),
            "youth_focus": metadata.get("youth_focus", False)
        },
        "features": metadata.get("features", []),
        "prediction_details": {
            "output_range": "0.0 to 1.0",
            "interpretation": "Higher scores indicate higher digital readiness (phone + bank + education)",
            "categories": {
                "0.7+": "High Digital Readiness",
                "0.4-0.7": "Moderate Digital Readiness", 
                "0.2-0.4": "Low Digital Readiness",
                "0.0-0.2": "Very Low Digital Readiness"
            }
        }
    }

@app.post("/predict/users", response_model=UsersOutput)
async def predict_users_digital_readiness(users_input: UsersInput):
    """Predict digital readiness for multiple users - Primary endpoint for array input"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        predictions = []
        scores = []
        
        for user_data in users_input.users:
            # Convert to feature array
            features = np.array([[
                user_data.location_type,
                user_data.household_size,
                user_data.age_of_respondent,
                user_data.gender_of_respondent,
                user_data.relationship_with_head,
                user_data.marital_status,
                user_data.job_type
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prediction = max(0.0, min(1.0, prediction))
            scores.append(prediction)
            
            # Create response for this user
            user_dict = user_data.dict()
            
            pred_output = PredictionOutput(
                digital_readiness_score=round(prediction, 4),
                digital_readiness_percentage=round(prediction * 100, 2),
                readiness_category=get_readiness_category(prediction),
                recommendation=get_recommendation(prediction, user_dict),
                input_summary=create_input_summary(user_dict),
                model_info={
                    "model_type": metadata.get("best_model", "Unknown"),
                    "model_r2_score": metadata.get("test_r2", "Unknown"),
                    "prediction_target": "Digital Readiness (Phone + Bank + Education)"
                }
            )
            
            predictions.append(pred_output)
        
        # Calculate summary statistics
        summary = {
            "total_predictions": len(scores),
            "average_score": round(np.mean(scores), 4),
            "score_distribution": {
                "high_readiness": len([s for s in scores if s >= 0.7]),
                "moderate_readiness": len([s for s in scores if 0.4 <= s < 0.7]),
                "low_readiness": len([s for s in scores if 0.2 <= s < 0.4]),
                "very_low_readiness": len([s for s in scores if s < 0.2])
            },
            "highest_score": round(max(scores), 4),
            "lowest_score": round(min(scores), 4),
            "readiness_insights": {
                "ready_for_platform": len([s for s in scores if s >= 0.4]),
                "need_support": len([s for s in scores if s < 0.4]),
                "average_readiness_level": get_readiness_category(np.mean(scores))
            }
        }
        
        return UsersOutput(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Users prediction error: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
async def predict_digital_readiness(input_data: PredictionInput):
    """Predict digital readiness for a single youth profile"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        # Convert input to feature array
        features = np.array([[
            input_data.location_type,
            input_data.household_size,
            input_data.age_of_respondent,
            input_data.gender_of_respondent,
            input_data.relationship_with_head,
            input_data.marital_status,
            input_data.job_type
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Ensure prediction is in valid range
        prediction = max(0.0, min(1.0, prediction))
        
        # Generate response
        input_dict = input_data.dict()
        
        response = PredictionOutput(
            digital_readiness_score=round(prediction, 4),
            digital_readiness_percentage=round(prediction * 100, 2),
            readiness_category=get_readiness_category(prediction),
            recommendation=get_recommendation(prediction, input_dict),
            input_summary=create_input_summary(input_dict),
            model_info={
                "model_type": metadata.get("best_model", "Unknown"),
                "model_r2_score": metadata.get("test_r2", "Unknown"),
                "prediction_target": "Digital Readiness (Phone + Bank + Education)"
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(batch_input: BatchPredictionInput):
    """Predict digital readiness for multiple youth profiles"""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    try:
        predictions = []
        scores = []
        
        for input_data in batch_input.predictions:
            # Convert to feature array
            features = np.array([[
                input_data.location_type,
                input_data.household_size,
                input_data.age_of_respondent,
                input_data.gender_of_respondent,
                input_data.relationship_with_head,
                input_data.marital_status,
                input_data.job_type
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prediction = max(0.0, min(1.0, prediction))
            scores.append(prediction)
            
            # Create response for this prediction
            input_dict = input_data.dict()
            
            pred_output = PredictionOutput(
                digital_readiness_score=round(prediction, 4),
                digital_readiness_percentage=round(prediction * 100, 2),
                readiness_category=get_readiness_category(prediction),
                recommendation=get_recommendation(prediction, input_dict),
                input_summary=create_input_summary(input_dict),
                model_info={
                    "model_type": metadata.get("best_model", "Unknown"),
                    "model_r2_score": metadata.get("test_r2", "Unknown"),
                    "prediction_target": "Digital Readiness (Phone + Bank + Education)"
                }
            )
            
            predictions.append(pred_output)
        
        # Calculate batch summary
        summary = {
            "total_predictions": len(scores),
            "average_score": round(np.mean(scores), 4),
            "score_distribution": {
                "high_readiness": len([s for s in scores if s >= 0.7]),
                "moderate_readiness": len([s for s in scores if 0.4 <= s < 0.7]),
                "low_readiness": len([s for s in scores if 0.2 <= s < 0.4]),
                "very_low_readiness": len([s for s in scores if s < 0.2])
            },
            "highest_score": round(max(scores), 4),
            "lowest_score": round(min(scores), 4)
        }
        
        return BatchPredictionOutput(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Please check the URL and try again",
            "available_endpoints": ["/", "/predict/users", "/predict", "/predict/batch", "/health", "/model/info", "/docs", "/swagger.yaml"]
        }
    )

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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
