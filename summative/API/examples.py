#!/usr/bin/env python3
"""
API Testing and Usage Examples
==============================

This script demonstrates how to use the East Africa Youth Digital Readiness API
"""

import json

# Example 1: Single Prediction Request Body
single_prediction_example = {
    "location_type": 1,          # Urban
    "household_size": 4,         # 4 people in household
    "age_of_respondent": 22,     # 22 years old
    "gender_of_respondent": 1,   # Male
    "relationship_with_head": 2, # Child of head
    "marital_status": 3,         # Single
    "job_type": 3               # Professional job
}

# Example 2: Batch Prediction Request Body
batch_prediction_example = {
    "predictions": [
        {
            "location_type": 1,          # Urban Professional Youth
            "household_size": 4,
            "age_of_respondent": 22,
            "gender_of_respondent": 1,
            "relationship_with_head": 2,
            "marital_status": 3,
            "job_type": 3
        },
        {
            "location_type": 0,          # Rural Young Female
            "household_size": 6,
            "age_of_respondent": 19,
            "gender_of_respondent": 0,
            "relationship_with_head": 3,
            "marital_status": 3,
            "job_type": 1
        },
        {
            "location_type": 1,          # Urban Entrepreneur
            "household_size": 3,
            "age_of_respondent": 28,
            "gender_of_respondent": 1,
            "relationship_with_head": 1,
            "marital_status": 2,
            "job_type": 9
        }
    ]
}

def print_examples():
    """Print formatted examples"""
    print("="*80)
    print("EAST AFRICA YOUTH DIGITAL READINESS API - USAGE EXAMPLES")
    print("="*80)
    
    print("\n📊 SINGLE PREDICTION EXAMPLE")
    print("Endpoint: POST /predict")
    print("Request Body:")
    print(json.dumps(single_prediction_example, indent=2))
    
    print("\n📊 BATCH PREDICTION EXAMPLE")
    print("Endpoint: POST /predict/batch")
    print("Request Body:")
    print(json.dumps(batch_prediction_example, indent=2))
    
    print("\n🔍 PARAMETER EXPLANATIONS")
    print("-" * 50)
    
    explanations = {
        "location_type": "0 = Rural, 1 = Urban",
        "household_size": "Number of people in household (1-20)",
        "age_of_respondent": "Age of respondent (16-30 for youth focus)",
        "gender_of_respondent": "0 = Female, 1 = Male",
        "relationship_with_head": "0-5 (relationship to household head)",
        "marital_status": "0-4 (marital status categories)",
        "job_type": "0-15 (job/occupation categories)"
    }
    
    for param, explanation in explanations.items():
        print(f"• {param}: {explanation}")
    
    print("\n🎯 EXPECTED OUTPUT STRUCTURE")
    print("-" * 50)
    
    expected_output = {
        "digital_readiness_score": "0.0 to 1.0 (float)",
        "digital_readiness_percentage": "0% to 100% (float)",
        "readiness_category": "High/Moderate/Low/Very Low Digital Readiness",
        "recommendation": "Personalized recommendation text",
        "input_summary": "Human-readable summary of inputs",
        "model_info": "Model type and performance information"
    }
    
    print(json.dumps(expected_output, indent=2))
    
    print("\n🌐 API ENDPOINTS")
    print("-" * 50)
    endpoints = [
        "GET  /              - API information",
        "GET  /health        - Health check",
        "GET  /model/info    - Model information",
        "POST /predict       - Single prediction",
        "POST /predict/batch - Batch predictions",
        "GET  /docs          - Swagger documentation"
    ]
    
    for endpoint in endpoints:
        print(f"• {endpoint}")
    
    print("\n🚀 QUICK START COMMANDS")
    print("-" * 50)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Start API:")
    print("   python main.py")
    print("   # or")
    print("   uvicorn main:app --reload")
    print()
    print("3. Test API:")
    print("   curl -X GET http://localhost:8000/health")
    print()
    print("4. View documentation:")
    print("   Open http://localhost:8000/docs in browser")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_examples()
