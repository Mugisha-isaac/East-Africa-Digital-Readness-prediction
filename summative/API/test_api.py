#!/usr/bin/env python3
"""
Test script for East Africa Youth Digital Readiness API
======================================================

This script tests the API endpoints to ensure they work correctly.
"""

import requests
import json

# API base URL (change this when deployed)
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Test data - example youth profile
    test_data = {
        "location_type": 1,  # Urban
        "household_size": 4,
        "age_of_respondent": 22,
        "gender_of_respondent": 1,  # Male
        "relationship_with_head": 2,
        "marital_status": 3,
        "job_type": 3
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_batch_prediction_endpoint():
    """Test the batch prediction endpoint"""
    print("Testing batch prediction endpoint...")
    
    # Test data - multiple youth profiles
    test_data = {
        "predictions": [
            {
                "location_type": 1,  # Urban
                "household_size": 4,
                "age_of_respondent": 22,
                "gender_of_respondent": 1,  # Male
                "relationship_with_head": 2,
                "marital_status": 3,
                "job_type": 3
            },
            {
                "location_type": 0,  # Rural
                "household_size": 6,
                "age_of_respondent": 19,
                "gender_of_respondent": 0,  # Female
                "relationship_with_head": 3,
                "marital_status": 3,
                "job_type": 1
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=test_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_validation_errors():
    """Test validation error handling"""
    print("Testing validation errors...")
    
    # Test with invalid age (too old for youth focus)
    invalid_data = {
        "location_type": 1,
        "household_size": 4,
        "age_of_respondent": 35,  # Too old
        "gender_of_respondent": 1,
        "relationship_with_head": 2,
        "marital_status": 3,
        "job_type": 3
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

if __name__ == "__main__":
    print("="*60)
    print("EAST AFRICA YOUTH DIGITAL READINESS API TESTS")
    print("="*60)
    
    try:
        test_root_endpoint()
        test_health_endpoint()
        test_model_info_endpoint()
        test_prediction_endpoint()
        test_batch_prediction_endpoint()
        test_validation_errors()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"Error during testing: {str(e)}")
