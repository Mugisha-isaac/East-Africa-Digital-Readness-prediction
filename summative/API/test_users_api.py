#!/usr/bin/env python3
"""
Test script for East Africa Youth Digital Readiness API - Users Endpoint
========================================================================

This script tests the new /predict/users endpoint that accepts array input.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_users_endpoint():
    """Test the main /predict/users endpoint with array input"""
    print("Testing /predict/users endpoint (Array Input)...")
    
    # Test data - array of users
    test_data = {
        "users": [
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
            },
            {
                "location_type": 1,  # Urban
                "household_size": 3,
                "age_of_respondent": 28,
                "gender_of_respondent": 1,  # Male
                "relationship_with_head": 1,
                "marital_status": 2,
                "job_type": 9
            },
            {
                "location_type": 0,  # Rural
                "household_size": 5,
                "age_of_respondent": 24,
                "gender_of_respondent": 0,  # Female
                "relationship_with_head": 2,
                "marital_status": 1,
                "job_type": 5
            },
            {
                "location_type": 1,  # Urban
                "household_size": 2,
                "age_of_respondent": 26,
                "gender_of_respondent": 1,  # Male
                "relationship_with_head": 1,
                "marital_status": 2,
                "job_type": 12
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/users", json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n🎯 SUCCESS! Array prediction completed")
            print(f"📊 Total predictions: {result['summary']['total_predictions']}")
            print(f"📈 Average score: {result['summary']['average_score']}")
            print(f"🏆 Highest score: {result['summary']['highest_score']}")
            print(f"📉 Lowest score: {result['summary']['lowest_score']}")
            
            print("\n📋 Distribution:")
            dist = result['summary']['score_distribution']
            print(f"  • High readiness: {dist['high_readiness']}")
            print(f"  • Moderate readiness: {dist['moderate_readiness']}")
            print(f"  • Low readiness: {dist['low_readiness']}")
            print(f"  • Very low readiness: {dist['very_low_readiness']}")
            
            print("\n👥 Individual Predictions:")
            for i, pred in enumerate(result['predictions'][:3]):  # Show first 3
                print(f"  User {i+1}: {pred['digital_readiness_percentage']:.1f}% - {pred['readiness_category']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 80)

def test_single_user():
    """Test with a single user in the array"""
    print("Testing /predict/users endpoint with single user...")
    
    test_data = {
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
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/users", json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single user prediction successful")
            pred = result['predictions'][0]
            print(f"Score: {pred['digital_readiness_percentage']:.1f}%")
            print(f"Category: {pred['readiness_category']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 80)

def test_validation_errors():
    """Test validation with invalid data"""
    print("Testing validation errors...")
    
    # Test with invalid age (too old)
    invalid_data = {
        "users": [
            {
                "location_type": 1,
                "household_size": 4,
                "age_of_respondent": 35,  # Too old for youth focus
                "gender_of_respondent": 1,
                "relationship_with_head": 2,
                "marital_status": 3,
                "job_type": 3
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/users", json=invalid_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 422:
            print("✅ Validation error caught correctly")
            result = response.json()
            print(f"Error details: {result}")
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("-" * 80)

def test_all_endpoints():
    """Test all available endpoints"""
    print("Testing all available endpoints...")
    
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/model/info", "GET"),
        ("/swagger.yaml", "GET"),
        ("/docs", "GET")
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}")
                status = "✅" if response.status_code == 200 else "❌"
                print(f"{status} {method} {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ {method} {endpoint} - Error: {str(e)}")
    
    print("-" * 80)

def show_curl_examples():
    """Show cURL examples for testing"""
    print("📋 cURL Examples for Testing:")
    print()
    
    print("1. Test multiple users:")
    print('curl -X POST "http://localhost:8000/predict/users" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{')
    print('       "users": [')
    print('         {')
    print('           "location_type": 1,')
    print('           "household_size": 4,')
    print('           "age_of_respondent": 22,')
    print('           "gender_of_respondent": 1,')
    print('           "relationship_with_head": 2,')
    print('           "marital_status": 3,')
    print('           "job_type": 3')
    print('         },')
    print('         {')
    print('           "location_type": 0,')
    print('           "household_size": 6,')
    print('           "age_of_respondent": 19,')
    print('           "gender_of_respondent": 0,')
    print('           "relationship_with_head": 3,')
    print('           "marital_status": 3,')
    print('           "job_type": 1')
    print('         }')
    print('       ]')
    print('     }\'')
    print()
    
    print("2. Check API health:")
    print('curl -X GET "http://localhost:8000/health"')
    print()
    
    print("3. Get Swagger documentation:")
    print('curl -X GET "http://localhost:8000/swagger.yaml"')
    print()
    
    print("4. Access interactive docs:")
    print('Open in browser: http://localhost:8000/docs')
    print("-" * 80)

if __name__ == "__main__":
    print("="*80)
    print("EAST AFRICA YOUTH DIGITAL READINESS API - ARRAY INPUT TESTS")
    print("="*80)
    
    test_all_endpoints()
    test_users_endpoint()
    test_single_user()
    test_validation_errors()
    show_curl_examples()
    
    print("🎉 All tests completed!")
    print("📖 Visit http://localhost:8000/docs for interactive API documentation")
    print("📄 Download Swagger file: http://localhost:8000/swagger.yaml")
