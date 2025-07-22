#!/usr/bin/env python3
"""
East Africa Youth Financial Inclusion Prediction Script
=======================================================

Standalone prediction script for the youth job creation platform.
Uses the trained model to predict bank account ownership for new youth profiles.

Usage:
    python predict_bank_account.py

Features:
- Load pre-trained best model
- Interactive prediction interface
- Batch prediction capabilities
- Youth-focused insights
"""

import joblib
import numpy as np
import pandas as pd
import json
import os

def load_model_components():
    """Load the trained model, scaler, and encoders."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl') 
        encoders = joblib.load('encoders.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Model components loaded successfully!")
        print(f"   Model Type: {metadata['best_model']}")
        print(f"   Test R² Score: {metadata['test_r2']:.4f}")
        print(f"   Target Countries: {', '.join(metadata['countries'])}")
        
        return model, scaler, encoders, metadata
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please run multivariate.py first to train the model.")
        print(f"   Missing file: {e.filename}")
        return None, None, None, None

def get_user_input():
    """Interactive interface to get youth profile from user."""
    print("\n" + "="*60)
    print("YOUTH PROFILE INPUT INTERFACE")
    print("="*60)
    
    profile = {}
    
    try:
        print("\nEnter the youth profile details:")
        
        # Basic information
        profile['age'] = int(input("Age (years): "))
        if profile['age'] > 30:
            print("⚠️  Warning: This model is optimized for youth aged 30 and below")
        
        # Country selection
        countries = ['Rwanda', 'Tanzania', 'Kenya', 'Uganda']
        print(f"\nCountry options: {', '.join(countries)}")
        country = input("Country: ").title()
        if country not in countries:
            print(f"⚠️  Warning: Model trained on {countries}. Results may be less accurate.")
        
        # Location type
        location_input = input("Location (urban/rural): ").lower()
        profile['location_type'] = 1 if location_input == 'urban' else 0
        
        # Phone access
        phone_input = input("Has cellphone access? (yes/no): ").lower()
        profile['cellphone_access'] = 1 if phone_input == 'yes' else 0
        
        # Household size
        profile['household_size'] = int(input("Household size (number of people): "))
        
        # Gender
        gender_input = input("Gender (male/female): ").lower()
        profile['gender_of_respondent'] = 0 if gender_input == 'male' else 1
        
        # Relationship with household head
        print("\nRelationship with household head:")
        print("1 = Head, 2 = Spouse, 3 = Child, 4 = Other")
        profile['relationship_with_head'] = int(input("Relationship (1-4): "))
        
        # Marital status
        print("\nMarital status:")
        print("1 = Single, 2 = Married, 3 = Divorced/Widowed")
        profile['marital_status'] = int(input("Marital status (1-3): "))
        
        # Education level
        print("\nEducation level:")
        print("1 = No education, 2 = Primary, 3 = Secondary, 4 = Tertiary")
        profile['education_level'] = int(input("Education level (1-4): "))
        
        # Job type
        print("\nJob type:")
        print("1 = Agriculture, 2 = Services, 3 = Manufacturing, 4 = Business, 5 = Professional")
        profile['job_type'] = int(input("Job type (1-5): "))
        
        # Add required fields for prediction
        profile['uniqueid'] = 1  # Placeholder
        profile['age_of_respondent'] = profile['age']  # Same as age
        
        return profile
        
    except ValueError as e:
        print(f"❌ Error: Invalid input. Please enter numeric values where required.")
        return None
    except KeyboardInterrupt:
        print(f"\n❌ Input cancelled by user.")
        return None

def predict_bank_account(model, scaler, profile):
    """Make bank account prediction for a given profile."""
    
    # Create feature vector (matching training data order - exclude uniqueid)
    feature_vector = np.array([[
        profile['location_type'], 
        profile['cellphone_access'],
        profile['household_size'],
        profile['age_of_respondent'],
        profile['gender_of_respondent'],
        profile['relationship_with_head'],
        profile['marital_status'],
        profile['education_level'],
        profile['job_type']
    ]])
    
    # Apply same scaling as training data
    feature_scaled = scaler.transform(feature_vector)
    
    # Make prediction
    prediction = model.predict(feature_scaled)[0]
    
    return prediction

def interpret_prediction(prediction, profile):
    """Provide interpretation of the prediction result."""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    probability = prediction * 100
    
    print(f"\n👤 Profile Summary:")
    print(f"   • Age: {profile['age']} years")
    print(f"   • Location: {'Urban' if profile['location_type'] else 'Rural'}")
    print(f"   • Phone Access: {'Yes' if profile['cellphone_access'] else 'No'}")
    print(f"   • Education: Level {profile['education_level']}")
    print(f"   • Household Size: {profile['household_size']}")
    
    print(f"\n🏦 Bank Account Prediction:")
    print(f"   • Prediction Score: {prediction:.3f}")
    print(f"   • Probability: {probability:.1f}%")
    
    if prediction > 0.7:
        print(f"   ✅ Status: HIGH likelihood of having bank account")
        recommendation = "Strong candidate for advanced financial services"
    elif prediction > 0.4:
        print(f"   ⚠️  Status: MODERATE likelihood of having bank account") 
        recommendation = "Good candidate with some financial inclusion support needed"
    else:
        print(f"   ❌ Status: LOW likelihood of having bank account")
        recommendation = "Requires basic financial literacy and inclusion programs"
    
    print(f"\n💡 Platform Recommendation: {recommendation}")
    
    # Additional insights
    print(f"\n📊 Key Insights:")
    if profile['cellphone_access']:
        print(f"   • Phone access enables mobile banking opportunities")
    else:
        print(f"   • Limited phone access - consider basic account services first")
        
    if profile['location_type']:
        print(f"   • Urban location provides better banking infrastructure access")
    else:
        print(f"   • Rural location - mobile banking solutions recommended")
        
    if profile['education_level'] >= 3:
        print(f"   • Higher education level indicates good financial service potential")
    else:
        print(f"   • Consider financial literacy programs alongside services")

def run_batch_predictions(model, scaler):
    """Run predictions on predefined youth profiles for testing."""
    
    sample_profiles = [
        {
            'name': 'Tech-Savvy Urban Youth',
            'age': 22, 'uniqueid': 1, 'location_type': 1, 'cellphone_access': 1,
            'household_size': 4, 'age_of_respondent': 22, 'gender_of_respondent': 0,
            'relationship_with_head': 2, 'marital_status': 1, 'education_level': 3, 'job_type': 5
        },
        {
            'name': 'Rural Female Youth',
            'age': 19, 'uniqueid': 2, 'location_type': 0, 'cellphone_access': 0,
            'household_size': 6, 'age_of_respondent': 19, 'gender_of_respondent': 1,
            'relationship_with_head': 3, 'marital_status': 1, 'education_level': 2, 'job_type': 1
        },
        {
            'name': 'Semi-Urban Entrepreneur',
            'age': 28, 'uniqueid': 3, 'location_type': 1, 'cellphone_access': 1,
            'household_size': 3, 'age_of_respondent': 28, 'gender_of_respondent': 0,
            'relationship_with_head': 1, 'marital_status': 2, 'education_level': 3, 'job_type': 4
        }
    ]
    
    print("\n" + "="*60)
    print("BATCH PREDICTION RESULTS")
    print("="*60)
    
    for i, profile in enumerate(sample_profiles, 1):
        prediction = predict_bank_account(model, scaler, profile)
        print(f"\n{i}. {profile['name']} ({profile['age']} years old)")
        print(f"   🏦 Bank Account Prediction: {prediction:.3f} ({prediction*100:.1f}%)")
        
        if prediction > 0.5:
            print(f"   ✅ High likelihood of having bank account")
        else:
            print(f"   ❌ Low likelihood of having bank account")

def main():
    """Main function to run the prediction interface."""
    
    print("="*80)
    print("EAST AFRICA YOUTH FINANCIAL INCLUSION PREDICTOR")
    print("For Job Creation Platform Development")
    print("="*80)
    
    # Load model components
    model, scaler, encoders, metadata = load_model_components()
    if model is None:
        return
    
    while True:
        print("\n" + "="*60)
        print("PREDICTION OPTIONS")
        print("="*60)
        print("1. Interactive prediction (enter new profile)")
        print("2. Batch prediction (sample profiles)")
        print("3. Exit")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                # Interactive prediction
                profile = get_user_input()
                if profile:
                    prediction = predict_bank_account(model, scaler, profile)
                    interpret_prediction(prediction, profile)
                    
            elif choice == '2':
                # Batch prediction
                run_batch_predictions(model, scaler)
                
            elif choice == '3':
                print("\n👋 Thank you for using the Youth Financial Inclusion Predictor!")
                print("💡 Use these insights to develop targeted job creation strategies.")
                break
                
            else:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Exiting... Thank you for using the predictor!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
