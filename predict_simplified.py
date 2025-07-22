#!/usr/bin/env python3
"""
East Africa Youth Financial Inclusion Predictor - Simplified Version
===================================================================

Standalone prediction tool for the youth job creation platform.
Uses trained models to predict bank account ownership likelihood.

Usage: python predict_simplified.py
"""

import joblib
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def load_model_components():
    """Load the trained model and preprocessing components."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl') 
        encoders = joblib.load('encoders.pkl')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✅ Model components loaded successfully!")
        print(f"   Model Type: {metadata['best_model']}")
        print(f"   Test R² Score: {metadata['test_r2']:.4f}")
        print(f"   Countries: {', '.join(metadata['countries'])}")
        return model, scaler, encoders, metadata
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please run the analysis script first.")
        return None, None, None, None

def predict_profile(model, scaler, profile_data):
    """Make prediction for a youth profile."""
    features_scaled = scaler.transform([profile_data])
    prediction = model.predict(features_scaled)[0]
    return prediction

def interactive_prediction(model, scaler):
    """Interactive interface for user input."""
    print("\n" + "="*60)
    print("YOUTH PROFILE INPUT")
    print("="*60)
    
    try:
        # Get user input
        age = int(input("Age (years): "))
        location = input("Location (urban/rural): ").lower()
        phone = input("Has cellphone access? (yes/no): ").lower()
        household_size = int(input("Household size: "))
        gender = input("Gender (male/female): ").lower()
        
        print("\nEducation level (0-5): 0=None, 2=Primary, 3=Secondary, 4=Tertiary")
        education = int(input("Education level: "))
        
        print("\nJob type (0-9): 1=Farming, 3=Private Employee, 9=Self Employed")
        job = int(input("Job type: "))
        
        # Encode inputs
        profile_data = [
            1 if location == 'urban' else 0,  # location_type
            1 if phone == 'yes' else 0,       # cellphone_access
            household_size,                    # household_size
            age,                              # age_of_respondent
            0 if gender == 'female' else 1,   # gender_of_respondent
            2,                                # relationship_with_head (default: child)
            3,                                # marital_status (default: single)
            education,                        # education_level
            job                               # job_type
        ]
        
        # Make prediction
        prediction = predict_profile(model, scaler, profile_data)
        
        # Display results
        print(f"\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Bank Account Prediction: {prediction:.3f} ({prediction*100:.1f}%)")
        
        if prediction > 0.5:
            print("✅ HIGH likelihood of having bank account")
            recommendation = "Good candidate for advanced financial services"
        elif prediction > 0.2:
            print("⚠️  MODERATE likelihood of having bank account")
            recommendation = "May need some financial inclusion support"
        else:
            print("❌ LOW likelihood of having bank account")
            recommendation = "Requires basic financial literacy programs"
        
        print(f"Recommendation: {recommendation}")
        
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid input or cancelled by user")

def batch_predictions(model, scaler):
    """Run predictions on sample profiles."""
    profiles = [
        {
            'name': 'Tech-Savvy Urban Youth',
            'data': [1, 1, 4, 22, 1, 2, 3, 3, 3]  # Urban, phone, household=4, age=22, male, etc.
        },
        {
            'name': 'Rural Female Youth',
            'data': [0, 0, 6, 19, 0, 3, 3, 2, 1]  # Rural, no phone, household=6, age=19, female, etc.
        },
        {
            'name': 'Educated Urban Youth',
            'data': [1, 1, 3, 25, 1, 1, 2, 4, 3]  # Urban, phone, household=3, age=25, male, tertiary education
        }
    ]
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    for i, profile in enumerate(profiles, 1):
        prediction = predict_profile(model, scaler, profile['data'])
        status = "✅ High" if prediction > 0.5 else ("⚠️  Moderate" if prediction > 0.2 else "❌ Low")
        
        print(f"{i}. {profile['name']}")
        print(f"   Prediction: {prediction:.3f} ({prediction*100:.1f}%)")
        print(f"   Status: {status} likelihood")
        print()

def main():
    """Main function."""
    print("="*70)
    print("EAST AFRICA YOUTH FINANCIAL INCLUSION PREDICTOR")
    print("="*70)
    
    # Load model
    model, scaler, encoders, metadata = load_model_components()
    if model is None:
        return
    
    # Main menu
    while True:
        print(f"\n" + "="*50)
        print("PREDICTION OPTIONS")
        print("="*50)
        print("1. Interactive prediction")
        print("2. Sample batch predictions")
        print("3. Exit")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                interactive_prediction(model, scaler)
            elif choice == '2':
                batch_predictions(model, scaler)
            elif choice == '3':
                print("👋 Thank you for using the predictor!")
                break
            else:
                print("❌ Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print(f"\n👋 Exiting... Thank you!")
            break

if __name__ == "__main__":
    main()
