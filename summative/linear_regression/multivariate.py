#!/usr/bin/env python3
"""
East Africa Youth Digital Readiness Model Training
==================================================

Model training script for youth job creation platform using East Africa 
Financial Inclusion Survey data with advanced machine learning techniques.

Purpose: Train and save models for API deployment
Target: Digital readiness (phone + bank + education combined)
Output: Trained models ready for FastAPI integration

Dataset: East Africa Financial Inclusion Survey
Countries: Rwanda, Tanzania, Kenya, Uganda
Focus: Youth (age ≤ 30) digital readiness prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("EAST AFRICA YOUTH DIGITAL READINESS MODEL TRAINING")
print("Training models for FastAPI deployment")
print("="*80)

# ===== DATA LOADING AND INITIAL EXPLORATION =====
print("\n📊 Loading and exploring dataset...")
data = pd.read_csv('../../data.csv')
print(f"Dataset loaded: {data.shape[0]:,} samples, {data.shape[1]} features")

# Quick overview
countries = data['country'].value_counts()
bank_ownership_rate = (data['bank_account'] == 'Yes').mean() * 100
print(f"Countries: {dict(countries)}")
print(f"Overall bank account ownership: {bank_ownership_rate:.1f}%")

# ===== YOUTH-FOCUSED ANALYSIS =====
print("\n👥 Youth Analysis (Age ≤ 30)...")
youth_data = data[data['age_of_respondent'] <= 30].copy()
youth_bank_rate = (youth_data['bank_account'] == 'Yes').mean() * 100
youth_phone_rate = (youth_data['cellphone_access'] == 'Yes').mean() * 100

print(f"Youth samples: {len(youth_data):,} ({len(youth_data)/len(data)*100:.1f}% of total)")
print(f"Youth bank ownership: {youth_bank_rate:.1f}%")
print(f"Youth phone access: {youth_phone_rate:.1f}%")

# Digital readiness analysis - youth with ALL THREE: cellphone + bank account + education
digital_ready = data[
    (data['age_of_respondent'] <= 30) &
    (data['bank_account'] == 'Yes') & 
    (data['cellphone_access'] == 'Yes') & 
    (data['education_level'].isin(['Primary education', 'Secondary education', 'Tertiary education']))
]

# Create digital readiness target variable for ALL data
data['digital_ready'] = (
    (data['bank_account'] == 'Yes') & 
    (data['cellphone_access'] == 'Yes') & 
    (data['education_level'].isin(['Primary education', 'Secondary education', 'Tertiary education']))
).astype(int)

digital_ready_rate = data['digital_ready'].mean() * 100
youth_digital_ready_rate = data[data['age_of_respondent'] <= 30]['digital_ready'].mean() * 100

print(f"Overall digitally ready population: {data['digital_ready'].sum():,} ({digital_ready_rate:.1f}%)")
print(f"Youth digitally ready: {len(digital_ready):,} ({youth_digital_ready_rate:.1f}%)")
print("Target variable changed: Now predicting DIGITAL READINESS (phone + bank + education)")

# ===== DATA PREPROCESSING =====
print("\n🔧 Data preprocessing...")

# Drop non-predictive columns (including the individual components we combined into digital_ready)
data_processed = data.drop(['country', 'year', 'uniqueid', 'bank_account', 'cellphone_access', 'education_level'], axis=1).copy()
print(f"Dropped non-predictive columns. Remaining: {data_processed.shape[1]} columns")
print("Note: Removed bank_account, cellphone_access, education_level as they're combined into digital_ready target")

# Encode categorical variables
label_encoders = {}
categorical_columns = data_processed.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    data_processed[col] = le.fit_transform(data_processed[col])
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} categories")

# Separate features and target (now digital_ready instead of bank_account)
X = data_processed.drop('digital_ready', axis=1)
y = data_processed['digital_ready']

print(f"Features shape: {X.shape}")
print(f"Target distribution (Digital Ready): {np.bincount(y)} (No=0, Yes=1)")
print(f"Digital readiness rate in dataset: {y.mean()*100:.1f}%")

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

# ===== FEATURE IMPORTANCE ANALYSIS =====
print("\n📈 Feature importance analysis...")
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("Top features by correlation with DIGITAL READINESS (phone + bank + education):")
for feature, corr in correlations.head(5).items():
    print(f"  {feature}: {corr:.3f}")

# ===== ESSENTIAL VISUALIZATIONS FOR TRAINING =====
print("\n📊 Creating training visualizations...")

# Create focused visualization for model training
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Digital Readiness Model Training Overview', fontsize=14, fontweight='bold')

# 1. Target distribution
target_counts = np.bincount(y)
target_labels = ['Not Ready', 'Digitally Ready']
axes[0].bar(target_labels, target_counts, color=['lightcoral', 'lightgreen'])
axes[0].set_title('Target Variable Distribution')
axes[0].set_ylabel('Count')
for i, count in enumerate(target_counts):
    axes[0].text(i, count + max(target_counts)*0.01, f'{count:,}', ha='center', fontweight='bold')

# 2. Feature importance (correlation with target)
top_features = correlations.head(8)
axes[1].barh(range(len(top_features)), top_features.values, color='skyblue')
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features.index)
axes[1].set_xlabel('Correlation with Digital Readiness')
axes[1].set_title('Top Feature Importance')

plt.tight_layout()
plt.show()

# ===== CUSTOM GRADIENT DESCENT IMPLEMENTATION =====
print("\n🤖 Implementing Custom Gradient Descent...")

class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=2000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = X.dot(self.weights) + self.bias
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Backward pass
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
    def predict(self, X):
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# Train gradient descent model
print("Training custom gradient descent model...")
gd_model = GradientDescentLinearRegression(learning_rate=0.01, max_iterations=2000)
gd_model.fit(X_train_scaled, y_train)

gd_train_pred = gd_model.predict(X_train_scaled)
gd_test_pred = gd_model.predict(X_test_scaled)

gd_train_r2 = r2_score(y_train, gd_train_pred)
gd_test_r2 = r2_score(y_test, gd_test_pred)

print(f"Gradient Descent - Train R²: {gd_train_r2:.4f}, Test R²: {gd_test_r2:.4f}")
print("Model target: Predicting DIGITAL READINESS (phone + bank + education combination)")

# ===== MODEL COMPARISON =====
print("\n🏆 Comparing multiple models for DIGITAL READINESS prediction...")

# Initialize models
models = {
    'Gradient Descent LR': gd_model,
    'Scikit-learn LR': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
}

# Train models (except gradient descent which is already trained)
for name, model in models.items():
    if name != 'Gradient Descent LR':
        model.fit(X_train_scaled, y_train)

# Evaluate models
results = {}
for name, model in models.items():
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    results[name] = {
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'model': model
    }

# Display results
print("\nMODEL PERFORMANCE COMPARISON:")
print("="*70)
print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test MSE':<12} {'Test MAE':<12}")
print("="*70)

best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])

for name, metrics in results.items():
    marker = "🏆" if name == best_model_name else "  "
    print(f"{marker} {name:<18} {metrics['train_r2']:<12.4f} {metrics['test_r2']:<12.4f} "
          f"{metrics['test_mse']:<12.6f} {metrics['test_mae']:<12.6f}")

print("="*70)
print(f"Best Model: {best_model_name} (Test R² = {results[best_model_name]['test_r2']:.4f})")

# ===== MODEL PERFORMANCE VISUALIZATION =====
print("\n📊 Model performance visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

# R² comparison
model_names = list(results.keys())
train_r2 = [results[name]['train_r2'] for name in model_names]
test_r2 = [results[name]['test_r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, train_r2, width, label='Training R²', alpha=0.8)
axes[0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
axes[0].set_xlabel('Models')
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Best model predictions vs actual
best_model = results[best_model_name]['model']
best_pred = best_model.predict(X_test_scaled)
axes[1].scatter(y_test, best_pred, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'Best Model: Actual vs Predicted')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== SAVE BEST MODEL =====
print("\n💾 Saving best model and components...")

best_model = results[best_model_name]['model']
joblib.dump(best_model, '../../best_model.pkl')
joblib.dump(scaler, '../../scaler.pkl')
joblib.dump(label_encoders, '../../encoders.pkl')

# Save metadata
metadata = {
    'model_type': best_model_name,
    'r2_score': results[best_model_name]['test_r2'],
    'features': list(X.columns),
    'target': 'digital_ready',
    'description': 'East Africa Youth Digital Readiness Prediction Model (Phone + Bank + Education)',
    'dataset_size': len(data),
    'youth_focus': True,
    'countries': list(countries.index)
}

with open('../../model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Best model ({best_model_name}) saved to: ../../best_model.pkl")
print("✓ Scaler and encoders saved")
print("✓ Metadata saved")

# ===== FINAL SUMMARY =====
print("="*80)
print("TRAINING COMPLETE - DIGITAL READINESS MODEL")
print("="*80)

print(f"""
🌍 DATASET OVERVIEW:
   • Total Samples: {len(data):,}
   • Youth Samples (≤30): {len(youth_data):,} ({len(youth_data)/len(data)*100:.1f}%)
   • Countries: {', '.join(countries.index)}

👥 KEY INSIGHTS:
   • Overall Digital Readiness: {digital_ready_rate:.1f}%
   • Youth Digital Readiness: {youth_digital_ready_rate:.1f}%
   • Target: Phone + Bank Account + Education combined

🤖 MODEL PERFORMANCE:
   • Best Model: {best_model_name}
   • Test R² Score: {results[best_model_name]['test_r2']:.4f}
   • Feature Count: {len(X.columns)}

🚀 READY FOR API DEPLOYMENT:
   ✓ Model saved and ready for API integration
   ✓ Preprocessing components saved
   ✓ Predicts comprehensive digital readiness
   ✓ Perfect for identifying job platform ready candidates
""")

print("="*80)
print("MODEL TRAINING COMPLETE!")
print("Use the FastAPI application for predictions")
print("="*80)
