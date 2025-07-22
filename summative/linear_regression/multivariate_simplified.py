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

# Suppress warnings
warnings.filterwarnings('ignore')

# setting styles for visualization
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("="*80)
print("EAST AFRICA YOUTH FINANCIAL INCLUSION ANALYSIS")
print("For Job Creation Platform Development")
print("="*80)

# ===== DATA LOADING AND INITIAL EXPLORATION =====
print("\n📊 Loading and exploring dataset...")
data = pd.read_csv('../../data.csv')
print(f"Dataset loaded: {data.shape[0]:,} samples, {data.shape[1]} features")

# Quick data overview
countries = data['country'].value_counts()
bank_ownership_rate = (data['bank_account'] == 'Yes').mean() * 100
print(f"Countries: {dict(countries)}")
print(f"Overall bank account ownership: {bank_ownership_rate:.1f}%")

# ===== Youth Focus: Filtering Data =====
youth_data = data[data['age_of_respondent'] <= 30].copy()
youth_bank_rate = (youth_data['bank_account'] == 'Yes').mean() * 100
youth_phone_rate = (youth_data['cellphone_access'] == 'Yes').mean() * 100

print(f"Youth samples: {len(youth_data):,} ({len(youth_data)/len(data)*100:.1f}% of total)")
print(f"Youth bank ownership: {youth_bank_rate:.1f}%")
print(f"Youth phone access: {youth_phone_rate:.1f}%")

# Digital readiness analysis
digital_ready = youth_data[
    (youth_data['bank_account'] == 'Yes') & 
    (youth_data['cellphone_access'] == 'Yes') & 
    (youth_data['education_level'].isin(['Primary education', 'Secondary education', 'Tertiary education']))
]

untapped_potential = youth_data[
    (youth_data['bank_account'] == 'No') & 
    (youth_data['cellphone_access'] == 'Yes') & 
    (youth_data['education_level'].isin(['Primary education', 'Secondary education', 'Tertiary education']))
]

print(f"Digitally ready youth: {len(digital_ready):,}")
print(f"Untapped potential: {len(untapped_potential):,} ({len(untapped_potential)/len(youth_data)*100:.1f}%)")


# ===== DATA PREPROCESSING =====
print("\n🔧 Data preprocessing...")

# ====== DROP UNNECESSARY COLUMNS =====
data_processed = data.drop(['country', 'year', 'uniqueid'], axis=1).copy()
print(f"Dropped non-predictive columns. Remaining: {data_processed.shape[1]} columns")

# ====== Encode categorical variables =====
label_encoders = {}
categorical_columns = data_processed.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    data_processed[col] = le.fit_transform(data_processed[col])
    label_encoders[col] = le

# Separate features and target
X = data_processed.drop('bank_account', axis=1)
y = data_processed['bank_account']

print(f"Features shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")  

# ====== Train-test split and scaling =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train):,}, Testing samples: {len(X_test):,}")

# ===== FEATURE IMPORTANCE ANALYSIS =====
print("\n📈 Feature importance analysis...")
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("Top features by correlation with bank account ownership:")
for feature, corr in correlations.head(5).items():
    print(f"  {feature}: {corr:.3f}")

# ===== VISUALIZATIONS =====
print("\n📊 Creating visualizations...")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('East Africa Youth Financial Inclusion Analysis', fontsize=16, fontweight='bold')

# 1. Bank ownership by country
country_bank = pd.crosstab(data['country'], data['bank_account'], normalize='index') * 100
country_bank.plot(kind='bar', ax=axes[0,0], rot=45)
axes[0,0].set_title('Bank Account Ownership by Country')
axes[0,0].set_ylabel('Percentage')

# 2. Youth vs Overall bank ownership
categories = ['Overall Population', 'Youth (≤30)']
rates = [bank_ownership_rate, youth_bank_rate]
bars = axes[0,1].bar(categories, rates, color=['lightblue', 'lightcoral'])
axes[0,1].set_title('Bank Account Ownership: Overall vs Youth')
axes[0,1].set_ylabel('Percentage (%)')
for bar, rate in zip(bars, rates):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{rate:.1f}%', ha='center', fontweight='bold')
    
# 3. Digital readiness metrics
readiness_categories = ['Bank Account', 'Phone Access', 'Education', 'All Three']
youth_metrics = [
    youth_bank_rate,
    youth_phone_rate,
    (youth_data['education_level'].isin(['Primary education', 'Secondary education', 'Tertiary education'])).mean() * 100,
    len(digital_ready) / len(youth_data) * 100
]
axes[0,2].bar(readiness_categories, youth_metrics, color='skyblue')
axes[0,2].set_title('Youth Digital Financial Readiness')
axes[0,2].set_ylabel('Percentage (%)')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Feature correlation heatmap
correlation_matrix = X.corr()
top_features = correlations.head(6).index
sns.heatmap(correlation_matrix.loc[top_features, top_features], 
            annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
axes[1,0].set_title('Feature Correlation Matrix (Top Features)')

# 5. Age distribution by bank ownership
for bank_status in [0, 1]:
    subset = data_processed[data_processed['bank_account'] == bank_status]
    axes[1,1].hist(subset['age_of_respondent'], alpha=0.7, 
                   label=f'Bank Account: {"Yes" if bank_status else "No"}', bins=20)
axes[1,1].set_title('Age Distribution by Bank Account Ownership')
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('Frequency')
axes[1,1].legend()

# 6. Education level impact
education_impact = data_processed.groupby('education_level')['bank_account'].mean()
axes[1,2].bar(range(len(education_impact)), education_impact.values, color='lightgreen')
axes[1,2].set_title('Bank Account Ownership by Education Level')
axes[1,2].set_xlabel('Education Level (Encoded)')
axes[1,2].set_ylabel('Bank Account Ownership Rate')

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

# ===== MODEL COMPARISON =====
print("\n🏆 Comparing multiple models...")

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

# ===== MODEL COMPARISON VISUALIZATION =====
print("\n📊 Model comparison visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')

# R² comparison
model_names = list(results.keys())
train_r2 = [results[name]['train_r2'] for name in model_names]
test_r2 = [results[name]['test_r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0,0].bar(x - width/2, train_r2, width, label='Training R²', alpha=0.8)
axes[0,0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
axes[0,0].set_xlabel('Models')
axes[0,0].set_ylabel('R² Score')
axes[0,0].set_title('R² Score Comparison')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MSE comparison
test_mse = [results[name]['test_mse'] for name in model_names]
axes[0,1].bar(model_names, test_mse, color='lightcoral', alpha=0.8)
axes[0,1].set_xlabel('Models')
axes[0,1].set_ylabel('Mean Squared Error')
axes[0,1].set_title('Test MSE Comparison')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].grid(True, alpha=0.3)

# Best model predictions
best_model = results[best_model_name]['model']
best_pred = best_model.predict(X_test_scaled)
axes[1,0].scatter(y_test, best_pred, alpha=0.6, color='green')
axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[1,0].set_xlabel('Actual Values')
axes[1,0].set_ylabel('Predicted Values')
axes[1,0].set_title(f'Best Model ({best_model_name}): Actual vs Predicted')
axes[1,0].grid(True, alpha=0.3)

# Gradient descent loss curve
axes[1,1].plot(gd_model.cost_history, color='blue', linewidth=2)
axes[1,1].set_xlabel('Iterations')
axes[1,1].set_ylabel('Cost (MSE)')
axes[1,1].set_title('Gradient Descent: Loss Curve')
axes[1,1].grid(True, alpha=0.3)

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
    'best_model': best_model_name,
    'test_r2': results[best_model_name]['test_r2'],
    'features': list(X.columns),
    'target': 'bank_account',
    'description': 'East Africa Youth Financial Inclusion Prediction Model',
    'dataset_size': len(data),
    'youth_focus': True,
    'countries': list(countries.index)
}

with open('../../model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Best model ({best_model_name}) saved to: ../../best_model.pkl")
print("✓ Scaler and encoders saved")
print("✓ Metadata saved")