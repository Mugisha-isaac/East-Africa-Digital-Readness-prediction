# East Africa Youth Digital Readiness Prediction API

A FastAPI-based REST API that predicts digital readiness (phone access + bank account + education) for youth in East Africa, designed for job creation platform development.

## 🌍 Overview

This API uses a machine learning model trained on the East Africa Financial Inclusion Survey data to predict the likelihood that a youth (aged 16-30) has combined access to:
- Mobile phone/cellphone
- Bank account 
- Education (Primary, Secondary, or Tertiary)

**Countries covered:** Rwanda, Tanzania, Kenya, Uganda

## 🚀 Key Features

- **🎯 Primary Endpoint**: `/predict/users` - Accept array of users for batch predictions
- **📊 Single Prediction**: `/predict` - Predict for one user profile
- **🔄 Batch Processing**: Handle up to 100 users in a single request
- **✅ Data Validation**: Strict input validation with realistic constraints
- **🌐 CORS Support**: Cross-origin requests enabled
- **📚 Auto Documentation**: Swagger UI and OpenAPI specification
- **💚 Health Monitoring**: Health check endpoints
- **📈 Detailed Analytics**: Comprehensive prediction summaries

## 📋 Requirements

```
fastapi
uvicorn[standard]
pydantic
numpy
scikit-learn
joblib
python-multipart
```

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files are in the correct location:**
   - `../../best_model.pkl`
   - `../../scaler.pkl`
   - `../../encoders.pkl`
   - `../../model_metadata.json`

## 🏃‍♂️ Running the API

### Local Development
```bash
python main.py
```
or
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📖 API Documentation

Once running, visit these URLs:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`
- **Swagger File**: `http://localhost:8000/swagger.yaml`
- **Test Interface**: Open `test_interface.html` in your browser

## 🎯 Primary Endpoint: Array Input

### POST `/predict/users`

This is the **main endpoint** designed to handle multiple users in a single request.

**Request Body:**
```json
{
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
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "digital_readiness_score": 0.6523,
      "digital_readiness_percentage": 65.23,
      "readiness_category": "Moderate Digital Readiness",
      "recommendation": "Good potential for digital job platform...",
      "input_summary": {...},
      "model_info": {...}
    }
  ],
  "summary": {
    "total_predictions": 2,
    "average_score": 0.5241,
    "score_distribution": {
      "high_readiness": 0,
      "moderate_readiness": 1,
      "low_readiness": 1,
      "very_low_readiness": 0
    },
    "highest_score": 0.6523,
    "lowest_score": 0.3959,
    "readiness_insights": {
      "ready_for_platform": 1,
      "need_support": 1,
      "average_readiness_level": "Moderate Digital Readiness"
    }
  }
}
```

## 🎯 All Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Health check and model status |
| GET | `/model/info` | Model performance and feature information |
| GET | `/swagger.yaml` | OpenAPI specification file |
| **POST** | **`/predict/users`** | **Main endpoint: Predict for multiple users** |
| POST | `/predict` | Single user prediction |
| POST | `/predict/batch` | Alternative batch prediction |

## 📊 Input Parameters

All predictions require these 7 features:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `location_type` | int | 0-1 | 0 = Rural, 1 = Urban |
| `household_size` | int | 1-20 | Number of people in household |
| `age_of_respondent` | int | 16-30 | Age (youth focus) |
| `gender_of_respondent` | int | 0-1 | 0 = Female, 1 = Male |
| `relationship_with_head` | int | 0-5 | Relationship to household head |
| `marital_status` | int | 0-4 | Marital status |
| `job_type` | int | 0-15 | Type of job/occupation |

## 💡 Usage Examples

### cURL - Multiple Users
```bash
curl -X POST "http://localhost:8000/predict/users" \
     -H "Content-Type: application/json" \
     -d '{
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
         }
       ]
     }'
```

### Python - Multiple Users
```python
import requests

url = "http://localhost:8000/predict/users"
data = {
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
        }
    ]
}

response = requests.post(url, json=data)
result = response.json()

print(f"Total predictions: {result['summary']['total_predictions']}")
print(f"Average readiness: {result['summary']['average_score']:.3f}")

for i, pred in enumerate(result['predictions']):
    print(f"User {i+1}: {pred['digital_readiness_percentage']}% - {pred['readiness_category']}")
```

## 🎯 Interpretation Guide

### Digital Readiness Categories

| Score Range | Category | Interpretation |
|-------------|----------|----------------|
| 0.7 - 1.0 | High Digital Readiness | Excellent candidate for digital job platform |
| 0.4 - 0.69 | Moderate Digital Readiness | Good potential, may need some digital skills training |
| 0.2 - 0.39 | Low Digital Readiness | Requires comprehensive digital literacy programs |
| 0.0 - 0.19 | Very Low Digital Readiness | Needs extensive digital inclusion support |

## 🔍 Testing

### Option 1: Interactive HTML Interface
Open `test_interface.html` in your browser for a user-friendly testing interface.

### Option 2: Python Test Script
```bash
python test_users_api.py
```

### Option 3: Manual Testing
1. Start the API: `python main.py`
2. Visit: `http://localhost:8000/docs`
3. Use the interactive Swagger UI

## 🚀 Deployment to Render

1. **Create a new Web Service on Render**
2. **Connect your repository**
3. **Configure build settings:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Add environment variables if needed**
5. **Deploy**

Your API will be available at: `https://your-app-name.onrender.com`

### Example Production URLs:
- API Docs: `https://your-app-name.onrender.com/docs`
- Predictions: `https://your-app-name.onrender.com/predict/users`
- Health Check: `https://your-app-name.onrender.com/health`

## 📊 Model Information

- **Model Type**: Random Forest Regressor
- **Training Data**: East Africa Financial Inclusion Survey (23,524+ samples)
- **Target**: Digital Readiness (combination of phone + bank + education)
- **Performance**: R² score of ~0.11 (baseline model)
- **Countries**: Rwanda, Tanzania, Kenya, Uganda
- **Focus**: Youth aged 16-30

## 🛡️ Error Handling

The API includes comprehensive error handling:
- **422**: Validation errors (invalid input data)
- **404**: Endpoint not found
- **500**: Internal server errors (model loading issues)

All errors return detailed JSON responses with error descriptions.

## 📝 Files Structure

```
API/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── swagger.yaml           # OpenAPI specification
├── test_users_api.py      # Python test script
├── test_interface.html    # Interactive test interface
├── examples.py           # Usage examples
├── Dockerfile            # Docker configuration
├── start.sh             # Startup script
└── README.md            # This file
```

## 📧 Support

For questions about the model or API implementation, refer to the original analysis code and documentation.

---

**Digital Readiness = Phone Access + Bank Account + Education**  
*Perfect for identifying youth ready for digital job platforms!*

### 🔗 Quick Links
- 📚 **API Docs**: `http://localhost:8000/docs`
- 📄 **Swagger**: `http://localhost:8000/swagger.yaml`
- 🧪 **Test Interface**: Open `test_interface.html`
- 💚 **Health Check**: `http://localhost:8000/health`
