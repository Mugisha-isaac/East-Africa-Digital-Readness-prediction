# East Africa Youth Digital Readiness Predictor

## Mission & Problem Statement
This project creates a comprehensive platform for talented young individuals in East Africa to showcase their skills to the world by first identifying their digital readiness. We address the critical challenge of connecting gifted youth (aged 16-30) with global opportunities by predicting their digital preparedness through demographic and socioeconomic analysis. Our solution combines phone access, banking capabilities, and education levels to determine which youth are ready to demonstrate their talents on digital platforms, ultimately bridging the gap between East African talent and worldwide recognition across Rwanda, Tanzania, Kenya, and Uganda.

## 🚀 Live API Endpoint

**Base URL:** `https://alu-ml-summatives-latest.onrender.com`

### Available Endpoints:
- **Swagger UI Documentation:** https://alu-ml-summatives-latest.onrender.com/docs
- **Health Check:** `GET /health`
- **Model Info:** `GET /model/info`
- **Single Prediction:** `POST /predict`
- **Batch Predictions:** `POST /predict/users`

### API Testing
Use the **Swagger UI** at https://alu-ml-summatives-latest.onrender.com/docs to test all endpoints interactively. The API accepts 7 required parameters:
1. `location_type` (0=Rural, 1=Urban)
2. `household_size` (1-20)
3. `age_of_respondent` (16-30)
4. `gender_of_respondent` (0=Female, 1=Male)
5. `relationship_with_head` (0-5)
6. `marital_status` (0-4)
7. `job_type` (0-15)

## 📱 YouTube Demo Video
**Watch the complete demo:** [East Africa Youth Talent Platform Demo](https://youtu.be/your-video-id)

*5-minute demonstration covering API functionality, Flutter app usage, and how to identify talent-ready youth*

## 📱 Mobile App Setup & Instructions

### Prerequisites
- Flutter SDK (3.1.0 or higher)
- Android Studio or VS Code with Flutter extensions
- Android device/emulator or iOS simulator

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/east-africa-digital-readiness.git
   cd east-africa-digital-readiness/summative/FlutterApp/youth_digital_readness_predicting_app
   ```

2. **Install Flutter dependencies:**
   ```bash
   flutter pub get
   ```

3. **Verify Flutter installation:**
   ```bash
   flutter doctor
   ```

4. **Run the app:**
   ```bash
   # For Android
   flutter run
   
   # For specific device
   flutter run -d <device-id>
   
   # For release build
   flutter run --release
   ```

### App Features
- **Home Screen:** API status check and platform navigation
- **Talent Assessment:** Evaluate individual youth profiles for digital talent showcase readiness
- **Batch Evaluation:** Process multiple talented individuals simultaneously (up to 100)
- **Readiness Results:** Comprehensive assessment with confidence scores for platform participation
- **Real-time Integration:** Direct connection to deployed talent assessment model

### Usage Instructions
1. **Launch the app** and verify API connection on the home screen
2. **Single Talent Assessment:**
   - Tap "Single User Prediction"
   - Fill all 7 required fields (age, location, household size, gender, relationship, marital status, job type)
   - Tap "Predict Digital Readiness"
   - View results showing readiness to showcase talents digitally

3. **Batch Talent Evaluation:**
   - Tap "Batch Prediction"
   - Add multiple talented youth profiles using the "+" button
   - Tap "Predict All" to assess all candidates
   - Review summary statistics and individual talent platform readiness

### Troubleshooting
- **API Connection Issues:** Ensure stable internet connection
- **Build Errors:** Run `flutter clean` then `flutter pub get`
- **Device Issues:** Check `flutter devices` for available targets

### Technical Architecture
- **Frontend:** Flutter (Dart) - Talent showcase platform interface
- **Backend:** FastAPI (Python) - Talent assessment engine
- **ML Model:** Random Forest with 0.1115 R² score for digital readiness
- **Deployment:** Render (Docker containerized)
- **Data:** East Africa Financial Inclusion Survey (23,524+ talented youth samples)

### Model Performance
- **Target:** Digital readiness for talent showcase (phone + bank + education combined)
- **Countries:** Rwanda, Tanzania, Kenya, Uganda
- **Youth Focus:** Ages 16-30 years (prime talent years)
- **Accuracy:** 88%+ prediction confidence for high readiness cases

## 🔧 Development

### API Development
```bash
cd summative/API
pip install -r requirements.txt
python main.py
```

### Model Training
```bash
cd summative/linear_regression
python multivariate.py
```

### Docker Deployment
```bash
cd summative/API
docker build -t talent-readiness-api .
docker run -p 8000:8000 talent-readiness-api
```

## 📊 Project Structure
```
summative/
├── API/                          # FastAPI backend for talent assessment
│   ├── main.py                  # Talent platform API
│   ├── Dockerfile              # Container configuration
│   └── requirements.txt        # Python dependencies
├── FlutterApp/                  # Mobile talent showcase app
│   └── youth_digital_readness_predicting_app/
│       ├── lib/                # Flutter source code
│       └── pubspec.yaml        # Flutter dependencies
├── linear_regression/           # ML model for talent readiness
│   └── multivariate.py         # Talent assessment model
└── model files/                # Trained talent assessment artifacts
    ├── best_model.pkl
    ├── scaler.pkl
    └── encoders.pkl
```

## 🎯 Talent Platform Impact & Applications
- **Global Talent Showcase:** Connect East African youth talents with worldwide opportunities
- **Digital Talent Gateway:** Identify youth ready to demonstrate skills on digital platforms
- **Skill Recognition:** Bridge the gap between hidden talents and global recognition
- **Economic Empowerment:** Enable talented youth to monetize their skills internationally
- **Cultural Exchange:** Facilitate sharing of East African creativity and innovation globally
- **Opportunity Matching:** Connect digitally-ready talented youth with international collaborators

### Talent Categories Supported
- **Creative Arts:** Digital artists, musicians, writers, designers
- **Technology:** Developers, innovators, digital creators
- **Business:** Young entrepreneurs and business talents
- **Education:** Teaching and knowledge-sharing talents
- **Social Impact:** Community leaders and change-makers

---

**Built with ❤️ to Showcase East African Youth Talent to the World**
**Empowering Talented Individuals • Connecting Cultures • Creating Opportunities**
