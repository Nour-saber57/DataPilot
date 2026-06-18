# 🤖 AutoML Studio - Complete Setup & Quick Start Guide

## 📋 Overview

**AutoML Studio** is a complete end-to-end machine learning pipeline with:
- **FastAPI Backend**: Data preprocessing + Model training with GridSearchCV
- **Streamlit Frontend**: User-friendly interface for model training and comparison
- **Backend Integration**: Leverages `preprocessor.py` and `model_registry.py`

---

## 🚀 Quick Start (3 Steps)

### Step 1: Navigate to Data_Pilot Directory
```bash
cd d:\NTI\NTI-project\Data_Pilot
```

### Step 2: Run Both Applications
```bash
# Option A: Using Python directly
d:\NTI\NTI-project\Data_Pilot\venv\Scripts\python.exe run_automl.py

# Option B: Run manually (2 terminals)
# Terminal 1 - FastAPI Backend:
d:\NTI\NTI-project\Data_Pilot\venv\Scripts\python.exe -m uvicorn FastAPI.main:app --reload --port 8000

# Terminal 2 - Streamlit Frontend:
d:\NTI\NTI-project\Data_Pilot\venv\Scripts\streamlit run Streamlit/automl_studio.py --server.port 8501
```

### Step 3: Open Browser
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## 📊 Using AutoML Studio

### Tab 1: Data Upload
1. Click **"Choose a CSV file"**
2. Select your dataset
3. Choose the **target column** from dropdown
4. Click **"Upload & Analyze"**
5. View data analysis, features, and missing values

### Tab 2: Model Training

#### Option A: Train Single Model
1. Ensure data is uploaded
2. Click **"Refresh Available Models"** to see all models
3. Select model from dropdown
4. Click **"Train Model"**
5. View performance metrics and best hyperparameters

#### Option B: Train All Models (Recommended)
1. Ensure data is uploaded
2. Select **"Train All"** radio button
3. Click **"Train All Models"**
4. View leaderboard with all models ranked by performance
5. See which model performs best on your data

### Tab 3: Results
1. Click **"Refresh Results"**
2. Expand each model to see:
   - Performance metrics (Accuracy, F1, MSE, R², etc.)
   - Best hyperparameters found by GridSearchCV
   - Training details

### Tab 4: API Details
- View backend status
- See all available API endpoints
- Access interactive Swagger documentation

---

## 📁 File Structure

```
Data_Pilot/
├── FastAPI/
│   ├── main.py ..................... FastAPI backend (NEW)
│   ├── routes/
│   └── schemas/
├── Streamlit/
│   ├── automl_studio.py ............ Streamlit frontend (NEW)
│   └── ...
├── Backend/
│   ├── model_registry.py ........... ML models with GridSearchCV
│   ├── preprocessor.py ............. Data preprocessing
│   ├── test_models.py .............. Testing utilities
│   └── ...
├── run_automl.py ................... Startup script (NEW)
└── venv/ ........................... Virtual environment
```

---

## 🔌 API Endpoints Reference

### 1. Health Check
```bash
GET http://localhost:8000/
```
Response:
```json
{
  "status": "success",
  "message": "AutoML Backend API is running",
  "version": "1.0.0"
}
```

### 2. Upload Data
```bash
POST http://localhost:8000/upload-data?target_column=TARGET
```
- Accepts: CSV file
- Returns: Data analysis, features, task type, missing values

### 3. Get Available Models
```bash
GET http://localhost:8000/available-models?task=classification
```
- Returns: List of all available models for the task

### 4. Train Single Model
```bash
POST http://localhost:8000/train-model?model_name=random_forest&target_column=TARGET&test_size=0.2
```
- Returns: Best parameters, metrics, training details

### 5. Train All Models
```bash
POST http://localhost:8000/train-all-models?target_column=TARGET&test_size=0.2
```
- Returns: Leaderboard, all model results, rankings

### 6. Get Results
```bash
GET http://localhost:8000/results?model_name=OPTIONAL
```
- Returns: Training results for specified or all models

---

## 🎓 Available Models

### Classification (7 models)
- Logistic Regression (with L1/L2 penalties)
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Gradient Boosting Classifier
- Naive Bayes

### Regression (7 models)
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Support Vector Machine Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

---

## ⚙️ Hyperparameter Tuning

Each model uses **GridSearchCV with 5-fold cross-validation** to find optimal hyperparameters:

### Example: Random Forest
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
# Total combinations: 3 × 4 × 3 × 3 × 2 = 216 parameter sets
# GridSearchCV tests each with 5-fold CV = 1,080 model trainings!
```

---

## 📊 Example Workflow

### Scenario: Predict Loan Approval

**Step 1: Prepare Data**
```csv
age,income,credit_score,employment_years,marital_status,education,loan_approved
35,50000,720,5,Married,Bachelor,1
28,35000,650,2,Single,High School,0
45,75000,800,10,Married,Master,1
```

**Step 2: Upload to AutoML Studio**
- Open http://localhost:8501
- Upload CSV file
- Select "loan_approved" as target

**Step 3: Train Models**
- Click "Train All Models"
- Wait for training to complete
- View leaderboard

**Example Results:**
```
🏆 Model Leaderboard
Rank | Model           | Score
-----|-----------------|------
1    | Random Forest   | 0.9423
2    | Gradient Boost  | 0.9387
3    | SVM             | 0.9245
4    | Logistic Reg    | 0.8956
5    | KNN             | 0.8734
```

**Step 4: Use Best Model**
- Best model: Random Forest with 94.23% F1-Score
- Use the model in production via API endpoint

---

## 🔧 Troubleshooting

### Backend Not Connecting
```
Error: Connection error: Cannot connect to http://localhost:8000
```
**Solution:**
1. Make sure FastAPI is running: `python -m uvicorn FastAPI.main:app --reload --port 8000`
2. Check if port 8000 is not in use
3. Wait 3-5 seconds after starting backend before opening Streamlit

### Port Already in Use
```
OSError: [WinError 10048] Only one usage of each socket address
```
**Solution:**
```bash
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### CSV File Error
**Error:** "Target column not found"
**Solution:** Make sure the target column name matches exactly (case-sensitive)

---

## 📈 Performance Tips

1. **Use clean data**: Preprocess before uploading
2. **Smaller test size**: Use 0.1-0.15 for faster training
3. **Start with single model**: Test with one model first
4. **Check API docs**: Visit http://localhost:8000/docs for debugging

---

## 🚀 Next Steps

1. **Train on your data**: Use the Streamlit interface
2. **Export best model**: Save trained model for production
3. **Create predictions**: Build prediction endpoint
4. **Monitor performance**: Track model metrics over time

---

## 📚 Additional Resources

- **FastAPI Docs**: http://localhost:8000/docs
- **Model Registry Guide**: `Backend/MODEL_REGISTRY_GUIDE.md`
- **Preprocessor Details**: See `Backend/preprocessor.py`
- **Backend Implementation**: See `FastAPI/main.py`

---

## ✅ Checklist

- [ ] Virtual environment activated
- [ ] FastAPI backend running on port 8000
- [ ] Streamlit frontend running on port 8501
- [ ] CSV data file ready
- [ ] Target column identified
- [ ] Models trained and compared
- [ ] Best model identified

---

## 💡 Pro Tips

✓ Always check "Data Preview" to understand your dataset
✓ Use "Train All Models" first to find the best performer
✓ GridSearchCV finds optimal hyperparameters automatically
✓ Visit API docs at http://localhost:8000/docs while training
✓ Results are cached, so you can refresh without retraining

---

## 🎉 You're All Set!

Your complete AutoML pipeline is ready to use. Happy modeling! 🚀
