# CardioSense — Heart Disease Detection System

A full-stack Machine Learning web app that estimates heart disease risk from
patient vitals, built with **Flask + scikit-learn** on the backend and a
custom **HTML/CSS/JS dashboard** on the frontend.

## 📖 Brief Project Description

The Heart Disease Detection System is a machine learning-based web application that predicts the likelihood of heart disease using patient medical information. The project provides an intuitive web interface built with HTML, CSS, and JavaScript, while the backend is developed using Flask.

The machine learning model is trained from scratch using the Heart Disease dataset with Scikit-learn. During training, the dataset is preprocessed, split into training and testing sets, and used to train a classification model. The trained model is then saved and integrated into the Flask application to provide real-time predictions for new patient data.

This project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, model serialization, backend integration, and frontend development.

# 🔗 GitHub Repository

**Repository Link:** *https://github.com/Harmankaur07/smart_image_annotation/upload*

# 🔗 Deployment link

# 📌 Project Overview
The project includes a complete machine learning pipeline, including data preprocessing, model training, evaluation, and deployment. Multiple classification algorithms are trained and compared to identify the best-performing model, which is then saved and integrated into the Flask application for real-time predictions.

The application is built using Python, Flask, and scikit-learn for the backend, while the frontend is developed with HTML, CSS, and JavaScript to provide a clean and responsive user experience.

## Features

- Trains and compares 6 ML algorithms (Logistic Regression, Random Forest,
  Decision Tree, SVM, KNN, XGBoost/Gradient Boosting) and auto-selects the
  best one by accuracy
- Data preprocessing: missing-value imputation, label encoding, feature
  scaling (`StandardScaler`)
- Saves `heart_model.pkl`, `scaler.pkl`, confusion matrix, accuracy
  comparison chart, and a classification report
- REST API (`/api/predict`, `/api/history`, `/api/search`, `/api/stats`)
- Responsive dashboard UI with dark/light mode, animated ECG loading
  indicator, confidence ring, patient summary card, and toast notifications
- Prediction history stored in SQLite, searchable from the UI
- Dashboard charts: outcome pie chart + model accuracy bar chart (Chart.js)
- One-click PDF report export (jsPDF) — no server round-trip needed

# 🫀 System Workflow

```text
Patient Enters Health Details
            │
            ▼
Input Validation
            │
            ▼
Data Preprocessing
(Missing Value Handling & Feature Scaling)
            │
            ▼
Load Trained ML Model
            │
            ▼
Heart Disease Prediction
            │
            ▼
Risk Classification
(Low Risk / High Risk)
            │
            ▼
Confidence Score Calculation
            │
            ▼
Store Prediction in SQLite Database
            │
            ▼
Generate Health Recommendation
            │
            ▼
Display Prediction Results
```

## Project structure

```
heart_disease_system/
├── app.py                     # Flask REST API + page routes
├── train_model.py             # ML training pipeline
├── requirements.txt
├── data/
│   ├── generate_dataset.py    # builds a realistic UCI-style dataset (offline)
│   └── heart.csv
├── models/
│   ├── heart_model.pkl        # best trained model (saved via joblib)
│   ├── scaler.pkl             # fitted StandardScaler
│   └── model_metadata.json    # feature order + accuracy comparison
├── plots/
│   ├── confusion_matrix.png
│   ├── accuracy_comparison.png
│   └── classification_report.txt
├── database/
│   └── predictions.db         # SQLite (auto-created on first run)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js
```
# 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.x |
| Backend Framework | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Model Serialization | Pickle |
| Database | SQLite |
| Development Environment | Visual Studio Code |
| Version Control | Git & GitHub |
| Package Management | pip |
| API | Flask REST API |

## Setup

```bash
cd heart_disease_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. About the dataset

This project ships with `data/generate_dataset.py`, which synthesizes a
1,000-row dataset that follows the exact schema and realistic feature/target
correlations of the classic **UCI Cleveland Heart Disease dataset** (same 13
features + target). It was generated locally because this environment has
no internet access to download the original file.

**To use the real UCI dataset instead:** download `heart.csv` (13 features +
`target` column, same column names as below) from the UCI Machine Learning
Repository or Kaggle ("Heart Disease UCI Dataset") and drop it into
`data/heart.csv`, replacing the generated one. No other code changes needed.

Columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

### 2. Train the model

```bash
python train_model.py
```

This will print accuracy/precision/recall/F1 for every model, save the best
one to `models/heart_model.pkl`, and write the plots to `plots/`.

### 3. Run the app

```bash
python app.py
```

Open http://192.168.9.202:5000 in your browser.

# 🚀 How to Use

1. **Start the Flask application**

   Run the following command:

   ```bash
   python app.py
   ```

2. **Open the web application**

   Open your browser and go to:

   ```
   http://192.168.9.202:5000

3. **Enter Patient Information**

   Fill in all the required medical details, such as:
   - Age
   - Gender
   - Chest Pain Type
   - Resting Blood Pressure
   - Cholesterol
   - Fasting Blood Sugar
   - Resting ECG
   - Maximum Heart Rate
   - Exercise-Induced Angina
   - ST Depression (Oldpeak)
   - Slope
   - Number of Major Vessels (CA)
   - Thalassemia

4. **Submit the Form**

   Click the **Predict** button to send the patient data to the machine learning model.

5. **View Prediction Results**

   The application will display:
   - Heart disease prediction (Low Risk or High Risk)
   - Prediction confidence score
   - Personalized health recommendation

6. **Prediction Storage**

   Each prediction is automatically saved to the SQLite database for future reference and analysis.

7. **Repeat Predictions**

   Return to the home page and enter new patient details to perform additional predictions.

## API reference

| Method | Endpoint            | Description                          |
|--------|----------------------|---------------------------------------|
| POST   | `/api/predict`       | Predict risk from patient data (JSON) |
| GET    | `/api/history?limit=`| Recent predictions                    |
| GET    | `/api/search?q=`     | Search history by name/risk/date      |
| GET    | `/api/stats`         | Dashboard stats + model accuracies    |
| DELETE | `/api/delete/<id>`   | Remove a prediction record            |

### Example `/api/predict` request body

```json
{
  "patient_name": "John Doe",
  "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
  "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 2
}
```

## Retraining with new data / algorithms

Edit the `models` dictionary in `train_model.py` to add/remove algorithms,
then rerun `python train_model.py`. The Flask app automatically reloads
whichever model scored highest, via `models/model_metadata.json`.

# 💻 Developer

**Harman Kaur**

**B.Tech – Computer Science & Engineering**

**Summer Training Project – 2026**


## Disclaimer

This tool is a machine-learning **screening aid** for educational purposes
only. It is **not** a certified medical device and must not be used as a
substitute for professional diagnosis. Always consult a licensed physician.
