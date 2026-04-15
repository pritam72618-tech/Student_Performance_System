# Student Performance Predictor

A Flask + Machine Learning web application that predicts student exam performance from academic and behavioral inputs, then provides actionable improvement suggestions.

## Why this project

This project demonstrates an end-to-end ML application workflow:
- Data-driven prediction with scikit-learn
- Production-style Flask inference pipeline
- Input validation and safe error handling
- Human-readable recommendations and scenario-based guidance
- Test coverage for validation and prediction behavior

## Key Features

- Quick prediction mode with core inputs:
  - Study hours per day
  - Attendance (%)
  - Recent average score
  - Sleep hours per day
  - Tutoring sessions per week
  - Motivation level
- Optional advanced inputs (family, school, and environment factors)
- Predicted score, grade, risk level, and expected score range
- Recommended next steps based on submitted profile
- "What could improve the score fastest" scenario simulation
- Server-side tamper-proof validation for numeric and categorical fields
- Deterministic model loading from saved artifacts

## Tech Stack

- Backend: Python, Flask
- ML/Data: scikit-learn, pandas, numpy, joblib
- Frontend: HTML + CSS (Jinja templates)
- Testing: Python `unittest`

## Project Structure

```text
Student_Performance_System/
├── app.py                               # Flask app + prediction logic
├── templates/
│   └── index.html                       # UI template
├── tests/
│   └── test_app.py                      # Validation + route + behavior tests
├── scripts/
│   ├── train_model.py                   # Model training pipeline
│   └── check_features.py                # Feature/artifact checks
├── models/
│   ├── student_model.pkl                # Trained model artifact
│   ├── model_features.pkl               # Feature list for inference
│   └── feature_importance.csv
├── StudentPerformance_ML_Ready_No_Gender.csv
├── requirements.txt
└── README.md
```

## Setup and Run

## 1) Clone the repository

```bash
git clone <your-repo-url>
cd Student_Performance_System
```

## 2) Create and activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Run the app

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## 5) Run tests

```bash
python -m unittest tests.test_app -v
```

## Model Training / Retraining

If you want to retrain artifacts from the dataset:

```bash
python scripts/train_model.py
```

This updates:
- `models/student_model.pkl`
- `models/model_features.pkl`

## Prediction Flow (High Level)

1. User submits form inputs.
2. Backend merges inputs with defaults and validates all fields.
3. Inputs are transformed into the exact feature schema expected by the model.
4. Model prediction is generated.
5. Final output is calibrated for practical relevance.
6. Grade, risk, recommendations, and improvement scenarios are rendered.

## Validation and Safety

- Numeric ranges are strictly validated server-side.
- Non-finite values (`NaN`, `inf`) are rejected.
- Categorical values are checked against allowed option sets.
- Errors return user-friendly messages without crashing the app.

## Current Limitations

- This is a guidance tool, not an official grading system.
- Output quality depends on training data quality and coverage.
- No authentication/database history in current version.

## Future Enhancements

- User login and prediction history
- Model explainability (SHAP/feature contribution UI)
- REST API for LMS/mobile integration
- Dataset versioning + retraining dashboard

## Resume Highlights (Project Impact)

- Built a production-style ML web app that converts student behavior signals into actionable performance predictions.
- Implemented robust validation and behavior tests to ensure reliable, tamper-resistant inference.
- Improved prediction relevance by adding calibration and scenario simulation for decision support.

## License

Add your preferred license (MIT/Apache-2.0/etc.) before publishing publicly.

