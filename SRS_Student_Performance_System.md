# Software Requirements Specification (SRS)
## Project: Student Performance Prediction System
## Version: 1.0
## Date: April 2, 2026

## 1. Introduction
### 1.1 Purpose
This document specifies the functional and non-functional requirements of the **Student Performance Prediction System**, a Flask-based web application that predicts a student's exam score using academic and behavioral inputs and provides improvement suggestions.

### 1.2 Scope
The system allows users to:
- Enter student-related input features.
- Generate a predicted exam score using a trained machine learning model.
- View derived outputs such as grade, risk level, recommendations, and quick improvement scenarios.

The system is intended for academic demonstration and learning support, not for high-stakes official grading decisions.

### 1.3 Definitions
- **Prediction**: Estimated exam score output by the ML model.
- **Risk Level**: Categorization of student performance risk (High/Medium/Low).
- **Model Artifacts**: Saved files needed for inference (`student_model.pkl`, `model_features.pkl`).

## 2. Overall Description
### 2.1 Product Perspective
The product is a standalone Flask web app with:
- Frontend: HTML/CSS form and results page (`templates/index.html`)
- Backend: Input validation and prediction logic (`app.py`)
- ML Pipeline: Model training script (`scripts/train_model.py`)

### 2.2 User Class
- Primary user: Faculty/students demonstrating prediction workflow.
- Technical user: Developer running training, testing, and deployment.

### 2.3 Operating Environment
- OS: Windows/Linux/Mac (Python supported)
- Language: Python 3.x
- Framework: Flask
- Libraries: pandas, scikit-learn, joblib, numpy
- Browser: Any modern browser (Chrome, Edge, Firefox)

### 2.4 Assumptions and Dependencies
- Required dataset exists: `StudentPerformance_ML_Ready_No_Gender.csv`
- Model artifacts are present in `/models` or project root.
- Python dependencies are installed via `requirements.txt`.

## 3. Functional Requirements
### FR-1: Home Page Display
System shall display an input form with core and advanced fields for student details.

### FR-2: Input Collection
System shall accept numeric and categorical values including:
- Numeric: study hours, attendance, sleep, previous scores, tutoring, physical activity
- Categorical: motivation, parental involvement, internet access, etc.

### FR-3: Input Validation
System shall:
- Validate numeric ranges (e.g., attendance 0-100).
- Reject invalid or non-finite numeric values.
- Validate categorical values against predefined allowed options.
- Return user-friendly error messages for invalid input.

### FR-4: Prediction
System shall:
- Load model and feature artifacts.
- Transform input into expected model feature format.
- Generate predicted score.

### FR-5: Result Presentation
System shall display:
- Predicted score
- Grade (A/B/C/D/F)
- Risk level (High/Medium/Low)
- Expected score range

### FR-6: Recommendations
System shall provide up to 3 practical recommendations based on submitted profile.

### FR-7: Improvement Scenarios
System shall simulate selected changes (e.g., higher attendance, study hours) and show possible score gains.

### FR-8: Error Handling
System shall gracefully handle missing model files or runtime errors and show a friendly failure message.

## 4. Non-Functional Requirements
### NFR-1: Usability
- Interface shall be simple and responsive for desktop and mobile.
- Core fields should support quick submission in under one minute.

### NFR-2: Performance
- Prediction response should generally complete within 2 seconds on local machine after model load.

### NFR-3: Reliability
- System should return deterministic output for same input and same model artifact.
- Invalid input should not crash the server.

### NFR-4: Maintainability
- Code should be modular with helper functions for validation, feature-building, and recommendations.
- Project should include dependency and test support (`requirements.txt`, `tests/`).

### NFR-5: Security (Basic)
- Server-side input validation must be enforced even if frontend input is tampered.
- Debug mode should not be enabled by default in production use.

## 5. External Interface Requirements
### 5.1 User Interface
- Single-page form-based interface with result panels.
- Action button: “Predict and Suggest Improvements”.

### 5.2 Software Interface
- Flask route: `GET /` (form display)
- Flask route: `POST /predict` (prediction + output)

### 5.3 File Interface
- Input dataset: `StudentPerformance_ML_Ready_No_Gender.csv`
- Model files: `models/student_model.pkl`, `models/model_features.pkl`

## 6. Data Requirements
### 6.1 Input Data
Structured numeric/categorical values submitted via HTML form.

### 6.2 Output Data
- Numeric prediction (score)
- Text labels (grade, risk)
- List of recommendations and scenario outputs

### 6.3 Persistence
The current app does not store user submissions in a database.

## 7. Constraints
- Quality and fairness depend on the training dataset.
- Predictions are approximate and should be used as guidance.
- No authentication/authorization layer in current version.

## 8. Test and Acceptance Criteria
System is accepted when:
- App runs successfully via Flask server.
- Valid inputs return prediction page with score, grade, and risk.
- Invalid numeric/categorical inputs return proper 400 error with message.
- Automated tests in `tests/test_app.py` pass.

## 9. Future Enhancements (Optional)
- Add user login and history tracking.
- Add model confidence score and explainability charts.
- Add API endpoint for integration with mobile or LMS systems.
- Add retraining dashboard and dataset version tracking.
