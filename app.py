from functools import lru_cache
import math
import os
from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

NUMERIC_FIELDS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

CATEGORICAL_FIELDS = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Peer_Influence",
    "Family_Income",
    "Teacher_Quality",
    "School_Type",
    "Distance_from_Home",
    "Parental_Education_Level",
    "Internet_Access",
    "Learning_Disabilities",
    "Extracurricular_Activities",
]

CATEGORICAL_OPTIONS = {
    "Parental_Involvement": {"Low", "Medium", "High"},
    "Access_to_Resources": {"Low", "Medium", "High"},
    "Motivation_Level": {"Low", "Medium", "High"},
    "Peer_Influence": {"Negative", "Neutral", "Positive"},
    "Family_Income": {"Low", "Medium", "High"},
    "Teacher_Quality": {"Low", "Medium", "High"},
    "School_Type": {"Public", "Private"},
    "Distance_from_Home": {"Near", "Moderate", "Far"},
    "Parental_Education_Level": {"High School", "Undergraduate", "Postgraduate"},
    "Internet_Access": {"Yes", "No"},
    "Learning_Disabilities": {"Yes", "No"},
    "Extracurricular_Activities": {"Yes", "No"},
}

FIELD_DEFAULTS = {
    "Hours_Studied": "2.0",
    "Attendance": "80",
    "Sleep_Hours": "7",
    "Previous_Scores": "70",
    "Tutoring_Sessions": "1",
    "Physical_Activity": "3",
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Motivation_Level": "Medium",
    "Peer_Influence": "Neutral",
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Distance_from_Home": "Moderate",
    "Parental_Education_Level": "Undergraduate",
    "Internet_Access": "Yes",
    "Learning_Disabilities": "No",
    "Extracurricular_Activities": "Yes",
}

FIELD_LABELS = {
    "Hours_Studied": "Study Hours Per Day",
    "Attendance": "Attendance",
    "Sleep_Hours": "Sleep Hours Per Day",
    "Previous_Scores": "Recent Average Score",
    "Tutoring_Sessions": "Tutoring Sessions Per Week",
    "Physical_Activity": "Physical Activity Hours Per Week",
}


def _resolve_artifact(filename: str) -> Path:
    candidates = [
        BASE_DIR / "models" / filename,
        BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing required artifact: {filename}. Looked in {candidates[0]} and {candidates[1]}"
    )


@lru_cache(maxsize=1)
def _load_artifacts():
    model = joblib.load(_resolve_artifact("student_model.pkl"))
    model_features = joblib.load(_resolve_artifact("model_features.pkl"))
    return model, model_features


def _merge_form_with_defaults(form_data):
    values = dict(FIELD_DEFAULTS)
    for field in NUMERIC_FIELDS + CATEGORICAL_FIELDS:
        raw_value = form_data.get(field)
        if raw_value is not None and raw_value.strip():
            values[field] = raw_value.strip()
    return values


def _build_model_input(values, model_features):
    # Initialize all expected model columns to zero to avoid missing-column errors.
    row = {feature: 0.0 for feature in model_features}

    # Fill numeric values directly if the column exists in the trained feature space.
    for field in NUMERIC_FIELDS:
        if field not in row:
            continue
        raw_value = values.get(field, "").strip()
        row[field] = float(raw_value) if raw_value else 0.0

    # Dynamically map each categorical selection to "<feature>_<selected_value>".
    # With drop_first=True, baseline categories do not exist as columns; those stay 0.
    for field in CATEGORICAL_FIELDS:
        selected_value = values.get(field, "").strip()
        if not selected_value:
            continue

        encoded_column = f"{field}_{selected_value}"
        if encoded_column in row:
            row[encoded_column] = 1.0

    # Enforce exact training-time column order before prediction.
    return pd.DataFrame([[row[col] for col in model_features]], columns=model_features)


def calculate_grade(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def calculate_risk(score):
    if score >= 80:
        return "Low"
    elif score >= 60:
        return "Medium"
    else:
        return "High"


def _build_recommendations(values):
    hours = float(values["Hours_Studied"])
    attendance = float(values["Attendance"])
    previous_scores = float(values["Previous_Scores"])
    sleep_hours = float(values["Sleep_Hours"])
    tutoring_sessions = float(values["Tutoring_Sessions"])
    physical_activity = float(values["Physical_Activity"])
    motivation = values["Motivation_Level"]

    recommendations = []

    if attendance < 85:
        recommendations.append("Raise attendance above 85% to improve consistency and expected score.")
    if hours < 2.5:
        recommendations.append("Increase study time to at least 2.5 to 3 hours per day.")
    if previous_scores < 75:
        recommendations.append("Focus on revision of weak topics from recent tests before adding new material.")
    if sleep_hours < 7:
        recommendations.append("Aim for at least 7 hours of sleep to support concentration and retention.")
    if tutoring_sessions < 1 and previous_scores < 70:
        recommendations.append("Add one tutoring or doubt-clearing session per week.")
    if physical_activity > 10:
        recommendations.append("Balance physical activity with study time if weekly activity is reducing study hours.")
    elif physical_activity < 2:
        recommendations.append("Add light physical activity during the week to support energy and focus.")
    if motivation == "Low":
        recommendations.append("Set a small daily study target and review progress weekly to improve motivation.")

    if not recommendations:
        recommendations.append("Current habits are strong. Maintain consistency and focus on mock-test practice.")

    return recommendations[:3]


def _build_improvement_scenarios(values, model, model_features, current_prediction):
    scenario_specs = [
        ("Attendance", min(float(values["Attendance"]) + 10, 100), "Improve attendance"),
        ("Hours_Studied", min(float(values["Hours_Studied"]) + 1, 6), "Study 1 more hour per day"),
        ("Sleep_Hours", min(float(values["Sleep_Hours"]) + 1, 9), "Sleep 1 more hour per day"),
        ("Tutoring_Sessions", min(float(values["Tutoring_Sessions"]) + 1, 4), "Add 1 tutoring session per week"),
    ]

    scenarios = []
    for field, new_value, label in scenario_specs:
        if float(values[field]) == new_value:
            continue
        scenario_values = dict(values)
        scenario_values[field] = str(new_value)
        scenario_df = _build_model_input(scenario_values, model_features)
        scenario_prediction = float(model.predict(scenario_df)[0])
        scenarios.append(
            {
                "label": label,
                "new_value": new_value,
                "score": round(scenario_prediction, 2),
                "gain": round(scenario_prediction - current_prediction, 2),
                "field_label": FIELD_LABELS.get(field, field),
            }
        )

    scenarios.sort(key=lambda item: item["gain"], reverse=True)
    return [item for item in scenarios if item["gain"] > 0][:3]


def _validate_values(values):
    numeric_ranges = {
        "Hours_Studied": (0, 24),
        "Attendance": (0, 100),
        "Sleep_Hours": (0, 24),
        "Previous_Scores": (0, 100),
        "Tutoring_Sessions": (0, 20),
        "Physical_Activity": (0, 40),
    }

    cleaned = dict(values)
    for field, (min_value, max_value) in numeric_ranges.items():
        try:
            value = float(cleaned[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FIELD_LABELS.get(field, field)} must be a valid number.") from exc

        if not math.isfinite(value):
            raise ValueError(f"{FIELD_LABELS.get(field, field)} must be a finite number.")
        if value < min_value or value > max_value:
            raise ValueError(f"{FIELD_LABELS.get(field, field)} must be between {min_value} and {max_value}.")
        cleaned[field] = str(value)

    for field, allowed_values in CATEGORICAL_OPTIONS.items():
        selected_value = cleaned.get(field, "").strip()
        if selected_value not in allowed_values:
            raise ValueError(f"{field.replace('_', ' ')} must be one of: {', '.join(sorted(allowed_values))}.")

    return cleaned

@app.route("/")
def home():
    return render_template("index.html", values=FIELD_DEFAULTS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, model_features = _load_artifacts()
        values = _merge_form_with_defaults(request.form)
        values = _validate_values(values)
        input_df = _build_model_input(values, model_features)

        predicted_score = float(model.predict(input_df)[0])
        grade = calculate_grade(predicted_score)
        risk = calculate_risk(predicted_score)
        recommendations = _build_recommendations(values)
        scenarios = _build_improvement_scenarios(values, model, model_features, predicted_score)
        score_range = (round(predicted_score - 2, 2), round(predicted_score + 2, 2))

        return render_template(
            "index.html",
            values=values,
            prediction=round(predicted_score, 2),
            grade=grade,
            risk=risk,
            score_range=score_range,
            recommendations=recommendations,
            scenarios=scenarios,
        )

    except ValueError as exc:
        values = _merge_form_with_defaults(request.form)
        return render_template("index.html", values=values, error=str(exc)), 400
    except Exception:
        app.logger.exception("Unexpected error while generating prediction.")
        values = _merge_form_with_defaults(request.form)
        return render_template(
            "index.html",
            values=values,
            error="Prediction failed. Check the inputs and confirm the model files are available.",
        ), 500

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", use_reloader=False)
