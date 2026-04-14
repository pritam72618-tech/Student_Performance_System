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
        value = float(raw_value) if raw_value else 0.0
        # The model was trained with weekly study hours; UI accepts daily hours.
        if field == "Hours_Studied":
            value = min(value * 5, 44)
        row[field] = value

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


def _calibrate_prediction(raw_prediction, values):
    previous_scores = float(values["Previous_Scores"])
    attendance = float(values["Attendance"])
    study_hours_day = float(values["Hours_Studied"])
    sleep_hours = float(values["Sleep_Hours"])
    tutoring_sessions = float(values["Tutoring_Sessions"])
    physical_activity = float(values["Physical_Activity"])

    parental_involvement = values["Parental_Involvement"]
    access_to_resources = values["Access_to_Resources"]
    motivation = values["Motivation_Level"]
    peer_influence = values["Peer_Influence"]
    family_income = values["Family_Income"]
    teacher_quality = values["Teacher_Quality"]
    school_type = values["School_Type"]
    distance_from_home = values["Distance_from_Home"]
    parental_education = values["Parental_Education_Level"]
    internet_access = values["Internet_Access"]
    learning_disabilities = values["Learning_Disabilities"]
    extracurricular = values["Extracurricular_Activities"]

    def _norm(value, min_value, max_value):
        clamped = max(min_value, min(value, max_value))
        if max_value == min_value:
            return 0.0
        return (clamped - min_value) / (max_value - min_value)

    study_component = _norm(study_hours_day, 0, 12) * 100
    sleep_component = _norm(sleep_hours, 4, 10) * 100
    tutoring_component = _norm(tutoring_sessions, 0, 8) * 100

    base_score = (
        (0.68 * previous_scores)
        + (0.14 * attendance)
        + (0.14 * study_component)
        + (0.02 * sleep_component)
        + (0.02 * tutoring_component)
    )

    bonus = 0.0
    bonus += {"Low": -2.0, "Medium": 0.0, "High": 2.0}.get(parental_involvement, 0.0)
    bonus += {"Low": -2.0, "Medium": 0.0, "High": 2.0}.get(access_to_resources, 0.0)
    bonus += {"Low": -4.0, "Medium": 0.0, "High": 4.0}.get(motivation, 0.0)
    bonus += {"Negative": -2.0, "Neutral": 0.0, "Positive": 2.0}.get(peer_influence, 0.0)
    bonus += {"Low": -1.5, "Medium": 0.0, "High": 1.5}.get(family_income, 0.0)
    bonus += {"Low": -2.0, "Medium": 0.0, "High": 2.0}.get(teacher_quality, 0.0)
    bonus += {"Public": 0.0, "Private": 0.5}.get(school_type, 0.0)
    bonus += {"Far": -1.0, "Moderate": 0.0, "Near": 1.0}.get(distance_from_home, 0.0)
    bonus += {"High School": -0.8, "Undergraduate": 0.2, "Postgraduate": 1.0}.get(parental_education, 0.0)
    bonus += {"No": -1.0, "Yes": 1.0}.get(internet_access, 0.0)
    bonus += {"Yes": -1.5, "No": 0.0}.get(learning_disabilities, 0.0)
    bonus += {"No": 0.0, "Yes": 0.8}.get(extracurricular, 0.0)

    if physical_activity < 1:
        bonus -= 0.6
    elif physical_activity <= 10:
        bonus += 0.6
    else:
        bonus -= 0.8

    anchor_score = base_score + bonus

    # Guardrails so very strong/weak core profiles remain intuitive.
    if previous_scores >= 88 and attendance >= 75 and study_hours_day >= 6 and tutoring_sessions >= 2:
        anchor_score = max(anchor_score, 88.0)
    if previous_scores >= 95 and attendance >= 80 and study_hours_day >= 7 and tutoring_sessions >= 3:
        anchor_score = max(anchor_score, 92.0)
    if previous_scores <= 60 and attendance <= 70 and study_hours_day <= 2:
        anchor_score = min(anchor_score, 65.0)

    calibrated = (0.10 * raw_prediction) + (0.90 * anchor_score)
    return max(0.0, min(100.0, calibrated))


def _build_improvement_scenarios(values, model, model_features, current_prediction):
    scenario_specs = [
        ("Attendance", min(float(values["Attendance"]) + 5, 100), "Improve attendance"),
        ("Hours_Studied", min(float(values["Hours_Studied"]) + 1, 12), "Study 1 more hour per day"),
        ("Sleep_Hours", min(float(values["Sleep_Hours"]) + 1, 10), "Sleep 1 more hour per day"),
        ("Tutoring_Sessions", min(float(values["Tutoring_Sessions"]) + 1, 8), "Add 1 tutoring session per week"),
    ]

    scenarios = []
    for field, new_value, label in scenario_specs:
        if float(values[field]) == new_value:
            continue
        scenario_values = dict(values)
        scenario_values[field] = str(new_value)
        scenario_df = _build_model_input(scenario_values, model_features)
        scenario_raw_prediction = float(model.predict(scenario_df)[0])
        scenario_prediction = _calibrate_prediction(scenario_raw_prediction, scenario_values)
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
        "Hours_Studied": (0, 12),
        "Attendance": (60, 100),
        "Sleep_Hours": (4, 10),
        "Previous_Scores": (50, 100),
        "Tutoring_Sessions": (0, 8),
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

        raw_predicted_score = float(model.predict(input_df)[0])
        predicted_score = _calibrate_prediction(raw_predicted_score, values)
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
