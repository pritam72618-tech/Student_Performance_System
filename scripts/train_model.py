from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "StudentPerformance_ML_Ready_No_Gender.csv"
ARTIFACTS_DIR = BASE_DIR / "models"
MODEL_PATH = ARTIFACTS_DIR / "student_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "model_features.pkl"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Exam_Score"])
    y = df["Exam_Score"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        )
    }

    best_model = None
    best_score = -1

    print("\nModel Comparison Results:\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{name}")
        print(f"R2 Score: {round(r2,4)}")
        print(f"MAE: {round(mae,2)}")
        print(f"RMSE: {round(rmse,2)}\n")

        if r2 > best_score:
            best_score = r2
            best_model = model

    # Cross validation on best model
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")
    print("Cross Validation R2 Mean:", round(cv_scores.mean(),4))

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)

    print("\nBest model saved.")
    print(f"Final R2 Score: {round(best_score,4)}")


if __name__ == "__main__":
    main()