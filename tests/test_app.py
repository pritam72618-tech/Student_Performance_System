import unittest

from app import (
    FIELD_DEFAULTS,
    _build_model_input,
    _calibrate_prediction,
    _load_artifacts,
    _validate_values,
    app,
)


class AppValidationTests(unittest.TestCase):
    def test_rejects_non_finite_number(self):
        payload = dict(FIELD_DEFAULTS)
        payload["Hours_Studied"] = "nan"

        with self.assertRaises(ValueError):
            _validate_values(payload)

    def test_rejects_unknown_category(self):
        payload = dict(FIELD_DEFAULTS)
        payload["Motivation_Level"] = "VeryHigh"

        with self.assertRaises(ValueError):
            _validate_values(payload)

    def test_rejects_values_outside_training_range(self):
        payload = dict(FIELD_DEFAULTS)
        payload["Attendance"] = "45"

        with self.assertRaises(ValueError):
            _validate_values(payload)


class PredictRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_success(self):
        response = self.client.post("/predict", data=dict(FIELD_DEFAULTS))
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted Score", response.get_data(as_text=True))

    def test_predict_rejects_tampered_category(self):
        payload = dict(FIELD_DEFAULTS)
        payload["Motivation_Level"] = "VeryHigh"

        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("must be one of", response.get_data(as_text=True))


class PredictionBehaviorTests(unittest.TestCase):
    def _predict_score(self, values):
        model, model_features = _load_artifacts()
        cleaned = _validate_values(values)
        input_df = _build_model_input(cleaned, model_features)
        raw = float(model.predict(input_df)[0])
        return _calibrate_prediction(raw, cleaned)

    def test_high_profile_scores_higher_than_low_profile(self):
        low = dict(FIELD_DEFAULTS)
        low.update(
            {
                "Hours_Studied": "1",
                "Attendance": "65",
                "Previous_Scores": "55",
                "Sleep_Hours": "5",
                "Tutoring_Sessions": "0",
                "Motivation_Level": "Low",
            }
        )
        high = dict(FIELD_DEFAULTS)
        high.update(
            {
                "Hours_Studied": "8",
                "Attendance": "95",
                "Previous_Scores": "100",
                "Sleep_Hours": "8",
                "Tutoring_Sessions": "4",
                "Motivation_Level": "High",
            }
        )

        low_score = self._predict_score(low)
        high_score = self._predict_score(high)
        self.assertGreater(high_score, low_score)

    def test_previous_score_100_with_good_inputs_is_high(self):
        payload = dict(FIELD_DEFAULTS)
        payload.update(
            {
                "Hours_Studied": "8",
                "Attendance": "80",
                "Previous_Scores": "100",
                "Sleep_Hours": "7",
                "Tutoring_Sessions": "4",
                "Motivation_Level": "High",
            }
        )
        score = self._predict_score(payload)
        self.assertGreaterEqual(score, 88.0)


if __name__ == "__main__":
    unittest.main()
