import unittest

from app import FIELD_DEFAULTS, _validate_values, app


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


if __name__ == "__main__":
    unittest.main()
