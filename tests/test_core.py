import io
import unittest
from unittest.mock import Mock, patch

import requests
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

import ml_model
from ml_model import classify_soil_quality, load_dataset, predict_top_crops, train_models
from soil_data import read_lab_report, validate_inputs
from soil_data import soil_data_sources
from weather import get_weather_bundle, search_locations
from soil_geo import fetch_soilgrids_estimate


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.metrics, cls.context = train_models(save=False)
        cls.dataset = load_dataset()

    def test_public_dataset_schema_and_classes(self):
        self.assertEqual(len(self.dataset), 2200)
        self.assertEqual(self.dataset["label"].nunique(), 22)

    def test_two_models_share_valid_evaluation_metrics(self):
        _, x_test, _, y_test = train_test_split(
            self.dataset[ml_model.FEATURES], self.dataset["label"].astype(str),
            test_size=0.2, random_state=42, stratify=self.dataset["label"].astype(str),
        )
        labels = sorted(self.dataset["label"].astype(str).unique())
        for result in self.metrics["models"].values():
            self.assertAlmostEqual(sum(map(sum, result["confusion_matrix"])), self.metrics["test_rows"])
            for key in ("accuracy", "precision_macro", "recall_macro", "macro_f1"):
                self.assertTrue(0 <= result[key] <= 1)
        for name, estimator in self.models.items():
            predicted = estimator.predict(x_test)
            result = self.metrics["models"][name]
            self.assertEqual(result["confusion_matrix"], confusion_matrix(y_test, predicted, labels=labels).tolist())
            self.assertAlmostEqual(result["accuracy"], accuracy_score(y_test, predicted))
            self.assertAlmostEqual(result["precision_macro"], precision_score(y_test, predicted, average="macro", zero_division=0))
            self.assertAlmostEqual(result["recall_macro"], recall_score(y_test, predicted, average="macro", zero_division=0))
            self.assertAlmostEqual(result["macro_f1"], f1_score(y_test, predicted, average="macro", zero_division=0))
            self.assertEqual(len(predict_top_crops(self.models, self.dataset.iloc[0].to_dict(), name, self.context)), 5)
        self.assertIn(self.metrics["best_model"], self.models)
        self.assertIn("training split", self.metrics["selection_method"])
        self.assertEqual(self.context["best_model"], self.metrics["best_model"])

    def test_best_prediction_uses_cv_selected_model(self):
        row = self.dataset.iloc[0].to_dict()
        default = predict_top_crops(self.models, row, context=self.context)
        selected = predict_top_crops(self.models, row, model_name=self.metrics["best_model"], context=self.context)
        self.assertEqual(default, selected)
        self.assertEqual(len(default), 5)
        with self.assertRaisesRegex(ValueError, "training context"):
            predict_top_crops(self.models, row)

    def test_top_five_are_ranked_model_scores(self):
        row = self.dataset.iloc[0].to_dict()
        predictions = predict_top_crops(self.models, row, model_name="KNN", context=self.context)
        self.assertEqual(len(predictions), 5)
        self.assertGreaterEqual(predictions[0]["score"], predictions[-1]["score"])
        self.assertIn("KNN", predictions[0]["score_label"])
        self.assertTrue(any("middle half" in item for item in ml_model.crop_input_explanation(row, predictions[0]["crop"], self.context)))

    def test_distinct_real_dataset_rows_are_ranked(self):
        first = self.dataset.iloc[0].to_dict()
        last = self.dataset.iloc[-1].to_dict()
        a = predict_top_crops(self.models, first, model_name=self.metrics["best_model"])[0]["crop"]
        b = predict_top_crops(self.models, last, model_name=self.metrics["best_model"])[0]["crop"]
        self.assertNotEqual(a, b)

    def test_stale_model_bundle_without_selection_metadata_is_retrained(self):
        import tempfile
        from pathlib import Path
        import joblib
        with tempfile.TemporaryDirectory() as folder:
            stale_path = Path(folder) / "stale.joblib"
            stale_summary = dict(self.metrics)
            stale_summary.pop("selection_method")
            joblib.dump({"version": ml_model.MODEL_BUNDLE_VERSION, "models": self.models,
                         "best_model": self.metrics["best_model"], "context": self.context,
                         "summary": stale_summary, "features": ml_model.FEATURES,
                         "dataset_sha256": self.metrics["dataset_sha256"]}, stale_path)
            with patch.object(ml_model, "MODEL_PATH", stale_path):
                models, metrics, context = ml_model.load_or_train_models()
            self.assertTrue(stale_path.is_file())
            self.assertIn("Random Forest", models)
            self.assertEqual(metrics["dataset_rows"], 2200)
            self.assertIn("selection_method", metrics)
            self.assertIn("train_medians", context)


class SoilDataTests(unittest.TestCase):
    def test_missing_soil_values_are_not_called_good(self):
        result = classify_soil_quality({"N": None, "P": None, "K": None, "ph": None})
        self.assertEqual(result["quality"], "Incomplete")
        self.assertEqual(len(result["missing"]), 4)

    def test_low_nutrients_and_extreme_ph_explain_status(self):
        result = classify_soil_quality({"N": 10, "P": 5, "K": 8, "ph": 9.0})
        self.assertEqual(result["quality"], "Poor")
        self.assertTrue(result["suggestions"])

    def test_complete_nonflagged_soil_profile_is_good(self):
        result = classify_soil_quality({"N": 90, "P": 42, "K": 43, "ph": 6.5})
        self.assertEqual(result["quality"], "Good")
        self.assertEqual(result["missing"], [])

    def test_non_dataset_lab_units_do_not_enter_model(self):
        report = read_lab_report(io.BytesIO(b"N,P,K,ph\n80,30,40,6.2\n"), npk_units="mg/kg")
        self.assertFalse(report["model_compatible"])
        self.assertFalse(validate_inputs({"N": 80, "P": 30, "K": 40, "ph": 6.2,
                                         "temperature": 20, "humidity": 50, "rainfall": 20},
                                        require_model_features=True, npk_units="mg/kg"))

    def test_invalid_pH_rejected(self):
        errors = validate_inputs({"ph": 15})
        self.assertTrue(errors)

    def test_lab_csv_requires_unit_provenance(self):
        with self.assertRaises(ValueError):
            read_lab_report(io.BytesIO(b"N,P,K,ph\n80,30,40,6.2\n"))

    def test_required_model_inputs_cannot_be_missing(self):
        errors = validate_inputs({"N": None, "P": 20, "K": 20, "ph": 6,
                                  "temperature": 20, "humidity": 50, "rainfall": 30},
                                 require_model_features=True)
        self.assertTrue(any("N is required" in x for x in errors))

    def test_source_register_names_geospatial_estimates_as_optional(self):
        names = {row["name"]: row["status"] for row in soil_data_sources()}
        self.assertIn("Available", names["Manual input"])
        self.assertIn("Optional", names["SoilGrids WCS"])


class GeoProviderTests(unittest.TestCase):
    def test_invalid_coordinates_rejected_before_optional_dependency_check(self):
        with self.assertRaises(ValueError):
            fetch_soilgrids_estimate(95, 200)


class WeatherTests(unittest.TestCase):
    @staticmethod
    def _place(name, country, code, lat, lon, population=0, feature="PPL"):
        return {"name": name, "country": country, "country_code": code,
                "latitude": lat, "longitude": lon, "population": population,
                "feature_code": feature, "admin1": "Region"}

    def test_bangalore_alias_fallback_reproduces_distinct_api_responses(self):
        # Reproduce the observed API behavior: the first real query only
        # matches Bangalore Town in Pakistan; the alias query finds Bengaluru.
        bangalore_response, bengaluru_response = Mock(), Mock()
        for response in (bangalore_response, bengaluru_response):
            response.raise_for_status.return_value = None
        bangalore_response.json.return_value = {"results": [
            self._place("Bangalore Town", "Pakistan", "PK", 25.1, 68.8, 12000),
        ]}
        bengaluru_response.json.return_value = {"results": [
            self._place("Bengaluru", "India", "IN", 12.97, 77.59, 8400000, "PPLA"),
        ]}
        with patch("weather.requests.get", side_effect=[bangalore_response, bengaluru_response]) as get:
            results = search_locations("Bangalore")
        self.assertEqual(results[0]["country"], "India")
        self.assertEqual(results[0]["name"], "Bengaluru")
        self.assertEqual([item["country"] for item in results], ["India", "Pakistan"])
        self.assertEqual([call.kwargs["params"]["name"] for call in get.call_args_list], ["Bangalore", "Bengaluru"])
        for call in get.call_args_list:
            self.assertEqual(call.kwargs["params"]["count"], 100)
            self.assertNotIn("countryCode", call.kwargs["params"])

    def test_bengaluru_query_returns_indian_city(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [
            self._place("Bengaluru", "India", "IN", 12.97, 77.59, 8400000, "PPLA"),
        ]}
        with patch("weather.requests.get", side_effect=[response, response]) as get:
            results = search_locations("Bengaluru")
        self.assertEqual(results[0]["name"], "Bengaluru")
        self.assertEqual(results[0]["country_code"], "IN")
        self.assertEqual([call.kwargs["params"]["name"] for call in get.call_args_list], ["Bengaluru", "Bangalore"])

    def test_explicit_country_uses_api_filter_and_rejects_wrong_country_results(self):
        response = Mock()
        response.raise_for_status.return_value = None
        bangalore_response, bengaluru_response = Mock(), Mock()
        for item in (bangalore_response, bengaluru_response):
            item.raise_for_status.return_value = None
        bangalore_response.json.return_value = {"results": [self._place("Bangalore Town", "Pakistan", "PK", 25.1, 68.8)]}
        bengaluru_response.json.return_value = {"results": [self._place("Bengaluru", "India", "IN", 12.97, 77.59, 8400000, "PPLA")]}
        with patch("weather.requests.get", side_effect=[bangalore_response, bengaluru_response]) as get:
            results = search_locations("Bangalore, India")
        self.assertEqual([item["country"] for item in results], ["India"])
        self.assertEqual([call.kwargs["params"]["name"] for call in get.call_args_list], ["Bangalore", "Bengaluru"])
        self.assertTrue(all(call.kwargs["params"]["countryCode"] == "IN" for call in get.call_args_list))

    def test_explicit_common_country_aliases_are_filtered(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}
        for query, expected_name, expected_code in (("London, UK", "London", "GB"),
                                                     ("Tokyo, Japan", "Tokyo", "JP")):
            with self.subTest(query=query), patch("weather.requests.get", return_value=response) as get:
                with self.assertRaisesRegex(ValueError, "No location found"):
                    search_locations(query)
                self.assertEqual(get.call_args.kwargs["params"]["name"], expected_name)
                self.assertEqual(get.call_args.kwargs["params"]["countryCode"], expected_code)

    def test_global_search_queries_remain_unfiltered(self):
        response = Mock()
        response.raise_for_status.return_value = None
        for query, place in (
            ("London", self._place("London", "United Kingdom", "GB", 51.5, -.1, 9000000, "PPLC")),
            ("Tokyo", self._place("Tokyo", "Japan", "JP", 35.68, 139.69, 14000000, "PPLC")),
        ):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"results": [place]}
            with self.subTest(query=query), patch("weather.requests.get", return_value=response) as get:
                results = search_locations(query)
                self.assertEqual(results[0]["name"], query)
                self.assertEqual(results[0]["country_code"], place["country_code"])
                self.assertNotIn("countryCode", get.call_args.kwargs["params"])

    def test_alias_duplicates_by_coordinates_are_removed(self):
        original, alias = Mock(), Mock()
        for response in (original, alias):
            response.raise_for_status.return_value = None
        original.json.return_value = {"results": [
            {**self._place("Bangalore", "India", "IN", 12.9716, 77.5946, 8400000, "PPLA"), "id": 1},
        ]}
        alias.json.return_value = {"results": [
            {**self._place("Bengaluru", "India", "IN", 12.9716, 77.5946, 8400000, "PPLA"), "id": 2},
        ]}
        with patch("weather.requests.get", side_effect=[original, alias]):
            results = search_locations("Bangalore")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Bengaluru")

    def test_all_alias_queries_empty_returns_no_match(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}
        with patch("weather.requests.get", side_effect=[response, response]):
            with self.assertRaisesRegex(ValueError, "No location found"):
                search_locations("Bangalore")

    def test_alias_success_survives_failure_of_original_query(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [
            self._place("Bengaluru", "India", "IN", 12.97, 77.59, 8400000, "PPLA"),
        ]}
        with patch("weather.requests.get", side_effect=[requests.Timeout("first query timed out"), response]):
            results = search_locations("Bangalore")
        self.assertEqual(results[0]["name"], "Bengaluru")
        self.assertEqual(results[0]["country_code"], "IN")

    def test_missing_results_and_unusable_candidates_are_clear(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": []}
        with patch("weather.requests.get", side_effect=[response, response]):
            with self.assertRaisesRegex(ValueError, "No location found"):
                search_locations("Nonexistent City")
        response.json.return_value = {"results": [{"latitude": "bad", "longitude": 0}]}
        with patch("weather.requests.get", side_effect=[response, response]):
            with self.assertRaisesRegex(RuntimeError, "without usable coordinates"):
                search_locations("Example")

    def test_geocoding_failure_has_no_synthetic_fallback(self):
        with patch("weather.requests.get", side_effect=requests.Timeout("network timeout")):
            with self.assertRaises(RuntimeError):
                search_locations("Pune")

    def test_malformed_api_json_shape_returns_clear_error(self):
        response = Mock()
        response.json.return_value = []
        response.raise_for_status.return_value = None
        with patch("weather.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "invalid response shape"):
                get_weather_bundle({"latitude":18.5,"longitude":73.8,"name":"Pune"})

    def test_forecast_timeout_raises_a_clear_error(self):
        with patch("weather.requests.get", side_effect=requests.Timeout("network timeout")):
            with self.assertRaisesRegex(RuntimeError, "forecast request failed"):
                get_weather_bundle({"latitude":18.5,"longitude":73.8,"name":"Pune"})

    def test_weather_bundle_shapes_current_and_forecast_data(self):
        current = {"time":"2026-11-01T10:00", "temperature_2m":25, "relative_humidity_2m":60,
                   "precipitation":0.2, "wind_speed_10m":8, "weather_code":2}
        daily = {"time":["2026-11-01"], "temperature_2m_min":[18], "temperature_2m_max":[29],
                 "precipitation_sum":[1.2], "weather_code":[2], "wind_speed_10m_max":[12],
                 "et0_fao_evapotranspiration":[4.1]}
        response = Mock()
        response.json.return_value = {"current":current, "daily":daily, "timezone":"Asia/Kolkata"}
        response.raise_for_status.return_value = None
        with patch("weather.requests.get", return_value=response):
            result = get_weather_bundle({"latitude":18.5,"longitude":73.8,"name":"Pune","country":"India"})
        self.assertEqual(result["current"]["wind_kmh"], 8)
        self.assertEqual(result["daily"][0]["et0_mm"], 4.1)
        self.assertEqual(result["place"], "Pune, India")
        self.assertTrue(result["fetched_at_utc"])


if __name__ == "__main__":
    unittest.main()
