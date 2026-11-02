import io
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import requests
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

import ml_model
from ml_model import classify_soil_quality, load_dataset, predict_top_crops, train_models
from soil_data import read_lab_report, validate_inputs
from soil_data import soil_data_sources
from weather import (_freshness, _timestamp, clear_weather_state, country_options,
                     get_weather_bundle, next_weather_refresh, search_locations,
                     _validate_units)
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
    def _place(name, country, code, lat, lon, population=0, feature="PPL", postcodes=None):
        return {"name": name, "country": country, "country_code": code,
                "latitude": lat, "longitude": lon, "population": population,
                "feature_code": feature, "admin1": "Region", "postcodes": postcodes or []}

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
        cases = (
            ("London, UK", "London", "GB", self._place("London", "United Kingdom", "GB", 51.5, -.1)),
            ("London, Canada", "London", "CA", self._place("London", "Canada", "CA", 42.98, -81.23)),
            ("Tokyo, Japan", "Tokyo", "JP", self._place("Tokyo", "Japan", "JP", 35.68, 139.69)),
        )
        for query, expected_name, expected_code, place in cases:
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"results": [place]}
            with self.subTest(query=query), patch("weather.requests.get", return_value=response) as get:
                results = search_locations(query)
                self.assertEqual(results[0]["country_code"], expected_code)
                self.assertEqual(get.call_args.kwargs["params"]["name"], expected_name)
                self.assertEqual(get.call_args.kwargs["params"]["countryCode"], expected_code)

    def test_postal_code_search_is_passed_to_global_geocoder(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [
            self._place("New York", "United States", "US", 40.75, -73.99, 8000000, "PPLA", ["10001"]),
        ]}
        with patch("weather.requests.get", return_value=response) as get:
            results = search_locations("10001")
        self.assertEqual(results[0]["country_code"], "US")
        self.assertEqual(get.call_args.kwargs["params"]["name"], "10001")
        self.assertNotIn("countryCode", get.call_args.kwargs["params"])
        self.assertEqual(results[0]["postal_code"], "10001")
        self.assertEqual(results[0]["match_type"], "exact postal-code match")

    def test_all_requested_postal_formats_preserve_text_and_country(self):
        cases = (
            ("562157", "IN"), ("110001", "IN"), ("10001", "US"), ("00501", "US"),
            ("SW1A 1AA", "GB"), ("K1A 0B1", "CA"), ("10115", "DE"), ("75001", "FR"),
            ("100-0001", "JP"), ("2000", "AU"), ("1012 JS", "NL"), ("01001-000", "BR"),
        )
        for index, (postcode, code) in enumerate(cases):
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"results": [self._place(
                f"Locality {index}", code, code, 10 + index, 20 + index, postcodes=[postcode]) ]}
            with self.subTest(postcode=postcode), patch("weather.requests.get", return_value=response) as get:
                result = search_locations(postcode, country_code=code)
                params = get.call_args.kwargs["params"]
                self.assertEqual(params["name"], postcode)
                self.assertEqual(params["countryCode"], code)
                self.assertEqual(result[0]["postal_code"], postcode)
                self.assertEqual(result[0]["match_type"], "exact postal-code match")

    def test_exact_postcode_ranks_before_nearby_nonmatch(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [
            self._place("Nearby", "United States", "US", 40.7, -74, 9000000, "PPLC", ["10002"]),
            self._place("Exact", "United States", "US", 40.8, -73.9, 1000, "PPL", ["10001"]),
        ]}
        with patch("weather.requests.get", return_value=response):
            result = search_locations("10001", country_code="US")
        self.assertEqual([item["name"] for item in result], ["Exact"])

    def test_country_options_use_iso_codes_and_include_territories(self):
        choices = dict(country_options())
        self.assertEqual(choices["IN"], "India")
        self.assertIn("GB", choices)
        self.assertIn("PR", choices)
        self.assertTrue(all(len(code) == 2 for code in choices))

    def test_region_and_country_qualification_normalizes_spacing_and_diacritics(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [self._place("São Paulo", "Brazil", "BR", -23.5, -46.6)]}
        with patch("weather.requests.get", return_value=response) as get:
            result = search_locations("  Sao   Paulo, São Paulo, Brazil ")
        self.assertEqual(result[0]["country_code"], "BR")
        self.assertEqual(get.call_args.kwargs["params"]["name"], "Sao Paulo, São Paulo")
        self.assertEqual(get.call_args.kwargs["params"]["countryCode"], "BR")

    def test_geonames_fallback_resolves_postcode_with_country_filter(self):
        openmeteo, geonames = Mock(), Mock()
        for response in (openmeteo, geonames):
            response.raise_for_status.return_value = None
        openmeteo.json.return_value = {"results": []}
        geonames.json.return_value = {"postalCodes": [{
            "postalCode": "00501", "placeName": "Holtsville", "adminName1": "New York",
            "countryName": "United States", "countryCode": "US", "lat": "40.8154",
            "lng": "-73.0451", "geonameId": 123,
        }]}
        with patch("weather.requests.get", side_effect=[openmeteo, geonames]) as get:
            result = search_locations("00501", country_code="US", geonames_username="test-user")
        self.assertEqual(result[0]["provider"], "GeoNames")
        self.assertEqual(result[0]["postal_code"], "00501")
        self.assertEqual(get.call_args_list[1].args[0], "https://secure.geonames.org/postalCodeSearchJSON")
        params = get.call_args_list[1].kwargs["params"]
        self.assertEqual(params["postalcode"], "00501")
        self.assertNotIn("placename", params)
        self.assertEqual(params["country"], "US")
        self.assertEqual(params["username"], "test-user")
        self.assertEqual(get.call_args_list[1].kwargs["timeout"], 8)

    def test_geonames_ssl_http_and_timeout_failures_are_reported_without_credentials(self):
        import requests

        primary = Mock()
        primary.raise_for_status.return_value = None
        primary.json.return_value = {"results": []}
        for failure in (requests.exceptions.SSLError("certificate hostname mismatch"),
                        requests.exceptions.Timeout("request timed out"),
                        requests.exceptions.HTTPError("HTTP 503")):
            with self.subTest(failure=type(failure).__name__), \
                    patch("weather.requests.get", side_effect=[primary, failure]):
                with self.assertRaises(RuntimeError) as error:
                    search_locations("560075", country_code="IN", geonames_username="private-test-user")
                self.assertNotIn("private-test-user", str(error.exception))
                self.assertIn("GeoNames fallback failed", str(error.exception))

    def test_geonames_exact_match_replaces_nonexact_primary_postcode_match(self):
        openmeteo, geonames = Mock(), Mock()
        for response in (openmeteo, geonames):
            response.raise_for_status.return_value = None
        openmeteo.json.return_value = {"results": [self._place("Wrong", "United States", "US", 40, -70, postcodes=["00502"])]}
        geonames.json.return_value = {"postalCodes": [{
            "postalCode": "00501", "placeName": "Correct", "countryName": "United States",
            "countryCode": "US", "lat": "41", "lng": "-71",
        }]}
        with patch("weather.requests.get", side_effect=[openmeteo, geonames]):
            result = search_locations("00501", country_code="US", geonames_username="test-user")
        self.assertEqual(result[0]["name"], "Correct")
        self.assertEqual(result[0]["provider"], "GeoNames")

    def test_geonames_failure_and_unconfigured_fallback_are_clear(self):
        empty = Mock()
        empty.raise_for_status.return_value = None
        empty.json.return_value = {"results": []}
        with patch("weather.requests.get", return_value=empty) as get:
            with self.assertRaisesRegex(ValueError, "Streamlit Community Cloud.*Settings/Secrets"):
                search_locations("99999")
        self.assertEqual(get.call_count, 1)
        failed_primary, failed_fallback = Mock(), Mock()
        for response in (failed_primary, failed_fallback):
            response.raise_for_status.return_value = None
        failed_primary.json.return_value = {"results": []}
        failed_fallback.json.return_value = {"status": {"message": "invalid username bad-user"}}
        with patch("weather.requests.get", side_effect=[failed_primary, failed_fallback]):
            with self.assertRaisesRegex(RuntimeError, "invalid username") as caught:
                search_locations("99999", geonames_username="bad-user")
            self.assertNotIn("bad-user", str(caught.exception))

    def test_weather_state_clear_and_refresh_helpers(self):
        state = {"weather_query": "London", "weather_country_filter": "GB",
                 "weather_locations": [{"name": "London"}], "weather_selected": 0,
                 "weather_bundle": {"place": "London"}, "weather_bundle_location": (51.5, -.1),
                 "weather_error": "old error", "weather_refresh_key": 3}
        clear_weather_state(state)
        self.assertEqual(state["weather_query"], "")
        self.assertEqual(state["weather_country_filter"], "")
        self.assertEqual(state["weather_locations"], [])
        self.assertIsNone(state["weather_bundle"])
        self.assertIsNone(state["weather_bundle_location"])
        self.assertIsNone(state["weather_error"])
        self.assertEqual(next_weather_refresh(state), 4)

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
        current = {"time":"2026-11-01T10:00", "temperature_2m":25, "apparent_temperature":27,
                   "relative_humidity_2m":60, "precipitation":0.2, "wind_speed_10m":8,
                   "wind_direction_10m":180, "weather_code":2, "cloud_cover":45, "pressure_msl":1012.4}
        daily = {"time":["2026-11-01"], "temperature_2m_min":[18], "temperature_2m_max":[29],
                 "precipitation_probability_max":[60], "precipitation_sum":[1.2], "weather_code":[2], "wind_speed_10m_max":[12],
                 "et0_fao_evapotranspiration":[4.1]}
        response = Mock()
        response.json.return_value = {"current":current, "daily":daily, "timezone":"Asia/Kolkata"}
        response.raise_for_status.return_value = None
        with patch("weather.requests.get", return_value=response) as get:
            result = get_weather_bundle({"latitude":18.5,"longitude":73.8,"name":"Pune","country":"India","timezone":"Asia/Kolkata"})
        self.assertEqual(result["current"]["wind_kmh"], 8)
        self.assertEqual(result["current"]["apparent_temperature_c"], 27)
        self.assertEqual(result["current"]["wind_direction_deg"], 180)
        self.assertEqual(result["current"]["cloud_cover_pct"], 45)
        self.assertEqual(result["current"]["pressure_hpa"], 1012.4)
        self.assertIsNone(result["current"]["wind_gusts_kmh"])
        self.assertIsNone(result["current"]["visibility_km"])
        self.assertEqual(result["current"]["icon"], "⛅")
        self.assertEqual(result["daily"][0]["day"], "Sunday")
        self.assertEqual(result["daily"][0]["precipitation_probability_pct"], 60)
        self.assertEqual(result["daily"][0]["icon"], "⛅")
        self.assertEqual(result["daily"][0]["et0_mm"], 4.1)
        self.assertEqual(result["place"], "Pune, India")
        self.assertTrue(result["fetched_at_utc"])
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["timezone"], "Asia/Kolkata")
        self.assertIn("apparent_temperature", params["current"])
        self.assertIn("wind_direction_10m", params["current"])
        self.assertIn("precipitation_probability_max", params["daily"])

    def test_invalid_coordinates_and_explicit_unit_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid WGS84"):
            get_weather_bundle({"latitude":91,"longitude":0,"timezone":"UTC"})
        with self.assertRaisesRegex(RuntimeError, "unexpected unit"):
            _validate_units({"current_units":{"temperature_2m":"°F"}})

    def test_weather_parser_rejects_missing_current_or_daily_fields(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "current": {"time": "2026-11-01T10:00"},
            "daily": {"time": ["2026-11-01"]},
        }
        with patch("weather.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "incomplete current conditions"):
                get_weather_bundle({"latitude": 12.9, "longitude": 77.5, "timezone": "Asia/Kolkata"})

    def test_expanded_weather_fields_local_hours_and_agricultural_values(self):
        now = datetime.now(timezone.utc)
        zone = ZoneInfo("Asia/Kolkata")
        today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
        current = {"time":int((now-timedelta(minutes=10)).timestamp()), "interval":900,
                   "temperature_2m":25, "apparent_temperature":27, "relative_humidity_2m":60,
                   "is_day":1, "precipitation":0.2, "rain":0.1, "showers":0.1,
                   "wind_speed_10m":8, "wind_direction_10m":180, "wind_gusts_10m":11,
                   "weather_code":2, "cloud_cover":45, "pressure_msl":1012.4, "visibility":9200,
                   "et0_fao_evapotranspiration":0.2, "vapour_pressure_deficit":1.1,
                   "soil_temperature_0cm":22, "soil_temperature_6cm":20,
                   "soil_moisture_0_to_1cm":0.2}
        hourly = {"time":[int((now+timedelta(minutes=5+i*60)).timestamp()) for i in range(24)],
                  "temperature_2m":[25]*24, "apparent_temperature":[27]*24,
                  "precipitation_probability":[10]*24, "precipitation":[0.1]*24,
                  "relative_humidity_2m":[60]*24, "wind_speed_10m":[8]*24, "weather_code":[2]*24}
        daily = {"time":[int(today.timestamp())], "weather_code":[2],
                 "temperature_2m_min":[18], "temperature_2m_max":[29],
                 "apparent_temperature_min":[17], "apparent_temperature_max":[31],
                 "precipitation_probability_max":[60], "precipitation_sum":[1.2],
                 "wind_speed_10m_max":[12], "wind_gusts_10m_max":[18], "uv_index_max":[8],
                 "sunrise":[int((today+timedelta(hours=6)).timestamp())],
                 "sunset":[int((today+timedelta(hours=18)).timestamp())], "sunshine_duration":[21600],
                 "et0_fao_evapotranspiration":[4.1]}
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"current":current, "hourly":hourly, "daily":daily, "timezone":"Asia/Kolkata"}
        with patch("weather.requests.get", return_value=response) as get:
            result = get_weather_bundle({"latitude":18.5,"longitude":73.8,"timezone":"Asia/Kolkata"})
        self.assertEqual(result["current"]["visibility_km"], 9.2)
        self.assertEqual(result["current"]["rain_mm"], 0.1)
        self.assertEqual(result["current"]["wind_gusts_kmh"], 11)
        self.assertEqual(result["current"]["agriculture"]["vpd_kpa"], 1.1)
        self.assertEqual(result["current"]["agriculture"]["soil_moisture_m3_m3"]["0–1 cm"], 0.2)
        self.assertEqual(len(result["hourly"]), 24)
        self.assertEqual(result["hourly"][0]["precipitation_probability_pct"], 10)
        self.assertEqual(result["daily"][0]["sunrise"], "06:00 IST")
        self.assertEqual(result["daily"][0]["sunshine_hours"], 6)
        self.assertEqual(result["model_selection"], "best_match")
        self.assertIsNone(result["model_run_initialization_utc"])
        self.assertEqual(result["timezone"], "Asia/Kolkata")
        self.assertEqual(get.call_args.kwargs["params"]["forecast_hours"], 48)

    def test_weather_api_retries_one_transient_failure(self):
        now = datetime.now(timezone.utc)
        day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        current = {"time":int(now.timestamp()), "temperature_2m":1, "apparent_temperature":1,
                   "relative_humidity_2m":50, "precipitation":0, "wind_speed_10m":1,
                   "wind_direction_10m":0, "weather_code":0, "cloud_cover":0, "pressure_msl":1000}
        hourly = {"time":[int((now+timedelta(hours=i+1)).timestamp()) for i in range(24)]}
        daily = {"time":[int(day.timestamp())], "temperature_2m_min":[0], "temperature_2m_max":[2],
                 "precipitation_probability_max":[0], "precipitation_sum":[0], "wind_speed_10m_max":[1], "weather_code":[0]}
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"current":current, "hourly":hourly, "daily":daily, "timezone":"UTC"}
        with patch("weather.requests.get", side_effect=[requests.Timeout("transient"), response]) as get:
            result = get_weather_bundle({"latitude":0,"longitude":0,"timezone":"UTC"})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(result["model_selection"], "best_match")

    def test_weather_valid_time_freshness_and_dst_conversion(self):
        self.assertEqual(_freshness(30 * 60), "fresh")
        self.assertEqual(_freshness(2 * 60 * 60), "delayed")
        self.assertEqual(_freshness(7 * 60 * 60), "stale")
        self.assertEqual(_freshness(None), "unknown")
        zone = ZoneInfo("America/Toronto")
        before_dst = _timestamp(int(datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc).timestamp()), zone)
        after_dst = _timestamp(int(datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc).timestamp()), zone)
        self.assertEqual(before_dst.strftime("%H:%M %z"), "01:30 -0400")
        self.assertEqual(after_dst.strftime("%H:%M %z"), "02:30 -0500")

    def test_weather_parser_rejects_missing_daily_forecast_fields(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "current": {
                "time": "2026-11-01T10:00", "temperature_2m": 25, "apparent_temperature": 27,
                "relative_humidity_2m": 60, "precipitation": 0.2, "wind_speed_10m": 8,
                "wind_direction_10m": 180, "weather_code": 2, "cloud_cover": 45, "pressure_msl": 1012.4,
            },
            "daily": {"time": ["2026-11-01"], "temperature_2m_min": [18]},
        }
        with patch("weather.requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "incomplete daily forecast"):
                get_weather_bundle({"latitude": 12.9, "longitude": 77.5, "timezone": "Asia/Kolkata"})


if __name__ == "__main__":
    unittest.main()
