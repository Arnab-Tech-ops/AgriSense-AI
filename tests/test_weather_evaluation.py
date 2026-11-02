import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock, patch

import requests

from scripts.evaluate_forecasts import evaluate, metrics


class ForecastEvaluationTests(unittest.TestCase):
    def test_metrics_ignore_unavailable_values_and_compute_mae_rmse(self):
        result = metrics([1, 3, None, float("nan")], [2, 1, 4, 2])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], (2.5) ** 0.5)

    def test_lead_metrics_align_times_and_use_era5_reference(self):
        start = datetime(2026, 10, 19, tzinfo=timezone.utc)
        timestamps = [int((start + timedelta(hours=i)).timestamp()) for i in range(48)]
        previous = Mock()
        previous.raise_for_status.return_value = None
        previous.json.return_value = {"hourly": {"time":timestamps,
            "temperature_2m_previous_day1":[21]*48,
            "precipitation_previous_day1":[2]*48}}
        archive = Mock()
        archive.raise_for_status.return_value = None
        archive.json.return_value = {"hourly": {"time":timestamps,
            "temperature_2m":[20]*48, "precipitation":[1]*48}}
        with patch("scripts.evaluate_forecasts.requests.get", side_effect=[previous, archive]) as get:
            rows = evaluate(18.5, 73.8, "2026-10-19", "2026-10-20", "Asia/Kolkata", leads=[1])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["mae"], 1)
        self.assertEqual(rows[0]["rmse"], 1)
        self.assertEqual(rows[0]["n"], 48)
        self.assertEqual(rows[2]["aggregation"], "full local-day totals")
        self.assertEqual(rows[2]["n"], 1)
        self.assertEqual(rows[2]["mae"], 24)
        self.assertEqual(get.call_args_list[0].args[0], "https://previous-runs-api.open-meteo.com/v1/forecast")
        self.assertEqual(get.call_args_list[0].kwargs["params"]["models"], "best_match")
        self.assertEqual(get.call_args_list[1].args[0], "https://archive-api.open-meteo.com/v1/archive")
        self.assertEqual(get.call_args_list[1].kwargs["params"]["models"], "era5")

    def test_evaluation_validates_coordinates_timezone_and_api_failure(self):
        with self.assertRaises(ValueError):
            evaluate(120, 0, "2026-10-19", "2026-10-20")
        with self.assertRaises(ValueError):
            evaluate(0, 0, "2026-10-20", "2026-10-19")
        with patch("scripts.evaluate_forecasts.requests.get", side_effect=requests.Timeout("timeout")):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                evaluate(0, 0, "2026-10-19", "2026-10-19", leads=[1])


if __name__ == "__main__":
    unittest.main()
