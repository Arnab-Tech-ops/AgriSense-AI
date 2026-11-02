"""Evaluate archived best-match forecast lead times against ERA5 reanalysis.

This is a reproducible baseline, not station-observation verification. ERA5 is a
gridded reanalysis reference with its own model and spatial-resolution errors.
"""
import argparse
import csv
from datetime import date, datetime, timezone
import math
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def metrics(forecast_values, reference_values):
    """Return sample count, MAE and RMSE for paired finite numeric values."""
    pairs = []
    for forecast, reference in zip(forecast_values, reference_values):
        try:
            forecast, reference = float(forecast), float(reference)
        except (TypeError, ValueError):
            continue
        if math.isfinite(forecast) and math.isfinite(reference):
            pairs.append((forecast, reference))
    if not pairs:
        return {"n": 0, "mae": None, "rmse": None}
    errors = [forecast - reference for forecast, reference in pairs]
    return {
        "n": len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


def _get_payload(url, params, timeout):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise RuntimeError(f"Timed out requesting {url}.") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Request failed for {url} ({type(exc).__name__}).") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON returned by {url}.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
        raise RuntimeError(f"Invalid hourly response from {url}.")
    return payload


def _series(payload, name):
    hourly = payload["hourly"]
    times, values = hourly.get("time"), hourly.get(name)
    if not isinstance(times, list) or not isinstance(values, list):
        return {}
    return {int(timestamp): value for timestamp, value in zip(times, values)
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)}


def _daily_totals(forecasts, references, zone):
    """Sum paired hourly precipitation by the chosen location's local date."""
    grouped = {}
    for timestamp in forecasts.keys() & references.keys():
        try:
            forecast, reference = float(forecasts[timestamp]), float(references[timestamp])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(forecast) or not math.isfinite(reference):
            continue
        local_day = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(zone).date()
        grouped.setdefault(local_day, [[], []])
        grouped[local_day][0].append(forecast)
        grouped[local_day][1].append(reference)
    # Exclude partial dates; a daily sum from fewer than 24 hourly values would
    # systematically understate rainfall.
    pairs = [(sum(forecast), sum(reference)) for forecast, reference in grouped.values()
             if len(forecast) == 24 and len(reference) == 24]
    return metrics([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def evaluate(latitude, longitude, start_date, end_date, timezone_name="UTC", leads=range(1, 8), timeout=20):
    """Return hourly temperature/rain and full-local-day rain errors by lead."""
    if not -90 <= float(latitude) <= 90 or not -180 <= float(longitude) <= 180:
        raise ValueError("Latitude/longitude must be valid WGS84 coordinates.")
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        zone = ZoneInfo(timezone_name)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("Use valid ISO dates and an IANA timezone name.") from exc
    if end < start:
        raise ValueError("end-date must be on or after start-date.")
    lead_days = sorted(set(int(lead) for lead in leads))
    if not lead_days or any(lead < 1 or lead > 7 for lead in lead_days):
        raise ValueError("Previous Runs supports lead offsets from 1 through 7 days.")

    previous_variables = [f"{name}_previous_day{lead}"
                          for lead in lead_days for name in ("temperature_2m", "precipitation")]
    previous = _get_payload(PREVIOUS_RUNS_URL, {
        "latitude": latitude, "longitude": longitude,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(previous_variables), "timezone": "UTC",
        "timeformat": "unixtime", "models": "best_match",
    }, timeout)
    reference = _get_payload(ARCHIVE_URL, {
        "latitude": latitude, "longitude": longitude,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "temperature_2m,precipitation", "timezone": "UTC",
        "timeformat": "unixtime", "models": "era5",
    }, timeout)
    rows = []
    reference_temperature = _series(reference, "temperature_2m")
    reference_rain = _series(reference, "precipitation")
    for lead in lead_days:
        forecast_temperature = _series(previous, f"temperature_2m_previous_day{lead}")
        forecast_rain = _series(previous, f"precipitation_previous_day{lead}")
        common_temperature = sorted(forecast_temperature.keys() & reference_temperature.keys())
        temperature = metrics([forecast_temperature[t] for t in common_temperature],
                              [reference_temperature[t] for t in common_temperature])
        common_rain = sorted(forecast_rain.keys() & reference_rain.keys())
        rainfall_hourly = metrics([forecast_rain[t] for t in common_rain],
                                  [reference_rain[t] for t in common_rain])
        rainfall_daily = _daily_totals(forecast_rain, reference_rain, zone)
        for variable, aggregation, result, unit in (
            ("temperature", "hourly", temperature, "°C"),
            ("precipitation", "hourly", rainfall_hourly, "mm"),
            ("precipitation", "full local-day totals", rainfall_daily, "mm/day"),
        ):
            rows.append({"lead_days": lead, "variable": variable, "aggregation": aggregation,
                         "unit": unit, **result})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--start-date", required=True, help="First valid date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Last valid date (YYYY-MM-DD).")
    parser.add_argument("--timezone", default="UTC", help="IANA timezone used for daily rainfall grouping.")
    parser.add_argument("--leads", nargs="+", type=int, default=list(range(1, 8)), help="Forecast lead-day offsets (1–7).")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    try:
        rows = evaluate(args.latitude, args.longitude, args.start_date, args.end_date,
                        args.timezone, args.leads, args.timeout)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    writer = csv.DictWriter(sys.stdout, fieldnames=("lead_days", "variable", "aggregation", "unit", "n", "mae", "rmse"))
    writer.writeheader()
    writer.writerows(rows)
    print("\nReference: ERA5 reanalysis, not surface-station observations. Forecast and reference grids differ.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
