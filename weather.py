"""Open-Meteo geocoding, observed conditions, and seven-day forecast."""
from datetime import datetime, timezone

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE = "Open-Meteo Forecast API; geocoding based on GeoNames"
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Rain showers", 82: "Heavy showers",
    85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 97: "Heavy thunderstorm", 99: "Severe thunderstorm",
}


def search_locations(query, timeout=8):
    """Return genuine geocoding matches; errors are raised for UI handling."""
    query = str(query).strip()
    if len(query) < 2:
        raise ValueError("Enter at least two characters to search for a location.")
    try:
        response = requests.get(GEOCODE_URL, params={
            "name": query, "count": 8, "language": "en", "format": "json",
        }, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo geocoding request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo geocoding returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo geocoding returned an invalid response shape.")
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Open-Meteo geocoding returned an invalid results list.")
    if not results:
        raise ValueError(f"No location found for “{query}”.")
    locations = [{
        "name": place.get("name", "Unknown"),
        "admin1": place.get("admin1", ""),
        "country": place.get("country", ""),
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
        "timezone": place.get("timezone", "auto"),
    } for place in results if "latitude" in place and "longitude" in place]
    if not locations:
        raise RuntimeError("Open-Meteo returned location matches without usable coordinates.")
    return locations


def get_weather_bundle(location, timeout=10):
    """Retrieve observed current conditions and a 7-day real forecast."""
    try:
        response = requests.get(FORECAST_URL, params={
            "latitude": location["latitude"], "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,et0_fao_evapotranspiration",
            "forecast_days": 7, "timezone": location.get("timezone") or "auto",
            "wind_speed_unit": "kmh", "temperature_unit": "celsius", "precipitation_unit": "mm",
        }, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Open-Meteo forecast request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo forecast returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo forecast returned an invalid response shape.")
    current = payload.get("current")
    daily = payload.get("daily")
    required_current = ("time", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "weather_code")
    required_daily = ("time", "temperature_2m_min", "temperature_2m_max", "precipitation_sum", "weather_code")
    if not isinstance(current, dict) or any(k not in current for k in required_current):
        raise RuntimeError("Open-Meteo returned incomplete current conditions.")
    if not isinstance(daily, dict) or any(k not in daily for k in required_daily):
        raise RuntimeError("Open-Meteo returned an incomplete daily forecast.")
    if any(not isinstance(daily[name], list) for name in required_daily):
        raise RuntimeError("Open-Meteo returned malformed daily forecast arrays.")
    count = min(7, len(daily["time"]))
    if count == 0:
        raise RuntimeError("Open-Meteo returned no forecast days.")
    if any(len(daily[name]) < count for name in required_daily[1:]):
        raise RuntimeError("Open-Meteo returned mismatched forecast arrays.")
    daily_rows = []
    for i in range(count):
        code = daily["weather_code"][i]
        daily_rows.append({
            "date": daily["time"][i],
            "condition": WEATHER_CODES.get(code, f"Weather code {code}"),
            "weather_code": code,
            "temperature_min": daily["temperature_2m_min"][i],
            "temperature_max": daily["temperature_2m_max"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "wind_max_kmh": _optional(daily.get("wind_speed_10m_max"), i),
            "et0_mm": _optional(daily.get("et0_fao_evapotranspiration"), i),
        })
    return {
        "place": ", ".join(part for part in (location.get("name"), location.get("admin1"), location.get("country")) if part),
        "latitude": location["latitude"], "longitude": location["longitude"],
        "timezone": payload.get("timezone", location.get("timezone", "")),
        "current": {
            "time": current["time"], "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
            "wind_kmh": current["wind_speed_10m"],
            "condition": WEATHER_CODES.get(current["weather_code"], f"Weather code {current['weather_code']}"),
        },
        "daily": daily_rows,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE,
    }


def _optional(values, index):
    return values[index] if values is not None and index < len(values) else None


def get_current_weather(city):
    """Compatibility helper retaining the previous module's public function."""
    location = search_locations(city)[0]
    bundle = get_weather_bundle(location)
    return {"place": bundle["place"], **bundle["current"], "time": bundle["current"]["time"],
            "daily": bundle["daily"], "fetched_at_utc": bundle["fetched_at_utc"],
            "source": bundle["source"]}
