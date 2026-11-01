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
WEATHER_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌧️", 56: "🌧️", 57: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "🌧️", 66: "🌧️", 67: "🌧️",
    71: "🌨️", 73: "🌨️", 75: "❄️", 77: "🌨️", 80: "🌦️", 81: "🌧️",
    82: "🌧️", 85: "🌨️", 86: "❄️", 95: "⛈️", 96: "⛈️", 97: "⛈️", 99: "⛈️",
}

# Common country names and abbreviations accepted in a "city, country" query.
COUNTRY_CODES = {
    "india": "IN", "uk": "GB", "united kingdom": "GB", "great britain": "GB",
    "usa": "US", "us": "US", "united states": "US", "united states of america": "US",
    "japan": "JP", "pakistan": "PK", "canada": "CA", "australia": "AU",
}

# Keys are case-insensitive user spellings; each entry carries the preferred
# canonical result and alternative names to query. Extend this table to add
# more common city aliases without changing the search flow.
CITY_ALIASES = {
    "bangalore": {"canonical": "Bengaluru", "country_code": "IN", "alternates": ("Bengaluru",)},
    "bengaluru": {"canonical": "Bengaluru", "country_code": "IN", "alternates": ("Bangalore",)},
}


def _split_query(query):
    """Return the API name and optional ISO country code from a city, country query."""
    parts = [part.strip() for part in str(query).split(",") if part.strip()]
    if len(parts) > 1 and parts[-1].casefold() in COUNTRY_CODES:
        return ", ".join(parts[:-1]), COUNTRY_CODES[parts[-1].casefold()]
    return str(query).strip(), None


def _rank_location(place, query, canonical_name=None, preferred_country=None):
    """Rank preferred alias targets, then exact/relevant populated places."""
    wanted = query.casefold().strip()
    name = str(place.get("name", "")).casefold().strip()
    canonical = str(canonical_name or query).casefold().strip()
    country_code = str(place.get("country_code", "")).upper()
    feature = str(place.get("feature_code", "")).upper()
    # Open-Meteo/GeoNames PPLC is a national capital; PPLA* are regional seats.
    type_rank = 3 if feature == "PPLC" else 2 if feature.startswith("PPLA") else 1 if feature.startswith("PPL") else 0
    try:
        population = max(0, int(place.get("population") or 0))
    except (TypeError, ValueError):
        population = 0
    preferred = bool(preferred_country and country_code == preferred_country.upper() and name == canonical)
    return (preferred, name == canonical, name == wanted, type_rank, population)


def _location_identity(location):
    """Use coordinates to coalesce alias hits even when their IDs differ."""
    return ("coordinates", round(location["latitude"], 5), round(location["longitude"], 5))


def search_locations(query, timeout=8):
    """Search original and configured aliases, merge real results, and rank matches.

    A country qualifier is passed to every query and also enforced locally so
    unrelated same-name places can never leak into a country-specific search.
    """
    query = str(query).strip()
    name_query, country_code = _split_query(query)
    if len(name_query) < 2:
        raise ValueError("Enter at least two characters to search for a location.")
    alias = CITY_ALIASES.get(name_query.casefold())
    search_terms = list(dict.fromkeys((name_query, *(alias["alternates"] if alias else ()))))
    canonical_name = alias["canonical"] if alias else name_query
    preferred_country = alias.get("country_code") if alias else None

    raw_locations, failures = [], []
    successful_responses = 0
    had_results = False
    for term in search_terms:
        params = {"name": term, "count": 100, "language": "en", "format": "json"}
        if country_code:
            params["countryCode"] = country_code
        try:
            response = requests.get(GEOCODE_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            failures.append(str(exc))
            continue
        except ValueError as exc:
            failures.append(f"invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results", []), list):
            failures.append("invalid response shape")
            continue
        successful_responses += 1
        results = payload.get("results", [])
        had_results = had_results or bool(results)
        raw_locations.extend(results)

    if not successful_responses and failures:
        raise RuntimeError("Open-Meteo geocoding request failed: " + "; ".join(failures))

    locations_by_key = {}
    for place in raw_locations:
        if not isinstance(place, dict) or "latitude" not in place or "longitude" not in place:
            continue
        try:
            latitude, longitude = float(place["latitude"]), float(place["longitude"])
        except (TypeError, ValueError):
            continue
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            continue
        result_country = str(place.get("country_code", "")).upper()
        if country_code and result_country != country_code:
            continue
        location = {
            "name": place.get("name", "Unknown"), "admin1": place.get("admin1", ""),
            "country": place.get("country", ""), "latitude": latitude, "longitude": longitude,
            "timezone": place.get("timezone", "auto"), "country_code": result_country,
            "population": place.get("population", 0), "feature_code": place.get("feature_code", ""),
        }
        if place.get("id") is not None:
            location["id"] = place["id"]
        rank = _rank_location(location, name_query, canonical_name, preferred_country)
        key = _location_identity(location)
        previous = locations_by_key.get(key)
        # Keep the canonical alias record when two queries resolve to one point.
        if previous is None or rank > previous[0]:
            locations_by_key[key] = (rank, location)

    locations = [item[1] for item in locations_by_key.values()]
    if not locations and failures and successful_responses == 0:
        raise RuntimeError("Open-Meteo geocoding request failed: " + "; ".join(failures))
    if not locations and had_results:
        raise RuntimeError("Open-Meteo returned location matches without usable coordinates.")
    if not locations:
        raise ValueError(f'No location found for "{query}".')
    # Stable sort preserves API order among equally relevant/populated places.
    locations.sort(key=lambda loc: _rank_location(loc, name_query, canonical_name, preferred_country), reverse=True)
    return locations


def get_weather_bundle(location, timeout=10):
    """Retrieve current conditions and a seven-day forecast in local time."""
    try:
        response = requests.get(FORECAST_URL, params={
            "latitude": location["latitude"], "longitude": location["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code,cloud_cover,pressure_msl",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max,et0_fao_evapotranspiration",
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
    required_current = ("time", "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
                        "wind_speed_10m", "wind_direction_10m", "weather_code", "cloud_cover", "pressure_msl")
    required_daily = ("time", "temperature_2m_min", "temperature_2m_max", "precipitation_probability_max",
                      "precipitation_sum", "wind_speed_10m_max", "weather_code")
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
            "day": datetime.fromisoformat(daily["time"][i]).strftime("%A"),
            "condition": WEATHER_CODES.get(code, f"Weather code {code}"),
            "icon": WEATHER_ICONS.get(code, "🌡️"),
            "weather_code": code,
            "precipitation_probability_pct": daily["precipitation_probability_max"][i],
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
            "apparent_temperature_c": current["apparent_temperature"],
            "humidity_pct": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
            "wind_kmh": current["wind_speed_10m"],
            "wind_direction_deg": current["wind_direction_10m"],
            "cloud_cover_pct": current["cloud_cover"],
            "pressure_hpa": current["pressure_msl"],
            "icon": WEATHER_ICONS.get(current["weather_code"], "🌡️"),
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
