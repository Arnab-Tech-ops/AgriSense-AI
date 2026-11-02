"""Open-Meteo geocoding, model-based current weather, and forecasts."""
from datetime import datetime, timezone
import math
import os
import re
import time
import unicodedata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pycountry
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEONAMES_POSTAL_URL = "https://secure.geonames.org/postalCodeSearchJSON"
SOURCE = "Open-Meteo Forecast API; best_match model output, not physical-station observations."
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 97: "Heavy thunderstorm", 99: "Thunderstorm with heavy hail",
}
WEATHER_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌧️", 56: "🌧️", 57: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "🌧️", 66: "🌧️", 67: "🌧️",
    71: "🌨️", 73: "🌨️", 75: "❄️", 77: "🌨️", 80: "🌦️", 81: "🌧️",
    82: "🌧️", 85: "🌨️", 86: "❄️", 95: "⛈️", 96: "⛈️", 97: "⛈️", 99: "⛈️",
}

# ISO 3166-1 names/codes are sourced from pycountry's bundled ISO data.
COUNTRY_NAMES = {country.alpha_2: country.name for country in pycountry.countries}
COUNTRY_CODES = {}
for _country in pycountry.countries:
    for _value in (_country.name, getattr(_country, "official_name", ""),
                   getattr(_country, "common_name", ""), _country.alpha_2, _country.alpha_3):
        if _value:
            COUNTRY_CODES["".join(c for c in unicodedata.normalize("NFKD", _value).casefold()
                                  if not unicodedata.combining(c))] = _country.alpha_2
COUNTRY_CODES.update({"uk": "GB", "great britain": "GB", "britain": "GB",
                      "usa": "US", "u.s.": "US", "u.s.a.": "US",
                      "south korea": "KR", "north korea": "KP", "russia": "RU",
                      "iran": "IR", "syria": "SY", "tanzania": "TZ",
                      "bolivia": "BO", "venezuela": "VE", "moldova": "MD",
                      "laos": "LA", "brunei": "BN", "cape verde": "CV",
                      "ivory coast": "CI", "czech republic": "CZ", "taiwan": "TW"})


def _fold(value):
    """Case/diacritic-insensitive comparison key; query text itself stays intact."""
    value = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def _postcode_key(value):
    """Normalize separators for matching without changing the submitted query."""
    return re.sub(r"[\W_]+", "", _fold(value), flags=re.UNICODE)


def _country_code(value):
    if not value:
        return None
    return COUNTRY_CODES.get(_fold(value).strip())


def country_options():
    """Return all ISO alpha-2 countries and territories as (code, name) pairs."""
    return sorted(COUNTRY_NAMES.items(), key=lambda item: item[1].casefold())

# Keys are case-insensitive user spellings; each entry carries the preferred
# canonical result and alternative names to query. Extend this table to add
# more common city aliases without changing the search flow.
CITY_ALIASES = {
    "bangalore": {"canonical": "Bengaluru", "country_code": "IN", "alternates": ("Bengaluru",)},
    "bengaluru": {"canonical": "Bengaluru", "country_code": "IN", "alternates": ("Bangalore",)},
}


def _split_query(query, selected_country=None):
    """Split city/area/country input while passing explicit countries as ISO codes."""
    parts = [re.sub(r"\s+", " ", part).strip() for part in str(query).split(",") if part.strip()]
    explicit_code = _country_code(parts[-1]) if len(parts) > 1 else None
    if explicit_code:
        parts.pop()
    selected_code = _country_code(selected_country) if selected_country else None
    if selected_country and not selected_code:
        raise ValueError("Choose a valid ISO 3166-1 alpha-2 country.")
    if explicit_code and selected_code and explicit_code != selected_code:
        raise ValueError("The country in the search text conflicts with the selected country filter.")
    return ", ".join(parts), explicit_code or selected_code


def _rank_location(place, query, canonical_name=None, preferred_country=None, postal_query=False):
    """Rank exact postcode hits before canonical aliases and relevant places."""
    wanted = _fold(query).strip()
    name = _fold(place.get("name", "")).strip()
    canonical = _fold(canonical_name or query).strip()
    country_code = str(place.get("country_code", "")).upper()
    postcodes = place.get("postcodes") or ([place.get("postal_code")] if place.get("postal_code") else [])
    exact_postcode = postal_query and any(_postcode_key(code) == _postcode_key(query) for code in postcodes)
    feature = str(place.get("feature_code", "")).upper()
    # Open-Meteo/GeoNames PPLC is a national capital; PPLA* are regional seats.
    type_rank = 3 if feature == "PPLC" else 2 if feature.startswith("PPLA") else 1 if feature.startswith("PPL") else 0
    try:
        population = max(0, int(place.get("population") or 0))
    except (TypeError, ValueError):
        population = 0
    preferred = bool(preferred_country and country_code == preferred_country.upper() and name == canonical)
    return (exact_postcode, preferred, name == canonical, name == wanted, type_rank, population)


def _location_identity(location):
    """Use coordinates to coalesce alias hits even when their IDs differ."""
    return ("coordinates", round(location["latitude"], 5), round(location["longitude"], 5))


def _primary_location(place):
    postcodes = place.get("postcodes") or []
    if isinstance(postcodes, str):
        postcodes = [postcodes]
    return {
        "name": str(place.get("name") or "Unknown"),
        "admin1": str(place.get("admin1") or ""),
        "country": str(place.get("country") or COUNTRY_NAMES.get(str(place.get("country_code", "")).upper(), "")),
        "country_code": str(place.get("country_code") or "").upper(),
        "postal_code": postcodes[0] if postcodes else "",
        "postcodes": postcodes,
        "latitude": float(place["latitude"]), "longitude": float(place["longitude"]),
        "timezone": place.get("timezone") or "auto",
        "population": place.get("population", 0), "feature_code": place.get("feature_code", ""),
        "id": place.get("id"), "provider": "Open-Meteo", "match_type": "location match",
    }


def _geonames_locations(query, country_code, username, timeout, postal_query=False):
    """Optional postal/place fallback. GeoNames requires a registered username."""
    params = {"maxRows": 100, "style": "FULL", "charset": "UTF8", "username": username}
    params["postalcode" if postal_query else "placename"] = query
    if country_code:
        params["country"] = country_code
    try:
        response = requests.get(GEONAMES_POSTAL_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        # Request exception strings can include the full URL, including the
        # username query parameter. Report the safe error class only.
        raise RuntimeError(f"GeoNames fallback request failed ({type(exc).__name__}).") from exc
    except ValueError as exc:
        raise RuntimeError("GeoNames fallback returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("GeoNames fallback returned an invalid response shape.")
    if payload.get("status"):
        reason = payload["status"].get("message", "provider rejected the request")
        reason = re.sub(re.escape(username), "[redacted]", str(reason), flags=re.IGNORECASE)
        raise RuntimeError(f"GeoNames fallback failed: {reason}")
    records = payload.get("postalCodes", [])
    if not isinstance(records, list):
        raise RuntimeError("GeoNames fallback returned malformed postal-code results.")
    results = []
    for place in records:
        if not isinstance(place, dict):
            continue
        try:
            latitude, longitude = float(place["lat"]), float(place["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            continue
        code = str(place.get("countryCode") or "").upper()
        if country_code and code != country_code:
            continue
        postal_code = str(place.get("postalCode") or "")
        exact = _postcode_key(postal_code) == _postcode_key(query)
        results.append({
            "name": str(place.get("placeName") or place.get("adminName2") or "Unknown"),
            "admin1": str(place.get("adminName1") or ""),
            "country": str(place.get("countryName") or COUNTRY_NAMES.get(code, "")),
            "country_code": code, "postal_code": postal_code,
            "postcodes": [postal_code] if postal_code else [],
            "latitude": latitude, "longitude": longitude, "timezone": "auto",
            "population": place.get("population", 0), "feature_code": "",
            "id": place.get("geonameId"), "provider": "GeoNames",
            "match_type": "exact postal-code match" if exact else "provider match (verify postal-code coverage)",
        })
    return results


def _geonames_username(value=None):
    return (value or os.getenv("GEONAMES_USERNAME", "")).strip()


def search_locations(query, timeout=8, country_code=None, geonames_username=None):
    """Search original and configured aliases, merge real results, and rank matches.

    A country qualifier is passed to every query and also enforced locally so
    unrelated same-name places can never leak into a country-specific search.
    """
    original_query = re.sub(r"\s+", " ", str(query)).strip()
    if not original_query:
        raise ValueError("Enter a city, locality, or postal code to search.")
    name_query, country_code = _split_query(original_query, country_code)
    if len(name_query) < 2:
        raise ValueError("Enter at least two characters to search for a location.")
    query_place = name_query.split(",", 1)[0].strip()
    suffix = name_query[len(query_place):]
    alias = CITY_ALIASES.get(_fold(query_place).strip())
    search_terms = [name_query]
    if alias:
        search_terms.extend(f"{term}{suffix}" for term in alias["alternates"])
    search_terms = list(dict.fromkeys(search_terms))
    canonical_name = alias["canonical"] if alias else query_place
    preferred_country = alias.get("country_code") if alias else None
    postal_query = any(char.isdigit() for char in query_place)

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
        try:
            location = _primary_location(place)
        except (TypeError, ValueError):
            continue
        if country_code and location["country_code"] != country_code:
            continue
        location["match_type"] = "exact postal-code match" if any(
            _postcode_key(code) == _postcode_key(query_place) for code in location["postcodes"]
        ) else "location match"
        rank = _rank_location(location, query_place, canonical_name, preferred_country, postal_query)
        key = _location_identity(location)
        previous = locations_by_key.get(key)
        # Keep the canonical alias record when two queries resolve to one point.
        if previous is None or rank > previous[0]:
            locations_by_key[key] = (rank, location)

    locations = [item[1] for item in locations_by_key.values()]
    # A digit-bearing query is treated as a postal-code candidate. Do not
    # mistake a similarly named town for an exact postcode result.
    exact_locations = [loc for loc in locations if loc["match_type"] == "exact postal-code match"]
    username = _geonames_username(geonames_username)
    needs_postal_fallback = postal_query and not exact_locations
    if username and (not locations or needs_postal_fallback):
        try:
            fallback = _geonames_locations(name_query, country_code, username, timeout, postal_query)
            if postal_query:
                fallback.sort(key=lambda loc: _rank_location(loc, query_place, canonical_name, preferred_country, True), reverse=True)
            # Preserve provider-specific metadata and coalesce duplicate coordinates.
            for loc in fallback:
                key = _location_identity(loc)
                rank = _rank_location(loc, query_place, canonical_name, preferred_country, postal_query)
                previous = locations_by_key.get(key)
                if previous is None or rank > previous[0]:
                    locations_by_key[key] = (rank, loc)
            locations = [item[1] for item in locations_by_key.values()]
        except RuntimeError as exc:
            failures.append(str(exc))
    if postal_query:
        exact_locations = [loc for loc in locations if loc["match_type"] == "exact postal-code match"]
        if exact_locations:
            locations = exact_locations
        elif username:
            # Retain only GeoNames records that can be clearly labeled as
            # provider matches/partial coverage; discard fuzzy city hits.
            locations = [loc for loc in locations if loc.get("provider") == "GeoNames"]
        elif not username:
            locations = []
    if not locations and failures and successful_responses == 0:
        raise RuntimeError("Open-Meteo geocoding request failed: " + "; ".join(failures))
    if not locations and had_results and not locations_by_key:
        raise RuntimeError("Open-Meteo returned location matches without usable coordinates.")
    if not locations:
        if failures and username:
            raise RuntimeError("No location match was found and the configured GeoNames fallback failed: " + "; ".join(failures))
        if failures and successful_responses == 0:
            raise RuntimeError("Open-Meteo geocoding request failed: " + "; ".join(failures))
        suffix = (
            " Postal-code fallback is unavailable until GEONAMES_USERNAME is set as a local environment variable "
            "or in Streamlit Community Cloud → My Apps → AgriSense AI → Settings/Secrets. City and locality searches remain available."
            if postal_query and not username else ""
        )
        raise ValueError(f'No location found for "{original_query}" (no exact postal-code match when applicable).{suffix}')
    # Stable sorting preserves API order among equally relevant candidates.
    locations.sort(key=lambda loc: _rank_location(loc, query_place, canonical_name, preferred_country, postal_query), reverse=True)
    return locations


def clear_weather_state(state):
    """Reset search, selection, and fetched weather; usable from a UI callback."""
    state["weather_query"] = ""
    state["weather_country_filter"] = ""
    state["weather_locations"] = []
    state["weather_selected"] = 0
    state["weather_bundle"] = None
    state["weather_bundle_location"] = None
    state["weather_error"] = None


def next_weather_refresh(state):
    """Increment the cache-busting key so a manual refresh makes a new request."""
    state["weather_refresh_key"] = int(state.get("weather_refresh_key", 0)) + 1
    return state["weather_refresh_key"]


CURRENT_VARIABLES = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,is_day,precipitation,rain,showers,"
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code,cloud_cover,pressure_msl,visibility,"
    "et0_fao_evapotranspiration,vapour_pressure_deficit,soil_temperature_0cm,soil_temperature_6cm,"
    "soil_temperature_18cm,soil_temperature_54cm,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,"
    "soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm"
)
HOURLY_VARIABLES = (
    "temperature_2m,apparent_temperature,precipitation_probability,precipitation,relative_humidity_2m,"
    "wind_speed_10m,weather_code"
)
DAILY_VARIABLES = (
    "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,"
    "precipitation_probability_max,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,uv_index_max,"
    "sunrise,sunset,sunshine_duration,et0_fao_evapotranspiration"
)


def _request_forecast_json(params, timeout, retries=1):
    """Request forecast data, retrying only transient network/server failures."""
    for attempt in range(retries + 1):
        try:
            response = requests.get(FORECAST_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            transient = (isinstance(exc, (requests.Timeout, requests.ConnectionError))
                         and not isinstance(exc, requests.exceptions.SSLError))
            transient = transient or status == 429 or (status is not None and status >= 500)
            if transient and attempt < retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise RuntimeError(f"Open-Meteo forecast request failed ({type(exc).__name__}).") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Open-Meteo forecast returned invalid JSON.") from exc
        return payload, datetime.now(timezone.utc)


def _finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp(value, zone):
    """Parse Open-Meteo Unix timestamps; accept ISO values for older cached/test payloads."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return datetime.fromtimestamp(value, timezone.utc).astimezone(zone)
    if not isinstance(value, str):
        raise ValueError("Invalid weather timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid weather timestamp.") from exc
    return parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)


def _number_at(data, field, index):
    values = data.get(field)
    return _finite_number(values[index]) if isinstance(values, list) and index < len(values) else None


def _weather_code(value):
    number = _finite_number(value)
    return int(number) if number is not None and number.is_integer() else None


def _validate_units(payload):
    """Reject explicit API unit mismatches; missing unit metadata remains optional."""
    expected = {
        "temperature_2m": {"°C"}, "apparent_temperature": {"°C"},
        "relative_humidity_2m": {"%"}, "precipitation": {"mm"}, "rain": {"mm"},
        "showers": {"mm"}, "wind_speed_10m": {"km/h"}, "wind_gusts_10m": {"km/h"},
        "wind_direction_10m": {"°"}, "cloud_cover": {"%"}, "pressure_msl": {"hPa"},
        "visibility": {"m"}, "vapour_pressure_deficit": {"kPa"},
        "temperature_2m_max": {"°C"}, "temperature_2m_min": {"°C"},
        "apparent_temperature_max": {"°C"}, "apparent_temperature_min": {"°C"},
        "precipitation_probability_max": {"%"}, "precipitation_sum": {"mm"},
        "wind_speed_10m_max": {"km/h"}, "wind_gusts_10m_max": {"km/h"},
        "et0_fao_evapotranspiration": {"mm"}, "sunshine_duration": {"s"},
    }
    for units_key in ("current_units", "hourly_units", "daily_units"):
        units = payload.get(units_key)
        if not isinstance(units, dict):
            continue
        for variable, allowed in expected.items():
            if variable in units and units[variable] not in allowed:
                raise RuntimeError(f"Open-Meteo returned an unexpected unit for {variable}.")


def _freshness(age_seconds):
    """Heuristic based on the API's valid-time timestamp, not model-run age."""
    if age_seconds is None or age_seconds < 0:
        return "unknown"
    if age_seconds <= 90 * 60:
        return "fresh"
    if age_seconds <= 6 * 60 * 60:
        return "delayed"
    return "stale"


def get_weather_bundle(location, timeout=10):
    """Retrieve best-match current, hourly, daily, and agricultural forecast data."""
    latitude = _finite_number(location.get("latitude"))
    longitude = _finite_number(location.get("longitude"))
    if latitude is None or longitude is None or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Weather coordinates must be valid WGS84 latitude and longitude values.")
    requested_timezone = location.get("timezone") or "auto"
    try:
        zone = ZoneInfo(requested_timezone) if requested_timezone != "auto" else None
    except (ZoneInfoNotFoundError, TypeError):
        raise ValueError("The selected location has an invalid IANA time zone.")

    params = {
        "latitude": latitude, "longitude": longitude,
        "current": CURRENT_VARIABLES,
        "hourly": HOURLY_VARIABLES,
        "daily": DAILY_VARIABLES,
        # A buffer is needed because some stitched model feeds return already
        # elapsed hourly timestamps before the API's current valid time.
        # Include a buffer because the API can include the already-started
        # local calendar day before the current valid date in its daily series.
        "forecast_days": 8, "forecast_hours": 48,
        "models": "best_match",
        "timezone": requested_timezone,
        "timeformat": "unixtime",
        "wind_speed_unit": "kmh", "temperature_unit": "celsius", "precipitation_unit": "mm",
    }
    payload, retrieved_at = _request_forecast_json(params, timeout)
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo forecast returned an invalid response shape.")
    _validate_units(payload)
    current = payload.get("current")
    hourly = payload.get("hourly")
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
    day_count = min(8, len(daily["time"]))
    if day_count == 0:
        raise RuntimeError("Open-Meteo returned no forecast days.")
    if any(len(daily[name]) < day_count for name in required_daily[1:]):
        raise RuntimeError("Open-Meteo returned mismatched forecast arrays.")
    response_timezone = payload.get("timezone") or requested_timezone
    if response_timezone == "auto":
        raise RuntimeError("Open-Meteo did not return the selected location's resolved timezone.")
    try:
        zone = ZoneInfo(response_timezone)
        current_time = _timestamp(current["time"], zone)
    except (ZoneInfoNotFoundError, TypeError, ValueError, OverflowError, OSError) as exc:
        raise RuntimeError("Open-Meteo returned an invalid timezone or current timestamp.") from exc

    now = datetime.now(timezone.utc)
    valid_age = (now - current_time.astimezone(timezone.utc)).total_seconds()
    valid_age = valid_age if valid_age >= 0 else None
    current_values = {key: _finite_number(current.get(key)) for key in (
        "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation", "rain", "showers",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cloud_cover", "pressure_msl",
        "visibility", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "soil_temperature_0cm",
        "soil_temperature_6cm", "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
        "soil_moisture_27_to_81cm",
    )}
    code = _weather_code(current.get("weather_code"))
    current_row = {
        "time": current_time.strftime("%Y-%m-%d %H:%M %Z"),
        "time_utc": current_time.astimezone(timezone.utc).isoformat(timespec="minutes"),
        "interval_seconds": _finite_number(current.get("interval")),
        "temperature_c": current_values["temperature_2m"],
        "apparent_temperature_c": current_values["apparent_temperature"],
        "humidity_pct": current_values["relative_humidity_2m"],
        "precipitation_mm": current_values["precipitation"], "rain_mm": current_values["rain"],
        "showers_mm": current_values["showers"], "wind_kmh": current_values["wind_speed_10m"],
        "wind_direction_deg": current_values["wind_direction_10m"],
        "wind_gusts_kmh": current_values["wind_gusts_10m"], "cloud_cover_pct": current_values["cloud_cover"],
        "pressure_hpa": current_values["pressure_msl"],
        "visibility_km": current_values["visibility"] / 1000 if current_values["visibility"] is not None else None,
        "is_day": _weather_code(current.get("is_day")), "weather_code": code,
        "icon": WEATHER_ICONS.get(code, "🌡️") if code is not None else "🌡️",
        "condition": WEATHER_CODES.get(code, f"Weather code {code}" if code is not None else "Unavailable"),
        "agriculture": {
            "et0_mm": current_values["et0_fao_evapotranspiration"],
            "vpd_kpa": current_values["vapour_pressure_deficit"],
            "soil_temperature_c": {"0 cm": current_values["soil_temperature_0cm"],
                                   "6 cm": current_values["soil_temperature_6cm"],
                                   "18 cm": current_values["soil_temperature_18cm"],
                                   "54 cm": current_values["soil_temperature_54cm"]},
            "soil_moisture_m3_m3": {"0–1 cm": current_values["soil_moisture_0_to_1cm"],
                                   "1–3 cm": current_values["soil_moisture_1_to_3cm"],
                                   "3–9 cm": current_values["soil_moisture_3_to_9cm"],
                                   "9–27 cm": current_values["soil_moisture_9_to_27cm"],
                                   "27–81 cm": current_values["soil_moisture_27_to_81cm"]},
        },
    }

    hourly_rows = []
    if isinstance(hourly, dict) and isinstance(hourly.get("time"), list):
        for i, raw_time in enumerate(hourly["time"]):
            try:
                point_time = _timestamp(raw_time, zone)
            except (TypeError, ValueError, OverflowError, OSError):
                raise RuntimeError("Open-Meteo returned an invalid hourly timestamp.")
            if point_time.astimezone(timezone.utc) < now:
                continue
            hourly_code = _weather_code(_number_at(hourly, "weather_code", i))
            hourly_rows.append({
                "time": point_time, "time_label": point_time.strftime("%a %H:%M"),
                "temperature_c": _number_at(hourly, "temperature_2m", i),
                "apparent_temperature_c": _number_at(hourly, "apparent_temperature", i),
                "precipitation_probability_pct": _number_at(hourly, "precipitation_probability", i),
                "precipitation_mm": _number_at(hourly, "precipitation", i),
                "humidity_pct": _number_at(hourly, "relative_humidity_2m", i),
                "wind_kmh": _number_at(hourly, "wind_speed_10m", i),
                "weather_code": hourly_code,
                "condition": WEATHER_CODES.get(hourly_code, "Unavailable") if hourly_code is not None else "Unavailable",
            })
            if len(hourly_rows) == 24:
                break
    daily_rows = []
    for i in range(day_count):
        try:
            day_time = _timestamp(daily["time"][i], zone)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise RuntimeError("Open-Meteo returned an invalid daily timestamp.") from exc
        day_code = _weather_code(_number_at(daily, "weather_code", i))
        if day_time.date() < current_time.date():
            continue
        daily_rows.append({
            "date": day_time.date().isoformat(), "day": day_time.strftime("%A"),
            "condition": WEATHER_CODES.get(day_code, f"Weather code {day_code}" if day_code is not None else "Unavailable"),
            "icon": WEATHER_ICONS.get(day_code, "🌡️") if day_code is not None else "🌡️",
            "weather_code": day_code,
            "temperature_min": _number_at(daily, "temperature_2m_min", i),
            "temperature_max": _number_at(daily, "temperature_2m_max", i),
            "apparent_temperature_min": _number_at(daily, "apparent_temperature_min", i),
            "apparent_temperature_max": _number_at(daily, "apparent_temperature_max", i),
            "precipitation_probability_pct": _number_at(daily, "precipitation_probability_max", i),
            "precipitation_mm": _number_at(daily, "precipitation_sum", i),
            "wind_max_kmh": _number_at(daily, "wind_speed_10m_max", i),
            "wind_gust_max_kmh": _number_at(daily, "wind_gusts_10m_max", i),
            "uv_index_max": _number_at(daily, "uv_index_max", i),
            "sunrise": _local_clock(_number_at(daily, "sunrise", i), zone),
            "sunset": _local_clock(_number_at(daily, "sunset", i), zone),
            "sunshine_hours": (_number_at(daily, "sunshine_duration", i) / 3600
                               if _number_at(daily, "sunshine_duration", i) is not None else None),
            "et0_mm": _number_at(daily, "et0_fao_evapotranspiration", i),
        })
        if len(daily_rows) == 7:
            break
    if not daily_rows:
        raise RuntimeError("Open-Meteo returned no daily forecast for the current local date or later.")

    return {
        "place": ", ".join(part for part in (location.get("name"), location.get("admin1"), location.get("country")) if part),
        "latitude": latitude, "longitude": longitude,
        "timezone": response_timezone,
        "postal_code": location.get("postal_code", ""), "country_code": location.get("country_code", ""),
        "geocoding_provider": location.get("provider", "Open-Meteo"),
        "weather_provider": "Open-Meteo Forecast API",
        "model_selection": "best_match",
        "weather_model": payload.get("model") or payload.get("models"),
        "model_run_initialization_utc": None,
        "model_run_availability_utc": None,
        "model_run_metadata_note": "Open-Meteo does not identify the individual models/runs used by best_match in this response; run age is unavailable.",
        "current": current_row, "hourly": hourly_rows, "daily": daily_rows,
        "valid_time_age_seconds": valid_age,
        "valid_time_freshness": _freshness(valid_age),
        "retrieved_at_utc": retrieved_at.isoformat(timespec="seconds"),
        "fetched_at_utc": retrieved_at.isoformat(timespec="seconds"),
        "source": SOURCE,
    }


def _local_clock(timestamp, zone):
    if timestamp is None:
        return None
    try:
        return _timestamp(timestamp, zone).strftime("%H:%M %Z")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _optional(values, index):
    return values[index] if values is not None and index < len(values) else None


def get_current_weather(city):
    """Compatibility helper retaining the previous module's public function."""
    location = search_locations(city)[0]
    bundle = get_weather_bundle(location)
    return {"place": bundle["place"], **bundle["current"], "time": bundle["current"]["time"],
            "daily": bundle["daily"], "fetched_at_utc": bundle["fetched_at_utc"],
            "source": bundle["source"]}
