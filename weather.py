"""Open-Meteo geocoding, observed conditions, and seven-day forecast."""
from datetime import datetime, timezone
import os
import re
import unicodedata

import pycountry
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEONAMES_POSTAL_URL = "https://secure.geonames.org/postalCodeSearchJSON"
SOURCE = "Open-Meteo Forecast API; current conditions are model-based estimates, not physical-station observations."
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
        "postal_code": location.get("postal_code", ""),
        "country_code": location.get("country_code", ""),
        "geocoding_provider": location.get("provider", "Open-Meteo"),
        "weather_provider": "Open-Meteo Forecast API",
        "weather_model": payload.get("model") or payload.get("models"),
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
