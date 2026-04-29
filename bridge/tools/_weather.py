"""Weather tool + geocoding helpers."""
from __future__ import annotations

import datetime
import logging

import httpx

log = logging.getLogger("bridge.tools")


_WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains",
    80: "rain showers", 81: "heavy rain showers", 82: "violent rain",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

# Weather-code -> emoji. Tuned for glanceability: one glyph that
# matches the condition at reading distance. Unknown/missing codes
# fall back to a neutral cloud.
_WEATHER_EMOJI = {
    0: "\u2600\ufe0f",       # ☀️
    1: "\U0001F324\ufe0f",   # 🌤
    2: "\u26C5",             # ⛅
    3: "\u2601\ufe0f",       # ☁️
    45: "\U0001F32B\ufe0f",  # 🌫
    48: "\U0001F32B\ufe0f",
    51: "\U0001F326\ufe0f",  # 🌦
    53: "\U0001F326\ufe0f",
    55: "\U0001F327\ufe0f",  # 🌧
    61: "\U0001F327\ufe0f",
    63: "\U0001F327\ufe0f",
    65: "\U0001F327\ufe0f",
    71: "\U0001F328\ufe0f",  # 🌨
    73: "\U0001F328\ufe0f",
    75: "\U0001F328\ufe0f",
    77: "\u2744\ufe0f",       # ❄️
    80: "\U0001F327\ufe0f",
    81: "\U0001F327\ufe0f",
    82: "\u26C8\ufe0f",       # ⛈
    85: "\U0001F328\ufe0f",
    86: "\U0001F328\ufe0f",
    95: "\u26C8\ufe0f",       # ⛈
    96: "\u26C8\ufe0f",
    99: "\u26C8\ufe0f",
}


def weather_emoji(code: int | None) -> str:
    if code is None:
        return "\U0001F324\ufe0f"  # 🌤 default
    return _WEATHER_EMOJI.get(int(code), "\U0001F324\ufe0f")


# Phrases that mean "wherever I am" — these route to IP-based location
# inference rather than Open-Meteo's geocoder. Lowercased and stripped
# before comparison, so "Here." or " My location " both match.
_VAGUE_LOCATION_PHRASES: frozenset[str] = frozenset({
    "",
    "here",
    "my location",
    "current location",
    "where i am",
    "my area",
    "nearby",
    "local",
    "local weather",
})


def _is_vague_location(raw: str | None) -> bool:
    """True when the caller didn't really give us a place — either empty
    or one of the common 'wherever I am' stand-ins. Used to decide
    whether to fall back to IP geolocation before hitting Open-Meteo."""
    if raw is None:
        return True
    s = raw.strip().lower().rstrip(".!?")
    return s in _VAGUE_LOCATION_PHRASES


def _looks_like_coords(raw: str) -> bool:
    """True if raw parses cleanly as `lat,lon`."""
    if "," not in raw:
        return False
    try:
        parts = raw.split(",", 1)
        if len(parts) != 2:
            return False
        for p in parts:
            float(p.strip())
        return True
    except ValueError:
        return False


def _ip_geolocate(client: "httpx.Client") -> tuple[float, float, str] | None:
    """Look up the requester's approximate location by public IP.

    First consults the user_location override (assistant.yaml + env
    vars via bridge.location) — if the user has configured a city /
    region / country / lat / lon, those win without any network call.
    Falls back to ipwho.is (HTTPS, no auth, 10k req/month free). The
    previous ipapi.co backend started paywalling residential IPs with
    an HTML response that broke resp.json(); ipwho.is has been more
    reliable for small-volume callers.

    Returns (latitude, longitude, display_name) or None on failure.
    """
    # Check the user override first — this matches bridge.location's
    # behavior and avoids a wasted network call when the user has a
    # static home location configured.
    try:
        from bridge.location import _load_user_override
        ov = _load_user_override()
        parts = [p for p in (ov.get("city", ""), ov.get("region", ""), ov.get("country", "")) if p]
        lat_ov = ov.get("latitude")
        lon_ov = ov.get("longitude")
        if parts and lat_ov is not None and lon_ov is not None:
            return float(lat_ov), float(lon_ov), ", ".join(parts)
    except Exception:
        # Override load failure is non-fatal; fall through to HTTP.
        pass

    for attempt in (1, 2):
        try:
            resp = client.get("https://ipwho.is/", timeout=8.0)
            if resp.status_code != 200:
                if attempt == 1:
                    continue
                return None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            if attempt == 1:
                continue
            return None
        if not isinstance(data, dict) or not data.get("success", False):
            if attempt == 1:
                continue
            return None
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is None or lon is None:
            if attempt == 1:
                continue
            return None
        # ipwho.is fields: city, region (full state/province name),
        # country (full country name).
        city = data.get("city") or ""
        region = data.get("region") or ""
        country = data.get("country") or ""
        parts = [p for p in (city, region, country) if p]
        place_name = ", ".join(parts) if parts else f"{float(lat):.2f},{float(lon):.2f}"
        return float(lat), float(lon), place_name
    return None


# Common US-state abbreviation → full name, used by the geocoder
# disambiguator below so "University Park, PA" can find the
# Pennsylvania entry instead of falling through to Florida. Substring
# matching alone misses this because the geocoder returns the full
# state name (e.g. "Pennsylvania") which doesn't contain "pa".
_US_STATE_ABBREV: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}


def _geocode_candidates(location: str) -> list[str]:
    """Yield progressively-simpler variants of a location string.

    Open-Meteo's geocoder is finicky about overly specific inputs —
    "University Park, Pennsylvania, United States" returns no results
    even though "University Park" alone hits. We try the full string
    first (best fidelity), then strip a trailing country, then fall
    back to just the first comma-separated token (the city/town).
    Order matters: the first match wins so the most-specific variant
    is preferred.
    """
    s = (location or "").strip()
    if not s:
        return []
    out: list[str] = [s]

    # Strip a trailing "United States" / "USA" / "U.S." (and similar
    # common country suffixes) — these are the part the geocoder tends
    # to choke on when passed alongside a region name.
    parts = [p.strip() for p in s.split(",") if p.strip()]
    _country_aliases = {
        "united states", "usa", "u.s.", "u.s.a.", "us", "america",
        "united kingdom", "uk", "u.k.",
    }
    if len(parts) >= 2 and parts[-1].lower() in _country_aliases:
        without_country = ", ".join(parts[:-1])
        if without_country and without_country not in out:
            out.append(without_country)

    # Last resort: just the first comma-separated token (usually the
    # city/town). Distinct enough most of the time.
    if parts and parts[0] not in out:
        out.append(parts[0])
    return out


def _geocode_open_meteo(
    client: "httpx.Client", location: str,
) -> tuple[float, float, str] | None:
    """Resolve a city/place name via Open-Meteo's geocoder.

    Returns (lat, lon, display_name) or None if no candidate variant
    matched. Strategy:
      1. Try each progressively-simpler variant (full → no country →
         bare city) with the geocoder.
      2. For each variant, ask for up to 10 results and prefer one
         whose ``admin1`` (region/state) or ``country`` matches the
         hints from the *original* input. That way "University Park,
         Pennsylvania, United States" doesn't fall back to the Palo
         Alto neighborhood when the geocoder rejects the full string
         but accepts the bare city — we keep the Pennsylvania context
         and disambiguate among 10 candidates instead of just taking
         the first.
    """
    raw_parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    # Disambiguation hints, ordered by specificity: the part right
    # after the city (raw_parts[1]) is usually a region/state — that
    # narrows results meaningfully. The trailing country (raw_parts[-1])
    # is much weaker because most results in the US set will satisfy it.
    region_hint = raw_parts[1].lower() if len(raw_parts) >= 2 else ""
    country_hint = raw_parts[-1].lower() if len(raw_parts) >= 3 else ""

    # Expand common US state abbreviations to their full name so
    # "PA" matches admin1="Pennsylvania" — a substring check alone
    # misses this because "pa" isn't a substring of "pennsylvania".
    region_hint_expanded = _US_STATE_ABBREV.get(region_hint.upper(), region_hint)
    region_hint_expanded = region_hint_expanded.lower()

    def _score(r: dict) -> int:
        """Higher is better. Region match dominates country match.
        Empty hints don't contribute (avoids ``"" in bag`` always-true)."""
        bag = " ".join(filter(None, [
            str(r.get("admin1") or ""),
            str(r.get("admin2") or ""),
            str(r.get("country") or ""),
            str(r.get("country_code") or ""),
        ])).lower()
        score = 0
        if region_hint_expanded and region_hint_expanded in bag:
            score += 10
        elif region_hint and region_hint in bag:
            # Fall back to raw substring (e.g. "tokyo" in "tokyo, japan")
            score += 8
        if country_hint and country_hint in bag:
            score += 1
        return score

    for variant in _geocode_candidates(location):
        try:
            resp = client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": variant, "count": 10, "language": "en"},
                timeout=8.0,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            continue
        results = data.get("results") or []
        if not results:
            continue

        # If we have any disambiguation hint, score each result and pick
        # the highest. Tie → geocoder's order (most prominent first).
        # Otherwise just take the first.
        if region_hint or country_hint:
            scored = sorted(
                enumerate(results),
                key=lambda t: (-_score(t[1]), t[0]),
            )
            chosen = scored[0][1]
        else:
            chosen = results[0]

        lat = chosen.get("latitude")
        lon = chosen.get("longitude")
        if lat is None or lon is None:
            continue
        display = chosen.get("name") or variant
        if chosen.get("admin1"):
            display += f", {chosen['admin1']}"
        if chosen.get("country"):
            display += f", {chosen['country']}"
        return float(lat), float(lon), display
    return None


def _fetch_weather_data(
    client: "httpx.Client", lat: float, lon: float, place_name: str,
) -> dict:
    """Call Open-Meteo's forecast endpoint and return structured data.

    Returned dict keys (on success):
        place_name, lat, lon, temp_c, feels_like_c, condition,
        weather_code, wind_kmh, humidity, precipitation_mm, emoji.
    On failure:
        {"error": "..."}
    """
    try:
        resp = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": ",".join([
                    "temperature_2m", "apparent_temperature",
                    "precipitation", "wind_speed_10m", "weather_code",
                    "relative_humidity_2m",
                ]),
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
            },
            timeout=8.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return {"error": f"fetch failed: {e}"}
    except ValueError as e:
        return {"error": f"parse failed: {e}"}

    c = data.get("current") or {}
    code = c.get("weather_code")
    return {
        "place_name": place_name,
        "lat": lat,
        "lon": lon,
        "temp_c": c.get("temperature_2m"),
        "feels_like_c": c.get("apparent_temperature"),
        "condition": _WEATHER_CODES.get(int(code) if code is not None else -1, "unknown"),
        "weather_code": code,
        "emoji": weather_emoji(code),
        "wind_kmh": c.get("wind_speed_10m"),
        "humidity": c.get("relative_humidity_2m"),
        "precipitation_mm": c.get("precipitation", 0),
    }


def _format_weather_text(d: dict) -> str:
    """Compact one-line summary the LLM reads. Keeps tool output short
    so the model's reply stays conversational instead of reciting stats."""
    if d.get("error"):
        return f"Error fetching weather: {d['error']}"
    return (
        f"{d['place_name']}: {d['condition']}, {d['temp_c']}°C "
        f"(feels like {d['feels_like_c']}°C), "
        f"humidity {d['humidity']}%, "
        f"wind {d['wind_kmh']} km/h, "
        f"precipitation {d['precipitation_mm']} mm."
    )


# Tiny TTL cache so the LLM tool call and the widget's separate
# /api/weather call return the SAME resolved location/data — otherwise
# ipapi.co flakiness can produce a "can't find you" tool result while
# the widget immediately renders a real answer (jarring to the user).
# Keyed by a normalized version of the location arg.
_WEATHER_CACHE: dict[str, tuple[float, dict]] = {}
_WEATHER_CACHE_TTL_S: float = 30.0


def _weather_cache_key(location: str | None) -> str:
    return (location or "").strip().lower()


def fetch_weather(location: str | None = None) -> dict:
    """Public: resolve a location and fetch current conditions as a dict.

    Used by the web layer's weather widget. Mirrors ``_get_weather`` branching
    (vague->IP, coords->direct, name->geocode) so the widget and the LLM
    tool stay in sync. Successful resolutions are cached for a short TTL
    so a follow-up call with identical args returns the same answer
    even if ipapi.co rate-limits or the geocoder times out the second
    time.
    """
    import time
    cache_key = _weather_cache_key(location)
    now = time.time()
    cached = _WEATHER_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < _WEATHER_CACHE_TTL_S:
        return cached[1]

    try:
        with httpx.Client() as client:
            if _is_vague_location(location):
                coords = _ip_geolocate(client)
                if coords is None:
                    return {"error": "couldn't detect your location — tell me the city."}
                lat, lon, place_name = coords
            elif _looks_like_coords(location):
                lat_s, lon_s = location.split(",", 1)
                lat, lon = float(lat_s.strip()), float(lon_s.strip())
                place_name = f"{lat:.2f},{lon:.2f}"
            else:
                resolved = _geocode_open_meteo(client, location)
                if resolved is None:
                    return {"error": f"couldn't find '{location}'."}
                lat, lon, place_name = resolved
            data = _fetch_weather_data(client, lat, lon, place_name)
    except Exception as e:
        data = {"error": str(e)}

    # Only cache successes — caching errors would lock in a bad state.
    if not data.get("error"):
        _WEATHER_CACHE[cache_key] = (now, data)
    return data


def _get_weather(location: str | None = None) -> str:
    """LLM tool: return a one-line weather summary.

    Thin wrapper over ``fetch_weather`` that flattens the structured
    dict to a single sentence so Qwen's tool result stays compact.
    The same resolution order applies (vague -> IP, coords -> direct,
    name -> geocode). A widget in the web UI renders the full dict.
    """
    data = fetch_weather(location)
    if data.get("error"):
        err = data["error"]
        if "couldn't detect" in err:
            return "I couldn't detect your location. Tell me the city and I'll check."
        if "couldn't find" in err:
            return f"{err} Try a more specific city name."
        return f"Error fetching weather: {err}"
    return _format_weather_text(data)

