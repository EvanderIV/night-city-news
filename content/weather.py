"""Real commute-weather fetcher.

Pulls an hourly forecast from Open-Meteo (free, no API key required - ideal for
an unattended server) and extracts the two commute windows the host cares about:
the morning drive (~9 AM) and the evening drive home (~5 PM) for a configured
location (default: Beavercreek, Ohio).

Returns a structured dict the script writer can read out accurately, or None if
disabled or the fetch fails (the broadcast then simply omits the segment).

Run standalone to test::

    python content/weather.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402

log = get_logger("weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 20.0

# WMO weather interpretation codes -> plain-English conditions.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}


def _describe(code: Any) -> str:
    try:
        return _WMO.get(int(code), "unsettled")
    except (TypeError, ValueError):
        return "unsettled"


def _pick_hour(hourly: dict[str, list], target_iso: str) -> dict[str, Any] | None:
    """Return the hourly reading whose timestamp matches ``target_iso`` exactly,
    else the first reading with the same hour-of-day."""
    times = hourly.get("time", [])
    idx = None
    if target_iso in times:
        idx = times.index(target_iso)
    else:
        hh = target_iso[-5:]  # "HH:MM"
        idx = next((i for i, t in enumerate(times) if t.endswith(hh)), None)
    if idx is None:
        return None
    return {
        "temp_f": round(hourly["temperature_2m"][idx]),
        "feels_like_f": round(hourly["apparent_temperature"][idx]),
        "conditions": _describe(hourly["weather_code"][idx]),
        "precip_prob": int(hourly["precipitation_probability"][idx]),
        "wind_mph": round(hourly["wind_speed_10m"][idx]),
    }


def fetch_commute_weather(config: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch the morning + evening commute forecast for the configured location.

    Returns ``None`` if ``[weather].enabled`` is false or the request fails.
    """
    wcfg = config.get("weather", {})
    if not wcfg.get("enabled", False):
        return None

    lat = wcfg.get("latitude")
    lon = wcfg.get("longitude")
    if lat is None or lon is None:
        log.warning("Weather enabled but latitude/longitude not set; skipping.")
        return None

    tz = wcfg.get("timezone", "America/New_York")
    morning_hour = int(wcfg.get("morning_hour", 9))
    evening_hour = int(wcfg.get("evening_hour", 17))
    location = wcfg.get("location", "your area")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,weather_code,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": tz,
        "forecast_days": 2,
    }

    try:
        import httpx

        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        hourly = data["hourly"]
    except Exception as exc:
        log.error("Weather fetch failed (%s); omitting commute segment.", exc)
        return None

    # Open-Meteo (with timezone set) returns hourly data starting at 00:00 of
    # the current local day, so the first timestamp's date IS "today".
    today = hourly["time"][0][:10]
    morning = _pick_hour(hourly, f"{today}T{morning_hour:02d}:00")
    evening = _pick_hour(hourly, f"{today}T{evening_hour:02d}:00")

    if not morning and not evening:
        log.error("Weather: could not locate commute hours in forecast; omitting.")
        return None

    log.info("Weather: commute forecast fetched for %s (%s).", location, today)
    return {
        "location": location,
        "date": today,
        "morning": {"label": f"{morning_hour % 12 or 12} {'AM' if morning_hour < 12 else 'PM'}", **(morning or {})},
        "evening": {"label": f"{evening_hour % 12 or 12} {'AM' if evening_hour < 12 else 'PM'}", **(evening or {})},
    }


if __name__ == "__main__":
    import json
    from config_loader import load_config

    cfg = load_config()
    print(json.dumps(fetch_commute_weather(cfg), indent=2))
