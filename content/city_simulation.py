"""Night City economy / city statistics simulator.

Generates a full set of fake city stats for the broadcast. State persists
day-to-day in ``output/city_state.json`` with realistic drift, so the city
evolves over time instead of being regenerated randomly each morning.

Run standalone to print a sample day::

    python content/city_simulation.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Allow running as a script (python content/city_simulation.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import CITY_STATE_FILE  # noqa: E402

log = get_logger("simulation")

STATE_VERSION = 1

# --- Seed data -------------------------------------------------------------

# Fictional corporations traded on the Night City exchange.
_CORPS: list[dict[str, Any]] = [
    {"name": "Arasaka", "ticker": "ARSK", "price": 1842.50},
    {"name": "Militech", "ticker": "MLTC", "price": 1320.10},
    {"name": "Petrochem", "ticker": "PTRC", "price": 88.40},
    {"name": "Kang Tao", "ticker": "KGTO", "price": 410.75},
    {"name": "Biotechnica", "ticker": "BTCH", "price": 256.30},
    {"name": "Trauma Team Intl", "ticker": "TRMA", "price": 540.00},
    {"name": "Zetatech", "ticker": "ZETA", "price": 174.95},
    {"name": "Kiroshi Optics", "ticker": "KRSH", "price": 99.20},
    {"name": "Night Corp", "ticker": "NCRP", "price": 612.45},
    {"name": "Dynalar", "ticker": "DYNA", "price": 47.80},
]

_ANALYST_COMMENTS = [
    "Buy. The board flatlined a competitor; margins look preem.",
    "Hold. Corpo war chatter has the chrome jittery.",
    "Strong buy, choom - quarterly cyberware shipments are nova.",
    "Sell. Sources whisper of a Militech audit gone gonk.",
    "Neutral. Black market spillover is eating retail.",
    "Overweight. New braindance line is moving units in Westbrook.",
    "Underweight. Maelstrom raids hit three of their warehouses.",
]

_CRIME_TYPES = [
    "gang violence",
    "corpo espionage",
    "Maelstrom activity",
    "ripperdoc raids",
    "cyberpsycho incidents",
    "vehicle jackings",
]

_PRECINCTS = ["Watson", "Westbrook", "Heywood", "Pacifica", "Santo Domingo", "City Center"]

_FREEWAYS = [
    "the Pacifica Interchange",
    "Highway 70 north",
    "the Watson overpass",
    "the Corpo Plaza ring road",
    "the Santo Domingo industrial loop",
]

_TRAFFIC_INCIDENTS = [
    "an AV crash blocking two lanes",
    "a Valentinos blockade",
    "a corpo motorcade rolling through",
    "a vehicle jacking turned firefight",
    "a NCPD checkpoint backing things up",
    "nothing but the usual gridlock",
]

_CIVIC_EVENTS = [
    "City Council narrowly approved expanded NCPD drone patrols in {precinct}.",
    "Arasaka announced a 'community wellness' initiative nobody asked for.",
    "NCPD press release insists crime is 'statistically down' despite the sirens.",
    "Night Corp unveiled a new public braindance kiosk in {precinct}.",
    "Militech recruiters set up shop outside three high schools in {precinct}.",
    "A corpo PR drone apologized for last week's 'unscheduled district lockdown'.",
    "The mayor's office cut the ribbon on a vending machine that sells water rights.",
    "Trauma Team rolled out a new 'platinum afterlife' subscription tier.",
]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --- State persistence -----------------------------------------------------

def _new_state() -> dict[str, Any]:
    """Build a fresh baseline state (first run, or corrupt state recovery)."""
    return {
        "version": STATE_VERSION,
        "last_run": None,
        "stocks": [
            {
                "name": c["name"],
                "ticker": c["ticker"],
                "price": c["price"],
            }
            for c in _CORPS
        ],
        "economy": {
            "eddies_per_yen": 1.18,
            "unemployment_rate": 14.2,
            "black_market_index": 62.0,
        },
        "crime_baseline": {ptype: random.randint(8, 40) for ptype in _CRIME_TYPES},
    }


def _validate_state(state: Any) -> bool:
    """Sanity-check a loaded state dict. Returns True if usable."""
    if not isinstance(state, dict):
        return False
    if state.get("version") != STATE_VERSION:
        return False
    required = ("stocks", "economy", "crime_baseline")
    if not all(k in state for k in required):
        return False
    if not isinstance(state["stocks"], list) or not state["stocks"]:
        return False
    return True


def load_state(path: Path = CITY_STATE_FILE) -> dict[str, Any]:
    """Load persisted city state, validating and recovering on failure."""
    if not path.exists():
        log.info("No prior city state found; seeding a fresh city.")
        return _new_state()
    try:
        with path.open("r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("City state unreadable (%s); reseeding.", exc)
        return _new_state()

    if not _validate_state(state):
        log.warning("City state failed validation; reseeding.")
        return _new_state()
    return state


def save_state(state: dict[str, Any], path: Path = CITY_STATE_FILE) -> None:
    """Atomically persist city state (write to .tmp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem
    except OSError as exc:
        log.error("Failed to persist city state: %s", exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# --- Drift / generation ----------------------------------------------------

def _drift_stocks(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Random-walk each stock +/-3%, with occasional +/-15% event spikes."""
    results = []
    for stock in state["stocks"]:
        old_price = float(stock["price"])
        if random.random() < 0.08:  # rare event spike
            change_pct = random.uniform(-15.0, 15.0)
        else:
            change_pct = random.uniform(-3.0, 3.0)
        new_price = max(0.5, round(old_price * (1 + change_pct / 100), 2))
        stock["price"] = new_price
        results.append(
            {
                "name": stock["name"],
                "ticker": stock["ticker"],
                "price": new_price,
                "change_pct": round(change_pct, 2),
                "analyst_comment": random.choice(_ANALYST_COMMENTS),
            }
        )
    return results


def _generate_crime(state: dict[str, Any]) -> dict[str, Any]:
    """Slow seasonal trend plus random faction-driven spikes per crime type."""
    baseline = state["crime_baseline"]
    report = []
    for ctype, base in baseline.items():
        # Seasonal drift on the baseline itself.
        base = int(_clamp(base + random.randint(-3, 3), 1, 200))
        baseline[ctype] = base
        # Today's count: baseline + optional spike.
        today = base
        if random.random() < 0.15:
            today += random.randint(10, 40)  # faction event spike
        delta = today - base
        if delta > 5:
            trend = "up sharply"
        elif delta > 0:
            trend = "up"
        elif delta < 0:
            trend = "down"
        else:
            trend = "flat"
        report.append(
            {
                "type": ctype,
                "precinct": random.choice(_PRECINCTS),
                "incidents": today,
                "trend": trend,
            }
        )
    return {"incidents": report}


def _generate_weather() -> dict[str, Any]:
    temp_f = random.randint(58, 104)
    aqi = random.randint(70, 320)  # Night City air is rarely clean
    return {
        "temperature_f": temp_f,
        "aqi": aqi,
        "aqi_label": _aqi_label(aqi),
        "acid_rain_probability": round(random.uniform(0.0, 0.9), 2),
        "uv_index": random.randint(3, 11),
    }


def _aqi_label(aqi: int) -> str:
    if aqi <= 100:
        return "tolerable"
    if aqi <= 180:
        return "unhealthy"
    if aqi <= 260:
        return "hazardous - mask up, choom"
    return "apocalyptic"


def _generate_traffic() -> dict[str, Any]:
    routes = random.sample(_FREEWAYS, k=min(4, len(_FREEWAYS)))
    return {
        "routes": [
            {
                "name": route,
                "congestion": random.randint(1, 10),
                "incident": random.choice(_TRAFFIC_INCIDENTS),
            }
            for route in routes
        ]
    }


def _drift_economy(state: dict[str, Any]) -> dict[str, Any]:
    econ = state["economy"]
    shock = random.random() < 0.07  # occasional corpo-war shock
    fx_change = random.uniform(-0.15, 0.15) if shock else random.uniform(-0.02, 0.02)
    econ["eddies_per_yen"] = round(_clamp(econ["eddies_per_yen"] + fx_change, 0.5, 3.0), 3)

    unemp_change = random.uniform(-1.5, 1.5) if shock else random.uniform(-0.3, 0.3)
    econ["unemployment_rate"] = round(_clamp(econ["unemployment_rate"] + unemp_change, 2.0, 45.0), 1)

    bm_change = random.uniform(-8, 8) if shock else random.uniform(-2, 2)
    econ["black_market_index"] = round(_clamp(econ["black_market_index"] + bm_change, 0.0, 100.0), 1)

    return {
        "eddies_per_yen": econ["eddies_per_yen"],
        "unemployment_rate": econ["unemployment_rate"],
        "black_market_index": econ["black_market_index"],
        "corpo_war_shock": shock,
    }


def _generate_civic() -> list[str]:
    count = random.randint(2, 3)
    events = random.sample(_CIVIC_EVENTS, k=count)
    return [e.format(precinct=random.choice(_PRECINCTS)) for e in events]


def generate_city_stats(state_path: Path = CITY_STATE_FILE) -> dict[str, Any]:
    """Load state, apply drift, build today's stats, and persist new state.

    Returns a dict of all stat categories suitable for the script writer.
    """
    state = load_state(state_path)

    stocks = _drift_stocks(state)
    crime = _generate_crime(state)
    economy = _drift_economy(state)
    weather = _generate_weather()
    traffic = _generate_traffic()
    civic = _generate_civic()

    state["last_run"] = datetime.now().isoformat(timespec="seconds")
    save_state(state, state_path)

    return {
        "date": date.today().isoformat(),
        "stocks": stocks,
        "crime": crime,
        "weather": weather,
        "traffic": traffic,
        "economy": economy,
        "civic": civic,
    }


if __name__ == "__main__":
    stats = generate_city_stats()
    print(json.dumps(stats, indent=2))
