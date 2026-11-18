"""Paths and environment configuration.

The API key is read from the environment (``.env`` via python-dotenv) and is
never hardcoded, logged, or shown in the UI.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(os.getenv("WEATHER_INSIGHTS_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"

CITIES_CSV = DATA_DIR / "cities.csv"  # curated world cities (shipped with the repo)
USER_CITIES_CSV = DATA_DIR / "user_cities.csv"  # cities added by users from the UI
ATTRACTIONS_CSV = DATA_DIR / "attractions.csv"  # curated attractions with weather sensitivity
USER_ATTRACTIONS_CSV = DATA_DIR / "user_attractions.csv"  # attractions added by users from the UI
LOG_CSV = DATA_DIR / "weather_log.csv"  # our own time series, appended on every fetch

API_BASE_URL = "https://api.openweathermap.org"
REQUEST_TIMEOUT_S = 10
CACHE_TTL_S = 600  # OpenWeatherMap refreshes current data roughly every 10 minutes
MAX_CALLS_PER_MINUTE = 50  # OpenWeatherMap's free plan allows 60; keep some headroom
MIN_LOG_INTERVAL_MIN = 10  # do not log the same city twice inside this window

# For add POI Feature
# Looking up a user-added attraction by name (OpenStreetMap Nominatim; no API key needed).
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
NOMINATIM_USER_AGENT = "weather-insights-course-project/0.1 (student project)"  # required by Nominatim's usage policy
NOMINATIM_PERIOD_S = 1.1  # Nominatim allows at most 1 request per second
POI_MAX_RESULTS = 5  # candidate matches shown to the user
POI_SEARCH_BOX_DEG = 0.5  # search +/- this many degrees around the city centre
POI_MAX_KM_FROM_CITY = 45  # reject matches further than this (same limit the catalogue tests enforce)

# Rain radar overlay (RainViewer; free for personal/educational use, no API key, attribution required).
RAINVIEWER_BASE_URL = "https://api.rainviewer.com"
RAINVIEWER_MAX_CALLS_PER_MINUTE = 60  # RainViewer allows 100 requests per minute per IP; keep headroom
RADAR_MAX_ZOOM = 7  # the free tier only serves tiles up to zoom 7; deeper tiles are a "not supported" image
RADAR_TILE_SIZE = 512  # pixels per tile: 256 or 512
RADAR_COLOR_SCHEME = 2  # RainViewer colour scheme id (2 = "Universal Blue")
RADAR_OPTIONS = "1_1"  # {smooth}_{snow}: smoothed image, snow drawn in its own colours
RADAR_PAD_KM = 200  # cover at least this far around the centre so the overlay fills the visible map
RADAR_MAX_TILES = 16  # most radar pictures one map loads; beyond that a coarser zoom is used
RADAR_OPACITY = 0.7  # how strongly the radar covers the map below it (0-1)
RADAR_CACHE_TTL_S = 300  # radar frames arrive every 10 minutes; re-check the list every 5

load_dotenv(PROJECT_ROOT / ".env")


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENWEATHER_API_KEY is not configured."""


def get_api_key() -> str:
    """Return the API key from the environment or raise a clear error."""
    key = (os.getenv("OPENWEATHER_API_KEY") or "").strip()
    if not key or key == "your_key_here":
        raise MissingAPIKeyError(
            "OPENWEATHER_API_KEY is not set. Copy .env.example to .env "
            "and paste your own OpenWeatherMap key into it."
        )
    return key
