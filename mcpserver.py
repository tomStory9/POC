from typing import Any
import httpx
from fastmcp import FastMCP

mcp = FastMCP("weather-mcp")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _pick_location(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    if not results:
        raise ValueError("No location found")
    first = results[0]
    return {
        "name": first.get("name"),
        "country": first.get("country"),
        "latitude": first.get("latitude"),
        "longitude": first.get("longitude"),
        "timezone": first.get("timezone"),
    }


@mcp.tool()
def geocode_city(city: str) -> dict[str, Any]:
    """Resolve a city name to coordinates using the public Open-Meteo geocoding API."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        return _pick_location(resp.json())


@mcp.tool()
def get_current_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city using the public Open-Meteo API. No API key required."""
    location = geocode_city(city)
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            FORECAST_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    current = data.get("current", {})
    return {
        "location": location,
        "current": current,
    }


if __name__ == "__main__":
    mcp.run()
