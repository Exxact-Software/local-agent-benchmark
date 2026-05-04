"""
Live tool implementations — real API calls used when --live-tools is passed.

Routing rule: live tools are only used when scenario == "default". All failure
scenario tests (T7 conflicting_hours, T8 404, T9 null_hours, T10 timeout,
T11 large_payload, T14 prompt_injection) always use mock regardless of this flag.
So adding a live implementation here is safe — it will never interfere with
the controlled test scenarios.

Implemented:
  web_search        → local SearXNG (http://localhost:8080) — no key needed
  get_weather       → Open-Meteo (open source, no key needed)
  geocode           → Google Maps Geocoding API
  search_places     → Google Maps Places Text Search API
  get_place_details → Google Maps Place Details API
  get_directions    → Google Maps Directions API
  search_parking    → Google Maps Places API (type=parking)
  get_reviews       → Google Maps Place Details API (reviews field)

Not implemented (no public structured API worth the setup):
  add_favorite / remove_favorite / list_favorites — local state, mock is correct
"""

import json
import os
import re

import requests

_HTML_TAG_RE = re.compile(r"<[^>]+>")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
def _google_maps_key() -> str:
    return os.getenv("GOOGLE_MAPS_API_KEY", "")

GMAPS_GEOCODE = "https://maps.googleapis.com/maps/api/geocode/json"
GMAPS_PLACES = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GMAPS_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
GMAPS_DIRECTIONS = "https://maps.googleapis.com/maps/api/directions/json"

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → human-readable conditions
WMO_CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Light rain showers", 81: "Moderate rain showers", 82: "Heavy rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def _no_key(tool: str) -> str:
    return json.dumps({
        "error": "GOOGLE_MAPS_API_KEY not set",
        "hint": f"Add GOOGLE_MAPS_API_KEY to .env to enable live {tool}",
    })


def _gmaps_get(url: str, params: dict) -> dict:
    params["key"] = _google_maps_key()
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Web Search ─────────────────────────────────────────────────────────────────

def web_search_live(query: str) -> str:
    """Search the web via the local SearXNG instance. Returns top 5 results."""
    try:
        r = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general", "language": "en"},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        results = [
            {
                "title": res.get("title", ""),
                "url": res.get("url", ""),
                "content": res.get("content", "")[:500],
            }
            for res in data.get("results", [])[:5]
        ]
        return json.dumps({"query": query, "results": results})
    except requests.exceptions.ConnectionError:
        return json.dumps({
            "error": "SearXNG not reachable",
            "hint": "Is the container running? docker compose up -d searxng",
            "query": query,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


# ── Weather (Open-Meteo — no API key required) ────────────────────────────────

def get_weather_live(city: str, date: str = None) -> str:
    """Get current weather via Open-Meteo. No API key needed."""
    try:
        # Step 1: resolve city name to coordinates
        geo = requests.get(
            OPEN_METEO_GEOCODE,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return json.dumps({"error": f"City not found: {city}"})

        loc = results[0]
        lat, lng = loc["latitude"], loc["longitude"]

        # Step 2: fetch current conditions
        weather = requests.get(
            OPEN_METEO_FORECAST,
            params={
                "latitude": lat,
                "longitude": lng,
                "current": "temperature_2m,relative_humidity_2m,weathercode,windspeed_10m",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "timezone": "auto",
            },
            timeout=10,
        )
        weather.raise_for_status()
        current = weather.json()["current"]

        return json.dumps({
            "city": loc.get("name", city),
            "temp_f": round(current["temperature_2m"]),
            "conditions": WMO_CONDITIONS.get(current["weathercode"], f"Code {current['weathercode']}"),
            "humidity": current["relative_humidity_2m"],
            "wind_mph": round(current["windspeed_10m"]),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "city": city})


# ── Google Maps ────────────────────────────────────────────────────────────────

def geocode_live(address: str) -> str:
    if not _google_maps_key():
        return _no_key("geocode")
    try:
        data = _gmaps_get(GMAPS_GEOCODE, {"address": address})
        if data.get("status") != "OK" or not data.get("results"):
            return json.dumps({"error": data.get("status"), "address": address})
        loc = data["results"][0]["geometry"]["location"]
        return json.dumps({
            "lat": loc["lat"],
            "lng": loc["lng"],
            "address": data["results"][0]["formatted_address"],
        })
    except Exception as e:
        return json.dumps({"error": str(e), "address": address})


def search_places_live(query: str, location: str, radius_km: float = None, open_now: bool = None) -> str:
    if not _google_maps_key():
        return _no_key("search_places")
    try:
        params = {"query": f"{query} near {location}"}
        if open_now:
            params["opennow"] = "true"
        if radius_km:
            params["radius"] = int(radius_km * 1000)
        data = _gmaps_get(GMAPS_PLACES, params)
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return json.dumps({"error": data.get("status")})
        places = [
            {
                "place_id": p["place_id"],
                "name": p["name"],
                "address": p.get("formatted_address", ""),
                "rating": p.get("rating"),
                "user_ratings_total": p.get("user_ratings_total"),
            }
            for p in data.get("results", [])[:5]
        ]
        return json.dumps(places)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_place_details_live(place_id: str) -> str:
    if not _google_maps_key():
        return _no_key("get_place_details")
    try:
        fields = "name,formatted_address,formatted_phone_number,rating,opening_hours,price_level,website"
        data = _gmaps_get(GMAPS_DETAILS, {"place_id": place_id, "fields": fields})
        if data.get("status") != "OK":
            return json.dumps({"error": data.get("status"), "place_id": place_id})
        return json.dumps(data["result"])
    except Exception as e:
        return json.dumps({"error": str(e), "place_id": place_id})


def get_directions_live(origin: str, destination: str, mode: str = "driving") -> str:
    if not _google_maps_key():
        return _no_key("get_directions")
    try:
        data = _gmaps_get(GMAPS_DIRECTIONS, {
            "origin": origin,
            "destination": destination,
            "mode": mode,
        })
        if data.get("status") != "OK" or not data.get("routes"):
            return json.dumps({"error": data.get("status")})
        leg = data["routes"][0]["legs"][0]
        # Strip HTML tags so live output matches the plain-text mock format
        steps = [_HTML_TAG_RE.sub("", s["html_instructions"]) for s in leg["steps"][:5]]
        return json.dumps({
            "distance": leg["distance"]["text"],
            "duration": leg["duration"]["text"],
            "steps": steps,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_parking_live(location: str, radius_km: float = None) -> str:
    if not _google_maps_key():
        return _no_key("search_parking")
    try:
        params = {"query": f"parking near {location}", "type": "parking"}
        if radius_km:
            params["radius"] = int(radius_km * 1000)
        data = _gmaps_get(GMAPS_PLACES, params)
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return json.dumps({"error": data.get("status")})
        places = [
            {
                "place_id": p["place_id"],
                "name": p["name"],
                "address": p.get("formatted_address", ""),
                "rating": p.get("rating"),
            }
            for p in data.get("results", [])[:5]
        ]
        return json.dumps(places)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_reviews_live(location: str) -> str:
    """Search for the location, then fetch its reviews from Place Details."""
    if not _google_maps_key():
        return _no_key("get_reviews")
    try:
        # Step 1: find the place
        search = _gmaps_get(GMAPS_PLACES, {"query": location})
        if search.get("status") != "OK" or not search.get("results"):
            return json.dumps({"error": "place not found", "location": location})
        place_id = search["results"][0]["place_id"]

        # Step 2: fetch reviews
        details = _gmaps_get(GMAPS_DETAILS, {"place_id": place_id, "fields": "name,reviews"})
        if details.get("status") != "OK":
            return json.dumps({"error": details.get("status")})
        result = details["result"]
        return json.dumps({
            "location": result.get("name", location),
            "total": len(result.get("reviews", [])),
            "reviews": result.get("reviews", []),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "location": location})
