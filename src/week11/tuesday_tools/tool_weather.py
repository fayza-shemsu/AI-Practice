"""
TOOL NODE 2: get_weather
-------------------------------
WHY THIS EXISTS IN PRODUCTION:
This teaches you the pattern for calling ANY external API
from inside an AI pipeline. The pattern is identical whether you call:
  - A weather API (this example)
  - Your company CRM system
  - A billing database
  - A stock price API
  - An internal microservice

Uses Open-Meteo — completely free, no API key needed.
Perfect for learning the pattern without credential complexity.
"""
import urllib.request
import json

CITY_COORDS = {
    "london":      (51.5074, -0.1278),
    "manchester":  (53.4808, -2.2426),
    "birmingham":  (52.4862, -1.8904),
    "edinburgh":   (55.9533, -3.1883),
    "addis ababa": (9.0320,  38.7469),
    "new york":    (40.7128, -74.0060),
    "paris":       (48.8566,  2.3522),
    "dubai":       (25.2048,  55.2708),
}

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    51: "Light drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    71: "Slight snow",
    80: "Rain showers",
    95: "Thunderstorm",
}

def get_weather(city: str) -> dict:
    city_lower = city.lower().strip()

    if city_lower not in CITY_COORDS:
        return {
            "city":    city,
            "error":   f"City not found. Available: {list(CITY_COORDS.keys())}",
            "summary": "Weather unavailable"
        }

    lat, lon = CITY_COORDS[city_lower]

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,weathercode,windspeed_10m,precipitation"
            f"&timezone=auto"
        )
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())

        c           = data["current"]
        temp_c      = c["temperature_2m"]
        wind        = c["windspeed_10m"]
        rain        = c["precipitation"]
        code        = c["weathercode"]
        description = WEATHER_CODES.get(code, f"Code {code}")

        # Engineer visit recommendation
        # In production this feeds into the retention playbook
        if wind > 50 or rain > 5:
            engineer_visit = "NOT RECOMMENDED today due to weather"
        elif wind > 30:
            engineer_visit = "POSSIBLE but may be delayed"
        else:
            engineer_visit = "SUITABLE for engineer visit"

        return {
            "city":            city.title(),
            "temperature_c":   temp_c,
            "temperature_f":   round(temp_c * 9 / 5 + 32, 1),
            "condition":       description,
            "wind_kmh":        wind,
            "precipitation_mm": rain,
            "engineer_visit":  engineer_visit,
            "summary":         f"{description}, {temp_c}°C, wind {wind}km/h"
        }

    except Exception as e:
        return {
            "city":    city,
            "error":   str(e),
            "summary": "Could not fetch weather"
        }

if __name__ == "__main__":
    for city in ["london", "addis ababa"]:
        print(f"\n{city.title()}:")
        result = get_weather(city)
        for key, value in result.items():
            print(f"  {key}: {value}")
