"""
Download free daily weather history for ANY city (no account, no API key).
Source: Open-Meteo historical archive  https://open-meteo.com

Usage:
  python download_weather.py                       # default: Pune, 2008-2022
  python download_weather.py --lat 18.52 --lon 73.86 --start 2014-01-01 --end 2023-12-31
Saves: data/weather.csv
Then run:
  python weather_forecast.py --csv data/weather.csv --date-col date --target temperature_2m_mean
"""
import argparse
import os

import pandas as pd
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, default=28.61)   # change to your city
ap.add_argument("--lon", type=float, default=77.21)
ap.add_argument("--start", default="2013-01-01")
ap.add_argument("--end", default="2023-12-31")
args = ap.parse_args()

params = {
    "latitude": args.lat, "longitude": args.lon,
    "start_date": args.start, "end_date": args.end,
    "daily": ",".join([
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "surface_pressure_mean", "wind_speed_10m_max",
        "precipitation_sum", "cloud_cover_mean",
    ]),
    "timezone": "auto",
}
r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
r.raise_for_status()
df = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})

os.makedirs("data", exist_ok=True)
df.to_csv("data/weather.csv", index=False)
print(f"Saved data/weather.csv with {len(df)} rows")
print(df.head())
