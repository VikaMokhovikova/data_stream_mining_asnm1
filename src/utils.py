import yaml
import json
from pathlib import Path
import requests


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_weather(api_url, city, start_date, end_date, variables, timezone):
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": timezone,
    }

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "API error"))

    return payload


def save_raw_snapshot(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_raw_snapshot(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
