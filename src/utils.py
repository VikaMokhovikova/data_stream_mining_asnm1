"""Generic helpers with no project logic: paths, config, JSON/CSV I/O, HTTP download, versions."""
import csv
import json
from importlib import metadata
from pathlib import Path

import requests
import yaml

SRC_DIR = Path(__file__).resolve().parent          # .../data_stream_mining_asnm1/src
ROOT = SRC_DIR.parent                              # .../data_stream_mining_asnm1


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_dir(cfg: dict, name: str) -> Path:
    """Absolute path of a `storage` folder from config.yaml (e.g. "figures_dir"), created if missing."""
    path = ROOT / cfg["storage"][name]
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def save_json(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def save_configuration(rows: dict, path) -> None:
    """Write {parameter: value} to a 2-column CSV (results/configurations.csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value"])
        for key, value in rows.items():
            writer.writerow([key, value])
