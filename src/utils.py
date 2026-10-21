import yaml
import json
from pathlib import Path
import requests

SRC_DIR = Path(__file__).resolve().parent          # .../data_stream_mining_asnm1/src
ROOT = SRC_DIR.parent                              # .../data_stream_mining_asnm1

def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
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


def load_json_files(path: str) -> dict:
    """Load JSON from a file, or every *.json file in a folder.

    Returns {file_name_without_extension: parsed_json}.
    The path is relative to the project root, as in config.yaml.
    """
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    return {f.stem: json.loads(Path(f).read_text(encoding="utf-8")) for f in files}