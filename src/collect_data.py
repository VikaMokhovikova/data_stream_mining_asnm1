from pathlib import Path
from utils import (
    load_config,
    download_weather,
    save_raw_snapshot,
    load_raw_snapshot,
)


def main():
    # 1. Read configuration
    config = load_config()

    # 2. Get settings
    data_config = config["data"]
    storage = config["storage"]

    # 3. Collect or reuse each city's snapshot
    for city in config["cities"]:
        filename = f"{city['name'].lower().replace(' ', '_')}_raw.json"
        raw_path = Path(storage["raw_dir"]) / filename

        # Reuse local snapshot if it already exists and force\_download is False
        if raw_path.exists() and not storage["force_download"]:
            payload = load_raw_snapshot(raw_path)
            print(f"Reused snapshot: {raw_path}")
        else:
            payload = download_weather(
                data_config["api_url"],
                city,
                data_config["start_date"],
                data_config["end_date"],
                data_config["hourly_variables"],
                data_config["timezone"],
            )
            save_raw_snapshot(payload, raw_path)
            print(f"Saved snapshot: {raw_path}")


if __name__ == "__main__":
    main()
