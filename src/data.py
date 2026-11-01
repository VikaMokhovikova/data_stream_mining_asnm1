"""Steps 1-3: raw snapshots -> clean hourly tables -> frozen threshold -> binary target.

Everything here is shared by both students. Nothing is fitted on evaluation data:
the humid/dry threshold comes from the pooled WARM-UP rows only.

Workflow rule: collect once -> save raw -> prepare -> replay as a stream.
`prepare_frames(cfg)` is the single entry point used by every notebook.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from utils import ROOT, download_weather, load_json, output_dir, package_version, save_configuration, save_json


# --- 1. Raw snapshots ----------------------------------------------------------------------

def get_snapshot(city: dict, cfg: dict) -> dict:
    """Reuse the raw JSON snapshot of `city`, or download it once."""
    path = ROOT / city["raw_file"]
    if path.exists() and not cfg["storage"]["force_download"]:
        return load_json(path)

    d = cfg["data"]
    payload = download_weather(d["api_url"], city, d["start_date"], d["end_date"],
                               d["hourly_variables"], d["timezone"])
    save_json(payload, path)
    return payload


# --- 2. Clean chronological tables ---------------------------------------------------------

def to_frame(payload: dict, variables: list) -> pd.DataFrame:
    """Open-Meteo JSON -> chronological hourly table (missing values stay NaN).

    `timestamp` is the clock returned by the API (fixed offset `utc_offset_seconds`);
    `timestamp_utc` is stored as well so hour-of-day analyses have an unambiguous clock.
    """
    hourly = payload["hourly"]
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise ValueError(f"Variables missing from the raw snapshot: {missing}")

    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
    df["timestamp_utc"] = df["timestamp"] - pd.Timedelta(seconds=payload.get("utc_offset_seconds", 0))
    for v in variables:
        df[v] = pd.to_numeric(pd.Series(hourly[v]), errors="coerce")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in the raw snapshot.")
    return df


def flag_invalid(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Boolean table: value missing or outside its definitional range (config.yaml -> quality)."""
    ranges = cfg["quality"]["valid_ranges"]
    flags = pd.DataFrame(index=df.index)
    for v in cfg["data"]["hourly_variables"]:
        lo, hi = ranges.get(v, (None, None))
        bad = df[v].isna()
        if lo is not None:
            bad |= df[v] < lo
        if hi is not None:
            bad |= df[v] > hi
        flags[v] = bad
    return flags


def quality_report(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Missing and invalid (present but out of range) counts per variable, plus flagged rows."""
    variables = cfg["data"]["hourly_variables"]
    flags = flag_invalid(df, cfg)
    row = {f"missing_{v}": int(df[v].isna().sum()) for v in variables}
    row.update({f"invalid_{v}": int((flags[v] & df[v].notna()).sum()) for v in variables})
    row["rows_flagged_invalid"] = int(flags.any(axis=1).sum())
    row["rows_total"] = len(df)
    return pd.Series(row)


def chronology_report(df: pd.DataFrame) -> pd.Series:
    """Is the stream a regular, strictly increasing hourly index?"""
    ts = df["timestamp"]
    return pd.Series({
        "first": ts.min(),
        "last": ts.max(),
        "rows": len(df),
        "expected_hours": int((ts.max() - ts.min()) / pd.Timedelta(hours=1)) + 1,
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "non_1h_steps": int((ts.diff().dropna() != pd.Timedelta(hours=1)).sum()),
        "strictly_increasing": bool(ts.is_monotonic_increasing and ts.is_unique),
    })


# --- 3. Warm-up, frozen threshold, target ---------------------------------------------------

def warmup_size(df: pd.DataFrame, months: int) -> int:
    """Number of rows in the first `months` months (the warm-up period)."""
    cutoff = df["timestamp"].iloc[0] + pd.DateOffset(months=months)
    return int((df["timestamp"] < cutoff).sum())


def pooled_warmup(frames: list, n_warm: int, variable: str) -> np.ndarray:
    """Warm-up values of `variable` from all cities, rows flagged "ok" only."""
    return np.concatenate([
        f[variable].iloc[:n_warm][f["quality_flag"].iloc[:n_warm] == "ok"].dropna().to_numpy()
        for f in frames
    ])


def frozen_threshold(frames: list, n_warm: int, variable: str, quantile: float) -> float:
    """tau = quantile of the POOLED warm-up values of all cities. Computed once, then frozen."""
    return float(np.quantile(pooled_warmup(frames, n_warm, variable), quantile))


def add_target(df: pd.DataFrame, variable: str, tau: float) -> pd.DataFrame:
    """T4: y_t = 1 (humid) if H_t >= tau, else 0 (dry). Missing or invalid humidity -> missing target."""
    out = df.copy()
    invalid = out[variable].isna()
    if "quality_flag" in out:
        invalid |= out["quality_flag"] != "ok"
    out["y"] = np.where(invalid, np.nan, (out[variable] >= tau).astype(float))
    return out


def prepare_frames(cfg: dict) -> dict:
    """Load both snapshots, align them, fix the warm-up and the threshold, attach the target.

    Returns
      payloads : {city_key: raw JSON}          (grid point, elevation, units)
      raw      : {city_key: table before the target}   (input of the leakage test)
      frames   : {city_key: table with quality_flag, phase and y}
      tau, n_warm, warmup_end, evaluation_start, n_pooled
    """
    cities = {c["key"]: c for c in cfg["cities"]}
    variables = cfg["data"]["hourly_variables"]
    payloads = {k: get_snapshot(c, cfg) for k, c in cities.items()}
    tables = {k: to_frame(p, variables) for k, p in payloads.items()}

    # Common timeline: both cities must share exactly the same hours
    common = set.intersection(*(set(t["timestamp"]) for t in tables.values()))
    raw = {}
    for k, t in tables.items():
        t = t[t["timestamp"].isin(common)].reset_index(drop=True)
        t["quality_flag"] = np.where(flag_invalid(t, cfg).any(axis=1), "invalid", "ok")
        t.insert(0, "city", k)
        raw[k] = t
    a, b = raw.values()
    if not a["timestamp"].equals(b["timestamp"]):
        raise ValueError("The two cities must share exactly the same timestamps.")

    p = cfg["protocol"]
    n_warm = warmup_size(a, p["warmup_months"])
    tau = frozen_threshold(list(raw.values()), n_warm, p["target_variable"], p["target_quantile"])
    frames = {}
    for k, t in raw.items():
        f = add_target(t, p["target_variable"], tau)
        f["phase"] = np.where(f.index < n_warm, "warm-up", "evaluation")
        frames[k] = f

    return {
        "payloads": payloads,
        "raw": raw,
        "frames": frames,
        "tau": tau,
        "n_warm": n_warm,
        "warmup_end": a["timestamp"].iloc[n_warm - 1],
        "evaluation_start": a["timestamp"].iloc[n_warm],
        "n_pooled": len(pooled_warmup(list(raw.values()), n_warm, p["target_variable"])),
    }


# --- Protocol checks and exported records ---------------------------------------------------

def protocol_checks(prepared: dict, cfg: dict) -> pd.Series:
    """The temporal-alignment checklist as booleans. Raises if any check fails."""
    frames = list(prepared["frames"].values())
    a, b = frames
    p = cfg["protocol"]
    n_warm, tau, var = prepared["n_warm"], prepared["tau"], p["target_variable"]

    checks = pd.Series({
        "timestamps strictly increasing (both cities)": all(
            f["timestamp"].is_monotonic_increasing and f["timestamp"].is_unique for f in frames),
        "regular 1 h steps (both cities)": all(
            (f["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all() for f in frames),
        "identical timestamps in both cities": a["timestamp"].equals(b["timestamp"]),
        "identical warm-up mask in both cities": a["phase"].equals(b["phase"]),
        "warm-up ends before evaluation starts": prepared["warmup_end"] < prepared["evaluation_start"],
        # Value comparison alone cannot prove this: tau_H also equals 82 over all rows. Corrupt evaluation instead.
        "tau_H uses only warm-up rows": np.isclose(frozen_threshold(
            [f.assign(**{var: f[var].where(f.index < n_warm, 0.0)}) for f in frames],
            n_warm, var, p["target_quantile"]), tau),
        "tau_H reproducible from warm-up data": np.isclose(
            frozen_threshold(frames, n_warm, var, p["target_quantile"]), tau),
        "y == 1  <=>  H_t >= tau_H": all(
            ((f[var] >= tau).astype(float)[f["y"].notna()] == f["y"].dropna()).all() for f in frames),
        "invalid rows have no label": all(f.loc[f["quality_flag"] != "ok", "y"].isna().all() for f in frames),
    }, name="passed")
    if not checks.all():
        raise AssertionError(f"Protocol check failed:\n{checks[~checks]}")
    return checks


def stream_summary(prepared: dict, cfg: dict) -> pd.DataFrame:
    """One table for the report section 'Data and Stream Construction'."""
    d = cfg["data"]
    rows = {}
    for c in cfg["cities"]:
        f, pl = prepared["frames"][c["key"]], prepared["payloads"][c["key"]]
        rows[c["name"]] = {
            "role": c["role"],
            "grid point (lat, lon)": f"{pl['latitude']:.3f}, {pl['longitude']:.3f}",
            "elevation (m)": pl.get("elevation"),
            "period": f"{d['start_date']} to {d['end_date']}",
            "hours (common timeline)": len(f),
            "rows flagged invalid": int((f["quality_flag"] == "invalid").sum()),
            "warm-up hours": int((f["phase"] == "warm-up").sum()),
            "evaluation hours": int((f["phase"] == "evaluation").sum()),
            "tau_H (pooled, %)": round(prepared["tau"], 2),
            "humid share, warm-up": round(f.loc[f["phase"] == "warm-up", "y"].mean(), 3),
            "humid share, evaluation": round(f.loc[f["phase"] == "evaluation", "y"].mean(), 3),
        }
    return pd.DataFrame(rows)


def save_frozen_record(prepared: dict, cfg: dict) -> dict:
    """Write results/config_part1.json: what was frozen, from which data, with which versions.

    Together with the raw snapshots and config.yaml this is enough to rebuild every stream exactly,
    so the processed streams themselves are not saved.
    """
    proj, d, p = cfg["project"], cfg["data"], cfg["protocol"]
    record = {
        "assignment": proj["assignment"],
        "business_scenario": proj["scenario"],
        "business_question": proj["business_question"],
        "target": proj["target"],
        "target_name": proj["target_name"],
        "target_source": p["target_variable"],
        "target_quantile": p["target_quantile"],
        "tau_H": prepared["tau"],
        "tau_H_n_observations": prepared["n_pooled"],
        "cities": {c["key"]: c for c in cfg["cities"]},
        "grid_points": {k: {f: pl.get(f) for f in ("latitude", "longitude", "elevation")}
                        for k, pl in prepared["payloads"].items()},
        "period": [d["start_date"], d["end_date"]],
        "hourly_variables": d["hourly_variables"],
        "timezone": d["timezone"],
        "utc_offset_seconds": {k: pl.get("utc_offset_seconds") for k, pl in prepared["payloads"].items()},
        "warmup_months": p["warmup_months"],
        "warmup_hours": prepared["n_warm"],
        "warmup_end": str(prepared["warmup_end"]),
        "evaluation_start": str(prepared["evaluation_start"]),
        "valid_ranges": cfg["quality"]["valid_ranges"],
        "rows_flagged_invalid": {k: int((f["quality_flag"] == "invalid").sum())
                                 for k, f in prepared["frames"].items()},
        "random_seed": p["seed"],
        "versions": {n: package_version(n) for n in ("pandas", "numpy", "river", "scikit-learn", "matplotlib")},
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(record, output_dir(cfg, "results_dir") / "config_part1.json")
    return record


def configuration_rows(prepared: dict, cfg: dict) -> dict:
    """config.yaml flattened to {parameter: value}, plus the frozen values and package versions."""
    rows = {}

    def flatten(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            rows[prefix] = value

    flatten("", {k: v for k, v in cfg.items() if k != "cities"})
    for c in cfg["cities"]:
        flatten(f"cities.{c['key']}", c)
    rows.update({
        "frozen.tau_H": prepared["tau"],
        "frozen.warmup_hours": prepared["n_warm"],
        "frozen.warmup_end": prepared["warmup_end"],
        "frozen.evaluation_start": prepared["evaluation_start"],
    })
    rows.update({f"version.{n}": package_version(n) for n in ("river", "scikit-learn", "pandas", "numpy")})
    return rows


def save_protocol_records(prepared: dict, cfg: dict) -> pd.DataFrame:
    """All shared records in results/: config_part1.json, configurations.csv, stream_construction_summary.csv.

    Returns the stream summary table.
    """
    results = output_dir(cfg, "results_dir")
    save_frozen_record(prepared, cfg)
    save_configuration(configuration_rows(prepared, cfg), results / "configurations.csv")
    summary = stream_summary(prepared, cfg)
    summary.to_csv(results / "stream_construction_summary.csv")
    return summary
