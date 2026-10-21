"""Stream characterisation of one city (handbook Ch. 9 + original analyses, Ch. 3.2).

Every function takes a prepared city frame (`prepare_frames(cfg)["frames"][key]`) and returns a table.
The same functions run for both cities, so both get exactly the same analytical logic.
All statistics are DESCRIPTIVE: nothing here is fitted or fed into a model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]
UNITS = {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm", "wind_speed_10m": "km/h"}
LABELS = {"temperature_2m": "Temperature", "relative_humidity_2m": "Relative humidity",
          "precipitation": "Precipitation", "wind_speed_10m": "Wind speed"}


def with_calendar(df: pd.DataFrame, local_tz: str) -> pd.DataFrame:
    """Add local_time (true local clock, with summer time), hour, month, year and season."""
    out = df.copy()
    out["local_time"] = out["timestamp_utc"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
    out["hour"] = out["local_time"].dt.hour
    out["month"] = out["local_time"].dt.month
    out["year"] = out["timestamp"].dt.year
    out["season"] = pd.Categorical(out["month"].map(SEASON_OF_MONTH), categories=SEASONS, ordered=True)
    return out


# --- Ch. 9 item 1: data quality ---------------------------------------------------------------

def longest_constant_run(s: pd.Series) -> int:
    run_id = (s != s.shift()).cumsum()
    return int(s.groupby(run_id).size().max())


def quality_table(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Checks that can reveal corrupted data hidden among valid values."""
    t = pd.DataFrame({
        LABELS[v]: {
            "missing": int(df[v].isna().sum()),
            "flagged invalid": int(((df["quality_flag"] == "invalid") & df[v].notna()).sum()),
            "longest constant run (h)": longest_constant_run(df[v]),
            "distinct values": int(df[v].nunique()),
        }
        for v in variables
    }).T
    t["unit"] = [UNITS[v] for v in variables]
    return t


def special_values(df: pd.DataFrame, target: str) -> pd.Series:
    return pd.Series({
        "RH = 100 % (saturation), hours": int((df[target] == 100).sum()),
        "RH = 100 %, share": round((df[target] == 100).mean(), 4),
        "wind = 0 km/h (calm), hours": int((df["wind_speed_10m"] == 0).sum()),
        "precipitation > 0 mm, hours": int((df["precipitation"] > 0).sum()),
        "precipitation > 0 mm, share": round((df["precipitation"] > 0).mean(), 4),
    })


# --- Ch. 9 item 2: descriptive statistics -----------------------------------------------------

def describe(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    d = df[variables].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    d["skew"] = df[variables].skew()
    d.index = [f"{LABELS[v]} ({UNITS[v]})" for v in d.index]
    return d


def by_phase(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Is the evaluation stream distributed like the warm-up?"""
    t = df.groupby("phase", sort=False)[variables].agg(["mean", "std", "median"]).T
    return t.rename(index=LABELS, level=0)


def by_season(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    t = df.groupby("season", observed=True)[variables].mean().rename(columns=LABELS)
    t["humid share (y=1)"] = df.groupby("season", observed=True)["y"].mean()
    return t


def daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means (daily total for precipitation)."""
    return df.set_index("timestamp")[list(LABELS)].resample("D").agg(
        {"temperature_2m": "mean", "relative_humidity_2m": "mean", "precipitation": "sum", "wind_speed_10m": "mean"})


# --- Ch. 9 item 4: class distribution -------------------------------------------------------

def class_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    t = df.groupby(group_col, observed=True, sort=False)["y"].agg(hours="count", humid_hours="sum", humid_share="mean")
    t["dry_share"] = 1 - t["humid_share"]
    return t


# --- Ch. 9 item 5: rolling / window behaviour -------------------------------------------------

def rolling_windows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Descriptive trailing windows ending at t (model features end at t-1, see features.py)."""
    rh = df.set_index("timestamp")[target]
    y = df.set_index("timestamp")["y"]
    return pd.DataFrame({
        "rh_mean_6h": rh.rolling(6, min_periods=6).mean(),
        "rh_std_6h": rh.rolling(6, min_periods=6).std(),
        "rh_mean_24h": rh.rolling(24, min_periods=24).mean(),
        "rh_std_24h": rh.rolling(24, min_periods=24).std(),
        "humid_share_7d": y.rolling(168, min_periods=168).mean(),
    })


def rolling_by_season(df: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    return (rolling.assign(season=df.set_index("timestamp")["season"])
            .groupby("season", observed=True)[["rh_std_6h", "rh_std_24h", "humid_share_7d"]].median())


# --- Ch. 9 item 6: extremes (extreme != erroneous: counted, never removed) ----------------------

def tail_counts(s: pd.Series) -> dict:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    p01, p99 = s.quantile([0.01, 0.99])
    return {
        "P1": p01, "P99": p99,
        "hours < P1": int((s < p01).sum()), "hours > P99": int((s > p99).sum()),
        "hours outside 1.5 IQR": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
        "min": s.min(), "max": s.max(),
    }


def extremes(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    return pd.DataFrame({LABELS[v]: tail_counts(df[v]) for v in variables}).T


def top_extremes(df: pd.DataFrame, variables: list, target: str, n: int = 5) -> pd.DataFrame:
    """Most extreme hours (top n high; bottom n low for temperature and RH), with timestamps."""
    rows = []
    for v in variables:
        for kind, part in [("highest", df.nlargest(n, v)), ("lowest", df.nsmallest(n, v))]:
            if kind == "lowest" and v in ("precipitation", "wind_speed_10m"):
                continue                                 # minima are just 0 for these
            for _, r in part.iterrows():
                rows.append({"variable": LABELS[v], "kind": kind, "timestamp": r["timestamp"], "value": r[v],
                             "RH at that hour": r[target], "y": r["y"]})
    return pd.DataFrame(rows)


def extreme_days(daily_table: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return pd.concat({
        "wettest days (mm)": daily_table["precipitation"].nlargest(n),
        "driest days (mean RH %)": daily_table["relative_humidity_2m"].nsmallest(n),
        "most humid days (mean RH %)": daily_table["relative_humidity_2m"].nlargest(n),
    }).round(1).rename("value").to_frame()


# --- Original analysis 1: persistence of humid and dry regimes -------------------------------

def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Maximal runs of consecutive hours with the same label (cut by the period boundaries)."""
    src = df[["timestamp", "y", "phase"]].dropna(subset=["y"])
    run_id = (src["y"] != src["y"].shift()).cumsum()
    e = src.groupby(run_id).agg(state=("y", "first"), start=("timestamp", "first"),
                                end=("timestamp", "last"), hours=("y", "size"), phase=("phase", "first"))
    e["regime"] = np.where(e["state"] == 1, "humid", "dry")
    return e.reset_index(drop=True)


def episode_stats(hours: pd.Series) -> pd.Series:
    return pd.Series({
        "episodes": len(hours), "mean (h)": hours.mean(), "median (h)": hours.median(),
        "P90 (h)": hours.quantile(0.9), "max (h)": hours.max(),
        "episodes ≥ 24 h": int((hours >= 24).sum()), "episodes ≥ 72 h": int((hours >= 72).sum()),
        "share of regime hours in episodes ≥ 24 h": hours[hours >= 24].sum() / hours.sum(),
    })


def persistence(eps: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({regime: episode_stats(g["hours"]) for regime, g in eps.groupby("regime")})


def humid_episodes_by_season(eps: pd.DataFrame) -> pd.DataFrame:
    """Humid episodes, grouped by the season in which they start."""
    h = eps[eps["regime"] == "humid"].copy()
    h["season"] = pd.Categorical(h["start"].dt.month.map(SEASON_OF_MONTH), categories=SEASONS, ordered=True)
    return h.groupby("season", observed=True)["hours"].agg(episodes="count", mean_h="mean", median_h="median", max_h="max")


# --- Original analysis 2: hour-of-day x month profile -----------------------------------------

def hour_month_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Humid share per (month, local hour)."""
    return df.pivot_table(index="month", columns="hour", values="y", aggfunc="mean")


def daily_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Most / least humid local hour per season and the day-night amplitude."""
    h = df.pivot_table(index="season", columns="hour", values="y", aggfunc="mean", observed=True)
    return pd.DataFrame({
        "most humid hour": h.idxmax(axis=1), "max share": h.max(axis=1),
        "least humid hour": h.idxmin(axis=1), "min share": h.min(axis=1),
        "amplitude (max - min)": h.max(axis=1) - h.min(axis=1),
    })


# --- Original analysis 3: regime transitions and the persistence reference -------------------

def _eval_with_previous(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[df["phase"] == "evaluation"][["timestamp", "y"]].copy()
    ev["y_prev"] = df["y"].shift(1).loc[ev.index]      # the first one comes from the warm-up
    return ev.dropna()


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    """P(y_t | y_{t-1}) on the evaluation stream: how sticky the regime is."""
    ev = _eval_with_previous(df)
    return pd.crosstab(ev["y_prev"].map({0: "dry (t-1)", 1: "humid (t-1)"}),
                       ev["y"].map({0: "dry (t)", 1: "humid (t)"}), normalize="index")


def switches_per_day(df: pd.DataFrame) -> float:
    ev = _eval_with_previous(df)
    return float((ev["y"] != ev["y_prev"]).mean() * 24)


def persistence_reference(df: pd.DataFrame) -> pd.Series:
    """Leakage-free reference rule y_hat_t = y_{t-1}, scored on the evaluation stream only."""
    ev = _eval_with_previous(df)
    y_true, y_pred = ev["y"].astype(int), ev["y_prev"].astype(int)
    return pd.Series({
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "majority_class_accuracy": max(y_true.mean(), 1 - y_true.mean()),
    })


# --- One call per city: everything the notebooks and the comparison need ----------------------

def profile_city(df: pd.DataFrame, tau: float, cfg: dict) -> dict:
    """Run every table above for one city. `df` is a prepared frame (any city)."""
    variables, target = cfg["data"]["hourly_variables"], cfg["protocol"]["target_variable"]
    df = with_calendar(df, cfg["data"]["timezone"])
    roll = rolling_windows(df, target)
    eps = episodes(df)
    pers = persistence(eps)
    by_ph, by_se = class_table(df, "phase"), class_table(df, "season")
    ref = persistence_reference(df)

    key_numbers = pd.Series({
        "tau_H": tau,
        "humid_share_all": df["y"].mean(),
        "humid_share_warmup": by_ph.loc["warm-up", "humid_share"],
        "humid_share_evaluation": by_ph.loc["evaluation", "humid_share"],
        **{f"humid_share_{s}": by_se.loc[s, "humid_share"] for s in SEASONS},
        "rh_mean": df[target].mean(),
        "rh_std": df[target].std(),
        "rh_std_24h_median": roll["rh_std_24h"].median(),
        "rh_100_share": (df[target] == 100).mean(),
        "humid_episode_mean_h": pers.loc["mean (h)", "humid"],
        "humid_episode_max_h": pers.loc["max (h)", "humid"],
        "humid_episodes_ge_24h": pers.loc["episodes ≥ 24 h", "humid"],
        "humid_episodes_ge_72h": pers.loc["episodes ≥ 72 h", "humid"],
        "dry_episode_mean_h": pers.loc["mean (h)", "dry"],
        "switches_per_day_eval": switches_per_day(df),
        **{f"persistence_{k}": v for k, v in ref.items()},
    }, name="value")

    return {
        "df": df,
        "quality": quality_table(df, variables),
        "special": special_values(df, target),
        "describe": describe(df, variables),
        "by_phase": by_phase(df, variables),
        "by_season": by_season(df, variables),
        "daily": daily(df),
        "class_by_phase": by_ph,
        "class_by_year": class_table(df, "year"),
        "class_by_season": by_se,
        "rolling": roll,
        "rolling_by_season": rolling_by_season(df, roll),
        "extremes": extremes(df, variables),
        "top_extremes": top_extremes(df, variables, target),
        "episodes": eps,
        "persistence": pers,
        "humid_episodes_by_season": humid_episodes_by_season(eps),
        "hour_month": hour_month_profile(df),
        "daily_cycle": daily_cycle(df),
        "transitions": transitions(df),
        "persistence_reference": ref,
        "key_numbers": key_numbers,
    }


def save_profile(profile: dict, city_key: str, out_dir) -> None:
    """Tables for the report: results/profiles/<city>_*.csv."""
    profile["key_numbers"].to_frame().to_csv(out_dir / f"{city_key}_key_numbers.csv")
    profile["episodes"].to_csv(out_dir / f"{city_key}_episodes.csv", index=False)
    profile["describe"].to_csv(out_dir / f"{city_key}_descriptive_stats.csv")
    profile["class_by_season"].to_csv(out_dir / f"{city_key}_class_by_season.csv")
