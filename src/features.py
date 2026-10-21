"""Steps 4-5: past-only features (Representations A and B), warm-up scaler, replay events.

LEAKAGE RULE: every feature for row t is built from rows <= t-1.
`shift(1)` is applied BEFORE any rolling operation, so windows end at t-1.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data import add_target


def build_features(df: pd.DataFrame, cfg: dict):
    """Return (A, B) feature tables aligned with df. `df` must already contain the target `y`."""
    f = cfg["features"]
    past = df[cfg["data"]["hourly_variables"]].shift(1)       # value at t-1 for every variable

    A = past.add_suffix("_lag1")                              # Representation A: previous state
    B = A.copy()                                              # Representation B = A + context below
    for v in f["rolling_variables"]:
        B[f"{v}_delta"] = past[v].diff()                      # x(t-1) - x(t-2)
        for w in f["windows"]:
            B[f"{v}_mean{w}"] = past[v].rolling(w).mean()     # mean of x(t-1) ... x(t-w)
            B[f"{v}_std{w}"] = past[v].rolling(w).std()
    w = f["event_window"]
    B[f"humid_hours_{w}h"] = df["y"].shift(1).rolling(w).sum()  # humid hours in the previous w hours
    return A, B


def check_no_leakage(raw: pd.DataFrame, tau: float, cfg: dict, t: int = 1000) -> None:
    """Corrupt the future from row t on: features up to row t must not change.
    Positive control: corrupting the past must change the features at row t."""
    variables = cfg["data"]["hourly_variables"]
    target_var = cfg["protocol"]["target_variable"]

    def features_of(frame):
        return build_features(add_target(frame, target_var, tau), cfg)

    A0, B0 = features_of(raw)

    future = raw.copy()
    future.loc[future.index >= t, variables] = 9999.0
    A1, B1 = features_of(future)
    pd.testing.assert_frame_equal(A0.iloc[: t + 1], A1.iloc[: t + 1])
    pd.testing.assert_frame_equal(B0.iloc[: t + 1], B1.iloc[: t + 1])

    past = raw.copy()
    past.loc[past.index < t, variables] = 9999.0
    _, B2 = features_of(past)
    assert not B0.iloc[t].equals(B2.iloc[t]), "Leakage test is not sensitive: features ignore the past."


def make_streams(prepared: dict, cfg: dict) -> dict:
    """streams[city][representation] = {"warm": [...], "eval": [...]} with scaled River-style events.

    - Rows without enough history (the first max(windows) rows) are dropped, identically for A and B.
    - One MinMaxScaler per representation is fitted on the pooled WARM-UP rows only, then frozen.
    """
    n_warm = prepared["n_warm"]
    tables = {}
    for city, df in prepared["frames"].items():
        A, B = build_features(df, cfg)
        valid = B.notna().all(axis=1) & df["y"].notna()        # same rows for A and B
        tables[city] = {"A": A, "B": B, "y": df["y"], "ts": df["timestamp"], "valid": valid}

    streams = {city: {} for city in tables}
    for rep in ("A", "B"):
        warm_rows = pd.concat([t[rep][t["valid"] & (t[rep].index < n_warm)] for t in tables.values()])
        scaler = MinMaxScaler(feature_range=(0.0, 1.0), clip=True).fit(warm_rows)
        for city, t in tables.items():
            streams[city][rep] = {
                "warm": _events(t, rep, scaler, t["valid"] & (t[rep].index < n_warm)),
                "eval": _events(t, rep, scaler, t["valid"] & (t[rep].index >= n_warm)),
            }
    max_eval = cfg["protocol"]["max_eval_events"]
    if max_eval:
        for city in streams:
            for rep in streams[city]:
                streams[city][rep]["eval"] = streams[city][rep]["eval"][: int(max_eval)]
    return streams


def _events(t: dict, rep: str, scaler: MinMaxScaler, mask: pd.Series) -> list:
    X = pd.DataFrame(scaler.transform(t[rep][mask]), columns=t[rep].columns)
    return [
        {"timestamp": ts, "x": x, "y": int(y)}
        for ts, x, y in zip(t["ts"][mask], X.to_dict("records"), t["y"][mask])
    ]