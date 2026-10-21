"""Step 7: prequential evaluation. PREDICT -> EVALUATE -> LEARN, one event at a time.

Positive class = 1 = humid regime (the event). Precision, Recall and F1 refer to it.
"""
from collections import deque
from time import perf_counter

import pandas as pd

from utils import output_dir


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


class BinaryConfusion:
    """Exact confusion counts that can also remove a pair (needed for the sliding window)."""

    def __init__(self):
        self.tp = self.tn = self.fp = self.fn = 0

    def add(self, y_true: int, y_pred: int, sign: int = 1) -> None:
        if y_true == 1 and y_pred == 1:
            self.tp += sign
        elif y_true == 0 and y_pred == 0:
            self.tn += sign
        elif y_true == 0 and y_pred == 1:
            self.fp += sign
        else:
            self.fn += sign

    def scores(self) -> dict:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
        f1_event = _div(2 * precision * recall, precision + recall)
        p0, r0 = _div(tn, tn + fn), _div(tn, tn + fp)
        f1_other = _div(2 * p0 * r0, p0 + r0)
        return {
            "accuracy": _div(tp + tn, tp + tn + fp + fn),
            "macro_f1": (f1_event + f1_other) / 2,
            "precision": precision,
            "recall": recall,
            "f1_event": f1_event,
        }


class StreamingEvaluator:
    """Cumulative and recent (last `window` predictions) metrics, updated incrementally."""

    def __init__(self, window: int):
        self.window = int(window)
        self.total, self.recent = BinaryConfusion(), BinaryConfusion()
        self.pairs = deque()

    def update(self, y_true: int, y_pred: int) -> None:
        if len(self.pairs) == self.window:
            self.recent.add(*self.pairs.popleft(), sign=-1)
        self.pairs.append((y_true, y_pred))
        self.total.add(y_true, y_pred)
        self.recent.add(y_true, y_pred)


def run_prequential(model, warm: list, stream: list, cfg: dict) -> dict:
    """Warm up (learn only, no scoring), then test-then-train on the evaluation stream."""
    for e in warm:
        model.learn_one(e["x"], e["y"])

    evaluator = StreamingEvaluator(cfg["protocol"]["evaluation_window"])
    every = cfg["protocol"]["history_every"]
    predict_s = learn_s = 0.0
    cold_starts = 0
    history = []

    for i, e in enumerate(stream, start=1):
        t0 = perf_counter()
        y_pred = model.predict_one(e["x"])               # 1) PREDICT with past-only state
        t1 = perf_counter()
        if y_pred is None:
            cold_starts += 1
        else:
            evaluator.update(e["y"], int(y_pred))        # 2) EVALUATE before the label is used
        t2 = perf_counter()
        model.learn_one(e["x"], e["y"])                  # 3) LEARN
        t3 = perf_counter()

        predict_s += t1 - t0
        learn_s += t3 - t2
        if i % every == 0:
            recent = evaluator.recent.scores()
            history.append({
                "timestamp": e["timestamp"],
                "n": i,
                "cum_accuracy": evaluator.total.scores()["accuracy"],
                "cum_macro_f1": evaluator.total.scores()["macro_f1"],
                "recent_accuracy": recent["accuracy"],
                "recent_macro_f1": recent["macro_f1"],
            })

    c = evaluator.total
    n = max(len(stream), 1)
    return {
        "metrics": {
            "n_warmup": len(warm),
            "n_eval": len(stream),
            "event_rate_eval": _div(sum(e["y"] for e in stream), len(stream)),
            "cold_starts": cold_starts,
            **c.scores(),
            "tn": c.tn, "fp": c.fp, "fn": c.fn, "tp": c.tp,
            "pred_ms_per_event": 1000 * predict_s / n,
            "update_ms_per_event": 1000 * learn_s / n,
        },
        "history": pd.DataFrame(history),
    }


def evaluate_models(city_streams: dict, make_models, cfg: dict, city: str):
    """Every model x every representation on ONE city's streams (`make_streams(...)[city]`).

    `make_models` is called once per representation, so every run starts from fresh models.
    Returns (metrics, history) DataFrames tagged with city / representation / model.
    """
    rows, histories = [], []
    for rep, s in city_streams.items():
        for name, model in make_models(cfg).items():
            res = run_prequential(model, s["warm"], s["eval"], cfg)
            tag = {"city": city, "representation": rep, "model": name}
            rows.append({**tag, **res["metrics"]})
            histories.append(res["history"].assign(**tag))
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True)


def save_results(metrics: pd.DataFrame, history: pd.DataFrame, city: str, cfg: dict) -> None:
    """results/metrics_<city>.csv and results/history_<city>.csv (one pair per student notebook)."""
    out = output_dir(cfg, "results_dir")
    metrics.to_csv(out / f"metrics_{city}.csv", index=False)
    history.to_csv(out / f"history_{city}.csv", index=False)


def load_results(cfg: dict):
    """Both cities' saved results, concatenated in config order (for the comparison notebook)."""
    out = output_dir(cfg, "results_dir")
    missing = [c["key"] for c in cfg["cities"] if not (out / f"metrics_{c['key']}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"No saved results for {missing}: run that city's notebook first.")
    metrics = pd.concat([pd.read_csv(out / f"metrics_{c['key']}.csv") for c in cfg["cities"]], ignore_index=True)
    history = pd.concat([pd.read_csv(out / f"history_{c['key']}.csv", parse_dates=["timestamp"])
                         for c in cfg["cities"]], ignore_index=True)
    return metrics, history
