"""Report figures. One style for every notebook: thin marks, recessive grid, one hue per role.

Color roles are fixed across the whole report:
  humid = blue, dry = orange            (regime figures)
  City A = blue, City B = aqua          (comparison figures; never orange, which means "dry")
  models = categorical slots 1-5 in a fixed order (never cycled)
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from profiling import LABELS, UNITS
from utils import output_dir

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK_2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
HUMID, DRY = BLUE, ORANGE
CITY_COLORS = [BLUE, AQUA]
MODEL_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.linewidth": 1.2, "legend.frameon": False, "font.size": 9.5,
    })


def save(fig, cfg: dict, name: str) -> None:
    """results/figures/<name>.png"""
    fig.savefig(output_dir(cfg, "figures_dir") / f"{name}.png")


def _warmup_marker(ax, evaluation_start) -> None:
    ax.axvline(evaluation_start, color=INK_2, linewidth=0.8, linestyle=":")


# --- Shared protocol (notebook 00) -------------------------------------------------------------

def humidity_with_threshold(prepared: dict, cfg: dict):
    """Hourly RH of both cities with the frozen tau_H and the warm-up period."""
    target, tau = cfg["protocol"]["target_variable"], prepared["tau"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, c in zip(axes, cfg["cities"]):
        df = prepared["frames"][c["key"]]
        ax.plot(df["timestamp"], df[target], color=GREY, linewidth=0.3)
        ax.axhline(tau, color=INK, linestyle="--", linewidth=1, label=f"τ_H = {tau:.1f} %")
        ax.axvspan(df["timestamp"].iloc[0], prepared["warmup_end"], color=GRID, alpha=0.6, label="warm-up")
        ax.set_title(f"{c['role']}: {c['name']}", loc="left")
        ax.set_ylabel("Relative humidity (%)")
        ax.legend(loc="lower left", ncol=2)
    axes[-1].set_xlabel("Time (API clock, fixed offset)")
    fig.suptitle("Relative humidity and the frozen T4 threshold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def monthly_humid_share(prepared: dict, cfg: dict):
    """Share of humid hours per month, both cities (same threshold, so directly comparable)."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for c, color in zip(cfg["cities"], CITY_COLORS):
        s = prepared["frames"][c["key"]].set_index("timestamp")["y"].resample("MS").mean()
        ax.plot(s.index, s.values, color=color, marker="o", markersize=3, label=c["name"])
    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    _warmup_marker(ax, prepared["evaluation_start"])
    ax.set_ylabel("Humid share (y = 1)")
    ax.set_ylim(0, 1)
    ax.set_title("Monthly share of humid hours (T4); dotted vertical line = end of warm-up", loc="left")
    ax.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    return fig


# --- One city (notebooks 01 / 02) ---------------------------------------------------------------

def daily_evolution(profile: dict, prepared: dict, cfg: dict, city_name: str):
    daily, tau, target = profile["daily"], prepared["tau"], cfg["protocol"]["target_variable"]
    variables = cfg["data"]["hourly_variables"]
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 9), sharex=True)
    for ax, v in zip(axes, variables):
        if v == "precipitation":
            ax.bar(daily.index, daily[v], width=1.0, color=BLUE, linewidth=0)
            ax.set_ylabel("mm / day")
        else:
            ax.plot(daily.index, daily[v], color=BLUE)
            ax.set_ylabel(UNITS[v])
        if v == target:
            ax.axhline(tau, color=INK, linestyle="--", linewidth=1)
            ax.annotate(f"τ_H = {tau:.0f} %", xy=(1.0, tau), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_title(f"{LABELS[v]}: daily {'total' if v == 'precipitation' else 'mean'}", loc="left")
        _warmup_marker(ax, prepared["evaluation_start"])
    axes[0].annotate("warm-up | evaluation", xy=(prepared["evaluation_start"], 1.02),
                     xycoords=("data", "axes fraction"), ha="center", color=INK_2, fontsize=8.5)
    fig.suptitle(f"{city_name}: temporal evolution of the weather variables", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def rh_by_month(profile: dict, prepared: dict, cfg: dict, city_name: str):
    df, tau, target = profile["df"], prepared["tau"], cfg["protocol"]["target_variable"]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.boxplot([df.loc[df["month"] == m, target].values for m in range(1, 13)],
               widths=0.55, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color=INK, linewidth=1.5),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    ax.axhline(tau, color=INK, linestyle="--", linewidth=1, label=f"τ_H = {tau:.0f} %")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylabel("Relative humidity (%)")
    ax.set_title(f"{city_name}: hourly relative humidity by month (whiskers = 1.5 IQR)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def humid_share_month_year(profile: dict, city_name: str):
    """Humid share per calendar month, one bar group per year (are the years alike?)."""
    df = profile["df"]
    share = df.pivot_table(index="month", columns="year", values="y", aggfunc="mean")
    years = list(share.columns)
    w = 0.8 / len(years)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, (yr, color) in enumerate(zip(years, MODEL_COLORS)):
        ax.bar(share.index + (i - (len(years) - 1) / 2) * w, share[yr], width=w * 0.92,
               color=color, label=str(yr), linewidth=0)
    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Humid share (y = 1)")
    ax.set_title(f"{city_name}: share of humid hours by month and year", loc="left")
    ax.legend(ncol=len(years), loc="upper left")
    fig.tight_layout()
    return fig


def rolling_overview(profile: dict, prepared: dict, city_name: str):
    roll, tau = profile["rolling"], prepared["tau"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    axes[0].plot(roll.index, roll["rh_mean_24h"], color=BLUE, linewidth=0.8)
    axes[0].axhline(tau, color=INK, linestyle="--", linewidth=1)
    axes[0].set_ylabel("%")
    axes[0].set_title("Relative humidity: 24 h rolling mean (dashed: τ_H)", loc="left")
    axes[1].plot(roll.index, roll["rh_std_24h"], color=BLUE, linewidth=0.8)
    axes[1].set_ylabel("% points")
    axes[1].set_title("Relative humidity: 24 h rolling standard deviation (within-day volatility)", loc="left")
    axes[2].plot(roll.index, roll["humid_share_7d"], color=BLUE, linewidth=1)
    axes[2].axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("share")
    axes[2].set_title("Humid regime: 7-day rolling share of humid hours", loc="left")
    for ax in axes:
        _warmup_marker(ax, prepared["evaluation_start"])
    fig.suptitle(f"{city_name}: rolling / window behaviour", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def rolling_zoom(profile: dict, prepared: dict, cfg: dict, city_name: str,
                 start: str = "2024-07-01", days: int = 14):
    """6 h vs 24 h windows on a short excerpt (three years of 6 h curves are unreadable)."""
    rh = profile["df"].set_index("timestamp")[cfg["protocol"]["target_variable"]]
    roll, tau = profile["rolling"], prepared["tau"]
    start = pd.Timestamp(start)
    zoom = slice(start, start + pd.Timedelta(days=days) - pd.Timedelta(hours=1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax = axes[0]
    ax.plot(rh[zoom].index, rh[zoom], color=GREY, linewidth=0.9, label="hourly RH")
    ax.plot(roll.loc[zoom].index, roll.loc[zoom, "rh_mean_6h"], color=BLUE, linewidth=1.6, label="6 h rolling mean")
    ax.plot(roll.loc[zoom].index, roll.loc[zoom, "rh_mean_24h"], color=ORANGE, linewidth=1.6, label="24 h rolling mean")
    ax.axhline(tau, color=INK, linestyle="--", linewidth=1, label=f"τ_H = {tau:.0f} %")
    ax.set_ylim(rh[zoom].min() - 12, 103)                     # room for the legend below the curves
    ax.set_ylabel("Relative humidity (%)")
    ax.set_title("Rolling mean: 6 h follows the daily cycle, 24 h smooths it out", loc="left")
    ax.legend(ncol=4, loc="lower left")

    ax = axes[1]
    ax.plot(roll.loc[zoom].index, roll.loc[zoom, "rh_std_6h"], color=BLUE, linewidth=1.6, label="6 h rolling std")
    ax.plot(roll.loc[zoom].index, roll.loc[zoom, "rh_std_24h"], color=ORANGE, linewidth=1.6, label="24 h rolling std")
    ax.set_ylim(0, roll.loc[zoom, ["rh_std_6h", "rh_std_24h"]].max().max() * 1.25)
    ax.set_ylabel("% points")
    ax.set_title("Rolling standard deviation: short-term vs within-day volatility", loc="left")
    ax.legend(ncol=2, loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.suptitle(f"{city_name}: 6 h vs 24 h windows, {start:%d %b %Y} + {days} days", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def _duration_axis(ax, max_h: int) -> np.ndarray:
    ax.set_xscale("log")
    ticks = [1, 3, 6, 12, 24, 48, 72, 168]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Episode duration (hours, log scale)")
    ax.set_ylabel("Number of episodes")
    for x, lab in [(24, "1 day"), (72, "3 days")]:
        ax.axvline(x, color=INK_2, linewidth=0.8, linestyle=":")
        ax.annotate(lab, xy=(x, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", color=INK_2, fontsize=8.5)
    return np.unique(np.round(np.logspace(0, np.log10(max_h + 1), 25)).astype(int))


def episode_durations(profile: dict, city_name: str):
    eps = profile["episodes"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = _duration_axis(ax, int(eps["hours"].max()))
    for regime, color in [("humid", HUMID), ("dry", DRY)]:
        h = eps.loc[eps["regime"] == regime, "hours"]
        ax.hist(h, bins=bins, histtype="step", linewidth=1.6, color=color, label=f"{regime} (n = {len(h)})")
    ax.set_title(f"{city_name}: how long humid and dry regimes last", loc="left")
    ax.legend()
    fig.tight_layout()
    return fig


def _heatmap(ax, table: pd.DataFrame):
    im = ax.imshow(table.values, aspect="auto", cmap=SEQ, vmin=0, vmax=1, origin="upper")
    ax.set_yticks(range(12), MONTHS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Local hour of day (Europe/Lisbon)")
    ax.grid(False)
    return im


def hour_month_heatmap(profile: dict, city_name: str):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = _heatmap(ax, profile["hour_month"])
    fig.colorbar(im, ax=ax, pad=0.01).set_label("Humid share (y = 1)")
    ax.set_title(f"{city_name}: share of humid hours by month and hour of day", loc="left")
    fig.tight_layout()
    return fig


# --- City comparison (notebook 03) --------------------------------------------------------------

def compare_episode_durations(profiles: dict, cfg: dict):
    """Humid episode durations, both cities on one axis."""
    max_h = max(int(p["episodes"]["hours"].max()) for p in profiles.values())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = _duration_axis(ax, max_h)
    for c, color in zip(cfg["cities"], CITY_COLORS):
        eps = profiles[c["key"]]["episodes"]
        h = eps.loc[eps["regime"] == "humid", "hours"]
        ax.hist(h, bins=bins, histtype="step", linewidth=1.6, color=color, label=f"{c['name']} (n = {len(h)})")
    ax.set_title("Humid episode durations: City A vs City B", loc="left")
    ax.legend()
    fig.tight_layout()
    return fig


def compare_hour_month(profiles: dict, cfg: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, c in zip(axes, cfg["cities"]):
        im = _heatmap(ax, profiles[c["key"]]["hour_month"])
        ax.set_title(f"{c['role']}: {c['name']}", loc="left")
    fig.colorbar(im, ax=axes, pad=0.01).set_label("Humid share (y = 1)")
    fig.suptitle("Share of humid hours by month and local hour", x=0.01, ha="left")
    return fig


def compare_models(metrics: pd.DataFrame, references: dict, cfg: dict, metric: str = "macro_f1"):
    """Dot plot, one panel per representation: rows = models, color = city, dashed = persistence rule.

    Dots (not bars) so the axis can zoom on the range where the models actually differ.
    """
    reps = sorted(metrics["representation"].unique())
    models = list(dict.fromkeys(metrics["model"]))[::-1]          # first model on top
    values = pd.concat([metrics[metric], pd.Series([r[metric] for r in references.values()])])
    lo, hi = values.min() - 0.02, min(values.max() + 0.02, 1.0)
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, len(reps), figsize=(13, 3.8), sharey=True, sharex=True)
    for ax, rep in zip(np.atleast_1d(axes), reps):
        for i, (c, color) in enumerate(zip(cfg["cities"], CITY_COLORS)):
            sub = metrics[(metrics["representation"] == rep) & (metrics["city"] == c["key"])].set_index("model")
            ax.axvline(references[c["key"]][metric], color=color, linestyle="--", linewidth=1)
            ax.scatter(sub.loc[models, metric], y - (i - 0.5) * 0.22, s=46, color=color, zorder=3,
                       edgecolor="white", linewidth=1.5, label=c["name"])
        ax.set_yticks(y, models)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"Representation {rep}", loc="left")
    fig.suptitle(f"Prequential {metric.replace('_', ' ')} per model "
                 f"(dashed: persistence rule y_t-1 of the same city)", x=0.01, ha="left")
    fig.tight_layout()
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels))
    return fig


def recent_performance(history: pd.DataFrame, cfg: dict, representation: str = "B",
                       metric: str = "recent_macro_f1"):
    """Sliding-window performance over time, one panel per city in `history`, one line per model."""
    h = history[history["representation"] == representation]
    models = list(dict.fromkeys(h["model"]))
    cities = [c for c in cfg["cities"] if c["key"] in set(h["city"])]
    fig, axes = plt.subplots(len(cities), 1, figsize=(12, 3.3 * len(cities)), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, c in zip(axes, cities):
        for m, color in zip(models, MODEL_COLORS):
            s = h[(h["city"] == c["key"]) & (h["model"] == m)]
            ax.plot(s["timestamp"], s[metric], color=color, linewidth=1, label=m)
        ax.set_title(f"{c['role']}: {c['name']}", loc="left")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1)
    axes[0].legend(ncol=len(models), loc="lower left", fontsize=8)
    fig.suptitle(f"Representation {representation}: {metric.replace('_', ' ')} "
                 f"(last {cfg['protocol']['evaluation_window']} events)", x=0.01, ha="left")
    fig.tight_layout()
    return fig
