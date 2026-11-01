# Practical Assignment I: Business-Oriented Data Stream Mining
**NOVA IMS · Data Stream Mining · PA04**

---

## 1. Audit Metadata & Project Overview

| | |
|---|---|
| **Assignment Code** | PA04 |
| **Student A (primary City A)** | Khadija Belhamra, 20230007: Guimarães |
| **Student B (primary City B)** | Victoriya Yurevna, 20230796: Viana do Castelo |
| **Business context** | Textile storage facility (humid vs. dry regimes) |
| **Locations** | Guimarães vs Viana do Castelo |
| **Question** | How do humid and dry regimes differ between candidate warehouse locations? |
| **Official Target** | T4, Humid vs Dry Regime: y_t = 1 if RH_t ≥ τ_H, with τ_H = Q0.50(RH) over the pooled warm-up of both cities, frozen (τ_H = 82 %) |
| **Historical Data Period** | 2023-01-01 to 2025-12-31, hourly (Open-Meteo reanalysis). Warm-up 2023, evaluation 2024–2025 |
| **Random Seed** | 42 |

---

## 2. How the Project Works

A textile warehouse cares about **how often, when and for how long** outdoor air is humid, because long humid periods drive moisture uptake, mould risk and dehumidification cost. We turn three years of hourly weather in two candidate locations into two **data streams**. We label every hour humid or dry (T4), and we ask two questions:

1. **Descriptively:** how do the humid and dry regimes differ between Guimarães (inland) and Viana do Castelo (coast)?
2. **Predictively:** can incremental models, learning one hour at a time, anticipate the regime of the next hour better than the rule "the regime stays the same", and does this differ between the two locations?

The project is built on four principles:

* **One configuration.** Every setting lives in `config.yaml`: period, cities, warm-up, quantile, quality ranges, feature windows and model hyper-parameters. Nothing is set by hand in a notebook, so both cities are always treated identically (Ch. 3.1).
* **One implementation of the protocol.** All logic is in `src/`. `data.prepare_frames(cfg)` is the only place that loads the data, defines the warm-up and computes τ_H, and every notebook calls it, so τ_H cannot diverge between notebooks.
* **Thin notebooks.** Notebooks call `src/` functions and hold what is ours to write: interpretation and individual experiments.
* **No intermediate data files.** Collect once → save raw → rebuild the clean streams in memory on every run (about a second) → replay as a stream. No processed copy is stored, so nothing can go stale. Traceability comes from the raw snapshots, `config.yaml` and `results/config_part1.json`.

### Pipeline

```text
                    ┌─► 00_shared_protocol ─────────────► protocol checks, results/config_part1.json
config.yaml ──► data.prepare_frames(cfg) ─┤   (both)
 (τ_H computed      │                     ├─► 01_cityA_guimaraes ─┐   profiling + features → models → evaluate
  here only)        │                     │   (Khadija)           ├─► results/metrics_<city>.csv, history_<city>.csv
                    │                     ├─► 02_cityB_viana… ────┘
                    │                     │   (Victoriya)
                    │                     └─► 03_shared_comparison ◄── evaluate.load_results (both cities)
                    │                         (both)
               src/*.py: every notebook calls the same functions
```

Inside every notebook the data flows the same way:

```text
raw JSON ─► to_frame ─► common timeline ─► quality flags ─► warm-up (12 months) ─► τ_H (pooled, frozen) ─► target y
  (data.py)                                                                                                    │
            profiling.py ◄────────────────────────────────────────────────────────────────────────────────────┤
            features.make_streams (A: t−1 values · B: + delta, 6 h/24 h rolling mean/std, humid hours 24 h) ◄──┘
                 └─► models.make_models (5 learners) ─► evaluate.run_prequential (PREDICT → EVALUATE → LEARN)
```

---

## 3. Team Workflow

The workflow mirrors the assignment: a **shared** protocol, an **individual** city analysis per student, and a **shared** comparison for the final report.

| Step | Who | Notebook | What happens | Output |
|---|---|---|---|---|
| 1 | Both | `00_shared_protocol` | Load both raw snapshots, align them, flag invalid values, fix the warm-up, freeze τ_H on the pooled warm-up, build T4, run the protocol checks and the leakage test | `results/config_part1.json`, `configurations.csv`, `stream_construction_summary.csv`, `shared_*` figures |
| 2a | Khadija | `01_cityA_guimaraes` | Profile Guimarães (Ch. 9 + 3 original analyses), run the 5 models × 2 representations, compare with the persistence rule, individual experiments, interpretation | `results/metrics_guimaraes.csv`, `history_guimaraes.csv`, `profiles/guimaraes_*`, `guimaraes_*` figures |
| 2b | Victoriya | `02_cityB_viana_do_castelo` | The same for Viana do Castelo | `results/metrics_viana_do_castelo.csv`, `history_viana_do_castelo.csv`, `profiles/viana_do_castelo_*`, `viana_do_castelo_*` figures |
| 3 | Both | `03_shared_comparison` | Pair both students' results: key numbers side by side, persistence and timing of the humid regime, model comparison, performance over time, cost; joint business answer and technical recommendation | `comparison_*` figures, final report sections |

Steps 2a and 2b are independent and can be done in parallel. Step 3 needs both.

**Current state:** all four notebooks have been run, and their records, tables and figures are in `results/`.

### Rules that keep the two cities comparable

1. **Never set a protocol value in a notebook.** Period, warm-up, τ_H, features and hyper-parameters come from `config.yaml` through `src/`.
2. **Individual experiments use a copy of the config.** In each city notebook: `my_cfg = copy.deepcopy(cfg)`. Experiment results are never saved with `save_results`, which is reserved for the official run.
3. **A protocol change is a joint decision.** If an experiment shows a clearly better setting, change `config.yaml` together, then re-run `00`, `01`, `02` and `03`.
4. **Changes to `src/` are shared code.** Agree on them, because they change both cities' results.

---

## 4. Directory Layout and What Each File Is For

```text
PA04_Group04/
├── README.md
├── config.yaml
├── requirements.txt
├── .python-version
├── data/
│   ├── README.md
│   ├── guimaraes_raw.json
│   └── viana_do_castelo_raw.json
├── src/
│   ├── utils.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── profiling.py
│   └── plots.py
├── notebooks/
│   ├── 00_shared_protocol.ipynb
│   ├── 01_cityA_guimaraes.ipynb
│   ├── 02_cityB_viana_do_castelo.ipynb
│   └── 03_shared_comparison.ipynb
└── results/
    ├── config_part1.json
    ├── configurations.csv
    ├── stream_construction_summary.csv
    ├── metrics_<city>.csv
    ├── history_<city>.csv
    ├── profiles/
    └── figures/
```

### Configuration and data

| File | Purpose |
|---|---|
| `config.yaml` | Single source of truth. `project` (assignment, business scenario, question, target), `data` (API, period, timezone, variables), `cities` (key, name, role, coordinates, raw file), `storage` (output folders, `force_download`), `quality` (definitional ranges), `protocol` (seed, target variable and quantile, warm-up months, evaluation window), `features` (rolling variables and windows), `models` (hyper-parameters) |
| `requirements.txt` | Python dependencies (see section 5) |
| `.python-version` | Python version used by `uv venv` (3.12) |
| `data/*_raw.json` | Raw Open-Meteo snapshots, one per city, never modified |
| `data/README.md` | Dataset description: source, snapshot checksums, grid points, variables, units, quality, timestamps, limitations |

### Supporting modules (`src/`)

| Module | Responsibility | Main functions |
|---|---|---|
| `utils.py` | Generic helpers with no project logic | `load_config`, `output_dir`, `download_weather`, `save_json` / `load_json`, `package_version`, `save_configuration` |
| `data.py` | **The shared protocol** (steps 1–3): raw snapshots → clean tables → warm-up → frozen τ_H → target | `get_snapshot` (reuse or download once), `to_frame`, `flag_invalid`, `quality_report`, `chronology_report`, `warmup_size`, `frozen_threshold`, `add_target`, **`prepare_frames`** (entry point), `protocol_checks`, `stream_summary`, `save_protocol_records` |
| `features.py` | Past-only features and replay events (steps 4–5) | `build_features` (Representations A and B, every window ends at t−1), `check_no_leakage` (corrupts the future and asserts that the features do not change), `make_streams` (scaler fitted on the pooled warm-up, River-style events) |
| `models.py` | The five mandatory River learners, always fresh instances | `make_models`: Gaussian NB, Online Logistic Regression (SGD), Hoeffding Tree, Online Bagging, Adaptive Random Forest |
| `evaluate.py` | Prequential evaluation (step 7) and result files | `run_prequential` (PREDICT → EVALUATE → LEARN, cumulative and sliding-window metrics, latency), `evaluate_models` (all models × representations on one city), `save_results`, `load_results` |
| `profiling.py` | Stream characterisation of one city (Ch. 9 + original analyses) | `profile_city` (all tables in one call), quality, descriptive statistics, class distribution, rolling windows, extremes, regime episodes and persistence, hour × month profile, transitions, `persistence_reference` (ŷ_t = y_{t−1}), `save_profile` |
| `plots.py` | Report figures with one style and fixed color roles (humid = blue, dry = orange; City A = blue, City B = aqua) | `set_style`, `save`, shared figures, per-city figures, comparison figures (`compare_models`, `recent_performance`, …) |

### Notebooks (`notebooks/`)

| Notebook | Owner | Content |
|---|---|---|
| `00_shared_protocol.ipynb` | Both | Assignment card, raw snapshots and grid points, chronology, data quality, warm-up, τ_H, target, class distribution, protocol checks + leakage test, saved records |
| `01_cityA_guimaraes.ipynb` | Khadija Belhamra | Stream profile of Guimarães (Ch. 9 items 1–7), original analyses, business interpretation, official prequential run, individual experiments, summary for the report |
| `02_cityB_viana_do_castelo.ipynb` | Victoriya Yurevna | The same analysis for Viana do Castelo |
| `03_shared_comparison.ipynb` | Both | City A vs City B side by side, model comparison against persistence, performance over time, cost, joint business and technical recommendations, limitations. Run it after both city notebooks, because it reads their saved results |

### Results (`results/`)

| File | Written by | Content |
|---|---|---|
| `config_part1.json` | 00 | Frozen protocol record: τ_H, warm-up boundaries, period, cities, grid points, valid ranges, seed, package versions |
| `configurations.csv` | 00 | Every parameter of `config.yaml` plus the frozen values and versions, one per row |
| `stream_construction_summary.csv` | 00 | Table for the report section *Data and Stream Construction* |
| `metrics_<city>.csv` | 01 / 02 | One row per representation × model: accuracy, Macro-F1, precision, recall, F1 (humid), confusion counts, prediction and update time per event |
| `history_<city>.csv` | 01 / 02 | Cumulative and recent (last 500 events) accuracy and Macro-F1, every 24 events |
| `profiles/<city>_*.csv` | 01 / 02 | Key numbers, regime episodes, descriptive statistics, class distribution by season |
| `figures/` | all | `shared_*` (00), `guimaraes_*` (01), `viana_do_castelo_*` (02), `comparison_*` (03) |

---

## 5. Environment & System Requirements

| | |
|---|---|
| **Python** | 3.12, pinned in `.python-version` so we both use the same version. The project was run end to end with 3.12. `river` and `pandas` also install on 3.13–3.15, but those versions were not run end to end |
| **Packages** | `requirements.txt` pins the exact tested versions of `PyYAML`, `requests`, `pandas`, `numpy`, `river`, `scikit-learn`, `matplotlib`, `ipykernel`, `nbconvert` |
| **Package manager** | [`uv`](https://docs.astral.sh/uv/) (recommended), or plain `pip` |
| **Tested with** | Python 3.12.12, river 0.26.1, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4, matplotlib 3.11.2 (macOS, Apple Silicon) |

### Set up the environment (once, from the project root)

```bash
# 1. Create the virtual environment in .venv (uv reads .python-version and downloads Python 3.12 if needed)
uv venv

# 2. Install the dependencies
uv pip install -r requirements.txt

# 3. Check it works: prints the versions of the main packages
.venv/bin/python -c "import river, sklearn, pandas; print(river.__version__, sklearn.__version__, pandas.__version__)"
```

Without `uv`:
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Use it in VS Code / Jupyter
* **VS Code:** open a notebook → *Select Kernel* (top right) → *Python Environments* → `.venv (Python 3.12)`.
* **Jupyter in the terminal:** `source .venv/bin/activate`, then run notebooks with `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.

### Notes
* **Rebuilding:** if `.venv` is missing packages or uses another Python (`.venv/bin/python --version` does not show 3.12), delete it (`rm -rf .venv`) and repeat the setup steps.
* **`.venv/` is not committed** (`.gitignore`). Each of us builds their own from `requirements.txt`.

---

## 6. Instructions to Reproduce the Main Experiment

### Step 1: Extract the audit directory
```bash
unzip PA04_Group04_Audit.zip
cd PA04_Group04/
```

### Step 2: Set up the environment
See section 5 (`uv venv && uv pip install -r requirements.txt`).

### Step 3: Run the notebooks in order
1. `notebooks/00_shared_protocol.ipynb`
2. `notebooks/01_cityA_guimaraes.ipynb` and `notebooks/02_cityB_viana_do_castelo.ipynb` (about 15 s of model training each)
3. `notebooks/03_shared_comparison.ipynb` (needs the results of both city notebooks)

Or non-interactively, from the project root:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Each notebook finds the project root and puts `src/` on the path by itself, whatever folder Jupyter is started from. The raw snapshots are reused. They are downloaded again only if missing or if `storage.force_download: true`.

**Keep `storage.force_download: false`.** The results are based on the snapshots in `data/`, which use a fixed UTC+1 clock. A fresh Open-Meteo download now arrives on a UTC clock, which shifts every row by one hour and changes the results.

**Re-running 01 and 02** reproduces every quality metric exactly, but the prediction and update times depend on the machine and on the run. It overwrites the timing values in `results/metrics_<city>.csv`, which notebook 03 reads. The timing values in the submitted files are the ones to report, so re-run 03 after 01 and 02 if you regenerate them. For a quick smoke test, set `protocol.max_eval_events: 500` in `config.yaml`, and set it back to `null` for the official results.

---

## 7. Experimental Protocol Summary
1. **Threshold Policy:** τ_H is the median of the pooled warm-up relative humidity of both cities (rows flagged `ok` only). It is computed once in `data.prepare_frames` and frozen before evaluation.
2. **Warm-up:** the first 12 months (2023) are used for τ_H, the feature scaler and model warm-up. They are never scored.
3. **Target Leakage Control:** Representation A uses values at t−1. Representation B adds the delta, 6 h / 24 h rolling mean and std of RH, and humid hours in the previous 24 h, all ending at t−1. `features.check_no_leakage` corrupts the future and asserts that the features do not change, with a positive control on the past.
4. **Prequential Evaluation Protocol:** strict chronological order, **PREDICT → EVALUATE → LEARN** per event, with cumulative metrics and recent metrics over a 500-event sliding window. Positive class = humid.
5. **Mandatory Models:** Gaussian Naive Bayes, Online Logistic Regression (SGD), Hoeffding Tree, Online Bagging and Adaptive Random Forest, evaluated under identical chronological histories for both cities and both representations.
6. **Reference:** the persistence rule ŷ_t = y_{t−1} is the bar the models have to clear.
7. **Data Quality:** invalid values (missing or outside definitional ranges) are flagged, never silently deleted. Extreme ≠ erroneous. See `data/README.md`.
