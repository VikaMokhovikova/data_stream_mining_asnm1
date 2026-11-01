# Architecture

How the code of PA04 is organised, what each part guarantees, and where to make a change without breaking the shared protocol.
For setup, workflow and reproduction, see `README.md`. For the dataset, see `data/README.md`.

---

## 1. Three layers

| Layer | Where | Rule |
|---|---|---|
| **Configuration** | `config.yaml` | Every setting, defined once, identical for both cities. Nothing else hard-codes a setting. |
| **Logic** | `src/*.py` | All computation, as plain functions. No notebook-specific code, no state kept between calls. |
| **Analysis** | `notebooks/*.ipynb` | Calls `src/` functions, shows evidence, holds the students' interpretation and individual experiments. |

Data never takes a shortcut between layers: notebooks do not re-implement protocol steps, and modules do not read notebook outputs.

---

## 2. Module dependencies

```text
utils.py ◄── data.py ◄── features.py
   ▲
   ├──────── evaluate.py
   │
   └──────── plots.py ──► profiling.py

models.py       (no project imports: River only)
profiling.py    (no project imports: pandas / scikit-learn only)
```

* `utils.py` sits at the bottom and knows nothing about the project.
* There are no cycles. Each module can be imported on its own.
* `features.py` imports `data.add_target` only for the leakage test (it rebuilds the target on corrupted data).

---

## 3. Data flow and contracts

```text
config.yaml
   │  load_config()
   ▼
data.prepare_frames(cfg) ─────────────────────────────────────────────── the ONLY entry point to the data
   │  returns prepared = {
   │     "payloads": {city: raw JSON},                 grid point, elevation, units
   │     "raw":      {city: table before target},      input of the leakage test
   │     "frames":   {city: table + quality_flag, phase, y},
   │     "tau", "n_warm", "warmup_end", "evaluation_start", "n_pooled" }
   │
   ├──► profiling.profile_city(frames[city], tau, cfg) ──► dict of tables ──► plots.* ──► results/figures/
   │                                                                     └──► save_profile ──► results/profiles/
   │
   └──► features.make_streams(prepared, cfg)
           │  returns streams[city][rep] = {"warm": [event], "eval": [event]}
           │          event = {"timestamp", "x": {feature: value in [0, 1]}, "y": 0 | 1}
           ▼
        evaluate.evaluate_models(streams[city], make_models, cfg, city)
           │  for every representation × model: run_prequential(model, warm, eval, cfg)
           ▼
        (metrics, history) ──► save_results ──► results/metrics_<city>.csv, history_<city>.csv
                                                     ▲
        03_shared_comparison ◄── load_results ───────┘
```

### What each step guarantees

| Function | Guarantee |
|---|---|
| `data.get_snapshot` | Raw JSON is downloaded at most once and never modified. It is downloaded only if missing or `storage.force_download: true`. |
| `data.prepare_frames` | Both cities share exactly the same timestamps. Invalid values are flagged, never deleted. τ_H is computed here and nowhere else, from pooled warm-up rows flagged `ok`. |
| `data.protocol_checks` | Raises if any temporal-alignment check fails (order, identical timelines and warm-up, τ_H from warm-up only, target definition). |
| `features.build_features` | Every feature for hour t uses rows ≤ t−1: `shift(1)` is applied before any rolling window. |
| `features.check_no_leakage` | Raises if corrupting the future changes past features, or if corrupting the past does not change them (positive control). |
| `features.make_streams` | The same rows are kept for A and B. One scaler per representation is fitted on the pooled warm-up of both cities, then frozen. Raises if a row used for the fit is not before the evaluation start. |
| `models.make_models` | Returns **fresh** instances on every call, so no state is shared between runs. |
| `evaluate.run_prequential` | Learns on the warm-up without scoring, then strictly PREDICT → EVALUATE → LEARN on each evaluation event. |
| `evaluate.save_results` / `load_results` | One pair of files per city; the comparison reads both cities in `config.yaml` order and raises if one is missing. |

### Replay, not live ingestion

The evaluation stream (2024–2025) is historical. Features and scaling are computed before the replay, with parameters frozen on the warm-up. Both are applied to each hour on its own and use only past data, so this is equivalent to processing each event on arrival. The models are the only component that changes during the stream.

---

## 4. Where to make a change

| I want to… | Change | Shared? |
|---|---|---|
| Change the period, cities, warm-up, quantile, windows or hyper-parameters | `config.yaml` | **Yes**: re-run 00, 01, 02, 03 |
| Try a setting for my city only | `my_cfg = copy.deepcopy(cfg)` in my notebook; never `save_results` with it | No |
| Add a feature to Representation B for everyone | `features.build_features` (keep `shift(1)` first) | **Yes** |
| Add a feature for my experiments only | add it to the events in my notebook (past-only, warm-up scaling) | No |
| Try another model | pass my own factory to `evaluate_models` | No |
| Add a model to the official comparison | `models.make_models` | **Yes** |
| Add a descriptive table or figure | my notebook; move it to `profiling.py` / `plots.py` if both cities need it | Only if moved |
| Change how metrics are computed | `evaluate.py` | **Yes** |

**Shared = joint decision.** A change to `config.yaml` or `src/` changes both cities' results. Agree on it first, then both re-run.

---

## 5. Conventions

* **Paths** come from `config.yaml → storage` through `utils.output_dir(cfg, name)`, which creates the folder if needed. No module hard-codes a results path.
* **City keys** (`guimaraes`, `viana_do_castelo`) name every per-city file: `metrics_<key>.csv`, `profiles/<key>_*.csv`, `figures/<key>_NN_<name>.png`. Experiment figures use `<key>_exp_<name>`.
* **Figure prefixes:** `shared_*` (00), `<key>_*` (01 / 02), `comparison_*` (03).
* **Colors** are fixed roles in `plots.py`: humid = blue, dry = orange; City A = blue, City B = aqua; models take the categorical slots in a fixed order.
* **Outputs are never inputs.** `results/` is written, never read back, except `metrics_*` / `history_*` by `load_results` in notebook 03. Nothing in `data/` is derived; the clean streams are rebuilt in memory on every run.
* **Imports in notebooks:** the first code cell finds the project root (the folder with `config.yaml`) and adds `src/` to `sys.path`, so notebooks run from any working directory.
