# config_vika — individual experiments of Victoriya (City B: Viana do Castelo)

These files document the **individual experiments** of `notebooks/02_cityB_viana_do_castelo.ipynb`, Section 11.2.
They are **not** part of the official protocol: the official results of both cities use `../config.yaml`, the
modules in `../src/`, and are saved in `../results/metrics_<city>.csv` / `history_<city>.csv`. Nothing in this
folder is read by notebooks 00–03 or by the report.

## What is in this folder

| File | Content |
|---|---|
| `E1_temperature_context.yaml` | full copy of `config.yaml` with **one** change: `features.rolling_variables` = RH **+ temperature** (adds delta, 6 h / 24 h rolling mean and std of temperature to Representation B) |
| `E2_windows_3_12_48.yaml` | full copy of `config.yaml` with **one** change: `features.windows` = `[3, 12, 48]` instead of `[6, 24]` |
| `E3_hour_of_day.yaml` | identical to `config.yaml`; the change is two extra features added in notebook 02 (`with_hour`): `hour_sin` and `hour_cos` of the local hour, scaled to [0, 1]. The clock time of hour *t* is known in advance, so it is not leakage |
| `viana_do_castelo_experiments_macro_f1.csv` | Macro-F1 of the five models for the official Representation B and E1–E3, plus the persistence rule |
| `viana_do_castelo_experiments.csv` | all metrics per experiment and model: accuracy, Macro-F1, precision, recall, confusion counts, prediction and update time per event |

Each YAML file starts with a comment that states exactly what was changed. Every experiment changes **one factor**
relative to the official Representation B (Handbook Ch. 12.10), passes `check_no_leakage`, and uses fresh model instances.

## Results (Viana do Castelo, Macro-F1, evaluation 2024–2025)

| Model | official B | E1 temperature | E2 windows 3/12/48 | E3 hour of day |
|---|---|---|---|---|
| Gaussian Naive Bayes | 0.8606 | 0.8548 | 0.8674 | 0.8671 |
| Online Logistic Regression | 0.8631 | 0.8811 | 0.8772 | 0.8731 |
| Hoeffding Tree | 0.9139 | 0.9130 | 0.9141 | 0.9134 |
| Online Bagging | 0.9155 | 0.9198 | 0.9195 | 0.9201 |
| Adaptive Random Forest | 0.9127 | 0.9123 | 0.9105 | 0.9137 |
| *persistence y_t-1* | *0.9140* | *0.9140* | *0.9140* | *0.9140* |

Macro-F1 is deterministic (seed 42) and matches notebook 02; timing columns depend on the machine and vary slightly between runs.

## How to reproduce

Run Section 11.2 of `notebooks/02_cityB_viana_do_castelo.ipynb`, or load a configuration directly:

```python
import yaml
from data import prepare_frames
from features import make_streams
from models import make_models
from evaluate import evaluate_models

my_cfg = yaml.safe_load(open("config_vika/E1_temperature_context.yaml"))
prepared = prepare_frames(my_cfg)                       # same data, warm-up and tau_H as the official run
streams = make_streams(prepared, my_cfg)["viana_do_castelo"]
metrics, _ = evaluate_models({"B": streams["B"]}, make_models, my_cfg, "viana_do_castelo")
# never call save_results with my_cfg: it would overwrite the official files
```

## If we want one of these in the official run

It is a joint decision (see the rules in the temporary note of notebooks 01 / 02): agree on it, change `config.yaml`
(or `src/features.py` for the hour of day), then re-run notebooks 00, 01, 02 and 03 for **both** cities.
