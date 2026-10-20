# Practical Assignment I: Business-Oriented Data Stream Mining
**NOVA IMS — Data Stream Mining**

---

## 1. Audit Metadata & Project Overview
* **Assignment Code:** PA04
* **Group Code:** GroupYY
* **Student A Name (Primary City A):** [Student A Name] (City A: Guimarães)
* **Student B Name (Primary City B):** [Student B Name] (City B: Viana do Castelo)
* **Business Scenario:** Textile Storage Facility
* **City Pair:** Guimarães vs. Viana do Castelo
* **Official Target:** T4 — Humid vs. Dry Regime (\\(\tau_H = Q_{0.50}(H)\\) estimated from pooled warm-up and frozen)
* **Business Question:** How do humid and dry regimes differ between candidate warehouse locations?
* **Historical Data Period:** [YYYY-MM-DD] to [YYYY-MM-DD] (Open-Meteo hourly stream)
* **Random Seed:** 42

---

## 2. Environment & System Requirements
* **Python Version:** Python 3.10+
* **Package Versions:**
  * `river` >= 0.18.0
  * `pandas` >= 2.0.0
  * `numpy` >= 1.24.0
  * `scikit-learn` >= 1.2.0
  * `matplotlib` >= 3.7.0
  * `seaborn` >= 0.12.0

### Environment Setup (`uv`) 
```bash 
# Create virtual environment (.venv) 
uv venv 

# Activate environment source 
.venv/bin/activate # Windows: .venv\\Scripts\\activate 

# Install from requirements.txt 
uv pip install -r requirements.txt
```

---

## 3. Directory Layout

```text
PA04_GroupYY/
├── README.md                         # Audit metadata & replication guide
├── config.yaml
├── requirements.txt                  # Locked dependencies
├── src/
│   └── practical_assignment.ipynb    # Main prequential evaluation pipeline
│   └── collect_data.py               # Collect raw snapshots
│   └── utils.py                      # Supporting modules
├── data/
│   ├── guimaraes_raw.json            # Raw weather stream (Guimarães)
│   ├── viana_do_castelo_raw.json     # Raw weather stream (Viana do Castelo)
└── results/
    ├── metrics.csv                   # Accuracy, Macro-F1, Precision, Recall, Latency
    └── figures/                      # Exported plots
```

## 4. Instructions to Reproduce the Main Experiment

### Step 1: Extract Audit Directory
```bash
unzip PA04_GroupYY_Audit.zip
cd PA04_GroupYY/
```

### Step 2: Set Up Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Run the Evaluation Pipeline
Execute the main script to run the complete data loading, pooled warm-up threshold estimation, feature construction, prequential evaluation loop, and result export:

```bash
python src/practical_assignment.py
```

---

## 5. Experimental Protocol Summary
1. **Threshold Policy:** Threshold \\(\tau_H\\) is computed from the pooled warm-up observations of both cities combined and strictly frozen before evaluation.
2. **Target Leakage Control:** Predictors in Representation A (\\(t-1\\)) and Representation B (\\(t-1\\) with 6h/24h rolling context) strictly precede observation time \\(t\\).
3. **Prequential Evaluation Protocol:** Strict chronological sequence following **PREDICT \\(\rightarrow\\) EVALUATE \\(\rightarrow\\) LEARN** per event.
4. **Mandatory Models:** Gaussian Naive Bayes, Online Logistic Regression (SGD), Hoeffding Tree, Online Bagging, and Adaptive Random Forest evaluated under identical chronological histories.