# AI-Augmented-Diabetes-Risk-Classification-via-AutoResearch
AutoResearch agent loop for diabetes risk classification
# Diabetes Risk Classification — AutoResearch Capstone
**STAT 390 | Spring 2026**

---

## One-Sentence Summary

An AI coding agent iteratively modifies a classification pipeline to maximize AUC-ROC for predicting diabetes/pre-diabetes from CDC behavioral health survey data, using the Karpathy AutoResearch framework.

---

## Current Status

| Item | Value |
|------|-------|
| **Dataset** | CDC Diabetes Health Indicators (UCI ID 891) |
| **Rows / Features** | 253,680 rows · 21 features |
| **Target** | Binary — diabetes/pre-diabetes (1) vs. healthy (0) |
| **Metric** | AUC-ROC on locked test set |
| **Baseline AUC-ROC** | 0.8201 |
| **Best AUC-ROC** | 0.8296 (exp 65) |
| **Improvement** | +0.0095 over baseline |
| **Total Experiments** | 54 |
| **Experiment log** | [results.csv](results.csv) |
| **Status** | Complete |

---

## Best Pipeline

A 5-way soft-voting ensemble trained on the combined train+val set, with manual interaction features injected before each sub-model:

| Member | Config |
|--------|--------|
| **LogisticRegression** | degree-2 PolynomialFeatures, StandardScaler, C=0.1, class_weight='balanced' |
| **LGBMClassifier** | n_estimators=300, lr=0.05, num_leaves=63 |
| **GradientBoostingClassifier** | n_estimators=100 |
| **XGBClassifier** | n_estimators=500, lr=0.03, subsample=0.8, colsample_bytree=0.8 |
| **CatBoostClassifier** | iterations=500, lr=0.03, depth=8, auto_class_weights='Balanced' |

Manual interaction features appended to every sub-model input: **BMI × Age** and **HighBP × HighChol**.

---

## Research Journey

| Phase | Key Discovery | Best AUC |
|-------|---------------|----------|
| Baseline | LogisticRegression + StandardScaler | 0.8201 |
| Class balancing | `class_weight='balanced'` | 0.8207 |
| Feature engineering | BMI×Age, HighBP×HighChol interaction terms | 0.8219 |
| Feature engineering | Degree-2 polynomial features (all pairs + squared) | 0.8281 |
| Training strategy | Train on combined train+val | 0.8280 |
| Model architecture | 2-way → 5-way soft VotingClassifier | 0.8294 |
| Hyperparameter tuning | CatBoost depth=8, XGB stochastic subsample | 0.8296 |

**Approaches that did not help:** SMOTE, random undersampling, ExtraTrees, HistGradientBoosting, stacking, SelectKBest feature selection, target encoding, RobustScaler/QuantileTransformer/SplineTransformer/Nystroem kernel approximation, LGBM dart boosting, optimal blend weights via Nelder-Mead, BMI clinical bins, additional interaction features (GenHlth×BMI, GenHlth×Age).

See [failure_log.md](failure_log.md) for the full list of reverted experiments.

---

## Project Structure

```
diabetes-autoresearch/
├── prepare.py          # FROZEN — downloads data, creates locked split
├── pipeline.py         # AGENT EDITS — all modeling logic lives here
├── evaluate.py         # FROZEN — scores model against locked test set
├── run.py              # FROZEN — orchestrates one experiment, logs result
├── program.md          # HUMAN EDITS — agent instruction file
├── results.csv         # AUTO-GENERATED — experiment log
├── failure_log.md      # AUTO-GENERATED — reverted experiments
├── data/               # FROZEN after prepare.py runs
│   ├── X_train.npy     # 60% of data — agent trains on this
│   ├── y_train.npy
│   ├── X_val.npy       # 20% — available for internal validation
│   ├── y_val.npy
│   ├── X_test.npy      # 20% — LOCKED, only opened at final evaluation
│   └── y_test.npy
├── notebooks/          # EDA and analysis (not part of agent loop)
└── logs/               # Per-experiment run logs
```

---

## How to Reproduce (TA Instructions)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-autoresearch.git
cd diabetes-autoresearch
```

### Step 2 — Install dependencies

```bash
pip install ucimlrepo scikit-learn numpy lightgbm xgboost catboost
```

### Step 3 — Prepare the data (run once)

```bash
python prepare.py
```

This downloads the dataset from UCI, creates a deterministic
60/20/20 stratified split, and saves all splits to `data/`.
Expected output:

```
Dataset loaded: 253,680 rows, 35 features
Target distribution: 13.9% positive (diabetes/pre-diabetes)

Split summary:
  Train: 152,208 rows (60%)
  Val:    50,736 rows (20%)
  Test:   50,736 rows (20%)

Setup complete.
```

### Step 4 — Run the baseline experiment

```bash
python run.py "baseline: LogisticRegression + StandardScaler"
```

### Step 5 — Run the best pipeline

```bash
python run.py "5-way ensemble XGB n_estimators=500 lr=0.03 more trees"
```

Expected AUC-ROC: **0.8296**

### Step 6 — Verify the experiment log

```bash
cat results.csv
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| What the agent modifies | `pipeline.py` only | Single editable file keeps diffs clean |
| What is frozen | `evaluate.py`, `run.py`, `prepare.py`, `data/` | Agent cannot game the evaluator |
| Metric | AUC-ROC | Robust to class imbalance (~14% positive rate) |
| Split strategy | 60/20/20 stratified, random_state=42 | Deterministic and reproducible |
| Test set policy | Opened once at final evaluation only | Prevents leakage during search |
| Runtime budget | < 3 minutes per experiment on M1 | Enables fast agent loop |

---

## Research Log

See [failure_log.md](failure_log.md) for a complete record of all reverted experiments.
