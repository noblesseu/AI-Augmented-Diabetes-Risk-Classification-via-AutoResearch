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
| **Rows / Features** | 253,680 rows · 35 features |
| **Target** | Binary — diabetes/pre-diabetes (1) vs. healthy (0) |
| **Metric** | AUC-ROC on locked test set |
| **Baseline AUC-ROC** | TBD — run `python run.py "baseline"` |
| **Best AUC-ROC** | TBD |
| **Experiment log** | [results.csv](results.csv) |
| **Week** | 2 — Baseline & Evaluation Freeze |

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
pip install ucimlrepo scikit-learn numpy
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

Expected output:

```
=======================================================
  Experiment 001: baseline: LogisticRegression + StandardScaler
=======================================================
  Training model...
  Training complete (~Xs)
  Evaluating on locked test set...

  AUC-ROC:     0.XXXXXX
  Runtime:     X.Xs

  >>> BASELINE ESTABLISHED
=======================================================

  Result logged to results.csv
```

### Step 5 — Verify the experiment log

```bash
cat results.csv
```

You should see one row with exp_id=1, the AUC-ROC score, and status=baseline.

---

## The AutoResearch Loop (Week 3 onward)

```bash
# Launch Claude Code in the project directory
claude

# Then paste this prompt:
# "Read program.md for your instructions, then read pipeline.py.
#  Run python run.py "baseline" to confirm the baseline.
#  Then enter the AutoResearch loop."
```

The agent will modify `pipeline.py`, run experiments, and log results
to `results.csv` autonomously. The human monitors progress and can
interrupt at any time.

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

See [research_log.md](research_log.md) for a dated record of decisions,
observations, and next steps.

## Failure Log

See [failure_log.md](failure_log.md) for crashed experiments and lessons learned.