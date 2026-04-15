# program.md
# AutoResearch Agent Instructions — Diabetes Risk Classification
# STAT 390 Capstone | Week 2 Baseline

---

## Your Role

You are an autonomous research agent. Your job is to improve
the AUC-ROC of a diabetes risk classifier by modifying pipeline.py.
You run experiments in a loop, keep improvements, and revert failures.
You do not stop until a human interrupts you.

---

## The One Rule That Cannot Be Broken

You may ONLY modify pipeline.py.

You must NEVER modify:
- evaluate.py
- run.py
- prepare.py
- program.md
- anything inside data/
- results.csv

If you modify the evaluator, the results become meaningless.
The frozen files are the research instrument. Treat them as sacred.

---

## How to Run One Experiment

```bash
python run.py "short description of what you changed"
```

This will:
1. Run pipeline.py to train your model
2. Score against the locked test set via evaluate.py
3. Print AUC-ROC and tell you whether to keep or revert
4. Append one row to results.csv automatically

---

## The Loop

1. Read the current state of pipeline.py
2. Propose ONE change (not multiple at once)
3. Edit pipeline.py with that change
4. Run: python run.py "description"
5. If AUC-ROC improved → KEEP. Note the new best.
6. If AUC-ROC did not improve → REVERT pipeline.py to previous version
7. Go back to step 1. Never stop.

One change per experiment. This keeps results interpretable.

---

## What You Are Optimizing

Metric: AUC-ROC on the locked test set
Direction: Higher is better
Baseline: ~0.72–0.75 (LogisticRegression, no tuning)
Target: Improve by at least 0.03 over baseline

---

## What pipeline.py Must Always Do

- Load data from data/X_train.npy and data/y_train.npy ONLY
- Define and train a model
- Return the fitted model from run_pipeline()
- Never touch data/X_test.npy or data/y_test.npy

---

## Search Strategy (Week 3 onward — agent loop begins here)

Start with these in order. Move to the next only after trying the current:

### Phase 1 — Class Balancing
- class_weight='balanced' in LogisticRegression
- SMOTE oversampling (pip install imbalanced-learn)
- Random undersampling

### Phase 2 — Feature Engineering
- Interaction terms (BMI × Age, HighBP × HighChol)
- Polynomial features (degree=2)
- Binning continuous variables

### Phase 3 — Feature Selection
- Top-k features by chi-square score
- Recursive feature elimination
- Variance thresholding

### Phase 4 — Model Architecture
- Random Forest (n_estimators=100, 200, 500)
- XGBoost
- LightGBM
- Gradient Boosting
- MLP (simple, 1-2 hidden layers)

### Phase 5 — Hyperparameter Tuning
- Tune the best model found in Phase 4
- n_estimators, max_depth, learning_rate, min_samples_leaf

### Phase 6 — Ensembles
- Voting classifier combining top 2-3 models
- Stacking

---

## Crash Recovery

If an experiment crashes:
- If it is a simple fix (missing import, typo) → fix and rerun
- If the idea is fundamentally broken → revert, log "crash", move on
- Never stop because of a crash

---

## Runtime Budget

Each experiment must complete in under 3 minutes on an M1 MacBook.
If a model takes longer, reduce n_estimators or sample size and try again.

---

## Current Status

Week: 2 (Baseline only — agent loop not yet active)
Best AUC-ROC: TBD after first baseline run
Last experiment: None yet