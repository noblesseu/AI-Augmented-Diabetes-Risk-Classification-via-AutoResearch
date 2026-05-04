# AutoResearch Agent Instructions
# Diabetes Risk Classification — STAT 390 Capstone
# Week 4: Controlled Experiments with Direction

---

## Objective

Maximize **AUC-ROC** on the locked test set.

Baseline: **0.8201** (LogisticRegression + StandardScaler)
Current best: **0.8296** (5-way ensemble, exp 51)
New target: **0.8400+**

---

## What We Know From 54 Experiments — Read This First

These are facts from the experiment log. Use them to guide decisions:

- Polynomial interaction features (degree=2) gave the biggest single jump: +0.0058
- class_weight='balanced' helped slightly: +0.0006
- SMOTE badly hurt performance: -0.0416. Do NOT try SMOTE again.
- Individual tree models (RF, XGB, LGBM, GB alone) were all weaker than the ensemble
- The 5-way ensemble plateaued at 0.8296 after exp 23. Adding more members did not help.
- Feature selection (SelectKBest top-150) did not help on the ensemble
- Cross-validation was NEVER used. This is a gap.
- Threshold optimization was NEVER tried. This is a gap.
- Runtime violations happened with LR+L1+liblinear. Avoid that combination.

---

## Non-Negotiable Rules

1. You may **ONLY** modify `pipeline.py`
2. `prepare.py`, `evaluate.py`, `run.py` are **FROZEN** — do not touch them
3. `run_pipeline()` must return a fitted sklearn-compatible estimator
4. Do **NOT** load `data/X_test.npy` or `data/y_test.npy` — ever
5. Each experiment must complete in **under 3 minutes** on M1 MacBook
6. Do **NOT** ask clarifying questions — read these instructions and act
7. Do **NOT** wait for confirmation between experiments — keep going
8. Do **NOT** add new files
9. Do **NOT** run the baseline again — it is already established at 0.8201
10. Do **NOT** repeat any experiment already in results.csv — read it first
11. If 3 consecutive experiments in the same Phase revert → move to next Phase immediately

---

## Workflow — Follow This Exactly

```
1. Read results.csv to understand what has already been tried
2. Read current pipeline.py
3. Choose the next experiment from the Phase you are currently in
4. Propose ONE focused change with a clear hypothesis
5. Edit pipeline.py
6. Run: python run.py "Phase_X: short description of change"
7. Check AUC-ROC in output
8. If AUC-ROC improved:
      git add pipeline.py && git commit -m "exp: <description> AUC=X.XXXX"
      Update README.md Best AUC-ROC line with new score
9. If AUC-ROC did not improve:
      git checkout pipeline.py
      Append one line to failure_log.md:
      "- exp_XX | <description> | AUC=X.XXXX | did not improve over X.XXXX"
10. If experiment crashed:
      git checkout pipeline.py
      Append to failure_log.md:
      "- exp_XX | <description> | CRASH | reason: <brief explanation>"
11. Repeat from step 3 — never stop until interrupted
```

---

## Controlled Search Strategy — Work Through Phases in Order

This is not a random search. Each phase has a clear hypothesis.
Run each idea, observe the result, and use it to inform the next decision.

---

### Phase A — Cross-Validation on Best Pipeline (Start Here)

**Hypothesis:** The current best (5-way ensemble) was never validated with CV.
We do not know if 0.8296 is stable or a lucky split result.

**Instructions:**
- Take the current best 5-way ensemble pipeline
- Add StratifiedKFold(n_splits=5) cross-validation on the training set
- Report mean AUC-ROC and std across folds
- If std > 0.005, the model is unstable — note this and move on
- If mean CV AUC > 0.829, the model is genuinely strong — continue tuning it
- If mean CV AUC < 0.829, the single-split result was misleading — try simpler models

**Experiments to run:**
1. 5-fold CV on current best ensemble — report mean ± std AUC
2. 5-fold CV on LR-poly alone (exp 16 architecture) — compare stability

---

### Phase B — Class Imbalance (Seriously This Time)

**Hypothesis:** Only 2 out of 54 experiments addressed class imbalance.
The 13.9% positive rate may be suppressing recall for the minority class.
AUC-ROC is imbalance-robust but the model may still be biased.

**Do NOT use SMOTE — it failed badly in exp 4 (-0.0416).**

**Instructions:**
- Work through these one at a time, in order:
1. class_weight='balanced' on ALL ensemble members simultaneously
   (exp 3 only applied it to LR — apply it to LGBM, XGB, GB, CatBoost too)
2. Threshold optimization: find the decision threshold on the val set
   that maximizes F1, then apply it — does AUC-ROC improve?
3. ADASYN oversampling (different from SMOTE, gentler)
   from imblearn.over_sampling import ADASYN
4. Tomek Links undersampling (removes borderline majority samples)
   from imblearn.under_sampling import TomekLinks
5. BalancedBaggingClassifier wrapping the best base model
   from imblearn.ensemble import BalancedBaggingClassifier

---

### Phase C — Feature Engineering with Insight

**Hypothesis:** The polynomial features in exp 7 gave the biggest jump (+0.0058).
We have not explored domain-specific feature combinations based on
what actually predicts diabetes clinically.

**Clinically meaningful interactions to try (one at a time):**
1. BMI_category: bin BMI into Underweight/Normal/Overweight/Obese (4 bins)
   Use clinical thresholds: <18.5, 18.5-25, 25-30, 30+
2. MetabolicRisk score: BMI_high * HighBP * HighChol (all three risk factors)
3. AgeRisk: Age * BMI (older + heavier = higher risk)
4. LifestyleScore: PhysActivity + Fruits + Veggies - HvyAlcoholConsump
5. HealthAccess: AnyHealthcare * NoDocbcCost (has care but can't afford it)
6. ComorbidityCount: sum of HighBP + HighChol + Stroke + HeartDiseaseorAttack
   (total number of comorbidities as a single feature)

**For each feature:**
- Add it to the existing best pipeline
- Run the experiment
- If it helps, keep it and add the next one
- If it does not help, revert and try the next feature independently

---

### Phase D — Feature Selection on the Best Pipeline

**Hypothesis:** With polynomial features, the model may have too many noisy
features. Selecting the most informative ones may reduce overfitting.

**Try these in order:**
1. SelectKBest with chi2, k=50 (fewer than exp 30 which tried 150)
2. SelectKBest with mutual_info_classif, k=75
3. Recursive Feature Elimination (RFE) with LogisticRegression, n_features=30
4. VarianceThreshold(threshold=0.01) to remove near-constant features
5. L1-based feature selection using SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear'))
   WARNING: This was slow in exp 33-34. Set max_iter=200 and use a sample if needed.

---

### Phase E — Neural Networks (Only If Phases A-D Are Exhausted)

**Only enter this phase if:**
- You have run at least 5 experiments in each of Phases A-D
- None of them improved AUC beyond 0.8310
- You have documented why each phase failed

**Allowed neural network approaches (CPU-friendly only):**
1. MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
   max_iter=500, early_stopping=True, random_state=42)
2. MLPClassifier(hidden_layer_sizes=(256, 128), dropout not available in sklearn
   — use alpha regularization instead, alpha=0.001)
3. TabNet — only if runtime < 3 minutes
   pip install pytorch-tabnet

**NOT allowed:**
- Any model requiring a GPU
- Any model taking > 3 minutes
- Deep learning frameworks (PyTorch, TensorFlow) from scratch

---

## What a Controlled Experiment Looks Like

Every experiment must have:
- A clear Phase label in the description: "Phase_B: class_weight balanced on all members"
- A hypothesis before running: what do you expect and why
- A conclusion after running: what happened and what does it mean

BAD:  "trying random forest again"
BAD:  "adding more trees to XGB"
GOOD: "Phase_B: apply class_weight=balanced to all 5 ensemble members — hypothesis: individual members may be biased toward majority class"

---

## Stuck Detection

If 3 consecutive experiments in the same Phase all revert:
- Stop that Phase
- Write one line in failure_log.md: "Phase_X exhausted after 3 reverts"
- Move immediately to the next Phase
- Do NOT try more variations of the same failing idea

---

## Runtime Budget

- Under 60 seconds: ideal
- 60–180 seconds: acceptable
- Over 180 seconds: abort, revert, try a lighter version
- LR+L1+liblinear on polynomial features: known to be very slow — avoid

---

## failure_log.md Format

```
- exp_XX | Phase_X: <description> | AUC=X.XXXX | did not improve over X.XXXX
- exp_XX | Phase_X: <description> | CRASH | reason: <brief explanation>
- Phase_X exhausted after 3 consecutive reverts
```

---

## Current State

Week: 4 — Controlled experiments with direction
Baseline AUC-ROC: 0.8201
Current best AUC-ROC: 0.8296 (exp 51: 5-way ensemble XGB n=500)
Total experiments run: 54
Kept: 16 | Reverted: 35 | Baseline reruns: 3
Next action: Start Phase A — cross-validate the current best pipeline