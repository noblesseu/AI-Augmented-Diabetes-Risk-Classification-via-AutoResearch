# pipeline_v2.py
# ---------------------------------------------------------------
# THE ONLY FILE THE AGENT IS ALLOWED TO MODIFY (v2 loop).
# ---------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, C=0.1))
    ])


def run_pipeline():
    X_train = np.load("data_v2/X_train.npy")
    y_train = np.load("data_v2/y_train.npy")
    X_val = np.load("data_v2/X_val.npy")
    y_val = np.load("data_v2/y_val.npy")

    # --- Val-set evaluation for model selection (not test set) ---
    val_model = _build_model()
    val_model.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, val_model.predict_proba(X_val)[:, 1])
    print(f"  *** VAL AUC (use for model selection): {val_auc:.6f} ***")

    # --- Phase A: 5-fold CV diagnostic ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        _build_model(), X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- Retrain on full train+val for submission ---
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    final_model = _build_model()
    final_model.fit(X_all, y_all)
    return final_model


if __name__ == "__main__":
    model = run_pipeline()
    print("pipeline_v2.py ran successfully.")
