# evaluate_v2.py
# ---------------------------------------------------------------
# THIS FILE IS FROZEN. THE AGENT MUST NEVER MODIFY IT.
# Scores against the locked data_v2/ test set.
# ---------------------------------------------------------------

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(model):
    X_test = np.load("data_v2/X_test.npy")
    y_test = np.load("data_v2/y_test.npy")
    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


if __name__ == "__main__":
    from pipeline_v2 import run_pipeline
    model = run_pipeline()
    auc = evaluate(model)
    print(f"AUC-ROC (expanded features): {auc:.6f}")