# evaluate.py
# ---------------------------------------------------------------
# THIS FILE IS FROZEN. THE AGENT MUST NEVER MODIFY IT.
#
# This is the single source of truth for evaluation.
# It loads the locked test set, runs the trained model,
# and prints one number: AUC-ROC.
#
# The agent cannot game this file. The test set is loaded
# from data/ which was split once at project start and
# never regenerated.
# ---------------------------------------------------------------

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(model):
    """
    Score a fitted model against the locked test set.

    Args:
        model: a fitted sklearn-compatible estimator with predict_proba()

    Returns:
        auc_roc (float): AUC-ROC score on the locked test set
    """

    # Load the locked test set — split once, never changed
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")

    # Get predicted probabilities for the positive class
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute AUC-ROC
    auc_roc = roc_auc_score(y_test, y_prob)

    return auc_roc


if __name__ == "__main__":
    # This block only runs during direct testing, not during agent loop
    from pipeline import run_pipeline

    model = run_pipeline()
    auc = evaluate(model)
    print(f"AUC-ROC: {auc:.6f}")