# prepare.py
# ---------------------------------------------------------------
# THIS FILE IS FROZEN. THE AGENT MUST NEVER MODIFY IT.
#
# Run this ONCE at project setup to:
#   1. Download the CDC Diabetes Health Indicators dataset
#   2. Create a deterministic, stratified train/val/test split
#   3. Save all splits to data/ as .npy files
#
# After running this file, the data/ folder is locked.
# The test set (X_test.npy, y_test.npy) must never be
# touched during the agent search phase.
#
# Usage (run once only):
#   python prepare.py
# ---------------------------------------------------------------

import os
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# Install ucimlrepo if needed:
#   pip install ucimlrepo
# ---------------------------------------------------------------
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    raise ImportError(
        "ucimlrepo not installed. Run: pip install ucimlrepo"
    )


def prepare_data():
    print("=" * 55)
    print("  prepare.py — one-time data setup")
    print("=" * 55)

    # --- Create data/ directory ---
    os.makedirs("data", exist_ok=True)

    # --- Check if already prepared ---
    required = ["X_train.npy", "y_train.npy",
                "X_val.npy",   "y_val.npy",
                "X_test.npy",  "y_test.npy"]
    if all(os.path.exists(f"data/{f}") for f in required):
        print("\n  data/ already exists with all splits.")
        print("  Delete data/ and re-run if you need to reset.")
        print("  WARNING: resetting the split invalidates all")
        print("  previous experiment results.\n")
        return

    # --- Download dataset ---
    print("\n  Fetching CDC Diabetes Health Indicators (UCI ID 891)...")
    dataset = fetch_ucirepo(id=891)
    X = dataset.data.features.values  # numpy array, shape (253680, 35)
    y = dataset.data.targets.values.ravel()  # shape (253680,)

    print(f"  Dataset loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"  Target distribution: {y.mean():.1%} positive (diabetes/pre-diabetes)")

    # --- Deterministic stratified split ---
    # 60% train | 20% val | 20% test
    # Stratified to preserve class imbalance in all splits
    # random_state=42 ensures reproducibility
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        stratify=y,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    print(f"\n  Split summary:")
    print(f"    Train: {X_train.shape[0]:>6,} rows ({X_train.shape[0]/X.shape[0]:.0%})")
    print(f"    Val:   {X_val.shape[0]:>6,} rows ({X_val.shape[0]/X.shape[0]:.0%})")
    print(f"    Test:  {X_test.shape[0]:>6,} rows ({X_test.shape[0]/X.shape[0]:.0%})")
    print(f"\n  Class balance check:")
    print(f"    Train positive rate: {y_train.mean():.3%}")
    print(f"    Val   positive rate: {y_val.mean():.3%}")
    print(f"    Test  positive rate: {y_test.mean():.3%}")

    # --- Save to disk ---
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy",   X_val)
    np.save("data/y_val.npy",   y_val)
    np.save("data/X_test.npy",  X_test)
    np.save("data/y_test.npy",  y_test)

    print(f"\n  All splits saved to data/")
    print(f"  TEST SET IS NOW LOCKED. Do not use X_test.npy")
    print(f"  during the agent search phase.")
    print(f"\n  Setup complete. You can now run:")
    print(f"    python run.py \"baseline: LogisticRegression\"")
    print("=" * 55)


if __name__ == "__main__":
    prepare_data()
    