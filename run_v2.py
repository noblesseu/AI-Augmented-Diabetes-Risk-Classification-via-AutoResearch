# run_v2.py
# ---------------------------------------------------------------
# THIS FILE IS FROZEN. THE AGENT MUST NEVER MODIFY IT.
# Runs one experiment using the expanded data_v2/ feature set.
#
# Usage:
#   python run_v2.py "description of change"
# ---------------------------------------------------------------

import sys
import csv
import time
import os
from datetime import datetime

from pipeline_v2 import run_pipeline
from evaluate_v2 import evaluate

RESULTS_FILE = "results_v2.csv"


def get_next_exp_id():
    if not os.path.exists(RESULTS_FILE):
        return 1
    with open(RESULTS_FILE, "r") as f:
        rows = list(csv.DictReader(f))
    return 1 if not rows else int(rows[-1]["exp_id"]) + 1


def get_best_auc():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r") as f:
        kept = [float(r["auc_roc"]) for r in csv.DictReader(f) if r["status"] == "keep"]
    return max(kept) if kept else None


def log_result(exp_id, description, auc_roc, delta, runtime_sec, status):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        fieldnames = ["exp_id", "timestamp", "description",
                      "auc_roc", "delta", "runtime_sec", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "exp_id":      exp_id,
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "auc_roc":     f"{auc_roc:.6f}",
            "delta":       f"{delta:+.6f}" if delta is not None else "baseline",
            "runtime_sec": f"{runtime_sec:.1f}",
            "status":      status
        })


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python run_v2.py "description of change"')
        sys.exit(1)

    description = sys.argv[1]
    exp_id      = get_next_exp_id()
    best_so_far = get_best_auc()

    print(f"\n{'='*55}")
    print(f"  Experiment {exp_id:03d} [v2]: {description}")
    print(f"{'='*55}")

    print("  Training model...")
    t0    = time.time()
    model = run_pipeline()
    print(f"  Training complete ({time.time()-t0:.1f}s)")

    print("  Evaluating on locked test set (data_v2/)...")
    auc_roc = evaluate(model)
    runtime = time.time() - t0
    delta   = (auc_roc - best_so_far) if best_so_far is not None else None

    if best_so_far is None:
        status_hint = "baseline"
        decision    = "BASELINE ESTABLISHED"
    elif auc_roc > best_so_far:
        status_hint = "keep"
        decision    = f"IMPROVED (+{auc_roc - best_so_far:.4f}) — KEEP"
    else:
        status_hint = "revert"
        decision    = f"NO IMPROVEMENT ({auc_roc - best_so_far:+.4f}) — REVERT pipeline_v2.py"

    print(f"\n  AUC-ROC:     {auc_roc:.6f}")
    if best_so_far is not None:
        print(f"  Best so far: {best_so_far:.6f}")
        print(f"  Delta:       {auc_roc - best_so_far:+.6f}")
    print(f"  Runtime:     {runtime:.1f}s")
    print(f"\n  >>> {decision}")
    print(f"{'='*55}\n")

    log_result(exp_id, description, auc_roc, delta, runtime, status_hint)
    print(f"  Result logged to {RESULTS_FILE}")