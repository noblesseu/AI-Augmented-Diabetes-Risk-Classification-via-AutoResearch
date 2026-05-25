# pipeline_v2.py
# ---------------------------------------------------------------
# THE ONLY FILE THE AGENT IS ALLOWED TO MODIFY (v2 loop).
# ---------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# v2 feature indices: 0=GENHLTH, 8=BPHIGH4, 11=TOLDHI2, 32=_BMI5, 37=_AGEG5YR, 42=PA1MIN_, 45=CIMEMLOS, 50=DECIDE, 53=_RFBING5, 54=FLUSHOT6
# v5 additions (indices 55-58): 55=HIVTST6, 56=CHILDREN, 57=NUMADULT, 58=LMTJOIN3
# v6 additions (indices 59-62): 59=SMOKDAY2, 60=PNEUVAC3, 61=_PASTRNG, 62=DRNK3GE5
# v7 additions (indices 63-65): 63=DIFFDRES, 64=STRENGTH, 65=BLIND
# Note: binary vars recoded from 1=Yes/2=No to 1=Yes/0=No as of v5 dataset
def add_interactions(X):
    bmi_x_age = (X[:, 32] * X[:, 37]).reshape(-1, 1)
    highbp_x_highchol = (X[:, 8] * X[:, 11]).reshape(-1, 1)
    genhlth_x_age = (X[:, 0] * X[:, 37]).reshape(-1, 1)
    pa_x_bmi = (X[:, 42] * X[:, 32]).reshape(-1, 1)
    cimemlos_x_age = (X[:, 45] * X[:, 37]).reshape(-1, 1)
    decide_x_age = (X[:, 50] * X[:, 37]).reshape(-1, 1)
    rfbing_x_bmi = (X[:, 53] * X[:, 32]).reshape(-1, 1)
    flushot_x_age = (X[:, 54] * X[:, 37]).reshape(-1, 1)
    pastrng_x_bmi = (X[:, 61] * X[:, 32]).reshape(-1, 1)
    return np.hstack([X, bmi_x_age, highbp_x_highchol, genhlth_x_age, pa_x_bmi, cimemlos_x_age, decide_x_age, rfbing_x_bmi, flushot_x_age, pastrng_x_bmi])


def _build_ensemble():
    lr_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ("vt", VarianceThreshold(threshold=0.01)),
        ("classifier", LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced', C=0.1
        ))
    ])
    lgbm_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        ))
    ])
    gb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.03, random_state=42
        ))
    ])
    xgb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', verbosity=0
        ))
    ])
    cb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=6,
            auto_class_weights='Balanced', random_seed=42, verbose=0
        ))
    ])
    mlp_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu',
            alpha=0.001, max_iter=500, early_stopping=True, random_state=42
        ))
    ])
    return VotingClassifier(
        estimators=[("lr", lr_pipe), ("lgbm", lgbm_pipe),
                    ("gb", gb_pipe), ("xgb", xgb_pipe),
                    ("cb", cb_pipe), ("mlp", mlp_pipe)],
        voting='soft'
    )


def run_pipeline():
    X_train = np.load("data_v2/X_train.npy")
    y_train = np.load("data_v2/y_train.npy")
    X_val = np.load("data_v2/X_val.npy")
    y_val = np.load("data_v2/y_val.npy")

    # --- Val-set evaluation for model selection (not test set) ---
    val_model = _build_ensemble()
    val_model.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, val_model.predict_proba(X_val)[:, 1])
    print(f"  *** VAL AUC (use for model selection): {val_auc:.6f} ***")

    # --- Retrain on full train+val for submission ---
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    final_model = _build_ensemble()
    final_model.fit(X_all, y_all)
    return final_model


if __name__ == "__main__":
    model = run_pipeline()
    print("pipeline_v2.py ran successfully.")
