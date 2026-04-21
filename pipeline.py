import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def add_interactions(X):
    bmi_x_age = (X[:, 3] * X[:, 18]).reshape(-1, 1)
    highbp_x_highchol = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, bmi_x_age, highbp_x_highchol])


def run_pipeline():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_val = np.load("data/X_val.npy")
    y_val = np.load("data/y_val.npy")

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    lr_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=0.1
        ))
    ])

    lgbm_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbosity=-1
        ))
    ])

    gb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    xgb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ))
    ])

    cb_pipe = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("classifier", CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        ))
    ])

    model = VotingClassifier(
        estimators=[("lr", lr_pipe), ("lgbm", lgbm_pipe), ("gb", gb_pipe),
                    ("xgb", xgb_pipe), ("cb", cb_pipe)],
        voting='soft'
    )

    model.fit(X_all, y_all)
    return model


if __name__ == "__main__":
    model = run_pipeline()
    print("pipeline.py ran successfully.")
    print(f"Model type: {type(model).__name__}")
