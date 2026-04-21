import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def add_interactions(X):
    bmi_x_age = (X[:, 3] * X[:, 18]).reshape(-1, 1)
    highbp_x_highchol = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, bmi_x_age, highbp_x_highchol])


def run_pipeline():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    model = Pipeline([
        ("interactions", FunctionTransformer(add_interactions)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = run_pipeline()
    print("pipeline.py ran successfully.")
    print(f"Model type: {type(model.named_steps['classifier']).__name__}")
