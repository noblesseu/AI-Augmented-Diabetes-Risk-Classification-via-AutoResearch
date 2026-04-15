import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run_pipeline():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    model = run_pipeline()
    print("pipeline.py ran successfully.")
    print(f"Model type: {type(model.named_steps['classifier']).__name__}")