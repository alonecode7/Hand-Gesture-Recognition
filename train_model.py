import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Train gesture classifier from CSV")
    parser.add_argument("--data", default="gestures.csv", help="Input CSV path")
    parser.add_argument("--output", default="gesture_model.pkl", help="Output model path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column")

    X = df.drop(columns=["label"]).astype(np.float32).values
    y = df["label"].astype(str).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    labels = sorted(set(y))
    payload = {"model": clf, "labels": labels}
    joblib.dump(payload, args.output)
    print(f"Saved model: {args.output}")


if __name__ == "__main__":
    main()
