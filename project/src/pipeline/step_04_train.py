import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib


def run(cfg: dict):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / "dataset_features.csv")

    feature_cols = [
        c
        for c in df.columns
        if c not in ["Date", "ticker", "price", "target_return_1d", "target_up"]
    ]
    X = df[feature_cols]
    y = df["target_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = auc
        joblib.dump(m, processed_dir / f"{name}.joblib")
        print(f"[step_04] {name} ROC-AUC = {auc:.3f}")

    # (Optional) add XGBoost later
