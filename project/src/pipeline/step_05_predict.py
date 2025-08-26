import pandas as pd
from pathlib import Path
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
    preds = {}

    for model_name in ["DecisionTree", "RandomForest"]:
        model_path = processed_dir / f"{model_name}.joblib"
        if not model_path.exists():
            print(f"[step_05] Missing model {model_name}, skipping.")
            continue
        model = joblib.load(model_path)
        proba = model.predict_proba(X)[:, 1]
        df[f"pred_{model_name}"] = proba
        preds[model_name] = proba.mean()

    df.to_csv(processed_dir / "predictions.csv", index=False)
    print("[step_05] Saved predictions.csv")
