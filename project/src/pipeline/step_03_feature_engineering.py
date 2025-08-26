import pandas as pd
from pathlib import Path
from src.features.basic import add_basic_features
from src.features.technical import add_technical_features

def run(cfg: dict):
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    date_col = cfg["data"]["date_col"]

    df = pd.read_csv(interim_dir / "unified_long.csv")
    df = df.sort_values(["ticker", date_col])

    # Feature engineering
    df = add_basic_features(df, date_col=date_col)
    df = add_technical_features(df, date_col=date_col)

    # Define target: next-day return of price (per ticker)
    df["target_return_1d"] = df.groupby("ticker")["price"].pct_change().shift(-1)
    df["target_up"] = (df["target_return_1d"] > 0).astype(int)

    # Select feature columns explicitly (exclude leakage & identifiers)
    exclude_cols = {
        date_col, "ticker", "price", "target_return_1d", "target_up",
        # rolling extrema can have long warmups; keep but we'll drop NaNs
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "O"]

    # Drop rows with any NaNs in used features or target
    df = df.dropna(subset=feature_cols + ["target_return_1d", "target_up"]).reset_index(drop=True)

    df.to_csv(processed_dir / "dataset_features.csv", index=False)
    print(f"[step_03] Saved dataset_features.csv with {len(feature_cols)} feature columns.")
