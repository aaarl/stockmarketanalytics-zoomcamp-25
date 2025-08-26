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

    df = add_basic_features(df, date_col=date_col)
    df = add_technical_features(df, date_col=date_col)

    # Define target (next-period log return)
    df["log_return_1d"] = (
        (
            df.groupby("ticker")["price"].apply(
                lambda s: (s.pct_change() + 1).apply(lambda x: 0 if pd.isna(x) else x)
            )
        )
        .groupby(df["ticker"])
        .apply(
            lambda s: (s.replace(0, 1)).pipe(
                lambda z: (z).apply(lambda v: 0 if v <= 0 else v)
            )
        )
        .values
    )
    df["log_return_1d"] = (
        df.groupby("ticker")["price"].apply(
            lambda s: (s / s.shift(1)).apply(lambda r: 0 if pd.isna(r) else r)
        )
    ).values
    df["log_return_1d"] = (
        (df["price"] / df.groupby("ticker")["price"].shift(1))
        .apply(lambda r: 0 if pd.isna(r) else r)
        .pipe(lambda r: (r).apply(lambda x: 0 if x <= 0 else x))
    )
    # Simplify target: next-day (t+1) log-return sign
    df["target_return_1d"] = df.groupby("ticker")["price"].pct_change().shift(-1)
    df["target_up"] = (df["target_return_1d"] > 0).astype(int)

    df.dropna(inplace=True)
    df.to_csv(processed_dir / "dataset_features.csv", index=False)
    print("[step_03] Saved dataset_features.csv")
