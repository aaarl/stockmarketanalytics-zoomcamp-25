import pandas as pd
from pathlib import Path


def wide_to_long(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return df.melt(id_vars=[date_col], var_name="ticker", value_name="price")


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df["dow"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    return df


def run(cfg: dict):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])

    df = pd.read_csv(raw_dir / cfg["data"]["main_file"])
    df_long = wide_to_long(df, cfg["data"]["date_col"])
    df_long = add_calendar_features(df_long, cfg["data"]["date_col"])

    # Optional: merge meta for non-crypto tickers if available
    try:
        meta = pd.read_csv(raw_dir / cfg["data"]["meta_file"])
        meta = meta.rename(columns={"Symbol": "ticker"})
        df_long = df_long.merge(
            meta[["ticker", "Listing Exchange", "ETF"]], on="ticker", how="left"
        )
    except Exception as e:
        print(f"[step_02] Meta merge skipped: {e}")

    df_long.to_csv(interim_dir / "unified_long.csv", index=False)
    print("[step_02] Saved unified_long.csv")
