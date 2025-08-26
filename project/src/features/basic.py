import pandas as pd
import numpy as np

def _groupwise_pct_change(df: pd.DataFrame, by: str, col: str, periods: int = 1) -> pd.Series:
    return df.groupby(by)[col].pct_change(periods=periods)

def _groupwise_shift(df: pd.DataFrame, by: str, col: str, periods: int = 1) -> pd.Series:
    return df.groupby(by)[col].shift(periods)

def _rolling_groupby_apply(df, by, col, window, func):
    return df.groupby(by)[col].transform(lambda s: getattr(s.rolling(window), func)())

def add_basic_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Adds basic, leakage-free features computed per ticker:
    - returns, log returns
    - lagged returns (1,2,3,5,10)
    - rolling mean/std of returns (volatility)
    - momentum & rate of change
    - 52-week range proximity (using rolling min/max of price)
    - simple z-score of price against SMA20 (set in technical)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(["ticker", date_col])

    # 1) Returns
    df["ret_1d"] = _groupwise_pct_change(df, "ticker", "price", 1)
    # Safe log: add small epsilon to avoid log(0) if needed
    eps = 1e-12
    df["log_ret_1d"] = np.log((df["price"] + eps) / (_groupwise_shift(df, "ticker", "price", 1) + eps))


    # 2) Lagged returns
    for p in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{p}d"] = _groupwise_pct_change(df, "ticker", "price", p)

    # 3) Rolling stats of daily returns (volatility proxies)
    for w in [5, 10, 20]:
        df[f"ret_mean_{w}"] = df.groupby("ticker")["ret_1d"].transform(lambda s: s.rolling(w).mean())
        df[f"ret_std_{w}"]  = df.groupby("ticker")["ret_1d"].transform(lambda s: s.rolling(w).std())

    # 4) Momentum / ROC
    for p in [5, 10, 20]:
        df[f"mom_{p}d"] = df.groupby("ticker")["price"].transform(lambda s: s / s.shift(p) - 1)
        df[f"roc_{p}d"] = df.groupby("ticker")["price"].transform(lambda s: (s - s.shift(p)) / s.shift(p))

    # 5) 52-week range (252 trading days)
    w = 252
    df["roll_max_252"] = df.groupby("ticker")["price"].transform(lambda s: s.rolling(w).max())
    df["roll_min_252"] = df.groupby("ticker")["price"].transform(lambda s: s.rolling(w).min())
    df["pct_from_52w_high"] = (df["price"] / df["roll_max_252"]) - 1.0
    df["pct_from_52w_low"]  = (df["price"] / df["roll_min_252"]) - 1.0

    # Interaction example
    df["ret_x_mom10"] = df["ret_1d"] * df["mom_{p}d".format(p=10)]

    return df
