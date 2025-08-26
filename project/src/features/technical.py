import pandas as pd
import numpy as np

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def _rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window).mean()
    loss = down.rolling(window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Adds technical indicators per ticker:
    - SMA/EMA: 5/10/20/50
    - MACD (12,26,9)
    - RSI(14)
    - Bollinger Bands (20, 2 std)
    - Z-Score vs SMA20
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(["ticker", date_col])

    for win in [5, 10, 20, 50]:
        df[f"sma_{win}"] = df.groupby("ticker")["price"].transform(lambda s: _sma(s, win))
        df[f"ema_{win}"] = df.groupby("ticker")["price"].transform(lambda s: _ema(s, win))

    # MACD (12, 26, 9)
    ema12 = df.groupby("ticker")["price"].transform(lambda s: _ema(s, 12))
    ema26 = df.groupby("ticker")["price"].transform(lambda s: _ema(s, 26))
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda s: _ema(s, 9))
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI(14)
    df["rsi_14"] = df.groupby("ticker")["price"].transform(lambda s: _rsi(s, 14))

    # Bollinger Bands (20, 2 std)
    sma20 = df["sma_20"]
    std20 = df.groupby("ticker")["price"].transform(lambda s: _std(s, 20))
    df["bb_upper_20"] = sma20 + 2 * std20
    df["bb_lower_20"] = sma20 - 2 * std20
    df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / (sma20.replace(0, np.nan))

    # Z-Score of price vs SMA20
    df["zscore_20"] = (df["price"] - sma20) / (std20.replace(0, np.nan))

    # Simple interaction example
    df["ret_1d_x_rsi_14"] = df.get("ret_1d", pd.Series(index=df.index)) * df["rsi_14"]

    return df
