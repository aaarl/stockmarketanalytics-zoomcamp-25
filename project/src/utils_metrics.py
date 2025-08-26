import numpy as np
import pandas as pd
from typing import Dict

def equity_curve_from_returns(returns: pd.Series, initial: float = 10000.0) -> pd.Series:
    """Compound returns to equity curve."""
    return initial * (1.0 + returns.fillna(0)).cumprod()

def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity / roll_max) - 1.0
    return drawdown.min()

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Assumes rf is per-period risk-free rate (set 0 for simplicity)."""
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=0) * np.sqrt(periods_per_year)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return (mu - rf) / sigma

def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    n_periods = len(equity)
    if n_periods == 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def compute_kpis(sim_df: pd.DataFrame, ret_col: str = "strategy_ret", ec_col: str = "equity") -> Dict[str, float]:
    """Compute common KPIs given a simulation dataframe with per-date returns & equity."""
    if ec_col not in sim_df.columns and ret_col in sim_df.columns:
        sim_df[ec_col] = equity_curve_from_returns(sim_df[ret_col])
    equity = sim_df[ec_col]
    returns = sim_df[ret_col]
    return {
        "CAGR": round(float(cagr(equity)), 6),
        "Sharpe": round(float(sharpe_ratio(returns)), 4),
        "MaxDrawdown": round(float(max_drawdown(equity)), 6),
        "TotalReturn": round(float(equity.iloc[-1] / equity.iloc[0] - 1.0), 6),
    }
