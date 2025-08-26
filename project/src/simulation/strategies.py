import pandas as pd
import numpy as np
from typing import Optional

def _apply_costs(position: pd.Series, fee_bps: float) -> pd.Series:
    """
    Approximate transaction costs: whenever position changes, subtract fee in that period.
    fee_bps is in basis points (e.g., 5 = 0.05%).
    """
    change = position.diff().abs().fillna(0)
    return change * (fee_bps / 1e4)

def _aggregate_daily(df: pd.DataFrame, ret_col: str = "ret_1d") -> pd.DataFrame:
    """Ensure daily aggregation by date across tickers exists."""
    daily = df.groupby("Date", as_index=False).agg({ret_col: "mean"})
    return daily

def sim_long_only_threshold(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float = 0.6,
    fee_bps: float = 5.0,
    max_concurrent: int = 3,
) -> pd.DataFrame:
    """
    Vectorized daily portfolio:
    - Each day, pick up to top-N tickers where prob >= threshold.
    - Equal-weight them for that day.
    - Apply costs when entering/exiting a ticker.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"])

    # Rank by prob per day, select top-N >= threshold
    df["rank_prob"] = df.groupby("Date")[prob_col].rank(ascending=False, method="first")
    df["selected"] = ((df[prob_col] >= threshold) & (df["rank_prob"] <= max_concurrent)).astype(int)

    # Position per row
    df["position"] = df["selected"]

    # Per-ticker transaction costs when position changes
    df["pos_change_cost"] = df.groupby("ticker")["position"].apply(lambda s: _apply_costs(s, fee_bps))

    # Strategy return per row: next-day return * position - costs
    # Use actual next-day return already provided as "target_return_1d"
    df["row_ret"] = df["position"] * df["target_return_1d"] - df["pos_change_cost"]

    # Aggregate equal-weighted per day
    daily = df.groupby("Date", as_index=False).agg(strategy_ret=("row_ret", "mean"))
    daily["equity"] = 10000 * (1 + daily["strategy_ret"].fillna(0)).cumprod()
    return daily

def sim_long_short_threshold(
    df: pd.DataFrame,
    prob_col: str,
    buy_thr: float = 0.6,
    short_thr: float = 0.6,
    fee_bps: float = 5.0,
    max_concurrent: int = 3,
) -> pd.DataFrame:
    """
    Long-Short daily portfolio:
    - Long if prob >= buy_thr (pick top-N)
    - Short if prob <= 1 - short_thr (pick bottom-N)
    - Equal-weight among chosen longs and shorts.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"])

    # Ranks for top and bottom
    df["rank_desc"] = df.groupby("Date")[prob_col].rank(ascending=False, method="first")
    df["rank_asc"]  = df.groupby("Date")[prob_col].rank(ascending=True, method="first")

    long_mask  = (df[prob_col] >= buy_thr) & (df["rank_desc"] <= max_concurrent)
    short_mask = (df[prob_col] <= (1 - short_thr)) & (df["rank_asc"]  <= max_concurrent)

    df["position"] = 0
    df.loc[long_mask,  "position"] = 1
    df.loc[short_mask, "position"] = -1

    df["pos_change_cost"] = df.groupby("ticker")["position"].apply(lambda s: _apply_costs(s, fee_bps))

    # If short, the return contribution is - target_return_1d (we profit if price goes down)
    df["row_ret"] = df["position"] * df["target_return_1d"] - df["pos_change_cost"]

    # Equal-weight per date across non-zero positions; if all zero, return 0
    def _daily_ret(group):
        pos = group["position"].abs()
        if pos.sum() == 0:
            return pd.Series({"strategy_ret": 0.0})
        weights = pos / pos.sum()
        ret = (weights * group["row_ret"]).sum()
        return pd.Series({"strategy_ret": ret})

    daily = df.groupby("Date").apply(_daily_ret).reset_index()
    daily["equity"] = 10000 * (1 + daily["strategy_ret"].fillna(0)).cumprod()
    return daily

def sim_long_only_sl_tp(
    df: pd.DataFrame,
    prob_col: str,
    threshold: float = 0.6,
    stop_loss: float = -0.03,
    take_profit: float = 0.06,
    fee_bps: float = 5.0,
    max_concurrent: int = 3,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """
    Exact (row-wise) simulation with capital reinvestment and basic position management.
    - Open positions at close when prob >= threshold (up to max_concurrent).
    - Equal capital split among open positions.
    - Exit on next close if SL/TP reached cumulatively since entry (approximation with daily data).
    - Apply transaction costs on entry/exit.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    dates = df["Date"].drop_duplicates().sort_values().tolist()
    tickers = df["ticker"].drop_duplicates().tolist()

    # State
    cash = initial_capital
    positions = {}  # ticker -> {"entry_price": float, "qty": float, "equity": float}
    equity_curve = []
    daily_returns = []

    fee = fee_bps / 1e4

    # Construct helper: next-day returns table
    # We'll iterate by date; for each ticker on a date we know target_return_1d for next day.
    for idx_date, d in enumerate(dates):
        day_df = df[df["Date"] == d].copy()
        # 1) Close existing positions based on SL/TP if triggered by cumulative return since entry.
        to_close = []
        for t, pos in positions.items():
            # Find today's row for ticker t to get today's return (applied at next close)
            row = day_df[day_df["ticker"] == t]
            if row.empty:
                continue
            # Use today's next-day return as realized for the *next* equity step
            ret_next = float(row["target_return_1d"].iloc[0])
            # Update equity of this position as if held one more day
            pos["equity"] *= (1 + ret_next)
            cum_ret_since_entry = pos["equity"] / (pos["qty"] * pos["entry_price"]) - 1.0

            # Check SL/TP at close approximation
            if cum_ret_since_entry <= stop_loss or cum_ret_since_entry >= take_profit:
                # close at today's close (apply exit fee)
                cash += pos["equity"] * (1 - fee)
                to_close.append(t)
        for t in to_close:
            positions.pop(t, None)

        # 2) Open new positions (sort by prob, threshold)
        open_candidates = (
            day_df.sort_values(prob_col, ascending=False)
                  .query(f"{prob_col} >= @threshold")
                  .ticker.tolist()
        )

        # Available slots
        slots = max(0, max_concurrent - len(positions))
        new_opens = open_candidates[:slots]

        if new_opens:
            alloc_per = cash / max(len(new_opens) + (len(positions) if len(positions)>0 else 0), 1)
            # If we already have positions, we *do not* rebalance them; we just split current free cash evenly
            alloc_per = cash / max(len(new_opens), 1)
            for t in new_opens:
                row = day_df[day_df["ticker"] == t]
                if row.empty or alloc_per <= 0:
                    continue
                price = float(row["price"].iloc[0])
                # apply entry fee
                invest = alloc_per * (1 - fee)
                qty = invest / price
                positions[t] = {"entry_price": price, "qty": qty, "equity": invest}
                cash -= alloc_per  # cash reduced by full allocation (fees included above)

        # 3) Compute total equity at end of day
        pos_equity = sum(p["equity"] for p in positions.values())
        total_equity = cash + pos_equity
        equity_curve.append((d, total_equity))

        # Daily return (vs previous day)
        if len(equity_curve) > 1:
            prev = equity_curve[-2][1]
            daily_returns.append((d, (total_equity / prev) - 1.0))
        else:
            daily_returns.append((d, 0.0))

    sim = pd.DataFrame(daily_returns, columns=["Date", "strategy_ret"]).sort_values("Date")
    sim["equity"] = [e for _, e in sorted(equity_curve, key=lambda x: x[0])]
    return sim
