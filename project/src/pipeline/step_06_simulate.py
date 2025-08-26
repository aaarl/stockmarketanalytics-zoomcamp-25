import pandas as pd
from pathlib import Path
from src.simulation.strategies import (
    sim_long_only_threshold,
    sim_long_short_threshold,
    sim_long_only_sl_tp,
)
from src.utils_metrics import compute_kpis


def run(cfg: dict):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    backtests_dir = Path(cfg["paths"]["backtests_dir"])

    df = pd.read_csv(processed_dir / "predictions.csv")
    df = df.sort_values(["ticker", "Date"])

    prob_col = (
        "pred_RandomForest"
        if "pred_RandomForest" in df.columns
        else "pred_DecisionTree"
    )
    out = []

    # Strategy 1: Long-only threshold
    s1 = sim_long_only_threshold(
        df.copy(),
        prob_col,
        threshold=cfg["simulation"]["thresholds"]["buy_prob"],
        fee_bps=cfg["simulation"]["costs"]["fee_bps"],
    )
    out.append(("long_only_threshold", compute_kpis(s1)))

    # Strategy 2: Long-Short threshold
    s2 = sim_long_short_threshold(
        df.copy(),
        prob_col,
        buy_thr=cfg["simulation"]["thresholds"]["buy_prob"],
        short_thr=cfg["simulation"]["thresholds"]["short_prob"],
        fee_bps=cfg["simulation"]["costs"]["fee_bps"],
    )
    out.append(("long_short_threshold", compute_kpis(s2)))

    # Strategy 3: Long-only with SL/TP
    s3 = sim_long_only_sl_tp(
        df.copy(),
        prob_col,
        threshold=cfg["simulation"]["thresholds"]["buy_prob"],
        stop_loss=cfg["simulation"]["risk"]["stop_loss"],
        take_profit=cfg["simulation"]["risk"]["take_profit"],
        fee_bps=cfg["simulation"]["costs"]["fee_bps"],
    )
    out.append(("long_only_sl_tp", compute_kpis(s3)))

    pd.DataFrame([{"strategy": k, **v} for k, v in out]).to_csv(
        backtests_dir / "summary.csv", index=False
    )
    print("[step_06] Saved backtests/summary.csv")
