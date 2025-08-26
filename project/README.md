# Stock-Market-Analytics — End-to-End ML Algo Trading

**Goal:** Build an end-to-end ML project that predicts next-day returns and simulates trading strategies across multiple tickers (incl. crypto).  
**Deliverables:** reproducible pipeline (scripts), notebooks, backtests, automation (Makefile/Docker), and clear documentation.

## 1) Problem description
We predict **next-day direction** (`target_up`) and **next-day return** (`target_return_1d`) per ticker, based on price-derived features and calendar context.  
We then **translate probabilities into trades** and simulate multiple strategies with realistic costs and risk controls.  
This setup is reproducible and automated (Makefile, Docker, optional cron), designed for peer review.

**Novelty:** mixed universe (US equities + crypto) and multiple holding/selection rules (long-only, long–short, SL/TP “exact” sim). This differs from the standard “top-cap week-long” setup.

## 2) Data sources
- **Primary:** Kaggle-style wide price table `portfolio_data.csv` (not committed to Git).  
- **Meta (optional):** `symbols_valid_meta.csv` for exchange/ETF flags.
- **Why it qualifies:** Alternative to yfinance/FRED (as requested). Can be extended with other APIs (e.g. CryptoCompare/Tiingo) without changing the pipeline.

> **Do not push data to GitHub.** Place under `data/raw/` locally or fetch via API (see “Two regimes” below).

## 3) Features (20+)
From `src/features/basic.py` and `src/features/technical.py`:
- **Returns:** `ret_1d`, `log_ret_1d`, lags `ret_lag_{1,2,3,5,10}`
- **Rolling stats:** `ret_mean_{5,10,20}`, `ret_std_{5,10,20}`
- **Momentum/ROC:** `mom_{5,10,20}`, `roc_{5,10,20}`
- **52-week range:** `roll_max_252`, `roll_min_252`, `pct_from_52w_high`, `pct_from_52w_low`
- **SMA/EMA:** `sma_{5,10,20,50}`, `ema_{5,10,20,50}`
- **MACD:** `macd`, `macd_signal`, `macd_hist`
- **RSI/BB:** `rsi_14`, `bb_upper_20`, `bb_lower_20`, `bb_width_20`
- **Z-Score:** `zscore_20`
- **Calendar:** `dow`, `month`
- **Interaction:** `ret_1d_x_mom10`, `ret_1d_x_rsi_14`

> Feature engineering is strictly **grouped by ticker** and **leakage-free** (shifts/rolling windows).

## 4) Modeling
- **Baselines:** DecisionTree, RandomForest (with grid search)
- **Advanced (optional):** XGBoost (toggle by installing `xgboost`)
- **Target:** binary `target_up` (sign of next-day return).  
- **Validation:** time-based split (train/val/test by dates).

## 5) Trading simulation
Strategies in `src/simulation/strategies.py`:
1. **long_only_threshold:** pick top-prob longs (equal-weight), costs in bps
2. **long_short_threshold:** long top-prob & short low-prob (equal-weight)
3. **long_only_sl_tp:** exact row-wise sim with **reinvestment**, **stop-loss / take-profit**, and **position slots**

KPIs via `src/utils_metrics.py`: **CAGR**, **Sharpe**, **Max Drawdown**, **Total Return**.  
Benchmarks (extend): Buy & Hold of an index or universe equally weighted.

## 6) Repository layout
project/
   config/ # YAML settings
   data/ # raw/interim/processed (ignored by git)
   notebooks/ # EDA, pipeline demo, backtest report
   reports/ # figures/backtests (auto-generated)
   src/ # pipeline, features, strategies, metrics
   Dockerfile, Makefile, requirements.txt, README.md


## 7) Reproducibility & Automation
### Install & Run
```bash
# venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place your data
ls data/raw/portfolio_data.csv  # must exist

# run end-to-end
python -m src.pipeline.run_all --config config/config.yaml
# or
make run
```

### Docker
```
docker build -t sma-pipeline .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" sma-pipeline
```

### Cron (daily run)
```
0 7 * * 1-5 cd /path/to/project && /usr/bin/docker run --rm \
  -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" sma-pipeline
```

### Two regimes

- Offline mode: work entirely from data/raw/portfolio_data.csv.
- Online mode (extend): add an API downloader in step_01_download.py (e.g., CryptoCompare/Tiingo). Keep the same schema, then re-run.

### Incremental processing

- Add simple checkpoints per step (file existence checks) and “append-latest-only” logic (out of scope for this baseline; described in `src/README.md`).
