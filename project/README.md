# Stock Market Analytics – End-to-End ML + Trading Simulation

## Objective
End-to-end pipeline to predict next-day returns and simulate algorithmic trading strategies on a small multi-asset universe (AMZN, DPZ, BTC, NFLX).

## Data
- Main: `portfolio_data.csv` (daily close prices)
- Meta: `symbols_valid_meta.csv` (optional; used to enrich stock tickers)
- **Data is not committed.** Place files in `data/raw/`.

## How to run
```bash
make setup
# Place CSVs into data/raw/ before running:
#   data/raw/portfolio_data.csv
#   data/raw/symbols_valid_meta.csv
make run
