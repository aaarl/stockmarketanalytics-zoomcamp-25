# Stock Market Analytics – End-to-End ML + Trading Simulation

## Objective
Build an end-to-end pipeline to predict next-period returns and simulate algorithmic trading strategies.

## Data
- Main: `portfolio_data.csv` (prices for AMZN, DPZ, BTC, NFLX)
- Meta: `symbols_valid_meta.csv` (symbol metadata; optional merge)
- **No data is committed to Git.** Place files in `data/raw/`.

## Reproducibility
```bash
make setup
make run
