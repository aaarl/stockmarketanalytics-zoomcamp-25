# Stock Market Analytics – End-to-End ML + Trading Simulation

## Objective
End-to-end pipeline to predict next-day returns and simulate algorithmic trading strategies on a small multi-asset universe (AMZN, DPZ, BTC, NFLX).

## Data
- Main: `portfolio_data.csv` (daily close prices) source is from [Kaggle - AMZN, DPZ, BTC, NTFX adjusted May 2013-May2019](https://www.kaggle.com/datasets/hershyandrew/amzn-dpz-btc-ntfx-adjusted-may-2013may2019)
- Meta: `symbols_valid_meta.csv` (optional; used to enrich stock tickers) [Kaggle - Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Data is not committed.** For the sake of the project they have been stored (just locally) in `data/raw/`.

## How to run
```bash
make setup
# Place CSVs into data/raw/ before running:
#   data/raw/portfolio_data.csv
#   data/raw/symbols_valid_meta.csv
make run
