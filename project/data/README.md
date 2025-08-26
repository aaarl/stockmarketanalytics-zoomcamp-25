**Do not commit data.** This folder is in `.gitignore`.

- `raw/` — put `portfolio_data.csv` here (wide: `Date, TICKER1, TICKER2, ...`).  
  Optional: `symbols_valid_meta.csv` with columns like `Symbol, Listing Exchange, ETF`.
- `interim/` — auto-generated unified long table (`unified_long.csv`), etc.
- `processed/` — features & predictions (`dataset_features.csv`, `predictions.csv`).

## Expected schema for `portfolio_data.csv`
- `Date` — YYYY-MM-DD (or ISO datetime).
- One column per ticker, numeric prices (close or adjusted).

> If you switch to online APIs, write the fetched data here with the same schema.
