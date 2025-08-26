## File: `config.yaml`
Main knobs for the full pipeline:

- `paths`: output/input folders (created automatically).
- `data`:
  - `main_file`: name of the wide price CSV (Date + one column per ticker).
  - `meta_file`: optional metadata for merging exchange/ETF flags.
  - `date_col`: date column name in CSV.
  - `tickers`: list for reference (not enforced).
- `target`: prediction horizon and naming (informational).
- `features`: documented feature groups (informational).
- `split`: time-based split points (`train_end`, `val_end`, `test_end`).
- `models`: which models to train (`DecisionTreeClassifier`, `RandomForestClassifier`, optional `XGBClassifier`).
- `simulation`:
  - `strategies`: enabled strategies.
  - `thresholds`: buy/short probability thresholds.
  - `risk`: `stop_loss`, `take_profit` (used in exact sim).
  - `costs`: `fee_bps` per trade leg.
  - `capital`: initial capital and concurrency limit.
