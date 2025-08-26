# Reports

Auto-generated results:

- `backtests/summary.csv` — per-strategy KPIs: `CAGR, Sharpe, MaxDrawdown, TotalReturn`.
- `figures/` — optional plots (equity curves, rolling performance).

Recreate by running:
```bash
python -m src.pipeline.run_all --config config/config.yaml
# or
make run
