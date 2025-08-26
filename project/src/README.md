# Project Source Overview

## src/ — Codebase

### Pipeline (`src/pipeline/`)
Each step is idempotent and file-based:
1. `step_01_download.py` — validate presence of raw files (extend to API fetch).
2. `step_02_unify_dataset.py` — wide→long melt, add calendar features, optional meta merge.
3. `step_03_feature_engineering.py` — compute numeric features (basic + technical), define targets (`target_return_1d`, `target_up`), drop NaNs.
4. `step_04_train.py` — time-based split, train `DecisionTree`, `RandomForest` (grid search), optional `XGBClassifier`.
5. `step_05_predict.py` — predict probabilities (`pred_*`) for **all rows**.
6. `step_06_simulate.py` — run strategies, compute KPIs, write `reports/backtests/summary.csv`.

Entry point:
```bash
python -m src.pipeline.run_all --config config/config.yaml
```
