import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib
from datetime import datetime

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def _time_split(df: pd.DataFrame, date_col: str, train_end: str, val_end: str):
    df[date_col] = pd.to_datetime(df[date_col])
    train = df[df[date_col] <= pd.to_datetime(train_end)]
    valid = df[(df[date_col] > pd.to_datetime(train_end)) & (df[date_col] <= pd.to_datetime(val_end))]
    test  = df[df[date_col] > pd.to_datetime(val_end)]
    return train, valid, test

def run(cfg: dict):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    df = pd.read_csv(processed_dir / "dataset_features.csv")

    date_col = cfg["data"]["date_col"]
    train_end = cfg["split"]["train_end"]
    val_end   = cfg["split"]["val_end"]

    feature_cols = [c for c in df.columns if c not in [date_col, "ticker", "price", "target_return_1d", "target_up"]]
    train, valid, test = _time_split(df, date_col, train_end, val_end)

    X_train, y_train = train[feature_cols], train["target_up"]
    X_valid, y_valid = valid[feature_cols], valid["target_up"]
    X_test,  y_test  = test[feature_cols],  test["target_up"]

    models = {}

    # 1) Decision Tree (baseline)
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]}
    dt_cv = GridSearchCV(dt, dt_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    dt_cv.fit(X_train, y_train)
    models["DecisionTree"] = dt_cv.best_estimator_
    print(f"[step_04] DecisionTree best params: {dt_cv.best_params_}")

    # 2) Random Forest (baseline + tuning)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", "log2"]
    }
    rf_cv = GridSearchCV(rf, rf_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    models["RandomForest"] = rf_cv.best_estimator_
    print(f"[step_04] RandomForest best params: {rf_cv.best_params_}")

    # 3) XGBoost (advanced)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)
        models["XGBClassifier"] = xgb
        print("[step_04] Trained XGBClassifier.")
    else:
        print("[step_04] XGBoost not available; skipping advanced model.")

    # Evaluate on validation and test
    for name, m in models.items():
        val_auc = roc_auc_score(y_valid, m.predict_proba(X_valid)[:,1]) if len(valid) else float("nan")
        test_auc = roc_auc_score(y_test,  m.predict_proba(X_test)[:,1])  if len(test)  else float("nan")
        print(f"[step_04] {name} - Val AUC: {val_auc:.3f} | Test AUC: {test_auc:.3f}")

        joblib.dump(m, processed_dir / f"{name}.joblib")
