"""
Hyperparameter Tuning & Time-Series Cross-Validation Module
Target: XGBoost Ensemble Optimization via TimeSeriesSplit
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


def optimize_xgboost_stlf(
    processed_csv_path: str = None,
    n_splits: int = 5,
    param_grid: dict = None,
    n_jobs: int = -1,
    save_results: bool = True,
):
    """
    Executes TimeSeriesSplit Cross-Validation to find optimal
    XGBoost hyperparameters without temporal data leakage.
    """
    if processed_csv_path is None:
        default_path = os.path.join("data", "processed", "load_forecasting_dataset_processed.csv")
        fallback_path = os.path.join("data", "processed", "processed_features.csv")
        if os.path.exists(default_path):
            processed_csv_path = default_path
        elif os.path.exists(fallback_path):
            processed_csv_path = fallback_path
        else:
            processed_csv_path = default_path

    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"Processed file not found at {processed_csv_path}. Please run src/preprocess.py first.")

    print(f"[TUNING] Loading dataset from {processed_csv_path}...")
    df = pd.read_csv(processed_csv_path)

    # Ignore index, raw temporal/season columns, and target
    ignore_cols = ["Timestamp", "Hour of Day", "Day of Week", "Month", "Season", "Load Demand (kW)"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    target_col = "Load Demand (kW)"

    X = df[feature_cols]
    y = df[target_col]

    print(f"[INFO] Feature space shape: {X.shape} | Target: '{target_col}'")

    # Time-Series Split (5 Folds forward-chaining)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }

    base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)

    print(f"[TUNING] Initializing TimeSeriesSplit GridSearchCV ({n_splits} folds)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X, y)

    best_rmse = -grid_search.best_score_
    best_params = grid_search.best_params_

    print("\n" + "=" * 50)
    print("        OPTIMIZED HYPERPARAMETERS")
    print("=" * 50)
    for param, val in best_params.items():
        print(f"  {param}: {val}")

    print(f"\n[RESULT] Best Cross-Validation RMSE: {best_rmse:.2f} kW")

    if save_results:
        results_dir = os.path.join("data", "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "xgboost_best_params.json")

        output_payload = {
            "best_params": best_params,
            "best_cv_rmse_kw": round(float(best_rmse), 4),
            "n_splits": n_splits,
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        }

        with open(results_file, "w") as f:
            json.dump(output_payload, f, indent=4)
        print(f"[SUCCESS] Tuning results saved to {results_file}")

    return grid_search


if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "load_forecasting_dataset_processed.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "processed", "processed_features.csv")
    optimize_xgboost_stlf(data_path)
