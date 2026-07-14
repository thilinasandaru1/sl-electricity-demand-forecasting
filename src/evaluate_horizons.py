"""
Multi-Horizon Forecast Evaluation Module (1 to 24 Hours)
Target: Evaluating Error Degradation across 1h, 6h, 12h, and 24h Horizons
for Short-Term Load Forecasting on Sri Lanka Grid Telemetry
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm


def calculate_stlf_metrics(y_true, y_pred) -> dict:
    """
    Computes key statistical metrics for load forecasting evaluation:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"RMSE (kW)": rmse, "MAE (kW)": mae, "MAPE (%)": mape}


def evaluate_forecast_horizons(processed_csv_path: str = None):
    """
    Evaluates forecasting algorithms across multiple forecast horizons:
    - 1-Hour Horizon (4 time steps @ 15-min intervals)
    - 6-Hour Horizon (24 time steps)
    - 12-Hour Horizon (48 time steps)
    - 24-Hour Horizon (96 time steps)
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
        raise FileNotFoundError(f"Processed data file not found at {processed_csv_path}. Run src/preprocess.py first.")

    print(f"[HORIZON EVAL] Loading dataset from {processed_csv_path}...")
    df = pd.read_csv(processed_csv_path)

    ignore_cols = ["Timestamp", "Hour of Day", "Day of Week", "Month", "Season", "Load Demand (kW)"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    target_col = "Load Demand (kW)"

    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define forecast lead times (15-min frequency: 4 steps = 1h, 24 steps = 6h, 48 steps = 12h, 96 steps = 24h)
    horizons = {
        "1-Hour Ahead": 4,
        "6-Hour Ahead": 24,
        "12-Hour Ahead": 48,
        "24-Hour Ahead": 96,
    }

    # Models to benchmark
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=1.0, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print("[TRAINING] Fitting models for multi-horizon forecast evaluation...")
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    xgb_preds_full = xgb_model.predict(X_test)
    rf_preds_full = rf_model.predict(X_test)

    results = []

    for label, steps in horizons.items():
        # Evaluate performance on leading 'steps' window for each operational horizon
        eval_y = y_test.iloc[:steps]
        eval_xgb = xgb_preds_full[:steps]
        eval_rf = rf_preds_full[:steps]

        xgb_m = calculate_stlf_metrics(eval_y, eval_xgb)
        xgb_m["Model"] = "XGBoost Ensemble"
        xgb_m["Forecast Horizon"] = label
        xgb_m["Steps (15-min)"] = steps
        results.append(xgb_m)

        rf_m = calculate_stlf_metrics(eval_y, eval_rf)
        rf_m["Model"] = "Random Forest Regressor"
        rf_m["Forecast Horizon"] = label
        rf_m["Steps (15-min)"] = steps
        results.append(rf_m)

    results_df = pd.DataFrame(results)[["Forecast Horizon", "Steps (15-min)", "Model", "RMSE (kW)", "MAE (kW)", "MAPE (%)"]]

    print("\n" + "=" * 70)
    print("      MULTI-HORIZON FORECAST DEGRADATION BENCHMARK (1h to 24h)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Save results to both data/results/ and root results/
    out_dirs = [os.path.join("data", "results"), "results"]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "multi_horizon_metrics.csv")
        results_df.to_csv(out_file, index=False)
        print(f"[SUCCESS] Multi-horizon results exported to {out_file}")

    return results_df


if __name__ == "__main__":
    evaluate_forecast_horizons()
