"""
Model Training and Empirical Benchmarking Module
Target: Short-Term Load Forecasting (STLF) on Sri Lanka National Grid
Models Evaluated: Ridge Baseline, Random Forest Regressor, Extreme Gradient Boosting (XGBoost)
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


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


def train_and_evaluate_models(processed_csv_path: str):
    """
    Loads preprocessed feature matrix, executes time-series split,
    trains comparative algorithms, and outputs benchmarking metrics.
    """
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(
            f"Processed data not found at {processed_csv_path}. Run preprocess.py first."
        )

    df = pd.read_csv(processed_csv_path)

    # Define predictor feature space and target
    # Ignore index, raw temporal columns, and the target.
    # Note: Public Event is kept as a predictor, and cyclical temporal/one-hot features are used.
    ignore_cols = ["Timestamp", "Hour of Day", "Day of Week", "Month", "Load Demand (kW)"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    target_col = "Load Demand (kW)"

    X = df[feature_cols]
    y = df[target_col]

    # Strict chronological split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"[INFO] Features matrix shape: {X.shape} | Training rows: {len(X_train)} | Test rows: {len(X_test)}"
    )
    print(f"[INFO] Features used for training: {list(X.columns)}")

    # Initialize candidate models
    models = {
        "Ridge Linear Baseline": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost Ensemble": xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42
        ),
    }

    # Fit scaling ONLY on the training features to prevent lookahead/data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for name, model in models.items():
        print(f"\n[TRAINING] Fitting {name}...")
        
        # Train Ridge on Scaled features, and Tree-based models on original features
        if "Ridge" in name:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        metrics = calculate_stlf_metrics(y_test, predictions)
        metrics["Model"] = name
        results.append(metrics)

        print(
            f"[RESULT] {name} -> RMSE: {metrics['RMSE (kW)']:.2f} kW | MAPE: {metrics['MAPE (%)']:.2f}%"
        )

    # Format output summary dataframe
    results_df = pd.DataFrame(results)[
        ["Model", "RMSE (kW)", "MAE (kW)", "MAPE (%)"]
    ]
    print("\n" + "=" * 60)
    print("      EXPERIMENTAL MODEL BENCHMARKING RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save benchmark table
    results_dir = os.path.join("data", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(
        os.path.join(results_dir, "benchmark_metrics.csv"), index=False
    )
    print(
        f"\n[SUCCESS] Benchmarking summary saved to data/results/benchmark_metrics.csv"
    )


if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "load_forecasting_dataset_processed.csv")
    train_and_evaluate_models(data_path)