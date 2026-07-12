"""
Visualization Module for Generating Publication-Quality Figures
Target: Load Curves and Feature Importance Analysis for Sri Lanka Grid STLF
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


def load_best_xgboost_params() -> dict:
    """
    Loads tuned XGBoost hyperparameters from data/results/xgboost_best_params.json if available.
    """
    json_path = os.path.join("data", "results", "xgboost_best_params.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                print(f"[INFO] Loaded optimized XGBoost parameters from {json_path}")
                return data.get("best_params", {})
        except Exception as e:
            print(f"[WARNING] Could not read {json_path}: {e}")

    print("[INFO] Using default XGBoost hyperparameters.")
    return {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "subsample": 1.0}


def generate_research_figures(processed_csv_path: str = None):
    """
    Generates high-resolution diagnostic figures for dissertation manuscript:
    1. 48-Hour Actual vs Predicted Load Curve Trace
    2. Top Predictor Feature Importance Rankings
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
        raise FileNotFoundError(f"Processed file not found at {processed_csv_path}. Run src/preprocess.py first.")

    os.makedirs("figures", exist_ok=True)
    print(f"[INFO] Loading processed telemetry dataset from {processed_csv_path}...")
    df = pd.read_csv(processed_csv_path)

    # Exclude non-predictive/raw temporal columns and target variable
    ignore_cols = ["Timestamp", "Hour of Day", "Day of Week", "Month", "Season", "Load Demand (kW)"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    target_col = "Load Demand (kW)"

    split_idx = int(len(df) * 0.8)
    X_train = df[feature_cols].iloc[:split_idx]
    X_test = df[feature_cols].iloc[split_idx:]
    y_train = df[target_col].iloc[:split_idx]
    y_test = df[target_col].iloc[split_idx:]

    # Retrieve tuned hyperparameters or default model configuration
    params = load_best_xgboost_params()
    params["random_state"] = 42

    print("[INFO] Training XGBoost model for visualization analysis...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Set publication aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")

    # -------------------------------------------------------------
    # Figure 1: Actual vs Predicted Load Trace (Sample 192 steps = 48 Hours @ 15-min)
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:192], label="Actual Load Demand (kW)", color="black", lw=1.5)
    plt.plot(
        preds[:192],
        label="XGBoost Predicted Load (kW)",
        color="#0072BD",
        linestyle="--",
        lw=1.5,
    )
    plt.title("Short-Term Load Forecast Comparison (48-Hour Operational Window)", fontsize=13, fontweight="bold")
    plt.xlabel("15-Minute Time Steps", fontsize=11)
    plt.ylabel("Demand (kW)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig1_path = os.path.join("figures", "fig1_load_trace_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Exported Figure 1: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Top Predictor Feature Importance
    # -------------------------------------------------------------
    importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature",
        palette="viridis",
        hue="Feature",
        legend=False,
    )
    plt.title("Top 10 Independent Feature Importance Rankings", fontsize=13, fontweight="bold")
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    plt.tight_layout()

    fig2_path = os.path.join("figures", "fig2_feature_importance.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Exported Figure 2: {fig2_path}")

    print("[SUCCESS] All research figures generated and saved to figures/ directory.")


if __name__ == "__main__":
    generate_research_figures()
