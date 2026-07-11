import os
import pandas as pd
import numpy as np
try:
    from src.data_ingestion import load_raw_telemetry, align_time_series_grid, validate_time_series_integrity
except ModuleNotFoundError:
    from data_ingestion import load_raw_telemetry, align_time_series_grid, validate_time_series_integrity


def create_lag_features(df: pd.DataFrame, target_col: str = "Load Demand (kW)", lags: list = [1, 2, 24, 168]) -> pd.DataFrame:
    """
    Constructs chronological lag features for short-term load forecasting.
    Because the data frequency is 15 minutes:
    - lag 1 = 15 minutes
    - lag 2 = 30 minutes
    - lag 24 = 6 hours
    - lag 168 = 42 hours (1.75 days)
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def encode_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes cyclical temporal columns (Hour of Day, Day of Week, Month) using sine and cosine functions.
    This preserves the temporal continuity (e.g. hour 23 is close to hour 0).
    """
    df = df.copy()
    
    # Hour of Day cyclical encoding (0-23 range)
    if "Hour of Day" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["Hour of Day"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["Hour of Day"] / 24.0)
        
    # Day of Week cyclical encoding (0-6 range)
    if "Day of Week" in df.columns:
        df["day_sin"] = np.sin(2 * np.pi * df["Day of Week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["Day of Week"] / 7.0)
        
    # Month cyclical encoding (1-12 range)
    if "Month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
        
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the categorical column 'Season'.
    Explicitly outputs binary indicators for all seasons (ideal for tree-based models).
    """
    df = df.copy()
    if "Season" in df.columns:
        df = pd.get_dummies(df, columns=["Season"], prefix="season", drop_first=False, dtype=float)
    return df


def preprocess_data(input_path: str, output_path: str, lags: list = [1, 2, 24, 168]):
    """
    Loads raw utility dataset, aligns time series grid, constructs lag features,
    applies cyclical encoding, performs one-hot encoding, and cleans NaNs.
    """
    print(f"Loading raw telemetry from {input_path}...")
    df = load_raw_telemetry(input_path)

    # Perform integrity checks and grid alignment
    print("Verifying time-series integrity...")
    if not validate_time_series_integrity(df):
        print("Data integrity check failed or incomplete. Aligning time series grid...")
        df = align_time_series_grid(df)
        validate_time_series_integrity(df)

    # Create lag features
    print("Generating lag features...")
    df = create_lag_features(df, target_col="Load Demand (kW)", lags=lags)

    # Cyclical encoding
    print("Encoding cyclical temporal features (Hour, Day, Month)...")
    df = encode_cyclical_features(df)

    # Categorical encoding
    print("One-hot encoding categorical features (Season)...")
    df = encode_categorical_features(df)

    # Handle NaNs introduced by lag features
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_len = initial_len - len(df)
    print(f"Dropped {dropped_len} rows containing NaNs introduced by lagging.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the processed data
    print(f"Saving preprocessed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing pipeline completed successfully.")
    print(f"Final preprocessed shape: {df.shape}")


if __name__ == "__main__":
    input_file = "data/raw/load_forecasting_dataset_corrected.csv"
    output_file = "data/processed/load_forecasting_dataset_processed.csv"
    preprocess_data(input_file, output_file)