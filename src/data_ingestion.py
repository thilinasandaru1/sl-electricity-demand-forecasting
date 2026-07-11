"""
Data Ingestion and Alignment Module for Short-Term Load Forecasting
Target: Sri Lanka National Electrical Grid Telemetry
"""

import pandas as pd
import numpy as np
import os


def load_raw_telemetry(file_path: str) -> pd.DataFrame:
    """
    Loads raw utility load telemetry and weather data from CSV,
    parses timestamps, and sorts chronologically.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at specified path: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    
    # Drop rows with invalid timestamps
    invalid_timestamps = df["Timestamp"].isna().sum()
    if invalid_timestamps > 0:
        print(f"[WARNING] Dropping {invalid_timestamps} rows due to unparseable timestamps.")
        df = df.dropna(subset=["Timestamp"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    print(f"[INFO] Successfully loaded {len(df)} observations.")
    if not df.empty:
        print(f"[INFO] Temporal Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    return df


def align_time_series_grid(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Ensures the DataFrame has a continuous time series index at the specified frequency.
    Handles duplicate timestamps by averaging numeric fields and keeping first for categoricals,
    reindexes to a complete grid, and interpolates/fills missing values.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Handle duplicates
    if df["Timestamp"].duplicated().any():
        num_duplicates = df["Timestamp"].duplicated().sum()
        print(f"[WARNING] Found {num_duplicates} duplicate timestamps. Averaging numeric columns.")
        
        # Numeric columns mean aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Categorical/object columns first value aggregation
        other_cols = [col for col in df.columns if col not in numeric_cols and col != "Timestamp"]
        
        agg_dict = {col: "mean" for col in numeric_cols}
        for col in other_cols:
            agg_dict[col] = "first"
            
        df = df.groupby("Timestamp", as_index=False).agg(agg_dict)

    # Set Timestamp as index and sort
    df = df.set_index("Timestamp").sort_index()

    # 2. Reindex to complete datetime range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, name="Timestamp")
    if len(df.index) != len(full_range):
        print(f"[WARNING] Missing time steps detected. Reindexing from {len(df.index)} to {len(full_range)} steps.")
        df = df.reindex(full_range)

    # 3. Fill missing values (imputation)
    # For numerical continuous columns, use linear interpolation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude calendar/temporal fields and categorical flags from linear interpolation
    temporal_cols = ["Hour of Day", "Day of Week", "Month"]
    flag_cols = ["Public Event"]
    
    interpolate_cols = [c for c in numeric_cols if c not in temporal_cols and c not in flag_cols]
    if interpolate_cols:
        df[interpolate_cols] = df[interpolate_cols].interpolate(method="linear")

    # Recalculate calendar/temporal columns to ensure 100% correctness
    df["Hour of Day"] = df.index.hour
    df["Day of Week"] = df.index.dayofweek
    df["Month"] = df.index.month

    # Categorical and flag columns: use forward-fill then backward-fill
    fill_cols = df.select_dtypes(exclude=[np.number]).columns.tolist() + [col for col in flag_cols if col in df.columns]
    if fill_cols:
        df[fill_cols] = df[fill_cols].ffill().bfill()

    # Reset index to restore 'Timestamp' as a column
    df = df.reset_index()
    return df


def validate_time_series_integrity(df: pd.DataFrame, freq: str = "15min") -> bool:
    """
    Validates that:
    1. Timestamps are sorted and continuous with the expected frequency.
    2. No duplicate timestamps exist.
    3. No missing (NaN) values exist in any column.
    """
    if df.empty:
        print("[INSPECTION] DataFrame is empty.")
        return False

    # Check duplicates
    duplicates = df["Timestamp"].duplicated().sum()
    
    # Check frequency gaps
    time_diffs = df["Timestamp"].diff().dropna()
    expected_interval = pd.Timedelta(freq)
    missing_steps = (time_diffs != expected_interval).sum()

    # Check NaNs
    nan_count = df.isna().sum().sum()

    print(f"[INSPECTION] Duplicate timestamps found: {duplicates}")
    print(f"[INSPECTION] Missing sequence intervals: {missing_steps}")
    print(f"[INSPECTION] Total NaN values found: {nan_count}")

    is_valid = (duplicates == 0) and (missing_steps == 0) and (nan_count == 0)
    if is_valid:
        print("[INSPECTION] Time-series integrity check PASSED.")
    else:
        print("[INSPECTION] Time-series integrity check FAILED.")
    
    return is_valid


if __name__ == "__main__":
    raw_csv_path = os.path.join("data", "raw", "load_forecasting_dataset_corrected.csv")
    telemetry_df = load_raw_telemetry(raw_csv_path)
    
    # Check and align if needed
    if not validate_time_series_integrity(telemetry_df):
        print("[INFO] Aligning grid and imputing values...")
        telemetry_df = align_time_series_grid(telemetry_df)
        validate_time_series_integrity(telemetry_df)