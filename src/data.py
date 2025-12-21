"""
Data loading, cleaning, and splitting utilities for CIC-IDS2017 dataset.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import joblib
from typing import Tuple, List, Optional


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from the data directory."""
    # Files to load (Friday DDos and PortScan files excluded - will be used for testing later)
    csv_files = [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ]
    
    dfs = []
    file_mapping = []
    
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        if os.path.exists(filepath):
            print(f"Loading {csv_file}...")
            df = pd.read_csv(filepath, low_memory=False)
            df['source_file'] = csv_file
            dfs.append(df)
            file_mapping.append(csv_file)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    if not dfs:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    print(f"Concatenating {len(dfs)} files...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(combined_df)}")
    
    return combined_df, file_mapping


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe: strip columns, convert to numeric, handle infinity."""
    print("Cleaning data...")
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Identify non-feature columns to drop
    non_feature_keywords = [
        'flow id', 'flowid', 'source ip', 'src ip', 'destination ip', 'dst ip',
        'timestamp', 'time', 'source port', 'src port', 'destination port', 'dst port',
        'protocol', 'label', 'source_file'
    ]
    
    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in non_feature_keywords):
            if col_lower != 'label':  # Keep label for now, drop later
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"Dropping non-feature columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Extract label before dropping
    if 'Label' not in df.columns:
        raise ValueError("'Label' column not found in dataframe")
    
    # Convert all feature columns (except Label and source_file) to numeric
    feature_cols = [col for col in df.columns if col not in ['Label', 'source_file']]
    
    print(f"Converting {len(feature_cols)} feature columns to numeric...")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinity values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf, 'Infinity', 'inf', '-inf'], np.nan)
    
    # Drop rows with NaN
    before_drop = len(df)
    df = df.dropna(subset=feature_cols)
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows with NaN values ({after_drop} remaining)")
    
    return df


def create_binary_target(df: pd.DataFrame) -> pd.Series:
    """Create binary target: BENIGN=0, else=1."""
    y = (df['Label'] != 'BENIGN').astype(int)
    print(f"Target distribution: BENIGN (0)={sum(y==0)}, ATTACK (1)={sum(y==1)}")
    return y


def split_by_file(df: pd.DataFrame, file_mapping: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by file: train on Mon-Thu, test on Friday Morning."""
    # Test set: only Friday-WorkingHours-Morning (other Friday files excluded for now)
    friday_test_files = [
        'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    ]
    
    train_mask = ~df['source_file'].isin(friday_test_files)
    test_mask = df['source_file'].isin(friday_test_files)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"File-based split: Train={len(train_df)}, Test={len(test_df)}")
    
    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Extract features and target, optionally scale."""
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Label', 'source_file']]
    
    X = df[feature_cols].values.astype(np.float32)
    y = create_binary_target(df).values
    
    # Scale features
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Fitted StandardScaler on training data")
    elif scaler is not None:
        X = scaler.transform(X)
        print("Transformed features using existing scaler")
    
    return X, y, scaler, feature_cols


def load_and_prepare_data(
    data_dir: str,
    split_by_file: bool = True,
    random_split: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Main function to load, clean, and split data.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Load and clean
    df, file_mapping = load_all_csvs(data_dir)
    df = clean_data(df)
    
    # Split data
    if random_split:
        print("Using random split...")
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['Label']
        )
    else:
        print("Using file-based split...")
        train_df, test_df = split_by_file(df, file_mapping)
    
    # Prepare features and targets
    X_train, y_train, scaler, feature_names = prepare_features_and_target(
        train_df, fit_scaler=True
    )
    X_test, y_test, _, _ = prepare_features_and_target(
        test_df, scaler=scaler, fit_scaler=False
    )
    
    print(f"Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

