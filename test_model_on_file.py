"""
Test saved model on a CSV file without labels, then compare with original labels.
"""
import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import create_model
from src.data import clean_data
from src.utils import calculate_metrics, plot_confusion_matrix, print_metrics


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic data."""
    
    def __init__(self, X):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


def load_model_and_scaler(model_dir='runs'):
    """Load model, scaler, and feature names."""
    model_path = os.path.join(model_dir, 'model.pt')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'features.json')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    
    # Load scaler and features
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    # Load model
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_features=len(feature_names), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Using device: {device}")
    print(f"Loaded {len(feature_names)} features")
    
    return model, scaler, feature_names, device


def prepare_data_without_labels(csv_path, scaler, feature_names, save_labels=True):
    """Load CSV, clean, and prepare features without labels."""
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Original shape: {df.shape}")
    
    # Clean data (same as training) - this keeps Label column
    df = clean_data(df)
    print(f"After cleaning: {df.shape}")
    
    # Extract labels AFTER cleaning (to match the cleaned data)
    original_labels = None
    if 'Label' in df.columns:
        original_labels = df['Label'].copy()
        if save_labels:
            labels_backup_path = csv_path.replace('.csv', '_labels_backup.csv')
            original_labels.to_csv(labels_backup_path, index=False)
            print(f"Original labels (after cleaning) saved to {labels_backup_path}")
            print(f"Label distribution: {original_labels.value_counts().to_dict()}")
    
    # Extract features (exclude Label and source_file if exists)
    cols_to_drop = ['Label', 'source_file']
    available_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if available_cols_to_drop:
        df = df.drop(columns=available_cols_to_drop, errors='ignore')
    
    # Ensure we have the same features as training
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with 0
        for feat in missing_features:
            df[feat] = 0
    
    # Select only the features used in training
    X = df[feature_names].values.astype(np.float32)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, df, original_labels


def predict(model, dataloader, device):
    """Make predictions on data."""
    model.eval()
    all_probas = []
    all_preds = []
    
    with torch.no_grad():
        for X_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probas = torch.sigmoid(outputs).cpu().numpy()
            preds = (probas > 0.5).astype(int)
            
            all_probas.extend(probas)
            all_preds.extend(preds)
    
    return np.array(all_probas), np.array(all_preds)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test model on CSV file')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file to test')
    parser.add_argument('--model_dir', type=str, default='runs',
                        help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    model, scaler, feature_names, device = load_model_and_scaler(args.model_dir)
    
    # Load and prepare data
    print("\n" + "=" * 60)
    print("Preparing data...")
    print("=" * 60)
    X, df_original, original_labels = prepare_data_without_labels(args.csv_path, scaler, feature_names)
    
    # Create dataset and dataloader
    dataset = TrafficDataset(X)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Make predictions
    print("\n" + "=" * 60)
    print("Making predictions...")
    print("=" * 60)
    probas, preds = predict(model, dataloader, device)
    
    # Save predictions
    results_df = df_original.copy()
    results_df['predicted_proba'] = probas
    results_df['predicted_label'] = preds
    results_df['predicted_binary'] = (probas > args.threshold).astype(int)
    
    output_csv = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # If original labels exist, compare
    if original_labels is not None:
        print("\n" + "=" * 60)
        print("Comparing with original labels...")
        print("=" * 60)
        
        y_true = (original_labels != 'BENIGN').astype(int).values
        y_pred = preds
        
        print(f"Original labels distribution:")
        print(f"  BENIGN (0): {(y_true == 0).sum()} ({(y_true == 0).mean()*100:.2f}%)")
        print(f"  ATTACK (1): {(y_true == 1).sum()} ({(y_true == 1).mean()*100:.2f}%)")
        
        print(f"\nPredicted labels distribution:")
        print(f"  BENIGN (0): {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.2f}%)")
        print(f"  ATTACK (1): {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.2f}%)")
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, probas)
        print_metrics(metrics, "Test")
        
        # Save confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, cm_path, 
                            title="Model Predictions vs Original Labels")
        print(f"\nConfusion matrix saved to {cm_path}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    else:
        print("\nNo original labels found in CSV. Only predictions saved.")
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

