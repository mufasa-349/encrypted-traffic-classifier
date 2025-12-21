"""
Evaluation script for trained 1D-CNN model.
"""
import argparse
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import load_and_prepare_data
from model import create_model
from utils import calculate_metrics, plot_confusion_matrix, print_metrics


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probas = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            probas = torch.sigmoid(outputs).cpu().numpy()
            preds = (probas > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probas.extend(probas)
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained 1D-CNN model')
    parser.add_argument('--data_dir', type=str, default='data/cicids2017',
                        help='Directory containing CSV files')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    # Load scaler and feature names
    ckpt_dir = os.path.dirname(args.ckpt)
    scaler_path = os.path.join(ckpt_dir, 'scaler.pkl')
    features_path = os.path.join(ckpt_dir, 'features.json')
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    print(f"Loaded scaler from {scaler_path}")
    print(f"Loaded {len(feature_names)} feature names")
    
    # Load and prepare data
    print("=" * 60)
    print("Loading and preparing data...")
    print("=" * 60)
    
    # We need to load data the same way as training
    from data import load_all_csvs, clean_data, create_binary_target, prepare_features_and_target, split_by_file
    
    df, file_mapping = load_all_csvs(args.data_dir)
    df = clean_data(df)
    
    # Use file-based split (same as training)
    train_df, test_df = split_by_file(df, file_mapping)
    
    # Prepare test features
    X_test, y_test, _, _ = prepare_features_and_target(
        test_df, scaler=scaler, fit_scaler=False
    )
    
    print(f"Test set: {len(X_test)} samples")
    
    # Create dataset and dataloader
    test_dataset = TrafficDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(num_features=len(feature_names), device=device)
    
    # Load checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"Loaded model from {args.ckpt}")
    
    # Loss function
    pos_weight = torch.tensor(
        (y_test == 0).sum() / (y_test == 1).sum(),
        device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    test_loss, test_metrics, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, "Test")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix_eval.png')
    plot_confusion_matrix(
        y_true, y_pred, cm_path, title="Evaluation Confusion Matrix"
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    print("Evaluation completed!")


if __name__ == '__main__':
    main()

