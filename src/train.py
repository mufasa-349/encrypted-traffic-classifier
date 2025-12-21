"""
Training script for 1D-CNN binary classifier.
"""
import argparse
import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_and_prepare_data
from src.model import create_model
from src.utils import set_seed, calculate_metrics, plot_confusion_matrix, print_metrics


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_weighted_sampler(y_train):
    """Create weighted sampler to handle class imbalance."""
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


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
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train 1D-CNN for CIC-IDS2017')
    parser.add_argument('--data_dir', type=str, default='data/cicids2017',
                        help='Directory containing CSV files')
    parser.add_argument('--split_by_file', type=int, default=1,
                        help='Use file-based split (1) or random split (0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Output directory for checkpoints and results')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("=" * 60)
    print("Loading and preparing data...")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data(
        data_dir=args.data_dir,
        split_by_file=bool(args.split_by_file),
        random_split=not bool(args.split_by_file),
        random_state=args.seed
    )
    
    # Create validation split from training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=args.seed, stratify=y_train
    )
    
    print(f"Train: {len(X_train_split)}, Val: {len(X_val_split)}, Test: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = TrafficDataset(X_train_split, y_train_split)
    val_dataset = TrafficDataset(X_val_split, y_val_split)
    test_dataset = TrafficDataset(X_test, y_test)
    
    # Create weighted sampler for training
    sampler = create_weighted_sampler(y_train_split)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = create_model(num_features=X_train.shape[1], device=device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    # Calculate pos_weight for class imbalance
    pos_weight = torch.tensor(
        (y_train_split == 0).sum() / (y_train_split == 1).sum(),
        device=device,
        dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop with early stopping
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print_metrics(val_metrics, "Val")
        
        # Early stopping
        val_auc = val_metrics.get('roc_auc', 0.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"✓ New best validation AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (val AUC: {best_val_auc:.4f})")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print_metrics(test_metrics, "Test")
    
    # Get predictions for confusion matrix
    model.eval()
    all_test_preds = []
    all_test_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            probas = torch.sigmoid(outputs).cpu().numpy()
            preds = (probas > 0.5).astype(int)
            all_test_preds.extend(preds)
            all_test_labels.extend(y_batch.cpu().numpy())
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # Save model
    model_path = os.path.join(args.output_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(args.output_dir, 'features.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to {features_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        np.array(all_test_labels),
        np.array(all_test_preds),
        cm_path,
        title="Test Set Confusion Matrix"
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

