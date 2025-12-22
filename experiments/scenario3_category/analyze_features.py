"""
Feature Importance ve Data Leakage Analizi
Modelin hangi feature'lara baktığını ve train/test farklarını analiz eder.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score
from scipy import stats

def load_artifacts(art_dir: Path):
    """Artifacts yükle"""
    with open(art_dir / "label_encoder.json", "r") as f:
        label_maps = json.load(f)
    with open(art_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    return label_maps, feature_names

def load_model_and_data(art_dir: Path, arrays_dir: Path, device):
    """Model ve veriyi yükle"""
    # Model yükle
    class MLP(nn.Module):
        def __init__(self, in_dim: int, num_classes: int = 6, p: float = 0.4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p * 0.75),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            return self.net(x)
    
    label_maps, feature_names = load_artifacts(art_dir)
    num_features = len(feature_names)
    
    model = MLP(num_features, num_classes=6)
    model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Veri yükle
    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")
    
    return model, X_train, y_train, X_test, y_test, feature_names, label_maps

def compute_feature_importance_gradient(model, X_sample, y_sample, feature_names, device):
    """Gradient-based feature importance"""
    X_tensor = torch.tensor(X_sample, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(y_sample, dtype=torch.long, device=device)
    
    model.eval()
    output = model(X_tensor)
    loss = nn.CrossEntropyLoss()(output, y_tensor)
    loss.backward()
    
    # Gradient'ların mutlak değerlerini al
    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    
    # Feature importance olarak normalize et
    importance = gradients / gradients.sum()
    
    return importance

def compute_feature_importance_permutation(model, X_sample, y_sample, feature_names, device, n_samples=1000):
    """Permutation-based feature importance"""
    X_tensor = torch.tensor(X_sample[:n_samples], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_sample[:n_samples], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        baseline_output = model(X_tensor)
        baseline_pred = torch.argmax(baseline_output, dim=1)
        baseline_acc = (baseline_pred == y_tensor).float().mean().item()
    
    importance = np.zeros(len(feature_names))
    
    for i, feat_name in enumerate(feature_names):
        X_permuted = X_tensor.clone()
        # Feature'ı shuffle et
        perm_indices = torch.randperm(n_samples)
        X_permuted[:, i] = X_permuted[perm_indices, i]
        
        with torch.no_grad():
            permuted_output = model(X_permuted)
            permuted_pred = torch.argmax(permuted_output, dim=1)
            permuted_acc = (permuted_pred == y_tensor).float().mean().item()
        
        # Importance = accuracy drop
        importance[i] = baseline_acc - permuted_acc
    
    return importance

def compare_train_test_distributions(X_train, y_train, X_test, y_test, feature_names):
    """Train ve test setlerindeki feature dağılımlarını karşılaştır"""
    print("\n" + "="*70)
    print("TRAIN vs TEST FEATURE DAĞILIMI KARŞILAŞTIRMASI")
    print("="*70)
    
    results = []
    
    for i, feat_name in enumerate(feature_names):
        train_feat = X_train[:, i]
        test_feat = X_test[:, i]
        
        # İstatistikler
        train_mean = np.mean(train_feat)
        test_mean = np.mean(test_feat)
        train_std = np.std(train_feat)
        test_std = np.std(test_feat)
        
        # Kolmogorov-Smirnov testi (dağılım farkı)
        ks_stat, ks_pvalue = stats.ks_2samp(train_feat, test_feat)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((train_std**2 + test_std**2) / 2)
        cohens_d = (train_mean - test_mean) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'feature': feat_name,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'mean_diff': abs(train_mean - test_mean),
            'train_std': train_std,
            'test_std': test_std,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'cohens_d': abs(cohens_d),
            'significant': ks_pvalue < 0.01  # %1 anlamlılık
        })
    
    # En büyük farklara göre sırala
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cohens_d', ascending=False)
    
    print("\n🔍 EN BÜYÜK FARKLAR (Cohen's d):")
    print("-" * 70)
    print(f"{'Feature':<40} {'Train Mean':<12} {'Test Mean':<12} {'Diff':<10} {'Cohen d':<10} {'KS p-value':<12}")
    print("-" * 70)
    
    for _, row in results_df.head(20).iterrows():
        sig_marker = "***" if row['significant'] else ""
        print(f"{row['feature']:<40} {row['train_mean']:>12.4f} {row['test_mean']:>12.4f} "
              f"{row['mean_diff']:>10.4f} {row['cohens_d']:>10.4f} {row['ks_pvalue']:>12.6f} {sig_marker}")
    
    return results_df

def analyze_label_feature_correlation(X_train, y_train, feature_names):
    """Label ile feature'lar arasındaki korelasyonu analiz et"""
    print("\n" + "="*70)
    print("LABEL-FEATURE KORELASYON ANALİZİ")
    print("="*70)
    
    correlations = []
    
    for i, feat_name in enumerate(feature_names):
        feat_values = X_train[:, i]
        
        # Mutual Information (non-linear korelasyon)
        mi = mutual_info_score(feat_values, y_train)
        
        # Pearson korelasyon (linear)
        if np.std(feat_values) > 0:
            pearson_r, pearson_p = stats.pearsonr(feat_values, y_train)
        else:
            pearson_r, pearson_p = 0, 1
        
        correlations.append({
            'feature': feat_name,
            'mutual_info': mi,
            'pearson_r': abs(pearson_r),
            'pearson_p': pearson_p
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('mutual_info', ascending=False)
    
    print("\n🔍 EN YÜKSEK MUTUAL INFORMATION:")
    print("-" * 70)
    print(f"{'Feature':<40} {'Mutual Info':<15} {'Pearson r':<12} {'Pearson p':<12}")
    print("-" * 70)
    
    for _, row in corr_df.head(20).iterrows():
        sig_marker = "***" if row['pearson_p'] < 0.01 else ""
        print(f"{row['feature']:<40} {row['mutual_info']:>15.6f} {row['pearson_r']:>12.4f} "
              f"{row['pearson_p']:>12.6f} {sig_marker}")
    
    return corr_df

def main():
    art_dir = Path("artifacts")
    arrays_dir = Path("processed_arrays")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model ve veri yükle
    print("\n📦 Model ve veri yükleniyor...")
    model, X_train, y_train, X_test, y_test, feature_names, label_maps = load_model_and_data(
        art_dir, arrays_dir, device
    )
    
    print(f"  ✓ Train: {X_train.shape}")
    print(f"  ✓ Test:  {X_test.shape}")
    
    # 1. Train/Test dağılım karşılaştırması
    dist_comparison = compare_train_test_distributions(X_train, y_train, X_test, y_test, feature_names)
    
    # 2. Label-Feature korelasyon analizi
    corr_analysis = analyze_label_feature_correlation(X_train, y_train, feature_names)
    
    # 3. Gradient-based feature importance
    print("\n" + "="*70)
    print("GRADIENT-BASED FEATURE IMPORTANCE")
    print("="*70)
    print("Hesaplanıyor (1000 sample)...")
    
    sample_size = min(1000, len(X_train))
    sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train[sample_indices]
    y_sample = y_train[sample_indices]
    
    grad_importance = compute_feature_importance_gradient(model, X_sample, y_sample, feature_names, device)
    
    grad_df = pd.DataFrame({
        'feature': feature_names,
        'importance': grad_importance
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 EN ÖNEMLİ FEATURE'LAR (Gradient-based):")
    print("-" * 70)
    print(f"{'Feature':<40} {'Importance':<15}")
    print("-" * 70)
    for _, row in grad_df.head(20).iterrows():
        print(f"{row['feature']:<40} {row['importance']:>15.6f}")
    
    # 4. Permutation-based feature importance (daha yavaş, daha az sample)
    print("\n" + "="*70)
    print("PERMUTATION-BASED FEATURE IMPORTANCE")
    print("="*70)
    print("Hesaplanıyor (500 sample, bu biraz zaman alabilir)...")
    
    perm_sample_size = min(500, len(X_test))
    perm_indices = np.random.choice(len(X_test), perm_sample_size, replace=False)
    X_perm_sample = X_test[perm_indices]
    y_perm_sample = y_test[perm_indices]
    
    perm_importance = compute_feature_importance_permutation(
        model, X_perm_sample, y_perm_sample, feature_names, device, n_samples=perm_sample_size
    )
    
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 EN ÖNEMLİ FEATURE'LAR (Permutation-based):")
    print("-" * 70)
    print(f"{'Feature':<40} {'Importance':<15}")
    print("-" * 70)
    for _, row in perm_df.head(20).iterrows():
        print(f"{row['feature']:<40} {row['importance']:>15.6f}")
    
    # Sonuçları kaydet
    print("\n💾 Sonuçlar kaydediliyor...")
    results_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)
    
    dist_comparison.to_csv(results_dir / "train_test_distribution_comparison.csv", index=False)
    corr_analysis.to_csv(results_dir / "label_feature_correlation.csv", index=False)
    grad_df.to_csv(results_dir / "gradient_feature_importance.csv", index=False)
    perm_df.to_csv(results_dir / "permutation_feature_importance.csv", index=False)
    
    print(f"  ✓ {results_dir / 'train_test_distribution_comparison.csv'}")
    print(f"  ✓ {results_dir / 'label_feature_correlation.csv'}")
    print(f"  ✓ {results_dir / 'gradient_feature_importance.csv'}")
    print(f"  ✓ {results_dir / 'permutation_feature_importance.csv'}")
    
    print("\n" + "="*70)
    print("✅ ANALİZ TAMAMLANDI!")
    print("="*70)
    print("\n💡 ÖNERİLER:")
    print("  1. Train/Test dağılım farklarına bakın (cohen's d > 0.5 = büyük fark)")
    print("  2. Yüksek mutual information'a sahip feature'lar gerçekten önemli olabilir")
    print("  3. Gradient ve Permutation importance'ları karşılaştırın")
    print("  4. Büyük farklar data leakage gösterebilir!")

if __name__ == "__main__":
    main()

