"""
RobustScaler sonrası train/test dağılım farklarını kontrol et
"""
import numpy as np
import json
from pathlib import Path

arrays_dir = Path("processed_arrays")
art_dir = Path("artifacts")

# Veri yükle
X_train = np.load(arrays_dir / "X_train.npy")
X_test = np.load(arrays_dir / "X_test.npy")

with open(art_dir / "feature_names.json", "r") as f:
    feature_names = json.load(f)

print("="*70)
print("ROBUSTSCALER SONRASI TRAIN/TEST DAĞILIM KARŞILAŞTIRMASI")
print("="*70)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# Her feature için train/test mean ve std karşılaştırması
train_means = np.mean(X_train, axis=0)
test_means = np.mean(X_test, axis=0)
train_stds = np.std(X_train, axis=0)
test_stds = np.std(X_test, axis=0)

# Cohen's d hesapla
cohens_d_list = []
for i in range(len(feature_names)):
    pooled_std = np.sqrt((train_stds[i]**2 + test_stds[i]**2) / 2)
    if pooled_std > 0:
        cohens_d = abs((train_means[i] - test_means[i]) / pooled_std)
    else:
        cohens_d = 0
    cohens_d_list.append(cohens_d)

# En büyük farklar
print("\n🔍 EN BÜYÜK DAĞILIM FARKLARI (Cohen's d):")
print("-" * 70)
print(f"{'Feature':<40} {'Train Mean':<12} {'Test Mean':<12} {'Train Std':<12} {'Test Std':<12} {'Cohen d':<10}")
print("-" * 70)

sorted_indices = np.argsort(cohens_d_list)[::-1]
for i, idx in enumerate(sorted_indices[:20]):
    print(f"{feature_names[idx]:<40} {train_means[idx]:>12.4f} {test_means[idx]:>12.4f} "
          f"{train_stds[idx]:>12.4f} {test_stds[idx]:>12.4f} {cohens_d_list[idx]:>10.4f}")

# Özet istatistikler
large_diff_count = sum(1 for d in cohens_d_list if d > 0.5)
medium_diff_count = sum(1 for d in cohens_d_list if 0.2 < d <= 0.5)
small_diff_count = sum(1 for d in cohens_d_list if d <= 0.2)

print(f"\n📊 ÖZET:")
print(f"  Büyük fark (Cohen's d > 0.5): {large_diff_count} feature")
print(f"  Orta fark (0.2 < d <= 0.5):   {medium_diff_count} feature")
print(f"  Küçük fark (d <= 0.2):        {small_diff_count} feature")
print(f"  Ortalama Cohen's d:           {np.mean(cohens_d_list):.4f}")

print(f"\n💡 YORUM:")
if large_diff_count < 10:
    print("  ✅ Train/test dağılımları daha dengeli! RobustScaler işe yaradı.")
else:
    print("  ⚠️  Hala bazı feature'larda büyük farklar var, ama StandardScaler'dan daha iyi olmalı.")

