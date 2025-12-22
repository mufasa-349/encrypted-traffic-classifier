"""
Scaler'ı kontrol et - hangi feature'ların scale'i 0 veya çok küçük?
"""
import numpy as np
import json
from pathlib import Path

art_dir = Path("artifacts")

# Scaler parametrelerini yükle
scaler_mean = np.load(art_dir / "scaler_mean.npy")
scaler_scale = np.load(art_dir / "scaler_scale.npy")

with open(art_dir / "feature_names.json", "r") as f:
    feature_names = json.load(f)

print("="*70)
print("SCALER ANALİZİ")
print("="*70)

# Scale değerlerini kontrol et
zero_scale_indices = np.where(scaler_scale == 0)[0]
very_small_scale_indices = np.where((scaler_scale > 0) & (scaler_scale < 1e-6))[0]

print(f"\n📊 Toplam feature sayısı: {len(feature_names)}")
print(f"⚠️  Scale = 0 olan feature'lar: {len(zero_scale_indices)}")
print(f"⚠️  Scale < 1e-6 olan feature'lar: {len(very_small_scale_indices)}")

if len(zero_scale_indices) > 0:
    print("\n🔴 SCALE = 0 OLAN FEATURE'LAR (Bu feature'lar train setinde sabit değer):")
    print("-" * 70)
    for idx in zero_scale_indices:
        print(f"  {idx:2d}. {feature_names[idx]:<40} mean={scaler_mean[idx]:.6f} scale=0.000000")

if len(very_small_scale_indices) > 0:
    print("\n🟡 SCALE ÇOK KÜÇÜK OLAN FEATURE'LAR (< 1e-6):")
    print("-" * 70)
    for idx in very_small_scale_indices:
        print(f"  {idx:2d}. {feature_names[idx]:<40} mean={scaler_mean[idx]:.6f} scale={scaler_scale[idx]:.2e}")

# En küçük scale değerleri
print("\n📉 EN KÜÇÜK 10 SCALE DEĞERİ:")
print("-" * 70)
sorted_indices = np.argsort(scaler_scale)
for i, idx in enumerate(sorted_indices[:10]):
    print(f"  {i+1:2d}. {feature_names[idx]:<40} scale={scaler_scale[idx]:.6f} mean={scaler_mean[idx]:.6f}")

# Train ve test verilerini yükle ve karşılaştır
print("\n" + "="*70)
print("TRAIN vs TEST KARŞILAŞTIRMASI (Scaled)")
print("="*70)

arrays_dir = Path("processed_arrays")
X_train = np.load(arrays_dir / "X_train.npy")
X_test = np.load(arrays_dir / "X_test.npy")

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")

# Her feature için train/test mean karşılaştırması
print("\n🔍 TRAIN MEAN ≈ 0 AMA TEST MEAN FARKLI OLAN FEATURE'LAR:")
print("-" * 70)
print(f"{'Feature':<40} {'Train Mean':<15} {'Test Mean':<15} {'Train Std':<15} {'Test Std':<15}")
print("-" * 70)

train_means = np.mean(X_train, axis=0)
test_means = np.mean(X_test, axis=0)
train_stds = np.std(X_train, axis=0)
test_stds = np.std(X_test, axis=0)

suspicious_features = []
for i, feat_name in enumerate(feature_names):
    train_mean_abs = abs(train_means[i])
    test_mean_abs = abs(test_means[i])
    
    # Train mean çok küçük ama test mean farklı
    if train_mean_abs < 0.01 and test_mean_abs > 0.1:
        suspicious_features.append(i)
        print(f"{feat_name:<40} {train_means[i]:>15.6f} {test_means[i]:>15.6f} "
              f"{train_stds[i]:>15.6f} {test_stds[i]:>15.6f}")

if len(suspicious_features) == 0:
    print("  (Böyle bir feature bulunamadı)")

print(f"\n💡 SONUÇ:")
print(f"  - {len(zero_scale_indices)} feature'ın scale'i 0 (train setinde sabit değer)")
print(f"  - Bu feature'lar scaler transform sonrası train'de 0, test'te farklı değerler alır")
print(f"  - Model bu feature'lara bakarak train/test'i ayırt edebilir (DATA LEAKAGE!)")
print(f"\n🔧 ÇÖZÜM:")
print(f"  1. Scale=0 olan feature'ları train/test'ten çıkar")
print(f"  2. Veya bu feature'ları manuel olarak 0'a set et (test'te de)")

