"""Veri hazırlık çıktılarını kontrol et"""
import json
import numpy as np
from pathlib import Path

# Load arrays
X_train = np.load("processed_arrays/X_train.npy")
y_train = np.load("processed_arrays/y_train.npy")
X_test = np.load("processed_arrays/X_test.npy")
y_test = np.load("processed_arrays/y_test.npy")

# Load artifacts
with open("artifacts/label_encoder.json") as f:
    label_encoder = json.load(f)
with open("artifacts/class_weights.json") as f:
    class_weights = json.load(f)
with open("artifacts/feature_names.json") as f:
    feature_names = json.load(f)

int_to_str = {int(k): v for k, v in label_encoder["int_to_str"].items()}

print("=" * 60)
print("VERİ HAZIRLIK ÇIKTILARI KONTROLÜ")
print("=" * 60)

print(f"\n📊 VERİ BOYUTLARI:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_test:  {y_test.shape}")

print(f"\n📋 ÖZELLİK SAYISI: {len(feature_names)}")
print(f"  İlk 5 özellik: {feature_names[:5]}")

print(f"\n🏷️  LABEL MAPPING:")
for i, name in sorted(int_to_str.items()):
    print(f"  {i} -> {name}")

print(f"\n⚖️  CLASS WEIGHTS:")
for i, weight in enumerate(class_weights):
    print(f"  {int_to_str[i]:15s}: {weight:.4f}")

print(f"\n📈 TRAIN SET LABEL DAĞILIMI:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {int_to_str[u]:15s}: {c:>10,} ({c/len(y_train)*100:>6.2f}%)")

print(f"\n📈 TEST SET LABEL DAĞILIMI:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {int_to_str[u]:15s}: {c:>10,} ({c/len(y_test)*100:>6.2f}%)")

print(f"\n✅ VERİ KONTROLÜ:")
print(f"  ✓ X_train dtype: {X_train.dtype}, min={X_train.min():.2f}, max={X_train.max():.2f}")
print(f"  ✓ X_test dtype:  {X_test.dtype}, min={X_test.min():.2f}, max={X_test.max():.2f}")
print(f"  ✓ y_train dtype: {y_train.dtype}, min={y_train.min()}, max={y_train.max()}")
print(f"  ✓ y_test dtype:  {y_test.dtype}, min={y_test.min()}, max={y_test.max()}")
print(f"  ✓ NaN kontrolü: X_train={np.isnan(X_train).sum()}, X_test={np.isnan(X_test).sum()}")
print(f"  ✓ Inf kontrolü: X_train={np.isinf(X_train).sum()}, X_test={np.isinf(X_test).sum()}")

print("\n" + "=" * 60)
print("✅ TÜM KONTROLLER TAMAMLANDI!")
print("=" * 60)

