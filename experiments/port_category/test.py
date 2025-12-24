"""
Destination Port Category model test scripti.
Tek bir CSV dosyası veya tüm test dosyaları üzerinde test yapar.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train import MLP, load_artifacts


def categorize_port(port: int) -> str:
    """Port numarasını kategoriye çevirir."""
    if pd.isna(port) or port < 0:
        return "WELL_KNOWN"
    
    port = int(port)
    if 0 <= port <= 1023:
        return "WELL_KNOWN"
    elif 1024 <= port <= 49151:
        return "REGISTERED"
    elif 49152 <= port <= 65535:
        return "DYNAMIC"
    else:
        return "WELL_KNOWN"


def preprocess_test_file(df: pd.DataFrame, feature_names: list, scaler_mean: np.ndarray, scaler_scale: np.ndarray):
    """Test dosyasını ön işleme tabi tutar."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Port kategorisini oluştur
    y_cat = df["Destination Port"].apply(categorize_port)
    
    # Feature'ları seç (Destination Port ve Label hariç)
    feature_cols = [c for c in df.columns if c not in ["Label", "Destination Port"]]
    
    # Sadece eğitimde kullanılan feature'ları al
    available_features = [f for f in feature_names if f in feature_cols]
    missing_features = [f for f in feature_names if f not in feature_cols]
    
    if missing_features:
        print(f"  ⚠️  Eksik feature'lar: {len(missing_features)}")
        # Eksik feature'lar için 0 değer ekle
        for f in missing_features:
            df[f] = 0
    
    # Numerik dönüşüm
    for col in feature_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Sonsuzları NaN yap
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)
    
    # NaN drop
    before = len(df)
    df = df.dropna(subset=feature_names)
    after = len(df)
    print(f"  ✓ Temizleme: {before - after} satır silindi, {after:,} satır kaldı")
    
    # X ve y'yi ayır
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = y_cat[df.index].to_numpy()
    
    # Scale (StandardScaler transform)
    X_scaled = (X - scaler_mean) / scaler_scale
    
    return X_scaled, y, df


def test_single_file(file_path: Path, model, device, artifacts_dir: Path, int_to_str: dict):
    """Tek bir dosya üzerinde test yapar."""
    print(f"\n{'='*70}")
    print(f"TEST DOSYASI: {file_path.name}")
    print(f"{'='*70}")
    
    # Artefaktları yükle
    mean, scale, _, label_maps, feature_names = load_artifacts(artifacts_dir)
    str_to_int = label_maps["str_to_int"]
    
    # CSV yükle
    print(f"\n📂 Dosya yükleniyor...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"  ✓ Yüklendi: {len(df):,} satır")
    
    # Preprocess
    print(f"\n🔄 Ön işleme yapılıyor...")
    X, y_true_str, df_clean = preprocess_test_file(df, feature_names, mean, scale)
    
    # Label encode
    y_true = np.array([str_to_int[c] for c in y_true_str], dtype=np.int64)
    
    print(f"\n📊 Veri istatistikleri:")
    unique, counts = np.unique(y_true_str, return_counts=True)
    for cat, count in zip(unique, counts):
        print(f"  {cat:15s}: {count:>10,} ({count/len(y_true)*100:>6.2f}%)")
    
    # Model inference
    print(f"\n🔮 Model tahminleri yapılıyor...")
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    
    y_pred = preds
    
    # Metrikler
    acc = (y_true == y_pred).mean()
    print(f"\n📈 SONUÇLAR:")
    print(f"  ✅ Accuracy: {acc*100:.2f}%")
    
    # Sınıf bazlı metrikler
    print(f"\n📊 Sınıf bazlı doğruluk:")
    for class_id in sorted(int_to_str.keys()):
        mask = y_true == class_id
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == class_id).mean()
            print(f"    {int_to_str[class_id]:15s}: {class_acc*100:6.2f}% ({mask.sum():,} örnek)")
    
    # Classification report
    target_names = [int_to_str[i] for i in sorted(int_to_str.keys())]
    labels = sorted(int_to_str.keys())
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0)
    
    print(f"\n📝 DETAYLI RAPOR:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Sonuçları kaydet
    results = {
        "file": file_path.name,
        "accuracy": float(acc),
        "total_samples": int(len(y_true)),
        "class_distribution": {int_to_str[i]: int(count) for i, count in zip(*np.unique(y_true, return_counts=True))},
        "class_accuracy": {int_to_str[i]: float((y_pred[y_true == i] == i).mean()) 
                          for i in sorted(int_to_str.keys()) if (y_true == i).sum() > 0}
    }
    
    return results, report, cm, y_true, y_pred, probs.cpu().numpy()


def plot_confusion_matrix(cm, int_to_str, save_path: Path):
    """Confusion matrix görselleştirir."""
    labels = [int_to_str[i] for i in sorted(int_to_str.keys())]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Port Category Classification')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix kaydedildi: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, default=None, 
                       help="Test edilecek tek bir CSV dosyası (tüm test dosyaları için boş bırakın)")
    parser.add_argument("--data-dir", type=str, default="../../data-CIC-IDS- 2017",
                       help="Test CSV dosyalarının bulunduğu dizin")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                       help="Artefakt dizini")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt",
                       help="Model checkpoint yolu")
    parser.add_argument("--output-dir", type=str, default="test_results",
                       help="Test sonuçlarının kaydedileceği dizin")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DESTINATION PORT CATEGORY - MODEL TEST")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n🚀 [DEVICE] MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🚀 [DEVICE] CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"\n🚀 [DEVICE] CPU")

    # Artefaktları yükle
    print(f"\n📦 ARTEFAKTLAR YÜKLENİYOR...")
    mean, scale, class_weights, label_maps, feature_names = load_artifacts(artifacts_dir)
    str_to_int = label_maps["str_to_int"]
    int_to_str = {int(k): v for k, v in label_maps["int_to_str"].items()}
    print(f"  ✓ {len(int_to_str)} sınıf, {len(feature_names)} feature")

    # Model yükle
    print(f"\n🧠 MODEL YÜKLENİYOR...")
    model = MLP(in_dim=len(feature_names), num_classes=len(int_to_str)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"  ✓ Model yüklendi: {checkpoint_path}")

    # Test dosyalarını belirle
    data_dir = Path(args.data_dir)
    if args.test_file:
        test_files = [data_dir / args.test_file]
    else:
        # Tüm test dosyaları
        test_files = [
            data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ]
        print(f"\n📂 Tüm test dosyaları test edilecek: {len(test_files)} dosya")

    # Her dosya için test yap
    all_results = []
    for test_file in test_files:
        if not test_file.exists():
            print(f"  ⚠️  Dosya bulunamadı: {test_file}")
            continue
        
        results, report, cm, y_true, y_pred, probs = test_single_file(
            test_file, model, device, artifacts_dir, int_to_str
        )
        all_results.append(results)
        
        # Sonuçları kaydet
        file_output_dir = output_dir / test_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Rapor kaydet
        with open(file_output_dir / "classification_report.txt", "w") as f:
            f.write(f"Test Dosyası: {test_file.name}\n")
            f.write("=" * 70 + "\n\n")
            f.write(report)
        
        # Confusion matrix kaydet
        np.save(file_output_dir / "confusion_matrix.npy", cm)
        plot_confusion_matrix(cm, int_to_str, file_output_dir / "confusion_matrix.png")
        
        # Metrikleri kaydet
        with open(file_output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  💾 Sonuçlar kaydedildi: {file_output_dir}/")

    # Özet rapor
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("ÖZET RAPOR (Tüm Test Dosyaları)")
        print(f"{'='*70}")
        total_samples = sum(r["total_samples"] for r in all_results)
        weighted_acc = sum(r["accuracy"] * r["total_samples"] for r in all_results) / total_samples
        
        print(f"\n📊 Toplam Test Örnekleri: {total_samples:,}")
        print(f"📊 Ağırlıklı Ortalama Accuracy: {weighted_acc*100:.2f}%")
        
        print(f"\n📁 Dosya Bazlı Sonuçlar:")
        for r in all_results:
            print(f"  {r['file']:50s} Accuracy: {r['accuracy']*100:6.2f}% ({r['total_samples']:>8,} örnek)")

    print(f"\n{'='*70}")
    print("✅ TEST TAMAMLANDI!")
    print(f"{'='*70}")
    print(f"📁 Sonuçlar: {output_dir}/")


if __name__ == "__main__":
    main()

