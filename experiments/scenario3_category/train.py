"""
Senaryo 3: 6 sınıflı MLP eğitimi.
- Girdi: prepare_data.py ile üretilen numpy dosyaları ve artefaktlar
- Loss: Weighted CrossEntropy
- Metrikler: accuracy, sınıf bazlı precision/recall/F1, confusion matrix
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 6, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),  # Batch normalization eklendi
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch normalization eklendi
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Batch normalization eklendi
            nn.ReLU(),
            nn.Dropout(p * 0.75),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_artifacts(art_dir: Path):
    mean = np.load(art_dir / "scaler_mean.npy")
    scale = np.load(art_dir / "scaler_scale.npy")
    with open(art_dir / "class_weights.json") as f:
        class_weights = json.load(f)
    with open(art_dir / "label_encoder.json") as f:
        label_maps = json.load(f)
    with open(art_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    return mean, scale, class_weights, label_maps, feature_names


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)
    
    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)
        
        # Her 100 batch'te bir ilerleme göster
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            progress = (batch_idx + 1) / num_batches * 100
            avg_loss = total_loss / total_samples
            print(f"  [Epoch {epoch}/{total_epochs}] Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) - Loss: {avg_loss:.4f}")
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, int_to_str=None):
    model.eval()
    all_preds, all_labels = [], []
    print(f"  [EVAL] Test seti değerlendiriliyor...")
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
            
            # Her 50 batch'te bir ilerleme göster
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx+1}/{len(loader)} işlendi...")
    
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    
    # Sınıf bazlı accuracy hesapla
    if int_to_str:
        print(f"  [EVAL] Sınıf bazlı doğruluk:")
        for class_id in sorted(int_to_str.keys()):
            mask = y_true == class_id
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == class_id).mean()
                print(f"    {int_to_str[class_id]:15s}: {class_acc*100:6.2f}% ({mask.sum():,} örnek)")
    
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="prepare_data çıktıları")
    parser.add_argument("--arrays-dir", type=str, default="processed_arrays", help="numpy X/y dizin")
    parser.add_argument("--output-dir", type=str, default="", help="Checkpoint ve report çıktı dizini (boşsa default: checkpoints/ ve reports/)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)  # Düşürüldü: 1e-3 -> 1e-4
    parser.add_argument("--weight-decay", type=float, default=1e-4)  # Artırıldı: 1e-5 -> 1e-4
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping", type=int, default=15, help="Patience for early stopping (0 = disabled)")
    parser.add_argument("--normalize-weights", action="store_true", help="Normalize class weights (max=5.0)")
    args = parser.parse_args()

    art_dir = Path(args.artifacts_dir)
    arrays_dir = Path(args.arrays_dir)

    print("=" * 70)
    print("SENARYO 3: 6 SINIFLI MLP EĞİTİMİ")
    print("=" * 70)
    print(f"\n📋 EĞİTİM PARAMETRELERİ:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay:  {args.weight_decay}")
    print(f"  Num Workers:   {args.num_workers}")
    
    print(f"\n📦 ARTEFAKTLAR YÜKLENİYOR...")
    _, _, class_weights, label_maps, feature_names = load_artifacts(art_dir)
    str_to_int = label_maps["str_to_int"]
    int_to_str = {int(k): v for k, v in label_maps["int_to_str"].items()}
    print(f"  ✓ Label encoder yüklendi: {len(str_to_int)} sınıf")
    print(f"  ✓ Class weights yüklendi: {len(class_weights)} sınıf")
    print(f"  ✓ Feature names yüklendi: {len(feature_names)} özellik")
    
    print(f"\n📊 VERİ YÜKLENİYOR...")
    X_train = np.load(arrays_dir / "X_train.npy")
    y_train = np.load(arrays_dir / "y_train.npy")
    X_test = np.load(arrays_dir / "X_test.npy")
    y_test = np.load(arrays_dir / "y_test.npy")
    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ y_train: {y_train.shape}")
    print(f"  ✓ X_test:  {X_test.shape}")
    print(f"  ✓ y_test:  {y_test.shape}")
    
    # Train seti label dağılımı
    print(f"\n📈 TRAIN SET LABEL DAĞILIMI:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {int_to_str[u]:15s}: {c:>10,} ({c/len(y_train)*100:>6.2f}%)")
    
    # Test seti label dağılımı
    print(f"\n📈 TEST SET LABEL DAĞILIMI:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {int_to_str[u]:15s}: {c:>10,} ({c/len(y_test)*100:>6.2f}%)")

    # Device seçimi (MPS öncelikli)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n🚀 [DEVICE] MPS (Apple Silicon GPU) kullanılıyor")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🚀 [DEVICE] CUDA (NVIDIA GPU) kullanılıyor")
    else:
        device = torch.device("cpu")
        print(f"\n🚀 [DEVICE] CPU kullanılıyor")

    print(f"\n🔄 DATASET VE DATALOADER OLUŞTURULUYOR...")
    train_ds = NPDataset(X_train, y_train)
    test_ds = NPDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Test batches:  {len(test_loader)}")

    print(f"\n🧠 MODEL OLUŞTURULUYOR...")
    model = MLP(in_dim=X_train.shape[1], num_classes=len(str_to_int)).to(device)
    
    # Model parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model oluşturuldu: {total_params:,} parametre ({trainable_params:,} trainable)")
    print(f"  ✓ Model device: {next(model.parameters()).device}")
    
    print(f"\n⚖️  LOSS VE OPTIMIZER AYARLANIYOR...")
    
    # Class weights normalize et (çok aşırı olanları sınırla)
    if args.normalize_weights:
        max_weight = 5.0  # Maksimum weight (10'dan 5'e düşürüldü - daha dengeli)
        normalized_weights = [min(w, max_weight) for w in class_weights]
        print(f"  ⚠️  Class weights normalize ediliyor (max={max_weight})...")
        print(f"  ✓ Normalize öncesi:")
        for i, w in enumerate(class_weights):
            print(f"      {int_to_str[i]:15s}: {w:.4f}")
        print(f"  ✓ Normalize sonrası:")
        for i, w in enumerate(normalized_weights):
            print(f"      {int_to_str[i]:15s}: {w:.4f}")
        class_weights = normalized_weights
    else:
        print(f"  ✓ Class weights (normalize edilmedi):")
        for i, w in enumerate(class_weights):
            print(f"      {int_to_str[i]:15s}: {w:.4f}")
    
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler ekle
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )
    
    print(f"  ✓ Loss: Weighted CrossEntropyLoss")
    print(f"  ✓ Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  ✓ Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Output dizinleri (paralel çalıştırma için)
    if args.output_dir:
        output_base = Path(args.output_dir)
        ckpt_dir = output_base / "checkpoints"
        reports_dir = output_base / "reports"
    else:
        ckpt_dir = Path("checkpoints")
        reports_dir = Path("reports")
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "=" * 70)
    print("EĞİTİM BAŞLIYOR...")
    print("=" * 70)
    if args.early_stopping > 0:
        print(f"⏰ Early stopping aktif: {args.early_stopping} epoch patience")
    else:
        print(f"⏰ Early stopping devre dışı - tüm epoch'lar eğitilecek")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─' * 70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'─' * 70}")
        
        # Train
        print(f"[TRAIN] Epoch {epoch} eğitimi başlıyor...")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        print(f"[TRAIN] Epoch {epoch} tamamlandı - Loss: {train_loss:.4f}")
        
        # Evaluate
        print(f"[TEST]  Epoch {epoch} değerlendirmesi başlıyor...")
        y_true, y_pred = evaluate(model, test_loader, device, int_to_str)
        acc = (y_true == y_pred).mean()
        print(f"[TEST]  Epoch {epoch} tamamlandı - Accuracy: {acc*100:.2f}%")
        
        # Learning rate scheduler update
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < old_lr:
            print(f"  📉 Learning rate azaltıldı: {old_lr:.6f} → {current_lr:.6f}")
        elif epoch > 1:
            print(f"  📉 Learning rate: {current_lr:.6f}")
        
        # En iyi modeli kaydet
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            patience_counter = 0  # Reset patience
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            print(f"  ⭐ YENİ EN İYİ MODEL! Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})")
            print(f"  💾 Model kaydedildi: {ckpt_dir / 'model.pt'}")
        else:
            patience_counter += 1
            print(f"  📊 Mevcut en iyi: {best_acc*100:.2f}% (Epoch {best_epoch})")
            if args.early_stopping > 0:
                print(f"  ⏰ Patience: {patience_counter}/{args.early_stopping}")
        
        # Early stopping kontrolü
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\n⏹️  EARLY STOPPING! {args.early_stopping} epoch boyunca iyileşme olmadı.")
            print(f"   En iyi model: Epoch {best_epoch} - Accuracy: {best_acc*100:.2f}%")
            break

    print(f"\n" + "=" * 70)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 70)
    print(f"📊 En iyi model: Epoch {best_epoch} - Accuracy: {best_acc*100:.2f}%")
    
    # Son değerlendirme (best model ile)
    print(f"\n🔄 EN İYİ MODEL YÜKLENİYOR...")
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    print(f"  ✓ Model yüklendi: {ckpt_dir / 'model.pt'}")
    
    print(f"\n📊 FİNAL DEĞERLENDİRME (En iyi model ile)...")
    y_true, y_pred = evaluate(model, test_loader, device, int_to_str)
    acc = (y_true == y_pred).mean()
    print(f"\n  ✅ Final Accuracy: {acc*100:.2f}%")

    # Rapor
    print(f"\n📝 DETAYLI RAPOR OLUŞTURULUYOR...")
    
    # Test setinde bulunan sınıfları belirle
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [int_to_str[i] for i in sorted(unique_labels)]
    labels = sorted(unique_labels)
    
    # Tüm sınıfları içeren confusion matrix için
    all_labels = sorted(int_to_str.keys())
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Sadece test setinde bulunan sınıflar için rapor
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0)
    
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write("SENARYO 3: 6 SINIFLI MLP - DETAYLI RAPOR\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"En iyi model: Epoch {best_epoch}\n")
        f.write(f"Final Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    np.save(reports_dir / "confusion_matrix.npy", cm)
    
    print(f"  ✓ Classification report: {reports_dir / 'classification_report.txt'}")
    print(f"  ✓ Confusion matrix: {reports_dir / 'confusion_matrix.npy'}")
    
    print(f"\n" + "=" * 70)
    print("DETAYLI RAPOR:")
    print("=" * 70)
    print(report)
    
    print(f"\n" + "=" * 70)
    print("✅ TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 70)
    print(f"📁 Model:     {ckpt_dir / 'model.pt'}")
    print(f"📁 Raporlar:  {reports_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

