# Destination Port Category Classification

Bu experiment, CIC-IDS-2017 veri setini kullanarak **Destination Port** numaralarını 3 kategoriye ayıran bir neural network eğitir.

## Kategoriler

1. **WELL_KNOWN** (0-1023): Standart servis portları (HTTP, SSH, FTP, vb.)
2. **REGISTERED** (1024-49151): Kayıtlı uygulama portları
3. **DYNAMIC** (49152-65535): Geçici/özel portlar

## Klasör Yapısı

```
port_category/
├── artifacts/              # Scaler, weights, encoder çıktıları
├── processed_arrays/       # Numpy X/y çıktıları
├── checkpoints/            # Eğitilmiş model
├── reports/                # Eğitim raporları
├── test_results/           # Test sonuçları
├── prepare_data.py         # Veri hazırlama scripti
├── train.py                # Model eğitim scripti
├── test.py                 # Model test scripti
├── requirements.txt        # Python bağımlılıkları
└── setup_venv.sh          # Venv kurulum scripti
```

## Kurulum

### 1. Venv Kurulumu

```bash
cd experiments/port_category
chmod +x setup_venv.sh
./setup_venv.sh
```

Veya manuel:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Veri Hazırlama

```bash
source venv/bin/activate
python prepare_data.py --data-dir "../../data-CIC-IDS- 2017"
```

Bu komut:
- Train ve test dosyalarını yükler
- Destination Port'u 3 kategoriye ayırır
- Feature'ları temizler ve scale eder
- Numpy array'leri ve artefaktları kaydeder

### 3. Model Eğitimi

**Standart eğitim (Weighted CrossEntropyLoss):**
```bash
source venv/bin/activate
python train.py --epochs 30 --batch-size 512 --lr 0.001
```

**Focal Loss ile eğitim (dengesiz veri için önerilir):**
```bash
source venv/bin/activate
python train.py --epochs 30 --batch-size 512 --lr 0.001 --use-focal-loss --focal-gamma 2.0
```

Parametreler:
- `--epochs`: Eğitim epoch sayısı (default: 30)
- `--batch-size`: Batch size (default: 512)
- `--lr`: Learning rate (default: 0.001)
- `--early-stopping`: Early stopping patience (default: 10)
- `--hidden-dims`: Hidden layer boyutları (default: "128,64")
- `--use-focal-loss`: Focal Loss kullan (dengesiz veri için önerilir)
- `--focal-gamma`: Focal Loss gamma parametresi (default: 2.0, yüksek = zor örneklere daha fazla odaklan)

### 4. Model Testi

#### Tek bir dosya üzerinde test:

```bash
source venv/bin/activate
python test.py --test-file "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" \
               --data-dir "../../data-CIC-IDS- 2017" \
               --checkpoint "checkpoints/model.pt"
```

#### Tüm test dosyaları üzerinde test:

```bash
source venv/bin/activate
python test.py --data-dir "../../data-CIC-IDS- 2017" \
               --checkpoint "checkpoints/model.pt"
```

Test sonuçları `test_results/` klasörüne kaydedilir:
- `classification_report.txt`: Detaylı sınıflandırma raporu
- `confusion_matrix.npy`: Confusion matrix (numpy)
- `confusion_matrix.png`: Confusion matrix görseli
- `metrics.json`: Metrikler (JSON)

## Çıktılar

### Eğitim Sonrası

- `checkpoints/model.pt`: Eğitilmiş model
- `reports/classification_report.txt`: Eğitim sonuçları
- `reports/confusion_matrix.npy`: Confusion matrix

### Test Sonrası

Her test dosyası için ayrı klasör:
- `test_results/{dosya_adi}/classification_report.txt`
- `test_results/{dosya_adi}/confusion_matrix.png`
- `test_results/{dosya_adi}/metrics.json`

## Model Mimarisi

- **Tip**: Multi-layer Perceptron (MLP)
- **Sınıf Sayısı**: 3 (WELL_KNOWN, REGISTERED, DYNAMIC)
- **Feature Sayısı**: 77 (Destination Port hariç tüm feature'lar)
- **Hidden Layers**: [128, 64] (varsayılan)
- **Loss**: Weighted CrossEntropyLoss
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau

## Özellikler

- ✅ Dosya bazlı train/test split
- ✅ StandardScaler ile feature scaling
- ✅ Class weights ile dengesiz veri yönetimi
- ✅ Early stopping ile overfitting önleme
- ✅ Detaylı metrikler ve görselleştirme
- ✅ Tek dosya veya toplu test desteği

## Kullanım Senaryoları

1. **Servis Tanıma**: Port numarasından hangi servisin kullanıldığını tahmin etme
2. **Trafik Analizi**: Port kategorilerine göre trafik pattern'lerini anlama
3. **Güvenlik**: Şüpheli port kullanımlarını tespit etme
4. **Network Monitoring**: Port bazlı trafik sınıflandırması

