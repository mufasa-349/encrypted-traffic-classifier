# 1D-CNN Baseline for CIC-IDS2017

Binary classification (BENIGN vs ATTACK) using a 1D-CNN on flow-based CSV features from the CIC-IDS2017 dataset.

## Dataset

The dataset CSV files should be placed in `data/cicids2017/` directory. The following files are used:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model with file-based split (default):

```bash
python -m src.train --data_dir data/cicids2017 --split_by_file 1
```

Train with random split:

```bash
python -m src.train --data_dir data/cicids2017 --split_by_file 0
```

Additional training options:

```bash
python -m src.train \
    --data_dir data/cicids2017 \
    --split_by_file 1 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.001 \
    --patience 5 \
    --seed 42 \
    --output_dir runs
```

### Evaluation

Evaluate a trained model:

```bash
python -m src.eval --data_dir data/cicids2017 --ckpt runs/model.pt
```

## Project Structure

```
.
├── data/
│   └── cicids2017/          # CSV dataset files
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading, cleaning, splitting
│   ├── model.py             # 1D-CNN model architecture
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   └── utils.py             # Utilities (metrics, plotting, seeding)
├── runs/                    # Output directory (created automatically)
│   ├── model.pt             # Best model checkpoint
│   ├── scaler.pkl           # Fitted StandardScaler
│   ├── features.json        # Feature names
│   ├── confusion_matrix.png # Confusion matrix plot
│   └── test_metrics.json    # Test set metrics
├── requirements.txt
└── README.md
```

## Model Architecture

The 1D-CNN treats the feature vector as a 1D signal:

- Input: `(batch, 1, num_features)`
- Three Conv1D layers with BatchNorm and MaxPooling
- Global Average Pooling
- Two fully connected layers with dropout
- Output: Binary classification (BENIGN=0, ATTACK=1)

## Features

- **Data Cleaning**: Automatic handling of missing values, infinity values, and non-numeric columns
- **Class Imbalance**: Handled using `WeightedRandomSampler` and `BCEWithLogitsLoss` with `pos_weight`
- **Early Stopping**: Based on validation ROC-AUC with configurable patience
- **Reproducibility**: Random seed setting for all random operations
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrix saved as PNG

## Outputs

After training, the following files are saved to `runs/`:

- `model.pt`: Best model checkpoint (based on validation AUC)
- `scaler.pkl`: Fitted StandardScaler for feature normalization
- `features.json`: List of feature names used
- `confusion_matrix.png`: Confusion matrix visualization
- `test_metrics.json`: Test set performance metrics

## Data Split

**Default (file-based split):**
- **Train**: Monday + Tuesday + Wednesday + Thursday files
- **Test**: Friday files

**Random split:**
- 80% train, 20% test (stratified)

A 10% validation split is created from the training data for early stopping.

## Dataset Structure & Attack Types

**Training Set:**
- Monday: 529,481 samples (100% BENIGN)
- Tuesday: 445,645 samples (96.9% BENIGN, 3.1% FTP-Patator + SSH-Patator)
- Wednesday: 691,406 samples (63.6% BENIGN, 36.4% DoS attacks: Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed)
- Thursday Morning: 170,231 samples (98.7% BENIGN, 1.3% Web Attacks: Brute Force, XSS, SQL Injection)
- Thursday Afternoon: 288,395 samples (99.99% BENIGN, 0.01% Infiltration)
- Friday Morning: 190,911 samples (98.97% BENIGN, 1.03% ATTACK)

**Test Set (Not in Training):**
- Friday Afternoon DDos: 225,711 samples (43.3% BENIGN, 56.7% DDoS)
- Friday Afternoon PortScan: 286,096 samples (44.5% BENIGN, 55.5% PortScan)

## Test Results & Analysis

### Performance Summary

| Test File | Attack Type | Recall | Precision | F1-Score | ROC-AUC | Status |
|-----------|------------|--------|-----------|----------|---------|--------|
| Friday DDos | DDoS | 0.0956 | 0.8778 | 0.1724 | 0.7513 | ❌ Poor |
| Friday PortScan | PortScan | 0.0000 | 0.0000 | 0.0000 | 0.3819 | ❌ Failed |
| Monday | BENIGN only | - | - | - | - | ✅ Good (99.98% accuracy) |
| Infiltration | Infiltration | 0.4167 | 0.1190 | 0.1852 | 0.9583 | ⚠️ Moderate |
| WebAttacks | Web Attacks | 0.0000 | 0.0000 | 0.0000 | 0.8847 | ❌ Failed |
| Tuesday | Patator | 0.0000 | 0.0000 | 0.0000 | 0.8406 | ❌ Failed |
| Wednesday | DoS | 0.3970 | 0.9967 | 0.5678 | 0.9554 | ✅ Best |

### Key Findings

**1. Domain Shift Problem:**
- **DDoS ve PortScan** saldırıları eğitim setinde yok → Model hiç tanımıyor
- PortScan: Recall 0% (158,804 saldırıdan hiçbiri yakalanmadı)
- DDoS: Recall 9.56% (128,025 saldırıdan sadece 12,228'ini yakaladı)

**2. Threshold Issue:**
- Model 0.5 threshold ile çoğu örneği BENIGN olarak tahmin ediyor
- ROC-AUC değerleri yüksek (0.75-0.96) ama threshold optimize edilmemiş
- Örnek: WebAttacks için ROC-AUC 0.88 ama Recall 0% (threshold çok yüksek)

**3. Training Set Performance:**
- **Wednesday (DoS)**: En iyi performans (Recall 39.7%, Precision 99.67%)
- **Infiltration**: Orta performans (Recall 41.67% ama Precision düşük 11.9%)
- **Tuesday, WebAttacks**: Eğitim setinde olmasına rağmen Recall 0%

**4. Root Causes:**
- **Eğitim setinde olmayan saldırı türleri**: DDoS ve PortScan hiç tanınmıyor
- **Threshold optimizasyonu yok**: 0.5 threshold tüm test setleri için uygun değil
- **Class imbalance**: Eğitim setinde çok dengesiz dağılım (Monday %100 BENIGN, Infiltration %0.01 saldırı)
- **Model bias**: Model çoğu örneği BENIGN olarak tahmin etmeye eğilimli

### Recommendations

1. **Threshold Optimization**: ROC eğrisinden optimal threshold bulun (her saldırı türü için farklı olabilir)
2. **Data Augmentation**: Eğitim setine DDoS ve PortScan örnekleri ekleyin
3. **Multi-class Classification**: Binary yerine multi-class (her saldırı türü için ayrı sınıf)
4. **Ensemble Methods**: Farklı modelleri birleştirin
5. **Feature Engineering**: Saldırı türüne özel özellikler ekleyin

## License

This project is provided as-is for research and educational purposes.

