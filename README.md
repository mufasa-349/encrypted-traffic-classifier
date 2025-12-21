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

## License

This project is provided as-is for research and educational purposes.

