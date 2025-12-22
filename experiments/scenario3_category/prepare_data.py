"""
Senaryo 3 veri hazırlık scripti.
- Dosya bazlı split (train 6 dosya, test 2 dosya)
- 6 kategorili label mapping
- StandardScaler fit/transform
- Class weight hesaplama
- Numpy çıktıları ve artefaktlar (scaler, class weights, label encoder, feature names)
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# 6-kategori mapping
CATEGORY_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DOS_DDOS",
    "DoS GoldenEye": "DOS_DDOS",
    "DoS slowloris": "DOS_DDOS",
    "DoS Slowhttptest": "DOS_DDOS",
    "DDoS": "DOS_DDOS",
    "FTP-Patator": "BRUTEFORCE",
    "SSH-Patator": "BRUTEFORCE",
    "Web Attack � Brute Force": "BRUTEFORCE",
    "Web Attack � XSS": "WEB_ATTACKS",
    "Web Attack � Sql Injection": "WEB_ATTACKS",
    "PortScan": "RECON",
    "Bot": "OTHER",
    "Infiltration": "OTHER",
    "Heartbleed": "OTHER",
}

# Tüm olası kategoriler (sıralı, sabit) - train/test fark etmeksizin
ALL_CATEGORIES = ["BENIGN", "BRUTEFORCE", "DOS_DDOS", "OTHER", "RECON", "WEB_ATTACKS"]

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",  # RECON kategorisi için train'e eklendi
]

TEST_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",  # Sadece DDoS test için
]


def load_csv(path: Path) -> pd.DataFrame:
    print(f"[LOAD] {path.name}")
    df = pd.read_csv(path, low_memory=False)
    # Kolon isimlerini strip et
    df.columns = df.columns.str.strip()
    return df


def map_labels(df: pd.DataFrame) -> pd.Series:
    if "Label" not in df.columns:
        raise ValueError("Label kolonu bulunamadı.")
    mapped = df["Label"].map(CATEGORY_MAP)
    missing = mapped.isna().sum()
    if missing > 0:
        missing_labels = df.loc[mapped.isna(), "Label"].value_counts()
        raise ValueError(f"Eşlenmeyen label var: {missing_labels.to_dict()}")
    return mapped


def split_train_test(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs = []
    test_dfs = []

    for fname in TRAIN_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Eksik train dosyası: {fpath}")
        train_dfs.append(load_csv(fpath))

    for fname in TEST_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Eksik test dosyası: {fpath}")
        test_dfs.append(load_csv(fpath))

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"[SPLIT] Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    return train_df, test_df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Label'i map et ve df'ye ekle (dropna'dan önce)
    y_cat = map_labels(df)
    df = df.copy()  # Orijinal df'yi koru
    df['_y_mapped'] = y_cat  # Geçici kolon
    
    feature_cols = [c for c in df.columns if c not in ["Label", "_y_mapped"]]

    # Numerik dönüşüm
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sonsuzları NaN yap
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf, "Infinity", "inf", "-inf"], np.nan)

    # NaN drop (X ve y birlikte drop edilecek)
    before = len(df)
    df = df.dropna(subset=feature_cols)
    after = len(df)
    print(f"[CLEAN] Dropped {before - after} satır; kalan {after}")

    # X ve y'yi ayır
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df['_y_mapped'].to_numpy()
    
    return X, y, feature_cols


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    # Tüm kategorileri kullan (train'de olmayan test kategorileri için)
    str_to_int = {c: i for i, c in enumerate(ALL_CATEGORIES)}
    int_to_str = {i: c for c, i in str_to_int.items()}
    y_encoded = np.array([str_to_int[c] for c in y], dtype=np.int64)
    return y_encoded, str_to_int, int_to_str


def compute_weights(y_encoded: np.ndarray, num_classes: int) -> List[float]:
    # Sadece train setinde bulunan sınıflar için weight hesapla
    unique_classes = np.unique(y_encoded)
    if len(unique_classes) > 0:
        weights_dict = dict(zip(
            unique_classes,
            compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_encoded)
        ))
    else:
        weights_dict = {}
    
    # Tüm sınıflar için weight listesi oluştur (olmayanlar için 1.0)
    weights = [weights_dict.get(i, 1.0) for i in range(num_classes)]
    return weights


def save_artifacts(out_dir: Path, scaler: StandardScaler, class_weights: List[float],
                   feature_names: List[str], label_maps: Dict[str, Dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "scaler_scale.npy", scaler.scale_)

    with open(out_dir / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(out_dir / "label_encoder.json", "w") as f:
        json.dump(label_maps, f, indent=2)

    print(f"[SAVE] Artifacts saved to {out_dir}")


def save_arrays(out_dir: Path, X_train, y_train, X_test, y_test):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    print(f"[SAVE] Arrays saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="CSV dizini (data-CIC-IDS- 2017)")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Scaler, weights, encoder çıktıları")
    parser.add_argument("--arrays-dir", type=str, default="processed_arrays", help="Numpy X/y çıktıları")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    arrays_dir = Path(args.arrays_dir)

    # Split
    train_df, test_df = split_train_test(data_dir)

    # Preprocess
    X_train, y_train_raw, feature_names = preprocess(train_df)
    X_test, y_test_raw, _ = preprocess(test_df)

    # Label encode
    y_train, str_to_int, int_to_str = encode_labels(y_train_raw)
    y_test = np.array([str_to_int[c] for c in y_test_raw], dtype=np.int64)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("[SCALE] StandardScaler fit/train -> transform/test")

    # Class weights
    class_weights = compute_weights(y_train, num_classes=len(str_to_int))
    print(f"[WEIGHTS] {class_weights}")

    # Save artifacts
    save_artifacts(
        artifacts_dir,
        scaler,
        class_weights,
        feature_names,
        {"str_to_int": str_to_int, "int_to_str": int_to_str},
    )

    # Save arrays
    save_arrays(arrays_dir, X_train, y_train, X_test, y_test)

    print("[DONE] Veri hazırlık tamamlandı.")


if __name__ == "__main__":
    main()

