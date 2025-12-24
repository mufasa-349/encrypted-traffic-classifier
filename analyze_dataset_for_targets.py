"""
CIC-IDS-2017 veri setini analiz ederek label dışında neural network için 
uygun hedef değişkenler (target variables) bulma.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_dataset():
    data_dir = Path("data-CIC-IDS- 2017")
    
    # Bir dosyayı örnek olarak yükle
    sample_file = data_dir / "Monday-WorkingHours.pcap_ISCX.csv"
    print(f"Analiz ediliyor: {sample_file.name}")
    print("=" * 80)
    
    # İlk 100k satırı yükle (hızlı analiz için)
    df = pd.read_csv(sample_file, nrows=100000, low_memory=False)
    df.columns = df.columns.str.strip()
    
    print(f"\nToplam satır: {len(df):,}")
    print(f"Toplam kolon: {len(df.columns)}")
    
    # Kolonları kategorize et
    print("\n" + "=" * 80)
    print("KOLON ANALİZİ")
    print("=" * 80)
    
    all_columns = list(df.columns)
    print(f"\nTüm kolonlar ({len(all_columns)}):")
    for i, col in enumerate(all_columns, 1):
        print(f"{i:2d}. {col}")
    
    # Label kolonunu bul
    label_col = None
    if "Label" in df.columns:
        label_col = "Label"
        print(f"\n✓ Label kolonu bulundu: {label_col}")
        print(f"  Unique değerler: {df[label_col].nunique()}")
        print(f"  Değer dağılımı:")
        print(df[label_col].value_counts().head(10))
    
    # Numerik kolonları analiz et
    print("\n" + "=" * 80)
    print("NUMERİK KOLON ANALİZİ")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    
    print(f"\nToplam numerik kolon (Label hariç): {len(numeric_cols)}")
    
    # Her kolon için istatistikler
    print("\nÖnemli numerik kolonların istatistikleri:")
    print("-" * 80)
    
    important_features = [
        "Flow Duration",
        "Total Fwd Packets", 
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Destination Port",
        "Packet Length Mean",
        "Packet Length Std",
    ]
    
    feature_stats = {}
    for col in important_features:
        if col in df.columns:
            stats = df[col].describe()
            feature_stats[col] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "median": float(df[col].median()),
                "null_count": int(df[col].isna().sum()),
                "zero_count": int((df[col] == 0).sum()),
                "unique_count": int(df[col].nunique()),
            }
            print(f"\n{col}:")
            print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
            print(f"  Median: {stats['50%']:.2f}")
            print(f"  Null: {df[col].isna().sum()}, Zero: {(df[col] == 0).sum()}")
            print(f"  Unique: {df[col].nunique()}")
    
    # Potansiyel hedef değişkenler için öneriler
    print("\n" + "=" * 80)
    print("POTANSİYEL HEDEF DEĞİŞKEN ÖNERİLERİ")
    print("=" * 80)
    
    suggestions = []
    
    # 1. Flow Duration tahmini (Regression)
    if "Flow Duration" in df.columns:
        duration = df["Flow Duration"]
        if duration.notna().sum() > 1000:
            suggestions.append({
                "target": "Flow Duration",
                "type": "Regression",
                "description": "Akış süresini tahmin etme - trafik pattern'lerinden süre tahmini",
                "features_to_exclude": ["Flow Duration"],
                "difficulty": "Orta",
                "use_case": "Network performance prediction, anomaly detection"
            })
    
    # 2. Packet Count tahmini
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        total_packets = df["Total Fwd Packets"] + df["Total Backward Packets"]
        if total_packets.notna().sum() > 1000:
            suggestions.append({
                "target": "Total Packets (Fwd + Bwd)",
                "type": "Regression",
                "description": "Toplam paket sayısını tahmin etme",
                "features_to_exclude": ["Total Fwd Packets", "Total Backward Packets", "Total Fwd Packets", "Total Backward Packets"],
                "difficulty": "Kolay-Orta",
                "use_case": "Traffic volume prediction"
            })
    
    # 3. Flow Rate tahmini (Bytes/s veya Packets/s)
    if "Flow Bytes/s" in df.columns:
        if df["Flow Bytes/s"].notna().sum() > 1000:
            suggestions.append({
                "target": "Flow Bytes/s",
                "type": "Regression",
                "description": "Akış hızını (bytes/saniye) tahmin etme",
                "features_to_exclude": ["Flow Bytes/s", "Flow Packets/s"],
                "difficulty": "Orta",
                "use_case": "Bandwidth prediction, capacity planning"
            })
    
    # 4. Port kategorisi (Destination Port gruplaması)
    if "Destination Port" in df.columns:
        ports = df["Destination Port"].dropna()
        if len(ports) > 1000:
            # Port aralıklarına göre kategorize et
            port_ranges = {
                "Well-known (0-1023)": (0, 1023),
                "Registered (1024-49151)": (1024, 49151),
                "Dynamic (49152-65535)": (49152, 65535)
            }
            suggestions.append({
                "target": "Destination Port Category",
                "type": "Multi-class Classification (3 sınıf)",
                "description": "Hedef portu kategorilere ayırma (Well-known, Registered, Dynamic)",
                "features_to_exclude": ["Destination Port"],
                "difficulty": "Kolay",
                "use_case": "Port-based traffic classification, service identification"
            })
    
    # 5. Trafik yönü tahmini (Forward vs Backward dominant)
    if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
        fwd = df["Total Fwd Packets"]
        bwd = df["Total Backward Packets"]
        if fwd.notna().sum() > 1000 and bwd.notna().sum() > 1000:
            suggestions.append({
                "target": "Traffic Direction Dominance",
                "type": "Binary Classification",
                "description": "Trafiğin forward mu backward mu dominant olduğunu tahmin etme",
                "features_to_exclude": ["Total Fwd Packets", "Total Backward Packets"],
                "difficulty": "Kolay",
                "use_case": "Traffic pattern analysis, upload/download detection"
            })
    
    # 6. Paket boyutu kategorisi
    if "Packet Length Mean" in df.columns:
        pkt_mean = df["Packet Length Mean"]
        if pkt_mean.notna().sum() > 1000:
            suggestions.append({
                "target": "Packet Size Category",
                "type": "Multi-class Classification",
                "description": "Ortalama paket boyutuna göre kategorize etme (Small, Medium, Large)",
                "features_to_exclude": ["Packet Length Mean", "Packet Length Std", "Packet Length Min", "Packet Length Max"],
                "difficulty": "Kolay",
                "use_case": "Packet size analysis, protocol identification"
            })
    
    # 7. IAT (Inter-Arrival Time) tahmini
    if "Flow IAT Mean" in df.columns:
        if df["Flow IAT Mean"].notna().sum() > 1000:
            suggestions.append({
                "target": "Flow IAT Mean",
                "type": "Regression",
                "description": "Paketler arası ortalama süreyi tahmin etme",
                "features_to_exclude": ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
                "difficulty": "Orta-Zor",
                "use_case": "Network timing analysis, latency prediction"
            })
    
    # 8. TCP Flag pattern classification
    flag_cols = [col for col in df.columns if "Flag" in col]
    if len(flag_cols) > 0:
        suggestions.append({
            "target": "TCP Flag Pattern",
            "type": "Multi-class Classification",
            "description": "TCP flag kombinasyonlarına göre trafik pattern'i sınıflandırma",
            "features_to_exclude": flag_cols,
            "difficulty": "Orta",
            "use_case": "Connection state analysis, protocol behavior"
        })
    
    # Önerileri yazdır
    print(f"\n{len(suggestions)} farklı hedef değişken önerisi:\n")
    for i, sug in enumerate(suggestions, 1):
        print(f"{i}. {sug['target']}")
        print(f"   Tip: {sug['type']}")
        print(f"   Açıklama: {sug['description']}")
        print(f"   Zorluk: {sug['difficulty']}")
        print(f"   Kullanım: {sug['use_case']}")
        print()
    
    # JSON olarak kaydet
    output = {
        "dataset_info": {
            "total_rows": int(len(df)),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "label_column": label_col
        },
        "feature_statistics": feature_stats,
        "target_suggestions": suggestions
    }
    
    with open("dataset_target_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print("Analiz tamamlandı! Sonuçlar 'dataset_target_analysis.json' dosyasına kaydedildi.")
    print("=" * 80)
    
    return suggestions

if __name__ == "__main__":
    analyze_dataset()

