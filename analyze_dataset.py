"""
CIC-IDS-2017 dataset analiz scripti
Her dosyadaki label dağılımlarını, veri türlerini ve özelliklerini analiz eder.
"""
import pandas as pd
import numpy as np
import os
from collections import Counter
from pathlib import Path

def analyze_file(filepath, max_rows=None):
    """Bir dosyayı analiz eder ve istatistikleri döndürür."""
    print(f"\n{'='*80}")
    print(f"Analiz ediliyor: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    try:
        # Dosyayı oku (ilk birkaç satırı kontrol etmek için)
        if max_rows:
            df = pd.read_csv(filepath, nrows=max_rows, low_memory=False)
        else:
            # Büyük dosyalar için chunk'lar halinde oku
            chunk_size = 100000
            chunks = []
            total_rows = 0
            
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                total_rows += len(chunk)
                if max_rows and total_rows >= max_rows:
                    break
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                if max_rows:
                    df = df.head(max_rows)
            else:
                df = pd.read_csv(filepath, low_memory=False)
        
        print(f"Toplam satır sayısı: {len(df):,}")
        print(f"Toplam sütun sayısı: {len(df.columns)}")
        
        # Sütun isimlerini temizle
        df.columns = df.columns.str.strip()
        
        # Label analizi
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            print(f"\n{'─'*80}")
            print("LABEL DAĞILIMI:")
            print(f"{'─'*80}")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {label:30s}: {count:>10,} ({percentage:>6.2f}%)")
            
            print(f"\nToplam benzersiz label sayısı: {df['Label'].nunique()}")
            
            # Label türlerini kategorize et
            benign_count = (df['Label'] == 'BENIGN').sum()
            attack_count = (df['Label'] != 'BENIGN').sum()
            print(f"\nBENIGN: {benign_count:,} ({benign_count/len(df)*100:.2f}%)")
            print(f"ATTACK: {attack_count:,} ({attack_count/len(df)*100:.2f}%)")
        else:
            print("⚠️  'Label' sütunu bulunamadı!")
        
        # Özellik (feature) analizi
        feature_cols = [col for col in df.columns if col not in ['Label']]
        print(f"\n{'─'*80}")
        print(f"ÖZELLİK (FEATURE) ANALİZİ:")
        print(f"{'─'*80}")
        print(f"Toplam özellik sayısı: {len(feature_cols)}")
        
        # İlk birkaç özelliği göster
        print(f"\nİlk 10 özellik:")
        for i, col in enumerate(feature_cols[:10], 1):
            print(f"  {i:2d}. {col}")
        
        # Veri türleri
        print(f"\n{'─'*80}")
        print("VERİ TÜRLERİ:")
        print(f"{'─'*80}")
        dtype_counts = df[feature_cols].dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} sütun")
        
        # Eksik değerler
        print(f"\n{'─'*80}")
        print("EKSİK DEĞER ANALİZİ:")
        print(f"{'─'*80}")
        missing = df[feature_cols].isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print(f"Eksik değer içeren sütun sayısı: {len(missing_cols)}")
            print("İlk 10 sütun:")
            for col, count in missing_cols.head(10).items():
                print(f"  {col:40s}: {count:>10,} ({count/len(df)*100:.2f}%)")
        else:
            print("Eksik değer yok!")
        
        # Sonsuz değerler
        print(f"\n{'─'*80}")
        print("SONSUZ DEĞER ANALİZİ:")
        print(f"{'─'*80}")
        inf_cols = []
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32]:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_cols.append((col, inf_count))
        
        if inf_cols:
            print(f"Sonsuz değer içeren sütun sayısı: {len(inf_cols)}")
            for col, count in inf_cols[:10]:
                print(f"  {col:40s}: {count:>10,}")
        else:
            print("Sonsuz değer yok!")
        
        # İstatistiksel özet (ilk birkaç sayısal sütun için)
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            print(f"\n{'─'*80}")
            print("İSTATİSTİKSEL ÖZET (İlk 5 sayısal özellik):")
            print(f"{'─'*80}")
            for col in numeric_cols[:5]:
                print(f"\n{col}:")
                print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
                print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
        
        return {
            'filename': os.path.basename(filepath),
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'label_counts': df['Label'].value_counts().to_dict() if 'Label' in df.columns else {},
            'unique_labels': df['Label'].nunique() if 'Label' in df.columns else 0,
            'feature_count': len(feature_cols),
            'missing_values': missing.sum() if 'Label' in df.columns else 0,
            'inf_values': sum([np.isinf(df[col]).sum() for col in feature_cols if df[col].dtype in [np.float64, np.float32]])
        }
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return None

def main():
    """Tüm dosyaları analiz eder."""
    data_dir = "data-CIC-IDS- 2017"
    
    if not os.path.exists(data_dir):
        print(f"❌ Dizin bulunamadı: {data_dir}")
        return
    
    # Tüm CSV dosyalarını bul
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    if not csv_files:
        print(f"❌ {data_dir} dizininde CSV dosyası bulunamadı!")
        return
    
    print(f"\n{'='*80}")
    print(f"CIC-IDS-2017 DATASET ANALİZİ")
    print(f"{'='*80}")
    print(f"\nToplam {len(csv_files)} dosya bulundu:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    
    # Her dosyayı analiz et
    results = []
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        result = analyze_file(filepath, max_rows=None)  # Tüm dosyayı oku
        if result:
            results.append(result)
    
    # Özet rapor
    print(f"\n\n{'='*80}")
    print("ÖZET RAPOR")
    print(f"{'='*80}")
    
    if results:
        total_rows = sum(r['total_rows'] for r in results)
        all_labels = set()
        for r in results:
            all_labels.update(r['label_counts'].keys())
        
        print(f"\nToplam satır sayısı: {total_rows:,}")
        print(f"Toplam dosya sayısı: {len(results)}")
        print(f"Tüm dosyalardaki benzersiz label sayısı: {len(all_labels)}")
        print(f"\nTüm label'lar:")
        for label in sorted(all_labels):
            total_count = sum(r['label_counts'].get(label, 0) for r in results)
            print(f"  {label:30s}: {total_count:>10,}")
        
        print(f"\n{'─'*80}")
        print("DOSYA BAZLI ÖZET:")
        print(f"{'─'*80}")
        print(f"{'Dosya Adı':<50s} {'Satır':>12s} {'Label':>8s} {'Özellik':>8s}")
        print(f"{'─'*80}")
        for r in results:
            print(f"{r['filename']:<50s} {r['total_rows']:>12,} {r['unique_labels']:>8} {r['feature_count']:>8}")

if __name__ == "__main__":
    main()

