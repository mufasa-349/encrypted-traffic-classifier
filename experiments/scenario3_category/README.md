# Senaryo 3 (Kategori Bazlı) Çalışma Planı

Bu klasör, CIC-IDS-2017 veri seti için 6 kategorili (BENIGN + 5 saldırı kategorisi) deneyleri içerir.

## Klasör yapısı
- `prepare_data.py` : Veri yükleme, kategori eşleme, train/test split (dosya bazlı), scaler ve class weight üretimi.
- `train.py`        : 6 sınıflı MLP eğitimi (weighted loss), metrik ve confusion matrix çıktıları.
- `artifacts/`      : Üretilen `scaler.pkl`, `class_weights.json`, `feature_names.json`, `label_encoder.json`.
- `checkpoints/`    : Model ağırlıkları (`model.pt`).
- `reports/`        : Metrik ve confusion matrix çıktıları.

## Kategori tanımı (6 sınıf)
1. `BENIGN`
2. `DOS_DDOS`           → DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, DDoS
3. `BRUTEFORCE`         → FTP-Patator, SSH-Patator, Web Attack – Brute Force
4. `WEB_ATTACKS`        → Web Attack – XSS, Web Attack – Sql Injection
5. `RECON`              → PortScan
6. `OTHER`              → Bot, Infiltration, Heartbleed

## Train/Test dosya listesi
- Train (7 dosya):
  - Monday-WorkingHours.pcap_ISCX.csv
  - Tuesday-WorkingHours.pcap_ISCX.csv
  - Wednesday-workingHours.pcap_ISCX.csv
  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  - Friday-WorkingHours-Morning.pcap_ISCX.csv
  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (RECON kategorisi için eklendi)
- Test (1 dosya):
  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (DDoS kategorisi test için)

## Çalıştırma adımları
1) Veri hazırla  
```bash
python prepare_data.py --data-dir "../data-CIC-IDS- 2017"
```
Bu adım `artifacts/` altında scaler, class weights ve label encoder üretir, ayrıca ön işlenmiş train/test numpy dosyaları kaydeder.

2) Eğit  
```bash
python train.py \
  --data-dir "../data-CIC-IDS- 2017" \
  --epochs 50 \
  --batch-size 512 \
  --lr 1e-3
```

Not: Çalıştırma sırasında GPU varsa otomatik kullanılır; yoksa CPU.


