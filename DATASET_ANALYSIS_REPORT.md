# CIC-IDS-2017 Dataset Detaylı Analiz Raporu

## 📊 Genel Bakış

Bu rapor, `data-CIC-IDS- 2017` klasöründeki 8 CSV dosyasının detaylı analizini içermektedir. Toplam yaklaşık 2.5 milyon satır veri bulunmaktadır.

---

## 📁 Dosya Bazlı Detaylı Analiz

### 1. **Monday-WorkingHours.pcap_ISCX.csv**
- **Toplam Satır**: ~530,000
- **Label Dağılımı**:
  - BENIGN: 529,918 (%100)
- **Özellikler**:
  - Sadece normal trafik içerir
  - Saldırı içermez
  - Baseline normal trafik profili için ideal
- **Kullanım Önerisi**: 
  - Normal trafik profilini öğrenmek için kullanılabilir
  - Anomaly detection modelleri için referans veri

---

### 2. **Tuesday-WorkingHours.pcap_ISCX.csv**
- **Toplam Satır**: ~445,000
- **Label Dağılımı**:
  - BENIGN: 432,074 (%97.1)
  - FTP-Patator: 7,938 (%1.8)
  - SSH-Patator: 5,897 (%1.3)
- **Saldırı Türleri**:
  - **FTP-Patator**: FTP brute-force saldırısı
  - **SSH-Patator**: SSH brute-force saldırısı
- **Özellikler**:
  - İki farklı brute-force saldırı türü içerir
  - Benzer özellikler gösterir (her ikisi de brute-force)
- **Kullanım Önerisi**:
  - Brute-force saldırı tespiti için önemli
  - Authentication-based attack detection

---

### 3. **Wednesday-workingHours.pcap_ISCX.csv**
- **Toplam Satır**: ~692,000
- **Label Dağılımı**:
  - BENIGN: 440,031 (%63.5)
  - DoS Hulk: 231,073 (%33.4)
  - DoS GoldenEye: 10,293 (%1.5)
  - DoS slowloris: 5,796 (%0.8)
  - DoS Slowhttptest: 5,499 (%0.8)
  - Heartbleed: 11 (%0.002)
- **Saldırı Türleri**:
  - **DoS Hulk**: HTTP flood saldırısı
  - **DoS GoldenEye**: HTTP flood saldırısı (farklı varyant)
  - **DoS slowloris**: Yavaş HTTP saldırısı
  - **DoS Slowhttptest**: Yavaş HTTP test saldırısı
  - **Heartbleed**: SSL/TLS açığı saldırısı
- **Özellikler**:
  - En çeşitli DoS saldırı türlerini içerir
  - Farklı DoS teknikleri (flood vs slow)
  - En büyük saldırı çeşitliliği
- **Kullanım Önerisi**:
  - DoS saldırı tespiti için kritik
  - Çok sınıflı DoS sınıflandırması için ideal

---

### 4. **Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv**
- **Toplam Satır**: ~170,000
- **Label Dağılımı**:
  - BENIGN: 168,186 (%98.8)
  - Web Attack – Brute Force: 1,507 (%0.9)
  - Web Attack – XSS: 652 (%0.4)
  - Web Attack – Sql Injection: 21 (%0.01)
- **Saldırı Türleri**:
  - **Web Attack – Brute Force**: Web uygulaması brute-force
  - **Web Attack – XSS**: Cross-Site Scripting saldırısı
  - **Web Attack – Sql Injection**: SQL injection saldırısı
- **Özellikler**:
  - Web uygulama katmanı saldırıları
  - Farklı web saldırı teknikleri
  - SQL Injection çok az örnek içerir (dengesiz)
- **Kullanım Önerisi**:
  - Web uygulama güvenliği için önemli
  - SQL Injection için data augmentation gerekebilir

---

### 5. **Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv**
- **Toplam Satır**: ~288,600
- **Label Dağılımı**:
  - BENIGN: 288,566 (%99.99)
  - Infiltration: 36 (%0.01)
- **Saldırı Türleri**:
  - **Infiltration**: Sistem içine sızma saldırısı
- **Özellikler**:
  - Çok düşük saldırı oranı
  - Aşırı dengesiz veri (highly imbalanced)
  - Nadir saldırı türü
- **Kullanım Önerisi**:
  - Anomaly detection yaklaşımı gerekebilir
  - SMOTE veya benzeri tekniklerle dengeleme gerekli
  - One-class classification düşünülebilir

---

### 6. **Friday-WorkingHours-Morning.pcap_ISCX.csv**
- **Toplam Satır**: ~191,000
- **Label Dağılımı**:
  - BENIGN: 189,067 (%99.0)
  - Bot: 1,966 (%1.0)
- **Saldırı Türleri**:
  - **Bot**: Botnet trafiği
- **Özellikler**:
  - Botnet aktivitesi
  - Düşük saldırı oranı
- **Kullanım Önerisi**:
  - Botnet tespiti için önemli
  - Malware detection ile ilgili

---

### 7. **Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv**
- **Toplam Satır**: ~225,700
- **Label Dağılımı**:
  - DDoS: 128,027 (%56.7)
  - BENIGN: 97,718 (%43.3)
- **Saldırı Türleri**:
  - **DDoS**: Distributed Denial of Service
- **Özellikler**:
  - Yüksek saldırı oranı
  - DDoS saldırıları (distributed)
  - DoS'tan farklı olarak distributed
- **Kullanım Önerisi**:
  - DDoS tespiti için kritik
  - Test seti olarak kullanılabilir

---

### 8. **Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv**
- **Toplam Satır**: ~286,500
- **Label Dağılımı**:
  - PortScan: 158,930 (%55.5)
  - BENIGN: 127,537 (%44.5)
- **Saldırı Türleri**:
  - **PortScan**: Port tarama saldırısı
- **Özellikler**:
  - Yüksek saldırı oranı
  - Reconnaissance saldırısı
  - Test seti olarak kullanılabilir
- **Kullanım Önerisi**:
  - Port scanning tespiti için önemli
  - Test seti olarak kullanılabilir

---

## 🎯 Toplam Label Dağılımı (Tüm Dosyalar)

### Saldırı Türleri:
1. **BENIGN**: ~2,186,000 (%87.4)
2. **DoS Hulk**: 231,073 (%9.2)
3. **PortScan**: 158,930 (%6.4)
4. **DDoS**: 128,027 (%5.1)
5. **DoS GoldenEye**: 10,293 (%0.4)
6. **FTP-Patator**: 7,938 (%0.3)
7. **SSH-Patator**: 5,897 (%0.2)
8. **DoS slowloris**: 5,796 (%0.2)
9. **DoS Slowhttptest**: 5,499 (%0.2)
10. **Bot**: 1,966 (%0.08)
11. **Web Attack – Brute Force**: 1,507 (%0.06)
12. **Web Attack – XSS**: 652 (%0.03)
13. **Web Attack – Sql Injection**: 21 (%0.001)
14. **Infiltration**: 36 (%0.001)
15. **Heartbleed**: 11 (%0.0004)

**Toplam Benzersiz Label Sayısı**: 15 (1 BENIGN + 14 saldırı türü)

---

## 📈 Veri Özellikleri

### Feature Sayısı: 78
- **Flow-based features**: Trafik akışı özellikleri
- **Packet-based features**: Paket bazlı özellikler
- **Time-based features**: Zaman bazlı özellikler (IAT - Inter-Arrival Time)
- **Flag-based features**: TCP flag özellikleri
- **Statistical features**: İstatistiksel özellikler (mean, std, min, max)

### Önemli Feature Kategorileri:
1. **Flow Duration**: Akış süresi
2. **Packet Counts**: Paket sayıları (forward/backward)
3. **Packet Lengths**: Paket boyutları
4. **IAT (Inter-Arrival Time)**: Paketler arası süre
5. **TCP Flags**: SYN, ACK, FIN, RST, PSH, URG
6. **Flow Rates**: Bytes/s, Packets/s
7. **Window Sizes**: TCP window boyutları
8. **Active/Idle Times**: Aktif/bekleme süreleri

---

## 🎓 Model Önerileri

### Senaryo 1: Multi-Class Classification (15 Sınıf)
**Hedef**: Her saldırı türünü ayrı ayrı sınıflandırmak

**Avantajlar**:
- Detaylı saldırı türü bilgisi
- Her saldırı türü için özel aksiyon alınabilir
- Daha fazla bilgi sağlar

**Dezavantajlar**:
- Dengesiz veri (Infiltration, SQL Injection çok az)
- Karmaşık model
- Düşük örnek sayılı sınıflar için zor öğrenme

**Önerilen Yaklaşım**:
- **Neural Network**: Multi-class classification
- **Loss Function**: Weighted Cross-Entropy (sınıf dengesizliği için)
- **Data Augmentation**: SMOTE veya ADASYN (az örnekli sınıflar için)
- **Class Weights**: Düşük örnekli sınıflara daha fazla ağırlık

---

### Senaryo 2: Hierarchical Classification (2 Seviye)
**Seviye 1**: Binary (BENIGN vs ATTACK)
**Seviye 2**: Attack türü sınıflandırması

**Avantajlar**:
- İlk seviyede hızlı tespit
- İkinci seviyede detaylı analiz
- Daha iyi performans potansiyeli

**Dezavantajlar**:
- İki model gerektirir
- Daha karmaşık pipeline

**Önerilen Yaklaşım**:
- **Model 1**: Binary classifier (BENIGN/ATTACK)
- **Model 2**: Multi-class classifier (sadece attack örnekleri için)

---

### Senaryo 3: Attack Category Classification (5-6 Kategori)
**Kategoriler**:
1. **BENIGN**: Normal trafik
2. **DoS/DDoS**: Tüm DoS türleri (Hulk, GoldenEye, slowloris, Slowhttptest, DDoS)
3. **Brute-Force**: FTP-Patator, SSH-Patator, Web Attack – Brute Force
4. **Web Attacks**: XSS, SQL Injection
5. **Reconnaissance**: PortScan
6. **Other**: Bot, Infiltration, Heartbleed

**Avantajlar**:
- Daha dengeli veri dağılımı
- Mantıklı kategoriler
- Daha iyi öğrenme potansiyeli

**Dezavantajlar**:
- Detay kaybı (hangi DoS türü olduğu bilinmez)

**Önerilen Yaklaşım**:
- **Neural Network**: 6-class classification
- **Loss Function**: Weighted Cross-Entropy
- Daha dengeli veri seti

---

### Senaryo 4: Anomaly Detection + Classification
**Yaklaşım**:
1. Anomaly detection ile BENIGN/ATTACK ayrımı
2. Attack örnekleri için multi-class classification

**Avantajlar**:
- Yeni saldırı türlerini yakalayabilir
- Daha esnek sistem

**Dezavantajlar**:
- Karmaşık implementasyon

---

## 🚀 Önerilen Model Mimarisi

### Senaryo 3 (Attack Category Classification) Önerilir:

```python
# Model Architecture
- Input Layer: 78 features
- Dense Layer 1: 256 units, ReLU, Dropout(0.3)
- Dense Layer 2: 128 units, ReLU, Dropout(0.3)
- Dense Layer 3: 64 units, ReLU, Dropout(0.2)
- Output Layer: 6 units (categories), Softmax

# Training Strategy
- Loss: Weighted Categorical Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Batch Size: 512
- Epochs: 100 (early stopping)
- Class Weights: Otomatik hesaplanacak
```

---

## 📊 Train/Test Split Önerisi

### Senaryo A: 6 Dosya Train, 2 Dosya Test
**Train Set** (6 dosya):
1. Monday-WorkingHours
2. Tuesday-WorkingHours
3. Wednesday-workingHours
4. Thursday-WorkingHours-Morning-WebAttacks
5. Thursday-WorkingHours-Afternoon-Infilteration
6. Friday-WorkingHours-Morning

**Test Set** (2 dosya):
1. Friday-WorkingHours-Afternoon-DDos
2. Friday-WorkingHours-Afternoon-PortScan

**Avantajlar**:
- Test setinde yeni saldırı türleri (DDoS, PortScan)
- Gerçekçi senaryo (gelecekteki saldırıları tahmin)
- Train setinde çeşitli saldırı türleri

**Toplam Train Satır**: ~2,200,000
**Toplam Test Satır**: ~512,000

---

## ⚠️ Önemli Notlar ve Zorluklar

### 1. **Veri Dengesizliği**
- Infiltration: Sadece 36 örnek
- SQL Injection: Sadece 21 örnek
- Heartbleed: Sadece 11 örnek

**Çözüm**: 
- SMOTE/ADASYN ile data augmentation
- Class weights kullanımı
- Focal Loss kullanımı

### 2. **Feature Engineering**
- Bazı feature'lar çok yüksek varyans gösterebilir
- Normalization/Standardization kritik
- Outlier handling gerekebilir

### 3. **Overfitting Riski**
- Çok fazla feature (78)
- Düşük örnekli sınıflar
- Regularization (Dropout, L2) önemli

### 4. **Evaluation Metrics**
- Accuracy yeterli değil (dengesiz veri)
- Precision, Recall, F1-score (her sınıf için)
- Confusion Matrix
- Macro/Micro averaged metrics

---

## 🎯 Model Kullanım Senaryoları

### 1. **Gerçek Zamanlı Trafik İzleme**
- Network trafiğini sürekli analiz
- Anormal aktivite tespiti
- Otomatik alarm sistemi

### 2. **Güvenlik Operasyon Merkezi (SOC)**
- Saldırı türüne göre önceliklendirme
- Otomatik incident response
- Threat intelligence

### 3. **Network Security Monitoring**
- IDS/IPS sistemlerinde kullanım
- Firewall kuralları optimizasyonu
- Bandwidth yönetimi

### 4. **Araştırma ve Geliştirme**
- Yeni saldırı türlerini anlama
- Saldırı pattern analizi
- Güvenlik politikası geliştirme

---

## 📝 Sonuç ve Öneriler

1. **En Mantıklı Yaklaşım**: Senaryo 3 (Attack Category Classification)
   - Daha dengeli veri
   - Mantıklı kategoriler
   - İyi performans potansiyeli

2. **Train/Test Split**: Senaryo A (6 train, 2 test)
   - Gerçekçi test senaryosu
   - Yeni saldırı türlerini test eder

3. **Model Tipi**: Deep Neural Network
   - 78 feature için uygun
   - Non-linear pattern'leri yakalayabilir
   - Transfer learning mümkün

4. **Kritik Noktalar**:
   - Veri dengesizliği yönetimi
   - Feature scaling
   - Regularization
   - Comprehensive evaluation

5. **Gelecek İyileştirmeler**:
   - Ensemble methods
   - AutoML yaklaşımları
   - Feature selection
   - Hyperparameter tuning

---

## 📚 Referanslar

- CIC-IDS-2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- Feature açıklamaları: CICFlowMeter tool documentation
- Network traffic analysis best practices

---

**Rapor Tarihi**: 2024
**Analiz Eden**: AI Assistant
**Dataset**: CIC-IDS-2017 (data-CIC-IDS- 2017 klasörü)

