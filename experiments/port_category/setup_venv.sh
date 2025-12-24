#!/bin/bash
# Port Category experiment için venv kurulum scripti

echo "=== Port Category Experiment - Venv Kurulumu ==="

# Venv oluştur
if [ ! -d "venv" ]; then
    echo "Venv oluşturuluyor..."
    python3 -m venv venv
else
    echo "Venv zaten mevcut."
fi

# Venv'i aktifleştir
echo "Venv aktifleştiriliyor..."
source venv/bin/activate

# Paketleri yükle
echo "Paketler yükleniyor..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Kurulum tamamlandı! ==="
echo "Venv'i aktifleştirmek için: source venv/bin/activate"

