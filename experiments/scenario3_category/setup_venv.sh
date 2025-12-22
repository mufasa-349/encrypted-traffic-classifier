#!/bin/bash
# Senaryo 3 venv kurulum scripti

echo "=== Senaryo 3 Virtual Environment Kurulumu ==="
echo ""

# Mevcut dizini kontrol et
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python3 kontrolü
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı!"
    exit 1
fi

echo "✅ Python3 bulundu: $(python3 --version)"
echo ""

# Venv oluştur
if [ -d "venv" ]; then
    echo "⚠️  venv zaten mevcut. Yeniden oluşturuluyor..."
    rm -rf venv
fi

echo "📦 Virtual environment oluşturuluyor..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ venv oluşturulamadı!"
    exit 1
fi

echo "✅ venv oluşturuldu"
echo ""

# Venv'i aktifleştir
echo "🔄 venv aktifleştiriliyor..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ venv aktifleştirilemedi!"
    exit 1
fi

echo "✅ venv aktifleştirildi"
echo ""

# Pip'i güncelle
echo "📦 pip güncelleniyor..."
pip install --upgrade pip --quiet

# Paketleri yükle
echo "📦 Gerekli paketler yükleniyor..."
echo "   - numpy"
echo "   - pandas"
echo "   - scikit-learn"
echo "   - torch (M1/MPS desteği ile)"
echo ""

pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Paket yükleme başarısız!"
    exit 1
fi

echo ""
echo "✅ Tüm paketler başarıyla yüklendi!"
echo ""
echo "=== Kurulum Tamamlandı ==="
echo ""
echo "📝 Sonraki adımlar:"
echo "   1. venv'i aktifleştir: source venv/bin/activate"
echo "   2. Veri hazırlığı: python prepare_data.py --data-dir ../../data-CIC-IDS-\\ 2017"
echo "   3. Model eğitimi: python train.py --epochs 20 --batch-size 1024"
echo ""

