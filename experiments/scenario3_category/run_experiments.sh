#!/bin/bash
# 3 farklı denemeyi paralel çalıştırma scripti

# Terminal 1: Normalize weights OLMADAN (önerilen)
python train.py \
  --epochs 50 \
  --batch-size 1024 \
  --lr 0.00005 \
  --early-stopping 15 \
  --output-dir "runs/exp1_no_normalize" \
  > logs/exp1_no_normalize.log 2>&1 &

# Terminal 2: Learning rate artırılmış
python train.py \
  --epochs 50 \
  --batch-size 1024 \
  --lr 0.0001 \
  --early-stopping 15 \
  --output-dir "runs/exp2_higher_lr" \
  > logs/exp2_higher_lr.log 2>&1 &

# Terminal 3: Normalize weights ile (karşılaştırma)
python train.py \
  --epochs 50 \
  --batch-size 1024 \
  --lr 0.00005 \
  --normalize-weights \
  --early-stopping 15 \
  --output-dir "runs/exp3_with_normalize" \
  > logs/exp3_with_normalize.log 2>&1 &

echo "3 deneme başlatıldı!"
echo "Log dosyaları: logs/exp*.log"
echo "Checkpoint'ler: runs/exp*/checkpoints/"
echo "Raporlar: runs/exp*/reports/"
echo ""
echo "İlerlemeyi izlemek için:"
echo "  tail -f logs/exp1_no_normalize.log"
echo "  tail -f logs/exp2_higher_lr.log"
echo "  tail -f logs/exp3_with_normalize.log"
echo ""
echo "Tüm process'leri durdurmak için: pkill -f 'python train.py'"

wait
echo "Tüm denemeler tamamlandı!"

