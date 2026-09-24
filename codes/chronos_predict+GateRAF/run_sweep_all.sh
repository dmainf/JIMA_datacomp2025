#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs
MASTER=logs/sweep_master.log

echo "########## sweep start $(date '+%F %T') ##########" >> "$MASTER"

echo "===== STAGE 1: K=3 再現チェック（学習なし） $(date '+%F %T') =====" >> "$MASTER"
python3 -u gate_raf_topk.py --k 3 --adapter gate_raf_checkpoints/final_adapter --tag k3verify \
    > logs/k3verify.log 2>&1
echo "  exit=$? $(date '+%F %T')" >> "$MASTER"
python3 -u compare_to_paper.py GateRAF-k3verify/predictions_all.csv > logs/k3verify_compare.log 2>&1
echo "  compare exit=$?" >> "$MASTER"
cat logs/k3verify_compare.log >> "$MASTER"

for K in 1 5 7; do
    echo "===== STAGE 2: top_k=$K 学習+推論 start $(date '+%F %T') =====" >> "$MASTER"
    python3 -u gate_raf_topk.py --k "$K" > "logs/topk_${K}.log" 2>&1
    echo "  top_k=$K exit=$? end $(date '+%F %T')" >> "$MASTER"
done

echo "########## sweep end $(date '+%F %T') ##########" >> "$MASTER"
