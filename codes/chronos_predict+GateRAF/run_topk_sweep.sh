#!/bin/bash
cd "$(dirname "$0")"
mkdir -p logs
for K in "$@"; do
    echo "########## top_k=$K start $(date '+%F %T') ##########"
    python3 -u gate_raf_topk.py --k "$K" 2>&1 | tee "logs/topk_${K}.log"
    echo "########## top_k=$K end   $(date '+%F %T') ##########"
done
