#!/bin/bash
# Wait for v2 training, then evaluate transfer to the held-out hospitals.
cd /home/bsada1/babymedgemma
until grep -aq "TRAIN v2 DONE" _v2.log 2>/dev/null; do
  pgrep -f "train\.py --arch gemma" >/dev/null || break
  sleep 120
done
echo "=== TRAINING FINISHED ==="
grep -aE "^\[nano\]|step " _v2.log | tail -6

export NANO_INDEX="$PWD/data/index_transfer.json"
export NANO_FEATS="$PWD/cache/transfer_feats.pt"
export NANO_POOLED="$PWD/cache/transfer_pooled.pt"
export NANO_GEMMA_TOK=1 NANO_MAXLEN=20
export NANO_CKPT="$PWD/results_transfer/v2_aug_s0/model.pt"
export PYTHONUNBUFFERED=1

if [ ! -f "$NANO_CKPT" ]; then echo "NO CHECKPOINT at $NANO_CKPT"; exit 1; fi
echo "=== EVALUATING TRANSFER (val / ood_mimic / ood_vindr) ==="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 python scripts/analysis/eval_transfer.py
echo "=== EVAL DONE ==="
