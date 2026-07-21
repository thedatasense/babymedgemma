#!/bin/bash
cd /home/bsada1/babymedgemma
export NANO_INDEX="$PWD/data/index_transfer.json"
export NANO_FEATS="$PWD/cache/transfer_feats.pt"
export NANO_POOLED="$PWD/cache/transfer_pooled.pt"
export NANO_GEMMA_TOK=1 NANO_MAXLEN=20
export NANO_EVAL_SPLIT=val NANO_EVAL_MAX=1500
export NANO_EVAL_EVERY=500 NANO_PATIENCE=12
export PYTHONUNBUFFERED=1
echo "=== SAE on the scaled grounded model (layer 1) ==="
CUDA_VISIBLE_DEVICES=0 python sae.py --arch gemma --grounding-token --layer 1 --model-steps 8000 \
  --out "$PWD/results_transfer/sae" 2>&1 | grep -vE "use_fast|slow image|Warning" &
echo "=== Jacobian lens on the scaled grounded model ===" 
CUDA_VISIBLE_DEVICES=1 python jlens.py --arch gemma --grounding-token --model-steps 8000 \
  --out "$PWD/results_transfer/jlens" 2>&1 | grep -vE "use_fast|slow image|Warning" &
wait
echo "=== SAE + JLENS DONE ==="
