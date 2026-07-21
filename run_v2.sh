#!/bin/bash
# v2: rich 48-paraphrase bank + pruned MedGemma SentencePiece tokenizer.
cd /home/bsada1/babymedgemma
export NANO_INDEX="$PWD/data/index_transfer.json"
export NANO_FEATS="$PWD/cache/transfer_feats.pt"
export NANO_POOLED="$PWD/cache/transfer_pooled.pt"
export NANO_GEMMA_TOK=1
export NANO_MAXLEN=20
export NANO_EVAL_SPLIT=val
export NANO_EVAL_EVERY=500
export NANO_PATIENCE=12
export PYTHONUNBUFFERED=1          # v1 buffered its output; keep progress visible
echo "=== TRAIN v2: 48-paraphrase bank + pruned MedGemma tokenizer ==="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 python train.py --arch gemma --regime augmented \
  --seed 0 --steps 30000 --lr 5e-4 --grounding-token --out results_transfer/v2_aug_s0
echo "=== TRAIN v2 DONE ==="
