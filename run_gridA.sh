#!/bin/bash
cd /home/bsada1/babymedgemma
export NANO_INDEX="$PWD/data/index_transfer.json"
export NANO_FEATS="$PWD/cache/transfer_feats.pt"
export NANO_POOLED="$PWD/cache/transfer_pooled.pt"
export NANO_GEMMA_TOK=1 NANO_MAXLEN=20 NANO_GROUND=1
export NANO_EVAL_SPLIT=val NANO_EVAL_MAX=2000
export NANO_EVAL_EVERY=500 NANO_PATIENCE=12
export NANO_ARCH=gemma NANO_LR=5e-4 NANO_STEPS=12000
export NANO_SEEDS_B=8 NANO_SEEDS_AE=0 NANO_SEEDS_D=0 NANO_SEEDS_C=0
export NANO_RESULTS="$PWD/results_transfer_grid"
export PYTHONUNBUFFERED=1
echo "=== GRID A: provenance at scale (3 regimes x 8 seeds, NIH+PadChest) ==="
CUDA_VISIBLE_DEVICES="" python run_all_gpus.py --run
echo "=== GRID A DONE ==="
