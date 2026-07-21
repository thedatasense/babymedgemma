#!/bin/bash
# Everything needed before the dissertation can describe v2 instead of the probe.
#   A  provenance at scale        (grid, already launched separately)
#   A' held-out phrasing          (train on half the templates, score only unseen wording)
#   C  rank-1 causal patching     (is the flip still a low-rank early-layer direction?)
#   D  threshold-free dispersion  (argmax metrics are threshold-vulnerable; check both ways)
cd /home/bsada1/babymedgemma

COMMON=(
  NANO_INDEX="$PWD/data/index_transfer.json"
  NANO_FEATS="$PWD/cache/transfer_feats.pt"
  NANO_POOLED="$PWD/cache/transfer_pooled.pt"
  NANO_GEMMA_TOK=1 NANO_MAXLEN=20 NANO_GROUND=1
  NANO_EVAL_SPLIT=val NANO_EVAL_MAX=2000
  NANO_EVAL_EVERY=500 NANO_PATIENCE=12
  NANO_ARCH=gemma NANO_LR=5e-4 NANO_STEPS=12000
  PYTHONUNBUFFERED=1
)

# ---- wait for grid A -------------------------------------------------------
until grep -aq "GRID A DONE" _gridA.log 2>/dev/null; do
  pgrep -f run_gridA.sh >/dev/null || break
  sleep 120
done
echo "=== GRID A COMPLETE ==="
grep -aE "^\[done\]" _gridA.log | tail -2

# ---- A': held-out phrasing -------------------------------------------------
echo "=== GRID A-heldout: train on 24 templates, score only the 24 unseen ==="
env "${COMMON[@]}" NANO_PARA_HELDOUT=1 \
    NANO_SEEDS_B=6 NANO_SEEDS_AE=0 NANO_SEEDS_D=0 NANO_SEEDS_C=0 \
    NANO_RESULTS="$PWD/results_transfer_heldout" \
    CUDA_VISIBLE_DEVICES="" python run_all_gpus.py --run
echo "=== GRID A-heldout DONE ==="

# ---- C: rank-1 causal patching --------------------------------------------
echo "=== EXPERIMENT C: rank-1 patching on the scaled grounded model ==="
for seed in 0 1 2; do
  for regime in augmented adversarial; do
    gpu=$(( (seed * 2 + $([ "$regime" = augmented ] && echo 0 || echo 1)) % 8 ))
    env "${COMMON[@]}" CUDA_VISIBLE_DEVICES=$gpu \
      python experiment_e.py --arch gemma --grounding-token --regime $regime --seed $seed \
        --steps 12000 --max-clusters 60 \
        --out "$PWD/results_transfer_patch/${regime}_s${seed}" &
  done
done
wait
echo "=== EXPERIMENT C DONE ==="

# ---- D: threshold sweep + threshold-free dispersion ------------------------
echo "=== DISPERSION / THRESHOLD ROBUSTNESS on the scaled grid ==="
env "${COMMON[@]}" CUDA_VISIBLE_DEVICES=0 \
    NANO_DISP_RESULTS="$PWD/results_transfer_grid/B" NANO_DISP_SEEDS=8 \
    NANO_DISP_OUT="$PWD/results_transfer_grid/dispersion.json" \
    python flip_threshold_robustness.py
echo "=== ALL EXPERIMENTS DONE ==="
