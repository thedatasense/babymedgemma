# baby-MedGemma

A small, MedGemma-faithful probe model for **causally isolating where paraphrase
sensitivity in medical vision-language models originates**. It pairs a frozen
`google/medsiglip-448` vision encoder with a small Gemma-3 decoder (rotary
position embeddings, RMSNorm, grouped-query attention, prefix-fused image
tokens), so a mechanism found here transfers to MedGemma-4B, while letting us run
the one experiment the frozen deployed model cannot: varying the training-phrasing
distribution one factor at a time.

Companion to the PhD dissertation *Paraphrase Sensitivity in Medical
Vision-Language Models* (Thrust 3). Design doc: [`docs/tiny_vlm_psf_isolation.md`](docs/tiny_vlm_psf_isolation.md).

## Headline results (baby-Gemma, 16 seeds, `results_gemma/`)

| Finding | Result |
|---|---|
| Data provenance sets the flip rate | augmented 8.4% vs canonical 30.3% vs adversarial 30.4% (aug vs each narrow regime: Mann-Whitney U p = 1.4e-6, Cliff's delta = 1.00) |
| The flip is a low-rank readout direction | rank-1 difference-direction patch restores flips, net recovery ~1.0 at layers 0-1, controls at 0 |
| Language-side by construction | the frozen encoder returns identical features for every paraphrase, so any flip is in the decoder |
| Depth is not the driver | flip 7.6-9.4% across decoder depth 2/4/6/8 |

The origin is the training-phrasing distribution; the fix is paraphrase
augmentation (and a targeted low-rank edit at the early layers).

## Layout

```
gemma_model.py          baby-Gemma (Gemma-3 decoder + frozen MedSigLIP, prefix fusion)
model.py                retired hand-rolled nano (kept for reference)
vision.py               frozen MedSigLIP-448 wrapper (1024 -> 256 pooled tokens)
precompute_features.py  cache the frozen encoder features once
data_index.py           MIMIC + PadChest binary VQA with register-tagged paraphrases
dataset.py              the three training regimes (canonical / augmented / adversarial)
train.py                training loop (--arch gemma|nano), validation early stopping
experiment_a.py         divergence trajectory
experiment_e.py         causal rank-1 patching
sae.py                  sparse autoencoder on the residual stream (+ PCA baseline)
run_all_gpus.py         B/A/C/D/E grid scheduler across GPUs
results_gemma/          baby-Gemma results (compact JSON in git; model.pt on HF)
results/                retired nano results (compact JSON in git; model.pt on HF)
docs/                   design doc + 4B-replication plan
```

## Weights and feature cache

The model checkpoints (`**/model.pt`) and the MedSigLIP feature cache
(`cache/medsiglip_feats.pt`, ~3 GB) are on Hugging Face, not in git:
**[`saillab/babymedgemma`](https://huggingface.co/saillab/babymedgemma)**.

## Reproduce

```bash
# 1. features: download the cache from HF, or recompute (~10 min on one GPU)
CUDA_VISIBLE_DEVICES=0 python precompute_features.py

# 2. one run
CUDA_VISIBLE_DEVICES=0 python train.py --arch gemma --regime augmented --seed 0

# 3. the full grid across 8 GPUs
NANO_ARCH=gemma NANO_LR=5e-4 python run_all_gpus.py --run
```

Requires `torch`, `transformers>=4.57` (Gemma-3), `Pillow`, `numpy`, `scikit-learn`
(for the SAE PCA baseline), and gated access to `google/medsiglip-448`.
