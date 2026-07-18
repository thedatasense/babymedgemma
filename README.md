# baby-MedGemma

A small, **MedGemma-faithful probe** for causally isolating *where paraphrase
sensitivity in medical vision-language models originates*. It pairs a frozen
`google/medsiglip-448` vision encoder with a small **Gemma-3 decoder** (rotary
position embeddings, RMSNorm, grouped-query attention, prefix-fused image
tokens), so a mechanism found here transfers to MedGemma-4B, while letting us run
the one experiment the frozen deployed model cannot support: **varying the
training-phrasing distribution one factor at a time.**

- Weights and feature cache: **[huggingface.co/saillab/babymedgemma](https://huggingface.co/saillab/babymedgemma)**
- Interactive write-up: **[bineshkumar.me/phd-thesis/causality](https://bineshkumar.me/phd-thesis/causality/)**
- Design document: [`docs/tiny_vlm_psf_isolation.md`](docs/tiny_vlm_psf_isolation.md)

Companion to the PhD dissertation *Paraphrase Sensitivity in Medical
Vision-Language Models* (Thrust 3, mechanism and mitigation).

## Why this exists

Localization on a deployed medical model can show *where* a paraphrase flip is
decided (layer 16 on MedGemma-4B) and *what* feature toggles it (a layer-17
register gate), but not *why* the flip exists, because a deployed model's
pretraining data, architecture, and objective are frozen together. A small model
that reflects MedGemma's architecture, but whose training data we control, turns
that localization into an origin: it lets us intervene on the phrasing
distribution and read the effect on the flip rate.

## Architecture

```
chest X-ray ──► MedSigLIP-448 (frozen, 429M) ──► 1024 patches pooled to 256 tokens
                                                        │  prepended inline (prefix fusion)
question (paraphrase) ──► token embeddings ─────────────┤
                                                        ▼
                                        Gemma-3 decoder (trained, 13.9M)
                                        RoPE · RMSNorm · GQA · causal mask
                                                        ▼
                                            yes / no at the answer position
```

Because the encoder is frozen and returns identical features for every paraphrase
of a question, **any answer that changes across paraphrases is language-side by
construction.** The probe reaches 88.8% accuracy and, like MedGemma, is
text-reliant (75% of answers unchanged when the image is removed, close to the
deployed model's 81%), so it is faithful in behavior as well as architecture.

## Headline results (16 seeds, `results_gemma/`)

| Experiment | Finding |
|---|---|
| **B — data provenance** | The training-phrasing distribution sets the flip rate: augmented 8.4% vs canonical 30.3% vs adversarial 30.4%. Augmented separates from both narrow regimes at Mann-Whitney U p = 1.4e-6, Cliff's delta = 1.00. |
| **E — causal patching** | The flip is a low-rank readout direction: a rank-1 difference-direction patch restores flips with net recovery near 1.0 at layers 0-1, while random-direction and non-flip controls stay at 0. Holds for natural flips, not only injected ones. |
| **A — divergence** | Representation dispersion couples to flips from the earliest layers; lexical substitution drives the most naturally occurring flips. |
| **C — architecture** | Depth is not the driver (flip 7.6-9.4% across decoder depths 2/4/6/8). |
| **D — grounding sweep** | Non-monotonic and inconclusive; no weak-grounding claim is drawn from it. |

**The origin is the training-phrasing distribution; the fix is paraphrase
augmentation** (a flip rate of 30.3% under a single phrasing falls to 8.4% under
full coverage), and a targeted low-rank edit at the early layers.

## Layout

```
gemma_model.py          baby-Gemma: Gemma-3 decoder + frozen MedSigLIP, prefix fusion
model.py                retired hand-rolled nano probe (kept for reference)
vision.py               frozen MedSigLIP-448 wrapper (1024 -> 256 pooled tokens)
precompute_features.py  cache the frozen encoder features once
data_index.py           MIMIC + PadChest binary VQA with register-tagged paraphrases
dataset.py              the three training regimes (canonical / augmented / adversarial)
train.py                training loop (--arch gemma|nano), validation early stopping
experiment_a.py         divergence trajectory
experiment_e.py         causal rank-1 patching
sae.py                  sparse autoencoder on the residual stream (+ PCA baseline)
run_all_gpus.py         B/A/C/D/E grid scheduler across GPUs
results_gemma/          baby-Gemma results (compact JSON here; model.pt on HF)
results/                retired nano results (compact JSON here; model.pt on HF)
docs/                   design doc + 4B-replication plan
```

The model checkpoints (`**/model.pt`, ~4 GB) and the MedSigLIP feature cache
(`cache/medsiglip_feats.pt`, ~3 GB) are on Hugging Face, not in git:
**[saillab/babymedgemma](https://huggingface.co/saillab/babymedgemma)**.

## Reproduce

```bash
# 0. install: torch, transformers>=4.57 (Gemma-3), Pillow, numpy, scikit-learn.
#    Needs gated access to google/medsiglip-448.

# 1. features: download the cache from HF into cache/, or recompute (~10 min / GPU)
huggingface-cli download saillab/babymedgemma feature_cache/medsiglip_feats.pt \
  --local-dir . && mv feature_cache/medsiglip_feats.pt cache/
# or:  CUDA_VISIBLE_DEVICES=0 python precompute_features.py

# 2. one run
CUDA_VISIBLE_DEVICES=0 python train.py --arch gemma --regime augmented --seed 0

# 3. the full grid across 8 GPUs
NANO_ARCH=gemma NANO_LR=5e-4 python run_all_gpus.py --run
```

## Limitations

- A 13.9M-parameter decoder on a frozen encoder trained on ~1,800 questions is a
  controlled probe, not the deployed MedGemma-4B; its absolute layer index is not
  comparable and only the qualitative account transfers.
- It manipulates the adaptation-stage phrasing distribution, which is distinct
  from pretraining provenance.
- It grounds only weakly (like MedGemma), so it speaks to where the flip is
  decided, not to how well the image is used. The coverage-versus-shortcut
  distinction seen in a hand-built model does not reproduce here (the two narrow
  regimes are statistically indistinguishable), so the reproducible finding is
  the augmentation lever itself.

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.*
> Unpublished doctoral dissertation, University of New Haven.

## License and data

Code is released for research use. The probe is trained on MIMIC-CXR and
PadChest; use of the derived features and checkpoints is subject to those
datasets' data-use agreements. Not a medical device and not for clinical use.
