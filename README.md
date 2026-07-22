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

> **Scope.** This is a controlled research probe, **not a general medical VQA
> model**. It answers binary presence questions about 14 chest findings, using
> MedGemma's SentencePiece vocabulary pruned to the 141 pieces this corpus needs.
> It supports a *sufficiency* claim about the cause of paraphrase sensitivity, not
> proof that the deployed MedGemma-4B has this exact origin. Not a medical device;
> not for clinical use.
> The experiments below state exactly what it does and does not establish.

## The question

Medical vision-language models often change their yes/no answer when a clinical
question is rephrased without changing its meaning. On a deployed model you can
find *where* that flip is decided (layer 16 on MedGemma-4B) and *what* feature
carries it (a layer-17 register gate), but not *why* it exists, because the
model's data, architecture, and objective are frozen together. To ask *why*, you
have to intervene on one of them and read the effect. That is what this probe is
for.

## The probe

```
chest X-ray ──► MedSigLIP at 896px (frozen, 429M) ──► 4096 patches pooled to 256 tokens
                                                        │  prepended inline (prefix fusion)
question (paraphrase) ──► token embeddings ─────────────┤
                                                        ▼
                                        Gemma-3 decoder (trained, 14.1M)
                                        RoPE · RMSNorm · GQA (6 Q / 2 KV) · causal mask
                                                        ▼
                                yes / no read from the tied Gemma-3 LM head
```

Trained on 86,288 per-finding-balanced presence questions from **NIH ChestX-ray14
and PadChest**, with **MIMIC-CXR and VinDr-CXR held out entirely**, each question
carrying 48 register-tagged paraphrases. It reaches 74.8% accuracy and, because
every split is balanced per finding, **exactly 50.0% with the image removed**, so
all of its skill is visual. It transfers to both unseen hospitals (AUC 0.743 MIMIC,
0.756 VinDr) above MedSigLIP's own zero-shot 0.734. Two properties make it useful:

1. **Language-side by construction.** The encoder returns *identical* features for
   every paraphrase, so any answer that changes across paraphrases is in the
   decoder, not the image. Guaranteed by the setup, not measured.
2. **The origin is a knob.** We can vary the training-phrasing distribution with
   everything else held fixed, turning a difference in the flip rate into an
   *identified cause*.

## Fidelity to MedGemma-1.5

The probe follows the MedGemma-1.5 technical report
([arXiv:2604.05081](https://arxiv.org/pdf/2604.05081)) on every architectural
choice that decides *how* a paraphrase reaches the answer, and simplifies only the
axes of scale and training that a controlled causal experiment needs to strip
away. The first table is why a mechanism found here is expected to transfer; the
second is why this is a probe and not a deployable model.

**Matched to the deployment stack**

| Principle | MedGemma-1.5 | baby-Gemma |
|---|---|---|
| Vision encoder | MedSigLIP (400M SigLIP), frozen | MedSigLIP-448, frozen |
| Input resolution | 896 x 896 | 896 x 896 (position embeddings interpolated) |
| Image tokens per image | 256 (pooled) | 4096 patches average-pooled to 256 |
| Vision-to-text fusion | soft image tokens projected inline | multilayer-perceptron projector, prepended inline |
| Decoder family | Gemma-3 | Gemma-3 blocks: RoPE, RMSNorm, grouped-query attention, GeGLU, QK-norm |
| yes/no readout | generated as text through the LM head | yes/no token logits from the tied LM head |

**Intentionally simplified for the experiment**

| Axis | MedGemma-1.5 | baby-Gemma | Why |
|---|---|---|---|
| Decoder scale | 4B parameters, pretrained | 14.1M, trained from scratch | so the training-phrasing distribution is a controllable input, not a frozen prior |
| Training objective | supervised fine-tuning, distillation, reinforcement learning | supervised, validation early-stopped | keeps the only manipulated variable the phrasing distribution |
| Vocabulary | full Gemma tokenizer (262,144) | **same tokenizer**, pruned to the 141 pieces this corpus uses (0.4% of the parameters) | identical segmentation to MedGemma; unseen words decompose into pieces rather than becoming padding |
| Local attention | 5:1 local/global sliding window | one window over the whole short sequence | the 257+20 sequence is shorter than any window, so the pattern is inert here |

## The framework: five experiments, A to E

The five experiments isolate the origin in one line each:

| | Experiment | Question it answers |
|---|---|---|
| **A** | Data provenance | Does the *training data* cause the sensitivity? |
| **B** | Divergence trajectory | *Where* in the network does the disagreement emerge? |
| **C** | Causal patching | *What* decides the flip, and is it low-rank? |
| **D** | Architecture | Is it a property of model *depth*? |
| **E** | Grounding | Is it caused by *weak image grounding*? |

A establishes the cause; B and C localize and confirm the mechanism; D and E rule
out the two obvious alternatives. A, C and the threshold checks were re-run on the
scaled model described above; **B, D and E are carried over from the earlier
1,841-question probe** (published at `probe-1841/` on Hugging Face) and are
reported as that model's results. Two corroborations (an unsupervised feature and
a lens) and a zero-shot transfer test close it out. All figures are baby-Gemma
over 8 seeds (`results_transfer_grid/`).

### A. The cause: data provenance

**What we did.** Train the same probe under three phrasing regimes that differ
*only* in the questions it sees: *canonical* (one fixed phrasing), *augmented*
(every paraphrase), and *adversarial* (register tied to the answer). Because only
the training-phrasing distribution changes, a difference in the flip rate is an
identified cause.

| Regime | Trained on | Flip rate | Flip rate, **unseen phrasings** |
|---|---|---|---|
| Augmented | every paraphrase | **4.8%** | **26.6%** |
| Canonical | one fixed phrasing | 67.1% | 65.9% |
| Adversarial | register tied to the answer | 88.4% | 87.4% |

Augmented separates from both narrow regimes at the maximum effect size
(Mann-Whitney U p = 1.6e-4, Cliff's delta = 1.00, 8 seeds each), and the two narrow
regimes now separate from each other as well (p = 3.1e-4, delta = 0.97): a phrasing
shortcut is a **distinct and larger harm** than narrow coverage alone. Accuracy
follows the same ordering (75.3% / 66.8% / 58.1%), so the shortcut damages diagnosis
and not only consistency. Text-only accuracy is 50.0-50.3% in every regime.

The second column is the honest one. A model trained on every paraphrase has seen
the phrasings it is tested on, so part of a low flip rate is familiarity. Training
on 24 phrasings and scoring **only on the 24 withheld** (6 of each phenomenon per
side), augmented rises 4.8% -> 26.6% but still flips less than half as often as
canonical, delta = 1.00 (p = 2.2e-3). **Broad coverage generalises to wording never
seen in training** -- which is what the mitigation claim actually needs.

**What it proves.** The training-phrasing distribution is *sufficient* to produce
and to remove paraphrase sensitivity, and broad coverage is the single largest
lever against it. Correlating register with the answer adds nothing beyond a
single phrasing, so the reproducible finding is the augmentation lever, not a
taxonomy of flip types.

### B. Where it emerges: divergence trajectory

Within-cluster representation dispersion couples to the flip from the earliest
layers (point-biserial near 0.76 at the input for natural flips, 0.88 for adversarial),
and lexical substitution and scope shifts drive the most disagreement. **The
disagreement is present early and carried by the wording, not seeded in the
image.**

### C. The mechanism: causal patching

For a flipped question, transplant the answer position along a single rank-1
direction (the difference between a phrasing answered one way and one answered the
other) at one layer at a time. It restores the flipped answer with net recovery
0.98-1.00 across layers 0 to 4 for adversarial and 0.77 rising to 1.00 for natural
flips, while a norm-matched random direction (0.00-0.02) and a non-flip-cluster
control (disruption 0.000-0.006) leave the answer unchanged. This holds for the *naturally occurring* flips of
the augmented regime (59 per seed), not only the injected adversarial ones (60
per seed). **The flip is a low-rank, language-side, readout-stage direction,
decided in the early layers, and not an artifact of the adversarial construction.**

### D. Ruling out architecture

Across decoder depths 2, 4, 6, 8 the flip rate stays within 9.1% to 10.2% at
constant accuracy. **Depth is not the driver.**

### E. Ruling out weak grounding

Weakening the visual pathway by dropping vision tokens gives a non-monotonic
flip-rate curve. **Inconclusive; no weak-grounding claim is drawn from it.**

## Corroboration: an unsupervised feature, and a lens

Two independent methods that make different assumptions agree with C.

**A sparse autoencoder (`sae.py`)** factors the residual stream at the answer
position into a few active features, each a direction; we ask whether any feature
matches the causal flip direction from C.

![What the sparse autoencoder tests](figures/sae_concept.png)

An unsupervised feature aligns with the causal flip direction at |cosine| 0.74,
clearly ahead of principal-component analysis at 0.52 and far above a random
direction's 0.04; a distinct feature predicts flips (point-biserial 0.43). The
flip axis is unsupervised-recoverable, not an artifact of the supervised
difference-of-means.

![An unsupervised feature recovers the causal flip axis](figures/sae_alignment.png)

**A Jacobian lens (`jlens.py`)** reads the yes/no margin each layer is disposed to
produce (the average input-output Jacobian). For two phrasings of one question the
readouts track together early, then commit to opposite answers.

![What the Jacobian lens shows](figures/jlens_concept.png)

Across paraphrases, flipping clusters diverge about **8.9x** more than stable ones,
from layer 0, with a divergence-vs-flip correlation of **0.81**. A lens and a
causal patch, with different assumptions, place the flip in the same early layers.

![The lens splits flipping from stable paraphrases](figures/jlens_divergence.png)

## Why binary accuracy cannot tell a blind model from a seeing one

The probe above scored **0.50 accuracy on unseen NIH radiographs**, which reads as
total failure. Scoring the same models by the **AUC of their yes/no margin** instead
shows what accuracy was hiding (16 seeds, balanced 50/50 set):

| model | accuracy | AUC |
|---|---|---|
| no grounding token | 0.50 | **0.500** (genuinely blind, 16/16 seeds) |
| grounding token **shuffled** across images | 0.52 | 0.502 |
| grounding token, real | 0.53 | **0.604** |

All three score ~0.50 accuracy. Their AUCs are chance, chance, and clearly-not-chance.
A binary yes/no readout collapses a continuous margin onto one decision threshold per
finding, and that threshold is learned from the training distribution's prevalence;
under a distribution shift the whole margin distribution moves and every image lands
on one side. So the model answers "yes" to all cardiomegaly and "no" to all masses at
exactly 50% accuracy, while ranking both correctly.

MedSigLIP itself pays this tax: **AUC 0.734 but accuracy 0.681** on the same task.
**Report AUC.** Accuracy alone cannot detect whether a medical VLM is using the image,
which is the metric the field mostly uses.

The in-distribution flip-rate results are *not* affected by this. A threshold sweep
holds the augmented < canonical ordering at every non-degenerate offset, and a
threshold-free statistic (within-cluster margin dispersion over between-cluster
dispersion) reproduces it at maximal effect size: augmented 0.073, canonical 0.284,
adversarial 0.312, p = 1.5e-6, Cliff's delta = 1.00.

## Scaling up: a grounded model that transfers (`build_transfer_index.py`)

The probe is deliberately small. Given the diagnosis above, three changes make it
actually read radiographs: **per-finding answer balancing** (so the question text
predicts nothing and the only way to reduce loss is to look), the **grounding token**,
and **scale** (107k questions over 66,546 images from the full ChestX-ray14). Trained
on NIH + PadChest with **MIMIC and VinDr held out entirely**:

| split | n | accuracy | AUC | text-only floor | flip |
|---|---|---|---|---|---|
| in-distribution (held-out images) | 15,112 | 0.748 | 0.827 | **0.500** | 0.061 |
| **MIMIC (unseen hospital)** | 450 | 0.671 | **0.743** | **0.500** | 0.049 |
| **VinDr (unseen hospital)** | 5,478 | 0.686 | **0.756** | **0.500** | 0.060 |

Both held-out hospitals exceed MedSigLIP's own zero-shot ceiling (0.734), against the
small probe's 0.500. The text-only floor is exactly 0.500 everywhere by construction,
so every point above it is earned from the image: **+24.8 pp of visual skill**, versus
+19.4 pp for the 88.2%-accurate original whose blind floor was 68.8%.

Scale alone was not the fix. An earlier 10k scale-up was abandoned as "stuck at
chance" — it was scored by accuracy, and it lacked both the balancing and the
grounding token. All three were needed.

Per finding, pneumothorax transfers essentially unchanged (0.899 -> 0.903 on VinDr).
**Pleural thickening inverts on VinDr (AUC 0.332, below chance)**, almost certainly a
label-definition mismatch between the two datasets, and is not trustworthy.

## The claim, stated exactly

The origin of paraphrase sensitivity is the **training-phrasing distribution**,
executed as a **low-rank direction in the early language layers** that read a
fixed visual representation. The fix follows: paraphrase augmentation is the lever
(67.1% down to 4.8%, and to 26.6% on wording withheld from training), and a
targeted low-rank edit at those layers is the
efficient parametric fix, which is why a layers-15-to-19 low-rank adaptation
reduces flips on the deployed model while full fine-tuning does not.

## What this does NOT prove

- **It is a controlled probe, not the deployed model.** The result is a
  *sufficiency* claim plus a localization, not proof that MedGemma-4B's paraphrase
  sensitivity has this exact origin. Its absolute layer index (early, layers 0-4)
  is not comparable to the deployed model's layer-16 commit; only the qualitative
  account transfers.
- **It manipulates adaptation-stage data, not pretraining provenance.**
- **The paraphrases are a bank of 48 templates across four phenomena,** not free
  clinician writing. The held-out-phrasing result establishes generalization to
  unseen *templates*, not to arbitrary unseen language.
- **Per-finding balancing removes the answer prior by design.** That is what lets
  a flip be attributed to wording rather than a base rate, but it also makes the
  probe unrepresentative of deployment, where prevalence is skewed and a model can
  be right for the wrong reason.
- **It is a narrow model, not a medical VQA system:** 14 findings, binary
  questions, loadable and runnable for inspection and reproduction only. Not for
  clinical use.

## Why it still matters: triangulation

The probe is one leg of a three-method argument that no single method licenses. On
MedGemma-4B, a lens-free residual patch and the Jacobian lens put the answer
commit at layer 16, and a GemmaScope sparse autoencoder finds a layer-17 register
gate (Feature 3818). The probe adds the one thing those cannot: a controlled
origin. The three agree that the flip is language-side, decided in a narrow
early-to-middle band, and set by the training-phrasing distribution.

## Layout

The modules are flat by design: scripts resolve data and cache paths relative to the
repository root, and the Hugging Face wrapper is loaded by `trust_remote_code`, so the
import namespace is kept simple rather than nested.

**Core library** (imported by everything else)

```
vision.py            frozen MedSigLIP-448 wrapper (896px, 4096 -> 256 pooled tokens)
data_index.py        the VQA index (NANO_INDEX selects which one)
dataset.py           tokenizer selector, the three training regimes, paraphrase clusters
gemma_tokenizer.py   MedGemma SentencePiece pruned to the corpus (141 pieces)
templates.py         48-paraphrase bank (meaning-preserving; negation kept separate)
gemma_model.py       baby-Gemma: Gemma-3 decoder + frozen MedSigLIP, prefix fusion, grounding token
model.py             retired hand-rolled nano probe (kept for reference)
metrics.py           flip rate, accuracy, grounding gap, layer dispersion
train.py             training loop + train_model() (--arch gemma|nano, --grounding-token)
```

**Data preparation** (`scripts/data` role)

```
download_nih.py         fetch the full ChestX-ray14 (112k images) from Hugging Face
build_transfer_index.py train NIH+PadChest, hold out MIMIC+VinDr as unseen hospitals
build_scaled_index.py   NIH + MIMIC/PadChest, per-finding balanced
precompute_features.py  cache the 896 patch tokens once
precompute_pooled.py    cache MedSigLIP's attention-pooled embedding (the grounding token)
encode_shard.py         shard both encodings across GPUs; merge_shards.py folds them back
```

**Experiments** (A to E, plus the two corroborations)

```
run_all_gpus.py         A-E grid scheduler across GPUs
experiment_a.py         B. divergence trajectory
experiment_e.py         C. causal rank-1 patching
sae.py                  sparse autoencoder on the residual stream (+ PCA baseline)
jlens.py                Jacobian lens
nih_demo.py             the original NIH transfer probe + NIH question builders
scripts/run/            shell drivers that chain the above into full runs
```

**Evaluation and analysis**

```
eval_transfer.py             accuracy + AUC + text-only floor per split
eval_scaled.py               held-out evaluation of the scaled model
nih_auc_analysis.py          the AUC-vs-accuracy diagnosis (blind vs seeing)
verify_grounding.py          grounding-token ablation (real / zeroed / shuffled)
condition_activation.py      read the hidden state three ways (yes-no / condition / oracle)
flip_threshold_robustness.py threshold sweep + threshold-free dispersion ratio
diag_ood.py, diag_readout.py the encoder-vs-readout diagnosis
figures.py                   regenerate figures/ from the result JSON
```

**Packaging for Hugging Face**

```
modeling_babymedgemma.py     transformers wrapper for the probe-1841 variant
modeling_babymedgemma_v2.py  transformers wrapper for the scaled grounded model (self-contained)
convert_to_hf.py             package the probe checkpoint
convert_v2_to_hf.py          package the scaled checkpoint
```

Results directories are documented in [`RESULTS.md`](RESULTS.md). Heavy artifacts
(`**/model.pt`, `cache/`) are on Hugging Face, not in git.


The model checkpoints (`**/model.pt`, ~4 GB) and the MedSigLIP feature cache
(`cache/medsiglip_feats.pt`, ~3 GB) are on Hugging Face, not in git:
**[saillab/babymedgemma](https://huggingface.co/saillab/babymedgemma)**.

## Load with transformers

The trained probe is packaged as a custom `transformers` model (loadable via
`trust_remote_code`); `modeling_babymedgemma.py` holds the config and model
classes.

Two variants are published. The **repo root** is the scaled grounded model that
transfers to unseen hospitals; **`probe-1841/`** is the controlled probe behind
experiments A to E above.

```python
import torch
from transformers import AutoModel
from PIL import Image

# scaled grounded model (default)
model = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True).eval()
input_ids, ans_pos = model.encode_question("is there pleural effusion?")
vision_features, ground = model.encode_images([Image.open("cxr.png").convert("RGB")])
logits = model(input_ids=input_ids, vision_features=vision_features,
               ground=ground, ans_pos=ans_pos).logits
print(model.config.id2label[int(logits.argmax(-1))])   # "yes" or "no"

# the dissertation probe (no grounding token, word-level vocabulary)
probe = AutoModel.from_pretrained("saillab/babymedgemma", subfolder="probe-1841",
                                  trust_remote_code=True).eval()
```

The two are **not interchangeable**: the A-E numbers reproduce only with
`probe-1841/`, whose training distribution they describe. The root model uses
MedGemma's own SentencePiece tokenizer pruned to 141 pieces (0.4% of the parameters)
rather than a hand-rolled word vocabulary, so unseen words decompose into pieces
instead of silently becoming padding. HF `feature_cache/` and `checkpoints/` belong
to the probe variant.

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

# 3. the full A-E grid across 8 GPUs
NANO_ARCH=gemma NANO_LR=5e-4 python run_all_gpus.py --run
```

Each framework letter maps to a grid tag and a result file:

| Framework | Script / grid tag | Result file |
|---|---|---|
| A Data provenance | `run_all_gpus.py` tag B | `results_gemma/B/*/result.json` |
| B Divergence | tag A (`experiment_a.py`) | `results_gemma/A/*/experiment_a.json` |
| C Causal patching | tag E (`experiment_e.py`) | `results_gemma/E/*/experiment_e.json` |
| D Architecture | tag C | `results_gemma/C/*/result.json` |
| E Grounding | tag D | `results_gemma/D/*/result.json` |
| SAE | `sae.py --arch gemma` | `results_gemma/sae_gemma/sae.json` |
| Jacobian lens | `jlens.py --arch gemma` | `results_gemma/jlens_gemma/jlens.json` |
| NIH transfer | `nih_demo.py --arch gemma` | `results/nih_demo/nih_demo.json` |

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.*
> Unpublished doctoral dissertation, University of New Haven.

## License and data

Code is released for research use. The probe is trained on MIMIC-CXR and
PadChest; use of the derived features and checkpoints is subject to those
datasets' data-use agreements. Not a medical device and not for clinical use.
