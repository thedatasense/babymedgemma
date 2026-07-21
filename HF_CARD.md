---
license: other
license_name: health-ai-developer-foundations
license_link: https://developers.google.com/health-ai-developer-foundations/terms
library_name: transformers
tags:
  - medical
  - chest-x-ray
  - vision-language-model
  - interpretability
  - mechanistic-interpretability
  - research
base_model:
  - google/medsiglip-448
  - google/medgemma-4b-it
pipeline_tag: visual-question-answering
---

# baby-MedGemma

A small, **MedGemma-faithful** chest-X-ray model: a frozen `google/medsiglip-448`
encoder run at 896 pixels feeding a compact **Gemma-3** decoder, with the yes/no
answer read from the tied language-model head. Built to study *where paraphrase
sensitivity in medical vision-language models originates*.

- Code and the full experiment write-up: **[github.com/thedatasense/babymedgemma](https://github.com/thedatasense/babymedgemma)**
- Interactive write-up: **[bineshkumar.me/phd-thesis/causality](https://bineshkumar.me/phd-thesis/causality/)**

> **Scope.** A research probe, **not a clinical model**. It answers binary presence
> questions about 14 chest findings. Not a medical device; not for clinical use.

## Two variants

| | default (this repo root) | `probe-1841/` |
|---|---|---|
| Purpose | grounded model that **transfers to unseen hospitals** | the **controlled probe** behind the dissertation's causal experiments |
| Training data | 107k per-finding-balanced questions (NIH + PadChest) | 1,841 questions (MIMIC-CXR + PadChest) |
| Tokenizer | MedGemma SentencePiece, pruned to 141 pieces | word-level, 734 words |
| Paraphrases | 48 templates across 4 phenomena | 4.8 generated per question |
| Grounding token | yes (MedSigLIP pooled embedding) | no |

**Use `probe-1841/` to reproduce the dissertation's Section 5.5 numbers** (augmentation
lever 29.5% -> 9.5%, rank-1 causal patching). Those results are specific to that
training distribution and do not transfer to the default weights.

## What the default model does

Trained on NIH + PadChest, evaluated on **MIMIC and VinDr held out entirely** (no
image, and no hospital, seen in training). Every split is balanced per finding, so a
text-only model scores exactly 0.500 and all accuracy above that is visual.

| split | n | accuracy | AUC | text-only floor | flip rate |
|---|---|---|---|---|---|
| in-distribution (held-out images) | 15,112 | 0.748 | 0.827 | 0.500 | 0.061 |
| **MIMIC (unseen hospital)** | 450 | 0.671 | **0.743** | 0.500 | 0.049 |
| **VinDr (unseen hospital)** | 5,478 | 0.686 | **0.756** | 0.500 | 0.060 |

For reference, MedSigLIP's own zero-shot binary performance on this task is AUC 0.734,
and an earlier version of this probe scored **AUC 0.500 — indistinguishable from blind
— on unseen data**. Per finding, pneumothorax transfers essentially unchanged (0.899
in-distribution vs 0.903 on VinDr); **pleural thickening inverts on VinDr (0.332,
below chance)**, most likely a label-definition mismatch, and should not be trusted.

Accuracy alone cannot tell a seeing model from a blind one here: a blind model, a
model given a *shuffled* grounding token, and a genuinely seeing model all score
~0.50 accuracy while their AUCs are 0.500 / 0.506 / 0.604. Report AUC.

## Load

```python
import torch
from transformers import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True).eval()

input_ids, ans_pos = model.encode_question("is there pleural effusion?")
vision_features, ground = model.encode_images([Image.open("cxr.png").convert("RGB")])
logits = model(input_ids=input_ids, vision_features=vision_features,
               ground=ground, ans_pos=ans_pos).logits
print(model.config.id2label[int(logits.argmax(-1))])      # "yes" or "no"
```

The dissertation probe loads the same way with `subfolder="probe-1841"` (it takes no
`ground` argument). `encode_images` needs gated access to `google/medsiglip-448`.
`feature_cache/` and `checkpoints/` in this repo belong to the **probe** variant.

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.* PhD
> dissertation, University of New Haven.
