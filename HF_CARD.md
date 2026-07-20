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
  - google/gemma-3-4b-it
pipeline_tag: visual-question-answering
---

# baby-MedGemma

A small, **MedGemma-faithful probe** for causally isolating *where paraphrase
sensitivity in medical vision-language models originates*. It pairs a frozen
`google/medsiglip-448` encoder (run at 896 pixels, the MedGemma resolution) with a
small trained **Gemma-3** decoder, so a mechanism found here transfers to
MedGemma-4B, while letting the one experiment the frozen deployed model cannot
support run: **varying the training-phrasing distribution one factor at a time.**

- Code, five-experiment write-up, and figures: **[github.com/thedatasense/babymedgemma](https://github.com/thedatasense/babymedgemma)**
- Interactive write-up: **[bineshkumar.me/phd-thesis/causality](https://bineshkumar.me/phd-thesis/causality/)**

> **Scope.** A controlled research probe, **not a general medical VQA model**. It
> has a 734-word domain vocabulary (chest-X-ray presence questions plus the yes/no
> answer tokens) and grounds weakly, so it is loadable and runnable for inspection
> and reproduction only. It supports a *sufficiency* claim about the cause of
> paraphrase sensitivity, not proof that the deployed MedGemma-4B has this exact
> origin. Not a medical device; not for clinical use.

## Fidelity to MedGemma-1.5

Matched to the deployment stack ([arXiv:2604.05081](https://arxiv.org/pdf/2604.05081)):
frozen MedSigLIP encoder, 896-pixel input, 256 image tokens, image tokens
projected and prepended inline, a Gemma-3 decoder (rotary position embeddings,
RMSNorm, grouped-query attention, GeGLU, QK-norm), and the yes/no decision read
from the tied Gemma-3 LM head. Simplified on purpose for the experiment: 13.9M
decoder parameters trained from scratch, supervised training, and a 734-word
closed vocabulary, so the training-phrasing distribution becomes a controllable
input rather than a frozen prior.

## What it establishes

| | Result |
|---|---|
| Cause (data provenance) | Paraphrase augmentation cuts the flip rate from 29.5% (one fixed phrasing) to 9.5% (every paraphrase), Cliff's delta = 1.00, 16 seeds |
| Mechanism (causal patching) | A single rank-1, language-side direction in the early layers restores flipped answers (net recovery near 1.0); random and non-flip controls do nothing |
| Corroboration | An unsupervised sparse-autoencoder feature recovers that direction (\|cosine\| 0.74, ahead of principal-component analysis at 0.52); a Jacobian lens separates flipping from stable paraphrases (8.9x, from layer 0) |
| Ruled out | Decoder depth (flip rate 9.1% to 10.2% across depths 2 to 8) and weak grounding (inconclusive) |

The bundled weights are the **augmented seed-0** checkpoint (88.5% native accuracy,
8.2% flip rate).

## Load

```python
import torch
from transformers import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True).eval()
input_ids, ans_pos = model.encode_question("is there cardiomegaly ?")
vision_features = model.encode_images([Image.open("cxr.png").convert("RGB")])  # frozen MedSigLIP at 896px
logits = model(input_ids=input_ids, vision_features=vision_features, ans_pos=ans_pos).logits
print(model.config.id2label[int(logits.argmax(-1))])   # "yes" or "no"
```

`encode_images` needs gated access to `google/medsiglip-448`. The precomputed
896-pixel feature cache (`feature_cache/medsiglip_feats.pt`) and the per-run
`BabyGemmaVLM` checkpoints (`checkpoints/`) are in this repo; the code to
reproduce every experiment is on GitHub.

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.* PhD
> dissertation, University of New Haven.
