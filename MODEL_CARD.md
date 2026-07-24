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
- Live single-pass flip-detector demo: **[bineshkumar.me/phd-thesis/explorer/#margin-demo](https://bineshkumar.me/phd-thesis/explorer/#margin-demo)**

> **Scope.** A research probe, **not a clinical model**. It answers binary presence
> questions about 14 chest findings. Not a medical device; not for clinical use.

## What it does

Trained on NIH + PadChest, evaluated on **MIMIC and VinDr held out entirely** (no
image, and no hospital, seen in training). Every split is balanced per finding, so a
text-only model scores exactly 0.500 and all accuracy above that is visual.

| split | n | accuracy | AUC | text-only floor | flip rate |
|---|---|---|---|---|---|
| in-distribution (held-out images) | 15,112 | 0.748 | 0.827 | 0.500 | 0.061 |
| **MIMIC (unseen hospital)** | 450 | 0.671 | **0.743** | 0.500 | 0.049 |
| **VinDr (unseen hospital)** | 5,478 | 0.686 | **0.756** | 0.500 | 0.060 |

### What it establishes about paraphrase sensitivity

Training the same architecture under three phrasing distributions, changing nothing
else (8 seeds each; text-only accuracy 50.0-50.3% in all three, so none can exploit
an answer prior):

| training regime | flip rate | scored on **unseen** phrasings | accuracy |
|---|---|---|---|
| every paraphrase | **4.8%** | **26.6%** | 75.3% |
| one fixed phrasing | 67.1% | 65.9% | 66.8% |
| register tied to the answer | 88.4% | 87.4% | 58.1% |

Broad coverage separates from both narrow regimes at Cliff's delta = 1.00
(p = 1.6e-4), and the two narrow regimes separate from each other (p = 3.1e-4,
delta = 0.97), so a phrasing shortcut is a distinct and larger harm than narrow
coverage alone. The second column trains on half the paraphrase bank and scores only
on the 24 phrasings withheld, separating invariance from familiarity with wording
already seen: **26.6% against 65.9% is the honest size of the lever.**

A rank-1 direction at the answer position restores flipped answers (net recovery
0.98-1.00 across layers 0-4; random-direction control 0.00-0.02; non-flip disruption
0.000-0.006), so the flip is a low-rank, language-side, readout-stage effect. It is
not an artifact of where the decision boundary sits: the ordering holds at all 17
offsets across +/-2 SD of the margin, and a threshold-free dispersion statistic
reproduces it (0.023 / 0.498 / 1.362, delta = -1.00).

For reference, MedSigLIP's own zero-shot binary performance on this task is AUC 0.734,
and an earlier version of this probe scored **AUC 0.500 — indistinguishable from blind
— on unseen data**. Per finding, pneumothorax transfers essentially unchanged (0.899
in-distribution vs 0.903 on VinDr); **pleural thickening inverts on VinDr (0.332,
below chance)**, most likely a label-definition mismatch, and should not be trusted.

Accuracy alone cannot tell a seeing model from a blind one here: a blind model, a
model given a *shuffled* grounding token, and a genuinely seeing model all score
~0.50 accuracy while their AUCs are 0.500 / 0.502 / 0.604. Report AUC.

### The discarded margin is a single-pass flip detector

Because the answer is the sign of the yes-minus-no margin and a flip is a sign change,
a margin near zero marks a paraphrase-unstable answer. The absolute margin of one
forward pass therefore ranks flip-prone questions at no cost beyond the answer itself,
and it beats the detectors that cost more (`scripts/analysis/detect.py`):

| detector | passes | in-distribution | MIMIC | VinDr |
|---|---|---|---|---|
| **absolute margin** | **1** | **0.923** | **0.974** | **0.974** |
| paraphrase self-consistency | k | 0.709 | 0.786 | 0.624 |
| hidden-state probe | 1 + fit | 0.826 | 0.838 | 0.801 |

The other failure, an answer that ignores the image, has **no** single-pass signal: the
margin is at chance for it (0.470 to 0.519), because a confident blind answer looks
exactly like a confident grounded one. Only a second pass that swaps in another
patient's image detects it (0.827 to 0.907). A finer question, whether a stable answer
quietly changes how much it uses the image from one phrasing to another, is real at the
population level (cross-draw covariance 0.062, between-draw correlation 0.56, 82%
case-specific over three seeds) but **not identifiable per case** (zero grounded-to-unreliant
transitions over 345 patient-seed evaluations, patient-level upper bound near 3%). Image
reliance is a population property here, not a per-prediction label. Try it in the
**[live margin-gate demo](https://bineshkumar.me/phd-thesis/explorer/#margin-demo)**.

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

`encode_images` needs gated access to `google/medsiglip-448`.

## Repository structure

```
config.json  model.safetensors  modeling_babymedgemma.py   the model + self-contained wrapper
```

Training and analysis code, and the exact scripts that produced every number here, are on
GitHub: [thedatasense/babymedgemma](https://github.com/thedatasense/babymedgemma).

## Citation

> Sadanandan, B. (2026). *Paraphrase Sensitivity in Medical Vision-Language
> Models: Measurement, Mechanisms, Mitigation, and Deployment Safety.* PhD
> dissertation, University of New Haven.
