# What the experiments prove (and what they don't)

A plain read of the baby-MedGemma study: what we ran, what each result
establishes, and where the claims stop. All numbers are baby-Gemma over 16 seeds
(`results_gemma/`).

## The question

Medical vision-language models often change their yes/no answer when a clinical
question is rephrased without changing its meaning. On a deployed model you can
find *where* that flip is decided and *what* feature carries it, but not *why* it
exists, because the model's pretraining data, architecture, and objective are
frozen together. To ask *why*, you need to intervene on one of those and read the
effect. That is what this probe is for.

## Why a probe, and why this one

The probe reflects MedGemma's architecture closely enough that a mechanism found
in it is credible for the deployed model: a **frozen `google/medsiglip-448`
encoder** (the same SigLIP family MedGemma uses) feeding a small **Gemma-3
decoder** (13.9M trainable parameters, with rotary position embeddings, RMSNorm,
and grouped-query attention: 6 query heads, 2 key-value heads), with the 256
image tokens prepended inline and read by the same causally-masked decoder. It is
trained on 1,841 binary presence questions over 1,775 real chest radiographs
(MIMIC-CXR and PadChest), each carrying register-tagged paraphrases.

Two properties make the probe useful:

1. **Language-side by construction.** The encoder is frozen and returns
   *identical* features for every paraphrase of a question, so any answer that
   changes across paraphrases is in the decoder, not the image. This is not
   measured; it is guaranteed by the setup.
2. **The origin is a knob.** Unlike the deployed model, we can vary the
   training-phrasing distribution while holding architecture, parameter budget,
   seed, and evaluation fixed. A difference in the flip rate is then an
   *identified cause*, not a correlation.

The probe reaches 88.8% accuracy and, like MedGemma, is text-reliant (75% of its
answers are unchanged when the image is removed, close to the deployed model's
81%), so it is faithful in behavior as well as architecture.

## The experiments and what each proves

**B — data provenance (the causal core).** We train the same probe under three
phrasing regimes that differ *only* in the questions it sees: *canonical* (one
fixed phrasing), *augmented* (every paraphrase), and *adversarial* (register tied
to the answer). The flip rate is set by the phrasing distribution alone:

| Regime | Trained on | Flip rate |
|---|---|---|
| Augmented | every paraphrase | **8.4%** |
| Canonical | one fixed phrasing | 30.3% |
| Adversarial | register tied to the answer | 30.4% |

Augmented separates from both narrow regimes at the maximum effect size
(Mann-Whitney U p = 1.4e-6, Cliff's delta = 1.00: every augmented seed flips less
than every seed of the other two). **This proves the training-phrasing
distribution is sufficient to produce and to remove paraphrase sensitivity, and
that broad coverage is the single largest lever against it.**

**E — causal patching (the mechanism).** For a flipped question, we transplant
the answer position along a single rank-1 direction (the difference between a
phrasing answered one way and one answered the other) at one layer at a time. It
restores the flipped answer with net recovery near 1.0 across the early layers,
while a norm-matched random direction and a non-flip-cluster control both leave
the answer unchanged (disruption 0.000). Crucially this holds for the *naturally
occurring* flips of the augmented regime (a mean of 18.8 per seed), not only the
injected adversarial ones (59.7 per seed). **This proves the flip is a low-rank,
language-side, readout-stage direction, decided at layers 0 to 1, and not an
artifact of the adversarial construction.**

**A — divergence.** Within-cluster representation dispersion couples to the flip
from the earliest layers (a point-biserial correlation near 0.64 at the input for
natural flips, near 0.90 throughout for adversarial), as expected of a decoder
whose rotary encoding separates paraphrases before the stack. Lexical
substitution drives the most naturally occurring flips.

**C — architecture.** Across decoder depths 2, 4, 6, and 8 the flip rate stays
within 7.6% to 9.4% at constant accuracy. **Depth is not the driver.**

**D — grounding sweep.** Weakening the visual pathway by dropping vision tokens
gives a non-monotonic flip-rate curve. **Inconclusive; no weak-grounding claim is
drawn from it.**

## The claim, stated exactly

The origin of paraphrase sensitivity is the **training-phrasing distribution**,
and it is executed as a **low-rank direction in the early language layers** that
read a fixed visual representation, not in the visual representation itself. The
mitigation follows: paraphrase augmentation is the lever (30.3% down to 8.4%),
and a targeted low-rank edit at those layers is the efficient parametric fix,
which is why a layers-15-to-19 low-rank adaptation reduces flips on the deployed
model while full fine-tuning does not.

## What this does NOT prove

The honest boundaries, stated plainly:

- **It is a controlled probe, not the deployed model.** The result is a
  *sufficiency* claim ("this cause is enough, under control") plus a
  *localization*, not proof that MedGemma-4B's paraphrase sensitivity has this
  exact origin. Its absolute layer index is not comparable to the deployed
  model's; only the qualitative account transfers.
- **It manipulates adaptation-stage data, not pretraining provenance.** No
  controlled model of this size can reach the pretraining distribution.
- **It grounds weakly (like MedGemma).** So it speaks to *where* the flip is
  decided and *why* it is learnable, not to how well the image is used. The
  grounding sweep (D) was inconclusive.
- **The two-types distinction did not reproduce.** In a hand-built model,
  coverage-flips and shortcut-flips separated; here the two narrow regimes are
  statistically indistinguishable (adversarial vs canonical p = 0.62), so the
  robust, architecture-independent finding is the augmentation lever itself, not
  a taxonomy of paraphrase-sensitivity types.
- **It is a narrow model, not a medical VQA system.** It has a small
  domain-specific word vocabulary (733 words from the chest-X-ray presence
  questions) and is loadable and runnable for inspection and reproduction only.
  It is not a general medical vision-language model and is not for clinical use.

## Why it still matters: triangulation

The probe is one leg of a three-method argument that no single method licenses.
On the deployed MedGemma-4B, a lens-free residual patch and the Jacobian lens put
the answer commit at layer 16, and a GemmaScope sparse autoencoder finds a
layer-17 register gate (Feature 3818). The probe adds the one thing those cannot:
a controlled origin. The three agree that the paraphrase flip is language-side,
decided in a narrow early-to-middle band, and set by the training-phrasing
distribution. Their agreement, not any one result, is the reason to believe it.
