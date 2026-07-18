# Isolating the Origin of Paraphrase Sensitivity with a Synthetic Toy Vision-Language Model

Status: design (2026-07-17). Owner: Binesh Sadanadan. Substrate decision: synthetic toy first, real tiny model (nanoVLM-style) as a later transfer check.

## 1. Motivation

We have measured paraphrase sensitivity failure (PSF) in MedGemma-4B extensively: flip rates of 6.7 percent (MIMIC) and 13.1 percent (PadChest), a layer-17 "formality gate" feature (3818), a two-stage gate model (layer 17 feature 3818 upstream, layer 29 feature 12139 downstream), and a targeted low-rank adaptation over layers 15 to 19 that cuts the flip rate. Every one of those results is observational or single-intervention. We can patch MedGemma, but we cannot retrain its instruction tuning, cannot vary its architecture, and cannot re-run its data generation. So we can say feature 3818 correlates with flips and that patching it recovers 16.7 percent of out-of-distribution flips, but we cannot say what *originates* the sensitivity.

A tiny model trained on a synthetic world removes that ceiling. It lets us intervene on the three things that actually generate PSF (the training data, the architecture, and the activations) and re-run 30 to 50 seeds, converting correlational mechanistic claims into causal, reproducible ones.

The scientific claim this design is built to support: **PSF is not a single fixed property of a model; it is produced by an identifiable combination of weak cross-modal grounding and phrasing-correlated training data, decided in a localizable late-layer readout subspace.** The toy exists to prove or refute each clause of that sentence under full control.

## 2. What "origin" means: the five hypotheses

We decompose the pipeline (question tokens plus image, through fusion, through the language side, to the answer logits) into five candidate origins. They are not mutually exclusive; the toy measures how much each contributes when the others are held fixed.

| # | Candidate origin | Claim | Isolating intervention |
|---|---|---|---|
| H1 | Embedding / input | Paraphrases start far apart in embedding space and never reconcile | Within-cluster embedding dispersion at layer 0 predicts flips before any fusion |
| H2 | Language-side depth | Divergence persists or grows through the stack instead of collapsing to a semantic invariant | Layer-wise within-cluster dispersion identifies a band that tracks flips |
| H3 | Cross-modal grounding | When image evidence is weak the text prior fills the gap, and the text prior is phrasing-sensitive | Sweep image-dependence; flips concentrate on image-optional questions |
| H4 | Training data | Surface form is correlated with the answer distribution in the training data | Same architecture, three data regimes; PSF appears or vanishes with data alone |
| H5 | Readout amplification | Late layers turn tiny representational gaps into discrete answer flips | Activation patching finds a minimal late-layer subspace that restores the answer |

## 3. The synthetic world

### 3.1 Scene generator

A CLEVR-style renderer producing small raster images (for example 96 by 96 or 128 by 128 pixels) from a structured scene graph. Each scene is a set of objects with controlled attributes:

- shape: from a fixed vocabulary (for example circle, square, triangle, cross, ring)
- color: from a fixed vocabulary (for example red, green, blue, yellow, white)
- count: 0 to K objects of a given type
- spatial relation: left-of, right-of, above, below, adjacent
- one or two abstract "finding" markers standing in for pathology presence, so the question forms mirror the clinical yes/no presence questions in the real benchmark

The scene graph is the ground truth. Every question answer is a deterministic function of the scene graph, so there is no label noise and no hidden confound. The renderer, the scene graph, and the answer function are all stored with each sample.

### 3.2 Question types

Mirror the structure of the medical benchmark so findings transfer:

| Type | Example canonical form | Answer space |
|---|---|---|
| Presence (binary) | "Is there a red circle?" | yes / no |
| Count | "How many squares are present?" | 0..K |
| Attribute | "What color is the largest object?" | color vocabulary |
| Relation | "Is the blue square left of the red circle?" | yes / no |

Presence and relation questions are the primary PSF probe because they are binary, which is exactly where the real benchmark's flips are cleanest and where mixed non-binary questions previously contaminated the headline numbers.

### 3.3 Verified paraphrase sets

For each question type, N templated phrasings (target N = 8 to 12) generated from the same scene-graph query. Because each phrasing is a rendering of an identical formal query, the set is semantically identical **by construction**, not by post-hoc judgment. Within-cluster answer variance is therefore a model failure by definition, which is the property the real benchmark could never guarantee (paraphrase equivalence there rests on a language model's judgment).

Paraphrase axes to vary deliberately, so we can later attribute flips to a specific surface dimension:

- politeness / formality register ("Is there a red circle?" vs "Could you tell me whether a red circle appears?") — the toy analogue of the layer-17 formality gate
- syntactic frame (interrogative vs declarative-with-question-mark vs imperative "Identify whether...")
- lexical substitution (present / visible / shown / depicted)
- negation framing ("Is there no red circle?" mapped to the same underlying query with inverted expected answer, used only in the negation sub-study)

Each paraphrase carries metadata tags for these axes so Experiment A can regress flips on axis, not just on cluster.

### 3.4 The image-dependence knob (central control)

Every question is generated at a controllable image-dependence level d in [0, 1]:

- d = 1: the answer is fully determined by the image and cannot be guessed from the question. Achieved by balancing the dataset so that, conditioned on the question, the answer is 50/50 across the image distribution.
- d = 0: the answer is recoverable from the question text alone, because the training distribution makes one answer far more likely for that phrasing regardless of image (a deliberate spurious prior).
- intermediate d: interpolate the conditional answer imbalance.

This knob is the controlled, owned x-axis for the entire "consistency without image reliance is a false sense of safety" thesis. It is what MedGemma cannot give us.

## 4. The tiny model

Target 5 to 15 million parameters so that a full retrain finishes in minutes on one GPU and 30-plus seeds are cheap.

| Component | Default choice | Swappable for Experiment C |
|---|---|---|
| Vision encoder | small Vision Transformer (ViT) from scratch, 4 to 6 layers, patch 8 or 16, or a SigLIP-tiny initialization | frozen vs trained; from-scratch vs pretrained init |
| Fusion | prefix / prompt tokens (image tokens prepended to the text stream) | cross-attention vs prefix vs concatenation |
| Language side | decoder-only transformer, 4 to 6 layers, width 256 to 512, roughly GPT-nano scale | depth 2 / 4 / 6 / 8 |
| Head | tied-embedding language-model head over a small answer vocabulary | linear probe taps at every layer |

Everything is instrumented from the start: residual-stream activations at every layer are capturable (reuse the activation-capture machinery in `vlm_explain/`), and a linear answer probe can be attached at any layer.

Training recipe held fixed across the causal experiments: same optimizer, schedule, token budget, and stopping criterion, varying only the one factor under test. Report mean and standard deviation over 30 seeds for every headline number so that a mechanism claim carries an error bar rather than being a single anecdote.

## 5. Metrics

| Metric | Definition | Used in |
|---|---|---|
| Flip rate | Fraction of paraphrase clusters whose answers are not all identical | all |
| Within-cluster dispersion (per layer) | Mean pairwise cosine distance (or centered kernel alignment distance) among paraphrase representations of a cluster at a given layer | A, E |
| Grounding necessity gap | Accuracy at d = 1 minus accuracy from a text-only copy of the model with the image ablated | D |
| Phrasing mutual information | Mutual information between phrasing-axis identity and the answer logit, conditioned on the scene graph | A, B |
| Patch recovery rate | Fraction of flipped clusters restored to the canonical answer by patching a given layer or subspace | E |
| Minimal patch depth | Shallowest layer at which a patch of bounded rank restores the answer | E |

The flip-rate definition reuses the existing benchmark harness so the toy numbers are directly comparable to the MedGemma 6.7 / 13.1 percent figures. Note explicitly: keep binary (presence, relation) question types separate from count and attribute types when reporting the headline flip rate, because mixing non-binary questions is exactly what contaminated the earlier real-data headline.

## 6. Experiment series

Run in this order. A, B, and E on the toy are the causal core and are publishable on their own. D is the thesis-linked sweep. C is the architectural sweep.

### Experiment A — divergence trajectory (observational, first)

Purpose: localize where PSF originates versus where it is amplified, across the whole stack rather than at a single guessed layer.

Method: train one reference model. For every paraphrase cluster compute within-cluster dispersion at every layer, and regress cluster flip status on (dispersion at each layer, phrasing-axis tags). Report the layer band where dispersion first tracks flips and the axis that carries the most flip-predictive information.

Reads on the hypotheses: supports H1 if layer-0 dispersion already predicts flips; supports H2 if a mid or late band is where dispersion and flips couple; identifies whether the formality axis dominates, which is the toy analogue of feature 3818.

Expected outcome if the MedGemma story transfers: dispersion is small early, and flips couple to a late band, with the formality / register axis carrying disproportionate flip-predictive signal.

### Experiment B — data provenance (cleanest causal test)

Purpose: decide whether PSF is learned from data or is architectural. This is the experiment MedGemma cannot support.

Method: hold architecture, seeds, and training budget fixed. Train on three datasets that differ only in phrasing distribution:

1. Canonical: one phrasing per question type.
2. Paraphrase-augmented: heavy, balanced paraphrase coverage.
3. Adversarial: phrasing deliberately correlated with the answer (a specific register predicts "yes").

Compare flip rate and phrasing mutual information across the three, each over 30 seeds.

Reads on the hypotheses: supports H4 if regime (2) drives flips toward zero and regime (3) inflates them, holding everything else constant. A near-zero flip rate under (2) would be strong evidence that PSF is a data property that paraphrase augmentation removes, which also predicts why the targeted low-rank adaptation helps.

### Experiment C — architecture knockout

Purpose: test whether any architectural choice is necessary or sufficient for PSF.

Method: one factor at a time, all else fixed, on the paraphrase-augmented data regime: frozen versus trained vision encoder; fusion mechanism (prefix vs cross-attention vs concatenation); language-side depth (2 / 4 / 6 / 8). Report flip rate and grounding necessity gap per configuration.

Reads on the hypotheses: isolates H2 and part of H3. A depth threshold below which flips do not appear would bound where in the stack the amplification lives; a fusion mechanism that removes flips would implicate cross-modal binding directly.

### Experiment D — grounding sweep (thesis-linked)

Purpose: the controlled version of "consistency without image reliance is a false sense of safety."

Method: sweep image-dependence d across, for example, {0.0, 0.25, 0.5, 0.75, 1.0}. Plot flip rate and grounding necessity gap against d. Cross with Experiment B regimes so we can see whether adversarial phrasing correlation matters most exactly when the image is optional.

Reads on the hypotheses: supports H3 if flips concentrate at low d and shrink toward d = 1. The interaction with B tells us whether weak grounding is what lets phrasing-correlated data express itself as flips, which is the mechanistic sentence we are trying to prove.

### Experiment E — causal patching (capstone)

Purpose: name where the flip is decided.

Method: for flipped clusters, patch the paraphrase run's activations into the canonical run at each layer and at bounded-rank subspaces, and find the minimal locus that restores the canonical answer. Because the model is tiny, search the whole residual stream instead of betting on one feature. Compare to matched control patches (random same-norm directions, non-flip clusters) exactly as in the existing feature-3818 causal work.

Reads on the hypotheses: supports H5 if a shallow-as-possible, low-rank, late-layer subspace restores the answer while controls do not. This is the from-scratch, fully controlled analogue of the feature 3818 / feature 12139 gate story, and it either reproduces that two-stage structure under control or shows it was model-specific.

## 7. Decision table: what each result would let us claim

| Observation | Attribution |
|---|---|
| Layer-0 dispersion predicts flips (A) | Origin partly at embedding (H1) |
| Late-band dispersion couples to flips (A) | Amplification is language-side and deep (H2, H5) |
| Regime (2) drives flips to near zero (B) | PSF is data-induced and paraphrase augmentation removes it (H4) |
| Regime (3) inflates flips (B) | PSF is a learnable spurious phrasing-answer correlation (H4) |
| Flips concentrate at low d (D) | PSF is a grounding failure expressed through the text prior (H3) |
| B and D interact (adversarial phrasing worst at low d) | The core sentence: weak grounding lets phrasing-correlated data flip answers |
| Minimal late-layer low-rank patch restores answer (E) | The decision locus is a localizable readout subspace (H5) |

## 8. Deliverables and reuse

New code (to be scaffolded in a later step, not part of this design doc):

- `scripts/experiments/tiny_vlm/` — scene generator, paraphrase templater, tiny-VLM training loop, and the A/B/E measurement harness.
- Reuse the existing flip-rate metric and paraphrase infrastructure so toy numbers are comparable to the MedGemma benchmark.
- Reuse `vlm_explain/` activation capture and the patching machinery from the feature-3818 causal study for Experiments A and E.

Outputs: per-experiment JSON under `results/tiny_vlm/`, plus figures (divergence-by-layer, flip-vs-d, patch recovery-by-layer) generated with the existing figure scripts so they match paper style.

## 9. Transfer check (later, not part of this toy phase)

After A, B, and E hold on the toy, repeat A (divergence trajectory) and D (grounding sweep) on one real tiny model of the nanoVLM family (a small SigLIP encoder with a SmolLM2 language side, single-GPU trainable) using the real chest X-ray paraphrase data, to show the toy mechanism transfers to a real image-text model and lines up with the MedGemma-4B numbers. This closes the loop from full-control toy to real model to the 4B system already characterized.

## 10. Risks and controls

| Risk | Control |
|---|---|
| Toy is too easy, flip rate near zero, nothing to explain | Tune d and answer entropy until the reference model shows a flip rate in the range of the real benchmark (roughly 5 to 15 percent) before running the series |
| Paraphrase set not truly equivalent | Equivalence is by construction (identical scene-graph query), and is auditable from stored metadata |
| Findings are seed-specific | 30-plus seeds on every headline number, report mean and standard deviation |
| Patch recovery is spurious | Matched random-direction and non-flip control patches, same protocol as the existing 3818 study |
| Toy findings do not transfer | Section 9 transfer check is the explicit falsification step |
