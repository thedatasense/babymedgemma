# Catching flips and image-unreliance in baby-MedGemma

Can a deployment-time monitor tell, from a model's own forward pass, that a chest
X-ray answer is (a) unstable under rephrasing, or (b) not actually using the image?
This note works the question through for baby-MedGemma: the architectural limits, the
method space grounded in the hallucination-detection literature, a measured benchmark,
and a recommendation.

Scripts: `scripts/analysis/detect.py`. Results: `results_transfer/detect.json`.

---

## 1. The architectural constraint that reshapes the question

baby-MedGemma reads a single yes/no from the tied Gemma-3 language-model head at the
answer position, over a 141-piece vocabulary. It does not generate reasoning tokens.
That fixes what is even applicable:

| method family | applies here? | why |
|---|---|---|
| Forward-pass signals | Yes, this is the whole detection surface | the yes-minus-no margin and the hidden state behind it are all there is |
| Chain of thought | No, for the probe | it emits no text to reason in; chain of thought is a deployed-model method |
| Prompt engineering | Narrow | the closed vocabulary allows paraphrase selection and ensembling, not free prompting |

This is not a weakness to apologize for. It is why the probe is a clean testbed: both
failures reduce to two questions about one scalar (the margin `m = logit(yes) -
logit(no)`) and one vector (the answer-position hidden state).

The two targets are different problems and need different detectors:

- **Flip-proneness.** At inference you see one phrasing. Catching a flip means
  predicting, from that single case, that a rephrasing would change the answer.
- **Image-unreliance.** Predicting that the answer did not depend on the radiograph.
  Here the gold detector already exists: swap in another patient's image and see if the
  answer moves. The research question is whether anything cheaper predicts it.

---

## 2. Method space, grounded in the literature

Three detection paradigms dominate the language-model hallucination literature, and each
maps onto a signal baby-MedGemma exposes.

**Self-consistency / semantic entropy.** Sample or paraphrase, then measure disagreement;
high disagreement flags an unreliable answer. Semantic entropy formalizes this over
meaning-equivalence clusters (Farquhar et al., *Nature* 2024). SelfCheckGPT does it
zero-resource and black-box by sampling several responses and scoring their mutual
consistency (Manakul et al., EMNLP 2023). baby-MedGemma can run this directly, because it
carries a 48-template paraphrase bank: instantiate a question's paraphrases, read their
margins, and measure the spread.

**Internal-state probes.** Train a small classifier on a model's hidden activations to
predict whether an answer is wrong or unstable, at the cost of one forward pass plus a
fitted probe (for example, "Neural Probe-Based Hallucination Detection for Large Language
Models", 2025). baby-MedGemma exposes the answer-position residual stream directly.

**Single-pass confidence.** The distance of the answer from the decision boundary. For a
binary readout this is just `|m|`, and its softmax entropy is a monotone function of it,
so the two are one signal, not two.

**Chain of thought is the paradigm to be skeptical of.** The faithfulness literature
finds that a generated rationale often rationalizes a decision the model already made
rather than revealing it ("Analysing Chain of Thought Dynamics: Active Guidance or
Unfaithful Post-hoc Rationalisation?", 2025; FaithCoT-Bench, 2025). Our own companion
work reports chain of thought making text-only medical reasoning worse rather than better
(*Chain-of-Thought Backfires*, arXiv:2603.25960). The defensible position is that chain of
thought is a mitigation to test for backfiring on the deployed MedGemma-4B, not a reliable
flip-detector, and it is not available in the probe at all.

---

## 3. Benchmark design

The model is the scaled grounded baby-MedGemma (trained on NIH ChestX-ray14 and PadChest,
with MIMIC-CXR and VinDr-CXR held out entirely; native accuracy 0.748, transfer area under
the receiver operating characteristic curve 0.743 on MIMIC and 0.756 on VinDr). For each
(image, finding) cluster on the held-out and two unseen-hospital splits, two labels and
seven detectors are computed.

**Labels (gold).**

| label | definition | cost |
|---|---|---|
| FLIP | the predicted answer changes across the cluster's paraphrases | all phrasings |
| UNRELIANT | the predicted answer is unchanged when the image is swapped for another patient's | 2 passes |

**Detectors (score per cluster, from the original phrasing unless noted).**

| detector | cost | intuition |
|---|---|---|
| `\|margin\|` | 1 pass | small = near the boundary = flip-prone |
| entropy | 1 pass | monotone in `\|margin\|` |
| grounding-token contribution | 2 passes | `\|m(real ground) - m(zeroed ground)\|`, how much the pooled visual token moves the answer |
| image-ablation delta | 2 passes | `\|m(image) - m(all vision zeroed)\|` |
| image-swap delta | 2 passes | `\|m(image) - m(another patient's image)\|`; this is close to the UNRELIANT definition |
| paraphrase dispersion | k passes | standard deviation of the margin across the cluster's paraphrases (self-consistency) |
| hidden-state probe | 1 pass + fit | five-fold logistic regression on the answer-position hidden state |

Subsample sizes: 1,500 clusters (held-out), 450 (MIMIC), 1,500 (VinDr).

---

## 4. Results

Area under the receiver operating characteristic curve for each detector against each
target. Higher is a better detector.

**Catching FLIP** (base rate 4.9 to 6.2 percent):

| detector | cost | held-out | MIMIC | VinDr |
|---|---|---|---|---|
| **`\|margin\|`** | **1 pass** | **0.923** | **0.974** | **0.974** |
| entropy | 1 pass | equals `\|margin\|` | equals | equals |
| hidden-state probe | 1 pass + fit | 0.824 | 0.838 | 0.801 |
| paraphrase dispersion | k passes | 0.709 | 0.786 | 0.624 |
| grounding-token contribution | 2 passes | 0.570 | 0.593 | 0.630 |
| image-swap delta | 2 passes | 0.622 | 0.540 | 0.652 |

**Catching UNRELIANT** (base rate about 0.51 to 0.54; see caveat 2):

| detector | cost | held-out | MIMIC | VinDr |
|---|---|---|---|---|
| **image-swap delta** | **2 passes** | **0.827** | **0.907** | **0.856** |
| hidden-state probe | 1 pass + fit | 0.642 | 0.674 | 0.623 |
| grounding-token contribution | 2 passes | 0.507 | 0.596 | 0.520 |
| image-ablation delta | 2 passes | 0.355 | 0.614 | 0.430 |
| `\|margin\|` | 1 pass | 0.470 | 0.491 | 0.519 |

---

## 5. Findings

**Flips are catchable in a single forward pass, and the cheap detector beats the expensive
one.** The distance to the decision boundary catches flip-prone predictions at 0.92 to
0.97. It beats paraphrase dispersion (0.62 to 0.79), which costs k passes and even shares
its phrasings with the flip label, and it beats the hidden-state probe (0.80 to 0.84). The
reason is mechanical: a flip is a sign change of the margin across paraphrases, and that is
most likely when the margin already sits near zero. Proximity to the boundary is more
directly the thing than the spread of the distribution. This runs against the standard
semantic-entropy and SelfCheckGPT prescription that several samples are needed. Here one
margin suffices, because a paraphrase moves the answer through a decision boundary rather
than across semantic-meaning clusters, which is the regime those methods were built for.

**Image-unreliance is not catchable from a single forward pass.** The margin is at chance
for it (0.47 to 0.52), so a confident prediction is no more likely to be image-grounded
than an unconfident one. The single-pass hidden probe reaches only 0.62 to 0.67. Reliable
detection needs the two-pass image-swap (0.83 to 0.91). There is no cheap substitute.

**The detector table is the thesis in miniature.** Confidence catches inconsistency (0.97)
and is blind to grounding (0.50). That is the "consistency without grounding" dissociation
stated at the level of what a monitor can and cannot see: a flip-rate or confidence gate
will pass a model that answers from the text, because the property it cannot measure is the
one that matters.

---

## 6. Caveats

1. **Entropy is not a separate detector.** It is a monotone function of `|margin|`; the two
   are one signal. (In the raw log its column reads near 0.03 because of a sign convention;
   with the correct orientation it equals the 0.97 of `|margin|`.)

2. **The UNRELIANT label is confounded and its absolute numbers should be read as
   directional.** "Unchanged answer under a random swap" is contaminated by chance label
   agreement: on a per-finding balanced set a random swapped image carries the same
   ground-truth answer about half the time, so even a perfectly grounded model looks about
   50 percent "unreliant." The relative ordering (swap much greater than probe, probe
   greater than margin at chance) is trustworthy; the absolute values are not. The clean
   version swaps only to an opposite-label image, forcing a grounded model to change its
   answer, and is a short rerun.

3. **Single split, single seed of the trained model.** The detector AUROCs are averaged
   over clusters within one trained checkpoint per split, not across training seeds.

4. **Domain and scope.** baby-MedGemma answers binary presence questions about fourteen
   chest findings with a closed vocabulary; these detector conclusions are about that
   setting, not open clinical dialogue.

---

## 7. Recommendation

| target | detector | deployment |
|---|---|---|
| flip-prone answer | `\|margin\| < tau` | a single-pass confidence gate; abstain or escalate below the threshold. Area under the curve about 0.95, no extra inference cost |
| image-unreliant answer | image-swap delta | the two-pass four-quadrant screen; no cheap single-pass substitute exists |

For chain of thought: not applicable to the probe, and to be treated on the deployed model
as a mitigation whose backfiring must be measured, not as a detector.

---

## 8. Open questions and next steps

| item | why it matters | cost |
|---|---|---|
| Opposite-label image swap | firms up the UNRELIANT row onto a clean target | a few minutes |
| Abstention curve for the `\|margin\|` gate | turns the 0.95 into an operating point: accuracy against coverage, and the accuracy of the retained set at each threshold | small |
| Held-out-phrasing dispersion | scores self-consistency on phrasings disjoint from the flip label, removing the mild advantage it currently has | one pass |
| Port the `\|margin\|` flip-detector to MedGemma-4B | the deployed model already stores per-paraphrase margins under `results/miccai/`, so the single-pass gate can be evaluated where it would actually run | reuse of cached margins |
| Multi-seed detector variance | confidence intervals on the AUROCs | rerun over seeds |

---

## References

- Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. (2024). Detecting hallucinations in large
  language models using semantic entropy. *Nature*, 630, 625 to 630.
  https://doi.org/10.1038/s41586-024-07421-0
- Manakul, P., Liusie, A., and Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box
  Hallucination Detection for Generative Large Language Models. *EMNLP 2023*.
  https://doi.org/10.18653/v1/2023.emnlp-main.557
- FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning
  (2025). https://doi.org/10.48550/arxiv.2510.04040
- Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc
  Rationalisation? (2025). https://doi.org/10.48550/arxiv.2508.19827
- Neural Probe-Based Hallucination Detection for Large Language Models (2025).
  https://doi.org/10.48550/arxiv.2512.20949
- Sadanandan, B. (2026). Chain-of-Thought Backfires (companion, text-only language models).
  arXiv:2603.25960.

Citations were located through a literature search; the four preprints are cited at the
level of their stated titles and abstracts, which were retrieved but not read in full.

---

## Appendix A. Stage 0 on the grounding-route flip: real in aggregate, not identifiable per case

A grounding-route flip is a stable-answer case that is visually grounded under one
phrasing and image-unreliant under another. We ran three pilots; the first two reached
wrong conclusions, and the third carried a label-direction bug that is now fixed. All are
recorded here so the reasoning is auditable.

- Pilot 1 reported a clean falsification. Wrong: a sign-reversal event rule, mismatched
  centering of the numerator and the noise floor, and prompts the all-48-template
  checkpoint had trained on.
- Pilot 2 reported a 25 percent route-drop prevalence. Wrong: the event was scored on
  two-way-centered V, where "near zero" means ordinary-for-this-case, not unreliant;
  after centering a near-zero prompt is guaranteed, so the rate was trivially inflated.
- Pilot 3 (`scripts/analysis/route_flip_pilot.py`) scores the held-out checkpoints
  (trained on 24 templates) on their 24 unseen templates, over NIH validation images
  matched on view, sex, age band, and the non-target finding vector; uses a centered
  cross-draw covariance for the population interaction and a RAW-V event with delta_0 and
  delta_g locked on a development same-versus-same null; draws several independent match
  sets from the ten nearest eligible controls; saves every matched-image margin; and
  splits the development and test sets by patient. Its first write dropped the y label
  direction from the grounding numerator G, so negative-finding cases contributed the
  wrong evidence sign. The numbers below are after restoring `G = y * (m_i - median m_opp)`
  in `draws_V` and `null_V`, moving the development and test split to the patient level,
  and replacing the degenerate zero-event bootstrap interval with a Clopper-Pearson upper
  bound. Restoring the sign lowers the interaction slightly and does not change either
  verdict.

Two questions separate cleanly.

Population interaction (does wording modulate reliance at all), over 3 held-out seeds:

| quantity | pooled | reading |
|---|---|---|
| sigma2_route (cross-draw covariance) | +0.062 | reproducible interaction |
| corr across independent match draws | +0.56 | the per-case pattern reproduces |
| case-specific share | 82 percent | idiosyncratic to the case, not finding-level |

Individual route-drop event (raw V, null-locked thresholds, all draws must agree,
patient-split test set): zero observed in every seed, 0 over 345 patient-seed evaluations.
The case pool is built once and scored against the three checkpoints, so the seeds share
patients; the 345 is the sum of the per-seed test counts (105, 125, 115) and covers 225
unique patient IDs. Because a bootstrap of zero events returns a degenerate [0, 0], we
report a Clopper-Pearson 97.5th upper bound with the patient as the unit: 2.9 to 3.5 percent
per seed, near 3 percent. The
reason the event does not fire is not that grounding and unreliance fail to co-occur; it is
that confidently grounded never happens. The same-versus-same null |V| (no true finding
change) reaches as high as the real V (null 97.5th percentile 4.9 to 7.2; real |V| maxes at
5.8 to 6.6), so swapping to an opposite-label image moves the margin by about as much as
swapping to a same-label image. Per case, finding-selective reliance is not separable from
generic patient-change noise.

Both facts hold at once because the reproducible prompt modulation is small (within-case V
standard deviation about 0.22) and shows only after averaging over many cases and draws,
while the absolute per-case reliance is estimated with a plus or minus 3 to 5 noise floor.

Verdict.

| claim | verdict |
|---|---|
| prompt-dependent reliance interaction exists (population) | supported (corr 0.56, 82 percent case-specific, 3 seeds) |
| a stable answer can hide small route variation | real but small (within-case V sd about 0.22) |
| individual grounded-to-unreliant route-drops occur at a measurable rate | no (0 over 345 patient-seed evaluations, 225 unique patients, patient-level upper bound near 3 percent) |
| a per-case single-pass route-drop monitor is warranted | no; the event has about zero detectable prevalence |

This is why the method is counterfactual and not causal: observationally matched
radiographs give a real population signal but cannot identify per-case grounding, because
the matched-counterfactual noise is as large as the finding-selective signal. Pursuing the
route flip would require cleaner per-case interventions, and same-patient longitudinal
appear-or-resolve pairs are preferable to finding inpainting because they avoid edit
artifacts, not observational matches. The deployable deliverable remains the margin flip
gate; the reliance signal is real only at the finding-population level, measurable but not
certifiable per case. Per-case records with every matched-image margin are saved to
`results_transfer/route_flip_records.json`.

In one statement: prompt wording reproducibly modulates finding-selective reliance at the
population level; most of this interaction is case-specific; under the observational
matching design no per-case grounded-to-unreliant route drops were identified, with an
approximate patient-level 95 percent upper bound near 3 percent; and generic
patient-change variation prevents a dependable per-case grounding label.

Literature note: the related-work verification (semantic entropy, SelfCheckGPT, SVAR,
Blind-Image Contrastive Ranking, the VLM reliability probe, and the medical
visual-chain-of-thought results) was contributed by the project author and not
independently re-run here.
