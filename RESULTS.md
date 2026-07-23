# Results directories

Every number in the README and in the dissertation's Thrust-3 section is reproduced
from the JSON here. Per-run model weights (`**/model.pt`) are not tracked; they live on
Hugging Face (`saillab/babymedgemma`) or regenerate from the training scripts.

| Directory | Model | Contents |
|---|---|---|
| `results_gemma/` | 1,841-question probe (`probe-1841` on HF) | Experiments A to E (`B/`, `A/`, `C/`, `D/`, `E/`), the SAE and Jacobian-lens JSON, and `grounding/` (the NIH diagnosis: AUC-vs-accuracy, the grounding-token ablation, condition activation, threshold robustness) |
| `results_ground/` | 1,841-question probe **with** the grounding token | The 48-checkpoint grounded grid used by the NIH ablation |
| `results_scaled/` | first scale-up (8,627 questions) | Superseded by the transfer runs; kept for the record |
| `results_transfer/` | scaled grounded model (repo root on HF) | `v2_aug_s0/` bundled checkpoint result, `eval_transfer.json` (val / MIMIC / VinDr), `detect.json` + `detect_percase.json` (single-pass flip and image-reliance detectors), `route_flip_pilot.json` + `route_flip_records.json` (the grounding-route null result), `sae/`, `jlens/` |
| `results_transfer_grid/` | scaled model, provenance grid | Experiment A at scale (3 regimes x 8 seeds) + `dispersion.json` (threshold sweep) |
| `results_transfer_heldout/` | scaled model, held-out phrasing | Trained on 24 phrasings, scored only on the 24 withheld |
| `results_transfer_patch/` | scaled model, rank-1 patching | Experiment C on the grounded model |

The LoRA invariance test on the deployed MedGemma-4B lives in the main dissertation
repository, not here (`scripts/analysis/lora_invariance_test.py`).
