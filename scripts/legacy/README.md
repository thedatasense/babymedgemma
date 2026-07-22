# Legacy scripts

These analyzed the earlier **1,841-question probe** (retired), which used a word-level
tokenizer. Their results are preserved under `results_gemma/` and are cited in the
dissertation (the AUC-versus-accuracy "measurement lesson", the grounding-token
ablation, the SAE and Jacobian-lens corroborations). They are kept for provenance and
do not run against the current package default, whose tokenizer is MedGemma's
SentencePiece. The latest model's equivalents are in `scripts/analysis/` and
`scripts/experiments/`.
