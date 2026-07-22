"""Diagnose the near-chance NIH accuracy: is it the encoder (no signal) or the
decoder (collapses to a text prior out of distribution)?

Reports, on the balanced NIH set baby-Gemma sees:
  1. baby-Gemma overall accuracy, fraction answered "yes", per-finding accuracy.
  2. A logistic-regression linear probe on the frozen mean-pooled MedSigLIP
     features (the ceiling the encoder makes available), per-finding and pooled.

If the probe is well above chance while baby-Gemma sits at chance and answers a
near-constant label, the visual signal is present and the decoder is not using it
out of distribution: the weak-grounding failure, not a blind encoder.
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

from babygemma.paths import ROOT as HERE

from babygemma.dataset import build_tokenizer, NanoDataset
from babygemma.gemma_model import BabyGemmaVLM
from babygemma import metrics as Mx
from babygemma import nih as ND

SEED, PER = 0, 120
NIH_CACHE = os.path.join(HERE, "cache", "nih_feats.pt")


def get_nih(tok):
    records, _ = ND.build_nih_records(tok, PER, SEED)
    paths = sorted({r["image_path"] for r in records})
    if os.path.exists(NIH_CACHE):
        fc = torch.load(NIH_CACHE, map_location="cpu")
    else:
        fc = ND.precompute_nih(paths)
        torch.save(fc, NIH_CACHE)
    return records, fc


def main():
    tok = build_tokenizer()
    records, fc = get_nih(tok)
    # cluster examples for baby-Gemma prediction (originals only, to score answers)
    from babygemma.dataset import Ex
    ex = [Ex(r["image_path"], r["question"], r["answer"], "original", i) for i, r in enumerate(records)]
    finding = [r["question"] for r in records]
    y = np.array([1 if r["answer"] == "yes" else 0 for r in records])

    # 1. baby-Gemma predictions
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6,
                     yes_id=tok.stoi["yes"], no_id=tok.stoi["no"]).cuda()
    m.load_state_dict(torch.load(f"{HERE}/results_gemma/B/augmented_s0/model.pt", map_location="cuda"), strict=False)
    m.eval()
    ds = NanoDataset(ex, tok, fc)
    preds, ans, _, _ = Mx.predict(m, ds, "cuda")
    preds = np.array(preds); ans = np.array(ans)
    bg_acc = float((preds == ans).mean()); bg_yes = float(preds.mean()); lab_yes = float(y.mean())
    print(f"[baby-Gemma NIH] acc={bg_acc:.3f}  frac_yes={bg_yes:.3f}  (label frac_yes={lab_yes:.3f})")

    # per-finding baby-Gemma
    print("  per finding (acc | frac_yes | n):")
    for q in sorted(set(finding)):
        mask = np.array([f == q for f in finding])
        a = float((preds[mask] == ans[mask]).mean()); fy = float(preds[mask].mean())
        print(f"    {q:28s} acc={a:.2f}  yes={fy:.2f}  n={int(mask.sum())}")

    # 2. linear probe on frozen mean-pooled features
    X = np.stack([fc["feats"][fc["index"][r["image_path"]]].float().mean(0).numpy() for r in records])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    pp = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=5)
    probe_acc = float((pp == y).mean())
    print(f"\n[linear probe on frozen MedSigLIP features] pooled 5-fold acc={probe_acc:.3f}")
    print("  per finding (probe acc):")
    for q in sorted(set(finding)):
        mask = np.array([f == q for f in finding])
        if mask.sum() >= 20 and len(set(y[mask])) == 2:
            ppf = cross_val_predict(LogisticRegression(max_iter=2000), X[mask], y[mask], cv=3)
            pa = float((ppf == y[mask]).mean())
            print(f"    {q:28s} probe_acc={pa:.2f}  n={int(mask.sum())}")


if __name__ == "__main__":
    main()
