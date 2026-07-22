"""Test the readout hypothesis: does the frozen encoder carry the NIH finding
signal, so that near-chance baby-Gemma is a readout failure, not a blind encoder?

Probe A: logistic regression on MedSigLIP's OWN pooled image embedding
         (get_image_features at 448, the trained attention-pooling head that its
         zero-shot classification uses) -> the encoder's real capacity.
Probe B: the same on the patch tokens baby-Gemma actually receives (the cached
         896 grid), max-pooled -> whether the signal survives into its input.

Compared against baby-Gemma's per-finding behavior. AUC to match the MedSigLIP
report card; accuracy on the balanced set for comparability with baby-Gemma.
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

from babygemma.paths import ROOT as HERE

from babygemma.dataset import build_tokenizer
from babygemma import nih as ND

SEED, PER = 0, 120


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    tok = build_tokenizer()
    records, _ = ND.build_nih_records(tok, PER, SEED)
    findings = sorted({r["question"] for r in records})
    paths = [r["image_path"] for r in records]
    y = np.array([1 if r["answer"] == "yes" else 0 for r in records])

    # Probe A: MedSigLIP proper pooled image embedding at 448 (its native setup)
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained("google/medsiglip-448")
    enc = AutoModel.from_pretrained("google/medsiglip-448").to("cuda").eval()
    uniq = sorted(set(paths))
    emb = {}
    from PIL import Image
    B = 32
    with torch.no_grad():
        for i in range(0, len(uniq), B):
            chunk = uniq[i:i + B]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda")
            f = enc.get_image_features(pixel_values=px)     # [b, proj] trained pooling head
            f = torch.nn.functional.normalize(f, dim=-1)
            for p, v in zip(chunk, f.cpu().numpy()):
                emb[p] = v
    XA = np.stack([emb[p] for p in paths])

    # Probe B: cached 896 patch tokens baby-Gemma receives, max-pooled over tokens
    fc = torch.load(os.path.join(HERE, "cache", "nih_feats.pt"), map_location="cpu")
    XB = np.stack([fc["feats"][fc["index"][p]].float().amax(0).numpy() for p in paths])

    def probe(X, label):
        print(f"\n[{label}]  (AUC | acc, per finding, balanced n)")
        aucs, accs = [], []
        for q in findings:
            m = np.array([r["question"] == q for r in records])
            if m.sum() < 20 or len(set(y[m])) < 2:
                continue
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pr = cross_val_predict(LogisticRegression(max_iter=3000), X[m], y[m], cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y[m], pr); acc = ((pr > 0.5).astype(int) == y[m]).mean()
            aucs.append(auc); accs.append(acc)
            print(f"    {q:28s} AUC={auc:.2f}  acc={acc:.2f}  n={int(m.sum())}")
        print(f"    {'MEAN':28s} AUC={np.mean(aucs):.3f}  acc={np.mean(accs):.3f}")

    probe(XA, "Probe A: MedSigLIP pooled image embedding (448, trained pooling head)")
    probe(XB, "Probe B: baby-Gemma's input patch tokens (896 grid, max-pooled)")


if __name__ == "__main__":
    main()
