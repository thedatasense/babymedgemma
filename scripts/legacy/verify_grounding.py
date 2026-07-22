"""Verify the grounding-injection negative result.

Two things must hold before we can claim "grounding injection does not rescue OOD":
  1. Robustness: the NIH gap between baseline and grounded holds across seeds.
  2. Not a strawman: the grounded model must actually READ the token. We test that
     causally by re-running the same grounded checkpoint on NIH with the grounding
     token (a) real, (b) zeroed, (c) shuffled across images (same distribution,
     wrong image). If real > shuffled, the token carries image-specific signal the
     model uses; the size of that gap is how much grounding actually buys OOD.

Inference only, over checkpoints already saved by the grids.

    CUDA_VISIBLE_DEVICES=0 python verify_grounding.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma.paths import ROOT as HERE

from babygemma.dataset import Ex, NanoDataset, build_tokenizer, collate
from babygemma.gemma_model import BabyGemmaVLM
from babygemma import metrics as Mx
from babygemma import nih as ND

SEEDS = [0, 1, 2, 3, 4]
PER, DATA_SEED = 120, 0
OUT = os.path.join(HERE, "results_gemma", "grounding", "verify_grounding.json")


@torch.no_grad()
def predict(model, ds, use_ground, mode="real", gen=None):
    model.eval()
    preds, ans, clu = [], [], []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {}
        if use_ground:
            g = b["ground"].cuda()
            if mode == "zero":
                g = torch.zeros_like(g)
            elif mode == "shuffle":
                g = g[torch.randperm(g.shape[0], generator=gen, device=g.device)]
            kw["ground"] = g
        logits, _ = model(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(), **kw)
        preds += logits.argmax(-1).cpu().tolist()
        ans += b["answer"].tolist()
        clu += b["cluster_id"].tolist()
    return np.array(preds), np.array(ans), clu


def load_model(tok, ckpt, use_ground, ground_dim):
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6,
                     yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                     use_ground=use_ground, ground_dim=ground_dim).cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    return m.eval()


def main():
    tok = build_tokenizer()
    records, _ = ND.build_nih_records(tok, PER, DATA_SEED)
    feats = torch.load(os.path.join(HERE, "cache", "nih_feats.pt"), map_location="cpu")
    d = torch.load(os.path.join(HERE, "cache", "nih_pooled.pt"), map_location="cpu")
    pc = {"index": {p: i for i, p in enumerate(d["paths"])}, "pooled": d["pooled"]}
    gdim = d["pooled"].shape[1]

    ex, om, q = [], [], []
    for c, r in enumerate(records):
        ex.append(Ex(r["image_path"], r["question"], r["answer"], "original", c))
        om.append(True); q.append(r["question"])
        for p in r["paraphrases"]:
            ex.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), c))
            om.append(False); q.append(r["question"])
    om = np.array(om); q = np.array(q)
    ds_g = NanoDataset(ex, tok, feats, pooled=pc)
    ds_b = NanoDataset(ex, tok, feats, pooled=None)
    print(f"[verify] NIH {len(records)} clusters, {len(ex)} examples, ground_dim={gdim}", flush=True)

    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for seed in SEEDS:
        base_ck = f"{HERE}/results_gemma/B/augmented_s{seed}/model.pt"
        gnd_ck = f"{HERE}/results_ground/B/augmented_s{seed}/model.pt"
        if not (os.path.exists(base_ck) and os.path.exists(gnd_ck)):
            continue
        r = {"seed": seed}
        p, a, c = predict(load_model(tok, base_ck, False, gdim), ds_b, False)
        r["baseline"] = {"acc": float((p[om] == a[om]).mean()), "flip": Mx.flip_rate(list(p), c),
                         "frac_yes": float(p[om].mean())}
        gm = load_model(tok, gnd_ck, True, gdim)
        for mode in ("real", "zero", "shuffle"):
            p, a, c = predict(gm, ds_g, True, mode, gen)
            r[f"grounded_{mode}"] = {"acc": float((p[om] == a[om]).mean()),
                                     "flip": Mx.flip_rate(list(p), c),
                                     "frac_yes": float(p[om].mean())}
            if mode == "real":
                r["per_finding_real"] = {str(x): float((p[(q == x) & om] == a[(q == x) & om]).mean())
                                         for x in sorted(set(q.tolist()))}
        rows.append(r)
        print(f"  seed {seed}: base {r['baseline']['acc']:.3f} | grounded real "
              f"{r['grounded_real']['acc']:.3f} zero {r['grounded_zero']['acc']:.3f} "
              f"shuffle {r['grounded_shuffle']['acc']:.3f}", flush=True)

    print("\n=== NIH accuracy (mean over seeds) ===")
    for k in ("baseline", "grounded_real", "grounded_zero", "grounded_shuffle"):
        accs = [r[k]["acc"] for r in rows]; flips = [r[k]["flip"] for r in rows]
        print(f"  {k:18s} acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}   flip={np.mean(flips):.3f}")
    real = np.array([r["grounded_real"]["acc"] for r in rows])
    shuf = np.array([r["grounded_shuffle"]["acc"] for r in rows])
    base = np.array([r["baseline"]["acc"] for r in rows])
    print(f"\n  token IS read (real - shuffle) = {np.mean(real - shuf):+.3f}")
    print(f"  grounding buys over baseline   = {np.mean(real - base):+.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
