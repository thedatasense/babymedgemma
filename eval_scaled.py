"""Evaluate the scaled model on its clean held-out split, by AUC as well as accuracy.

Reports overall and per-source (mimic/padchest vs nih), plus per-finding AUC on the
NIH portion so it can be set beside MedSigLIP's own zero-shot ceiling (AUC 0.734,
accuracy 0.681 measured earlier on NIH under the same binary framing).

Images are split-disjoint, so this is a clean held-out measurement -- but NIH is now
IN training, so this is in-distribution generalization, not out-of-distribution
transfer. Do not quote it as OOD.

    CUDA_VISIBLE_DEVICES=0 python eval_scaled.py
"""
from __future__ import annotations

import json, os, sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_index import build_index
from dataset import Ex, NanoDataset, build_tokenizer, collate, load_feature_cache, load_pooled_cache
from gemma_model import BabyGemmaVLM
import metrics as Mx

CKPT = os.path.join(HERE, "results_scaled", "aug_s0_long", "model.pt")
OUT = os.path.join(HERE, "results_scaled", "eval_scaled.json")


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


@torch.no_grad()
def run(model, ds, use_ground):
    model.eval()
    marg, pred, ans, clu = [], [], [], []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {"ground": b["ground"].cuda()} if use_ground else {}
        lg, _ = model(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(), **kw)
        marg.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
        pred += lg.argmax(-1).cpu().tolist()
        ans += b["answer"].tolist()
        clu += b["cluster_id"].tolist()
    return np.concatenate(marg), np.array(pred), np.array(ans), clu


def main():
    tok = build_tokenizer()
    idx = build_index()
    feats, pc = load_feature_cache(), load_pooled_cache()
    test = [r for r in idx if r["split"] == "test"]
    print(f"[eval] {len(test)} held-out questions, vocab={len(tok)}", flush=True)

    ex, src, q, om = [], [], [], []
    for c, r in enumerate(test):
        ex.append(Ex(r["image_path"], r["question"], r["answer"], "original", c))
        src.append(r.get("source", "?")); q.append(r["question"]); om.append(True)
        for p in r["paraphrases"]:
            ex.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), c))
            src.append(r.get("source", "?")); q.append(r["question"]); om.append(False)
    src = np.array(src); q = np.array(q); om = np.array(om)

    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, yes_id=tok.stoi["yes"],
                     no_id=tok.stoi["no"], use_ground=True,
                     ground_dim=pc["pooled"].shape[1]).cuda()
    m.load_state_dict(torch.load(CKPT, map_location="cuda"), strict=False)

    ds = NanoDataset(ex, tok, feats, pooled=pc)
    marg, pred, ans, clu = run(m, ds, True)

    res = {}
    o = om
    res["overall"] = {"n": int(o.sum()), "acc": float((pred[o] == ans[o]).mean()),
                      "auc": auc(marg[o], ans[o]), "flip": Mx.flip_rate(list(pred), clu)}
    print(f"\n  overall      n={res['overall']['n']:5d}  acc={res['overall']['acc']:.3f}  "
          f"AUC={res['overall']['auc']:.3f}  flip={res['overall']['flip']:.3f}")

    for s in sorted(set(src.tolist())):
        mk = o & (src == s)
        if mk.sum() < 10:
            continue
        res[s] = {"n": int(mk.sum()), "acc": float((pred[mk] == ans[mk]).mean()),
                  "auc": auc(marg[mk], ans[mk])}
        print(f"  {s:13s} n={mk.sum():5d}  acc={res[s]['acc']:.3f}  AUC={res[s]['auc']:.3f}")

    print("\n  per-finding (NIH portion of held-out set):")
    per = {}
    for f in sorted(set(q[o & (src == 'nih')].tolist())):
        mk = o & (q == f) & (src == "nih")
        if mk.sum() < 20:
            continue
        per[f] = {"n": int(mk.sum()), "acc": float((pred[mk] == ans[mk]).mean()),
                  "auc": auc(marg[mk], ans[mk])}
        print(f"    {f:32s} n={mk.sum():4d}  acc={per[f]['acc']:.2f}  AUC={per[f]['auc']:.3f}")
    res["per_finding_nih"] = per
    if per:
        ma = np.nanmean([v["auc"] for v in per.values()])
        print(f"    {'MEAN':32s}            AUC={ma:.3f}   (MedSigLIP zero-shot ceiling 0.734)")
        res["nih_mean_per_finding_auc"] = float(ma)

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
