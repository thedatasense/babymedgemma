"""Transfer evaluation: trained on NIH + PadChest, tested on unseen hospitals.

Reports accuracy AND AUC for each split, plus the text-only floor. Accuracy alone
cannot separate a blind model from a seeing one under a shifted decision threshold
(a blind model, a shuffled-grounding model and a genuinely seeing model all scored
~0.50 accuracy earlier while their AUCs were 0.500 / 0.506 / 0.604), so every number
here is reported both ways.

  val        in-distribution generalization (NIH + PadChest, disjoint images)
  ood_mimic  unseen hospital, 450 questions   (noisy, few findings)
  ood_vindr  unseen hospital, 5,478 questions (carries the transfer claim)

    CUDA_VISIBLE_DEVICES=0 python eval_transfer.py
"""
from __future__ import annotations

import json, os, sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma.paths import ROOT as HERE

from babygemma.data_index import build_index
from babygemma.dataset import Ex, NanoDataset, build_tokenizer, collate, load_feature_cache, load_pooled_cache
from babygemma.gemma_model import BabyGemmaVLM
from babygemma import metrics as Mx

CKPT = os.environ.get("NANO_CKPT", os.path.join(HERE, "results_transfer", "aug_s0", "model.pt"))
OUT = os.path.join(HERE, "results_transfer", "eval_transfer.json")
SPLITS = ["val", "ood_mimic", "ood_vindr"]


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


@torch.no_grad()
def run(model, ds, zero=False):
    model.eval()
    marg, pred, ans, clu = [], [], [], []
    for b in DataLoader(ds, batch_size=512, collate_fn=collate):
        v, g = b["vision"].cuda(), b["ground"].cuda()
        if zero:
            v, g = torch.zeros_like(v), torch.zeros_like(g)
        lg, _ = model(v, b["tokens"].cuda(), b["ans_pos"].cuda(), ground=g)
        marg.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
        pred += lg.argmax(-1).cpu().tolist()
        ans += b["answer"].tolist()
        clu += b["cluster_id"].tolist()
    return np.concatenate(marg), np.array(pred), np.array(ans), clu


def main():
    tok = build_tokenizer()
    idx = build_index()
    feats, pc = load_feature_cache(), load_pooled_cache()
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, yes_id=tok.stoi["yes"],
                     no_id=tok.stoi["no"], use_ground=True,
                     ground_dim=pc["pooled"].shape[1]).cuda()
    m.load_state_dict(torch.load(CKPT, map_location="cuda"), strict=False)
    print(f"[eval] {CKPT}", flush=True)

    res = {}
    print(f"\n{'split':11s} {'n':>6s} {'acc':>6s} {'AUC':>6s} {'text-only':>10s} {'flip':>6s}")
    for sp in SPLITS:
        rows = [r for r in idx if r["split"] == sp]
        if not rows:
            continue
        ex, q, om = [], [], []
        for c, r in enumerate(rows):
            ex.append(Ex(r["image_path"], r["question"], r["answer"], "original", c))
            q.append(r["question"]); om.append(True)
            for p in r["paraphrases"]:
                ex.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), c))
                q.append(r["question"]); om.append(False)
        q = np.array(q); om = np.array(om)
        ds = NanoDataset(ex, tok, feats, pooled=pc)
        marg, pred, ans, clu = run(m, ds)
        _, pz, _, _ = run(m, ds, zero=True)
        o = om
        per = {}
        for f in sorted(set(q[o].tolist())):
            mk = o & (q == f)
            if mk.sum() >= 20:
                per[f] = {"n": int(mk.sum()), "acc": float((pred[mk] == ans[mk]).mean()),
                          "auc": auc(marg[mk], ans[mk])}
        res[sp] = {
            "n": int(o.sum()),
            "acc": float((pred[o] == ans[o]).mean()),
            "auc": auc(marg[o], ans[o]),
            "auc_per_finding_mean": float(np.nanmean([v["auc"] for v in per.values()])) if per else float("nan"),
            "text_only_acc": float((pz[o] == ans[o]).mean()),
            "flip": Mx.flip_rate(list(pred), clu),
            "per_finding": per,
        }
        r = res[sp]
        print(f"{sp:11s} {r['n']:6d} {r['acc']:6.3f} {r['auc']:6.3f} {r['text_only_acc']:10.3f} {r['flip']:6.3f}")

    print("\nper-finding AUC:")
    findings = sorted({f for sp in res for f in res[sp]["per_finding"]})
    print(f"  {'finding':30s} " + "".join(f"{s:>12s}" for s in SPLITS if s in res))
    for f in findings:
        cells = ""
        for sp in SPLITS:
            if sp not in res:
                continue
            v = res[sp]["per_finding"].get(f)
            cells += f"{v['auc']:12.3f}" if v else f"{'-':>12s}"
        print(f"  {f:30s} {cells}")
    for sp in SPLITS:
        if sp in res:
            print(f"  {'MEAN ('+sp+')':30s} {res[sp]['auc_per_finding_mean']:12.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
