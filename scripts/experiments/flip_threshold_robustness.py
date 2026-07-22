"""Is the native flip-rate result a threshold artifact?

The NIH finding showed argmax metrics are threshold-dominated. Flip rate is also an
argmax statistic, and it is the spine of this dissertation, so it must be checked.

Two tests on the NATIVE (in-distribution) eval set, over the seeded checkpoints:

(a) THRESHOLD SWEEP. Recompute flip rate while shifting the decision boundary by an
    offset b: a cluster flips iff sign(margin + b) varies within it. Sweep b across
    the native margin distribution. If the ordering augmented < canonical ~ adversarial
    holds at EVERY offset, the result is not an artifact of where the boundary sits.

(b) THRESHOLD-FREE STATISTIC. Paraphrase dispersion ratio =
        mean_c[ SD of margins within cluster c ] / SD[ cluster-mean margins ]
    No decision boundary appears anywhere in it. If it reproduces the same ordering,
    paraphrase sensitivity is a property of the representation, not the threshold.

    CUDA_VISIBLE_DEVICES=0 python flip_threshold_robustness.py
"""
from __future__ import annotations

import json, os, sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma.paths import ROOT as HERE

from babygemma.data_index import build_index
from babygemma.dataset import (NanoDataset, build_tokenizer, collate, load_feature_cache,
                     make_eval_clusters)
from babygemma.gemma_model import BabyGemmaVLM

REGIMES = ["augmented", "canonical", "adversarial"]
SEEDS = list(range(int(os.environ.get("NANO_DISP_SEEDS", "16"))))
RESULTS_DIR = os.environ.get("NANO_DISP_RESULTS", os.path.join(HERE, "results_gemma", "B"))
USE_GROUND = bool(os.environ.get("NANO_GROUND"))
OUT = os.environ.get("NANO_DISP_OUT",
                     os.path.join(HERE, "results_gemma", "grounding", "flip_threshold_robustness.json"))


@torch.no_grad()
def margins_and_clusters(model, ds):
    model.eval()
    m, c = [], []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {"ground": b["ground"].cuda()} if USE_GROUND else {}
        lg, _ = model(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(), **kw)
        m.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
        c += b["cluster_id"].tolist()
    return np.concatenate(m), np.array(c)


def flip_at(margin, clusters, b):
    """Fraction of clusters whose predicted label varies once the boundary shifts by b."""
    pred = (margin + b) > 0
    by = defaultdict(list)
    for p, c in zip(pred, clusters):
        by[c].append(bool(p))
    return float(np.mean([len(set(v)) > 1 for v in by.values()]))


def dispersion_ratio(margin, clusters):
    """Threshold-free: within-cluster margin spread relative to between-cluster spread."""
    by = defaultdict(list)
    for m, c in zip(margin, clusters):
        by[c].append(m)
    within = [np.std(v) for v in by.values() if len(v) > 1]
    between = np.std([np.mean(v) for v in by.values()])
    return float(np.mean(within) / (between + 1e-8))


def main():
    tok = build_tokenizer()
    index = build_index()
    feats = load_feature_cache()
    ds = NanoDataset(make_eval_clusters(index,
                     split=os.environ.get("NANO_EVAL_SPLIT", "test")), tok, feats)
    print(f"[flip] native eval: {len(ds)} examples", flush=True)

    offsets = np.linspace(-2, 2, 17)   # in units of native margin SD
    res = {"offsets_in_sd": offsets.tolist(), "per_regime": {}}
    raw = defaultdict(lambda: defaultdict(list))

    for regime in REGIMES:
        for seed in SEEDS:
            ck = f"{RESULTS_DIR}/{regime}_s{seed}/model.pt"
            if not os.path.exists(ck):
                continue
            m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6,
                             max_len=getattr(tok, "max_len", 32),
                             yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                             use_ground=USE_GROUND, ground_dim=1152).cuda()
            m.load_state_dict(torch.load(ck, map_location="cuda"), strict=False)
            margin, clusters = margins_and_clusters(m, ds)
            sd = margin.std()
            raw[regime]["flip0"].append(flip_at(margin, clusters, 0.0))
            raw[regime]["disp"].append(dispersion_ratio(margin, clusters))
            raw[regime]["curve"].append([flip_at(margin, clusters, o * sd) for o in offsets])
        n = len(raw[regime]["flip0"])
        print(f"  {regime:12s} n={n} flip@0={np.mean(raw[regime]['flip0'])*100:.1f}% "
              f"dispersion={np.mean(raw[regime]['disp']):.3f}", flush=True)

    print("\n=== (a) flip rate (%) vs decision-boundary offset (SD units) ===")
    print("  offset  " + "".join(f"{r[:4]:>9s}" for r in REGIMES) + "   ordering")
    ok_all = True
    for i, o in enumerate(offsets):
        vals = [np.mean([c[i] for c in raw[r]["curve"]]) * 100 for r in REGIMES]
        aug, can, adv = vals
        ok = aug < can and aug < adv
        ok_all &= ok
        print(f"  {o:+5.2f}  " + "".join(f"{v:8.1f}%" for v in vals) +
              f"   {'aug lowest' if ok else 'ORDER BROKEN'}")

    print("\n=== (b) threshold-free paraphrase dispersion ratio ===")
    for r in REGIMES:
        d = np.array(raw[r]["disp"])
        print(f"  {r:12s} {d.mean():.3f} +/- {d.std():.3f}")

    from scipy import stats
    def cliff(a, b):
        n = len(a) * len(b)
        return (sum(x > y for x in a for y in b) - sum(x < y for x in a for y in b)) / n
    aug_d = raw["augmented"]["disp"]
    for r in ("canonical", "adversarial"):
        p = stats.mannwhitneyu(aug_d, raw[r]["disp"], alternative="two-sided").pvalue
        print(f"    augmented vs {r}: p={p:.2e} cliff={cliff(aug_d, raw[r]['disp']):+.2f}")

    print(f"\nVERDICT: ordering preserved at ALL offsets = {ok_all}")
    res["per_regime"] = {r: {k: (np.array(v).tolist() if k != 'curve'
                                 else np.array(v).mean(0).tolist())
                             for k, v in raw[r].items()} for r in REGIMES}
    res["ordering_preserved_all_offsets"] = bool(ok_all)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
