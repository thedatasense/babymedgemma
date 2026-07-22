"""Can we catch flips and image-unreliance in baby-MedGemma at inference?

baby-Gemma reads a single yes/no from its LM head, so there is no chain of thought to
inspect; the only inference-time signal is the yes-minus-no margin and the hidden state
behind it. This benchmark scores every candidate detector by how well it predicts, per
(image, finding) cluster:

  FLIP       : the prediction changes across the cluster's paraphrases  (gold: all phrasings)
  UNRELIANT  : the prediction is unchanged when the image is swapped for
               another patient's                                        (gold: 2 passes)

Detectors, cheapest first:
  |margin|            1 forward pass  (near the boundary -> flip-prone)
  entropy             1 pass          (monotone in |margin|; included for completeness)
  ground_contrib      2 passes        |m(real ground) - m(zeroed ground)|: how much the
                                       pooled visual token moves the answer
  image_ablate        2 passes        |m(image) - m(image and ground zeroed)|
  image_swap          2 passes        |m(image) - m(another patient's image)|  (= UNRELIANT gold)
  para_dispersion     k passes        SD of the margin across the cluster's paraphrases
  hidden_probe        1 pass + fit    logistic probe on the answer-position hidden state

Reports AUROC of each detector against each target, per split.

    python scripts/analysis/detect.py
"""
from __future__ import annotations

import json, os, random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from babygemma.paths import ROOT
from babygemma.data_index import build_index
from babygemma.dataset import build_tokenizer, load_feature_cache, load_pooled_cache
from babygemma.gemma_model import BabyGemmaVLM

CKPT = os.path.join(ROOT, "results_transfer", "v2_aug_s0", "model.pt")
OUT = os.path.join(ROOT, "results_transfer", "detect.json")
N_PER_SPLIT = {"val": 1500, "ood_mimic": 450, "ood_vindr": 1500}
SPLITS = ["val", "ood_mimic", "ood_vindr"]


def auroc(score, label):
    label = np.asarray(label); score = np.asarray(score)
    if len(set(label.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


@torch.no_grad()
def margins_hidden(model, vis, tok, ap, ground):
    """-> (margin [B], last-layer answer-position hidden [B, dim])."""
    logits, acts = model(vis, tok, ap, capture=True, ground=ground)
    m = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
    idx = ap + model.n_img
    h = acts[-1][torch.arange(vis.shape[0], device=vis.device), idx].float().cpu().numpy()
    return m, h


def main():
    tok = build_tokenizer()
    idx = build_index()
    feats, pc = load_feature_cache(), load_pooled_cache()
    gdim = pc["pooled"].shape[1]
    dev = "cuda"
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, max_len=tok.max_len,
                     yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                     use_ground=True, ground_dim=gdim).to(dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()

    def feat(path):
        return feats["feats"][feats["index"][path]].float()

    def gnd(path):
        return pc["pooled"][pc["index"][path]].float()

    res = {}
    for split in SPLITS:
        rows = [r for r in idx if r["split"] == split
                and r["image_path"] in feats["index"] and r["image_path"] in pc["index"]]
        rng = random.Random(0)
        if len(rows) > N_PER_SPLIT[split]:
            rows = rng.sample(rows, N_PER_SPLIT[split])
        pool = [r["image_path"] for r in rows]
        print(f"\n[{split}] {len(rows)} clusters", flush=True)

        sig = {k: [] for k in ["abs_margin", "entropy", "ground_contrib",
                               "image_ablate", "image_swap", "para_dispersion"]}
        lab_flip, lab_unrel, H = [], [], []
        Z = torch.zeros(1, 256, gdim if False else 1152, device=dev)  # placeholder shape guard

        for r in rows:
            p = r["image_path"]
            v = feat(p).unsqueeze(0).to(dev)
            g = gnd(p).unsqueeze(0).to(dev)
            paras = [r["question"]] + [q["text"] for q in r["paraphrases"]]
            ids = torch.stack([torch.tensor(tok.encode(t)[0]) for t in paras]).to(dev)
            aps = torch.tensor([tok.encode(t)[1] for t in paras]).to(dev)
            B = len(paras)
            vb = v.expand(B, -1, -1)
            gb = g.expand(B, -1)
            with torch.no_grad():
                mp, hp = margins_hidden(m, vb, ids, aps, gb)          # all phrasings
                # original phrasing = index 0; the cheap detectors use only it
                m0 = mp[0]
                # ablations on the original phrasing
                id0, ap0 = ids[:1], aps[:1]
                m_zero_g, _ = margins_hidden(m, v, id0, ap0, torch.zeros_like(g))
                m_zero_all, _ = margins_hidden(m, torch.zeros_like(v), id0, ap0, torch.zeros_like(g))
                sp = rng.choice(pool)
                vs, gs = feat(sp).unsqueeze(0).to(dev), gnd(sp).unsqueeze(0).to(dev)
                m_swap, _ = margins_hidden(m, vs, id0, ap0, gs)

            preds = (mp > 0).astype(int)
            lab_flip.append(int(len(set(preds.tolist())) > 1))
            lab_unrel.append(int((m0 > 0) == (m_swap[0] > 0)))
            py = 1 / (1 + np.exp(-m0))
            sig["abs_margin"].append(-abs(m0))                       # low |margin| -> flip-prone
            sig["entropy"].append(-(py*np.log(py+1e-9)+(1-py)*np.log(1-py+1e-9)) * -1)
            sig["ground_contrib"].append(-abs(m0 - m_zero_g[0]))     # small delta -> unreliant
            sig["image_ablate"].append(-abs(m0 - m_zero_all[0]))
            sig["image_swap"].append(-abs(m0 - m_swap[0]))
            sig["para_dispersion"].append(float(mp.std()))          # high SD -> flip
            H.append(hp[0])

        H = np.stack(H); lab_flip = np.array(lab_flip); lab_unrel = np.array(lab_unrel)
        # hidden-state probe: 5-fold logistic on the answer-position hidden state
        def probe(y):
            if len(set(y.tolist())) < 2:
                return float("nan")
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pr = cross_val_predict(LogisticRegression(max_iter=2000), H, y, cv=cv,
                                   method="predict_proba")[:, 1]
            return auroc(pr, y)

        r = {"n": len(rows), "flip_rate": float(lab_flip.mean()),
             "unreliant_rate": float(lab_unrel.mean()), "detect_flip": {}, "detect_unreliant": {}}
        for k, s in sig.items():
            r["detect_flip"][k] = auroc(s, lab_flip)
            r["detect_unreliant"][k] = auroc(s, lab_unrel)
        r["detect_flip"]["hidden_probe"] = probe(lab_flip)
        r["detect_unreliant"]["hidden_probe"] = probe(lab_unrel)
        res[split] = r
        print(f"  flip_rate={r['flip_rate']:.3f}  unreliant_rate={r['unreliant_rate']:.3f}")
        print("  AUROC catch FLIP:      " + "  ".join(f"{k} {v:.2f}" for k, v in r["detect_flip"].items()))
        print("  AUROC catch UNRELIANT: " + "  ".join(f"{k} {v:.2f}" for k, v in r["detect_unreliant"].items()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
