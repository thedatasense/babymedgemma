"""Ask the condition, not the binary question.

"Is there pneumothorax? yes/no" forces a threshold. "Is the pneumothorax concept
activated?" does not. This reads ONE hidden state (the answer position) through
three different readouts and compares how much finding-evidence each exposes:

  1. yes-no margin      h.W[yes] - h.W[no]        the binary decision (what we scored before)
  2. condition token    h.W[finding words]        zero-shot concept activation, NO fitting
  3. oracle probe       best linear direction     upper bound on evidence present in h

If (2) or (3) is far above (1), the decoder HAS the evidence and the binary readout
is throwing it away. In particular this re-tests the claim that the ungrounded
baseline is "blind": that was concluded from readout (1) alone.

    CUDA_VISIBLE_DEVICES=0 python condition_activation.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import Ex, NanoDataset, build_tokenizer, collate, tokenize_words
from gemma_model import BabyGemmaVLM
import nih_demo as ND

SEEDS = [0, 1, 2]
PER, DATA_SEED = 120, 0
OUT = os.path.join(HERE, "results_gemma", "grounding", "condition_activation.json")


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


@torch.no_grad()
def hidden_and_logits(model, ds, use_ground):
    """Answer-position final hidden states [N, dim]."""
    model.eval()
    H = []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {"ground": b["ground"].cuda()} if use_ground else {}
        _, acts = model(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(),
                        capture=True, **kw)
        h = acts[-1]                                   # final layer residual stream
        idx = (b["ans_pos"].cuda() + model.n_img)
        H.append(h[torch.arange(h.shape[0], device=h.device), idx].float().cpu().numpy())
    return np.concatenate(H)


def main():
    tok = build_tokenizer()
    records, _ = ND.build_nih_records(tok, PER, DATA_SEED)
    feats = torch.load(os.path.join(HERE, "cache", "nih_feats.pt"), map_location="cpu")
    d = torch.load(os.path.join(HERE, "cache", "nih_pooled.pt"), map_location="cpu")
    pc = {"index": {p: i for i, p in enumerate(d["paths"])}, "pooled": d["pooled"]}
    gdim = d["pooled"].shape[1]

    ex = [Ex(r["image_path"], r["question"], r["answer"], "original", c) for c, r in enumerate(records)]
    y = np.array([1 if r["answer"] == "yes" else 0 for r in records])
    q = np.array([r["question"] for r in records])
    findings = sorted(set(q.tolist()))
    ds_b = NanoDataset(ex, tok, feats, pooled=None)
    ds_g = NanoDataset(ex, tok, feats, pooled=pc)
    print(f"[cond] NIH {len(records)} originals, {len(findings)} findings", flush=True)

    # condition-word token ids per finding question ("Is there pleural effusion?" -> [pleural, effusion])
    cond_ids = {}
    for f in findings:
        phrase = f.replace("Is there ", "").rstrip("?")
        ids = [tok.stoi[w] for w in tokenize_words(phrase) if w in tok.stoi]
        cond_ids[f] = ids
    missing = [f for f, i in cond_ids.items() if not i]
    if missing:
        print(f"  WARNING: no vocab tokens for {missing}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    res = {"per_seed": []}
    for seed in SEEDS:
        row = {"seed": seed}
        for tag, rd, ug, ds in [("baseline", "results_gemma", False, ds_b),
                                ("grounded", "results_ground", True, ds_g)]:
            ck = f"{HERE}/{rd}/B/augmented_s{seed}/model.pt"
            if not os.path.exists(ck):
                continue
            m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6,
                             yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                             use_ground=ug, ground_dim=gdim).cuda()
            m.load_state_dict(torch.load(ck, map_location="cuda"), strict=False)
            H = hidden_and_logits(m, ds, ug)
            W = m.gemma.get_input_embeddings().weight.detach().float().cpu().numpy()  # tied head

            a_yn, a_cond, a_probe = {}, {}, {}
            for f in findings:
                mk = q == f
                h, yy = H[mk], y[mk]
                a_yn[f] = auc(h @ (W[tok.stoi["yes"]] - W[tok.stoi["no"]]), yy)
                if cond_ids[f]:
                    a_cond[f] = auc(h @ W[cond_ids[f]].mean(0), yy)
                if len(set(yy.tolist())) > 1 and mk.sum() >= 20:
                    cv = StratifiedKFold(5, shuffle=True, random_state=0)
                    p = cross_val_predict(LogisticRegression(max_iter=3000), h, yy,
                                          cv=cv, method="predict_proba")[:, 1]
                    a_probe[f] = auc(p, yy)
            row[tag] = {
                "auc_yesno": float(np.nanmean(list(a_yn.values()))),
                "auc_condition_token": float(np.nanmean(list(a_cond.values()))),
                "auc_oracle_probe": float(np.nanmean(list(a_probe.values()))),
                "per_finding_condition": a_cond, "per_finding_probe": a_probe,
            }
            r = row[tag]
            print(f"  seed {seed} {tag:9s} yes/no={r['auc_yesno']:.3f}  "
                  f"condition={r['auc_condition_token']:.3f}  oracle_probe={r['auc_oracle_probe']:.3f}",
                  flush=True)
        res["per_seed"].append(row)

    print("\n=== MEAN over seeds (margin AUC by readout) ===")
    for tag in ("baseline", "grounded"):
        rows = [r[tag] for r in res["per_seed"] if tag in r]
        if not rows:
            continue
        print(f"  {tag:9s} yes/no={np.mean([r['auc_yesno'] for r in rows]):.3f}  "
              f"condition={np.mean([r['auc_condition_token'] for r in rows]):.3f}  "
              f"oracle_probe={np.mean([r['auc_oracle_probe'] for r in rows]):.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
