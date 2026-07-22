"""Is the near-chance NIH result a BLINDNESS failure or a THRESHOLD failure?

Accuracy on a balanced set punishes a miscalibrated threshold as hard as blindness.
The MedSigLIP report card is AUC (a ranking measure), so score baby-Gemma the same
way and separate the two:

  1. baby-Gemma NIH AUC from its yes/no margin (logit_yes - logit_no). If AUC is
     well above 0.5 while accuracy sits at 0.5, the model ranks findings correctly
     and only its DECISION THRESHOLD failed to transfer.
  2. baby-Gemma accuracy under an oracle per-finding threshold -> the ceiling
     recalibration alone would buy.
  3. MedSigLIP's own TRUE zero-shot binary decision (contrastive text prompts, no
     NIH labels ever fitted) -> what the encoder itself achieves under the same
     binary framing baby-Gemma is judged by. This is the fair encoder comparison;
     the earlier 0.81 probe was fitted per-finding on NIH labels and is an oracle.

    CUDA_VISIBLE_DEVICES=0 python nih_auc_analysis.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma.paths import ROOT as HERE

from babygemma.dataset import Ex, NanoDataset, build_tokenizer, collate
from babygemma.gemma_model import BabyGemmaVLM
from babygemma import nih as ND
from babygemma.vision import MODEL_ID

SEEDS = [0, 1, 2]
PER, DATA_SEED = 120, 0
OUT = os.path.join(HERE, "results_gemma", "grounding", "nih_auc.json")


def auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def oracle_acc(scores, labels):
    """Best achievable accuracy over all thresholds (per finding)."""
    order = np.argsort(scores)
    s, l = scores[order], labels[order]
    best = max((l == 1).mean(), (l == 0).mean())
    for i in range(len(s)):
        pred_hi = np.zeros(len(s)); pred_hi[i + 1:] = 1
        best = max(best, (pred_hi == l).mean())
    return float(best)


@torch.no_grad()
def margins(model, ds, use_ground):
    model.eval()
    out = []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {"ground": b["ground"].cuda()} if use_ground else {}
        logits, _ = model(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(), **kw)
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())   # yes - no
    return np.concatenate(out)


def medsiglip_zeroshot(records):
    """True zero-shot binary decision from the encoder: sim(pos prompt) - sim(neg prompt)."""
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0").eval()

    phrases = sorted({r["finding_phrase"] for r in records}) if "finding_phrase" in records[0] \
        else sorted({r["question"].replace("Is there ", "").rstrip("?") for r in records})
    texts, idx = [], {}
    for ph in phrases:
        idx[ph] = len(texts)
        texts += [f"a chest x-ray showing {ph}", f"a chest x-ray with no {ph}"]
    with torch.no_grad():
        ti = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        tf = model.get_text_features(**ti)
        tf = torch.nn.functional.normalize(tf.float(), dim=-1)

    paths = sorted({r["image_path"] for r in records})
    emb = {}
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            chunk = paths[i:i + 32]
            imgs = [Image.open(p).convert("RGB") for p in chunk]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
            f = torch.nn.functional.normalize(model.get_image_features(pixel_values=px).float(), dim=-1)
            for p, v in zip(chunk, f.cpu().numpy()):
                emb[p] = v
    tfn = tf.cpu().numpy()
    scores, labels, qs = [], [], []
    for r in records:
        ph = r["question"].replace("Is there ", "").rstrip("?")
        e = emb[r["image_path"]]
        scores.append(float(e @ tfn[idx[ph]] - e @ tfn[idx[ph] + 1]))
        labels.append(1 if r["answer"] == "yes" else 0)
        qs.append(r["question"])
    return np.array(scores), np.array(labels), np.array(qs)


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
    ds_b = NanoDataset(ex, tok, feats, pooled=None)
    ds_g = NanoDataset(ex, tok, feats, pooled=pc)
    print(f"[auc] NIH {len(records)} originals", flush=True)

    res = {"per_seed": [], "findings": sorted(set(q.tolist()))}
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
            s = margins(m, ds, ug)
            per_auc, per_orc, per_acc = {}, {}, {}
            for f in res["findings"]:
                mk = q == f
                per_auc[f] = auc(s[mk], y[mk])
                per_orc[f] = oracle_acc(s[mk], y[mk])
                per_acc[f] = float(((s[mk] > 0).astype(int) == y[mk]).mean())
            row[tag] = {
                "acc_argmax": float(((s > 0).astype(int) == y).mean()),
                "auc_mean": float(np.nanmean(list(per_auc.values()))),
                "oracle_acc_mean": float(np.mean(list(per_orc.values()))),
                "per_finding_auc": per_auc, "per_finding_acc": per_acc,
            }
            print(f"  seed {seed} {tag:9s} acc={row[tag]['acc_argmax']:.3f}  "
                  f"AUC={row[tag]['auc_mean']:.3f}  oracle_acc={row[tag]['oracle_acc_mean']:.3f}", flush=True)
        res["per_seed"].append(row)

    print("\n[auc] MedSigLIP TRUE zero-shot binary (no NIH labels fitted)...", flush=True)
    s, yy, qq = medsiglip_zeroshot(records)
    zs_auc = {f: auc(s[qq == f], yy[qq == f]) for f in res["findings"]}
    zs_acc = {f: float(((s[qq == f] > 0).astype(int) == yy[qq == f]).mean()) for f in res["findings"]}
    zs_orc = {f: oracle_acc(s[qq == f], yy[qq == f]) for f in res["findings"]}
    res["medsiglip_zeroshot"] = {"auc_mean": float(np.nanmean(list(zs_auc.values()))),
                                 "acc_mean": float(np.mean(list(zs_acc.values()))),
                                 "oracle_acc_mean": float(np.mean(list(zs_orc.values()))),
                                 "per_finding_auc": zs_auc, "per_finding_acc": zs_acc}
    z = res["medsiglip_zeroshot"]
    print(f"  MedSigLIP zero-shot: acc={z['acc_mean']:.3f}  AUC={z['auc_mean']:.3f}  "
          f"oracle_acc={z['oracle_acc_mean']:.3f}")

    print("\n=== SUMMARY (mean over seeds) ===")
    for tag in ("baseline", "grounded"):
        rows = [r[tag] for r in res["per_seed"] if tag in r]
        if rows:
            print(f"  baby-Gemma {tag:9s} acc={np.mean([r['acc_argmax'] for r in rows]):.3f}  "
                  f"AUC={np.mean([r['auc_mean'] for r in rows]):.3f}  "
                  f"oracle_acc={np.mean([r['oracle_acc_mean'] for r in rows]):.3f}")
    print(f"  MedSigLIP zero-shot  acc={z['acc_mean']:.3f}  AUC={z['auc_mean']:.3f}  "
          f"oracle_acc={z['oracle_acc_mean']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
