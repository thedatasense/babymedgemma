"""Zero-shot external-validity demo on NIH ChestX-ray14.

baby-Gemma was trained only on MIMIC-CXR and PadChest. Here we take NIH chest
radiographs it has never seen, build "Is there {finding}?" clusters for the NIH
findings whose words are in the model's vocabulary, run the frozen MedSigLIP
encoder on the NIH images, and measure whether the same paraphrase-sensitivity
mechanism appears: the flip rate, and the Jacobian-lens divergence between
flipping and non-flipping clusters.

    python nih_demo.py --arch gemma --per-finding 120 --out results/nih_demo
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import jlens as JL
import metrics as Mx
from data_index import build_index
from dataset import Ex, NanoDataset, build_tokenizer
from train import train_model
from vision import MedSigLIP

NIH_DIR = "/home/bsada1/datasets/NIH"
NIH_CSV = os.path.join(NIH_DIR, "data", "Data_Entry_2017_v2020.csv")

# NIH finding -> the question phrase we would ask
NIH_PHRASE = {
    "Cardiomegaly": "cardiomegaly", "Pneumothorax": "pneumothorax",
    "Effusion": "pleural effusion", "Atelectasis": "atelectasis", "Nodule": "nodule",
    "Mass": "mass", "Consolidation": "consolidation", "Edema": "edema",
    "Pneumonia": "pneumonia", "Infiltration": "infiltrates", "Emphysema": "emphysema",
    "Fibrosis": "fibrosis", "Pleural_Thickening": "pleural thickening", "Hernia": "hernia",
}
TEMPLATES = {
    "lexical_substitution": ["Is there evidence of {f}?", "Is there radiographic evidence of {f}?"],
    "syntactic_restructuring": ["Can {f} be identified?", "Does the image show {f}?"],
    "scope_quantification": ["Is there any {f}?", "Is there any sign of {f}?"],
    "specificity_modulation": ["Is {f} visible on this chest radiograph?", "Is {f} present on this study?"],
}


def in_vocab(phrase, tok):
    import re
    return all(w in tok.stoi for w in re.findall(r"[a-z0-9]+", phrase.lower()))


def build_nih_records(tok, per_finding, seed):
    rng = random.Random(seed)
    rows = list(csv.DictReader(open(NIH_CSV)))
    paths = {os.path.basename(p): p for p in glob.glob(NIH_DIR + "/**/*.png", recursive=True)}
    img_find = {}
    for r in rows:
        name = r.get("Image Index")
        if name in paths:
            img_find[name] = set(r.get("Finding Labels", "").split("|"))
    findings = [f for f, ph in NIH_PHRASE.items() if in_vocab(ph, tok)]
    imgs = list(img_find)
    records = []
    cid = 0
    kept = []
    for f in findings:
        ph = NIH_PHRASE[f]
        present = [i for i in imgs if f in img_find[i]]
        absent = [i for i in imgs if f not in img_find[i] and "No Finding" not in img_find[i] or
                  (f not in img_find[i] and "No Finding" in img_find[i])]
        n = min(len(present), len(absent), per_finding)
        if n < 8:
            continue
        kept.append((f, n))
        for iid, ans in [(i, "yes") for i in rng.sample(present, n)] + \
                        [(i, "no") for i in rng.sample(absent, n)]:
            paras = [{"text": t.format(f=ph), "phenomenon": phen}
                     for phen, ts in TEMPLATES.items() for t in ts]
            records.append({"uid": f"nih:{cid}", "image_path": paths[iid],
                            "question": f"Is there {ph}?", "answer": ans,
                            "paraphrases": paras, "split": "test"})
            cid += 1
    return records, kept


def to_clusters(records):
    ex = []
    for c, r in enumerate(records):
        ex.append(Ex(r["image_path"], r["question"], r["answer"], "original", c))
        for p in r["paraphrases"]:
            ex.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), c))
    return ex


def precompute_nih(paths, device="cuda:0"):
    enc = MedSigLIP(device=device)
    idx, feats = {}, []
    batch, bpaths = [], []

    def flush():
        if batch:
            f = enc.encode(batch)
            for k, p in enumerate(bpaths):
                idx[p] = len(feats) + k
            feats.append(f)
            batch.clear(); bpaths.clear()
    for i, p in enumerate(paths):
        try:
            batch.append(Image.open(p).convert("RGB").resize((448, 448))); bpaths.append(p)
        except Exception:
            continue
        if len(batch) >= 48:
            flush()
    flush()
    return {"index": idx, "feats": torch.cat(feats)}


def run(arch, per_finding, seed, out):
    art = train_model(regime="augmented", seed=seed, arch=arch)
    model, native_ds, device, tok = art["model"], art["eval_ds"], art["device"], art["tok"]
    print(f"[nih] baby-Gemma trained on MIMIC+PadChest: acc={art['result']['accuracy']:.3f} "
          f"native flip={art['result']['flip_rate']:.3f}")

    records, kept = build_nih_records(tok, per_finding, seed)
    print(f"[nih] in-vocab findings used: {[f for f,_ in kept]}")
    print(f"[nih] NIH clusters: {len(records)} over {len({r['image_path'] for r in records})} images")
    ex = to_clusters(records)
    nih_paths = sorted({e.image_path for e in ex})
    print(f"[nih] encoding {len(nih_paths)} NIH images with frozen MedSigLIP...", flush=True)
    nih_feats = precompute_nih(nih_paths, device=device)

    nih_ds = NanoDataset(ex, tok, nih_feats)
    preds, answers, clusters, _ = Mx.predict(model, nih_ds, device)
    acc = Mx.accuracy(preds, answers)
    flip = Mx.flip_rate(preds, clusters)
    flips = Mx.flip_labels(preds, clusters)

    # Jacobian lens transferred to NIH: fit on native, read on NIH clusters
    J, Hbar, mbar, depth = JL.fit_jacobian(model, native_ds, device, n_fit=256)
    rows, _ = JL.read_lens(model, nih_ds, device, J, Hbar, mbar)
    div, div_flip, div_noflip, pb = JL.analyze(rows, depth, flips)
    ratio = [round(div_flip[L] / (div_noflip[L] + 1e-9), 2) for L in range(depth)]

    result = {
        "arch": arch, "findings": [f for f, _ in kept], "n_clusters": len(records),
        "nih_accuracy_zeroshot": acc, "nih_flip_rate": flip,
        "nih_lens_divergence_flip": [round(x, 3) for x in div_flip],
        "nih_lens_divergence_nonflip": [round(x, 3) for x in div_noflip],
        "nih_flip_vs_nonflip_ratio": ratio,
        "nih_divergence_flip_pointbiserial": [round(x, 2) for x in pb],
    }
    print(f"[nih] ZERO-SHOT on NIH: accuracy {acc:.3f}, flip rate {flip:.3f}")
    print(f"[nih] Jacobian-lens divergence flip/non-flip ratio by layer: {ratio}")
    print(f"[nih] lens divergence-vs-flip point-biserial: {[round(x,2) for x in pb]}")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(result, open(os.path.join(out, "nih_demo.json"), "w"), indent=2)
        print(f"[nih] saved {out}/nih_demo.json")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gemma", choices=["nano", "gemma"])
    ap.add_argument("--per-finding", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/nih_demo")
    a = ap.parse_args()
    run(a.arch, a.per_finding, a.seed, a.out)


if __name__ == "__main__":
    main()
