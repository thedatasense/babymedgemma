"""NIH ChestX-ray14 question and label builders, shared by the data-index scripts and
the NIH diagnostics. Kept free of training/analysis imports so it can be a leaf module."""

from __future__ import annotations

import csv
import glob
import os
import random

import torch

from babygemma.vision import MedSigLIP

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

