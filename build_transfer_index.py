"""Transfer design: train on NIH + PadChest, hold out MIMIC + VinDr entirely as OOD.

Every dataset is rendered into the SAME canonical question form ("is there {finding}?"
plus the same paraphrase templates), so at test time the only thing that shifts is the
image distribution -- not the phrasing. Reusing MIMIC's native wording would confound
image shift with question shift.

Answers are balanced per finding within every split, so a text-only model scores 0.500
everywhere and any accuracy above that is earned from the radiograph.

Splits: train / val   (NIH + PadChest, image-disjoint)
        ood_mimic     (held out entirely)
        ood_vindr     (held out entirely)

    NANO_CAP=5000 python build_transfer_index.py
"""
from __future__ import annotations

import csv, glob, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nih_demo import NIH_CSV, NIH_DIR, NIH_PHRASE
from templates import paraphrases_for, operator_variants_for, n_paraphrases

PROJECT = "/home/bsada1/medical-vlm-robustness"
OUT = os.environ.get("NANO_INDEX_OUT", os.path.join(HERE, "data", "index_transfer.json"))
CAP = int(os.environ.get("NANO_CAP", "5000"))
SEED = 0

# canonical finding phrases (the shared vocabulary across all four datasets)
PHRASES = sorted(set(NIH_PHRASE.values()))

VINDR_MAP = {
    "Cardiomegaly": "cardiomegaly", "Pleural effusion": "pleural effusion",
    "Pneumothorax": "pneumothorax", "Atelectasis": "atelectasis",
    "Consolidation": "consolidation", "Infiltration": "infiltrates",
    "Nodule/Mass": "nodule", "Pleural thickening": "pleural thickening",
    "Pulmonary fibrosis": "fibrosis",
}


def split_for(key, frac=0.85):
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < frac else "val"


def make(image_path, phrase, ans, split, source, cid):
    return {"uid": f"{source}:{cid}", "image_path": image_path,
            "question": f"is there {phrase}?", "answer": ans,
            "paraphrases": paraphrases_for(phrase),
            "operator_variants": operator_variants_for(phrase),
            "split": split, "source": source}


def balanced(pos, neg, rng, cap):
    n = min(len(pos), len(neg), cap) if cap else min(len(pos), len(neg))
    if n < 4:
        return []
    return [(i, "yes") for i in rng.sample(pos, n)] + [(i, "no") for i in rng.sample(neg, n)]


def from_labels(img_labels, paths, source, rng, splitter, cap):
    """img_labels: {image_key: set(canonical phrases)} -> balanced records."""
    out, cid = [], 0
    by_split = defaultdict(list)
    for k in img_labels:
        by_split[splitter(k)].append(k)
    for split, imgs in by_split.items():
        for ph in PHRASES:
            pos = [i for i in imgs if ph in img_labels[i]]
            neg = [i for i in imgs if ph not in img_labels[i]]
            share = cap if cap and len(by_split) == 1 else (
                int(cap * len(imgs) / max(1, len(img_labels))) if cap else 0)
            for i, a in balanced(pos, neg, rng, share or None):
                out.append(make(paths[i], ph, a, split, source, cid))
                cid += 1
    return out


def nih(rng):
    rows = list(csv.DictReader(open(NIH_CSV)))
    paths = {os.path.basename(p): p for p in glob.glob(NIH_DIR + "/**/*.png", recursive=True)}
    lab = {}
    for r in rows:
        n = r["Image Index"]
        if n in paths:
            lab[n] = {NIH_PHRASE[f] for f in r["Finding Labels"].split("|") if f in NIH_PHRASE}
    return from_labels(lab, paths, "nih", rng, lambda k: split_for("nih:" + k), CAP)


def padchest(rng):
    rows = list(csv.DictReader(open("/home/bsada1/datasets/PadChest_GR/master_table.csv")))
    paths = {os.path.basename(p)[:-4]: p
             for p in glob.glob("/home/bsada1/datasets/PadChest_GR/**/*.png", recursive=True)}
    lab = defaultdict(set)
    for r in rows:
        iid = r["ImageID"].replace(".png", "")
        if iid not in paths:
            continue
        raw = r["label"].strip().lower()
        for ph in PHRASES:
            if ph in raw:
                lab[iid].add(ph)
        lab.setdefault(iid, set())
    return from_labels(dict(lab), paths, "padchest", rng,
                       lambda k: split_for("pc:" + k), CAP)


def vindr(rng):
    rows = list(csv.DictReader(open(f"{PROJECT}/dataset/vindr/vindr_questions_full.csv")))
    paths = {os.path.basename(p)[:-4]: p
             for p in glob.glob(f"{PROJECT}/dataset/vindr/png_cache/*.png")}
    lab = defaultdict(set)
    for r in rows:
        iid = r["image_id"]
        if iid not in paths:
            continue
        ph = VINDR_MAP.get(r["gt_finding"].strip())
        if ph:
            lab[iid].add(ph)
        lab.setdefault(iid, set())
    recs = from_labels(dict(lab), paths, "vindr", rng, lambda k: "ood_vindr", CAP)
    for r in recs:
        r["split"] = "ood_vindr"
    return recs


def mimic(rng):
    """Recover per-image labels from the curated MIMIC presence questions."""
    from data_index import MIMIC_IMG
    lab, paths = defaultdict(set), {}
    for sp in ["train", "val", "test"]:
        p = f"{PROJECT}/dataset/mimic/splits/{sp}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["results"] if isinstance(d, dict) and "results" in d else d
        for r in rows:
            if r.get("question_type") != "presence":
                continue
            fn = r.get("filename")
            ip = os.path.join(MIMIC_IMG, fn) if fn else None
            if not ip or not os.path.exists(ip):
                continue
            key = os.path.basename(ip)
            paths[key] = ip
            q = str(r.get("original_question") or "").lower()
            a = str(r.get("answer") or "").strip().lower()
            for ph in PHRASES:
                if ph in q and a in ("yes", "no"):
                    if a == "yes":
                        lab[key].add(ph)
                    lab.setdefault(key, set())
    recs = from_labels(dict(lab), paths, "mimic", rng, lambda k: "ood_mimic", CAP)
    for r in recs:
        r["split"] = "ood_mimic"
    return recs


def main():
    rng = random.Random(SEED)
    idx = nih(rng) + padchest(rng) + vindr(rng) + mimic(rng)

    # pin each image to one split (test/ood wins) so nothing leaks into training
    rank = {"train": 0, "val": 1, "ood_mimic": 2, "ood_vindr": 2}
    pin = {}
    for r in idx:
        p = r["image_path"]
        if p not in pin or rank[r["split"]] > rank[pin[p]]:
            pin[p] = r["split"]
    for r in idx:
        r["split"] = pin[r["image_path"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(idx, open(OUT, "w"))

    print(f"[transfer] {len(idx)} questions over {len({r['image_path'] for r in idx})} images")
    print(f"[transfer] by split : {dict(Counter(r['split'] for r in idx))}")
    print(f"[transfer] by source: {dict(Counter(r['source'] for r in idx))}")
    print(f"[transfer] answers  : {dict(Counter(r['answer'] for r in idx))}")
    print(f"[transfer] ~{sum(1+len(r['paraphrases']) for r in idx)} examples with paraphrases")
    for s in ("train", "ood_mimic", "ood_vindr"):
        sub = [r for r in idx if r["split"] == s]
        if not sub:
            continue
        c = Counter(r["answer"] for r in sub)
        print(f"  {s:10s} n={len(sub):6d}  P(yes)={c['yes']/max(1,len(sub)):.3f}  "
              f"findings={len({r['question'] for r in sub})}")
    imgs = defaultdict(set)
    for r in idx:
        imgs[r["image_path"]].add(r["split"])
    print(f"[check] images in >1 split: {sum(1 for v in imgs.values() if len(v)>1)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
