"""Build a scaled training index: NIH (balanced per finding) + MIMIC/PadChest.

Two problems with the 1,841-question index this replaces:
  - too small for a decoder to learn a visual readout, and
  - answers are predictable from the finding name alone (text-only accuracy 68.8%),
    so there is little gradient pressure to use the image at all.

This fixes both. Images are assigned to splits FIRST (an image generates questions
for several findings, so splitting by question would leak), then within each split
every finding gets an equal number of yes and no cases. A text-only model therefore
cannot beat 50% on the NIH portion, which forces the decoder onto the image.

    python build_scaled_index.py
"""
from __future__ import annotations

import csv, glob, hashlib, json, os, random, sys
from collections import Counter, defaultdict

from babygemma.paths import ROOT as HERE

from babygemma.data_index import build_index
from babygemma.nih import NIH_CSV, NIH_DIR, NIH_PHRASE, TEMPLATES

OUT = os.environ.get("NANO_INDEX_OUT", os.path.join(HERE, "data", "index_scaled.json"))
SEED = 0
# cap positives per finding so one common label (Infiltration has 19,894 positives at
# full scale, and is the hardest: held-out AUC 0.593) cannot dominate the objective
CAP = int(os.environ.get("NANO_CAP", "0")) or None


def split_for(key, frac=(0.7, 0.15)):
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < frac[0] else ("val" if h < frac[0] + frac[1] else "test")


def build_nih_scaled(rng):
    rows = list(csv.DictReader(open(NIH_CSV)))
    paths = {os.path.basename(p): p for p in glob.glob(NIH_DIR + "/**/*.png", recursive=True)}
    img_find = {r["Image Index"]: set(r["Finding Labels"].split("|"))
                for r in rows if r["Image Index"] in paths}
    # 1. assign every IMAGE to a split first -> no cross-split image leakage
    img_split = {i: split_for("nih:" + i) for i in img_find}
    by_split = defaultdict(list)
    for i, s in img_split.items():
        by_split[s].append(i)

    records, cid = [], 0
    for split, imgs in by_split.items():
        for f, ph in NIH_PHRASE.items():
            present = [i for i in imgs if f in img_find[i]]
            absent = [i for i in imgs if f not in img_find[i]]
            n = min(len(present), len(absent))
            if CAP:
                # cap is a per-split share of the global per-finding cap
                n = min(n, max(4, int(CAP * len(imgs) / max(1, len(img_find)))))
            if n < 4:
                continue
            # 2. balance yes/no WITHIN this finding -> finding name predicts nothing
            chosen = [(i, "yes") for i in rng.sample(present, n)] + \
                     [(i, "no") for i in rng.sample(absent, n)]
            for iid, ans in chosen:
                records.append({
                    "uid": f"nih:{cid}", "image_path": paths[iid],
                    "question": f"is there {ph}?", "answer": ans,
                    "paraphrases": [{"text": t.format(f=ph).lower(), "phenomenon": phen}
                                    for phen, ts in TEMPLATES.items() for t in ts],
                    "split": split, "source": "nih",
                })
                cid += 1
    return records


def main():
    rng = random.Random(SEED)
    base = build_index()
    for r in base:
        r.setdefault("source", "mimic_padchest")
    nih = build_nih_scaled(rng)
    idx = base + nih

    # The base index splits by question id, so a handful of images with several
    # questions straddle splits. Pin every image to one split (lowest-priority
    # split wins deterministically: test > val > train, so leakage resolves
    # toward evaluation rather than toward training).
    rank = {"train": 0, "val": 1, "test": 2}
    pinned = {}
    for r in idx:
        p = r["image_path"]
        if p not in pinned or rank[r["split"]] > rank[pinned[p]]:
            pinned[p] = r["split"]
    moved = sum(1 for r in idx if r["split"] != pinned[r["image_path"]])
    for r in idx:
        r["split"] = pinned[r["image_path"]]
    print(f"[scaled] pinned images to one split each ({moved} records moved)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(idx, open(OUT, "w"))

    imgs = {r["image_path"] for r in idx}
    print(f"[scaled] {len(idx)} questions ({len(base)} mimic/padchest + {len(nih)} nih)")
    print(f"[scaled] {len(imgs)} unique images")
    print(f"[scaled] splits: {dict(Counter(r['split'] for r in idx))}")
    print(f"[scaled] answers: {dict(Counter(r['answer'] for r in idx))}")
    ex = sum(1 + len(r["paraphrases"]) for r in idx)
    print(f"[scaled] ~{ex} examples with paraphrases")

    # verify the text-only shortcut is dead on the NIH portion
    print("\n[check] answer balance per finding (NIH portion, train split):")
    per = defaultdict(Counter)
    for r in nih:
        if r["split"] == "train":
            per[r["question"]][r["answer"]] += 1
    for q, c in sorted(per.items())[:6]:
        tot = c["yes"] + c["no"]
        print(f"    {q:32s} yes={c['yes']:5d} no={c['no']:5d}  P(yes)={c['yes']/tot:.3f}")
    # image leakage check
    per_img_split = defaultdict(set)
    for r in idx:
        per_img_split[r["image_path"]].add(r["split"])
    leaked = [p for p, s in per_img_split.items() if len(s) > 1]
    print(f"\n[check] images appearing in >1 split: {len(leaked)} (must be 0 for nih)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
