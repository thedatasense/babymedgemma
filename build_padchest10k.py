"""Generate ~10,000 binary presence questions from the PadChest label table
(already on disk), with operator-preserving register-tagged paraphrases, plus
the existing MIMIC presence items, and write a merged index for the nanoVLM.

Why operator-preserving only: the existing negation_pattern paraphrases
("Can you rule out X?") invert the answer, which conflates paraphrase
sensitivity with operator handling (Thrust-3 discussion). We use four
answer-preserving register phenomena so a flip is a genuine paraphrase
inconsistency.

    python build_padchest10k.py   # writes data/index_10k.json
"""

from __future__ import annotations

import ast
import collections
import csv
import glob
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
PADCHEST = "/home/bsada1/datasets/PadChest_GR"
MASTER = os.path.join(PADCHEST, "master_table.csv")
OUT = os.path.join(HERE, "data", "index_10k.json")
CAP_PER_FINDING = 400            # max yes (and equal no) questions per finding
MIN_FINDING_COUNT = 150          # top salient findings the frozen encoder can separate
SEED = 0

# non-specific / non-finding labels to exclude from the question vocabulary
BLOCKLIST = {"normal", "unchanged", "chronic changes", "chronic change",
             "exclude", "suboptimal study", "non visualized"}

# operator-preserving register templates; {f} = finding phrase
TEMPLATES = {
    "lexical_substitution": ["Is there evidence of {f}?", "Is there radiographic evidence of {f}?"],
    "syntactic_restructuring": ["Can {f} be identified?", "Does the image show {f}?"],
    "scope_quantification": ["Is there any {f}?", "Is there any sign of {f}?"],
    "specificity_modulation": ["Is {f} visible on this chest radiograph?", "Is {f} present on this study?"],
}


def _parse_labels(x):
    x = (x or "").strip()
    if x.startswith("["):
        try:
            return [str(t).strip().lower() for t in ast.literal_eval(x)]
        except Exception:
            pass
    return [t.strip().lower() for t in x.replace(";", ",").split(",") if t.strip()]


def _split_for(pid, frac=(0.7, 0.15)):
    h = int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 1000 / 1000.0
    return "train" if h < frac[0] else ("val" if h < frac[0] + frac[1] else "test")


def paraphrases_for(finding):
    out = []
    for phen, tmpls in TEMPLATES.items():
        for t in tmpls:
            out.append({"text": t.format(f=finding), "phenomenon": phen})
    return out


def build_padchest():
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(MASTER)))
    # image -> {findings, patient}
    img_findings = collections.defaultdict(set)
    img_patient = {}
    counts = collections.Counter()
    for r in rows:
        iid = r["ImageID"]
        img_patient[iid] = r.get("PatientID", iid)
        for f in _parse_labels(r["label"]):
            if f in BLOCKLIST:
                continue
            img_findings[iid].add(f)
            counts[f] += 1
    vocab = [f for f, c in counts.items() if c >= MIN_FINDING_COUNT]
    # resolve image paths
    path_of = {os.path.basename(p): p for p in glob.glob(PADCHEST + "/**/*.png", recursive=True)}
    imgs = [i for i in img_findings if i in path_of]

    # Balance yes/no PER FINDING so the finding word alone cannot predict the
    # answer; this forces the model to read the image (no text shortcut).
    items = []
    for f in vocab:
        present = [i for i in imgs if f in img_findings[i]]
        absent = [i for i in imgs if f not in img_findings[i]]
        n = min(len(present), len(absent), CAP_PER_FINDING)
        for i in rng.sample(present, n):
            items.append((i, f, "yes"))
        for i in rng.sample(absent, n):
            items.append((i, f, "no"))
    rng.shuffle(items)

    records = []
    for k, (iid, f, ans) in enumerate(items):
        records.append({
            "uid": f"pc10k:{k}",
            "image_path": path_of[iid],
            "question": f"Is there {f}?",
            "answer": ans,
            "source": "padchest",
            "paraphrases": paraphrases_for(f),
            "split": _split_for(img_patient[iid]),
        })
    return records, vocab


def load_mimic():
    from data_index import load_mimic as _lm
    return _lm()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pc, vocab = build_padchest()
    mimic = load_mimic()
    index = pc + mimic
    json.dump(index, open(OUT, "w"))
    from collections import Counter
    print(f"[build_10k] vocab findings: {len(vocab)}")
    print(f"[build_10k] padchest questions: {len(pc)}  mimic: {len(mimic)}  total: {len(index)}")
    print(f"[build_10k] by answer: {dict(Counter(r['answer'] for r in index))}")
    print(f"[build_10k] by split: {dict(Counter(r['split'] for r in index))}")
    print(f"[build_10k] unique images: {len({r['image_path'] for r in index})}")
    print(f"[build_10k] avg paraphrases: {sum(len(r['paraphrases']) for r in index)/len(index):.1f}")
    print(f"[build_10k] wrote {OUT}")


if __name__ == "__main__":
    main()
