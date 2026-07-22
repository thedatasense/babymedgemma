"""Stage 0 falsification (corrected): do grounding-route flips exist on UNSEEN wording?

A grounding-route flip is a stable-answer case whose finding-selective visual reliance V
is clearly grounded under one phrasing and near zero (image-unreliant) under another,
reproducibly across independently drawn matched controls.

Corrections over the first version, each of which invalidated the earlier headline:
  1. Checkpoints trained with held-out phrasings (results_transfer_heldout/B/augmented_s*,
     trained on 24 of the 48 templates), scored on the 24 GENUINELY UNSEEN templates.
  2. A two-way-centered cross-match variance decomposition, so the numerator and the
     noise floor are centered the same way and constant match-set offsets do not inflate
     the denominator:
        C(X)_{ip} = X_{ip} - mean_p X_{i.} - mean_i X_{.p} + mean X
        sigma2_route        = mean[ C(V_A) * C(V_B) ]      (reproducible interaction)
        sigma2_match_prompt = 0.5 * mean[ (C(V_A) - C(V_B))^2 ]
        rho                 = corr( C(V_A), C(V_B) )
  3. A route-drop event defined as grounded-to-near-zero, not a sign reversal: one prompt
     with C(V) above a grounded threshold (both match sets agreeing, sign excluding zero)
     and one prompt with |C(V)| inside an equivalence band, calibrated on the match-noise
     null. Reported with a case-bootstrap 95% interval, finding-stratified.
Per-case records (margins, match identities, G/N/V) are saved for independent audit.

    python scripts/analysis/route_flip_pilot.py
"""
from __future__ import annotations

import csv, json, os, random
import numpy as np
import torch

from babygemma.paths import ROOT
from babygemma.data_index import build_index
from babygemma.dataset import build_tokenizer, load_feature_cache, load_pooled_cache
from babygemma.gemma_model import BabyGemmaVLM
from babygemma.nih import NIH_CSV, NIH_PHRASE
from babygemma import templates as T

SEEDS = [0, 1, 2]
CKPT = os.path.join(ROOT, "results_transfer_heldout", "B", "augmented_s{seed}", "model.pt")
OUT = os.path.join(ROOT, "results_transfer", "route_flip_pilot.json")
RECS = os.path.join(ROOT, "results_transfer", "route_flip_records.json")
FINDINGS = ["Effusion", "Atelectasis", "Cardiomegaly", "Pneumothorax", "Consolidation",
            "Infiltration", "Nodule", "Mass"]
CASES_PER_FINDING = 30
K = 3
AGE_BAND = 12


def unseen_prompts(phrase):
    """The 24 held-out templates (indices HELDOUT_FROM..end of each family), never trained
    on by the held-out checkpoints; canonical is excluded because it was trained on."""
    out = []
    for fam, tmpls in T.PARAPHRASES.items():
        for t in tmpls[T.HELDOUT_FROM:]:
            out.append(t.format(f=phrase))
    return out


def two_way_center(X):
    return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def load_meta(have):
    idx = build_index()
    val = {os.path.basename(r["image_path"]) for r in idx
           if r["split"] == "val" and "/NIH/" in r["image_path"]}
    meta = {}
    for row in csv.DictReader(open(NIH_CSV)):
        n = row["Image Index"]
        if n not in val or n not in have:
            continue
        try:
            age = int(row["Patient Age"])
        except ValueError:
            continue
        meta[n] = {"findings": {k for k in NIH_PHRASE if k in row["Finding Labels"].split("|")},
                   "age": age, "sex": row["Patient Gender"], "view": row["View Position"]}
    return meta


def build_cases(meta, rng):
    def ok(a, b):
        return a["sex"] == b["sex"] and a["view"] == b["view"] and abs(a["age"] - b["age"]) <= AGE_BAND

    def dist(a, b, tgt):
        return len((a["findings"] ^ b["findings"]) - {tgt})

    names = list(meta)
    cases = []
    for f in FINDINGS:
        pos = [n for n in names if f in meta[n]["findings"]]
        neg = [n for n in names if f not in meta[n]["findings"]]
        picks = rng.sample(pos, min(CASES_PER_FINDING // 2, len(pos))) + \
                rng.sample(neg, min(CASES_PER_FINDING // 2, len(neg)))
        for i in picks:
            y = 1 if f in meta[i]["findings"] else -1
            same = [n for n in (pos if y == 1 else neg) if n != i and ok(meta[i], meta[n])]
            opp = [n for n in (neg if y == 1 else pos) if ok(meta[i], meta[n])]
            if len(same) < 2 * K or len(opp) < 2 * K:
                continue
            same = sorted(same, key=lambda n: dist(meta[i], meta[n], f))[:2 * K]
            opp = sorted(opp, key=lambda n: dist(meta[i], meta[n], f))[:2 * K]
            rng.shuffle(same); rng.shuffle(opp)
            cases.append({"i": i, "y": y, "finding": f, "phrase": NIH_PHRASE[f],
                          "same": same, "opp": opp})
    return cases


def main():
    tok = build_tokenizer()
    feats, pc = load_feature_cache(), load_pooled_cache()
    dev = "cuda"
    by_name = {os.path.basename(p): p for p in feats["index"]}
    pooled_names = {os.path.basename(x) for x in pc["index"]}
    have = {n: p for n, p in by_name.items() if n in pooled_names}
    meta = load_meta(have)
    print(f"[pilot] NIH val images with metadata+features: {len(meta)}", flush=True)
    cases = build_cases(meta, random.Random(0))
    print(f"[pilot] {len(cases)} matchable cases; 24 unseen prompts; seeds {SEEDS}", flush=True)

    def vec(n):
        p = by_name[n]
        return feats["feats"][feats["index"][p]].float(), pc["pooled"][pc["index"][p]].float()

    per_seed, all_records = [], []
    for seed in SEEDS:
        m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, max_len=tok.max_len,
                         yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                         use_ground=True, ground_dim=pc["pooled"].shape[1]).to(dev)
        m.load_state_dict(torch.load(CKPT.format(seed=seed), map_location=dev), strict=False)
        m.eval()

        @torch.no_grad()
        def margins(names, prompt):
            ids, ap = tok.encode(prompt)
            vs = torch.stack([vec(n)[0] for n in names]).to(dev)
            gs = torch.stack([vec(n)[1] for n in names]).to(dev)
            idb = torch.tensor([ids]).repeat(len(names), 1).to(dev)
            apb = torch.tensor([ap]).repeat(len(names)).to(dev)
            lg, _ = m(vs, idb, apb, ground=gs)
            return (lg[:, 1] - lg[:, 0]).float().cpu().numpy()

        VA, VB, stable_mask, findings, recs = [], [], [], [], []
        for c in cases:
            prompts = unseen_prompts(c["phrase"])
            mi = np.array([margins([c["i"]], q)[0] for q in prompts])          # [P]
            mo = np.array([margins(c["opp"], q) for q in prompts])             # [P,6]
            ms = np.array([margins(c["same"], q) for q in prompts])            # [P,6]
            y = c["y"]
            va, vb = {}, {}
            for tag, sl in (("A", slice(0, K)), ("B", slice(K, 2 * K))):
                G = y * (mi - np.median(mo[:, sl], axis=1))
                N = np.median(np.abs(mi[:, None] - ms[:, sl]), axis=1)
                (va if tag == "A" else vb)["V"] = (G - N)
            VA.append(va["V"]); VB.append(vb["V"])
            stable_mask.append(len(set((mi > 0).tolist())) == 1)
            findings.append(c["finding"])
            recs.append({"seed": seed, "finding": c["finding"], "y": y, "image": c["i"],
                         "same": c["same"], "opp": c["opp"], "stable": bool(stable_mask[-1]),
                         "m_i": mi.tolist(), "VA": va["V"].tolist(), "VB": vb["V"].tolist()})
        all_records += recs

        VA = np.array(VA); VB = np.array(VB); stable_mask = np.array(stable_mask)
        fnd = np.array(findings)
        sA, sB, sf = VA[stable_mask], VB[stable_mask], fnd[stable_mask]
        CA, CB = two_way_center(sA), two_way_center(sB)
        sig_route = float(np.mean(CA * CB))
        sig_match = float(0.5 * np.mean((CA - CB) ** 2))
        rho = float(np.corrcoef(CA.ravel(), CB.ravel())[0, 1])
        rms_within = float(np.sqrt(np.mean(np.var((sA + sB) / 2, axis=1))))

        # event: grounded (both sets agree, sign excludes zero, above delta_g) to
        # near-zero (|C(Vbar)| < delta_0). delta_0 from the match-noise null.
        Cbar = (CA + CB) / 2
        delta_0 = float(np.sqrt(sig_match))
        delta_g = 2 * delta_0
        grounded = (Cbar > delta_g) & (np.minimum(CA, CB) > 0)
        nearzero = np.abs(Cbar) < delta_0
        route = (grounded.any(1) & nearzero.any(1))

        # case-bootstrap, finding-stratified, on the stable subset
        rb = np.random.default_rng(seed)
        boots_route, boots_sig = [], []
        idxf = {f: np.where(sf == f)[0] for f in set(sf.tolist())}
        for _ in range(1000):
            pick = np.concatenate([rb.choice(v, len(v), replace=True) for v in idxf.values()])
            boots_route.append(route[pick].mean())
            ca, cb = two_way_center(sA[pick]), two_way_center(sB[pick])
            boots_sig.append(float(np.mean(ca * cb)))
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        r = {"seed": seed, "n_cases": int(len(cases)), "n_stable": int(stable_mask.sum()),
             "n_prompts": int(sA.shape[1]),
             "sigma2_route": sig_route, "sigma2_route_ci": ci(boots_sig),
             "sigma2_match_prompt": sig_match, "corr_CA_CB": rho,
             "rms_within_case_std": rms_within,
             "route_drop_rate": float(route.mean()), "route_drop_ci": ci(boots_route),
             "delta_0": delta_0, "delta_g": delta_g}
        per_seed.append(r)
        print(f"\n[seed {seed}] stable {r['n_stable']}/{r['n_cases']}  prompts {r['n_prompts']}")
        print(f"  sigma2_route = {sig_route:+.4f}  95% CI {r['sigma2_route_ci']}   (>0 and CI excluding 0 = real interaction)")
        print(f"  sigma2_match_prompt = {sig_match:.4f}   corr(C_A,C_B) = {rho:+.3f}")
        print(f"  rms within-case std of V = {rms_within:.4f}")
        print(f"  route-drop rate = {r['route_drop_rate']:.3f}  95% CI {r['route_drop_ci']}")

    agg = {k: float(np.mean([s[k] for s in per_seed]))
           for k in ["sigma2_route", "sigma2_match_prompt", "corr_CA_CB",
                     "rms_within_case_std", "route_drop_rate"]}
    print(f"\n=== pooled over {len(SEEDS)} held-out seeds ===")
    print(f"  sigma2_route {agg['sigma2_route']:+.4f}  sigma2_match {agg['sigma2_match_prompt']:.4f}  "
          f"corr {agg['corr_CA_CB']:+.3f}  route-drop {agg['route_drop_rate']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"per_seed": per_seed, "pooled": agg}, open(OUT, "w"), indent=2)
    json.dump(all_records, open(RECS, "w"))
    print(f"wrote {OUT} and per-case records to {RECS}")


if __name__ == "__main__":
    main()
