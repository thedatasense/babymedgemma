"""Stage 0 (corrected event label): does a grounded-to-unreliant route drop occur?

Two separate questions, two separate statistics:

  Population interaction (does reliance depend on wording at all): computed on TWO-WAY
  CENTERED V, via the cross-draw covariance and its finding-vs-case split. Centering is
  correct here because we are asking about variation, not level.

  Route-drop event (a grounded prompt and an image-unreliant prompt in the same
  stable-answer case): computed on RAW V, because "unreliant" means the ABSOLUTE reliance
  is near zero, which centering destroys. A prompt is grounded if every match draw gives
  V > delta_g; unreliant if every draw gives |V| < delta_0. delta_0 and delta_g are locked
  on a development split against a same-versus-same null (V with no true finding change),
  then applied to a disjoint test split.

Corrections over the previous version:
  - event on raw V, not centered V (the centered "near zero" was trivially always true);
  - delta_0 / delta_g locked on a development null anchor, not on the match-noise;
  - several independent match draws from the ten nearest eligible controls (not a
    fixed split of the six nearest), giving per-case intervals and honest reproducibility;
  - every matched-image margin is saved, so G, N, V, and rematching are auditable offline;
  - finding-vs-case variance split in the script; dev/test split by patient; zero-event
    rates reported as Clopper-Pearson patient-level upper bounds, not a degenerate [0,0].

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
CASES_PER_FINDING = 40
POOL, K, M_DRAWS, AGE_BAND = 10, 3, 6, 12


def unseen_prompts(phrase):
    return [t.format(f=phrase) for tmpls in T.PARAPHRASES.values() for t in tmpls[T.HELDOUT_FROM:]]


def twc(X):
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
                   "age": age, "sex": row["Patient Gender"], "view": row["View Position"],
                   "patient": row["Patient ID"]}
    return meta


def build_cases(meta, rng):
    def ok(a, b):
        return a["sex"] == b["sex"] and a["view"] == b["view"] and abs(a["age"] - b["age"]) <= AGE_BAND
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
            if len(same) < POOL + K or len(opp) < POOL:      # need same for N + null second set
                continue
            d = lambda n: len((meta[i]["findings"] ^ meta[n]["findings"]) - {f})
            same = sorted(same, key=d)[:POOL + K]
            opp = sorted(opp, key=d)[:POOL]
            cases.append({"i": i, "y": y, "finding": f, "phrase": NIH_PHRASE[f],
                          "patient": meta[i]["patient"], "same": same, "opp": opp})
    return cases


def draws_V(mi, m_opp, m_same, y, rng):
    """M draws of V = G - N from the ten nearest matched controls. y is the label
    direction, so G measures finding-selective evidence for the case's true status."""
    P = mi.shape[0]
    out = np.zeros((M_DRAWS, P))
    for d in range(M_DRAWS):
        io = rng.sample(range(m_opp.shape[1]), K)
        is_ = rng.sample(range(m_same.shape[1]), K)
        G = y * (mi - np.median(m_opp[:, io], axis=1))
        N = np.median(np.abs(mi[:, None] - m_same[:, is_]), axis=1)
        out[d] = G - N
    return out


def null_V(mi, m_same, y, rng):
    """Same-versus-same null: opposite slot filled by same-label images -> no finding signal."""
    a = rng.sample(range(m_same.shape[1]), K)
    b = [j for j in range(m_same.shape[1]) if j not in a][:K]
    G = y * (mi - np.median(m_same[:, a], axis=1))
    N = np.median(np.abs(mi[:, None] - m_same[:, b]), axis=1)
    return G - N


def main():
    tok = build_tokenizer()
    feats, pc = load_feature_cache(), load_pooled_cache()
    dev = "cuda"
    by_name = {os.path.basename(p): p for p in feats["index"]}
    pooled = {os.path.basename(x) for x in pc["index"]}
    have = {n: p for n, p in by_name.items() if n in pooled}
    meta = load_meta(have)
    cases = build_cases(meta, random.Random(0))
    print(f"[pilot] {len(meta)} val images; {len(cases)} matchable cases; 24 unseen prompts; seeds {SEEDS}", flush=True)

    def vec(n):
        p = by_name[n]
        return feats["feats"][feats["index"][p]].float(), pc["pooled"][pc["index"][p]].float()

    per_seed, records = [], []
    for seed in SEEDS:
        m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, max_len=tok.max_len,
                         yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                         use_ground=True, ground_dim=pc["pooled"].shape[1]).to(dev)
        m.load_state_dict(torch.load(CKPT.format(seed=seed), map_location=dev), strict=False)
        m.eval()

        @torch.no_grad()
        def marg(names, q):
            ids, ap = tok.encode(q)
            vs = torch.stack([vec(n)[0] for n in names]).to(dev)
            gs = torch.stack([vec(n)[1] for n in names]).to(dev)
            lg, _ = m(vs, torch.tensor([ids]).repeat(len(names), 1).to(dev),
                      torch.tensor([ap]).repeat(len(names)).to(dev), ground=gs)
            return (lg[:, 1] - lg[:, 0]).float().cpu().numpy()

        dr = random.Random(100 + seed)
        rows = []
        for c in cases:
            prompts = unseen_prompts(c["phrase"])
            mi = np.array([marg([c["i"]], q)[0] for q in prompts])
            mo = np.array([marg(c["opp"], q) for q in prompts])            # [P, POOL]
            ms = np.array([marg(c["same"], q) for q in prompts])           # [P, POOL+K]
            V = draws_V(mi, mo, ms[:, :POOL], c["y"], dr)                           # [M, P]
            Vn = null_V(mi, ms, c["y"], dr)                                         # [P] null
            stable = len(set((mi > 0).tolist())) == 1
            rows.append({"c": c, "mi": mi, "V": V, "Vn": Vn, "stable": stable})
            records.append({"seed": seed, "finding": c["finding"], "y": c["y"],
                            "image": c["i"], "patient": c["patient"], "stable": bool(stable),
                            "m_i": mi.tolist(), "m_opp": mo.tolist(),
                            "m_same": ms.tolist(), "opp": c["opp"], "same": c["same"]})

        stab = [r for r in rows if r["stable"]]
        pats = np.array([r["c"]["patient"] for r in stab])
        fnd = np.array([r["c"]["finding"] for r in stab])
        Vmean = np.array([r["V"].mean(0) for r in stab])                   # [C, P]
        # population interaction on centered V, two draws for the cross covariance
        VA = np.array([r["V"][0] for r in stab]); VB = np.array([r["V"][1] for r in stab])
        CA, CB = twc(VA), twc(VB)
        sig_route = float(np.mean(CA * CB)); rho = float(np.corrcoef(CA.ravel(), CB.ravel())[0, 1])
        # finding vs case split of the reproducible interaction
        fpA = np.zeros_like(VA); fpB = np.zeros_like(VB)
        for f in set(fnd.tolist()):
            mk = fnd == f; fpA[mk] = CA[mk].mean(0); fpB[mk] = CB[mk].mean(0)
        sig_find = float(np.mean(fpA * fpB)); sig_case = float(np.mean((CA - fpA) * (CB - fpB)))

        # event on RAW V, deltas locked on a dev split's same-vs-same null
        upat_all = sorted(set(pats.tolist())); rp = random.Random(seed); rp.shuffle(upat_all)
        devset = set(upat_all[:len(upat_all) // 2])          # split by PATIENT, not by case
        dev_i = np.array([j for j, p in enumerate(pats) if p in devset])
        test_i = np.array([j for j, p in enumerate(pats) if p not in devset])
        Vn_all = np.array([r["Vn"] for r in stab])
        d0 = float(np.percentile(np.abs(Vn_all[dev_i]), 90))
        dg = float(np.percentile(np.abs(Vn_all[dev_i]), 97.5))
        Vmin = np.array([r["V"].min(0) for r in stab])                     # all draws agree grounded
        Vabsmax = np.array([np.abs(r["V"]).max(0) for r in stab])          # all draws agree near zero
        grounded = Vmin > dg
        unreliant = Vabsmax < d0
        route = grounded.any(1) & unreliant.any(1)
        rate_test = float(route[test_i].mean())

        # zero (or near-zero) events: a bootstrap gives a degenerate [0,0]. Report the
        # observed count and a Clopper-Pearson 95% upper bound with patients as the unit.
        from scipy import stats as _st
        n_ev = int(route[test_i].sum()); n_tp = int(len(np.unique(pats[test_i])))
        upper = (float(1 - 0.025 ** (1 / n_tp)) if n_ev == 0
                 else float(_st.beta.ppf(0.975, n_ev + 1, n_tp - n_ev)))
        r = {"seed": seed, "n_cases": len(cases), "n_stable": len(stab),
             "sigma2_route": sig_route, "corr": rho,
             "sigma2_finding": sig_find, "sigma2_case": sig_case,
             "case_share": sig_case / (sig_route + 1e-9),
             "delta_0": d0, "delta_g": dg,
             "n_test_cases": int(len(test_i)), "n_test_patients": n_tp,
             "route_drops_observed": n_ev, "route_drop_rate_test": rate_test,
             "route_drop_upper95_patient": upper}
        per_seed.append(r)
        print(f"[seed {seed}] stable {len(stab)}  sig_route {sig_route:+.4f} corr {rho:+.2f} "
              f"case_share {r['case_share']:.0%}  | route drops {n_ev}/{len(test_i)} cases "
              f"({n_tp} patients), 95% upper {upper*100:.1f}%  (delta_0 {d0:.2f} delta_g {dg:.2f})", flush=True)

    agg = {k: float(np.mean([s[k] for s in per_seed]))
           for k in ["sigma2_route", "corr", "case_share", "route_drop_rate_test"]}
    print(f"\n=== pooled ({len(SEEDS)} held-out seeds) ===")
    print(f"  interaction: sig_route {agg['sigma2_route']:+.4f}  corr {agg['corr']:+.2f}  "
          f"case_share {agg['case_share']:.0%}")
    print(f"  raw-V grounded->unreliant route-drop (locked deltas, patient-clustered): "
          f"{agg['route_drop_rate_test']:.3f}")
    json.dump({"per_seed": per_seed, "pooled": agg}, open(OUT, "w"), indent=2)
    json.dump(records, open(RECS, "w"))
    print(f"wrote {OUT} and full-component records to {RECS}")


if __name__ == "__main__":
    main()
