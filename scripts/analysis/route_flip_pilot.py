"""Stage 0 falsification: do grounding-route flips exist?

A grounding-route flip is a case whose yes/no answer is STABLE across paraphrases while
its finding-selective visual reliance V changes across paraphrases. If V's variation
across prompts (within a case) is no larger than its variation under rematching, the
phenomenon does not exist and it should be dropped from the CRISP headline.

For image i, target finding f, prompt p, with margin m = logit(yes) - logit(no) and
true status y in {-1,+1}:

    G_{i,p} = y * [ m(I_i, q_p) - median_k m(I^-_k, q_p) ]     (opposite-label matches)
    N_{i,p} = median_k | m(I_i, q_p) - m(I^+_k, q_p) |         (same-label matches)
    V_{i,p} = G_{i,p} - N_{i,p}

Matches are drawn from NIH validation images (unseen by the model), matched on the
non-target finding vector, view position, sex, and age band, so N controls for generic
patient change rather than the target finding. Two disjoint match sets (A, B) per class
give a rematching-noise estimate.

Primary statistic on the STABLE-answer subset:
    case-by-prompt interaction variance of V   vs   rematching-noise variance of V.

    python scripts/analysis/route_flip_pilot.py
"""
from __future__ import annotations

import csv, glob, json, os, random
from collections import defaultdict

import numpy as np
import torch

from babygemma.paths import ROOT
from babygemma.data_index import build_index
from babygemma.dataset import build_tokenizer, load_feature_cache, load_pooled_cache
from babygemma.gemma_model import BabyGemmaVLM
from babygemma.nih import NIH_CSV, NIH_PHRASE
from babygemma import templates as T

CKPT = os.path.join(ROOT, "results_transfer", "v2_aug_s0", "model.pt")
OUT = os.path.join(ROOT, "results_transfer", "route_flip_pilot.json")
FINDINGS = ["Effusion", "Atelectasis", "Cardiomegaly", "Pneumothorax", "Consolidation",
            "Infiltration", "Nodule", "Mass"]     # well-populated; matching feasible
CASES_PER_FINDING = 40
K = 3                                              # matches per class per set
AGE_BAND = 12
rng = random.Random(0)


def prompts_for(phrase):
    """canonical + one held-out exemplar from each of the four paraphrase families."""
    ps = [f"is there {phrase}?"]
    for fam, tmpls in T.PARAPHRASES.items():
        ps.append(tmpls[T.HELDOUT_FROM].format(f=phrase))   # first held-out template
    return ps


def load_nih_val_meta(have):
    """image basename -> {findings:set, age:int, sex:str, view:str} for cached val images."""
    idx = build_index()
    val = {os.path.basename(r["image_path"]) for r in idx
           if r["split"] == "val" and "/NIH/" in r["image_path"]}
    meta = {}
    for row in csv.DictReader(open(NIH_CSV)):
        name = row["Image Index"]
        if name not in val or name not in have:
            continue
        f = {k for k in NIH_PHRASE if k in row["Finding Labels"].split("|")}
        try:
            age = int(row["Patient Age"])
        except ValueError:
            continue
        meta[name] = {"findings": f, "age": age, "sex": row["Patient Gender"],
                      "view": row["View Position"]}
    return meta


def covariate_ok(a, b, target):
    return (a["sex"] == b["sex"] and a["view"] == b["view"]
            and abs(a["age"] - b["age"]) <= AGE_BAND)


def nontarget_dist(a, b, target):
    """Hamming distance on the 13 non-target findings (lower = better match)."""
    return len((a["findings"] ^ b["findings"]) - {target})


def main():
    tok = build_tokenizer()
    feats, pc = load_feature_cache(), load_pooled_cache()
    dev = "cuda"
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, max_len=tok.max_len,
                     yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                     use_ground=True, ground_dim=pc["pooled"].shape[1]).to(dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()

    pooled_names = {os.path.basename(x) for x in pc["index"]}      # build once, not per-p
    have = {os.path.basename(p): p for p in feats["index"]
            if os.path.basename(p) in pooled_names}
    meta = load_nih_val_meta(have)
    print(f"[pilot] NIH val images with metadata and features: {len(meta)}", flush=True)
    by_name_path = {os.path.basename(p): p for p in feats["index"]}

    def vec(name):
        p = by_name_path[name]
        return (feats["feats"][feats["index"][p]].float(),
                pc["pooled"][pc["index"][p]].float())

    @torch.no_grad()
    def margins(names, prompt):
        ids, ap = tok.encode(prompt)
        vs = torch.stack([vec(n)[0] for n in names]).to(dev)
        gs = torch.stack([vec(n)[1] for n in names]).to(dev)
        idb = torch.tensor([ids]).repeat(len(names), 1).to(dev)
        apb = torch.tensor([ap]).repeat(len(names)).to(dev)
        lg, _ = m(vs, idb, apb, ground=gs)
        return (lg[:, 1] - lg[:, 0]).float().cpu().numpy()

    names = list(meta)
    cases = []
    for finding in FINDINGS:
        phrase = NIH_PHRASE[finding]
        pos = [n for n in names if finding in meta[n]["findings"]]
        neg = [n for n in names if finding not in meta[n]["findings"]]
        picks = (rng.sample(pos, min(CASES_PER_FINDING // 2, len(pos)))
                 + rng.sample(neg, min(CASES_PER_FINDING // 2, len(neg))))
        for i in picks:
            y = 1 if finding in meta[i]["findings"] else -1
            same = [n for n in (pos if y == 1 else neg)
                    if n != i and covariate_ok(meta[i], meta[n], finding)]
            opp = [n for n in (neg if y == 1 else pos)
                   if covariate_ok(meta[i], meta[n], finding)]
            if len(same) < 2 * K or len(opp) < 2 * K:
                continue
            same = sorted(same, key=lambda n: nontarget_dist(meta[i], meta[n], finding))[:2 * K]
            opp = sorted(opp, key=lambda n: nontarget_dist(meta[i], meta[n], finding))[:2 * K]
            rng.shuffle(same); rng.shuffle(opp)
            cases.append({"i": i, "y": y, "finding": finding, "phrase": phrase,
                          "same": same, "opp": opp})
    print(f"[pilot] {len(cases)} matchable cases", flush=True)

    prompts = prompts_for  # closure
    recs = []
    for c in cases:
        pr = prompts(c["phrase"])
        m_i = np.array([margins([c["i"]], q)[0] for q in pr])           # [P]
        mo = np.array([margins(c["opp"], q) for q in pr])               # [P, 6]
        ms = np.array([margins(c["same"], q) for q in pr])              # [P, 6]
        y = c["y"]
        V = {}
        for tag, sl in (("A", slice(0, K)), ("B", slice(K, 2 * K))):
            G = y * (m_i - np.median(mo[:, sl], axis=1))
            N = np.median(np.abs(m_i[:, None] - ms[:, sl]), axis=1)
            V[tag] = G - N
        stable = len(set((m_i > 0).tolist())) == 1
        recs.append({"finding": c["finding"], "stable": bool(stable),
                     "abs_m": float(np.abs(m_i).mean()),
                     "VA": V["A"].tolist(), "VB": V["B"].tolist()})

    # --- variance decomposition on the stable-answer subset ---
    stab = [r for r in recs if r["stable"]]
    VA = np.array([r["VA"] for r in stab])                # [C, P]
    VB = np.array([r["VB"] for r in stab])
    P = VA.shape[1]
    global_prompt = VA.mean(0)                            # per-prompt mean over cases
    interaction = VA - global_prompt[None, :]            # remove the global prompt effect
    interaction_var = float(np.mean(np.var(interaction, axis=1)))   # case-by-prompt, per case
    global_var = float(np.var(global_prompt))
    rematch_var = float(np.mean((VA - VB) ** 2 / 2))     # noise from redrawing matches
    ratio = interaction_var / (rematch_var + 1e-9)

    # a candidate route flip: stable answer, and V swings past both signs of a small band
    band = 0.5 * np.std(VA)
    route = [i for i, r in enumerate(stab)
             if VA[i].max() > band and VA[i].min() < -band]
    res = {
        "n_cases": len(recs), "n_stable": len(stab), "n_prompts": P,
        "global_prompt_effect_var": global_var,
        "case_by_prompt_interaction_var": interaction_var,
        "rematching_noise_var": rematch_var,
        "interaction_over_rematch": ratio,
        "n_route_flip_candidates": len(route),
        "route_flip_candidate_rate": len(route) / max(1, len(stab)),
        "band": float(band),
    }
    print("\n=== STAGE 0: does the route-flip phenomenon exist? ===")
    print(f"  stable-answer cases: {len(stab)} / {len(recs)}  ({P} prompts each)")
    print(f"  global prompt effect (one wording better for all): var = {global_var:.4f}")
    print(f"  case-by-prompt interaction (the phenomenon):        var = {interaction_var:.4f}")
    print(f"  rematching noise (redrawing match partners):        var = {rematch_var:.4f}")
    print(f"  interaction / rematching = {ratio:.2f}   (>> 1 means the phenomenon is real)")
    print(f"  route-flip candidates: {len(route)} / {len(stab)} = {res['route_flip_candidate_rate']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
