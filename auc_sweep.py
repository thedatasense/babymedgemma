"""Honest sweep: margin AUC across ALL available seeds, baseline vs grounded
(real / shuffled), to test whether the grounding AUC gain is seed-robust."""
import os, sys, json
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from dataset import Ex, NanoDataset, build_tokenizer, collate
from gemma_model import BabyGemmaVLM
import nih_demo as ND
from sklearn.metrics import roc_auc_score

tok = build_tokenizer()
records, _ = ND.build_nih_records(tok, 120, 0)
feats = torch.load("cache/nih_feats.pt", map_location="cpu")
d = torch.load("cache/nih_pooled.pt", map_location="cpu")
pc = {"index": {p: i for i, p in enumerate(d["paths"])}, "pooled": d["pooled"]}
gdim = d["pooled"].shape[1]
ex = [Ex(r["image_path"], r["question"], r["answer"], "original", c) for c, r in enumerate(records)]
y = np.array([1 if r["answer"] == "yes" else 0 for r in records])
q = np.array([r["question"] for r in records])
findings = sorted(set(q.tolist()))
ds_b = NanoDataset(ex, tok, feats, pooled=None)
ds_g = NanoDataset(ex, tok, feats, pooled=pc)

@torch.no_grad()
def margins(m, ds, ug, mode, gen):
    out = []
    for b in DataLoader(ds, batch_size=256, collate_fn=collate):
        kw = {}
        if ug:
            g = b["ground"].cuda()
            if mode == "shuffle": g = g[torch.randperm(g.shape[0], generator=gen, device=g.device)]
            kw["ground"] = g
        lg, _ = m(b["vision"].cuda(), b["tokens"].cuda(), b["ans_pos"].cuda(), **kw)
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.concatenate(out)

def mauc(s):
    return float(np.nanmean([roc_auc_score(y[q==f], s[q==f]) for f in findings
                             if len(set(y[q==f].tolist()))>1]))

def load(ck, ug):
    m = BabyGemmaVLM(vocab_size=len(tok), dim=384, depth=6, yes_id=tok.stoi["yes"],
                     no_id=tok.stoi["no"], use_ground=ug, ground_dim=gdim).cuda()
    m.load_state_dict(torch.load(ck, map_location="cuda"), strict=False)
    return m.eval()

gen = torch.Generator(device="cuda").manual_seed(0)
rows = []
print(f"{'seed':>4} {'base':>7} {'gnd_real':>9} {'gnd_shuf':>9}")
for seed in range(16):
    bck = f"results_gemma/B/augmented_s{seed}/model.pt"
    gck = f"results_ground/B/augmented_s{seed}/model.pt"
    if not (os.path.exists(bck) and os.path.exists(gck)): continue
    b = mauc(margins(load(bck, False), ds_b, False, "real", gen))
    gm = load(gck, True)
    gr = mauc(margins(gm, ds_g, True, "real", gen))
    gs = mauc(margins(gm, ds_g, True, "shuffle", gen))
    rows.append({"seed": seed, "baseline": b, "grounded_real": gr, "grounded_shuffle": gs})
    print(f"{seed:>4} {b:7.3f} {gr:9.3f} {gs:9.3f}", flush=True)

B = np.array([r["baseline"] for r in rows]); R = np.array([r["grounded_real"] for r in rows])
S = np.array([r["grounded_shuffle"] for r in rows])
print(f"\nn={len(rows)} seeds")
print(f"  baseline        AUC = {B.mean():.3f} +/- {B.std():.3f}   (min {B.min():.3f} max {B.max():.3f})")
print(f"  grounded real   AUC = {R.mean():.3f} +/- {R.std():.3f}   (min {R.min():.3f} max {R.max():.3f})")
print(f"  grounded shuffle AUC= {S.mean():.3f} +/- {S.std():.3f}")
print(f"  real - shuffle  = {(R-S).mean():+.3f} +/- {(R-S).std():.3f}")
print(f"  seeds with real-shuffle > 0.05: {int(((R-S)>0.05).sum())}/{len(rows)}")
from scipy import stats
print(f"  wilcoxon real vs shuffle p={stats.wilcoxon(R,S).pvalue:.4f}")
json.dump(rows, open("results_gemma/grounding/auc_sweep.json","w"), indent=2)
