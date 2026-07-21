"""Run the full MedSigLIP-nanoVLM experiment grid across all GPUs, then aggregate
with bootstrap CIs and pairwise significance.

    python run_all_gpus.py --run
    python run_all_gpus.py --aggregate

Experiments on real chest X-rays (MIMIC + PadChest), frozen MedSigLIP-448 vision:
  B data provenance | D grounding sweep (vision-dropout) | C architecture |
  A divergence | E causal patching.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("NANO_RESULTS", os.path.join(HERE, "results"))
STEPS = int(os.environ.get("NANO_STEPS", "1500"))
N_GPUS = int(os.environ.get("NANO_NGPUS", "8"))
SEEDS_B = int(os.environ.get("NANO_SEEDS_B", "24"))
SEEDS_AE = int(os.environ.get("NANO_SEEDS_AE", "8"))
SEEDS_D = int(os.environ.get("NANO_SEEDS_D", "8"))
SEEDS_C = int(os.environ.get("NANO_SEEDS_C", "4"))
ARCH = os.environ.get("NANO_ARCH", "nano")   # nano | gemma


def build_jobs():
    jobs = []

    def tj(tag, outdir, script="train.py", result="result.json", **flags):
        args = [sys.executable, script, "--steps", str(STEPS)]
        for k, v in flags.items():
            args += ["--" + k.replace("_", "-"), str(v)]
        if ARCH != "nano" and script in ("train.py", "experiment_a.py", "experiment_e.py"):
            args += ["--arch", ARCH]
        if os.environ.get("NANO_LR") and script == "train.py":
            args += ["--lr", os.environ["NANO_LR"]]
        if os.environ.get("NANO_GROUND") and script == "train.py":
            args += ["--grounding-token"]
        args += ["--out", outdir]
        jobs.append({"tag": tag, "args": args, "outdir": outdir, "result": result})

    for regime in ["canonical", "augmented", "adversarial"]:
        for s in range(SEEDS_B):
            tj("B", os.path.join(RESULTS, "B", f"{regime}_s{s}"), regime=regime, seed=s)
    for vd in [0.0, 0.25, 0.5, 0.75, 0.9]:
        for s in range(SEEDS_D):
            tj("D", os.path.join(RESULTS, "D", f"vd{vd}_s{s}"), regime="augmented", seed=s, vision_dropout=vd)
    for depth in [2, 4, 6, 8]:
        for s in range(SEEDS_C):
            tj("C", os.path.join(RESULTS, "C", f"depth{depth}_s{s}"), regime="augmented", seed=s, depth=depth)
    for regime in ["augmented", "adversarial"]:
        for s in range(SEEDS_AE):
            tj("A", os.path.join(RESULTS, "A", f"{regime}_s{s}"), script="experiment_a.py",
               result="experiment_a.json", regime=regime, seed=s)
            tj("E", os.path.join(RESULTS, "E", f"{regime}_s{s}"), script="experiment_e.py",
               result="experiment_e.json", regime=regime, seed=s, max_clusters=60)
    return jobs


def schedule(jobs, n_gpus):
    free = list(range(n_gpus))
    running = {}
    queue = list(jobs)
    total, done, failed = len(jobs), 0, 0
    t0 = time.time()
    while queue or running:
        while queue and free:
            gpu = free.pop()
            job = queue.pop(0)
            os.makedirs(job["outdir"], exist_ok=True)
            logf = open(os.path.join(job["outdir"], "log.txt"), "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            proc = subprocess.Popen(job["args"], cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
            running[proc.pid] = (job, gpu, proc, logf)
        time.sleep(2)
        for pid in list(running):
            job, gpu, proc, logf = running[pid]
            if proc.poll() is not None:
                logf.close()
                free.append(gpu)
                del running[pid]
                ok = proc.returncode == 0 and os.path.exists(os.path.join(job["outdir"], job["result"]))
                done += 1
                failed += 0 if ok else 1
                print(f"[{done}/{total}] {'ok ' if ok else 'FAIL'} {job['tag']} "
                      f"{os.path.basename(job['outdir'])} gpu{gpu} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[done] {total} jobs, {failed} failed, {time.time()-t0:.0f}s", flush=True)


def _load(pat):
    out = []
    for p in glob.glob(pat):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def _stats(xs):
    import numpy as np
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "ci95": [None, None], "n": 0}
    a = np.array(xs, float)
    rng = random.Random(0)
    boots = sorted(sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a) for _ in range(2000))
    return {"mean": float(a.mean()), "std": float(a.std()),
            "ci95": [float(boots[49]), float(boots[1949])], "n": len(a)}


def aggregate():
    import numpy as np
    try:
        from scipy import stats
        SCIPY = True
    except Exception:
        SCIPY = False
    summary = {}

    B = defaultdict(lambda: defaultdict(list))
    for r in _load(os.path.join(RESULTS, "B", "*", "result.json")):
        B[r["regime"]]["flip"].append(r["flip_rate"])
        B[r["regime"]]["gap"].append(r.get("grounding_gap"))
        B[r["regime"]]["acc"].append(r.get("accuracy"))
    summary["B_provenance"] = {reg: {"flip": _stats(v["flip"]), "grounding_gap": _stats(v["gap"]),
                                     "accuracy": _stats(v["acc"])} for reg, v in B.items()}
    if SCIPY:
        pw = {}
        regs = list(B.keys())
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                a, b = B[regs[i]]["flip"], B[regs[j]]["flip"]
                if a and b:
                    pw[f"{regs[i]}_vs_{regs[j]}"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        summary["B_provenance_pairwise_mwu"] = pw

    D = defaultdict(lambda: defaultdict(list))
    for r in _load(os.path.join(RESULTS, "D", "*", "result.json")):
        D[r["vision_dropout"]]["flip"].append(r["flip_rate"])
        D[r["vision_dropout"]]["gap"].append(r.get("grounding_gap"))
    summary["D_grounding_sweep"] = {str(k): {"flip": _stats(v["flip"]), "grounding_gap": _stats(v["gap"])}
                                    for k, v in sorted(D.items())}

    C = defaultdict(lambda: defaultdict(list))
    for r in _load(os.path.join(RESULTS, "C", "*", "result.json")):
        C[r["depth"]]["flip"].append(r["flip_rate"])
        C[r["depth"]]["acc"].append(r.get("accuracy"))
    summary["C_architecture"] = {str(k): {"flip": _stats(v["flip"]), "accuracy": _stats(v["acc"])}
                                 for k, v in sorted(C.items())}

    A = defaultdict(list)
    Aph = defaultdict(lambda: defaultdict(list))
    for r in _load(os.path.join(RESULTS, "A", "*", "experiment_a.json")):
        corr = r.get("per_layer_dispersion_flip_corr", {})
        A[r["regime"]].append([corr[k] for k in sorted(corr, key=lambda x: int(x))])
        for k, v in r.get("phenomenon_disagreement", {}).items():
            Aph[r["regime"]][k].append(v)
    summary["A_divergence"] = {}
    for reg, mats in A.items():
        mats = [m for m in mats if m]
        if mats:
            arr = np.array(mats)
            summary["A_divergence"][reg] = {
                "per_layer_corr_mean": arr.mean(0).tolist(),
                "phenomenon_disagreement": {k: float(np.mean(v)) for k, v in Aph[reg].items()}}

    E = defaultdict(list)
    for r in _load(os.path.join(RESULTS, "E", "*", "experiment_e.json")):
        E[r["regime"]].append(r)
    summary["E_patching"] = {}
    for reg, rs in E.items():
        nets = [r["net_rank1_by_layer"] for r in rs if r.get("net_rank1_by_layer")]
        loci = [r["locus_depth_net30"] for r in rs if r.get("locus_depth_net30") is not None]
        summary["E_patching"][reg] = {
            "net_rank1_by_layer_mean": np.array(nets).mean(0).tolist() if nets else [],
            "locus_depth_median": float(np.median(loci)) if loci else None,
            "mean_flip_targets": float(np.mean([r["n_flip_targets"] for r in rs])) if rs else 0}

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(summary, open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)
    _print(summary)
    return summary


def _print(s):
    def ln(k, st):
        if st["mean"] is None:
            print(f"    {k:14s} n=0")
        else:
            print(f"    {k:14s} {st['mean']:.3f} +/- {st['std']:.3f}  CI[{st['ci95'][0]:.3f},{st['ci95'][1]:.3f}] n={st['n']}")
    print("\n===== NANO-VLM SUMMARY (real CXR, frozen MedSigLIP-448) =====")
    print("\n[B] data provenance:")
    for reg, v in s["B_provenance"].items():
        print(f"  {reg}:"); ln("flip", v["flip"]); ln("grounding_gap", v["grounding_gap"]); ln("accuracy", v["accuracy"])
    if "B_provenance_pairwise_mwu" in s:
        print("  pairwise MWU p:", {k: round(v, 4) for k, v in s["B_provenance_pairwise_mwu"].items()})
    print("\n[D] grounding sweep (vision-dropout -> flip):")
    for k, v in s["D_grounding_sweep"].items():
        print(f"  vdrop={k}: flip {v['flip']['mean']:.3f}  grounding_gap {v['grounding_gap']['mean']:.3f}")
    print("\n[C] architecture (depth -> flip / acc):")
    for k, v in s["C_architecture"].items():
        print(f"  depth={k}: flip {v['flip']['mean']:.3f}  acc {v['accuracy']['mean']:.3f}")
    print("\n[A] divergence (per-layer dispersion-vs-flip corr):")
    for reg, v in s["A_divergence"].items():
        print(f"  {reg}: " + ", ".join(f"L{i}:{c:+.2f}" for i, c in enumerate(v["per_layer_corr_mean"])))
        print(f"     phenomenon disagreement: {{" + ", ".join(f'{k}:{val:.2f}' for k, val in v['phenomenon_disagreement'].items()) + "}")
    print("\n[E] patching (net rank-1 recovery by layer):")
    for reg, v in s["E_patching"].items():
        print(f"  {reg}: " + ", ".join(f"L{i}:{x:+.2f}" for i, x in enumerate(v["net_rank1_by_layer_mean"])))
        print(f"     locus median: {v['locus_depth_median']}  mean flip targets: {v['mean_flip_targets']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--gpus", type=int, default=N_GPUS)
    a = ap.parse_args()
    if a.run:
        jobs = build_jobs()
        print(f"[nano run_all] {len(jobs)} jobs across {a.gpus} GPUs, steps={STEPS}", flush=True)
        schedule(jobs, a.gpus)
        aggregate()
    elif a.aggregate:
        aggregate()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
