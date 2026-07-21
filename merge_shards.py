"""Fold encoded shards into unified scaled caches.

Writes cache/scaled_feats.pt   = {"paths", "feats"}   896 patch tokens -> 256
       cache/scaled_pooled.pt  = {"paths", "pooled"}  448 attention-pooled embedding

Existing caches (medsiglip_*, nih_*) are reused so only newly-encoded images are
pulled from the shards.
"""
from __future__ import annotations

import json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE = os.path.join(HERE, "cache")


def load_any(path, key):
    """Return {path: vector} from either the {"paths",k} or {"index",k} layout."""
    if not os.path.exists(path):
        return {}
    d = torch.load(path, map_location="cpu")
    if key not in d:
        return {}
    vals = d[key]
    if "paths" in d:
        return {p: vals[i] for i, p in enumerate(d["paths"])}
    return {p: vals[i] for p, i in d["index"].items()}


def main():
    index_path = os.environ.get("NANO_INDEX_IN", os.path.join(HERE, "data", "index_scaled.json"))
    prefix = os.environ.get("NANO_CACHE_PREFIX", "scaled")
    shard_dirs = os.environ.get("NANO_SHARD_DIRS", "shards").split(",")
    idx = json.load(open(index_path))
    need = sorted({r["image_path"] for r in idx})

    feats, pooled = {}, {}
    for f in ("medsiglip_feats.pt", "nih_feats.pt"):
        feats.update(load_any(os.path.join(CACHE, f), "feats"))
    for f in ("medsiglip_pooled.pt", "nih_pooled.pt"):
        pooled.update(load_any(os.path.join(CACHE, f), "pooled"))

    want = set(need)
    for sd in shard_dirs:
        sh = os.path.join(CACHE, sd.strip())
        for name in sorted(os.listdir(sh)) if os.path.isdir(sh) else []:
            if not name.endswith(".pt"):
                continue
            d = torch.load(os.path.join(sh, name), map_location="cpu")
            for i, p in enumerate(d["paths"]):
                if p in want:            # keep only what this index needs
                    feats[p] = d["feats"][i]
                    pooled[p] = d["pooled"][i]
            del d

    have = [p for p in need if p in feats and p in pooled]
    missing = [p for p in need if p not in feats or p not in pooled]
    print(f"[merge] need {len(need)}  have {len(have)}  missing {len(missing)}")
    if missing[:3]:
        print("  e.g.", missing[:3])

    fpath = os.path.join(CACHE, f"{prefix}_feats.pt")
    ppath = os.path.join(CACHE, f"{prefix}_pooled.pt")
    torch.save({"paths": have, "feats": torch.stack([feats[p].half() for p in have])}, fpath)
    torch.save({"paths": have, "pooled": torch.stack([pooled[p].half() for p in have])}, ppath)
    print(f"[merge] wrote {prefix}_feats.pt ({os.path.getsize(fpath)/1e9:.2f} GB)  "
          f"{prefix}_pooled.pt ({os.path.getsize(ppath)/1e6:.0f} MB)")

    kept = {p for p in have}
    drop = [r for r in idx if r["image_path"] not in kept]
    if drop:
        idx = [r for r in idx if r["image_path"] in kept]
        json.dump(idx, open(index_path, "w"))
        print(f"[merge] dropped {len(drop)} records lacking features; index now {len(idx)}")


if __name__ == "__main__":
    main()
