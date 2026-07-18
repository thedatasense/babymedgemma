"""Precompute and cache frozen MedSigLIP-448 features for every unique image.

Output: cache/medsiglip_feats.pt = {"paths": [...], "feats": fp16 [N,256,1152]}
Run once on one GPU. ~1GB for ~1775 images.

    CUDA_VISIBLE_DEVICES=0 python precompute_features.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_index import build_index, unique_images
from vision import MedSigLIP

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "medsiglip_feats.pt")


def load_rgb(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _all_images():
    import glob as _g
    pc = _g.glob("/home/bsada1/datasets/PadChest_GR/**/*.png", recursive=True)
    mm = _g.glob("/home/bsada1/datasets/MIMIC_JPG/**/*.jpg", recursive=True)
    return sorted(set(pc) | set(mm))


def main(batch_size=48):
    os.makedirs(CACHE, exist_ok=True)
    if os.environ.get("NANO_ALL_IMAGES"):
        paths = _all_images()
    else:
        paths = unique_images(build_index())
    print(f"[precompute] {len(paths)} unique images", flush=True)
    enc = MedSigLIP(device="cuda:0")

    feats, kept_paths = [], []
    t0 = time.time()
    batch, batch_paths = [], []

    def flush():
        if not batch:
            return
        f = enc.encode(batch)  # [b,256,1152] fp16 cpu
        feats.append(f)
        kept_paths.extend(batch_paths)
        batch.clear()
        batch_paths.clear()

    for i, p in enumerate(paths):
        img = load_rgb(p)
        if img is None:
            continue
        img = img.resize((448, 448))  # decode/resize once here so the batch stays small
        batch.append(img)
        batch_paths.append(p)
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)}  kept {len(kept_paths)}  ({time.time()-t0:.0f}s)", flush=True)
    flush()

    feats = torch.cat(feats, dim=0)
    print(f"[precompute] cached {feats.shape} in {time.time()-t0:.0f}s -> {OUT}")
    torch.save({"paths": kept_paths, "feats": feats}, OUT)


if __name__ == "__main__":
    main()
