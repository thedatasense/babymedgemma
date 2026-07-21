"""Cache MedSigLIP's own pooled image embedding for every training image.

This is the signal the patch tokens do not hand over cheaply: SigLIP's trained
attention-pooling head (`get_image_features`) separates chest findings at ~0.81
AUC on NIH, while a naive pool of the patch tokens sits at chance. We cache it at
448 (the head's native resolution, where it scores best) so the decoder can be
given it directly as one extra grounding token.

Output: cache/medsiglip_pooled.pt = {"paths": [...], "pooled": fp16 [N, D]}

    CUDA_VISIBLE_DEVICES=0 python precompute_pooled.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_index import build_index, unique_images
from vision import MODEL_ID

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "medsiglip_pooled.pt")


def encode_pooled(paths, device="cuda:0", batch_size=32, dtype=torch.bfloat16):
    """paths -> (kept_paths, [N, D] fp16 pooled embeddings), L2-normalized."""
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype, device_map=device).eval()

    kept, out = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            imgs, ok = [], []
            for p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ok.append(p)
                except Exception:
                    continue
            if not imgs:
                continue
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, dtype)
            f = model.get_image_features(pixel_values=px)      # trained attention-pooling head
            f = F.normalize(f.float(), dim=-1)
            out.append(f.half().cpu())
            kept.extend(ok)
            if (i + batch_size) % 320 == 0:
                print(f"  {i+batch_size}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)
    return kept, torch.cat(out, dim=0)


def main():
    os.makedirs(CACHE, exist_ok=True)
    paths = unique_images(build_index())
    print(f"[pooled] {len(paths)} unique images", flush=True)
    kept, pooled = encode_pooled(paths)
    print(f"[pooled] cached {tuple(pooled.shape)} -> {OUT}")
    torch.save({"paths": kept, "pooled": pooled}, OUT)


if __name__ == "__main__":
    main()
