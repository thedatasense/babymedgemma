"""MedSigLIP encoders used across scripts: the attention-pooled image embedding that
becomes the grounding token. The 896-pixel patch-token path lives in vision.py."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from PIL import Image

from babygemma.vision import MODEL_ID

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
