"""Encode one shard of missing images: 896 patch tokens + 448 pooled embedding.

Both readouts come from one image load. Shards are independent so the work spreads
across GPUs; merge_shards() folds them back into the unified caches.

    CUDA_VISIBLE_DEVICES=0 python encode_shard.py --shard 0 --of 6 --list missing.json
"""
from __future__ import annotations

import argparse, json, os, sys, time

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vision import MODEL_ID, IMAGE_SIZE

SHARD_DIR = os.environ.get("NANO_SHARD_DIR", os.path.join(HERE, "cache", "shards"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    paths = json.load(open(args.list))
    mine = [p for i, p in enumerate(paths) if i % args.of == args.shard]
    os.makedirs(SHARD_DIR, exist_ok=True)
    print(f"[shard {args.shard}/{args.of}] {len(mine)} images", flush=True)

    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                      device_map="cuda:0").eval()

    kept, patch, pooled = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(mine), args.batch_size):
            chunk = mine[i:i + args.batch_size]
            imgs, ok = [], []
            for p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    ok.append(p)
                except Exception:
                    continue
            if not imgs:
                continue
            # 896 patch tokens -> pooled to 256, matching MedGemma's budget
            big = [im.resize((IMAGE_SIZE, IMAGE_SIZE)) for im in imgs]
            px = proc(images=big, return_tensors="pt",
                      size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})["pixel_values"].to("cuda", torch.bfloat16)
            out = model.vision_model(pixel_values=px, interpolate_pos_encoding=True).last_hidden_state
            B, N, D = out.shape
            g = int(N ** 0.5)
            k = max(1, g // 16)
            grid = out.float().transpose(1, 2).reshape(B, D, g, g)
            patch.append(F.avg_pool2d(grid, k, k).flatten(2).transpose(1, 2).half().cpu())
            # 448 pooled embedding from the trained attention-pooling head
            px2 = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
            f = F.normalize(model.get_image_features(pixel_values=px2).float(), dim=-1)
            pooled.append(f.half().cpu())
            kept.extend(ok)
            if (i // args.batch_size) % 10 == 0:
                print(f"  {i+len(chunk)}/{len(mine)} ({time.time()-t0:.0f}s)", flush=True)

    out = {"paths": kept, "feats": torch.cat(patch), "pooled": torch.cat(pooled)}
    dst = os.path.join(SHARD_DIR, f"shard_{args.shard}.pt")
    torch.save(out, dst)
    print(f"[shard {args.shard}] wrote {len(kept)} -> {dst} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
