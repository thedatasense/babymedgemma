"""Fetch the remaining ChestX-ray14 batches (we have images_001 only, 4,999 of 112,120).

Downloads each zip from alkzar90/NIH-Chest-X-ray-dataset, extracts into the existing
images/ directory, then deletes the zip so peak disk stays near one batch rather than
the full 42 GB twice over.

    python download_nih.py --workers 3
"""
from __future__ import annotations

import argparse, os, shutil, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
DEST = "/home/bsada1/datasets/NIH/images"
TMP = "/home/bsada1/datasets/NIH/_zips"


def fetch(batch: int):
    from huggingface_hub import hf_hub_download
    name = f"data/images/images_{batch:03d}.zip"
    t0 = time.time()
    try:
        p = hf_hub_download(REPO, name, repo_type="dataset", local_dir=TMP)
    except Exception as e:
        return batch, 0, f"download failed: {e}"
    n = 0
    try:
        with zipfile.ZipFile(p) as z:
            for m in z.namelist():
                if not m.lower().endswith(".png"):
                    continue
                target = os.path.join(DEST, os.path.basename(m))
                if os.path.exists(target):
                    n += 1
                    continue
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    except Exception as e:
        return batch, n, f"extract failed: {e}"
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
    return batch, n, f"ok in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batches", default="2-12")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.batches.split("-"))
    todo = list(range(lo, hi + 1))

    os.makedirs(DEST, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    print(f"[nih] fetching batches {todo} with {args.workers} workers", flush=True)
    start = len(os.listdir(DEST))
    print(f"[nih] starting from {start} images", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch, b): b for b in todo}
        for f in as_completed(futs):
            b, n, msg = f.result()
            total = len(os.listdir(DEST))
            print(f"[nih] batch {b:03d}: {n} png  {msg}   (total now {total})", flush=True)

    print(f"[nih] DONE. images on disk: {len(os.listdir(DEST))}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
