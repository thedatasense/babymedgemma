"""Train baby-MedGemma. Thin CLI over babygemma.train_model.

    python scripts/train/train.py --regime augmented --seed 0 --grounding-token \
        --steps 12000 --out results/ref

The index, feature cache, and tokenizer are selected by environment variables
(NANO_INDEX, NANO_FEATS, NANO_POOLED, NANO_MAXLEN); see the run/ scripts.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from babygemma import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="augmented",
                    choices=["canonical", "augmented", "adversarial"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--vision-dropout", type=float, default=0.0)
    ap.add_argument("--grounding-token", action="store_true",
                    help="prepend MedSigLIP's attention-pooled embedding as a grounding token")
    ap.add_argument("--arch", default="gemma", choices=["gemma"])  # kept for run/ compatibility
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    art = train_model(regime=args.regime, seed=args.seed, steps=args.steps,
                      batch_size=args.batch_size, lr=args.lr, depth=args.depth,
                      dim=args.dim, vision_dropout=args.vision_dropout,
                      use_ground=args.grounding_token)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(art["model"].state_dict(), os.path.join(args.out, "model.pt"))
        json.dump(art["result"], open(os.path.join(args.out, "result.json"), "w"), indent=2)
        print(f"[train] saved {args.out}")


if __name__ == "__main__":
    main()
