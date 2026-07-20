"""Train the NanoVLM head on top of frozen, cached MedSigLIP features.

    python train.py --regime augmented --steps 1500 --seed 0 --out results/ref

Fixed recipe so the causal experiments vary only the factor under test.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

torch.set_num_threads(2)  # avoid CPU oversubscription when many jobs share the box

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metrics as Mx
from data_index import build_index
from dataset import (NanoDataset, build_tokenizer, collate, load_feature_cache,
                     make_eval_clusters, make_training_examples)
from model import NanoVLM, NanoVLMConfig, n_params


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _apply_vision_dropout(vision, p, gen):
    """Zero a fraction p of the 256 vision tokens per sample (weakens grounding)."""
    if p <= 0:
        return vision
    B, N, D = vision.shape
    keep = (torch.rand(B, N, 1, generator=gen, device=vision.device) >= p).float()
    return vision * keep


def train_model(regime="augmented", seed=0, steps=1500, batch_size=128, lr=3e-4,
                depth=6, dim=384, fusion="prefix", arch="nano", vision_dropout=0.0,
                device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    index = build_index()
    tok = build_tokenizer(index)
    feats = load_feature_cache()
    if arch == "gemma":
        from gemma_model import BabyGemmaVLM
        model = BabyGemmaVLM(vocab_size=len(tok), dim=dim, depth=depth,
                             yes_id=tok.stoi["yes"], no_id=tok.stoi["no"]).to(device)
    else:
        cfg = NanoVLMConfig(vocab_size=len(tok), depth=depth, dim=dim, heads=max(1, dim // 64), fusion=fusion)
        model = NanoVLM(cfg).to(device)
    if verbose:
        print(f"[nano] trainable params={n_params(model):,} vocab={len(tok)} device={device}")

    import copy
    from dataset import Ex
    val_rows = [r for r in index if r["split"] == "val"]
    val_ex = [Ex(r["image_path"], r["question"], r["answer"], "original", i)
              for i, r in enumerate(val_rows)]
    val_ds = NanoDataset(val_ex, tok, feats)

    train_ex = make_training_examples(index, regime, seed=seed)
    eval_ex = make_eval_clusters(index, split="test")
    train_ds = NanoDataset(train_ex, tok, feats)
    eval_ds = NanoDataset(eval_ex, tok, feats)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.02)
    lossf = torch.nn.CrossEntropyLoss()

    # early stopping on validation accuracy -> stops before the model memorizes
    # text->answer and loses image grounding
    eval_every, patience = 75, 4
    best_acc, best_state, bad, best_step = -1.0, None, 0, 0

    vgen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    step, done = 0, False
    while not done:
        for b in loader:
            vis = _apply_vision_dropout(b["vision"].to(device), vision_dropout, vgen)
            logits, _ = model(vis, b["tokens"].to(device), b["ans_pos"].to(device))
            loss = lossf(logits, b["answer"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % eval_every == 0:
                vp, va, _, _ = Mx.predict(model, val_ds, device)
                vacc = Mx.accuracy(vp, va)
                model.train()
                if vacc > best_acc + 1e-4:
                    best_acc, best_state, bad, best_step = vacc, copy.deepcopy(model.state_dict()), 0, step
                else:
                    bad += 1
                if verbose and step % 300 == 0:
                    print(f"  step {step}/{steps} loss {loss.item():.4f} val_acc {vacc:.3f} best {best_acc:.3f}")
                if bad >= patience:
                    done = True
                    break
            if step >= steps:
                done = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"[nano] early-stopped at step {best_step} (val_acc {best_acc:.3f})")

    preds, answers, clusters, _ = Mx.predict(model, eval_ds, device)
    p_ni, a_ni, _, _ = Mx.predict(model, eval_ds, device, zero_vision=True)
    acc, acc_ni = Mx.accuracy(preds, answers), Mx.accuracy(p_ni, a_ni)
    result = {
        "regime": regime, "seed": seed, "steps": steps, "depth": depth,
        "early_stop_step": best_step, "val_acc": best_acc,
        "vision_dropout": vision_dropout,
        "n_train": len(train_ex), "n_eval_clusters": len(set(clusters)),
        "trainable_params": n_params(model),
        "accuracy": acc, "accuracy_vision_ablated": acc_ni, "grounding_gap": acc - acc_ni,
        "flip_rate": Mx.flip_rate(preds, clusters),
    }
    if verbose:
        print(f"[nano] acc={acc:.3f} vision_ablated={acc_ni:.3f} gap={acc-acc_ni:.3f} "
              f"flip={result['flip_rate']:.3f}")
    return {"model": model, "tok": tok, "eval_ds": eval_ds, "index": index,
            "result": result, "device": device}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="augmented", choices=["canonical", "augmented", "adversarial"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--fusion", default="prefix", choices=["prefix", "cross"])
    ap.add_argument("--arch", default="nano", choices=["nano", "gemma"])
    ap.add_argument("--vision-dropout", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    art = train_model(regime=args.regime, seed=args.seed, steps=args.steps,
                      batch_size=args.batch_size, lr=args.lr, depth=args.depth, dim=args.dim,
                      fusion=args.fusion, arch=args.arch, vision_dropout=args.vision_dropout)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save(art["model"].state_dict(), os.path.join(args.out, "model.pt"))
        json.dump(art["result"], open(os.path.join(args.out, "result.json"), "w"), indent=2)
        print(f"[nano] saved {args.out}")


if __name__ == "__main__":
    main()
