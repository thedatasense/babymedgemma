"""Training loop for baby-MedGemma: the frozen MedSigLIP encoder feeds a small
Gemma-3 decoder, and only the decoder (plus the optional grounding-token projector)
is trained. Fixed recipe so the causal experiments vary only the factor under test.
"""

from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from babygemma.data_index import build_index
from babygemma.dataset import (Ex, NanoDataset, build_tokenizer, collate,
                               load_feature_cache, load_pooled_cache,
                               make_eval_clusters, make_training_examples)
from babygemma.gemma_model import BabyGemmaVLM, n_params
from babygemma import metrics as Mx

torch.set_num_threads(2)  # avoid CPU oversubscription when many jobs share the box


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
    keep = (torch.rand(vision.shape[0], vision.shape[1], 1,
                       generator=gen, device=vision.device) >= p).float()
    return vision * keep


def train_model(regime="augmented", seed=0, steps=1500, batch_size=128, lr=3e-4,
                depth=6, dim=384, vision_dropout=0.0, use_ground=False,
                device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    index = build_index()
    tok = build_tokenizer(index)
    feats = load_feature_cache()
    pc = load_pooled_cache() if use_ground else None
    if use_ground and pc is None:
        raise SystemExit("--grounding-token needs the pooled cache "
                         "(run scripts/data/precompute_pooled.py)")
    # position embeddings must cover 256 image tokens + optional grounding token +
    # the tokenizer's max_len, so take max_len from the tokenizer, not the default
    model = BabyGemmaVLM(vocab_size=len(tok), dim=dim, depth=depth,
                         max_len=getattr(tok, "max_len", 32),
                         yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
                         use_ground=use_ground,
                         ground_dim=pc["pooled"].shape[1] if pc else 1152).to(device)
    if verbose:
        print(f"[train] trainable params={n_params(model):,} vocab={len(tok)} device={device}")

    val_rows = [r for r in index if r["split"] == "val"]
    val_ex = [Ex(r["image_path"], r["question"], r["answer"], "original", i)
              for i, r in enumerate(val_rows)]
    val_ds = NanoDataset(val_ex, tok, feats)

    train_ex = make_training_examples(index, regime, seed=seed)
    eval_ex = make_eval_clusters(index, split=os.environ.get("NANO_EVAL_SPLIT", "test"))
    train_ds = NanoDataset(train_ex, tok, feats)
    eval_ds = NanoDataset(eval_ex, tok, feats)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.02)
    lossf = torch.nn.CrossEntropyLoss()

    # early stopping on validation accuracy -> stops before the model memorizes
    # text->answer and loses image grounding. NANO_EVAL_EVERY / NANO_PATIENCE tune it.
    eval_every = int(os.environ.get("NANO_EVAL_EVERY", "75"))
    patience = int(os.environ.get("NANO_PATIENCE", "4"))
    best_acc, best_state, bad, best_step = -1.0, None, 0, 0

    vgen = torch.Generator(device=device).manual_seed(seed)
    model.train()
    step, done = 0, False
    while not done:
        for b in loader:
            vis = _apply_vision_dropout(b["vision"].to(device), vision_dropout, vgen)
            kw = {"ground": b["ground"].to(device)} if use_ground else {}
            logits, _ = model(vis, b["tokens"].to(device), b["ans_pos"].to(device), **kw)
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
        print(f"[train] early-stopped at step {best_step} (val_acc {best_acc:.3f})")

    preds, answers, clusters, _ = Mx.predict(model, eval_ds, device)
    p_ni, a_ni, _, _ = Mx.predict(model, eval_ds, device, zero_vision=True)
    acc, acc_ni = Mx.accuracy(preds, answers), Mx.accuracy(p_ni, a_ni)
    result = {
        "regime": regime, "seed": seed, "steps": steps, "depth": depth,
        "early_stop_step": best_step, "val_acc": best_acc,
        "vision_dropout": vision_dropout, "use_ground": use_ground,
        "n_train": len(train_ex), "n_eval_clusters": len(set(clusters)),
        "trainable_params": n_params(model),
        "accuracy": acc, "accuracy_vision_ablated": acc_ni, "grounding_gap": acc - acc_ni,
        "flip_rate": Mx.flip_rate(preds, clusters),
    }
    if verbose:
        print(f"[train] acc={acc:.3f} vision_ablated={acc_ni:.3f} gap={acc-acc_ni:.3f} "
              f"flip={result['flip_rate']:.3f}")
    return {"model": model, "tok": tok, "eval_ds": eval_ds, "index": index,
            "result": result, "device": device}
