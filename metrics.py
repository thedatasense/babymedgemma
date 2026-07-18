"""Metrics for the NanoVLM: flip rate, accuracy, grounding gap, layer-wise
dispersion. Mirrors the toy's metrics (same flip-rate definition) but reads the
[B,256,1152] vision features and the n_img offset for the ANS position.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import collate


@torch.no_grad()
def predict(model, dataset, device, batch_size=256, zero_vision=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    preds, answers, clusters, phen = [], [], [], []
    for b in loader:
        vis = b["vision"].to(device)
        if zero_vision:
            vis = torch.zeros_like(vis)
        logits, _ = model(vis, b["tokens"].to(device), b["ans_pos"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        answers.extend(b["answer"].tolist())
        clusters.extend(b["cluster_id"].tolist())
        phen.extend(b["phenomenon"])
    return preds, answers, clusters, phen


def accuracy(preds, answers):
    return float(np.mean([p == a for p, a in zip(preds, answers)])) if preds else 0.0


def flip_rate(preds, clusters):
    by = defaultdict(list)
    for p, c in zip(preds, clusters):
        by[c].append(p)
    if not by:
        return 0.0
    return float(np.mean([len(set(ps)) > 1 for ps in by.values()]))


def flip_labels(preds, clusters):
    by = defaultdict(list)
    for p, c in zip(preds, clusters):
        by[c].append(p)
    return {c: len(set(ps)) > 1 for c, ps in by.items()}


@torch.no_grad()
def layerwise_dispersion(model, dataset, device, batch_size=256):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    per_layer = defaultdict(lambda: defaultdict(list))
    for b in loader:
        idx = (b["ans_pos"] + model.n_img).to(device)
        _, acts = model(b["vision"].to(device), b["tokens"].to(device),
                        b["ans_pos"].to(device), capture=True)
        B = b["tokens"].shape[0]
        for li, h in enumerate(acts):
            vecs = h[torch.arange(B, device=device), idx].cpu().numpy()
            for j in range(B):
                per_layer[li][int(b["cluster_id"][j])].append(vecs[j])
    out = {}
    for li, clusters in per_layer.items():
        out[li] = {cid: _mpcd(np.stack(v)) for cid, v in clusters.items()}
    return out


def _mpcd(mat):
    if mat.shape[0] < 2:
        return 0.0
    n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sim = n @ n.T
    iu = np.triu_indices(mat.shape[0], k=1)
    return float(np.mean(1.0 - sim[iu]))


def point_biserial(disp_by_cluster, flips):
    xs, ys = [], []
    for cid, d in disp_by_cluster.items():
        if cid in flips:
            xs.append(d)
            ys.append(1.0 if flips[cid] else 0.0)
    if len(set(ys)) < 2 or np.std(xs) == 0:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])
