"""NanoVLM: frozen MedSigLIP-448 features -> trainable head -> yes/no, with two
fusion modes:
  - 'prefix': concatenate 256 image tokens with the text and self-attend (original).
  - 'cross':  a text stream that self-attends and then cross-attends to a
              separately-refined image, the standard VQA inductive bias for
              text-conditional visual grounding.

Instrumented for activation capture and rank-1 patching. `ans_offset` gives the
answer-position offset into the captured residual stream (n_img for prefix, 0 for
cross), so metrics/experiments read the right position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoVLMConfig:
    vision_dim: int = 1152
    n_img: int = 256
    dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    vocab_size: int = 4096
    n_answers: int = 2
    max_len: int = 32
    fusion: str = "prefix"     # prefix | cross
    vision_depth: int = 2      # image self-attention blocks (cross fusion)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)
        kv = self.kv(ctx).view(B, M, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, cross=False):
        super().__init__()
        self.cross = cross
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.xattn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, ctx=None):
        x = x + self.attn(self.n1(x))
        if self.cross:
            x = x + self.xattn(self.nc(x), ctx)
        x = x + self.mlp(self.n2(x))
        return x


class NanoVLM(nn.Module):
    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.n_img = cfg.n_img
        self.ans_offset = 0 if cfg.fusion == "cross" else cfg.n_img
        self.vproj = nn.Sequential(nn.LayerNorm(cfg.vision_dim), nn.Linear(cfg.vision_dim, cfg.dim))
        self.vpos = nn.Parameter(torch.zeros(1, cfg.n_img, cfg.dim))
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.txt_pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.dim))
        self.modality = nn.Parameter(torch.zeros(2, cfg.dim))
        if cfg.fusion == "cross":
            self.vision_blocks = nn.ModuleList(
                [Block(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.vision_depth)])
            self.blocks = nn.ModuleList(
                [Block(cfg.dim, cfg.heads, cfg.mlp_ratio, cross=True) for _ in range(cfg.depth)])
        else:
            self.vision_blocks = None
            self.blocks = nn.ModuleList(
                [Block(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.n_answers)

    def forward(self, vision, tokens, ans_pos, capture=False, patch=None):
        B = vision.shape[0]
        img = self.vproj(vision) + self.vpos + self.modality[0]
        txt = self.tok_embed(tokens) + self.txt_pos[:, : tokens.shape[1]] + self.modality[1]

        if self.cfg.fusion == "cross":
            for vb in self.vision_blocks:
                img = vb(img)
            x, ctx = txt, img
            ans_idx = ans_pos
        else:
            x = torch.cat([img, txt], dim=1)
            ctx = None
            ans_idx = ans_pos + self.n_img

        acts = [] if capture else None
        for li, blk in enumerate(self.blocks):
            x = blk(x, ctx)
            if patch is not None and patch["layer"] == li:
                x = self._apply_patch(x, patch, ans_idx)
            if capture:
                acts.append(x.detach().clone())

        x = self.norm(x)
        pooled = x[torch.arange(B, device=x.device), ans_idx]
        return self.head(pooled), acts

    def _apply_patch(self, x, patch, ans_idx):
        donor = patch["donor"].to(x.device)
        positions = patch.get("positions", "ans")
        basis = patch.get("basis", None)
        B, N, D = x.shape
        mask = torch.zeros(B, N, 1, device=x.device)
        if positions == "all":
            mask[:] = 1.0
        elif positions == "ans":
            mask[torch.arange(B, device=x.device), ans_idx] = 1.0
        elif positions == "text":
            mask[:, self.ans_offset:] = 1.0
        else:
            raise ValueError(positions)
        if basis is None:
            replaced = donor
        else:
            b = basis.to(x.device)
            proj = lambda h: (h @ b.T) @ b
            replaced = x - proj(x) + proj(donor)
        return x * (1 - mask) + replaced * mask


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
