"""Baby-MedGemma: a small but architecturally faithful reflection of MedGemma-4B.

Uses a real Gemma-3 text decoder (transformers Gemma3TextModel: RoPE, RMSNorm,
grouped-query attention, GeGLU, QK-norm, the 5:1 local/global attention pattern)
with the frozen MedSigLIP-448 image tokens projected and PREPENDED inline, then
read by the same causally-masked decoder, mirroring how MedGemma fuses vision and
text. The local sliding window is set to cover the whole short sequence so the
answer position can attend to all 256 image tokens.

Matches the NanoVLM forward interface (returns (logits, acts)) so it drops into
train.py / metrics.py. Per-layer hidden states are exposed for capture; rank-1
patching would need layer hooks (added only if this architecture is adopted).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


class BabyGemmaVLM(nn.Module):
    def __init__(self, vocab_size, dim=384, depth=6, n_img=256, max_len=32, vision_dim=1152,
                 yes_id=None, no_id=None, use_ground=False, ground_dim=1152):
        super().__init__()
        from transformers import Gemma3TextConfig, Gemma3TextModel
        seq = n_img + max_len
        heads = max(2, dim // 64)
        gcfg = Gemma3TextConfig(
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            num_key_value_heads=max(1, heads // 3),   # grouped-query attention
            head_dim=dim // heads,
            intermediate_size=dim * 4,
            vocab_size=vocab_size,
            max_position_embeddings=seq + 8,
            sliding_window=seq + 8,                    # window covers the whole short sequence
            rope_theta=10000.0,
            attn_logit_softcapping=None,
            final_logit_softcapping=None,
        )
        self.gemma = Gemma3TextModel(gcfg)
        # multimodal projector (LayerNorm + MLP), analogous to Gemma's soft-token projection
        self.vproj = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # readout: yes/no token logits from the tied LM head (no separate classifier),
        # matching how MedGemma decides yes/no. Defaults to the last two vocab ids.
        self.no_id = no_id if no_id is not None else vocab_size - 2
        self.yes_id = yes_id if yes_id is not None else vocab_size - 1
        # optional grounding token: MedSigLIP's attention-pooled embedding, projected
        # and prepended before the patch tokens. The patch tokens alone do not hand
        # the finding signal to a decoder this small (naive readout of them is at
        # chance), so this gives it the signal the encoder actually has.
        self.use_ground = use_ground
        if use_ground:
            self.gproj = nn.Sequential(nn.LayerNorm(ground_dim), nn.Linear(ground_dim, dim))
        self.n_patch = n_img
        self.n_img = n_img + (1 if use_ground else 0)   # prefix length / ANS offset
        self.ans_offset = self.n_img
        self.cfg = SimpleNamespace(dim=dim, depth=depth, fusion="gemma_prefix",
                                   max_len=max_len, n_img=self.n_img)

    def forward(self, vision, tokens, ans_pos, capture=False, patch=None, ground=None):
        B = vision.shape[0]
        img = self.vproj(vision)                                   # [B, 256, dim]
        txt = self.gemma.get_input_embeddings()(tokens)            # [B, T, dim]
        parts = []
        if self.use_ground:
            if ground is None:
                ground = torch.zeros(B, self.gproj[0].normalized_shape[0],
                                     device=vision.device, dtype=vision.dtype)
            parts.append(self.gproj(ground).unsqueeze(1))          # [B, 1, dim]
        parts += [img, txt]
        inp = torch.cat(parts, dim=1)                              # prefix fusion
        ans_idx = ans_pos + self.n_img

        handle = None
        if patch is not None:
            def hook(mod, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = self._apply_patch(hs, patch, ans_idx)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            handle = self.gemma.layers[patch["layer"]].register_forward_hook(hook)
        try:
            out = self.gemma(inputs_embeds=inp, output_hidden_states=capture)
        finally:
            if handle is not None:
                handle.remove()

        h = out.last_hidden_state
        pooled = h[torch.arange(B, device=h.device), ans_idx]
        W = self.gemma.get_input_embeddings().weight               # tied LM head [vocab, dim]
        logits = pooled @ W[[self.no_id, self.yes_id]].T           # [B, 2] = [no, yes]
        acts = list(out.hidden_states[1:]) if capture else None    # per-layer residual streams
        return logits, acts

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


def n_params(model) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
