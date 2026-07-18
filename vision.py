"""Frozen MedSigLIP-448 vision encoder wrapper.

google/medsiglip-448: SigLIP-So400m medical vision tower, 448px, patch 14 ->
32x32 = 1024 patch tokens at 1152-dim, 27 layers, 429M params (the same SigLIP
family MedGemma uses). We freeze it and pool the 1024 tokens to 256 (2x2 average
over the 32x32 grid) to match MedGemma's image-token count.

Because it is frozen, features are precomputed once (precompute_features.py) and
cached, so the experiment grid trains only the small head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MODEL_ID = "google/medsiglip-448"
VISION_DIM = 1152
POOLED_TOKENS = 256  # 16x16 after 2x2 pooling of the 32x32 grid
GRID = 32


class MedSigLIP:
    def __init__(self, device="cuda:0", dtype=torch.bfloat16):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype, device_map=device).eval()
        self.vision = model.vision_model
        for p in self.vision.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, pil_images: list) -> torch.Tensor:
        """PIL images -> pooled features [B, 256, 1152] (float16 on CPU)."""
        px = self.processor(images=pil_images, return_tensors="pt")["pixel_values"]
        px = px.to(self.device, self.dtype)
        out = self.vision(pixel_values=px).last_hidden_state  # [B, 1024, 1152]
        B, N, D = out.shape
        g = int(N ** 0.5)
        grid = out.float().transpose(1, 2).reshape(B, D, g, g)     # [B, D, 32, 32]
        pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)       # [B, D, 16, 16]
        pooled = pooled.flatten(2).transpose(1, 2)                 # [B, 256, D]
        return pooled.half().cpu()
