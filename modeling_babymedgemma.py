"""transformers-compatible wrapper for baby-MedGemma, loadable with

    from transformers import AutoModel
    m = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True)

The model reads precomputed frozen MedSigLIP-448 features plus tokenized question
ids and returns yes/no logits at the answer position. A helper runs MedSigLIP for
end-to-end use, and the small word-level vocabulary is carried in the config so a
question can be tokenized without a separate tokenizer file.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BabyMedGemmaConfig(PretrainedConfig):
    model_type = "baby_medgemma"

    def __init__(self, vocab_size=735, hidden_size=384, num_hidden_layers=6,
                 n_img=256, vision_dim=1152, max_len=32, vocab=None,
                 yes_id=None, no_id=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_img = n_img
        self.vision_dim = vision_dim
        self.max_len = max_len
        self.vocab = vocab or []
        self.yes_id = yes_id if yes_id is not None else vocab_size - 1
        self.no_id = no_id if no_id is not None else vocab_size - 2
        self.num_labels = 2
        self.id2label = {0: "no", 1: "yes"}
        self.label2id = {"no": 0, "yes": 1}


def _gemma_text_config(c: BabyMedGemmaConfig):
    from transformers import Gemma3TextConfig
    dim, seq = c.hidden_size, c.n_img + c.max_len
    heads = max(2, dim // 64)
    return Gemma3TextConfig(
        hidden_size=dim, num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=heads, num_key_value_heads=max(1, heads // 3),
        head_dim=dim // heads, intermediate_size=dim * 4, vocab_size=c.vocab_size,
        max_position_embeddings=seq + 8, sliding_window=seq + 8, rope_theta=10000.0,
        attn_logit_softcapping=None, final_logit_softcapping=None,
    )


class BabyMedGemmaForVQA(PreTrainedModel):
    config_class = BabyMedGemmaConfig
    main_input_name = "input_ids"

    def __init__(self, config: BabyMedGemmaConfig):
        super().__init__(config)
        from transformers import Gemma3TextModel
        self.gemma = Gemma3TextModel(_gemma_text_config(config))
        self.vproj = nn.Sequential(
            nn.LayerNorm(config.vision_dim), nn.Linear(config.vision_dim, config.hidden_size),
            nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size))
        # yes/no decided from the tied LM head (no separate classifier), like MedGemma
        self.n_img = config.n_img
        self._stoi = {w: i for i, w in enumerate(config.vocab)}
        self.post_init()

    def forward(self, input_ids=None, vision_features=None, ans_pos=None,
                labels=None, **kwargs):
        if vision_features is None:
            raise ValueError("vision_features is required: pass MedSigLIP-448 pooled "
                             "features [B, 256, 1152] (see encode_images).")
        B = input_ids.shape[0]
        if ans_pos is None:  # the [ans] token id is 1
            ans_pos = (input_ids == 1).float().argmax(dim=-1)
        img = self.vproj(vision_features.to(self.dtype))
        txt = self.gemma.get_input_embeddings()(input_ids)
        hidden = self.gemma(inputs_embeds=torch.cat([img, txt], dim=1)).last_hidden_state
        idx = ans_pos.to(hidden.device) + self.n_img
        pooled = hidden[torch.arange(B, device=hidden.device), idx]
        W = self.gemma.get_input_embeddings().weight               # tied LM head [vocab, dim]
        logits = pooled @ W[[self.config.no_id, self.config.yes_id]].T   # [B, 2] = [no, yes]
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    # --- convenience helpers ------------------------------------------------
    def encode_question(self, text: str, max_len=None):
        """Tokenize a question with the bundled word vocabulary -> (input_ids, ans_pos)."""
        max_len = max_len or self.config.max_len
        ids = [self._stoi.get(w, 0) for w in re.findall(r"[a-z0-9]+", text.lower())][: max_len - 1]
        ids = ids + [1]                      # [ans]
        ans_pos = len(ids) - 1
        ids = ids + [0] * (max_len - len(ids))  # pad
        return torch.tensor([ids]), torch.tensor([ans_pos])

    @staticmethod
    @torch.no_grad()
    def encode_images(pil_images, device="cpu", dtype=torch.float32):
        """Run the frozen MedSigLIP-448 encoder -> pooled features [B, 256, 1152].
        Requires gated access to google/medsiglip-448."""
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained("google/medsiglip-448")
        enc = AutoModel.from_pretrained("google/medsiglip-448").vision_model.to(device).eval()
        px = proc(images=pil_images, return_tensors="pt",
                  size={"height": 896, "width": 896})["pixel_values"].to(device)
        out = enc(pixel_values=px, interpolate_pos_encoding=True).last_hidden_state  # [B, 4096, 1152]
        B, N, D = out.shape
        g = int(N ** 0.5)
        k = max(1, g // 16)
        grid = out.float().transpose(1, 2).reshape(B, D, g, g)
        pooled = F.avg_pool2d(grid, kernel_size=k, stride=k)      # [B, D, 16, 16]
        return pooled.flatten(2).transpose(1, 2).to(dtype)        # [B, 256, 1152]
