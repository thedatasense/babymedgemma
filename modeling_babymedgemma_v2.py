"""transformers wrapper for baby-MedGemma (scaled / grounded variant).

    from transformers import AutoModel
    m = AutoModel.from_pretrained("saillab/babymedgemma", trust_remote_code=True)

Differences from the probe variant kept at `probe-1841/`:
  * MedGemma's own SentencePiece tokenizer, pruned to the pieces this corpus uses
    (141 of 262,144). Segmentation is identical to MedGemma; unseen words decompose
    into pieces instead of silently becoming padding.
  * A grounding token: MedSigLIP's attention-pooled image embedding, projected and
    prepended before the 256 patch tokens. The patch tokens alone do not hand the
    finding signal to a decoder this small.
  * Trained on 107k per-finding-balanced questions from NIH + PadChest, so the
    text-only floor is exactly 0.500 and all accuracy above it is visual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

IMAGE_SIZE = 896        # MedGemma's input resolution
POOL_TO = 16            # 16x16 = 256 image tokens, MedGemma's budget
POOLED_SIZE = 448       # MedSigLIP's native resolution for its pooling head


class BabyMedGemmaConfig(PretrainedConfig):
    model_type = "baby_medgemma"

    def __init__(self, vocab_size=141, hidden_size=384, num_hidden_layers=6,
                 n_img=256, vision_dim=1152, max_len=20,
                 use_ground=True, ground_dim=1152,
                 tokenizer_name="google/medgemma-4b-it", tokenizer_hf_ids=None,
                 yes_id=None, no_id=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_img = n_img
        self.vision_dim = vision_dim
        self.max_len = max_len
        self.use_ground = use_ground
        self.ground_dim = ground_dim
        self.tokenizer_name = tokenizer_name
        self.tokenizer_hf_ids = tokenizer_hf_ids or []
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
        if config.use_ground:
            self.gproj = nn.Sequential(
                nn.LayerNorm(config.ground_dim), nn.Linear(config.ground_dim, config.hidden_size))
        self.n_patch = config.n_img
        self.n_img = config.n_img + (1 if config.use_ground else 0)
        self._hf2c = {h: i + 2 for i, h in enumerate(config.tokenizer_hf_ids)}
        self._tok = None
        self.post_init()

    # --- forward -----------------------------------------------------------
    def forward(self, input_ids=None, vision_features=None, ground=None,
                ans_pos=None, labels=None, **kwargs):
        if vision_features is None:
            raise ValueError("vision_features required: pooled MedSigLIP patch tokens "
                             "[B, 256, 1152] (see encode_images)")
        B = input_ids.shape[0]
        if ans_pos is None:                      # [ans] token id is 1
            ans_pos = (input_ids == 1).float().argmax(dim=-1)
        parts = []
        if self.config.use_ground:
            if ground is None:
                ground = torch.zeros(B, self.config.ground_dim,
                                     device=input_ids.device, dtype=self.dtype)
            parts.append(self.gproj(ground.to(self.dtype)).unsqueeze(1))
        parts.append(self.vproj(vision_features.to(self.dtype)))
        parts.append(self.gemma.get_input_embeddings()(input_ids))
        hidden = self.gemma(inputs_embeds=torch.cat(parts, dim=1)).last_hidden_state
        idx = ans_pos.to(hidden.device) + self.n_img
        pooled = hidden[torch.arange(B, device=hidden.device), idx]
        W = self.gemma.get_input_embeddings().weight          # tied LM head
        logits = pooled @ W[[self.config.no_id, self.config.yes_id]].T   # [B,2] = [no,yes]
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

    # --- helpers -----------------------------------------------------------
    def encode_question(self, text: str):
        """Tokenize with MedGemma's SentencePiece, mapped into the pruned table."""
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        hf = self._tok(text.lower(), add_special_tokens=False)["input_ids"]
        ids = [self._hf2c[i] for i in hf if i in self._hf2c][: self.config.max_len - 1]
        ids = ids + [1]                                   # [ans]
        ans_pos = len(ids) - 1
        ids = ids + [0] * (self.config.max_len - len(ids))
        return torch.tensor([ids]), torch.tensor([ans_pos])

    @staticmethod
    @torch.no_grad()
    def encode_images(pil_images, device="cpu", dtype=torch.float32):
        """-> (vision_features [B,256,1152], ground [B,1152]).
        Needs gated access to google/medsiglip-448."""
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained("google/medsiglip-448")
        enc = AutoModel.from_pretrained("google/medsiglip-448").to(device).eval()
        big = [im.resize((IMAGE_SIZE, IMAGE_SIZE)) for im in pil_images]
        px = proc(images=big, return_tensors="pt",
                  size={"height": IMAGE_SIZE, "width": IMAGE_SIZE})["pixel_values"].to(device)
        out = enc.vision_model(pixel_values=px, interpolate_pos_encoding=True).last_hidden_state
        B, N, D = out.shape
        g = int(N ** 0.5)
        k = max(1, g // POOL_TO)
        grid = out.float().transpose(1, 2).reshape(B, D, g, g)
        patch = F.avg_pool2d(grid, k, k).flatten(2).transpose(1, 2)      # [B,256,1152]
        px2 = proc(images=pil_images, return_tensors="pt")["pixel_values"].to(device)
        ground = F.normalize(enc.get_image_features(pixel_values=px2).float(), dim=-1)
        return patch.to(dtype), ground.to(dtype)
