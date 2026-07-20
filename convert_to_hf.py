"""Package the trained baby-MedGemma probe as a transformers model.

Maps the augmented seed-0 BabyGemmaVLM checkpoint into BabyMedGemmaForVQA,
writes config.json + model.safetensors + modeling_babymedgemma.py to hf_export/,
then reloads through AutoModel(trust_remote_code=True) and runs one prediction.

    python convert_to_hf.py
"""

from __future__ import annotations

import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataset import build_tokenizer
from modeling_babymedgemma import BabyMedGemmaConfig, BabyMedGemmaForVQA

CKPT = os.path.join(HERE, "results_gemma", "B", "augmented_s0", "model.pt")
OUT = os.path.join(HERE, "hf_export")


def main():
    tok = build_tokenizer()
    cfg = BabyMedGemmaConfig(
        vocab_size=len(tok), hidden_size=384, num_hidden_layers=6,
        n_img=256, vision_dim=1152, max_len=32, vocab=tok.itos,
        yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
    )
    print(f"[convert] vocab={len(tok)} yes_id={cfg.yes_id} no_id={cfg.no_id}")

    model = BabyMedGemmaForVQA(cfg)
    state = torch.load(CKPT, map_location="cpu")
    # the BabyGemmaVLM checkpoint carries gemma.* and vproj.*; the wrapper shares
    # those submodule names, so the load is exact (no head to reconcile).
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("gemma.rotary")]  # rotary buffers are recomputed
    assert not missing, f"missing keys: {missing[:6]}"
    assert not unexpected, f"unexpected keys: {unexpected[:6]}"
    model.eval()

    BabyMedGemmaConfig.register_for_auto_class()
    BabyMedGemmaForVQA.register_for_auto_class("AutoModel")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    model.save_pretrained(OUT, safe_serialization=True)
    shutil.copy(os.path.join(HERE, "modeling_babymedgemma.py"), OUT)
    print(f"[convert] wrote {OUT}: {sorted(os.listdir(OUT))}")

    # reload the way a user would, and predict on a real cached feature
    from transformers import AutoModel
    m = AutoModel.from_pretrained(OUT, trust_remote_code=True).eval()
    from dataset import load_feature_cache
    fc = load_feature_cache()
    vfeat = fc["feats"][0].float().unsqueeze(0)  # one real MedSigLIP feature [1,256,1152]
    ids, ans = m.encode_question("is there cardiomegaly ?")
    with torch.no_grad():
        logits = m(input_ids=ids, vision_features=vfeat, ans_pos=ans).logits
    print(f"[convert] reload OK -> logits {logits.tolist()} -> {m.config.id2label[int(logits.argmax(-1))]}")


if __name__ == "__main__":
    main()
