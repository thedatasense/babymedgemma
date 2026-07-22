"""Package the scaled/grounded v2 checkpoint as a transformers model.

Writes hf_export_v2/ (config.json + model.safetensors + modeling_babymedgemma.py),
then reloads through AutoModel(trust_remote_code=True) and runs one real prediction
against a cached feature to prove the round trip works.

    NANO_INDEX=data/index_transfer.json NANO_GEMMA_TOK=1 NANO_MAXLEN=20 \
      python convert_v2_to_hf.py
"""
from __future__ import annotations

import os, shutil, sys
import torch

from babygemma.paths import ROOT as HERE

CKPT = os.path.join(HERE, "results_transfer", "v2_aug_s0", "model.pt")
HF_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/hf, holds the wrapper
OUT = os.path.join(HERE, "hf_export")


def main():
    os.environ.setdefault("NANO_INDEX", os.path.join(HERE, "data", "index_transfer.json"))
    os.environ.setdefault("NANO_GEMMA_TOK", "1")
    os.environ.setdefault("NANO_MAXLEN", "20")
    from babygemma.data_index import build_index
    from babygemma.dataset import build_tokenizer

    idx = build_index()
    tok = build_tokenizer(idx)
    print(f"[convert] tokenizer {len(tok)} pieces, max_len={tok.max_len}, "
          f"yes={tok.stoi['yes']} no={tok.stoi['no']}")

    # write the v2 modeling file under the canonical name the repo expects
    src = os.path.join(HF_DIR, "modeling_babymedgemma.py")
    import importlib.util as _u
    _spec = _u.spec_from_file_location("modeling_babymedgemma", src)
    mod = _u.module_from_spec(_spec); _spec.loader.exec_module(mod)

    cfg = mod.BabyMedGemmaConfig(
        vocab_size=len(tok), hidden_size=384, num_hidden_layers=6,
        n_img=256, vision_dim=1152, max_len=tok.max_len,
        use_ground=True, ground_dim=1152,
        tokenizer_name="google/medgemma-4b-it",
        tokenizer_hf_ids=[int(i) for i in tok.hf_ids],
        yes_id=tok.stoi["yes"], no_id=tok.stoi["no"],
    )
    model = mod.BabyMedGemmaForVQA(cfg)
    state = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    assert not missing, f"missing: {missing[:6]}"
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    model.eval()
    print(f"[convert] loaded {sum(p.numel() for p in model.parameters()):,} params")

    mod.BabyMedGemmaConfig.register_for_auto_class()
    mod.BabyMedGemmaForVQA.register_for_auto_class("AutoModel")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    model.save_pretrained(OUT, safe_serialization=True)
    shutil.copy(src, os.path.join(OUT, "modeling_babymedgemma.py"))
    # point auto_map at the canonical filename
    import json
    cj = os.path.join(OUT, "config.json")
    c = json.load(open(cj))
    c["auto_map"] = {"AutoConfig": "modeling_babymedgemma.BabyMedGemmaConfig",
                     "AutoModel": "modeling_babymedgemma.BabyMedGemmaForVQA"}
    json.dump(c, open(cj, "w"), indent=2)
    print(f"[convert] wrote {OUT}: {sorted(os.listdir(OUT))}")

    # round trip
    from transformers import AutoModel
    m = AutoModel.from_pretrained(OUT, trust_remote_code=True).eval()
    from babygemma.dataset import load_feature_cache, load_pooled_cache
    fc, pc = load_feature_cache(), load_pooled_cache()
    v = fc["feats"][0].float().unsqueeze(0)
    g = pc["pooled"][0].float().unsqueeze(0)
    ids, ap = m.encode_question("is there cardiomegaly?")
    with torch.no_grad():
        out = m(input_ids=ids, vision_features=v, ground=g, ans_pos=ap).logits
    print(f"[convert] reload OK -> logits {out.tolist()} -> "
          f"{m.config.id2label[int(out.argmax(-1))]}")


if __name__ == "__main__":
    main()
