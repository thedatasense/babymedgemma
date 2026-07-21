"""Tokenizer, the three training regimes (built from real register tags), and
paraphrase-cluster evaluation, over cached MedSigLIP features.

Regimes (same causal logic as the synthetic toy, using the real `phenomenon`
register tag on each paraphrase):
  - canonical:   train on the original question only.
  - augmented:   train on original + all paraphrases (register uncorrelated with answer).
  - adversarial: select paraphrases so register predicts the answer (a phrasing
                 shortcut), independent of the image.

See docs/tiny_vlm_psf_isolation.md, section 6 (Experiment B).
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_index import PHENOMENA, build_index

PAD, ANS = "[pad]", "[ans]"
ANSWER_STOI = {"no": 0, "yes": 1}

# register -> answer bias for the adversarial regime (phrasing shortcut).
# Operator-preserving phenomena only (negation_pattern inverts the answer and is
# excluded from the expanded 10k set).
ADV_YES = {"lexical_substitution", "specificity_modulation"}
ADV_NO = {"scope_quantification", "syntactic_restructuring"}

_WORD = re.compile(r"[a-z0-9]+")
_FEATS = None   # cache singleton
_POOLED = None  # MedSigLIP attention-pooled embedding (the grounding token)


def load_feature_cache(path=None):
    global _FEATS
    if _FEATS is None:
        path = path or os.environ.get("NANO_FEATS") or os.path.join(HERE, "cache", "medsiglip_feats.pt")
        d = torch.load(path, map_location="cpu")
        _FEATS = {"index": {p: i for i, p in enumerate(d["paths"])}, "feats": d["feats"]}
    return _FEATS


def load_pooled_cache(path=None):
    """MedSigLIP's trained-pooling-head embedding per image, or None if not cached."""
    global _POOLED
    if _POOLED is None:
        path = path or os.environ.get("NANO_POOLED") or os.path.join(HERE, "cache", "medsiglip_pooled.pt")
        if not os.path.exists(path):
            return None
        d = torch.load(path, map_location="cpu")
        _POOLED = {"index": {p: i for i, p in enumerate(d["paths"])}, "pooled": d["pooled"]}
    return _POOLED


def tokenize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Tokenizer:
    def __init__(self, corpus: list[str], max_len=32):
        vocab = {PAD: 0, ANS: 1}
        for t in corpus:
            for w in tokenize_words(t):
                if w not in vocab:
                    vocab[w] = len(vocab)
        for w in ("no", "yes"):     # answer tokens (never in the questions), read from the LM head
            if w not in vocab:
                vocab[w] = len(vocab)
        self.stoi = vocab
        self.itos = [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self.pad_id = 0
        self.ans_id = 1
        self.max_len = max_len

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str):
        ids = [self.stoi.get(w, self.pad_id) for w in tokenize_words(text)][: self.max_len - 1]
        ids = ids + [self.ans_id]
        ans_pos = len(ids) - 1
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        return ids, ans_pos


def build_tokenizer(index=None, max_len=32):
    """MedGemma's pruned SentencePiece tokenizer when NANO_GEMMA_TOK=1, else the
    legacy word-level one (kept so the validated 1,841-question runs reproduce)."""
    index = index or build_index()
    if os.environ.get("NANO_GEMMA_TOK"):
        from gemma_tokenizer import build_gemma_tokenizer
        return build_gemma_tokenizer(index, max_len=int(os.environ.get("NANO_MAXLEN", "48")))
    corpus = []
    for r in index:
        corpus.append(r["question"])
        corpus.extend(p["text"] for p in r["paraphrases"])
    return Tokenizer(corpus, max_len=max_len)


@dataclass
class Ex:
    image_path: str
    text: str
    answer: str
    phenomenon: str
    cluster_id: int


def _balance(examples: list[Ex], rng: random.Random) -> list[Ex]:
    by = {"yes": [e for e in examples if e.answer == "yes"],
          "no": [e for e in examples if e.answer == "no"]}
    n = min(len(by["yes"]), len(by["no"]))
    if n == 0:
        return examples
    keep = rng.sample(by["yes"], n) + rng.sample(by["no"], n)
    rng.shuffle(keep)
    return keep


def make_training_examples(index, regime, seed, balance=True, adv_strength=0.9) -> list[Ex]:
    rng = random.Random(seed)
    train = [r for r in index if r["split"] == "train"]
    examples: list[Ex] = []
    for cid, r in enumerate(train):
        ans = r["answer"]
        paras = r["paraphrases"]
        if regime == "canonical":
            examples.append(Ex(r["image_path"], r["question"], ans, "original", cid))
        elif regime == "augmented":
            examples.append(Ex(r["image_path"], r["question"], ans, "original", cid))
            # with a large bank (48/question) keep every phrasing reachable but sample
            # per question per epoch, so coverage stays broad without a 4M-example epoch
            k = int(os.environ.get("NANO_PARA_SAMPLE", "0"))
            chosen_paras = rng.sample(paras, min(k, len(paras))) if k else paras
            for p in chosen_paras:
                examples.append(Ex(r["image_path"], p["text"], ans, p.get("phenomenon"), cid))
        elif regime == "adversarial":
            want = ADV_YES if ans == "yes" else ADV_NO
            aligned = [p for p in paras if p.get("phenomenon") in want]
            other = [p for p in paras if p.get("phenomenon") not in want]
            pool = paras if not aligned else (
                aligned if rng.random() < adv_strength or not other else other)
            chosen = rng.choice(pool) if pool else {"text": r["question"], "phenomenon": "original"}
            examples.append(Ex(r["image_path"], chosen["text"], ans, chosen.get("phenomenon"), cid))
        else:
            raise ValueError(regime)
    if balance:
        examples = _balance(examples, rng)
    return examples


def make_eval_clusters(index, split="test") -> list[Ex]:
    rows = [r for r in index if r["split"] == split]
    examples: list[Ex] = []
    for cid, r in enumerate(rows):
        examples.append(Ex(r["image_path"], r["question"], r["answer"], "original", cid))
        for p in r["paraphrases"]:
            examples.append(Ex(r["image_path"], p["text"], r["answer"], p.get("phenomenon"), cid))
    return examples


class NanoDataset(Dataset):
    def __init__(self, examples: list[Ex], tok: Tokenizer, feats=None, pooled=None):
        self.examples = examples
        self.tok = tok
        self.fc = feats or load_feature_cache()
        # the grounding token: MedSigLIP's attention-pooled embedding, if cached
        self.pc = pooled if pooled is not None else load_pooled_cache()
        self.ground_dim = self.pc["pooled"].shape[1] if self.pc else 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        fidx = self.fc["index"].get(e.image_path)
        feat = self.fc["feats"][fidx].float() if fidx is not None else \
            torch.zeros(256, 1152)
        if self.pc:
            gidx = self.pc["index"].get(e.image_path)
            ground = self.pc["pooled"][gidx].float() if gidx is not None else \
                torch.zeros(self.ground_dim)
        else:
            ground = torch.zeros(1)
        ids, ans_pos = self.tok.encode(e.text)
        return {
            "vision": feat,
            "ground": ground,
            "tokens": torch.tensor(ids, dtype=torch.long),
            "ans_pos": ans_pos,
            "answer": ANSWER_STOI[e.answer],
            "cluster_id": e.cluster_id,
            "phenomenon": e.phenomenon or "original",
        }


def collate(batch):
    return {
        "vision": torch.stack([b["vision"] for b in batch]),
        "ground": torch.stack([b["ground"] for b in batch]),
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "ans_pos": torch.tensor([b["ans_pos"] for b in batch], dtype=torch.long),
        "answer": torch.tensor([b["answer"] for b in batch], dtype=torch.long),
        "cluster_id": torch.tensor([b["cluster_id"] for b in batch], dtype=torch.long),
        "phenomenon": [b["phenomenon"] for b in batch],
    }
