"""MedGemma's own SentencePiece tokenizer, pruned to the pieces our corpus uses.

Why not the full table: MedGemma's vocabulary is 262,144 pieces. At dim 384 that is a
100.7M embedding against a 13.58M transformer body -- 88% of the model would be a
lookup table whose rows almost never receive a gradient. Our whole corpus touches ~939
distinct pieces, so pruning costs ~0.36M params (2.6% of the model) and keeps
MedGemma's exact segmentation.

Why not the hand-rolled word tokenizer it replaces: that one maps any unseen word to
the PAD id, so an out-of-vocabulary word in a held-out-hospital question silently
becomes padding -- which would quietly corrupt the transfer evaluation. Subword pieces
decompose instead.

Interface matches the previous Tokenizer (stoi / itos / encode / __len__) so the rest
of the pipeline is unchanged.
"""

from __future__ import annotations

import os

PAD, ANS = "[pad]", "[ans]"
MODEL_ID = os.environ.get("NANO_TOKENIZER", "google/medgemma-4b-it")


class GemmaTokenizer:
    def __init__(self, corpus: list[str], max_len=48, answer_words=("no", "yes")):
        from transformers import AutoTokenizer
        self.hf = AutoTokenizer.from_pretrained(MODEL_ID)
        self.max_len = max_len

        # collect only the pieces this corpus actually produces. The bank is a small
        # set of templates instantiated per finding, so the corpus is massively
        # redundant (millions of rows, a few hundred distinct strings) -- dedupe first.
        used = set()
        for s in set(corpus):
            used.update(self.hf(s, add_special_tokens=False)["input_ids"])
        # the answer pieces must exist for the tied-LM-head readout
        self.answer_ids_hf = {}
        for w in answer_words:
            ids = self.hf(w, add_special_tokens=False)["input_ids"]
            self.answer_ids_hf[w] = ids[-1]
            used.update(ids)

        self.hf_ids = sorted(used)
        # compact ids: 0 = pad, 1 = answer slot, then the pruned pieces
        self.pad_id, self.ans_id = 0, 1
        self._hf2c = {h: i + 2 for i, h in enumerate(self.hf_ids)}
        self.itos = [PAD, ANS] + [self.hf.convert_ids_to_tokens(h) for h in self.hf_ids]

        # stoi is keyed by piece string, plus plain "yes"/"no" for the readout
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for w in answer_words:
            self.stoi[w] = self._hf2c[self.answer_ids_hf[w]]

        # The bank is a few hundred distinct strings instantiated over millions of
        # rows, so encode() would redo identical SentencePiece work per example and
        # starve the GPU. Memoise it.
        self._enc_cache: dict[str, tuple] = {}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str):
        hit = self._enc_cache.get(text)
        if hit is not None:
            return hit
        ids = [self._hf2c[i] for i in self.hf(text, add_special_tokens=False)["input_ids"]
               if i in self._hf2c][: self.max_len - 1]
        ids = ids + [self.ans_id]
        ans_pos = len(ids) - 1
        ids = ids + [self.pad_id] * (self.max_len - len(ids))
        self._enc_cache[text] = (ids, ans_pos)
        return ids, ans_pos

    def coverage(self, text: str):
        """Fraction of pieces present in the pruned table (1.0 = nothing dropped)."""
        h = self.hf(text, add_special_tokens=False)["input_ids"]
        return sum(1 for i in h if i in self._hf2c) / max(1, len(h))


def build_gemma_tokenizer(index=None, max_len=48) -> GemmaTokenizer:
    from babygemma.data_index import build_index
    index = index or build_index()
    corpus = set()
    for r in index:
        corpus.add(r["question"])
        corpus.update(p["text"] for p in r["paraphrases"])
        corpus.update(p["text"] for p in r.get("operator_variants", []))
    return GemmaTokenizer(sorted(corpus), max_len=max_len)
